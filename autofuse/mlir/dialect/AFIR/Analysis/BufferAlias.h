/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_BUFFERALIAS_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_BUFFERALIAS_H

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {

// 一个维度上的起点。多缓冲按 ADR-12 表达为 memref 的 stage 维，轮转下标因此不是常量，
// 但也不是完全未知——它是 `(iv + phase) mod modulus`。这三种形态在别名判定里给出的
// 结论完全不同，合并成「常量 / 未知」两档会直接丢掉多缓冲的全部信息。
struct DimensionOffset {
  enum class Kind : uint8_t {
    Static,    // 常量下标
    Rotating,  // (循环归纳变量 + phase) mod modulus
    Unknown,   // 其余：动态且无法归类
  };

  Kind kind = Kind::Unknown;
  int64_t value = 0;    // Static：常量值；Rotating：phase
  int64_t modulus = 0;  // Rotating：轮转级数
  // Rotating 的归纳变量本身。**必须带上**：相位差只有在两个下标由同一个归纳变量驱动
  // 时才有意义。丢掉它，两个各自轮转的兄弟循环会被按相位差判成「永不撞同一个槽」，
  // 而它们其实各自遍历了全部槽——那是漏报，不是误报。
  Value inductionVariable;

  static DimensionOffset getStatic(int64_t value);
  static DimensionOffset getRotating(Value inductionVariable, int64_t modulus, int64_t phase);
  static DimensionOffset getUnknown();
};

// 值追到根之后的存储切片：哪块存储、在它的下标空间里占哪一段。
struct BufferSlice {
  Value base;                                  // 存储的根定义点；空表示追不到
  llvm::SmallVector<DimensionOffset> offsets;  // 相对 base 的每维起点
  llvm::SmallVector<int64_t> sizes;            // 每维长度，负值表示未知
  bool wholeBase = false;                      // 追到了根但放弃了段信息，按整块处理

  bool isValid() const {
    return static_cast<bool>(base);
  }
};

enum class AliasResult : uint8_t {
  NoAlias,   // 可证明不重叠
  MayAlias,  // 证明不了
};

// 追到值的根存储。识别的取视图形式：`memref.subview`（静态 offset/size、单位 stride）、
// `memref.cast`，以及 destination-style Op 的 tie——结果与它绑定的 destination 是同一块
// 存储，这一条与 `getBufferDescAttr` 走的是同一个事实（设计文档 5.1 步 8）。
// 识别不了的形式一律退化成 `wholeBase`，宁可多报不可漏报。
BufferSlice resolveBufferSlice(Value value);

// 两个切片是否可能指向同一块存储。`iterationDistance` 是 `rhs` 相对 `lhs` 的循环迭代
// 距离（0 = 同一次迭代）——轮转下标只有在距离是级数的整数倍时才会撞上，这正是
// 「双缓冲把冒险推远，但推不掉」的形式化。
//
// `loopInductionVariable` 是这个距离所属的循环的归纳变量。只有当轮转下标正是由它驱动
// 时，模运算才谈得上成立；由内层循环驱动的下标配上外层循环的距离是两套坐标系，一律
// 按「可能重叠」处理。传空表示只问同一次迭代内的重叠（此时只要求两边同源即可）。
AliasResult aliasSlices(const BufferSlice &lhs, const BufferSlice &rhs, int64_t iterationDistance = 0,
                        Value loopInductionVariable = {});

// 两个访问在跨迭代方向上第一次可能重叠的迭代距离（`rhs` 比 `lhs` 晚这么多次迭代）。
// 返回 std::nullopt 表示任何距离都不会重叠；返回 1 表示相邻迭代就撞上（单缓冲）；
// 返回 S 表示 S 路轮转把它推到了 S。
std::optional<int64_t> firstAliasingIterationDistance(const BufferSlice &lhs, const BufferSlice &rhs,
                                                      Value loopInductionVariable = {});

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_BUFFERALIAS_H
