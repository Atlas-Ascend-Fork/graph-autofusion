/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ReducePlan.h - AFIR reduction analysis ------------------*- C++ -*-===//
// 分析 tensor_desc、sched 和 asc_graph，提供累加语义、实现布局与逻辑归约域。
// 语义 pass 仅依赖累加分析；AR/RA 布局与 scratch 要求由完整计划检查。
// 分析函数不发诊断；diagnostic 非空时返回失败原因，成功时清空。

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_REDUCEPLAN_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_REDUCEPLAN_H

#include <cstdint>
#include <optional>
#include <string>

#include "dialect/AFIR/IR/AFIR.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Operation;
}  // namespace mlir

namespace mlir::afir {

// 二维归约布局：A 为保留维，R 为归约维；AR = [A, R]，RA = [R, A]。
enum class ReducePattern : uint8_t { AR, RA };

// 合并维的轴 ID；此处不物化 SSA，也不添加物理对齐。
struct ReduceMergedExtent {
  llvm::SmallVector<int64_t, 2> axes;  // 外 -> 内，按向量化顺序
};

// 跨轮累加语义，独立于实现布局和 scratch 要求。
struct ReduceAccumulationInfo {
  // 调度路径中的累加轴，外到内；合法的非累加归约返回空列表。
  llvm::SmallVector<int64_t, 2> accumulationAxes;

  bool isAccumulating() const {
    return !accumulationAxes.empty();
  }
};

// 仅分析跨轮累加，不检查 AR/RA 布局或 scratch 数量。
FailureOr<ReduceAccumulationInfo> analyzeReduceAccumulation(Operation *op, std::string *diagnostic = nullptr);

struct ReducePlan {
  // 来自 analyzeReduceAccumulation 的跨轮累加轴。
  llvm::SmallVector<int64_t, 2> accumulationAxes;

  ReducePattern pattern = ReducePattern::AR;

  // first 为保留维 A，last 为归约维 R，不代表物理外/内顺序。
  // materializePhysicalReduceShape 按 pattern 排列物理维度并对齐最内维。
  ReduceMergedExtent first;
  ReduceMergedExtent last;

  // 是否恰有一个向量化归约轴，用于 ReduceInit 的 isTailLast 参数。
  bool tailLast = false;

  // scratch 段内的零基下标，由 afir.scratch_roles 解析，不依赖调度器排列顺序。
  unsigned apiScratchIndex = 0;
  std::optional<unsigned> accumulationScratchIndex;

  bool isAccumulating() const {
    return !accumulationAxes.empty();
  }
};

// 构造 buffer 归约计划，检查 descriptor、二维 A/R 布局及 scratch 要求。
FailureOr<ReducePlan> buildReducePlan(Operation *op, std::string *diagnostic = nullptr);

// 调度轴追溯到的 Original 轴及其尺寸。
// Mean 分母取完整逻辑域尺寸的乘积，不能用单 tile 范围或带 padding 的物理跨度代替。
struct LogicalReductionDomain {
  // Original 轴 id，按 destination 的轴序去重后给出（稳定、不依赖哈希序）。
  llvm::SmallVector<int64_t, 2> originalAxes;
  // 与 `originalAxes` 一一对应的尺寸表达式。`N` 是它们的乘积。
  llvm::SmallVector<ExprAttr, 2> sizes;
};

// 从轴图追溯归约域，不创建 SSA；描述缺失、来源不可追溯或尺寸非法时失败。
// 跨阶段生成的 Original 轴可能仅表示 partial 数量，本函数不证明域的完整性。
FailureOr<LogicalReductionDomain> analyzeLogicalReductionDomain(Operation *op, std::string *diagnostic = nullptr);

// Bridge 建完模块、内存物化前追踪 Mean 输入来源，并标注已证明的 reduction_domain。
// 来源涉及携带归约产物的跨阶段 workspace 时不标注；同函数内的已完成归约不因此被拒。
// 无法证明时不写属性，由消费侧处理失败。
void annotateReductionDomains(ModuleOp module);

// 消费已证明的域：缺少属性即失败，不回落到轴图推断。
// 若当前轴图也能推导出域，必须与属性一致。
FailureOr<LogicalReductionDomain> getProvenReductionDomain(Operation *op, std::string *diagnostic = nullptr);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_REDUCEPLAN_H
