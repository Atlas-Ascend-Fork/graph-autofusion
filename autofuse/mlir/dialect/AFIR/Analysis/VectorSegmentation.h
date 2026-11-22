/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- VectorSegmentation.h - Collapse AFIR vectorized axes -----*- C++ -*-===//
//
// 向量化轴合并（合轴），legacy 所说的阶段 2在 MLIR 侧的唯一实现。
//
// legacy 的合轴分两个阶段，划分依据是作用对象：
//
//   - 阶段 1（图级）：`optimize/optimize.cpp:845` `MergeContinuousAxis`，改写全图共享的
//     轴集合（`ApplyMerge`），下游 tiling / block 切分 / 内存规划 / 代价模型都读它，所以
//     它必须全图一致。它的结果已经烘进 `afir.asc_graph` 的轴表传到这里——4 维 elementwise
//     的 z0/z1/z2/z3 到 MLIR 手上已经是一根 `z0z1z2z3`，本文件不重做这一层。
//   - 阶段 2（指令映射级）：只看本算子自己的 operand，因此可以合得比阶段 1 更多。
//     legacy 这一层有两份判据不同的实现：
//       `codegen/api_call/utils/api_call_utils.cpp:343` `CheckAxisContinuous`（stride 连续性）
//       `codegen/api_call/broadcast/broadcast_api_call.cpp:40` `DimensionCollapse`（按
//       `is_brc` 标志分组，**不查 stride 连续性**）
//     本文件只实现前者。后者不是「在另一个维度上更激进」，是判据更弱：UB tile 上出现对齐
//     填充时它会把两根普通轴合并、尺寸直接相乘，而那个乘积不等于实际内存跨度。
//
// 判据与 PyTorch TensorIterator 的 `can_coalesce` 逐字同构：从最内往外，若对所有参与的
// 张量都有 `stride[外] == stride[内] × extent[内]`，两轴合并。逐对比较相邻轴，不是拿
// 累积段长去比——legacy 在 `CheckAxisContinuous` 之后把 `prev_repeat` 覆写回刚处理那一根轴
// 自己的 repeat（`api_call_utils.cpp` 合并分支末尾），所以它比的也是相邻对。
//
// 广播轴（stride 为 0）只与 stride 为 0 的轴合并，这不是特例，是同一条判据的结果：
// `0 == 0 × e` 恒真，而 `s == 0 × e` 在 `s ≠ 0` 时恒假。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_VECTORSEGMENTATION_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_VECTORSEGMENTATION_H

#include <cstdint>

#include "dialect/AFIR/IR/AFIR.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir {

// 参与合轴判据的一个张量，以及它在 IR 里的出处。
//
// 出处要带上，因为下降侧要按段的 stride 给对应那个 operand 生成偏移，而
// `afir.operand_descs` 里并非每一项都参与（Load 的 GM 侧那一项角色是 `input`，
// 没有向量化轴），按数组下标硬对是错的。
struct VectorTensor {
  enum class Side { Operand, Destination };

  Side side = Side::Operand;
  // 在 `afir.operand_descs` / `afir.destination_descs` 数组里的下标。
  unsigned index = 0;
  TensorDescAttr descriptor;
};

// 一段合并后的向量化轴。
struct VectorSegment {
  // 合进本段的向量化轴 id，外→内。至少一个。
  SmallVector<int64_t> axes;
  // 段内各轴 extent 之积，已 Simplify。
  ExprAttr extent;
  // 逐轴 extent，与 `axes` 一一对应。
  //
  // 有了 `extent` 还留这一份，是因为下降侧必须能把单独一根轴的 extent 换掉：切分轴
  // （`TileInner`）在最后一个 tile 上的实际长度是尾块长度，不是轴长，而 `extent` 已经乘成
  // 一个数，换不回去。少了这一份，含切分轴的段在尾块上会算多。
  SmallVector<ExprAttr> axisExtents;
  // 与 `VectorPlan::tensors` 一一对应：本段最内那根轴上的 stride。
  //
  // 取最内轴而不是最外轴：这个 stride 的用途是「本段前进一格，地址前进多少」，
  // 而段内各轴已证连续，所以整段等价于一根 stride 为最内轴 stride 的轴。legacy 存的
  // 也是这一个（`SaveApiLoopAxisStatus` 在开段时压入，合并分支不改它）。
  SmallVector<ExprAttr> strides;
};

struct VectorPlan {
  // 参与判据的张量，顺序为：operand 侧（按下标升序）在前，destination 侧在后。
  SmallVector<VectorTensor> tensors;
  // 合并后的段，外→内。
  SmallVector<VectorSegment> segments;
};

// 对 `op` 的向量化轴做阶段 2 合轴。
//
// 取哪些张量：`afir.operand_descs` 与 `afir.destination_descs` 里角色为 `compute`
// 且有向量化轴的那些。没有向量化轴的 compute 张量是标量（调度给它的 descriptor 是
// `repeats = [1, 1], strides = [0, 0]`，一个 `vectorized_axis` 都没有），它在整条循环里
// 就是同一个值，既约束不了合轴也不需要偏移。
// 角色过滤正好选出片上侧的张量——Load 的 GM 侧角色是 `input`、Store 的是 `output`，它们
// 没有向量化轴，也不该参与判据。这与 legacy 拿 `ub_inputs` / `ub_outputs` 是同一件事。
//
// 段 extent 是推出来的，不是从某个「基准张量」挑的：一根轴上各张量的 repeat 中，凡不可证
// 为 1 的必须互相可证相等，extent 取那个公共值；全为 1 则 extent 为 1，该轴整根丢掉
// （不贡献元素也不贡献迭代）。丢轴在逐对比较之前做，所以被丢的轴不会把两侧的相邻关系
// 割断。legacy 是挑一个 `GetBaseTensor` 再取它的 `axis_size`；推导比挑选多一道「各张量
// 是否一致」的检查，不一致时报失败而不是静默采用其中一个。
//
// 失败不输出诊断：调用方在 capability 查询里把它翻译成
// `CapabilitySupport::unsupported(MissingInformation)`，整图回落 legacy。在这里 emitError
// 会让 pass 直接失败，那不是要的行为。与 `buildTransferPlan` 同一约定。
//
// 失败的情形：没有一个 compute 角色的张量、各张量的 `vectorized_axis` 不是同一串、
// descriptor 的数组长度自相矛盾、某根向量化轴在 `axis` 里找不到、一根轴上各张量的 repeat
// 互相证不了相等。
//
// 「证不了」一律封段（保守方向）：多分一段的后果是多一层循环，少分一段的后果是算错地址。
FailureOr<VectorPlan> collapseVectorizedAxes(Operation *op, bool compactTail = false);
// Explicit participants retain their original operand/destination indices.
// Callers remove scalar API operands before merging, not from the completed plan.
FailureOr<VectorPlan> collapseVectorizedAxes(Operation *op, ArrayRef<VectorTensor> tensors, bool compactTail = false);

// 两个 `ExprAttr` 可证相等吗。文本相同先短路，再走 `ProveExpressionsEqual`；任一侧为空
// 或证不了（含 `Unknown` 与解析失败）返回 false。`VectorSegmentation.cpp` 合轴判据与
// `Implementations/` 侧（`Transpose.cpp` / `Pad.cpp` / `Broadcast.cpp` / `RemovePad.cpp`）
// 的形态判据共用——两侧都是「判不了就拒」的保守方向，短路与 SymEngine 语义必须同一份。
bool provablyEqual(ExprAttr lhs, ExprAttr rhs);

// 表达式可证为字面量 1 吗。同样保守：证不了就是 false。
bool provablyOne(ExprAttr expression);

// `axisId` 在 `descriptor.getAxis()` 里的下标；轴不在 `axis` 里时失败。
FailureOr<size_t> axisPosition(TensorDescAttr descriptor, int64_t axisId);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_VECTORSEGMENTATION_H
