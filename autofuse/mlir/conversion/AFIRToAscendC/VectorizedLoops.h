/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- VectorizedLoops.h - Materialize collapsed vectorized axes -*- C++ -*-===//
//
// 把 `collapseVectorizedAxes` 的分段结果落成 IR：除最内若干段之外的每一段生成一层
// `scf.for`，各 tile 偏移到本迭代的起点，最内那几段的元素数交给 API。
//
// 这是 legacy `CreateComputeNodeOuterFor`（`codegen/api_call/utils/api_call_utils.cpp:329`）
// 在 MLIR 侧的对应物，偏移算式与它的 `CalcInnerOffset` 是同一个：`Σ iv_j × stride_j`。
//
// **为什么落在 conversion 而不是 `LoopGenerate`**：分段是 per-op 的，而 `LoopGenerate` 的
// 循环按 `sched.axis` 前缀在相邻 op 之间**共享**。要在那里做，得先引入「某个 op 私有的
// 循环」这个概念，并为新段造轴 id 与 `AxisAttr`。另一条同样要紧：`AFIRPipelineSync` 在
// conversion 之前跑完，Enque / DeQue 与事件对已经排在本 op 周围，新循环只能包住计算调用
// 本身——在 conversion 层建循环天然满足这一条。
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_VECTORIZEDLOOPS_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_VECTORIZEDLOOPS_H

#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"

namespace mlir::ascend::kernel {

struct VectorizedWindow {
  // 与 `VectorPlan::tensors` 一一对应：已偏移到本迭代起点的 tile。
  SmallVector<Value> tiles;
  // 最内 `apiWindowSize` 段的元素数，外→内。**含尾块修正**，见下。
  SmallVector<Value> extents;
};

// 为 `plan` 里除最内 `apiWindowSize` 段之外的每一段生成一层 `scf.for`。
//
// `tiles` 按 `plan.tensors` 的顺序给出各张量在目标方言里的值（调用方从 adaptor 取，
// 用 `VectorTensor::side` / `::index` 对位）。
//
// **尾块**：切分轴（`TransferPlan::tileInnerAxis`）在最后一个 tile 上的
// 实际长度是尾块长度而不是轴长，这个值由 `materializeScheduledTileIndices` 算出。本函数把
// 它替换进**含该轴的那一段**的 extent 里——所以段 extent 是按 `VectorSegment::axisExtents`
// 逐轴相乘得到的，不是直接用 `VectorSegment::extent`。少了这一步，尾块那一趟会多算。
//
// **副作用**：返回时 `rewriter` 的插入点在最内层**新建**循环的体内（终结符之前）。
// 没有新建循环时插入点不变。调用方随后就在那里建 API 调用。这不是疏忽——外层段的循环体
// 正是 API 调用该待的地方。
//
// 取 `OpBuilder &` 而不是 `ConversionPatternRewriter &`：本函数只用到 `create` 与
// `setInsertionPoint`，而 `OpBuilder::create` 走的是 listener，传一个
// `ConversionPatternRewriter` 进来照样进它的建档。换成基类只为一件事——UT 里造得出
// 调用方。`materializeScheduledTileIndices` 也是这个签名。
//
// **一段都没有**（`plan.segments` 为空，即每根向量化轴的 extent 都是 1）时，`apiWindowSize`
// 为 1 的调用方拿到的是一个**单元素窗口**：不建循环、各 tile 偏移常量 0、`extents` 是常量 1。
// 这是搬运侧 `materializeScheduledTileIndices` 收的同一档形状，不是对空计划的圆场。
// 要两段以上窗口的调用方在这个形状上仍旧拿到 `failure()`——空段答不出各段在 shape 数组里
// 占多少。
//
// `plan.segments.size() < apiWindowSize`（非空时）、`tiles` 与 `plan.tensors` 不等长、
// 尺寸表达式物化不了，都返回 `failure()`。
FailureOr<VectorizedWindow> materializeVectorizedSegments(Operation *op, OpBuilder &rewriter,
                                                          const afir::VectorPlan &plan,
                                                          const afir::TransferPlan &transferPlan, ArrayRef<Value> tiles,
                                                          unsigned apiWindowSize);

// 按 `VectorTensor` 的出处把 tile 值排成 `plan.tensors` 的顺序。
//
// `inputs` 是 adaptor 的输入操作数，与 `afir.operand_descs` 同序（`getOperandPartition`
// 的 `inputCount` 就是这么划的）；`destination` 是目的地。
//
// 出处取不到（下标越界）返回 `failure()`，不静默跳过——跳过会让 `tiles` 与 `tensors`
// 错位，之后按下标取 stride 就取到别的张量头上了。
FailureOr<SmallVector<Value>> orderTilesForPlan(const afir::VectorPlan &plan, ArrayRef<Value> inputs,
                                                Value destination);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_VECTORIZEDLOOPS_H
