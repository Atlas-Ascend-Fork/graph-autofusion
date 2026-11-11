/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 分段循环属于单个 op，而 LoopGenerate 按 sched.axis 前缀共享相邻 op 的循环。
// 放在 conversion 可避免为私有分段循环新增轴身份，并确保循环只包围计算调用。
// AFIRPipelineSync 已在之前插入 Enque / DeQue 和事件对，不能将这些操作包入新循环。

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
  // 最内 `apiWindowSize` 段的元素数，外→内。含尾块修正，见下。
  SmallVector<Value> extents;
};

// 为最内 apiWindowSize 段之外的各段建立 scf.for。
// tiles 按 plan.tensors 排列，由调用方按 VectorTensor 的 side / index 从 adaptor 取值。
// 尾块使用 materializeScheduledTileIndices 得到的 tileInnerAxis 实际长度，
// 并按 axisExtents 逐轴相乘计算段 extent，不能直接使用完整 VectorSegment::extent。
//
// 返回时插入点位于最内新建循环的终结符之前，供调用方生成 API；未建循环时保持插入点。
// 接口只需 OpBuilder 的 create / setInsertionPoint；传入 ConversionPatternRewriter
// 仍通过其 listener 跟踪新算子，也便于 UT 直接调用，与 materializeScheduledTileIndices 一致。
//
// 空 segments 且 apiWindowSize 为 1 时返回偏移 0、extent 1 的单元素窗口，不建循环。
// 空计划请求多段窗口、非空段数不足、tiles 数量不匹配或尺寸表达式无法物化时返回 failure。
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
