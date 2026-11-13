/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISELOCALVIEW_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISELOCALVIEW_H_

#include "dialect/AFIR/Analysis/ComputeExtent.h"

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel::elementwise {

// 行广播形态：二元 elementwise 的一侧是 `(1, A)`、另一侧是 `(B, A)`，判据与 legacy
// `IsGeneralizeBrcInlineScene`（`common/common_utils.cpp:856`）等价。设计见
// docs/development/2026-11-10-afir-broadcast-inline-design.md 功能需求 1。
struct BroadcastInlineOperands {
  // AFIR 操作数下标（0 或 1），不是发射顺序——这个判据不知道哪一侧会被挪到第一个
  // 实参，那是 pattern 的事。
  unsigned broadcastOperandIndex;
  // 非广播侧合并后的两组 repeat：outer（legacy 的 "B"，`shape0`）、
  // inner（legacy 的 "A"，`shape1`）。
  afir::ExprAttr outerRepeat;
  afir::ExprAttr innerRepeat;
  // `first_axis_v_stride`：沿广播侧自身的向量化轴走，在第一个 repeat 不为 1 的位置之前，
  // 取目的地在同一位置上的 `vectorized_strides` 里最后一个非零值。
  afir::ExprAttr firstAxisVStride;
};

// 两个输入都不是标量时才可能命中；标量形态走的是另一组 implementation。失败（含「不是
// 这个形态」）统一答 `std::nullopt`，调用方据此回落到别的判据或整体返回不支持。
std::optional<BroadcastInlineOperands> matchBroadcastInlineOperands(Operation *op);

enum class LocalViewCompatibility {
  Compatible,
  MissingInformation,
  UnsupportedShape,
};

FailureOr<afir::TensorDescAttr> getOperandDescriptor(Operation *op, unsigned index = 0);
FailureOr<afir::TensorDescAttr> getDestinationDescriptor(Operation *op, unsigned index = 0);

// A missing plan denotes the explicit minimal rank-one whole-tensor contract.
// A present plan with no segments denotes a single logical element.
struct ElementwiseLocalView {
  std::optional<afir::VectorPlan> vectorPlan;
  afir::ExtentSource extentSource = afir::ExtentSource::Nominal;
};
// Interpret descriptors and construct the common loop plan, without imposing
// continuous API lane, alignment, capacity or alias requirements.
LocalViewCompatibility analyzeElementwiseLocalView(Operation *op, ArrayRef<unsigned> scalarInputs, bool compactTail,
                                                   ElementwiseLocalView &view);
// Continuous elementwise API contract: equal vector lane domains, independent
// aligned addresses, known static bounds and conservative shared-storage checks.
// Dynamic storage sufficiency remains the allocation/caller contract.
LocalViewCompatibility prepareElementwiseLocalView(Operation *op, ArrayRef<unsigned> scalarInputs, bool compactTail,
                                                   ElementwiseLocalView &view);
LocalViewCompatibility checkLocalViewCompatibility(Operation *op, ArrayRef<afir::TensorDescAttr> descriptors,
                                                   bool *wholeTensor = nullptr);
bool isSingletonTensorInput(Operation *op, unsigned index);
bool hasSameLocalLayout(Operation *op, unsigned inputIndex);
LocalViewCompatibility checkElementwiseLocalViews(Operation *op, bool *wholeTensor = nullptr);

}  // namespace mlir::ascend::kernel::elementwise
#endif
