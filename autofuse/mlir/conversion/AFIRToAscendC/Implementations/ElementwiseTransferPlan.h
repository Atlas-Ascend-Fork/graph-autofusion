/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_

#include "dialect/AFIR/Analysis/TransferPlan.h"

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

// `kind` 是可选出参：判定为 `Compatible` 时写入参与比较的那一组计划的共同种类，其余
// 情况不写。调用方要它，是因为「未切 tile」与「已切 tile」两种形态对 LocalTensor 的秩
// 要求不同，而这个区分只有计划知道。
LocalViewCompatibility checkLocalViewCompatibility(Operation *op, ArrayRef<afir::TensorDescAttr> descriptors,
                                                   afir::TransferPlanKind *kind = nullptr);
// A logical singleton that must be widened to the destination domain.
bool isSingletonTensorInput(Operation *op, unsigned index);

// 参与比较的是这个算子**有形状的那些输入**加上 destination，由操作数类型现场认出来，
// 不由调用方声明形态。声明形态在 `ScalarOperand::Any` 上认不出谁是 tile（标量可以落在
// 第一个操作数上），而形态与实际操作数对不对得上是 `getTensorTypes` / `emissionOrder`
// 那一处的判据，不在这里重复一份——重复的那份会先于它答话，把「形态不符」报成别的理由。
LocalViewCompatibility checkElementwiseLocalViews(Operation *op, afir::TransferPlanKind *kind = nullptr);

}  // namespace mlir::ascend::kernel::elementwise

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_
