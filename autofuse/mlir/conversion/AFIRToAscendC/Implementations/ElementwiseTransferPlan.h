/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details on how to use this file.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_

#include "dialect/AFIR/Analysis/TransferPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ascend::kernel::elementwise {

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
// 参与比较的是这个算子**有形状的那些输入**加上 destination，由操作数类型现场认出来，
// 不由调用方声明形态。声明形态在 `ScalarOperand::Any` 上认不出谁是 tile（标量可以落在
// 第一个操作数上），而形态与实际操作数对不对得上是 `getTensorTypes` / `emissionOrder`
// 那一处的判据，不在这里重复一份——重复的那份会先于它答话，把「形态不符」报成别的理由。
LocalViewCompatibility checkElementwiseLocalViews(Operation *op, afir::TransferPlanKind *kind = nullptr);

}  // namespace mlir::ascend::kernel::elementwise

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_IMPLEMENTATIONS_ELEMENTWISETRANSFERPLAN_H_
