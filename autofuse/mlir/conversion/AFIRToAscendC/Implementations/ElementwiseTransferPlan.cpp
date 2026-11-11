/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details on how to use this file.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::afir;

namespace mlir::ascend::kernel::elementwise {

bool isSingletonTensorInput(Operation *op, unsigned index) {
  auto partition = afir::getOperandPartition(op);
  if (failed(partition) || index >= partition->inputCount ||
      !isa<ascendc::LocalTensorType>(partition->inputs(op)[index].getType())) {
    return false;
  }
  auto descriptor = getOperandDescriptor(op, index);
  auto destination = getDestinationDescriptor(op);
  if (failed(descriptor) || failed(destination) || descriptor->getAxis() != destination->getAxis() ||
      descriptor->getVectorizedAxis().empty() || descriptor->getVectorizedAxis() != destination->getVectorizedAxis() ||
      descriptor->getRepeats().empty() || descriptor->getRepeats().size() != descriptor->getAxis().size() ||
      descriptor->getRepeats() == destination->getRepeats()) {
    return false;
  }
  return llvm::all_of(descriptor->getRepeats(),
                      [](afir::ExprAttr expr) { return afir::ProveExpressionsEqual(expr.getValue().getValue(), "1"); });
}

FailureOr<TensorDescAttr> getOperandDescriptor(Operation *op, unsigned index) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  if (!descriptors || index >= descriptors.size()) {
    return failure();
  }
  return dyn_cast<TensorDescAttr>(descriptors[index]);
}

FailureOr<TensorDescAttr> getDestinationDescriptor(Operation *op, unsigned index) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  if (!descriptors || index >= descriptors.size()) {
    return failure();
  }
  return dyn_cast<TensorDescAttr>(descriptors[index]);
}

LocalViewCompatibility checkLocalViewCompatibility(Operation *op, ArrayRef<TensorDescAttr> descriptors,
                                                   TransferPlanKind *kind) {
  if (!op || !hasCanonicalDescriptors(op)) {
    return LocalViewCompatibility::MissingInformation;
  }
  if (descriptors.empty()) {
    return LocalViewCompatibility::MissingInformation;
  }

  SmallVector<TransferPlan, 3> plans;
  plans.reserve(descriptors.size());
  for (TensorDescAttr descriptor : descriptors) {
    if (!descriptor) {
      return LocalViewCompatibility::MissingInformation;
    }
    // 建不出计划一律是「标注缺失」：寻址侧没有形状天花板，失败只在标注破损时发生。
    // 形状不兼容要到下面比较各方视图时才判得出来。
    FailureOr<TransferPlan> plan = buildTransferPlanForDescriptor(op, descriptor);
    if (failed(plan)) {
      return LocalViewCompatibility::MissingInformation;
    }
    plans.push_back(*plan);
  }

  const TransferPlan &reference = plans.front();
  for (const TransferPlan &plan : llvm::drop_begin(plans)) {
    if (plan.kind != reference.kind || plan.dims.size() != reference.dims.size()) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    if (plan.kind == TransferPlanKind::Identity) {
      continue;
    }
    for (const auto [referenceDim, dim] : llvm::zip_equal(reference.dims, plan.dims)) {
      if (proveEqual(referenceDim.extent, dim.extent) != afir::detail::TruthValue::True ||
          proveEqual(referenceDim.localStride, dim.localStride) != afir::detail::TruthValue::True) {
        return LocalViewCompatibility::UnsupportedShape;
      }
    }
  }
  if (kind) {
    *kind = reference.kind;
  }
  return LocalViewCompatibility::Compatible;
}

LocalViewCompatibility checkElementwiseLocalViews(Operation *op, TransferPlanKind *kind) {
  if (!op) {
    return LocalViewCompatibility::MissingInformation;
  }
  FailureOr<OperandPartition> partition = getOperandPartition(op);
  if (failed(partition)) {
    return LocalViewCompatibility::MissingInformation;
  }
  // 根据实际操作数是否为标量选择 tile，不固定取第 0 个输入。ScalarOperand::Any
  // 允许标量在左侧，例如 Add(two, load0)，生成时才由 emissionOrder 调整实参顺序。
  // 此检查同时用于 memref 能力查询和目标方言改写，因此排除标量，而不限定某种张量类型。
  // 这里只选取有局部视图的输入，不重复检查注册形态；形态匹配由 getTensorTypes /
  // emissionOrder 负责，避免提前把形态错误报告为 MissingInformation。
  SmallVector<TensorDescAttr, 3> descriptors;
  for (auto [index, input] : llvm::enumerate(partition->inputs(op))) {
    if (input.getType().isIntOrIndexOrFloat()) {
      continue;
    }
    FailureOr<TensorDescAttr> descriptor = getOperandDescriptor(op, static_cast<unsigned>(index));
    if (failed(descriptor)) {
      return LocalViewCompatibility::MissingInformation;
    }
    descriptors.push_back(*descriptor);
  }
  FailureOr<TensorDescAttr> destination = getDestinationDescriptor(op);
  if (failed(destination)) {
    return LocalViewCompatibility::MissingInformation;
  }
  descriptors.push_back(*destination);
  return checkLocalViewCompatibility(op, descriptors, kind);
}

}  // namespace mlir::ascend::kernel::elementwise
