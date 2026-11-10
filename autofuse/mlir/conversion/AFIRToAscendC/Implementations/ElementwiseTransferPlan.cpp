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
  // 按**是不是标量**挑输入，不按下标挑。标量可以落在第一个操作数上——`Add(two, load0)`
  // 这种图 legacy 靠改图消化，MLIR 这边是 `ScalarOperand::Any` 那一档，发射时才把有形状
  // 的那个挪到第一个实参（`emissionOrder`）。按下标取第 0 个会取到标量那一侧：它没有
  // 可比的局部视图，判出来是 MissingInformation，整个算子于是报「没有可用实现」。
  //
  // 判据写成「不是标量」而不是「是某个张量类型」：这道检查在 memref 形态（能力查询）和
  // 目标方言形态（改写）上都要成立，两处的张量类型不是同一个。
  //
  // 挑出来的块数**不与注册形态对账**。对账是 `getTensorTypes` / `emissionOrder` 的事，
  // 在那之前先答一次会把「形态不符」报成 MissingInformation——标量形态的 implementation
  // 收到两块 tile 时本该由「没有向量核」之类更靠前的判据先答话。
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
