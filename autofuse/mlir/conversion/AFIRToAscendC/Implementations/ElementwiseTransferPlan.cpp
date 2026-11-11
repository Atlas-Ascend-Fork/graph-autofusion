/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::afir;

namespace mlir::ascend::kernel::elementwise {

namespace {

bool isProvenEqualToLiteral(ExprAttr expr, StringRef literal) {
  return expr && ProveExpressionsEqual(expr.getValue().getValue(), literal);
}

// `lhs * rhs` 的规范形式，与 `TransferPlan.cpp` 里同名的私有版本做同一件事，但这里要
// 导出给 `matchBroadcastInlineOperands` 用——两处各自是私有实现，不是共用的理由见
// `TransferPlan.cpp` 的注释（该函数也是文件内 `static`）。
FailureOr<ExprAttr> multiplyExpressionsHere(MLIRContext *context, ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return failure();
  }
  Result<std::string> product = MultiplyExpressions(lhs.getValue().getValue(), rhs.getValue().getValue());
  if (!product) {
    return failure();
  }
  return makeExprAttr(context, product.value());
}

// 合并连续的广播轴和非广播轴，与 legacy `MergeBrcAxisRepeats`
// （`common/common_utils.cpp:789`）逐步对齐，包括它的两个怪癖：
//
//   - 两侧的「最内轴要求对齐」那一步都读**同一张** `otherStrides`
//     表——legacy 的两个 `MergeBrcAxisParams` 构造时传的是同一个实参，不是各自的步长表；
//   - 合并后的「最后一组」不是整组 repeat 的累乘，只取该组**第一个**位置的
//     `repeat * stride`——legacy 的 for 循环只填到 `size()-2`，最后一格由循环外单独一行
//     赋值覆盖。这与直觉不符，但在今天唯一验证过的形态（每组恰好一根轴）里两者等价，
//     而本函数的职责是复刻 legacy，不是重新设计这条合轴规则。
//
// 剔除「两侧都是 1」的轴之后两侧必然等长；返回的两个数组也必然等长（同一份 partition）。
FailureOr<std::pair<SmallVector<ExprAttr>, SmallVector<ExprAttr>>> mergeBrcAxisRepeats(
    MLIRContext *context, ArrayRef<ExprAttr> repeats0, ArrayRef<ExprAttr> repeats1, ArrayRef<ExprAttr> otherStrides) {
  FailureOr<ExprAttr> one = makeExprAttr(context, "1");
  if (failed(one)) {
    return failure();
  }
  SmallVector<ExprAttr> repeatsNoOne0, repeatsNoOne1, stridesNoOne0, stridesNoOne1;
  SmallVector<bool> isAxisBrc0, isAxisBrc1;
  for (auto [repeat0, repeat1, stride] : llvm::zip_equal(repeats0, repeats1, otherStrides)) {
    if (isProvenEqualToLiteral(repeat0, "1") && isProvenEqualToLiteral(repeat1, "1")) {
      continue;
    }
    repeatsNoOne0.push_back(repeat0);
    repeatsNoOne1.push_back(repeat1);
    ExprAttr strideOrOne = isProvenEqualToLiteral(stride, "0") ? *one : stride;
    stridesNoOne0.push_back(strideOrOne);
    stridesNoOne1.push_back(strideOrOne);
    const bool repeatsDiffer = !ProveExpressionsEqual(repeat0.getValue().getValue(), repeat1.getValue().getValue());
    isAxisBrc0.push_back(repeatsDiffer && isProvenEqualToLiteral(repeat0, "1"));
    isAxisBrc1.push_back(repeatsDiffer && isProvenEqualToLiteral(repeat1, "1"));
  }
  SmallVector<ExprAttr> merged0, merged1;
  if (repeatsNoOne0.empty()) {
    merged0.push_back(*one);
    merged1.push_back(*one);
    return std::make_pair(merged0, merged1);
  }

  bool previousIsSame =
      ProveExpressionsEqual(repeatsNoOne0[0].getValue().getValue(), repeatsNoOne1[0].getValue().getValue());
  SmallVector<size_t> partition = {0};
  for (size_t i = 1; i < repeatsNoOne0.size(); ++i) {
    const bool currentIsSame =
        ProveExpressionsEqual(repeatsNoOne0[i].getValue().getValue(), repeatsNoOne1[i].getValue().getValue());
    const bool canMerge0 = (isAxisBrc0[i - 1] == isAxisBrc0[i]);
    const bool canMerge1 = (isAxisBrc1[i - 1] == isAxisBrc1[i]);
    if (previousIsSame != currentIsSame || !canMerge0 || !canMerge1) {
      partition.push_back(i);
      previousIsSame = currentIsSame;
    }
  }
  merged0.resize(partition.size());
  merged1.resize(partition.size());

  size_t start = 0;
  for (size_t i = 1; i < partition.size(); ++i) {
    const size_t end = partition[i];
    ExprAttr acc0 = *one;
    ExprAttr acc1 = *one;
    for (size_t j = start; j < end; ++j) {
      FailureOr<ExprAttr> next0 = multiplyExpressionsHere(context, acc0, repeatsNoOne0[j]);
      FailureOr<ExprAttr> next1 = multiplyExpressionsHere(context, acc1, repeatsNoOne1[j]);
      if (failed(next0) || failed(next1)) {
        return failure();
      }
      acc0 = *next0;
      acc1 = *next1;
    }
    merged0[i - 1] = acc0;
    merged1[i - 1] = acc1;
    start = end;
  }
  if (start < stridesNoOne0.size()) {
    FailureOr<ExprAttr> last0 = multiplyExpressionsHere(context, repeatsNoOne0[start], stridesNoOne0[start]);
    FailureOr<ExprAttr> last1 = multiplyExpressionsHere(context, repeatsNoOne1[start], stridesNoOne1[start]);
    if (failed(last0) || failed(last1)) {
      return failure();
    }
    merged0.back() = *last0;
    merged1.back() = *last1;
  }
  return std::make_pair(merged0, merged1);
}

// 取某个 descriptor 在各向量化轴位置上的 repeat，按 `vectorizedAxis` 的顺序排列——
// 与 legacy `i0_vectorized_axis_pos` / `i1_vectorized_axis_pos` 同一件事。
FailureOr<SmallVector<ExprAttr>> vectorizedRepeats(TensorDescAttr descriptor) {
  SmallVector<ExprAttr> result;
  for (int64_t axisId : descriptor.getVectorizedAxis()) {
    const auto *found = llvm::find(descriptor.getAxis(), axisId);
    if (found == descriptor.getAxis().end()) {
      return failure();
    }
    const size_t index = std::distance(descriptor.getAxis().begin(), found);
    if (index >= descriptor.getRepeats().size()) {
      return failure();
    }
    result.push_back(descriptor.getRepeats()[index]);
  }
  return result;
}

}  // namespace

std::optional<BroadcastInlineOperands> matchBroadcastInlineOperands(Operation *op) {
  FailureOr<TensorDescAttr> operand0 = getOperandDescriptor(op, 0);
  FailureOr<TensorDescAttr> operand1 = getOperandDescriptor(op, 1);
  FailureOr<TensorDescAttr> destination = getDestinationDescriptor(op);
  if (failed(operand0) || failed(operand1) || failed(destination)) {
    return std::nullopt;
  }
  if (operand0->getVectorizedAxis().size() != operand1->getVectorizedAxis().size() ||
      operand0->getVectorizedAxis().empty()) {
    return std::nullopt;
  }
  FailureOr<SmallVector<ExprAttr>> repeats0 = vectorizedRepeats(*operand0);
  FailureOr<SmallVector<ExprAttr>> repeats1 = vectorizedRepeats(*operand1);
  if (failed(repeats0) || failed(repeats1)) {
    return std::nullopt;
  }

  bool operand0HasBrcAxis = false;
  bool operand1HasBrcAxis = false;
  for (auto [repeat0, repeat1] : llvm::zip_equal(*repeats0, *repeats1)) {
    operand0HasBrcAxis |= isProvenEqualToLiteral(repeat0, "1") && !isProvenEqualToLiteral(repeat1, "1");
    operand1HasBrcAxis |= isProvenEqualToLiteral(repeat1, "1") && !isProvenEqualToLiteral(repeat0, "1");
  }
  if (operand0HasBrcAxis == operand1HasBrcAxis) {
    // 两侧都有广播轴、或都没有：legacy 两种都返回不支持。
    return std::nullopt;
  }
  const unsigned broadcastIndex = operand0HasBrcAxis ? 0 : 1;
  const unsigned otherIndex = 1 - broadcastIndex;
  ArrayRef<ExprAttr> broadcastRepeats = broadcastIndex == 0 ? ArrayRef(*repeats0) : ArrayRef(*repeats1);
  ArrayRef<ExprAttr> otherRepeats = broadcastIndex == 0 ? ArrayRef(*repeats1) : ArrayRef(*repeats0);
  TensorDescAttr otherDescriptor = otherIndex == 0 ? *operand0 : *operand1;
  if (otherDescriptor.getVectorizedStrides().size() != otherRepeats.size()) {
    return std::nullopt;
  }

  MLIRContext *context = op->getContext();
  FailureOr<std::pair<SmallVector<ExprAttr>, SmallVector<ExprAttr>>> merged =
      mergeBrcAxisRepeats(context, broadcastRepeats, otherRepeats, otherDescriptor.getVectorizedStrides());
  if (failed(merged)) {
    return std::nullopt;
  }
  const auto &[brcMerged, otherMerged] = *merged;
  if (brcMerged.size() != 2 || otherMerged.size() != 2) {
    return std::nullopt;
  }
  if (!isProvenEqualToLiteral(brcMerged[0], "1") && !isProvenEqualToLiteral(otherMerged[0], "1")) {
    return std::nullopt;
  }

  // `first_axis_v_stride`：沿广播侧自身的向量化轴走，读目的地同一位置的
  // `vectorized_strides`，在第一个 repeat 不为 1 的位置之前，留最后一个非零值。
  if (destination->getVectorizedStrides().size() != broadcastRepeats.size()) {
    return std::nullopt;
  }
  FailureOr<ExprAttr> zero = makeExprAttr(context, "0");
  if (failed(zero)) {
    return std::nullopt;
  }
  ExprAttr firstAxisVStride = *zero;
  for (auto [repeat, destinationStride] : llvm::zip_equal(broadcastRepeats, destination->getVectorizedStrides())) {
    if (!isProvenEqualToLiteral(repeat, "1")) {
      break;
    }
    if (!isProvenEqualToLiteral(destinationStride, "0")) {
      firstAxisVStride = destinationStride;
    }
  }

  BroadcastInlineOperands result;
  result.broadcastOperandIndex = broadcastIndex;
  result.outerRepeat = otherMerged[0];
  result.innerRepeat = otherMerged[1];
  result.firstAxisVStride = firstAxisVStride;
  return result;
}

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
