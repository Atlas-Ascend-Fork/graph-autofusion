/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details on how to use this file.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ElementwiseLocalView.h"

#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"
#include "dialect/AFIR/Analysis/BufferAlias.h"
#include "dialect/AFIR/Analysis/ResourceIds.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using namespace mlir::afir;

namespace mlir::ascend::kernel::elementwise {

namespace {

bool isProvenEqualToLiteral(ExprAttr expr, StringRef literal) {
  return expr && ProveExpressionsEqual(expr.getValue().getValue(), literal);
}

// `mergeBrcAxisRepeats` 合并后必须恰好剩外、内两组，对应 `BroadcastInlineOperands` 的
// `outerRepeat`/`innerRepeat` 两个字段，也是底层 AscendC broadcast API 的 `shape0`/
// `shape1` 两个形参——合并结果组数超过这个值就超出了该 API 能表达的形态。
constexpr size_t kInlineBroadcastMergedAxisCount = 2;

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

// `mergeBrcAxisRepeats` 第一阶段的结果：剔除「两侧都是 1」的轴之后的 repeat/步长表，
// 以及每根保留轴在哪一侧是广播轴（`isAxisBrc*`，供第二阶段判断能否与前一根轴合并）。
struct FilteredBrcAxisRepeats {
  SmallVector<ExprAttr> repeatsNoOne0, repeatsNoOne1;
  SmallVector<ExprAttr> stridesNoOne0, stridesNoOne1;
  SmallVector<bool> isAxisBrc0, isAxisBrc1;
};

// 剔除两侧 repeat 都为 1 的轴；步长为 0（该轴不产生地址偏移）按 1 处理，配合调用方把
// `repeat * stride` 当缩放因子使用。两侧的「最内轴要求对齐」那一步都读**同一张**
// `otherStrides` 表——legacy 的两个 `MergeBrcAxisParams` 构造时传的是同一个实参，不是
// 各自的步长表。
FailureOr<FilteredBrcAxisRepeats> filterBroadcastAxisRepeats(MLIRContext *context, ArrayRef<ExprAttr> repeats0,
                                                             ArrayRef<ExprAttr> repeats1,
                                                             ArrayRef<ExprAttr> otherStrides) {
  FailureOr<ExprAttr> one = makeExprAttr(context, "1");
  if (failed(one)) {
    return failure();
  }
  FilteredBrcAxisRepeats result;
  for (auto [repeat0, repeat1, stride] : llvm::zip_equal(repeats0, repeats1, otherStrides)) {
    if (isProvenEqualToLiteral(repeat0, "1") && isProvenEqualToLiteral(repeat1, "1")) {
      continue;
    }
    result.repeatsNoOne0.push_back(repeat0);
    result.repeatsNoOne1.push_back(repeat1);
    ExprAttr strideOrOne = isProvenEqualToLiteral(stride, "0") ? *one : stride;
    result.stridesNoOne0.push_back(strideOrOne);
    result.stridesNoOne1.push_back(strideOrOne);
    const bool repeatsDiffer = !ProveExpressionsEqual(repeat0.getValue().getValue(), repeat1.getValue().getValue());
    result.isAxisBrc0.push_back(repeatsDiffer && isProvenEqualToLiteral(repeat0, "1"));
    result.isAxisBrc1.push_back(repeatsDiffer && isProvenEqualToLiteral(repeat1, "1"));
  }
  return result;
}

// 合并连续的广播轴和非广播轴，与 legacy `MergeBrcAxisRepeats`
// （`common/common_utils.cpp:789`）逐步对齐，包括它的一个怪癖：合并后的「最后一组」不是
// 整组 repeat 的累乘，只取该组**第一个**位置的 `repeat * stride`——legacy 的 for 循环只
// 填到 `size()-2`，最后一格由循环外单独一行赋值覆盖。这与直觉不符，但在今天唯一验证过的
// 形态（每组恰好一根轴）里两者等价，而本函数的职责是复刻 legacy，不是重新设计这条合轴规则。
// 剔除「两侧都是 1」的轴之后两侧必然等长；返回的两个数组也必然等长（同一份 partition）。
FailureOr<std::pair<SmallVector<ExprAttr>, SmallVector<ExprAttr>>> mergeBrcAxisRepeats(
    MLIRContext *context, ArrayRef<ExprAttr> repeats0, ArrayRef<ExprAttr> repeats1, ArrayRef<ExprAttr> otherStrides) {
  FailureOr<ExprAttr> one = makeExprAttr(context, "1");
  if (failed(one)) {
    return failure();
  }
  FailureOr<FilteredBrcAxisRepeats> filtered = filterBroadcastAxisRepeats(context, repeats0, repeats1, otherStrides);
  if (failed(filtered)) {
    return failure();
  }
  const auto &[repeatsNoOne0, repeatsNoOne1, stridesNoOne0, stridesNoOne1, isAxisBrc0, isAxisBrc1] = *filtered;

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

// 两个操作数里哪一个带广播轴（恰好一个有）：按向量化轴逐一比较 repeat，`1` 对 `非 1` 的
// 一侧是广播侧。两侧都有、或都没有时 legacy 一律回退，所以这里也返回空。
std::optional<unsigned> findBroadcastOperandIndex(ArrayRef<ExprAttr> repeats0, ArrayRef<ExprAttr> repeats1) {
  bool operand0HasBrcAxis = false;
  bool operand1HasBrcAxis = false;
  for (auto [repeat0, repeat1] : llvm::zip_equal(repeats0, repeats1)) {
    operand0HasBrcAxis |= isProvenEqualToLiteral(repeat0, "1") && !isProvenEqualToLiteral(repeat1, "1");
    operand1HasBrcAxis |= isProvenEqualToLiteral(repeat1, "1") && !isProvenEqualToLiteral(repeat0, "1");
  }
  if (operand0HasBrcAxis == operand1HasBrcAxis) {
    return std::nullopt;
  }
  return operand0HasBrcAxis ? 0 : 1;
}

// `first_axis_v_stride`：沿广播侧自身的向量化轴走，读目的地同一位置的
// `vectorized_strides`，在第一个 repeat 不为 1 的位置之前，留最后一个非零值。
FailureOr<ExprAttr> computeFirstAxisVStride(MLIRContext *context, ArrayRef<ExprAttr> broadcastRepeats,
                                            TensorDescAttr destination) {
  if (destination.getVectorizedStrides().size() != broadcastRepeats.size()) {
    return failure();
  }
  FailureOr<ExprAttr> zero = makeExprAttr(context, "0");
  if (failed(zero)) {
    return failure();
  }
  ExprAttr firstAxisVStride = *zero;
  for (auto [repeat, destinationStride] : llvm::zip_equal(broadcastRepeats, destination.getVectorizedStrides())) {
    if (!isProvenEqualToLiteral(repeat, "1")) {
      break;
    }
    if (!isProvenEqualToLiteral(destinationStride, "0")) {
      firstAxisVStride = destinationStride;
    }
  }
  return firstAxisVStride;
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

  std::optional<unsigned> broadcastIndexOrNone = findBroadcastOperandIndex(*repeats0, *repeats1);
  if (!broadcastIndexOrNone) {
    // 两侧都有广播轴、或都没有：legacy 两种都返回不支持。
    return std::nullopt;
  }
  const unsigned broadcastIndex = *broadcastIndexOrNone;
  const unsigned otherIndex = 1 - broadcastIndex;
  ArrayRef<ExprAttr> broadcastRepeats = broadcastIndex == 0 ? ArrayRef(*repeats0) : ArrayRef(*repeats1);
  ArrayRef<ExprAttr> otherRepeats = broadcastIndex == 0 ? ArrayRef(*repeats1) : ArrayRef(*repeats0);
  TensorDescAttr broadcastDescriptor = broadcastIndex == 0 ? *operand0 : *operand1;
  TensorDescAttr otherDescriptor = otherIndex == 0 ? *operand0 : *operand1;
  if (otherDescriptor.getVectorizedStrides().size() != otherRepeats.size()) {
    return std::nullopt;
  }
  // `computeFirstAxisVStride` 按位置把 `broadcastRepeats`（广播侧的 `vectorized_axis`
  // 顺序）与 `destination.getVectorizedStrides()` 配对；只有两者的 `vectorized_axis`
  // 完全一致（根数和顺序都一致）时这种按位置配对才成立。destination 的 `vectorized_axis`
  // 是独立属性，调度允许其与广播侧重新排序，不能假设天然对齐。
  if (destination->getVectorizedAxis() != broadcastDescriptor.getVectorizedAxis()) {
    return std::nullopt;
  }

  MLIRContext *context = op->getContext();
  FailureOr<std::pair<SmallVector<ExprAttr>, SmallVector<ExprAttr>>> merged =
      mergeBrcAxisRepeats(context, broadcastRepeats, otherRepeats, otherDescriptor.getVectorizedStrides());
  if (failed(merged)) {
    return std::nullopt;
  }
  const auto &[brcMerged, otherMerged] = *merged;
  if (brcMerged.size() != kInlineBroadcastMergedAxisCount || otherMerged.size() != kInlineBroadcastMergedAxisCount) {
    return std::nullopt;
  }
  if (!isProvenEqualToLiteral(brcMerged[0], "1") && !isProvenEqualToLiteral(otherMerged[0], "1")) {
    return std::nullopt;
  }

  FailureOr<ExprAttr> firstAxisVStride = computeFirstAxisVStride(context, broadcastRepeats, *destination);
  if (failed(firstAxisVStride)) {
    return std::nullopt;
  }

  BroadcastInlineOperands result;
  result.broadcastOperandIndex = broadcastIndex;
  result.outerRepeat = otherMerged[0];
  result.innerRepeat = otherMerged[1];
  result.firstAxisVStride = *firstAxisVStride;
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
  auto descriptor = dyn_cast<TensorDescAttr>(descriptors[index]);
  if (!descriptor) {
    return failure();
  }
  return descriptor;
}

FailureOr<TensorDescAttr> getDestinationDescriptor(Operation *op, unsigned index) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  if (!descriptors || index >= descriptors.size()) {
    return failure();
  }
  auto descriptor = dyn_cast<TensorDescAttr>(descriptors[index]);
  if (!descriptor) {
    return failure();
  }
  return descriptor;
}

namespace {
constexpr int64_t kDataBlockBits = 32 * 8;

// AFIR alias analysis resolves destination ties and memrefs. At conversion time
// also peel AscendC local views; conservatively treat slices of one TBuf as shared.
Value localStorageRoot(Value value) {
  while (value) {
    Value root = resolveBufferSlice(value).base;
    if (root && root != value) {
      value = root;
    } else if (auto sub = value.getDefiningOp<ascendc::LocalTensorSubIndexOp>()) {
      value = sub.getTensor();
    } else if (auto cast = value.getDefiningOp<ascendc::LocalTensorReinterpretCastOp>()) {
      value = cast.getIn();
    } else if (auto tensor = value.getDefiningOp<ascendc::TBufGetTensorOp>()) {
      return tensor.getBuffer();
    } else if (auto tensor = value.getDefiningOp<ascendc::TBufGetWithOffsetOp>()) {
      return tensor.getBuffer();
    } else {
      return value;
    }
  }
  return {};
}

bool hasUniqueAxes(ArrayRef<int64_t> axes) {
  llvm::SmallDenseSet<int64_t, 4> seen;
  return llvm::all_of(axes, [&](int64_t axis) { return seen.insert(axis).second; });
}

bool sameStrides(TensorDescAttr lhs, TensorDescAttr rhs) {
  if (!lhs || !rhs || lhs.getVectorizedAxis() != rhs.getVectorizedAxis() ||
      lhs.getVectorizedStrides().size() != rhs.getVectorizedStrides().size()) {
    return false;
  }
  return llvm::all_of(llvm::zip(lhs.getVectorizedStrides(), rhs.getVectorizedStrides()),
                      [](auto pair) { return provablyEqual(std::get<0>(pair), std::get<1>(pair)); });
}

// Detect known invalid static geometry. Symbolic capacity is supplied by the same
// allocation/caller contract for equal and unequal layouts; it is not inferred
// from the logical LocalTensor shape.
std::optional<int64_t> staticSpan(TensorDescAttr descriptor) {
  int64_t span = 1;
  for (size_t i = descriptor.getVectorizedAxis().size(); i > 0; --i) {
    auto position = axisPosition(descriptor, descriptor.getVectorizedAxis()[i - 1]);
    if (failed(position)) {
      return std::nullopt;
    }
    auto extent = getLiteralStride(descriptor.getRepeats()[*position]);
    auto stride = getLiteralStride(descriptor.getVectorizedStrides()[i - 1]);
    if (failed(extent) || failed(stride) || *extent <= 0 || *stride < 0) {
      return std::nullopt;
    }
    if (*extent == 1) {
      continue;
    }
    if (*stride < span || *stride > (std::numeric_limits<int64_t>::max() - span) / (*extent - 1)) {
      return std::nullopt;
    }
    span += (*extent - 1) * *stride;
  }
  return span;
}
}  // namespace

bool hasSameLocalLayout(Operation *op, unsigned inputIndex) {
  auto input = getOperandDescriptor(op, inputIndex);
  auto destination = getDestinationDescriptor(op);
  return succeeded(input) && succeeded(destination) && sameStrides(*input, *destination);
}

LocalViewCompatibility checkLocalViewCompatibility(Operation *op, ArrayRef<TensorDescAttr> descriptors,
                                                   bool *wholeTensor) {
  if (!op || !hasCanonicalDescriptors(op) || descriptors.empty() ||
      llvm::any_of(descriptors, [](TensorDescAttr desc) { return !desc; })) {
    return LocalViewCompatibility::MissingInformation;
  }
  const auto reference = descriptors.back();
  const bool whole = reference.getVectorizedAxis().empty();
  for (TensorDescAttr desc : descriptors) {
    if (whole) {
      // Only the explicit minimal tensor contract may use GetSize. A partial
      // scheduled view must never silently fall back to the buffer capacity.
      if (!desc.getVectorizedAxis().empty() || !desc.getVectorizedStrides().empty() || !desc.getRepeats().empty()) {
        return LocalViewCompatibility::MissingInformation;
      }
      continue;
    }
    if (!hasUniqueAxes(desc.getAxis()) || !hasUniqueAxes(desc.getVectorizedAxis()) ||
        desc.getAxis().size() != desc.getRepeats().size() ||
        desc.getVectorizedAxis().size() != desc.getVectorizedStrides().size() || desc.getVectorizedAxis().empty()) {
      return LocalViewCompatibility::MissingInformation;
    }
    if (desc.getVectorizedAxis() != reference.getVectorizedAxis()) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    for (int64_t axis : reference.getVectorizedAxis()) {
      auto a = axisPosition(desc, axis);
      auto b = axisPosition(reference, axis);
      if (failed(a) || failed(b) || *b >= reference.getRepeats().size()) {
        return LocalViewCompatibility::MissingInformation;
      }
    }
  }
  if (wholeTensor) {
    *wholeTensor = whole;
  }
  return LocalViewCompatibility::Compatible;
}

LocalViewCompatibility analyzeElementwiseLocalView(Operation *op, ArrayRef<unsigned> scalarInputs, bool compactTail,
                                                   ElementwiseLocalView &view) {
  if (!op) {
    return LocalViewCompatibility::MissingInformation;
  }
  auto partition = getOperandPartition(op);
  if (failed(partition) || partition->destinationCount != 1) {
    return LocalViewCompatibility::MissingInformation;
  }
  SmallVector<VectorTensor> tensors;
  SmallVector<TensorDescAttr> descriptors;
  SmallVector<Value> values;
  for (auto [index, input] : llvm::enumerate(partition->inputs(op))) {
    if (input.getType().isIntOrIndexOrFloat() || llvm::is_contained(scalarInputs, index)) {
      continue;
    }
    auto desc = getOperandDescriptor(op, index);
    if (failed(desc)) {
      return LocalViewCompatibility::MissingInformation;
    }
    tensors.push_back({VectorTensor::Side::Operand, static_cast<unsigned>(index), *desc});
    descriptors.push_back(*desc);
    values.push_back(input);
  }
  auto destination = getDestinationDescriptor(op);
  if (failed(destination)) {
    return LocalViewCompatibility::MissingInformation;
  }
  tensors.push_back({VectorTensor::Side::Destination, 0, *destination});
  descriptors.push_back(*destination);
  values.push_back(partition->destinations(op)[0]);
  bool whole = false;
  auto compatibility = checkLocalViewCompatibility(op, descriptors, &whole);
  if (compatibility != LocalViewCompatibility::Compatible) {
    return compatibility;
  }
  view.vectorPlan.reset();
  view.extentSource = ExtentSource::Nominal;
  if (whole) {
    for (Value value : values) {
      auto type = dyn_cast<ascendc::LocalTensorType>(value.getType());
      if (!type || !type.hasRank() || type.getRank() != 1) {
        return LocalViewCompatibility::UnsupportedShape;
      }
    }
    return LocalViewCompatibility::Compatible;
  }
  auto plan = collapseVectorizedAxes(op, tensors, compactTail);
  if (failed(plan)) {
    return LocalViewCompatibility::MissingInformation;
  }
  view.vectorPlan = std::move(*plan);
  auto func = op->getParentOfType<func::FuncOp>();
  view.extentSource = func && func->hasAttr(kAscGraphAttrName) ? ExtentSource::Actual : ExtentSource::Nominal;
  return LocalViewCompatibility::Compatible;
}

LocalViewCompatibility prepareElementwiseLocalView(Operation *op, ArrayRef<unsigned> scalarInputs, bool compactTail,
                                                   ElementwiseLocalView &view) {
  auto result = analyzeElementwiseLocalView(op, scalarInputs, compactTail, view);
  if (result != LocalViewCompatibility::Compatible) {
    return result;
  }
  auto partition = getOperandPartition(op);
  SmallVector<TensorDescAttr> descriptors;
  SmallVector<Value> values;
  if (!view.vectorPlan) {
    // GetSize counts the whole destination; every tensor input must have the same shape.
    auto destinationType = cast<ascendc::LocalTensorType>(partition->destinations(op)[0].getType());
    for (auto [index, input] : llvm::enumerate(partition->inputs(op))) {
      if (input.getType().isIntOrIndexOrFloat() || llvm::is_contained(scalarInputs, index)) {
        continue;
      }
      if (cast<ascendc::LocalTensorType>(input.getType()).getShape() != destinationType.getShape()) {
        return LocalViewCompatibility::UnsupportedShape;
      }
    }
    return LocalViewCompatibility::Compatible;
  }
  const auto destinationDesc = view.vectorPlan->tensors.back().descriptor;
  for (const auto &tensor : view.vectorPlan->tensors) {
    auto desc = tensor.descriptor;
    Value value = tensor.side == VectorTensor::Side::Destination ? partition->destinations(op)[tensor.index]
                                                                 : partition->inputs(op)[tensor.index];
    // Continuous elementwise APIs do not repeat lanes inside their API window.
    for (int64_t axis : destinationDesc.getVectorizedAxis()) {
      auto a = axisPosition(desc, axis);
      auto b = axisPosition(destinationDesc, axis);
      if (!provablyEqual(desc.getRepeats()[*a], destinationDesc.getRepeats()[*b])) {
        return LocalViewCompatibility::UnsupportedShape;
      }
    }
    // Match the full logical domain by axis identity, not by LocalTensor shape positions.
    bool sameDomain = desc.getAxis().size() == destinationDesc.getAxis().size();
    for (int64_t axis : destinationDesc.getAxis()) {
      auto a = axisPosition(desc, axis);
      auto b = axisPosition(destinationDesc, axis);
      sameDomain &= succeeded(a) && provablyEqual(desc.getRepeats()[*a], destinationDesc.getRepeats()[*b]);
    }
    if (!sameDomain && (tensor.side == VectorTensor::Side::Destination ||
                        !afir::elementwise::sourceBroadcastsToDestination(op, value, partition->destinations(op)[0]))) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    descriptors.push_back(desc);
    values.push_back(value);
  }
  if (!view.vectorPlan->segments.empty() && !llvm::all_of(view.vectorPlan->segments.back().strides, provablyOne)) {
    return LocalViewCompatibility::UnsupportedShape;
  }
  const auto destination = descriptors.back();
  const bool unequal = llvm::any_of(descriptors, [&](auto desc) { return !sameStrides(desc, destination); });
  for (auto [tensorIndex, value] : llvm::enumerate(values)) {
    auto desc = descriptors[tensorIndex];
    auto type = dyn_cast<ascendc::LocalTensorType>(value.getType());
    if (!type) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    auto span = staticSpan(desc);
    const bool staticGeometry =
        llvm::all_of(desc.getVectorizedAxis(),
                     [&](int64_t axis) {
                       auto position = axisPosition(desc, axis);
                       return succeeded(position) && succeeded(getLiteralStride(desc.getRepeats()[*position]));
                     }) &&
        llvm::all_of(desc.getVectorizedStrides(), [](ExprAttr stride) { return succeeded(getLiteralStride(stride)); });
    if (staticGeometry && !span) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    int64_t capacity = 1;
    bool staticCapacity = type.hasRank();
    if (staticCapacity) {
      for (int64_t size : type.getShape()) {
        if (ShapedType::isDynamic(size)) {
          staticCapacity = false;
          break;
        }
        if (size <= 0 || capacity > std::numeric_limits<int64_t>::max() / size) {
          return LocalViewCompatibility::UnsupportedShape;
        }
        capacity *= size;
      }
    }
    if (span && staticCapacity && *span > capacity) {
      return LocalViewCompatibility::UnsupportedShape;
    }
    if (unequal) {
      // Each API invocation starts on a data block. Merged inner axes do not
      // introduce new API starts and therefore need no separate alignment test.
      for (size_t i = 0; i + 1 < view.vectorPlan->segments.size(); ++i) {
        auto stride = view.vectorPlan->segments[i].strides[tensorIndex];
        const auto bits = type.getElementType().getIntOrFloatBitWidth();
        if (bits == 0 || kDataBlockBits % bits != 0 ||
            !afir::ProveExpressionMultipleOf(stride.getValue().getValue(), kDataBlockBits / bits)) {
          return LocalViewCompatibility::UnsupportedShape;
        }
      }
      const auto sourceRoot = localStorageRoot(value);
      const auto destinationRoot = localStorageRoot(values.back());
      const auto sourceBuffer = getBufferDescAttr(value);
      const auto destinationBuffer = getBufferDescAttr(values.back());
      // TQue and TBuf IDs are allocated independently; q0 and b0 are distinct resources.
      const bool sharedStorage =
          value == values.back() || (sourceRoot && sourceRoot == destinationRoot) ||
          (sourceBuffer && destinationBuffer && sourceBuffer.getBufferId() >= 0 &&
           sourceBuffer.getBufferId() == destinationBuffer.getBufferId() &&
           isCanonicalQueueResource(sourceBuffer) == isCanonicalQueueResource(destinationBuffer) &&
           sourceBuffer.getSpace() == destinationBuffer.getSpace());
      if (sharedStorage && !sameStrides(desc, destination)) {
        return LocalViewCompatibility::UnsupportedShape;
      }
    }
  }
  return LocalViewCompatibility::Compatible;
}

LocalViewCompatibility checkElementwiseLocalViews(Operation *op, bool *wholeTensor) {
  ElementwiseLocalView view;
  auto result = prepareElementwiseLocalView(op, {}, false, view);
  if (result == LocalViewCompatibility::Compatible && wholeTensor) {
    *wholeTensor = !view.vectorPlan;
  }
  return result;
}

}  // namespace mlir::ascend::kernel::elementwise
