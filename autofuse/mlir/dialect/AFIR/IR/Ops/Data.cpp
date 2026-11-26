/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Data.cpp - AFIR data op implementations ------------------*- C++ -*-===//

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::afir;

namespace {
constexpr unsigned kMinOperands = 2;
constexpr unsigned kMaxOperands = 3;

using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

LogicalResult verifyNullaryOp(Operation *op) {
  if (op->getNumOperands() != 0) {
    return op->emitOpError("expected 0 operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!resultType) {
    return op->emitOpError("expected tensor result");
  }
  return success();
}

// load/store verifier. Unlike unary *elementwise* math ops (verified in math.cpp),
// load/store reshape across the tiling boundary: a load maps a GM tensor at its
// logical rank onto a merged/tiled local view, and a store maps back. The
// scheduler (which runs before ScheduleToAfir) legitimately assigns operand and
// result DIFFERENT ranks/shapes here — e.g. GM <?x?x?xf16> load → tiled
// <?x?x?xf16> over merged+tiled axes. So we only require both sides be shaped
// tensors of the same element type; rank/shape may differ by design.
LogicalResult verifyUnaryOp(Operation *op) {
  auto types = getUnaryShapedTypes(op);
  if (failed(types)) {
    return failure();
  }
  ShapedType inputType = types->input;
  ShapedType resultType = types->result;
  if (!isa<TensorType>(inputType) || !isa<TensorType>(resultType)) {
    return op->emitOpError("expected tensor operand and result");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return op->emitOpError("operand and result must have the same element type");
  }
  return success();
}

LogicalResult verifyBufferForm(Operation *op, unsigned expectedResults) {
  // Exactly two: the input and the destination. `afir.load`'s base also carries
  // an optional tmp buffer, but no transfer API on either family takes a tmp buffer
  // (see the op lowering matrix in docs/guides/), so one here is a mistake
  // rather than a form this verifier should accommodate.
  if (op->getNumOperands() != 2) {
    return op->emitOpError("buffer form expects one input and one destination");
  }
  if (op->getNumResults() != expectedResults) {
    return op->emitOpError("buffer form has an invalid result count");
  }
  if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
      llvm::any_of(op->getResultTypes(), [](Type type) { return isa<TensorType>(type); })) {
    return op->emitOpError("buffer form expects non-tensor input, destination, and results");
  }
  if (expectedResults == 1 && op->getResult(0).getType() != op->getOperand(1).getType()) {
    return op->emitOpError("buffer result and destination must have the same type");
  }
  return success();
}

FailureOr<SmallVector<AffineMap>> collectIndexingMaps(Operation *op) {
  auto mapsAttr = op->getAttrOfType<ArrayAttr>(kIndexingMapsAttrName);
  if (!mapsAttr) {
    return failure();
  }

  SmallVector<AffineMap> maps;
  maps.reserve(mapsAttr.size());
  for (Attribute attr : mapsAttr) {
    auto mapAttr = dyn_cast<AffineMapAttr>(attr);
    if (!mapAttr) {
      return failure();
    }
    maps.push_back(mapAttr.getValue());
  }
  return maps;
}

std::optional<uint64_t> getStaticElementCount(MemRefType type) {
  if (!type.hasStaticShape()) {
    return std::nullopt;
  }
  uint64_t count = 1;
  for (int64_t dim : type.getShape()) {
    uint64_t unsignedDim = static_cast<uint64_t>(dim);
    if (unsignedDim != 0 && count > std::numeric_limits<uint64_t>::max() / unsignedDim) {
      return std::nullopt;
    }
    count *= unsignedDim;
  }
  return count;
}

FailureOr<std::string> getCanonicalElementCountExpression(Operation *op, StringRef descriptorName, unsigned rank) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(descriptorName);
  if (!descriptors || descriptors.empty()) {
    return op->emitOpError("InvalidTransferShape: missing canonical descriptors for dynamic transfer");
  }
  auto descriptor = dyn_cast<TensorDescAttr>(descriptors[0]);
  if (!descriptor || descriptor.getRepeats().size() != rank) {
    return op->emitOpError("InvalidTransferShape: canonical transfer shape expressions must match the view rank");
  }
  std::string expression;
  for (auto [index, repeat] : llvm::enumerate(descriptor.getRepeats())) {
    if (index != 0) {
      expression += " * ";
    }
    expression += repeat.getValue().getValue().str();
  }
  if (descriptor.getRepeats().empty()) {
    expression = "1";
  }
  return expression;
}

FailureOr<Expression> parseGeneratedExpression(StringRef raw) {
  // Text assembled here is not canonical: `ExpectEq(a, b)` canonicalizes to
  // `True` when both sides agree, which is the case this verifier wants to
  // accept.
  auto parsed = Expression::ParseRelaxed(raw);
  if (!parsed) {
    return failure();
  }
  return parsed.value();
}

bool isSchedulerTiledTransfer(Operation *op) {
  // ScheduleToAfir carries the post-tiling descriptor expressions on transfer
  // ops. Their local view may use merged/block/tile axes, so its element count
  // is not always algebraically provable from the GM view without the runtime
  // tiling-data positivity assumptions. The queue lowering consumes the
  // scheduler descriptors directly and performs the runtime flat-count copy.
  // Axis identity is not a precondition here: scheduler views may keep the same
  // axes or merge/re-cut them into a different axis space.
  auto sched = op->getAttrOfType<mlir::afir::ScheduleAttr>(mlir::afir::kSchedAttrName);
  if (!sched || sched.getLoopAxis() < 0) {
    return false;
  }
  auto operandDescs = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  auto destinationDescs = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  if (!operandDescs || !destinationDescs || operandDescs.size() != 1 || destinationDescs.size() != 1 ||
      op->getNumOperands() != 2) {
    return false;
  }
  auto sourceTensor = dyn_cast<TensorDescAttr>(operandDescs[0]);
  auto destinationTensor = dyn_cast<TensorDescAttr>(destinationDescs[0]);
  // 空间从值的定义点取，一个值一份（设计文档 5.1 步 8）。
  BufferDescAttr sourceBuffer = getBufferDescAttr(op->getOperand(0));
  BufferDescAttr destinationBuffer = getBufferDescAttr(op->getOperand(1));
  if (!sourceTensor || !destinationTensor || !sourceBuffer || !destinationBuffer) {
    return false;
  }
  Space sourceSpace = sourceBuffer.getSpace().getValue();
  Space destinationSpace = destinationBuffer.getSpace().getValue();
  return (sourceSpace == Space::GM && destinationSpace == Space::UB) ||
         (sourceSpace == Space::UB && destinationSpace == Space::GM);
}

FailureOr<ExprAttr> descriptorElementCount(Operation *op, TensorDescAttr tensor) {
  FailureOr<ExprAttr> count = makeExprAttr(op->getContext(), "1");
  if (failed(count)) {
    return failure();
  }
  for (ExprAttr repeat : tensor.getRepeats()) {
    count = multiplyExprs(op->getContext(), *count, repeat);
    if (failed(count)) {
      return failure();
    }
  }
  return count;
}

bool isSchedulerTiledTransferWithCompatibleView(Operation *op) {
  if (!isSchedulerTiledTransfer(op)) {
    return false;
  }
  auto operandDescs = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  auto destinationDescs = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  auto sourceTensor = dyn_cast<TensorDescAttr>(operandDescs[0]);
  auto destinationTensor = dyn_cast<TensorDescAttr>(destinationDescs[0]);
  if (!sourceTensor || !destinationTensor) {
    return false;
  }
  // Inductor 的外层 Data/Output 可能只保留 dtype/index：一侧 descriptor 没有轴，
  // 对应 memref 退成 scalar。这种退化形态由调度 descriptor 描述，跳过元素数比较。
  if (sourceTensor.getAxis().empty() || destinationTensor.getAxis().empty()) {
    return true;
  }

  FailureOr<ExprAttr> sourceCount = descriptorElementCount(op, sourceTensor);
  FailureOr<ExprAttr> destinationCount = descriptorElementCount(op, destinationTensor);
  StringRef sourceText;
  StringRef destinationText;
  if (succeeded(sourceCount) && succeeded(destinationCount)) {
    sourceText = sourceCount->getValue().getValue();
    destinationText = destinationCount->getValue().getValue();
    if (sourceText == destinationText || ProveExpressionsEqual(sourceText, destinationText)) {
      return true;
    }
  }

  // 轴空间不同是调度切分/合轴的正常结果：外层调度循环会重复或重组 tile，
  // 两个 descriptor 的元素总量不必相等。此时由调度 descriptor 与下降侧消费它。
  if (sourceTensor.getAxis() != destinationTensor.getAxis()) {
    return true;
  }

  // 元素数不相等时，只有同轴视图能由「一侧 repeat 1、stride 0」的广播解释。
  // 调度外层循环会重复这张小 tile，覆盖 destination 的完整元素数。
  if (sourceTensor.getRepeats().size() != sourceTensor.getAxis().size() ||
      destinationTensor.getRepeats().size() != destinationTensor.getAxis().size() ||
      sourceTensor.getStrides().size() != sourceTensor.getAxis().size() ||
      destinationTensor.getStrides().size() != destinationTensor.getAxis().size()) {
    return false;
  }
  auto expressionsAgree = [](ExprAttr lhs, ExprAttr rhs) {
    StringRef lhsText = lhs.getValue().getValue();
    StringRef rhsText = rhs.getValue().getValue();
    return lhsText == rhsText || ProveExpressionsEqual(lhsText, rhsText);
  };
  // 广播侧判据分读/写两侧。读(源)侧 stride==0 即合法广播——同一元素被重复读无害，
  // 分块后呈 repeat==(N/内块) 且 stride 0（如中间轴 expand 的 a1T：repeat 3/a1t、
  // stride 0，目的侧 stride 400*a1t），repeat==1 只覆盖整轴广播的特例；调度外层循环按
  // repeat 重复这张 tile、两侧 stride 已对上内块，元素数差异由此解释。
  // 写(目的)侧仍要求 repeat==1 && stride==0：repeat>1 且写侧 stride 0 表示 N 份 tile
  // 全部写回同一 GM 地址（正是「第二份 expand 拷贝写回第 0 份位置」类错数据），必须
  // 编译期拒绝而非静默放行。本函数只对带 afir.sched 的搬运生效，循环覆盖由调度构造保证。
  auto isReadBroadcastSide = [](ExprAttr stride) { return stride.getValue().getValue() == "0"; };
  auto isWriteBroadcastSide = [](ExprAttr repeat, ExprAttr stride) {
    return repeat.getValue().getValue() == "1" && stride.getValue().getValue() == "0";
  };
  const size_t axisCount = sourceTensor.getAxis().size();
  llvm::SmallVector<bool, 4> broadcastAxis(axisCount, false);
  for (size_t axis = 0; axis < axisCount; ++axis) {
    if (expressionsAgree(sourceTensor.getRepeats()[axis], destinationTensor.getRepeats()[axis]) &&
        expressionsAgree(sourceTensor.getStrides()[axis], destinationTensor.getStrides()[axis])) {
      continue;
    }
    // 写侧违规独立拒绝：repeat>1 且 stride 0 表示 N 份 tile 全部写回同一 GM 地址，
    // 即使读侧同轴也是广播（stride 0）也不放行——析取短路会把这类组合静默放过。
    if (destinationTensor.getStrides()[axis].getValue().getValue() == "0") {
      int64_t destinationRepeat = 0;
      if (destinationTensor.getRepeats()[axis].getValue().getValue().getAsInteger(10, destinationRepeat) ||
          destinationRepeat > 1) {
        return false;
      }
    }
    if (isReadBroadcastSide(sourceTensor.getStrides()[axis]) ||
        isWriteBroadcastSide(destinationTensor.getRepeats()[axis], destinationTensor.getStrides()[axis])) {
      broadcastAxis[axis] = true;
      continue;
    }
  }
  for (size_t axis = 0; axis < axisCount; ++axis) {
    if (broadcastAxis[axis] ||
        (expressionsAgree(sourceTensor.getRepeats()[axis], destinationTensor.getRepeats()[axis]) &&
         expressionsAgree(sourceTensor.getStrides()[axis], destinationTensor.getStrides()[axis]))) {
      continue;
    }
    // 剩下的差异轴是「内层广播压缩外层行距」：小 tile 一侧的外层 stride =
    // 全形一侧的 stride ÷ 内层广播轴的 repeat。判据与桥接层
    // DifferenceIsExplainedByBroadcast 一致——广播发生在更内层时外层紧凑行距
    // 跟着缩小（合并轴模板 B0Y2 的 a0：源 (5,400)、目的 (5,1200)，内层 a1 广播
    // repeat 3）。交叉比较：源 stride × 目的内层广播 repeat 积 = 目的 stride，
    // 或反之；内层没有广播轴则不可解释。
    // 与桥接层 DifferenceIsExplainedByBroadcast 同判：本轴两侧 repeat 必须相等——
    // 行距被内层广播压缩只在「同一逻辑行被分块」时成立；repeat 不等而 stride 恰成
    // 倍数是真实错配，不能靠交叉乘积的巧合放行、跳过元素数校验。
    if (!expressionsAgree(sourceTensor.getRepeats()[axis], destinationTensor.getRepeats()[axis]) ||
        std::none_of(broadcastAxis.begin() + static_cast<ptrdiff_t>(axis) + 1, broadcastAxis.end(),
                     [](bool value) { return value; })) {
      return false;
    }
    std::string sourceExpected = sourceTensor.getStrides()[axis].getValue().getValue().str();
    std::string destinationExpected = destinationTensor.getStrides()[axis].getValue().getValue().str();
    for (size_t inner = axis + 1; inner < axisCount; ++inner) {
      if (!broadcastAxis[inner]) {
        continue;
      }
      auto sourceProduct =
          MultiplyExpressions(sourceExpected, destinationTensor.getRepeats()[inner].getValue().getValue());
      auto destinationProduct =
          MultiplyExpressions(destinationExpected, sourceTensor.getRepeats()[inner].getValue().getValue());
      if (!sourceProduct || !destinationProduct) {
        return false;
      }
      sourceExpected = sourceProduct.value();
      destinationExpected = destinationProduct.value();
    }
    if (!ProveExpressionsEqual(destinationTensor.getStrides()[axis].getValue().getValue(), sourceExpected) &&
        !ProveExpressionsEqual(sourceTensor.getStrides()[axis].getValue().getValue(), destinationExpected)) {
      return false;
    }
  }
  return true;
}

LogicalResult verifyDynamicTransferElementCount(Operation *op, MemRefType sourceType, MemRefType destinationType) {
  FailureOr<std::string> sourceExpression =
      getCanonicalElementCountExpression(op, kOperandDescsAttrName, sourceType.getRank());
  FailureOr<std::string> destinationExpression =
      getCanonicalElementCountExpression(op, kDestinationDescsAttrName, destinationType.getRank());
  if (failed(sourceExpression) || failed(destinationExpression)) {
    return failure();
  }
  std::string expectedText = "ExpectEq(" + *sourceExpression + ", " + *destinationExpression + ")";
  auto expected = parseGeneratedExpression(expectedText);
  if (failed(expected)) {
    return op->emitOpError("InvalidTransferShape: malformed canonical element-count expression");
  }
  auto expectedSimplified = expected.value().Simplify();
  if (!expectedSimplified) {
    return op->emitOpError("InvalidTransferShape: cannot simplify canonical element-count expression: ")
           << expectedSimplified.error().message;
  }
  auto proof = afir::detail::ExpressionAccess::Prove(expectedSimplified.value());
  if (!proof || proof.value() != afir::detail::TruthValue::True) {
    return op->emitOpError(
        "UnsupportedDynamicShape: source and destination element counts are not statically proven equal");
  }
  return success();
}

LogicalResult verifyTransferElementCount(Operation *op, MemRefType sourceType, MemRefType destinationType) {
  std::optional<uint64_t> sourceElementCount = getStaticElementCount(sourceType);
  std::optional<uint64_t> destinationElementCount = getStaticElementCount(destinationType);
  if (sourceType.hasStaticShape() && destinationType.hasStaticShape() &&
      (!sourceElementCount || !destinationElementCount)) {
    return op->emitOpError("transfer static element count exceeds the supported range");
  }
  // Inductor 的 `AscBackend` 外层图可能只在 Load descriptor 上携带 GM→UB 的 tile 映射，
  // scheduler-tiled transfer 的每趟可以只搬一个 tile，再由外层调度循环覆盖整图。
  // 轴空间不同（合轴/重切）时交给调度 descriptor；轴空间相同时，两侧 descriptor 的
  // repeats 乘积相等，或差异能由「一侧 repeat 1、stride 0」的广播轴解释，才跳过元素数
  // 比较；否则继续走静态/动态校验，避免真实 mismatch 被放行。
  if (isSchedulerTiledTransferWithCompatibleView(op)) {
    return success();
  }
  if (!sourceType.hasStaticShape() || !destinationType.hasStaticShape()) {
    return verifyDynamicTransferElementCount(op, sourceType, destinationType);
  }
  if (sourceElementCount && destinationElementCount && *sourceElementCount != *destinationElementCount) {
    return op->emitOpError("transfer source and destination must have the same static element count");
  }
  return success();
}

// `equalElementCount` 说的是「两端搬的是同样多个元素」。`afir.load` / `afir.store` 是
// 真，`afir.nddma` 是假——它一边搬一边广播，落点的元素数本来就比源多，那不是形态错误。
//
// 关掉它之后两端的形状在 ODS 这一层就不再被约束，这是有意的：三个搬运 op 的两端都活在
// 不同的下标空间，形状对不对只有 descriptor 答得出来，而 verifier 失败是硬失败、不是回落
// （`afir.nddma` 的 ODS 上记着实测数字）。真正的形状判据在下降侧按 descriptor 查。
LogicalResult verifyTransferIndexingMaps(Operation *op, bool equalElementCount) {
  // The canonical buffer form is memref-based and always carries explicit
  // source/destination maps. Existing AscendC lowering tests use typed
  // GlobalTensor/LocalTensor values as an intermediate form; preserve that
  // legacy boundary until those producers are migrated to memref.
  auto sourceType = dyn_cast<MemRefType>(op->getOperand(0).getType());
  auto destinationType = dyn_cast<MemRefType>(op->getOperand(1).getType());
  if (!sourceType || !destinationType) {
    return success();
  }
  auto maps = collectIndexingMaps(op);
  if (failed(maps)) {
    return op->emitOpError("InvalidIndexingMaps: buffer form requires an ArrayAttr of AffineMapAttr");
  }
  if (maps->size() != 2) {
    return op->emitOpError("InvalidIndexingMaps: transfer requires exactly two indexing maps");
  }
  if ((*maps)[0].getNumDims() != (*maps)[1].getNumDims()) {
    return op->emitOpError("InvalidIndexingMaps: source and destination maps must share the same iteration rank");
  }
  if ((*maps)[0].getNumSymbols() != (*maps)[1].getNumSymbols()) {
    return op->emitOpError("InvalidIndexingMaps: source and destination maps must share the same symbol count");
  }
  if ((*maps)[0].getNumResults() != sourceType.getRank()) {
    return op->emitOpError("InvalidIndexingMaps: source map result rank must match source view rank");
  }
  if ((*maps)[1].getNumResults() != destinationType.getRank()) {
    return op->emitOpError("InvalidIndexingMaps: destination map result rank must match destination view rank");
  }
  if (sourceType.getElementType() != destinationType.getElementType()) {
    return op->emitOpError("transfer source and destination must have the same element type");
  }
  if (!equalElementCount) {
    return success();
  }
  return verifyTransferElementCount(op, sourceType, destinationType);
}

void getBufferFormEffects(Operation *op, SmallVectorImpl<MemoryEffect> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &op->getOpOperand(0), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &op->getOpOperand(1), SideEffects::DefaultResource::get());
}

}  // namespace

LogicalResult ScalarOp::verify() {
  // 操作数与结果的个数不用在这里查：ODS 生成的 `ZeroOperands` / `OneResult` 已经查了
  // （`ConstantLike` 对这两个特征有 static_assert）。这里只钉一件事——
  // 字面量的类型就是结果的元素类型：buffer 形态是结果自己，tensor 形态是结果的元素类型。
  // 对不上时下降出来的常量会带着另一种 dtype 被使用点读走。
  Type resultType = getResult().getType();
  Type elementType = isa<ShapedType>(resultType) ? cast<ShapedType>(resultType).getElementType() : resultType;
  if (cast<TypedAttr>(getValue()).getType() != elementType) {
    return emitOpError("value attribute type must match the result element type");
  }
  return success();
}

// 字面量折叠成它自己的属性：`ConstantLike` 要求这一条，下游（归约缓存依赖分析）要的是按特征
// 认出字面量。AFIR 方言没有实现 `materializeConstant`，所以折叠器只会拿它合并重复的字面量
// （`--cse` 下两个同值同类型的 `afir.scalar` 并成一个），不会把别的 op 的结果换成新的常量 op。
OpFoldResult ScalarOp::fold(FoldAdaptor) {
  return getValue();
}
LogicalResult IndexExprOp::verify() {
  return verifyNullaryOp(getOperation());
}
LogicalResult LoadOp::verify() {
  if (isBufferizedForm()) {
    if (failed(verifyBufferForm(getOperation(), 1))) {
      return failure();
    }
    if (failed(verifyIndexingMaps())) {
      return failure();
    }
    return verifyCanonicalBufferDescriptors(getOperation());
  }
  return verifyUnaryOp(getOperation());
}
LogicalResult StoreOp::verify() {
  if (isBufferizedForm()) {
    if (failed(verifyBufferForm(getOperation(), 0))) {
      return failure();
    }
    if (failed(verifyIndexingMaps())) {
      return failure();
    }
    return verifyCanonicalBufferDescriptors(getOperation());
  }
  return verifyUnaryOp(getOperation());
}

FailureOr<Role> LoadOp::getRole() {
  return getCanonicalTargetRole(getOperation());
}

FailureOr<Role> StoreOp::getRole() {
  return getCanonicalTargetRole(getOperation());
}

FailureOr<unsigned> LoadOp::getTiedResultIndex(unsigned destinationIndex) {
  if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0;
}

FailureOr<SmallVector<AffineMap>> LoadOp::getIndexingMapList() {
  return collectIndexingMaps(getOperation());
}

LogicalResult LoadOp::verifyIndexingMaps() {
  return verifyTransferIndexingMaps(getOperation(), /*equalElementCount=*/true);
}

// `afir.store` has no tmp buffer operand at all: its base omits it because no
// transfer API on either family takes one.
FailureOr<unsigned> StoreOp::getTiedResultIndex(unsigned destinationIndex) {
  if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0;
}

FailureOr<SmallVector<AffineMap>> StoreOp::getIndexingMapList() {
  return collectIndexingMaps(getOperation());
}

LogicalResult StoreOp::verifyIndexingMaps() {
  return verifyTransferIndexingMaps(getOperation(), /*equalElementCount=*/true);
}

void LoadOp::getEffects(SmallVectorImpl<MemoryEffect> &effects) {
  if (isBufferizedForm()) {
    getBufferFormEffects(getOperation(), effects);
  }
}

void StoreOp::getEffects(SmallVectorImpl<MemoryEffect> &effects) {
  if (isBufferizedForm()) {
    getBufferFormEffects(getOperation(), effects);
  }
}

Speculation::Speculatability LoadOp::getSpeculatability() {
  return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}

Speculation::Speculatability StoreOp::getSpeculatability() {
  return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// afir.nddma
//===----------------------------------------------------------------------===//
//
// 与 `afir.load` 共用搬运那一整套（buffer 形态、indexing maps、descriptor 契约、
// destination-style、pipe、副作用），只有一处不同：indexing maps 不要求两端元素数相等
// ——一边搬一边广播改的正是元素数。两端的形状关系不在这里查，理由见 ODS 上那段。

LogicalResult NddmaOp::verify() {
  if (isBufferizedForm()) {
    if (failed(verifyBufferForm(getOperation(), 1))) {
      return failure();
    }
    if (failed(verifyIndexingMaps())) {
      return failure();
    }
    return verifyCanonicalBufferDescriptors(getOperation());
  }
  return verifyUnaryOp(getOperation());
}

FailureOr<Role> NddmaOp::getRole() {
  return getCanonicalTargetRole(getOperation());
}

FailureOr<unsigned> NddmaOp::getTiedResultIndex(unsigned destinationIndex) {
  if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0;
}

FailureOr<SmallVector<AffineMap>> NddmaOp::getIndexingMapList() {
  return collectIndexingMaps(getOperation());
}

LogicalResult NddmaOp::verifyIndexingMaps() {
  return verifyTransferIndexingMaps(getOperation(), /*equalElementCount=*/false);
}

void NddmaOp::getEffects(SmallVectorImpl<MemoryEffect> &effects) {
  if (isBufferizedForm()) {
    getBufferFormEffects(getOperation(), effects);
  }
}

Speculation::Speculatability NddmaOp::getSpeculatability() {
  return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// afir.transpose
//===----------------------------------------------------------------------===//
//
// 与 load/store/nddma 共用搬运那一整套（indexing maps、descriptor 契约、destination-style、
// pipe、副作用），差别在两处：两端 rank 必须相同（重排不增删轴），以及 tmp buffer 是
// 正当形态——两侧转置包装都吃 tmp（mem `ConfusionTranspose` / reg `TransposeExtend`），
// 不能复用 load/store 的「恰好两个操作数」判据。

// 转置重排轴序、不增删轴，两端 rank 必须相同（legacy `IsNodeValid` 同一判据，
// `v2_ascir_codegen_impl.h:3750-3754`）。无秩的一端不比：target 方言的中间形态
// （`!ascendc.local_tensor`）可能无秩，它们的形状关系由 descriptor 与 maps 表达。
LogicalResult verifyTransposeSameRank(Operation *op) {
  auto inputType = dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto outputType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!inputType || !outputType || !inputType.hasRank() || !outputType.hasRank()) {
    return success();
  }
  if (inputType.getRank() != outputType.getRank()) {
    return op->emitOpError("transpose must not add or remove axes: input and output ranks differ");
  }
  return success();
}

LogicalResult TransposeOp::verify() {
  if (isBufferizedForm()) {
    // 操作数结构：input + destination + 可选 tmp（ODS 三段划分，段长由
    // AttrSizedOperandSegments 保证）。tmp 在这里不像 load/store 那样报错——见上。
    // 以下 result/destination/type 判据与 verifyBufferForm 保持一致。
    if (getOperation()->getNumOperands() < kMinOperands || getOperation()->getNumOperands() > kMaxOperands) {
      return getOperation()->emitOpError("buffer form expects an input, a destination, and an optional tmp buffer");
    }
    if (getOperation()->getNumResults() != 1) {
      return getOperation()->emitOpError("buffer form has an invalid result count");
    }
    if (llvm::any_of(getOperation()->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
        llvm::any_of(getOperation()->getResultTypes(), [](Type type) { return isa<TensorType>(type); })) {
      return getOperation()->emitOpError("buffer form expects non-tensor input, destination, and results");
    }
    if (getOperation()->getResult(0).getType() != getOperation()->getOperand(1).getType()) {
      return getOperation()->emitOpError("buffer result and destination must have the same type");
    }
    if (failed(verifyTransposeSameRank(getOperation()))) {
      return failure();
    }
    if (failed(verifyIndexingMaps())) {
      return failure();
    }
    return verifyCanonicalBufferDescriptors(getOperation());
  }
  if (failed(verifyTransposeSameRank(getOperation()))) {
    return failure();
  }
  return verifyUnaryOp(getOperation());
}

FailureOr<Role> TransposeOp::getRole() {
  return getCanonicalTargetRole(getOperation());
}

FailureOr<unsigned> TransposeOp::getTiedResultIndex(unsigned destinationIndex) {
  if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0;
}

FailureOr<SmallVector<AffineMap>> TransposeOp::getIndexingMapList() {
  return collectIndexingMaps(getOperation());
}

// 转置保元素数（两端是同一批元素换摆法），与 load/store 同档；nddma 才是那个改元素数的。
LogicalResult TransposeOp::verifyIndexingMaps() {
  return verifyTransferIndexingMaps(getOperation(), /*equalElementCount=*/true);
}

void TransposeOp::getEffects(SmallVectorImpl<MemoryEffect> &effects) {
  if (isBufferizedForm()) {
    getBufferFormEffects(getOperation(), effects);
  }
}

Speculation::Speculatability TransposeOp::getSpeculatability() {
  return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}
