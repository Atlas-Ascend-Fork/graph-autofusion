/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Elementwise.cpp - AFIR elementwise op implementations ----*- C++ -*-===//
//
// Verifiers + return-type inference for AFIR math ops. Ported from
// Ascend-MLIR-gsr.
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"

#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

// elementwise 结果形状只在这里推一次：操作数形状的广播。arity 无关的 verifier 与
// 多输入的 `inferReturnType*` 都走它，于是「被校验的形状」与「被推出的形状」不会
// 分叉。一元 Op 的结果形状就是输入形状，`inferReturnTypes` 直接取，不绕这一圈。
// 某个操作数无秩时 `hasRankedShape` 置假，此时不写 `newShape`。
// 标量操作数不参与形状：它没有 tile，广播到哪个形状都是它自己。跳过和退回无秩
// 在这里是两件相反的事——一个无秩的 shaped 操作数说明「结果的形状不知道」，一个标量
// 操作数什么都没说。混为一谈的后果是 `afir.add %tile, %scalar` 的 buffer 形态
// verifier 直接报「形状对不上」，而它其实完全合法。
bool contributesShape(Type type) {
  return llvm::isa<ShapedType>(type);
}

LogicalResult inferElementwiseShape(ValueRange operands, SmallVectorImpl<int64_t> &newShape, bool &hasRankedShape) {
  SmallVector<ShapedType> shapedOperands;
  for (Value operand : operands) {
    auto opType = llvm::dyn_cast<ShapedType>(operand.getType());
    if (!opType) {
      continue;  // 标量：不参与广播。
    }
    if (!opType.hasRank()) {
      hasRankedShape = false;
      return success();
    }
    shapedOperands.push_back(opType);
  }
  if (shapedOperands.empty()) {
    // 全是标量的 elementwise op 不存在（至少有一个 tile 才有得算），但真碰上了也不该
    // 断言出一个形状来。
    hasRankedShape = false;
    return success();
  }

  int64_t newShapeRank = 0;
  for (ShapedType opType : shapedOperands) {
    newShapeRank = std::max(newShapeRank, opType.getRank());
  }

  hasRankedShape = true;
  newShape.assign(newShapeRank, 1);
  SmallVector<int64_t> operOffset(shapedOperands.size());
  for (size_t i = 0; i < shapedOperands.size(); ++i) {
    operOffset[i] = newShapeRank - shapedOperands[i].getRank();
  }
  for (int64_t i = newShapeRank; i != 0; --i) {
    const int64_t index = i - 1;
    for (size_t j = 0; j < shapedOperands.size(); ++j) {
      if (index < operOffset[j]) {
        continue;
      }
      auto shape = shapedOperands[j].getShape();
      if (shape[index - operOffset[j]] == 1) {
        continue;
      }
      if (newShape[index] == ShapedType::kDynamic || newShape[index] == 1) {
        newShape[index] = shape[index - operOffset[j]];
      }
      if (shape[index - operOffset[j]] == ShapedType::kDynamic) {
        continue;
      }
      if (newShape[index] != shape[index - operOffset[j]]) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult verifyElementwiseOp(Operation *op) {
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!llvm::all_of(op->getOperandTypes(), [](Type type) { return mlir::isa<ShapedType>(type); }) || !resultType) {
    return op->emitOpError("expected tensor operands and result");
  }

  SmallVector<int64_t> inferredShape;
  bool hasRankedShape = false;
  if (failed(inferElementwiseShape(op->getOperands(), inferredShape, hasRankedShape))) {
    return op->emitOpError("operands could not be broadcast to a compatible shape");
  }
  if (hasRankedShape &&
      (!resultType.hasRank() || failed(verifyCompatibleShape(inferredShape, resultType.getShape())))) {
    return op->emitOpError("result shape must match the inferred broadcast shape");
  }
  return success();
}

// The buffer-form contract shared by every destination-style elementwise
// operation, whatever its arity: no tensors anywhere, result type equal to the
// destination type, one element type across inputs/destination/result, and a
// destination shaped like the broadcast of the inputs. The tmp buffer operand is
// excluded from all of it -- it is bytes, not a tile of the computation.
LogicalResult verifyBufferizedElementwise(Operation *op, ValueRange inputs, Value destination) {
  if (!destination) {
    return op->emitOpError("buffer form expects a destination");
  }
  // 至少一个输入要有形状。ODS 的类型约束是逐个操作数的，说不出「至多一个标量」——
  // `scalarOperand = "any"` 的 op 两个操作数各自都可以是标量，两个一起是标量就得在这里
  // 挡。挡住的不是一个假想形态：`afir.add %a, %b` 两个都是 `f32` 时 ODS 全过，而
  // `inferElementwiseShape` 会答「没有形状」，目的地的形状无从校验，生成的调用把
  // 两个值当成两块 tile。
  if (!inputs.empty() && llvm::none_of(inputs, [](Value input) { return contributesShape(input.getType()); })) {
    return op->emitOpError(
        "buffer form expects at least one shaped input; all-scalar inputs have no tile to compute over");
  }
  if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
      isa<TensorType>(op->getResult(0).getType())) {
    return op->emitOpError("buffer form expects non-tensor inputs, destination, and result");
  }
  if (op->getResult(0).getType() != destination.getType()) {
    return op->emitOpError("buffer result and destination must have the same type");
  }
  auto destinationType = dyn_cast<MemRefType>(destination.getType());
  auto resultType = dyn_cast<MemRefType>(op->getResult(0).getType());
  // 标量输入豁免：它不是一块 buffer。其余输入仍然必须是 memref，不然形状检查没有
  // 可比的东西，只能整段跳过。
  if (!destinationType || !resultType || llvm::any_of(inputs, [](Value input) {
        return contributesShape(input.getType()) && !isa<MemRefType>(input.getType());
      })) {
    return success();
  }
  // 元素类型的一致性不在这里查：输入与结果由 ODS 的 `AllElementTypesMatch` 管（见
  // Elementwise.td 的 `sameElementType`），destination 与结果由上面那条同型检查管。
  // 在这里再查一遍既到不了，也会挡住结果 dtype 本就不同的 Op（比较族、cast）。
  SmallVector<int64_t> inferredShape;
  bool hasRankedShape = false;
  if (failed(inferElementwiseShape(inputs, inferredShape, hasRankedShape))) {
    return op->emitOpError("operands could not be broadcast to a compatible shape");
  }
  const char *broadcastReason = nullptr;
  const bool shapedInputsBroadcast = llvm::all_of(inputs, [&](Value input) {
    return !isa<ShapedType>(input.getType()) ||
           elementwise::sourceBroadcastsToDestination(op, input, destination, &broadcastReason);
  });
  if (!hasRankedShape ||
      (!shapedInputsBroadcast && failed(verifyCompatibleShape(inferredShape, destinationType.getShape())))) {
    if (broadcastReason != nullptr) {
      return op->emitOpError(broadcastReason);
    }
    return op->emitOpError("result shape must match the inferred broadcast shape");
  }
  return success();
}

void getBufferizedElementwiseEffects(Operation *op, OperandRange inputs, MutableOperandRange destination,
                                     MutableOperandRange tmpBuffer, SmallVectorImpl<MemoryEffect> &effects) {
  for (unsigned index = 0; index < inputs.size(); ++index) {
    // 标量操作数不是一次内存读。给它挂上 Read 会让 `arith.constant` 参与访存排序，
    // 说的是一件不存在的事。
    if (!contributesShape(inputs[index].getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), &op->getOpOperand(index), SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &destination[0], SideEffects::DefaultResource::get());
  // A tmp buffer, but a written one: without the effect nothing stops two operations
  // sharing a tmp buffer from being reordered against each other.
  if (!tmpBuffer.empty()) {
    effects.emplace_back(MemoryEffects::Write::get(), &tmpBuffer[0], SideEffects::DefaultResource::get());
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// 供 ODS 生成的方法体调用的那一组（AFIRElementwiseSupport.h）
//===----------------------------------------------------------------------===//
namespace mlir::afir::elementwise {

namespace {

bool isLiteral(ExprAttr expression, int64_t expected) {
  int64_t literal = 0;
  return expression && !expression.getValue().getValue().getAsInteger(10, literal) && literal == expected;
}

bool axisIsVectorized(TensorDescAttr descriptor, size_t axisIndex) {
  if (!descriptor || descriptor.getAxis().size() <= axisIndex) {
    return true;
  }
  return llvm::is_contained(descriptor.getVectorizedAxis(), descriptor.getAxis()[axisIndex]);
}

}  // namespace

static bool verifyBroadcastOuterStrides(Operation *op, ArrayRef<int64_t> sourceShape,
                                        ArrayRef<int64_t> destinationShape, TensorDescAttr sourceDescriptor,
                                        TensorDescAttr destinationDescriptor, ArrayRef<size_t> broadcastAxes,
                                        const char **reason) {
  auto reject = [&](const char *message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  auto hasLaterBroadcastAxis = [&broadcastAxes](size_t axis) {
    return llvm::any_of(broadcastAxes, [&](size_t broadcastAxis) { return broadcastAxis > axis; });
  };
  for (size_t axis = 0; axis < destinationShape.size(); ++axis) {
    if (sourceShape[axis] != destinationShape[axis] ||
        sourceDescriptor.getStrides()[axis] == destinationDescriptor.getStrides()[axis]) {
      continue;
    }
    if (!hasLaterBroadcastAxis(axis)) {
      return reject("outer stride difference requires a later broadcast axis");
    }

    FailureOr<ExprAttr> expectedStride = makeExprAttr(op->getContext(), "1");
    if (failed(expectedStride)) {
      return false;
    }
    for (size_t broadcastAxis : broadcastAxes) {
      if (broadcastAxis <= axis) {
        continue;
      }
      expectedStride =
          multiplyExprs(op->getContext(), *expectedStride, destinationDescriptor.getRepeats()[broadcastAxis]);
      if (failed(expectedStride)) {
        return false;
      }
    }
    expectedStride = multiplyExprs(op->getContext(), *expectedStride, sourceDescriptor.getStrides()[axis]);
    if (failed(expectedStride)) {
      return false;
    }
    StringRef expectedText = expectedStride->getValue().getValue();
    StringRef destinationText = destinationDescriptor.getStrides()[axis].getValue().getValue();
    if (expectedText != destinationText && !ProveExpressionsEqual(expectedText, destinationText)) {
      return reject("destination outer stride must equal source stride multiplied by later broadcast extents");
    }
  }
  return true;
}

static bool verifyBroadcastGeometry(Operation *op, ShapedType sourceType, ShapedType destinationType,
                                    TensorDescAttr sourceDescriptor, TensorDescAttr destinationDescriptor,
                                    const char **reason) {
  auto reject = [&](const char *message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  const ArrayRef<int64_t> sourceShape = sourceType.getShape();
  const ArrayRef<int64_t> destinationShape = destinationType.getShape();
  llvm::SmallVector<size_t, 4> broadcastAxes;
  for (size_t axis = 0; axis < destinationShape.size(); ++axis) {
    if (sourceShape[axis] == destinationShape[axis]) {
      continue;
    }
    if (sourceShape[axis] != 1) {
      return reject("broadcast source dimension must be size one");
    }
    if (!isLiteral(sourceDescriptor.getStrides()[axis], 0)) {
      return reject("broadcast source stride must be zero");
    }
    if (axisIsVectorized(sourceDescriptor, axis) || axisIsVectorized(destinationDescriptor, axis)) {
      return reject("broadcast axis must not be vectorized");
    }
    broadcastAxes.push_back(axis);
  }
  if (broadcastAxes.empty() && sourceDescriptor.getStrides() == destinationDescriptor.getStrides()) {
    return true;
  }

  return verifyBroadcastOuterStrides(op, sourceShape, destinationShape, sourceDescriptor, destinationDescriptor,
                                     broadcastAxes, reason);
}

static bool haveSameRank(ShapedType sourceType, ShapedType destinationType) {
  return sourceType && destinationType && sourceType.hasRank() && destinationType.hasRank() &&
         sourceType.getRank() == destinationType.getRank();
}

static bool verifyBroadcastDescriptors(Operation *op, unsigned sourceIndex, ShapedType sourceType,
                                       ShapedType destinationType, const char **reason) {
  auto operandDescriptors = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  auto destinationDescriptors = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  if (!operandDescriptors || operandDescriptors.size() <= sourceIndex || !destinationDescriptors ||
      destinationDescriptors.empty()) {
    return false;
  }
  TensorDescAttr sourceDescriptor = dyn_cast<TensorDescAttr>(operandDescriptors[sourceIndex]);
  TensorDescAttr destinationDescriptor = dyn_cast<TensorDescAttr>(destinationDescriptors[0]);
  if (!sourceDescriptor || !destinationDescriptor || sourceDescriptor.getAxis() != destinationDescriptor.getAxis() ||
      sourceDescriptor.getAxis().size() != static_cast<size_t>(sourceType.getRank()) ||
      destinationDescriptor.getAxis().size() != static_cast<size_t>(destinationType.getRank()) ||
      sourceDescriptor.getStrides().size() != static_cast<size_t>(sourceType.getRank())) {
    return false;
  }

  return verifyBroadcastGeometry(op, sourceType, destinationType, sourceDescriptor, destinationDescriptor, reason);
}

bool sourceBroadcastsToDestination(Operation *op, Value source, Value destination, const char **reason) {
  if (!op || !source || !destination) {
    return false;
  }
  FailureOr<OperandPartition> partition = getOperandPartition(op);
  if (failed(partition) || partition->destinationCount != 1) {
    return false;
  }

  std::optional<unsigned> sourceIndex;
  OperandRange inputs = partition->inputs(op);
  for (unsigned index = 0; index < partition->inputCount; ++index) {
    if (inputs[index] == source) {
      sourceIndex = index;
      break;
    }
  }
  OperandRange destinations = partition->destinations(op);
  if (!sourceIndex || destinations.empty() || destinations[0] != destination) {
    return false;
  }

  auto sourceType = dyn_cast<ShapedType>(source.getType());
  auto destinationType = dyn_cast<ShapedType>(destination.getType());
  if (!haveSameRank(sourceType, destinationType)) {
    return false;
  }

  return verifyBroadcastDescriptors(op, *sourceIndex, sourceType, destinationType, reason);
}

}  // namespace mlir::afir::elementwise

LogicalResult mlir::afir::elementwise::verifyLogicalForm(Operation *op) {
  if (!llvm::all_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
      !isa<TensorType>(op->getResult(0).getType())) {
    return op->emitOpError("expected tensor operands and result");
  }
  return verifyElementwiseOp(op);
}

LogicalResult mlir::afir::elementwise::verifyBufferForm(Operation *op, ValueRange inputs, Value destination) {
  if (failed(verifyBufferizedElementwise(op, inputs, destination))) {
    return failure();
  }
  return verifyCanonicalBufferDescriptors(op);
}

void mlir::afir::elementwise::getBufferFormEffects(Operation *op, MutableOperandRange destination,
                                                   MutableOperandRange tmpBuffer,
                                                   SmallVectorImpl<MemoryEffect> &effects) {
  FailureOr<OperandPartition> partition = getOperandPartition(op);
  OperandRange inputs = succeeded(partition) ? partition->inputs(op) : op->getOperands();
  getBufferizedElementwiseEffects(op, inputs, destination, tmpBuffer, effects);
}

LogicalResult mlir::afir::elementwise::inferLogicalReturnType(ValueRange inputs, Type resultElementType,
                                                              SmallVectorImpl<Type> &inferredReturnTypes) {
  SmallVector<int64_t> newShape;
  bool hasRankedShape = false;
  if (failed(inferElementwiseShape(inputs, newShape, hasRankedShape))) {
    return failure();
  }
  // 不能假定第一个输入有形状：`scalarOperand = "any"` 的 op 两边都可以是标量
  // （`afir.add %scalar, %tile`），取 `inputs.front()` 会在那个方向上直接 `failure()`。
  // 结果的元素类型与秩由有形状的那一个给；「一个都没有」由
  // `verifyBufferizedElementwise` 挡，这里只需要退出。
  ShapedType inputType;
  for (Value input : inputs) {
    if (auto shaped = dyn_cast<ShapedType>(input.getType())) {
      inputType = shaped;
      break;
    }
  }
  if (!inputType) {
    return failure();
  }
  const Type elementType = resultElementType ? resultElementType : inputType.getElementType();
  if (!hasRankedShape) {
    // 有操作数无秩，广播结果的秩就是未知的——第一个输入哪怕自己有秩也说明不了结果的
    // 形状。断言成它的类型会是一个没人校验得出来的错误结论（无秩时 verifier 跳过形状
    // 检查），所以这里退回无秩形态。
    inferredReturnTypes.push_back(inputType.cloneWith(std::nullopt, elementType));
    return success();
  }
  inferredReturnTypes.push_back(inputType.cloneWith(newShape, elementType));
  return success();
}

Type mlir::afir::elementwise::inferResultElementType(MLIRContext *context, Type input, ResultElementRule rule) {
  const Type element = getElementTypeOrSelf(input);
  switch (rule) {
    case ResultElementRule::Same:
      return element;
    case ResultElementRule::Uint8:
      return IntegerType::get(context, 8, IntegerType::Unsigned);
    case ResultElementRule::Int32:
      // 无符号性，不是 `IntegerType::Signed`——AFIR 的 i32 全程是 signless。
      return IntegerType::get(context, 32);
    case ResultElementRule::SameOrF32ForI32:
      return element.isSignlessInteger(32) ? Type(Float32Type::get(context)) : element;
    case ResultElementRule::CastPair:
      // 推不出来：合法组合是一张逐行对表，一个输入对多个合法输出。
      return {};
  }
  return {};
}

bool mlir::afir::elementwise::matchesResultElementRule(Type input, Type result, ResultElementRule rule) {
  const Type expected = inferResultElementType(getElementTypeOrSelf(input).getContext(), input, rule);
  return expected && expected == getElementTypeOrSelf(result);
}

LogicalResult mlir::afir::elementwise::verifyLogicalNotPair(Operation *op, Type input, Type result) {
  const Type inputElement = getElementTypeOrSelf(input);
  const Type resultElement = getElementTypeOrSelf(result);
  // 并集只有两行：mem 侧的 `T -> T` 与 reg 侧的 `T -> ui8`。`ui8 -> ui8` 两侧都有，
  // 落在第一条上。
  if (inputElement == resultElement || resultElement.isUnsignedInteger(8)) {
    return success();
  }
  return op->emitOpError("unsupported logical_not result element type ")
         << resultElement << " for input " << inputElement;
}

// 二元 elementwise 已使用 AFIR_BufferizableBinaryElementwiseOp，返回类型由 ODS
// InferTypeOpInterface 生成，逻辑结果的 ui8 类型由 logicalResultElementType 提供。
// 共享 verifyLogicalForm / verifyBufferForm 已覆盖原手写校验。
// 更换基类时需同步移除对应的手写类型推导和 verify 定义，保持生成声明与实现一致。
