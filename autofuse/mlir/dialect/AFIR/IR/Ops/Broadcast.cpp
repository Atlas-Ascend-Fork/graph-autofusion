/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Broadcast.cpp - AFIR broadcast op implementation ---------*- C++ -*-===//

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

using MemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;

LogicalResult verifyBroadcastShapeImpl(Operation *op) {
  if (op->getNumOperands() < 1 || op->getNumResults() != 1) {
    return op->emitOpError("expected one shaped input and one shaped result");
  }
  auto inputType = dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!inputType || !resultType) {
    return op->emitOpError("expected one shaped input and one shaped result");
  }

  if (inputType.getElementType() != resultType.getElementType()) {
    return op->emitOpError("broadcast input and result must have the same element type");
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    const size_t inputRank = inputShape.size();
    const size_t resultRank = resultShape.size();
    if (resultRank < inputRank) {
      return op->emitOpError("result rank must be >= input rank for broadcast");
    }

    const size_t offset = resultRank - inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
      const int64_t inputDim = inputShape[i];
      const int64_t resultDim = resultShape[i + offset];
      if (inputDim != ShapedType::kDynamic && resultDim != ShapedType::kDynamic && inputDim != 1 &&
          inputDim != resultDim) {
        return op->emitOpError("input shape must be broadcastable to result shape");
      }
    }
  }
  return success();
}

LogicalResult verifyBufferizedBroadcast(BroadcastOp op) {
  if (op->getNumResults() != 1) {
    return op->emitOpError("buffer form expects one result");
  }
  if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<TensorType>(type); }) ||
      isa<TensorType>(op->getResult(0).getType())) {
    return op->emitOpError("buffer form expects non-tensor input, destination, and result");
  }
  if (op->getResult(0).getType() != op.getDestination().getType()) {
    return op->emitOpError("buffer result and destination must have the same type");
  }
  return success();
}

void getBufferizedBroadcastEffects(BroadcastOp op, SmallVectorImpl<MemoryEffect> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &op.getInputMutable(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &op.getDestinationMutable()[0],
                       SideEffects::DefaultResource::get());
  // The mem-based Broadcast API writes caller-provided scratch. Declare that
  // write effect to prevent reordering broadcasts that share the buffer.
  if (op.getTmpBuffer()) {
    effects.emplace_back(MemoryEffects::Write::get(), &op.getTmpBufferMutable()[0],
                         SideEffects::DefaultResource::get());
  }
}

}  // namespace

LogicalResult BroadcastOp::verify() {
  if (isBufferizedForm()) {
    if (failed(verifyBufferizedBroadcast(*this))) {
      return failure();
    }
  } else if (!isa<TensorType>(getInput().getType()) || !isa<TensorType>(getResult().getType())) {
    return emitOpError("logical form expects tensor input and result; memref form requires a destination");
  }
  if (failed(verifyBroadcastShape())) {
    return failure();
  }
  if (isBufferizedForm()) {
    return verifyCanonicalBufferDescriptors(getOperation());
  }
  return success();
}

FailureOr<Role> BroadcastOp::getRole() {
  return getCanonicalTargetRole(getOperation());
}

LogicalResult BroadcastOp::verifyBroadcastShape() {
  return verifyBroadcastShapeImpl(getOperation());
}

FailureOr<unsigned> BroadcastOp::getTiedResultIndex(unsigned destinationIndex) {
  if (destinationIndex != 0 || !getDestination() || getOperation()->getNumResults() != 1) {
    return failure();
  }
  return 0;
}

void BroadcastOp::getEffects(SmallVectorImpl<MemoryEffect> &effects) {
  if (isBufferizedForm()) {
    getBufferizedBroadcastEffects(*this, effects);
  }
}

Speculation::Speculatability BroadcastOp::getSpeculatability() {
  return isBufferizedForm() ? Speculation::NotSpeculatable : Speculation::Speculatable;
}
