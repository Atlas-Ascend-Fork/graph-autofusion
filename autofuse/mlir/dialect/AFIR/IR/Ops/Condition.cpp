/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Condition.cpp - AFIR condition op implementation ---------*- C++ -*-===//
//
// Ported from Ascend-MLIR-gsr: select / where ternary verifiers.
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

bool areCompatibleDims(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

LogicalResult verifyValueShapes(Operation *op, ShapedType lhs, ShapedType rhs, ShapedType result) {
  if (!lhs.hasRank() || !rhs.hasRank() || !result.hasRank()) return success();
  if (lhs.getRank() != rhs.getRank() || lhs.getRank() != result.getRank()) {
    return op->emitOpError("value operands and result must have the same rank");
  }
  for (auto [lhsDim, rhsDim, resultDim] : llvm::zip(lhs.getShape(), rhs.getShape(), result.getShape())) {
    if (!areCompatibleDims(lhsDim, rhsDim) || !areCompatibleDims(lhsDim, resultDim)) {
      return op->emitOpError("value operands and result must have compatible shapes");
    }
  }
  return success();
}

LogicalResult verifyConditionShape(Operation *op, ShapedType condition, ShapedType result) {
  if (!condition.hasRank() || !result.hasRank()) return success();
  ArrayRef<int64_t> conditionShape = condition.getShape();
  ArrayRef<int64_t> resultShape = result.getShape();
  if (resultShape.size() < conditionShape.size()) {
    return op->emitOpError("condition rank must be <= result rank");
  }
  const size_t offset = resultShape.size() - conditionShape.size();
  for (size_t i = 0; i < conditionShape.size(); ++i) {
    const int64_t conditionDim = conditionShape[i];
    if (conditionDim != 1 && !areCompatibleDims(conditionDim, resultShape[i + offset])) {
      return op->emitOpError("condition shape must be broadcastable to result shape");
    }
  }
  return success();
}

LogicalResult verifyTernaryOp(Operation *op) {
  if (op->getNumOperands() != 3) {
    return op->emitOpError("expected 3 operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto conditionType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto input1Type = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto input2Type = mlir::dyn_cast<ShapedType>(op->getOperand(2).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!conditionType || !input1Type || !input2Type || !resultType) {
    return op->emitOpError("expected tensor operands and result");
  }
  if (failed(verifyValueShapes(op, input1Type, input2Type, resultType))) return failure();
  return verifyConditionShape(op, conditionType, resultType);
}

}  // namespace

LogicalResult SelectOp::verify() {
  return verifyTernaryOp(getOperation());
}
LogicalResult WhereOp::verify() {
  return verifyTernaryOp(getOperation());
}
