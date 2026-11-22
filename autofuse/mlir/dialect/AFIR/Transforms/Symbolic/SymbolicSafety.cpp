/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"

#include <string>

#include "dialect/AFIR/Support/SymbolicExpression.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::afir::detail {
namespace {

bool isPositiveArgument(SymbolicEvalOp op, unsigned index) {
  DenseI64ArrayAttr positiveArguments = op.getPositiveArgsAttr();
  if (!positiveArguments) {
    return false;
  }
  return llvm::is_contained(positiveArguments.asArrayRef(), static_cast<int64_t>(index));
}

bool isKnownPositive(const ExpressionNode &node, SymbolicEvalOp op) {
  if (FailureOr<int64_t> value = parseInteger(node); succeeded(value)) {
    return *value > 0;
  }
  if (FailureOr<unsigned> index = getPlaceholderIndex(node); succeeded(index)) {
    return *index < op.getInputs().size() && isPositiveArgument(op, *index);
  }
  if (node.kind == NodeKind::Real) {
    try {
      return std::stod(node.value) > 0.0;
    } catch (...) {
      return false;
    }
  }
  if (node.kind == NodeKind::Rational && node.arguments.size() == 2) {
    FailureOr<int64_t> numerator = parseInteger(node.arguments[0]);
    FailureOr<int64_t> denominator = parseInteger(node.arguments[1]);
    return succeeded(numerator) && succeeded(denominator) && ((*numerator > 0) == (*denominator > 0));
  }
  if (node.kind == NodeKind::Mul) {
    return llvm::all_of(node.arguments, [&](const ExpressionNode &argument) {
      return !isInverse(argument) && isKnownPositive(argument, op);
    });
  }
  return false;
}

LogicalResult verifyF64Safety(const ExpressionNode &node, SymbolicEvalOp op, bool inverseAllowed = false) {
  if (node.kind == NodeKind::Mod) {
    return op.emitOpError("floating-point modulo is not supported");
  }
  if (node.kind == NodeKind::Log && (node.arguments.size() != 1 || !isKnownPositive(node.arguments.front(), op))) {
    return op.emitOpError("cannot prove logarithm argument is positive");
  }
  if (node.kind == NodeKind::Pow) {
    if (inverseAllowed && isInverse(node)) {
      if (!isKnownPositive(node.arguments[0], op)) {
        return op.emitOpError("cannot prove floating-point divisor is positive");
      }
      return success();
    }
    if (node.arguments.size() != 2 || !isKnownPositive(node.arguments[0], op)) {
      return op.emitOpError("cannot prove floating-point power base is positive");
    }
  }
  for (const ExpressionNode &argument : node.arguments) {
    if (failed(verifyF64Safety(argument, op, node.kind == NodeKind::Mul))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyIndexSafety(const ExpressionNode &node, SymbolicEvalOp op, bool inverseAllowed = false,
                                bool rationalAllowed = false);

LogicalResult verifyRationalSafety(const ExpressionNode &node, SymbolicEvalOp op, bool rationalAllowed) {
  if (!rationalAllowed || node.arguments.size() != 2) {
    return op.emitOpError("contains a rational that is not a ceiling/floor scaling factor");
  }
  FailureOr<int64_t> numerator = parseInteger(node.arguments[0]);
  FailureOr<int64_t> denominator = parseInteger(node.arguments[1]);
  if (failed(numerator) || failed(denominator) || *numerator < 0 || *denominator <= 0) {
    return op.emitOpError("contains an unsafe rational scaling factor");
  }
  return success();
}

LogicalResult verifyPowerSafety(const ExpressionNode &node, SymbolicEvalOp op, bool inverseAllowed) {
  if (isInverse(node)) {
    if (!inverseAllowed || !isKnownPositive(node.arguments[0], op)) {
      return op.emitOpError(
          "contains an unsafe or unsupported power; integer lowering only accepts a positive division denominator");
    }
    return success();
  }
  if (node.arguments.size() != 2) {
    return op.emitOpError("contains an unsupported power form");
  }
  FailureOr<int64_t> exponent = parseInteger(node.arguments[1]);
  if (failed(exponent) || *exponent < 0) {
    return op.emitOpError("contains an unsupported negative or symbolic power");
  }
  if (*exponent > kMaxSymbolicPowExponent) {
    return op.emitOpError("integer power exponent exceeds the lowering limit of ") << kMaxSymbolicPowExponent;
  }
  if (*exponent > 1 && !op.getOverflowCheckedAttr()) {
    return op.emitOpError("integer power requires an explicit overflow_checked proof");
  }
  return verifyIndexSafety(node.arguments[0], op);
}

LogicalResult verifyMultiplicationSafety(const ExpressionNode &node, SymbolicEvalOp op, bool rationalAllowed) {
  const size_t numeratorCount =
      llvm::count_if(node.arguments, [](const ExpressionNode &argument) { return !isInverse(argument); });
  if (numeratorCount > 1 && !op.getOverflowCheckedAttr()) {
    return op.emitOpError("integer multiplication requires an explicit overflow_checked proof");
  }
  for (const ExpressionNode &argument : node.arguments) {
    if (failed(verifyIndexSafety(argument, op, true, rationalAllowed))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyIndexSafety(const ExpressionNode &node, SymbolicEvalOp op, bool inverseAllowed,
                                bool rationalAllowed) {
  if (node.kind == NodeKind::Real || node.kind == NodeKind::Log) {
    return op.emitOpError("contains a node that is not valid in the exact integer profile");
  }
  if (node.kind == NodeKind::Rational) {
    return verifyRationalSafety(node, op, rationalAllowed);
  }
  if (node.kind == NodeKind::Pow) {
    return verifyPowerSafety(node, op, inverseAllowed);
  }
  if (node.kind == NodeKind::Ceiling || node.kind == NodeKind::Floor) {
    if (node.arguments.size() != 1) {
      return op.emitOpError("contains an unsupported rounding form");
    }
    return verifyIndexSafety(node.arguments.front(), op, false, true);
  }
  if (node.kind == NodeKind::Mod && (node.arguments.size() != 2 || !isKnownPositive(node.arguments[1], op))) {
    return op.emitOpError("cannot prove modulo divisor is positive");
  }
  if (node.kind == NodeKind::Add && !op.getOverflowCheckedAttr()) {
    return op.emitOpError("integer addition requires an explicit overflow_checked proof");
  }
  if (node.kind == NodeKind::Mul) {
    return verifyMultiplicationSafety(node, op, rationalAllowed);
  }
  for (const ExpressionNode &argument : node.arguments) {
    if (failed(verifyIndexSafety(argument, op, node.kind == NodeKind::Mul, false))) {
      return failure();
    }
  }
  return success();
}
}  // namespace

LogicalResult verifySymbolicIndexSafety(const ExpressionNode &node, SymbolicEvalOp op) {
  return verifyIndexSafety(node, op);
}

LogicalResult verifySymbolicSafety(SymbolicEvalOp op) {
  const bool indexProfile = llvm::all_of(op.getInputs(), [](Value value) { return value.getType().isIndex(); }) &&
                            (op.getResult().getType().isIndex() || op.getResult().getType().isInteger(1));
  const bool f64Profile = llvm::all_of(op.getInputs(), [](Value value) { return value.getType().isF64(); }) &&
                          (op.getResult().getType().isF64() || op.getResult().getType().isInteger(1));
  if (!indexProfile && !f64Profile) {
    return op.emitOpError("does not use a supported homogeneous safety profile");
  }
  Result<Expression> expression = Expression::Parse(op.getExpr().getValue().getValue());
  Result<ExpressionNode> tree =
      expression ? ExpressionAccess::GetTree(expression.value()) : Result<ExpressionNode>(expression.error());
  if (!tree) {
    op.emitOpError("cannot inspect expression safety: ") << tree.error().message;
    return failure();
  }
  return indexProfile ? verifySymbolicIndexSafety(tree.value(), op) : verifyF64Safety(tree.value(), op);
}

}  // namespace mlir::afir::detail
