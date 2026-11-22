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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::afir {
namespace {

using detail::ExpressionNode;
using detail::getPlaceholderIndex;
using detail::isInverse;
using detail::kMaxSymbolicPowExponent;
using detail::NodeKind;
using detail::parseBoolean;
using detail::parseInteger;

bool multiplyInto(int64_t lhs, int64_t rhs, int64_t &result) {
  return !__builtin_mul_overflow(lhs, rhs, &result);
}

FailureOr<Value> lowerBoolean(PatternRewriter &rewriter, Location location, const ExpressionNode &node) {
  FailureOr<bool> value = parseBoolean(node);
  if (failed(value)) {
    return failure();
  }
  return rewriter.create<arith::ConstantOp>(location, rewriter.getBoolAttr(*value)).getResult();
}

class IndexExpressionLowerer {
 public:
  IndexExpressionLowerer(PatternRewriter &rewriter, SymbolicEvalOp op)
      : rewriter_(rewriter), op_(op), location_(op.getLoc()) {}

  FailureOr<Value> lower(const ExpressionNode &node) {
    switch (node.kind) {
      case NodeKind::Integer:
        return lowerInteger(node);
      case NodeKind::Symbol:
        return lowerSymbol(node);
      case NodeKind::Add:
        return lowerNary<index::AddOp>(node);
      case NodeKind::Mul:
        return lowerMultiply(node, false);
      case NodeKind::Pow:
        return lowerPow(node);
      case NodeKind::Max:
        return lowerNary<index::MaxUOp>(node);
      case NodeKind::Min:
        return lowerNary<index::MinUOp>(node);
      case NodeKind::Mod:
        return lowerBinary<index::RemUOp>(node);
      case NodeKind::Ceiling:
        return lowerRound(node, true);
      case NodeKind::Floor:
        return lowerRound(node, false);
      case NodeKind::Abs:
        return node.arguments.size() == 1 ? lower(node.arguments.front()) : FailureOr<Value>(failure());
      case NodeKind::Equal:
        return lowerCompare(node, index::IndexCmpPredicate::EQ);
      case NodeKind::NotEqual:
        return lowerCompare(node, index::IndexCmpPredicate::NE);
      case NodeKind::LessEqual:
        return lowerCompare(node, index::IndexCmpPredicate::ULE);
      case NodeKind::LessThan:
        return lowerCompare(node, index::IndexCmpPredicate::ULT);
      case NodeKind::LogicAnd:
        return lowerNary<arith::AndIOp>(node);
      case NodeKind::LogicOr:
        return lowerNary<arith::OrIOp>(node);
      case NodeKind::Boolean:
        return lowerBoolean(rewriter_, location_, node);
      default:
        return failure();
    }
  }

 private:
  FailureOr<Value> lowerInteger(const ExpressionNode &node) {
    FailureOr<int64_t> value = parseInteger(node);
    if (failed(value) || *value < 0) {
      return failure();
    }
    return rewriter_.create<arith::ConstantIndexOp>(location_, *value).getResult();
  }

  FailureOr<Value> lowerSymbol(const ExpressionNode &node) {
    FailureOr<unsigned> index = getPlaceholderIndex(node);
    if (failed(index) || *index >= op_.getInputs().size()) {
      return failure();
    }
    return op_.getInputs()[*index];
  }

  template <typename OpTy>
  FailureOr<Value> lowerBinary(const ExpressionNode &node) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<Value> lhs = lower(node.arguments[0]);
    FailureOr<Value> rhs = lower(node.arguments[1]);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return rewriter_.create<OpTy>(location_, *lhs, *rhs).getResult();
  }

  template <typename OpTy>
  FailureOr<Value> lowerNary(const ExpressionNode &node) {
    if (node.arguments.empty()) {
      return failure();
    }
    FailureOr<Value> result = lower(node.arguments.front());
    for (const ExpressionNode &argument : llvm::drop_begin(node.arguments)) {
      FailureOr<Value> rhs = lower(argument);
      if (failed(result) || failed(rhs)) {
        return failure();
      }
      result = rewriter_.create<OpTy>(location_, *result, *rhs).getResult();
    }
    return result;
  }

  FailureOr<Value> lowerProduct(ArrayRef<const ExpressionNode *> arguments) {
    if (arguments.empty()) {
      return Value();
    }
    FailureOr<Value> result = lower(*arguments.front());
    if (failed(result)) {
      return failure();
    }
    for (const ExpressionNode *argument : llvm::drop_begin(arguments)) {
      FailureOr<Value> rhs = lower(*argument);
      if (failed(rhs)) {
        return failure();
      }
      result = rewriter_.create<index::MulOp>(location_, *result, *rhs).getResult();
    }
    return result;
  }

  LogicalResult foldRationalFactor(const ExpressionNode &argument, int64_t &constantNumerator,
                                   int64_t &constantDenominator) {
    if (argument.arguments.size() != 2) {
      return failure();
    }
    FailureOr<int64_t> numerator = parseInteger(argument.arguments[0]);
    FailureOr<int64_t> denominator = parseInteger(argument.arguments[1]);
    if (failed(numerator) || failed(denominator) || *denominator == 0) {
      return failure();
    }
    if (!multiplyInto(constantNumerator, *numerator, constantNumerator) ||
        !multiplyInto(constantDenominator, *denominator, constantDenominator)) {
      return failure();
    }
    return success();
  }

  FailureOr<Value> lowerMultiply(const ExpressionNode &node, bool ceilDivision) {
    SmallVector<const ExpressionNode *> numerators;
    SmallVector<const ExpressionNode *> denominators;
    int64_t constantNumerator = 1;
    int64_t constantDenominator = 1;
    for (const ExpressionNode &argument : node.arguments) {
      if (argument.kind == NodeKind::Rational) {
        if (failed(foldRationalFactor(argument, constantNumerator, constantDenominator))) {
          return failure();
        }
      } else if (isInverse(argument)) {
        denominators.push_back(&argument.arguments[0]);
      } else {
        numerators.push_back(&argument);
      }
    }
    if (numerators.empty() && constantNumerator == 1 && constantDenominator == 1) {
      return failure();
    }

    FailureOr<Value> numeratorProduct = lowerProduct(numerators);
    if (failed(numeratorProduct)) {
      return failure();
    }
    Value numeratorValue = *numeratorProduct;
    if (constantNumerator != 1 || !numeratorValue) {
      Value constant = rewriter_.create<arith::ConstantIndexOp>(location_, constantNumerator).getResult();
      numeratorValue =
          numeratorValue ? rewriter_.create<index::MulOp>(location_, numeratorValue, constant).getResult() : constant;
    }

    FailureOr<Value> denominatorProduct = lowerProduct(denominators);
    if (failed(denominatorProduct)) {
      return failure();
    }
    Value denominatorValue = *denominatorProduct;
    if (constantDenominator != 1) {
      Value constant = rewriter_.create<arith::ConstantIndexOp>(location_, constantDenominator).getResult();
      denominatorValue = denominatorValue
                             ? rewriter_.create<index::MulOp>(location_, denominatorValue, constant).getResult()
                             : constant;
    }

    if (denominatorValue) {
      return ceilDivision ? rewriter_.create<index::CeilDivUOp>(location_, numeratorValue, denominatorValue).getResult()
                          : rewriter_.create<index::DivUOp>(location_, numeratorValue, denominatorValue).getResult();
    }
    return numeratorValue;
  }

  FailureOr<Value> lowerPow(const ExpressionNode &node) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<int64_t> exponent = parseInteger(node.arguments[1]);
    if (failed(exponent) || *exponent < 0 || *exponent > kMaxSymbolicPowExponent) {
      return failure();
    }
    FailureOr<Value> base = lower(node.arguments[0]);
    if (failed(base)) {
      return failure();
    }
    if (*exponent == 0) {
      return rewriter_.create<arith::ConstantIndexOp>(location_, 1).getResult();
    }
    Value result = *base;
    for (int64_t index = 1; index < *exponent; ++index) {
      result = rewriter_.create<index::MulOp>(location_, result, *base).getResult();
    }
    return result;
  }

  FailureOr<Value> lowerRound(const ExpressionNode &node, bool ceiling) {
    if (node.arguments.size() != 1) {
      return failure();
    }
    const ExpressionNode &argument = node.arguments.front();
    if (argument.kind == NodeKind::Mul) {
      return lowerMultiply(argument, ceiling);
    }
    return lower(argument);
  }

  FailureOr<Value> lowerCompare(const ExpressionNode &node, index::IndexCmpPredicate predicate) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<Value> lhs = lower(node.arguments[0]);
    FailureOr<Value> rhs = lower(node.arguments[1]);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return rewriter_.create<index::CmpOp>(location_, predicate, *lhs, *rhs).getResult();
  }

  PatternRewriter &rewriter_;
  SymbolicEvalOp op_;
  Location location_;
};

class F64ExpressionLowerer {
 public:
  F64ExpressionLowerer(PatternRewriter &rewriter, SymbolicEvalOp op)
      : rewriter_(rewriter), op_(op), location_(op.getLoc()) {}

  FailureOr<Value> lower(const ExpressionNode &node) {
    switch (node.kind) {
      case NodeKind::Integer:
      case NodeKind::Real:
        return lowerNumber(node);
      case NodeKind::Rational:
        return lowerRational(node);
      case NodeKind::Symbol:
        return lowerSymbol(node);
      case NodeKind::Add:
        return lowerNary<arith::AddFOp>(node);
      case NodeKind::Mul:
        return lowerMultiply(node);
      case NodeKind::Pow:
        return lowerBinary<math::PowFOp>(node);
      case NodeKind::Max:
        return lowerNary<arith::MaximumFOp>(node);
      case NodeKind::Min:
        return lowerNary<arith::MinimumFOp>(node);
      case NodeKind::Log:
        return lowerUnary<math::LogOp>(node);
      case NodeKind::Ceiling:
        return lowerUnary<math::CeilOp>(node);
      case NodeKind::Floor:
        return lowerUnary<math::FloorOp>(node);
      case NodeKind::Abs:
        return lowerUnary<math::AbsFOp>(node);
      case NodeKind::Equal:
        return lowerCompare(node, arith::CmpFPredicate::OEQ);
      case NodeKind::NotEqual:
        return lowerCompare(node, arith::CmpFPredicate::UNE);
      case NodeKind::LessEqual:
        return lowerCompare(node, arith::CmpFPredicate::OLE);
      case NodeKind::LessThan:
        return lowerCompare(node, arith::CmpFPredicate::OLT);
      case NodeKind::LogicAnd:
        return lowerNary<arith::AndIOp>(node);
      case NodeKind::LogicOr:
        return lowerNary<arith::OrIOp>(node);
      case NodeKind::Boolean:
        return lowerBoolean(rewriter_, location_, node);
      default:
        return failure();
    }
  }

 private:
  FailureOr<Value> lowerNumber(const ExpressionNode &node) {
    try {
      const double value =
          node.kind == NodeKind::Integer ? static_cast<double>(std::stoll(node.value)) : std::stod(node.value);
      return rewriter_.create<arith::ConstantOp>(location_, rewriter_.getF64FloatAttr(value)).getResult();
    } catch (...) {
      return failure();
    }
  }

  FailureOr<Value> lowerRational(const ExpressionNode &node) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<int64_t> numerator = parseInteger(node.arguments[0]);
    FailureOr<int64_t> denominator = parseInteger(node.arguments[1]);
    if (failed(numerator) || failed(denominator) || *denominator == 0) {
      return failure();
    }
    const double value = static_cast<double>(*numerator) / static_cast<double>(*denominator);
    return rewriter_.create<arith::ConstantOp>(location_, rewriter_.getF64FloatAttr(value)).getResult();
  }

  FailureOr<Value> lowerSymbol(const ExpressionNode &node) {
    FailureOr<unsigned> index = getPlaceholderIndex(node);
    if (failed(index) || *index >= op_.getInputs().size()) {
      return failure();
    }
    return op_.getInputs()[*index];
  }

  template <typename OpTy>
  FailureOr<Value> lowerUnary(const ExpressionNode &node) {
    if (node.arguments.size() != 1) {
      return failure();
    }
    FailureOr<Value> value = lower(node.arguments.front());
    return succeeded(value) ? FailureOr<Value>(rewriter_.create<OpTy>(location_, *value).getResult())
                            : FailureOr<Value>(failure());
  }

  template <typename OpTy>
  FailureOr<Value> lowerBinary(const ExpressionNode &node) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<Value> lhs = lower(node.arguments[0]);
    FailureOr<Value> rhs = lower(node.arguments[1]);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return rewriter_.create<OpTy>(location_, *lhs, *rhs).getResult();
  }

  template <typename OpTy>
  FailureOr<Value> lowerNary(const ExpressionNode &node) {
    if (node.arguments.empty()) {
      return failure();
    }
    FailureOr<Value> result = lower(node.arguments.front());
    for (const ExpressionNode &argument : llvm::drop_begin(node.arguments)) {
      FailureOr<Value> rhs = lower(argument);
      if (failed(result) || failed(rhs)) {
        return failure();
      }
      result = rewriter_.create<OpTy>(location_, *result, *rhs).getResult();
    }
    return result;
  }

  FailureOr<Value> lowerMultiply(const ExpressionNode &node) {
    SmallVector<const ExpressionNode *> numerators;
    SmallVector<const ExpressionNode *> denominators;
    for (const ExpressionNode &argument : node.arguments) {
      (isInverse(argument) ? denominators : numerators)
          .push_back(isInverse(argument) ? &argument.arguments[0] : &argument);
    }
    if (numerators.empty()) {
      return failure();
    }
    FailureOr<Value> numerator = lower(*numerators.front());
    for (const ExpressionNode *argument : llvm::drop_begin(numerators)) {
      FailureOr<Value> rhs = lower(*argument);
      if (failed(numerator) || failed(rhs)) {
        return failure();
      }
      numerator = rewriter_.create<arith::MulFOp>(location_, *numerator, *rhs).getResult();
    }
    for (const ExpressionNode *argument : denominators) {
      FailureOr<Value> denominator = lower(*argument);
      if (failed(numerator) || failed(denominator)) {
        return failure();
      }
      numerator = rewriter_.create<arith::DivFOp>(location_, *numerator, *denominator).getResult();
    }
    return numerator;
  }

  FailureOr<Value> lowerCompare(const ExpressionNode &node, arith::CmpFPredicate predicate) {
    if (node.arguments.size() != 2) {
      return failure();
    }
    FailureOr<Value> lhs = lower(node.arguments[0]);
    FailureOr<Value> rhs = lower(node.arguments[1]);
    if (failed(lhs) || failed(rhs)) {
      return failure();
    }
    return rewriter_.create<arith::CmpFOp>(location_, predicate, *lhs, *rhs).getResult();
  }

  PatternRewriter &rewriter_;
  SymbolicEvalOp op_;
  Location location_;
};

struct LowerSymbolicEvalPattern : public OpRewritePattern<SymbolicEvalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SymbolicEvalOp op, PatternRewriter &rewriter) const override {
    const bool indexProfile = llvm::all_of(op.getInputs(), [](Value value) { return value.getType().isIndex(); }) &&
                              (op.getResult().getType().isIndex() || op.getResult().getType().isInteger(1));
    const bool f64Profile = llvm::all_of(op.getInputs(), [](Value value) { return value.getType().isF64(); }) &&
                            (op.getResult().getType().isF64() || op.getResult().getType().isInteger(1));
    if (!indexProfile && !f64Profile) {
      return rewriter.notifyMatchFailure(op, "operation does not use a supported homogeneous lowering profile");
    }
    Result<Expression> expression = Expression::Parse(op.getExpr().getValue().getValue());
    if (!expression) {
      return rewriter.notifyMatchFailure(op, expression.error().message);
    }
    Result<ExpressionNode> tree = detail::ExpressionAccess::GetTree(expression.value());
    if (!tree) {
      return rewriter.notifyMatchFailure(op, tree.error().message);
    }
    FailureOr<Value> lowered = indexProfile ? IndexExpressionLowerer(rewriter, op).lower(tree.value())
                                            : F64ExpressionLowerer(rewriter, op).lower(tree.value());
    if (failed(lowered)) {
      return rewriter.notifyMatchFailure(op, "expression is unsupported by the index lowering profile");
    }
    rewriter.replaceOp(op, *lowered);
    return success();
  }
};

}  // namespace

void detail::populateSymbolicLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerSymbolicEvalPattern>(patterns.getContext());
}

}  // namespace mlir::afir
