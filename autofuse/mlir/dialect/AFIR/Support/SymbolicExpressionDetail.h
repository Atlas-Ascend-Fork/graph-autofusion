/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSIONDETAIL_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSIONDETAIL_H

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "dialect/AFIR/Support/SymbolicExpression.h"

namespace mlir::afir::detail {

inline constexpr std::string_view kOperandPlaceholderPrefix = "__arg";
inline constexpr std::string_view kBooleanTrueLiteral = "True";
inline constexpr std::string_view kBooleanFalseLiteral = "False";

enum class NodeKind {
  Integer,
  Real,
  Boolean,
  Symbol,
  Add,
  Mul,
  Pow,
  Max,
  Min,
  Mod,
  Log,
  Ceiling,
  Floor,
  Abs,
  Rational,
  Equal,
  NotEqual,
  LessEqual,
  LessThan,
  LogicAnd,
  LogicOr,
};

struct ExpressionNode {
  NodeKind kind;
  std::string value;
  std::vector<ExpressionNode> arguments;
};

struct NormalizedExpression {
  Expression expression;
  std::vector<std::string> symbols;
};

enum class TruthValue {
  False,
  True,
  Unknown,
};

// Internal symbolic interface, kept outside the public Expression API to avoid
// exposing SymEngine tree traversal to ordinary callers.
class ExpressionAccess {
 public:
  static Result<ExpressionNode> GetTree(const Expression &expression);
  static Result<std::vector<std::string>> GetFreeSymbols(const Expression &expression);
  static Result<NormalizedExpression> NormalizeFreeSymbols(const Expression &expression);
  static Result<Expression> Substitute(const Expression &expression,
                                       const std::map<std::string, std::string> &replacements);
  static Result<double> Evaluate(const Expression &expression, const std::map<std::string, double> &bindings);
  static Result<TruthValue> Prove(const Expression &predicate);
};

}  // namespace mlir::afir::detail

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSIONDETAIL_H
