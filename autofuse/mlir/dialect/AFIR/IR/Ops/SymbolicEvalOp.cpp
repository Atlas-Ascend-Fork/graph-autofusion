/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include <set>
#include <string>

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::afir {

LogicalResult SymbolicEvalOp::verify() {
  Result<Expression> expression = Expression::Parse(getExpr().getValue().getValue());
  if (!expression) {
    return emitOpError("contains an invalid expression: ") << expression.error().message;
  }
  Result<std::vector<std::string>> symbols = detail::ExpressionAccess::GetFreeSymbols(expression.value());
  if (!symbols) {
    return emitOpError("cannot inspect expression symbols: ") << symbols.error().message;
  }

  std::set<std::string> actualSymbols(symbols.value().begin(), symbols.value().end());
  if (actualSymbols.size() != getInputs().size()) {
    return emitOpError("requires exactly one operand for every placeholder; expression has ")
           << actualSymbols.size() << " placeholders but op has " << getInputs().size() << " operands";
  }
  for (size_t index = 0; index < getInputs().size(); ++index) {
    const std::string expected = std::string(detail::kOperandPlaceholderPrefix) + std::to_string(index);
    if (actualSymbols.erase(expected) == 0) {
      return emitOpError("is missing placeholder '") << expected << "'";
    }
  }
  if (!actualSymbols.empty()) {
    return emitOpError("contains non-placeholder symbol '") << *actualSymbols.begin() << "'";
  }

  Type inputType;
  for (Value input : getInputs()) {
    if (!inputType) {
      inputType = input.getType();
    } else if (input.getType() != inputType) {
      return emitOpError("requires operands to have one common type");
    }
  }
  Type resultType = getResult().getType();
  Type numericType = inputType ? inputType : resultType;
  const bool supportedNumericType = numericType.isIndex() || numericType.isInteger(64) || numericType.isF64();
  if (!supportedNumericType && !numericType.isInteger(1)) {
    return emitOpError("supports only index, i64, f64, and i1 profiles");
  }
  if (resultType != numericType && !resultType.isInteger(1)) {
    return emitOpError("result must match the operand type or be i1 for a predicate");
  }
  return success();
}

}  // namespace mlir::afir
