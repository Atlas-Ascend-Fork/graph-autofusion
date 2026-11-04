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

#include <limits>
#include <string>

namespace mlir::afir::detail {

FailureOr<int64_t> parseInteger(const ExpressionNode &node) {
  if (node.kind != NodeKind::Integer) {
    return failure();
  }
  try {
    size_t parsed = 0;
    int64_t value = std::stoll(node.value, &parsed);
    return parsed == node.value.size() ? FailureOr<int64_t>(value) : FailureOr<int64_t>(failure());
  } catch (...) {
    return failure();
  }
}

FailureOr<bool> parseBoolean(const ExpressionNode &node) {
  if (node.kind != NodeKind::Boolean) {
    return failure();
  }
  if (node.value == kBooleanTrueLiteral) {
    return true;
  }
  if (node.value == kBooleanFalseLiteral) {
    return false;
  }
  return failure();
}

bool isInverse(const ExpressionNode &node) {
  if (node.kind != NodeKind::Pow || node.arguments.size() != 2) {
    return false;
  }
  FailureOr<int64_t> exponent = parseInteger(node.arguments[1]);
  return succeeded(exponent) && *exponent == -1;
}

FailureOr<unsigned> getPlaceholderIndex(const ExpressionNode &node) {
  if (node.kind != NodeKind::Symbol || node.value.rfind(kOperandPlaceholderPrefix, 0) != 0 ||
      node.value.size() == kOperandPlaceholderPrefix.size()) {
    return failure();
  }
  try {
    size_t parsed = 0;
    unsigned long value = std::stoul(node.value.substr(kOperandPlaceholderPrefix.size()), &parsed);
    if (parsed != node.value.size() - kOperandPlaceholderPrefix.size() ||
        value > std::numeric_limits<unsigned>::max()) {
      return failure();
    }
    return static_cast<unsigned>(value);
  } catch (...) {
    return failure();
  }
}

}  // namespace mlir::afir::detail
