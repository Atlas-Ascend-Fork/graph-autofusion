/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_SYMBOLIC_SYMBOLICTRANSFORMDETAIL_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_SYMBOLIC_SYMBOLICTRANSFORMDETAIL_H

#include <cstdint>

#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::afir::detail {

// Positive powers are expanded during lowering. This is a code-size bound, not an overflow proof;
// overflow is still governed by `overflow_checked`.
inline constexpr int64_t kMaxSymbolicPowExponent = 8;

FailureOr<int64_t> parseInteger(const ExpressionNode &node);
FailureOr<bool> parseBoolean(const ExpressionNode &node);
bool isInverse(const ExpressionNode &node);
FailureOr<unsigned> getPlaceholderIndex(const ExpressionNode &node);

LogicalResult verifySymbolicSafety(SymbolicEvalOp op);
LogicalResult verifySymbolicIndexSafety(const ExpressionNode &node, SymbolicEvalOp op);
void populateSymbolicLoweringPatterns(RewritePatternSet &patterns);

}  // namespace mlir::afir::detail

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_TRANSFORMS_SYMBOLIC_SYMBOLICTRANSFORMDETAIL_H
