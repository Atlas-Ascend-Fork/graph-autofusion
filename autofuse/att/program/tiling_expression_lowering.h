/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_TILING_EXPRESSION_LOWERING_H_
#define ATT_PROGRAM_TILING_EXPRESSION_LOWERING_H_
#include <array>
#include "tiling_expression.h"
namespace att {
// First three identities come from the reference expression IR; typed extensions are explicit.
enum class TilingProgramIRExpressionKind : uint8_t {
  kConstant = 1,
  kSymbol = 2,
  kOperation = 3,
  kCast = 4,
  kConditional = 5
};
struct TilingProgramIRExpression {
  TilingProgramIRExpressionKind kind = TilingProgramIRExpressionKind::kConstant;
  TilingProgramIRValueType value_type = TilingProgramIRValueType::kInt64;
  TilingProgramIROpcode opcode = TilingProgramIROpcode::kNone;
  uint32_t constant_id = 0, symbol_id = 0, arg_count = 0;
  std::array<uint32_t, kTilingTernaryArity> args{};
};
struct TilingProgramIRExpressionDefinition {
  // Topological order, last node is the root, all nodes must be reachable from that root.
  std::vector<TilingProgramIRExpression> nodes;
  std::vector<TilingProgramIRValue> constants;
  std::vector<TilingProgramIRValueType> input_types;
};
struct TilingExpressionLoweringError {
  TilingExpressionStatus status = TilingExpressionStatus::kOk;
  uint32_t node = UINT32_MAX;
  TilingExpressionError code_error;
};
// Compile-time only. No evaluation, folding, reassociation, reciprocal rewrite or common-value cache.
// Shared nodes are expanded at each occurrence, preserving branch-local reads and eager arguments.
// Failure preserves output; code is published only after the common VM verifier succeeds.
TilingExpressionLoweringError LowerTilingProgramIRExpression(const TilingProgramIRExpressionDefinition &definition,
                                                             TilingProgramIRExpressionCode *output);
}  // namespace att
#endif  // ATT_PROGRAM_TILING_EXPRESSION_LOWERING_H_
