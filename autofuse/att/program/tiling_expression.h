/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_TILING_EXPRESSION_H_
#define ATT_PROGRAM_TILING_EXPRESSION_H_
#include <cstddef>
#include <memory>
#include <vector>
#include "tiling_program_value.h"
namespace att {
// Preserve reference instruction identities. Cast is the first typed extension.
enum class TilingProgramIRInstructionKind : uint8_t {
  kConstant = 1,
  kSymbol = 2,
  kOperation = 3,
  kJumpIfFalse = 4,
  kJumpIfTrue = 5,
  kPushBoolean = 6,
  kJump = 7,
  kReciprocal = 8,
  kCast = 9
};
struct TilingProgramIRInstruction {
  TilingProgramIRInstructionKind kind = TilingProgramIRInstructionKind::kConstant;
  TilingProgramIROpcode opcode = TilingProgramIROpcode::kNone;
  uint32_t constant_id = 0, symbol_id = 0, jump = 0, arg_count = 0;
  bool boolean_value = false;
  TilingProgramIRValueType value_type = TilingProgramIRValueType::kInt64;
};
struct TilingProgramIRExpressionCode {
  std::vector<TilingProgramIRInstruction> instructions;
  std::vector<TilingProgramIRValue> constants;
  std::vector<TilingProgramIRValueType> input_types;
  TilingProgramIRValueType result_type = TilingProgramIRValueType::kInt64;
};
enum class TilingExpressionStatus {
  kOk,
  kInvalidArgument,
  kInvalidProgram,
  kUnsupported,
  kInputError,
  kEvaluationError,
  kBufferTooSmall
};
struct TilingExpressionError {
  TilingExpressionStatus status = TilingExpressionStatus::kOk;
  uint32_t instruction = UINT32_MAX;
  TilingProgramValueStatus value_status = TilingProgramValueStatus::kOk;
};
constexpr size_t kTilingExpressionMaxItems = 65536U;
constexpr size_t kTilingExpressionMaxStack = 256U;
using TilingExpressionReadSlot = TilingProgramValueStatus (*)(void *, uint32_t, TilingProgramIRValue *);
class TilingProgramIRCompiledExpression;
TilingExpressionError BuildTilingProgramIRExpression(const TilingProgramIRExpressionCode &code,
                                                     std::shared_ptr<const TilingProgramIRCompiledExpression> *output);
TilingExpressionError EvaluateTilingProgramIRExpression(const TilingProgramIRCompiledExpression &code, void *context,
                                                        TilingExpressionReadSlot read_slot,
                                                        TilingProgramIRValue *scratch, size_t scratch_elements,
                                                        TilingProgramIRValue *output);
// Query is O(1) and never reads inputs or executes code. Capacity is in live, aligned Value objects.
size_t GetTilingExpressionScratchElements(const TilingProgramIRCompiledExpression &code);
// Scratch is exclusive to this invocation, disjoint from input storage/reader state and output.
// Sequential reuse is allowed; simultaneous or nested evaluations require separate regions.
// Immutable after validation. Slot reader accesses caller-owned variables.
class TilingProgramIRCompiledExpression final {
 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit TilingProgramIRCompiledExpression(ConstructionKey, const TilingProgramIRExpressionCode &code)
      : code_(code) {}

 private:
  TilingProgramIRExpressionCode code_;
  size_t stack_size_ = 0;
  friend TilingExpressionError BuildTilingProgramIRExpression(
      const TilingProgramIRExpressionCode &, std::shared_ptr<const TilingProgramIRCompiledExpression> *);
  friend TilingExpressionError EvaluateTilingProgramIRExpression(const TilingProgramIRCompiledExpression &, void *,
                                                                 TilingExpressionReadSlot, TilingProgramIRValue *,
                                                                 size_t, TilingProgramIRValue *);
  friend size_t GetTilingExpressionScratchElements(const TilingProgramIRCompiledExpression &);
};
}  // namespace att
#endif  // ATT_PROGRAM_TILING_EXPRESSION_H_
