/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_TILING_STAGE_H_
#define ATT_PROGRAM_TILING_STAGE_H_
#include "tiling_expression.h"
namespace att {
struct TilingProgramIRStageConditionalReturn {
  uint32_t condition_step = 0;
  TilingProgramIRValue value;
  // UINT32_MAX selects the constant above. Otherwise value must be default-initialized.
  uint32_t value_step = UINT32_MAX;
};
struct TilingProgramIRStageDefinition {
  std::vector<TilingProgramIRValueType> input_types;
  // Each expression initializes one local slot. Inputs, then previous locals, form its exact type table.
  // The last local is the normal return value. All assignment/return conversions must be explicit in the code.
  std::vector<TilingProgramIRExpressionCode> expressions;
  // Strictly increasing bool steps. A true condition returns the constant or an earlier local before subsequent steps.
  std::vector<TilingProgramIRStageConditionalReturn> conditional_returns{};
};
struct TilingStageError {
  TilingExpressionStatus status = TilingExpressionStatus::kOk;
  uint32_t step = UINT32_MAX;
  TilingExpressionError expression_error;
};
class TilingProgramIRCompiledStage;
TilingStageError BuildTilingProgramIRStage(const TilingProgramIRStageDefinition &definition,
                                           std::shared_ptr<const TilingProgramIRCompiledStage> *output);
size_t GetTilingStageScratchElements(const TilingProgramIRCompiledStage &stage);
// Scratch contains live Value objects, is exclusive to this invocation, and must not alias reader state/input/output.
// No runtime allocation. Failure preserves output, but scratch contents are unspecified.
TilingStageError EvaluateTilingProgramIRStage(const TilingProgramIRCompiledStage &stage, void *context,
                                              TilingExpressionReadSlot read_slot, TilingProgramIRValue *scratch,
                                              size_t scratch_elements, TilingProgramIRValue *output);
class TilingProgramIRCompiledStage final {
 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit TilingProgramIRCompiledStage(ConstructionKey) {}

 private:
  std::vector<std::shared_ptr<const TilingProgramIRCompiledExpression>> expressions_;
  std::vector<TilingProgramIRStageConditionalReturn> conditional_returns_;
  size_t input_count_ = 0;
  size_t expression_scratch_ = 0;
  friend TilingStageError BuildTilingProgramIRStage(const TilingProgramIRStageDefinition &,
                                                    std::shared_ptr<const TilingProgramIRCompiledStage> *);
  friend size_t GetTilingStageScratchElements(const TilingProgramIRCompiledStage &);
  friend TilingStageError EvaluateTilingProgramIRStage(const TilingProgramIRCompiledStage &, void *,
                                                       TilingExpressionReadSlot, TilingProgramIRValue *, size_t,
                                                       TilingProgramIRValue *);
};
}  // namespace att
#endif  // ATT_PROGRAM_TILING_STAGE_H_
