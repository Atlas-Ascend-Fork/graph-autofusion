/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_program_evaluation.h"
namespace att {
namespace {
using Kind = AxesProgramInputKind;
using Role = AxesProgramEvaluationRole;
using Type = TilingProgramIRValueType;
TilingStageError Invalid() {
  return {TilingExpressionStatus::kInvalidProgram, UINT32_MAX, {}};
}
bool ValidInput(const AxesProgramEvaluationDefinition &definition, const AxesProgramInput &input) {
  switch (input.kind) {
    case Kind::kRelatedTilingVariable:
      return definition.role == Role::kConstraint && input.index < definition.related_tiling_count;
    case Kind::kRelatedInputShape:
      return definition.role == Role::kConstraint && input.index < definition.related_input_count;
    case Kind::kResourceValue:
      return definition.role == Role::kConstraint && input.index == 0;
    case Kind::kParentVariable:
      return definition.role == Role::kUpperBound && input.index < definition.parent_count;
    default:
      return false;
  }
}
}  // namespace
TilingStageError BuildAxesProgramEvaluation(const AxesProgramEvaluationDefinition &definition,
                                            std::shared_ptr<const AxesProgramEvaluation> *output) {
  if (output == nullptr) {
    return {TilingExpressionStatus::kInvalidArgument, UINT32_MAX, {}};
  }
  if ((definition.role != Role::kConstraint && definition.role != Role::kUpperBound) ||
      definition.related_tiling_count > kTilingExpressionMaxItems ||
      definition.related_input_count > kTilingExpressionMaxItems ||
      definition.parent_count > kTilingExpressionMaxItems ||
      (definition.role == Role::kConstraint && definition.parent_count != 0) ||
      (definition.role == Role::kUpperBound &&
       (definition.related_tiling_count != 0 || definition.related_input_count != 0)) ||
      definition.inputs.size() > kTilingExpressionMaxItems ||
      definition.inputs.size() != definition.stage.input_types.size() || definition.stage.expressions.empty() ||
      definition.stage.expressions.back().result_type != Type::kInt64) {
    return Invalid();
  }
  for (size_t i = 0; i < definition.inputs.size(); ++i) {
    if (definition.stage.input_types[i] != Type::kInt64 || !ValidInput(definition, definition.inputs[i])) {
      return Invalid();
    }
  }
  std::shared_ptr<const TilingProgramIRCompiledStage> stage;
  const auto error = BuildTilingProgramIRStage(definition.stage, &stage);
  if (error.status != TilingExpressionStatus::kOk) {
    return error;
  }
  auto evaluation = std::make_shared<AxesProgramEvaluation>(AxesProgramEvaluation::ConstructionKey{});
  evaluation->role_ = definition.role;
  evaluation->related_tiling_count_ = definition.related_tiling_count;
  evaluation->related_input_count_ = definition.related_input_count;
  evaluation->parent_count_ = definition.parent_count;
  evaluation->inputs_ = definition.inputs;
  evaluation->stage_ = std::move(stage);
  *output = std::move(evaluation);
  return {};
}
}  // namespace att
