/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_stage.h"
#include <algorithm>
namespace att {
namespace {
using Status = TilingExpressionStatus;
using ValueStatus = TilingProgramValueStatus;
TilingStageError Error(Status status, size_t step = UINT32_MAX, TilingExpressionError expression = {}) {
  return {status, static_cast<uint32_t>(step), expression};
}
struct StageReader {
  void *context;
  TilingExpressionReadSlot read_slot;
  size_t input_count;
  const TilingProgramIRValue *locals;
  size_t initialized;
};
ValueStatus ReadStageSlot(void *context, uint32_t slot, TilingProgramIRValue *output) {
  const auto &reader = *static_cast<StageReader *>(context);
  if (slot < reader.input_count) {
    return reader.read_slot == nullptr ? ValueStatus::kInvalidArgument : reader.read_slot(reader.context, slot, output);
  }
  const size_t local = static_cast<size_t>(slot) - reader.input_count;
  if (local >= reader.initialized) {
    return ValueStatus::kInvalidArgument;
  }
  *output = reader.locals[local];
  return ValueStatus::kOk;
}
}  // namespace
TilingStageError BuildTilingProgramIRStage(const TilingProgramIRStageDefinition &definition,
                                           std::shared_ptr<const TilingProgramIRCompiledStage> *output) {
  if (output == nullptr) {
    return Error(Status::kInvalidArgument);
  }
  if (definition.expressions.empty() || definition.expressions.size() > kTilingExpressionMaxItems ||
      definition.input_types.size() > kTilingExpressionMaxItems - definition.expressions.size()) {
    return Error(Status::kInvalidProgram);
  }
  if (definition.conditional_returns.size() > kTilingExpressionMaxItems - definition.input_types.size()) {
    return Error(Status::kInvalidProgram);
  }
  for (size_t index = 0; index < definition.conditional_returns.size(); ++index) {
    const auto &branch = definition.conditional_returns[index];
    if (branch.condition_step >= definition.expressions.size() ||
        (index != 0 && branch.condition_step <= definition.conditional_returns[index - 1].condition_step) ||
        definition.expressions[branch.condition_step].result_type != TilingProgramIRValueType::kBoolean) {
      return Error(Status::kInvalidProgram, branch.condition_step);
    }
  }
  for (const auto &branch : definition.conditional_returns) {
    const auto result_type = definition.expressions.back().result_type;
    if (branch.value_step == UINT32_MAX) {
      if (branch.value.type != result_type || ValidateTilingProgramIRValue(branch.value) != ValueStatus::kOk) {
        return Error(Status::kInvalidProgram, branch.condition_step);
      }
    } else if (branch.value_step >= branch.condition_step ||
               definition.expressions[branch.value_step].result_type != result_type ||
               branch.value.type != TilingProgramIRValueType::kInt64 || branch.value.raw_bits != 0) {
      return Error(Status::kInvalidProgram, branch.condition_step);
    }
  }
  auto compiled = std::make_shared<TilingProgramIRCompiledStage>(TilingProgramIRCompiledStage::ConstructionKey{});
  compiled->input_count_ = definition.input_types.size();
  auto available = definition.input_types;
  size_t items = available.size() + definition.conditional_returns.size();
  for (size_t step = 0; step < definition.expressions.size(); ++step) {
    const auto &code = definition.expressions[step];
    for (size_t count : {code.input_types.size(), code.constants.size(), code.instructions.size()}) {
      if (count > kTilingExpressionMaxItems - items) {
        return Error(Status::kInvalidProgram, step);
      }
      items += count;
    }
    if (code.input_types != available) {
      return Error(Status::kInvalidProgram, step);
    }
    std::shared_ptr<const TilingProgramIRCompiledExpression> expression;
    const auto error = BuildTilingProgramIRExpression(code, &expression);
    if (error.status != Status::kOk) {
      return Error(error.status, step, error);
    }
    compiled->expression_scratch_ =
        std::max(compiled->expression_scratch_, GetTilingExpressionScratchElements(*expression));
    compiled->expressions_.push_back(std::move(expression));
    available.push_back(code.result_type);
  }
  compiled->conditional_returns_ = definition.conditional_returns;
  *output = std::move(compiled);
  return {};
}
size_t GetTilingStageScratchElements(const TilingProgramIRCompiledStage &stage) {
  return stage.expressions_.size() + stage.expression_scratch_;
}
TilingStageError EvaluateTilingProgramIRStage(const TilingProgramIRCompiledStage &stage, void *context,
                                              TilingExpressionReadSlot read_slot, TilingProgramIRValue *scratch,
                                              size_t scratch_elements, TilingProgramIRValue *output) {
  if (output == nullptr || (stage.input_count_ != 0 && read_slot == nullptr) || stage.expressions_.empty()) {
    return Error(Status::kInvalidArgument);
  }
  const size_t required = GetTilingStageScratchElements(stage);
  if (scratch_elements < required) {
    return Error(Status::kBufferTooSmall);
  }
  if (scratch == nullptr) {
    return Error(Status::kInvalidArgument);
  }
  const auto begin = reinterpret_cast<uintptr_t>(scratch);
  const auto result = reinterpret_cast<uintptr_t>(output);
  const size_t bytes = required * sizeof(TilingProgramIRValue);
  if (begin % alignof(TilingProgramIRValue) != 0 || result % alignof(TilingProgramIRValue) != 0 ||
      begin > UINTPTR_MAX - bytes || result > UINTPTR_MAX - sizeof(*output) ||
      (begin < result + sizeof(*output) && result < begin + bytes)) {
    return Error(Status::kInvalidArgument);
  }
  StageReader reader{context, read_slot, stage.input_count_, scratch, 0};
  auto *expression_scratch = scratch + stage.expressions_.size();
  size_t return_index = 0;
  for (size_t step = 0; step < stage.expressions_.size(); ++step) {
    const auto error = EvaluateTilingProgramIRExpression(*stage.expressions_[step], &reader, ReadStageSlot,
                                                         expression_scratch, stage.expression_scratch_, scratch + step);
    if (error.status != Status::kOk) {
      return Error(error.status, step, error);
    }
    ++reader.initialized;
    if (return_index < stage.conditional_returns_.size() &&
        stage.conditional_returns_[return_index].condition_step == step) {
      const auto &branch = stage.conditional_returns_[return_index++];
      if (scratch[step].raw_bits != 0) {
        *output = branch.value_step == UINT32_MAX ? branch.value : scratch[branch.value_step];
        return {};
      }
    }
  }
  *output = scratch[stage.expressions_.size() - 1];
  return {};
}
}  // namespace att
