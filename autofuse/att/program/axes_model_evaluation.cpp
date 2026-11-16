/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_model_evaluation.h"
namespace att {
TilingProgramIRValueType GetAxesModelInputType(AxesModelInputKind kind) {
  using Kind = AxesModelInputKind;
  using Type = TilingProgramIRValueType;
  switch (kind) {
    case Kind::kInputShape:
    case Kind::kPureMcVariable:
    case Kind::kLocalBufferVariable:
    case Kind::kCacheLocalValue:
      return Type::kInt64;
    case Kind::kUbSize:
      return Type::kUint32;
    case Kind::kUbThreshold:
      return Type::kFloat64;
    default:
      return static_cast<Type>(0);
  }
}
TilingStageError BuildAxesModelEvaluation(const AxesModelEvaluationDefinition &definition,
                                          std::shared_ptr<const AxesModelEvaluation> *output) {
  using Type = TilingProgramIRValueType;
  using Function = AxesModelFunction;
  const TilingStageError invalid{TilingExpressionStatus::kInvalidProgram, UINT32_MAX, {}};
  if (output == nullptr) {
    return {TilingExpressionStatus::kInvalidArgument, UINT32_MAX, {}};
  }
  const bool core = definition.function == Function::kUsedCoreNum || definition.function == Function::kRealUsedCoreNum;
  const bool threshold = definition.function == Function::kUbThreshold;
  const bool cache = definition.function == Function::kUbCacheLine;
  const bool perf = definition.function == Function::kPerf;
  if ((!core && !threshold && !cache && !perf) || definition.input_count > kTilingExpressionMaxItems ||
      definition.pure_mc_count > kTilingExpressionMaxItems ||
      definition.local_buffer_count > kTilingExpressionMaxItems ||
      definition.inputs.size() > kTilingExpressionMaxItems ||
      definition.inputs.size() != definition.stage.input_types.size()) {
    return invalid;
  }
  std::shared_ptr<const TilingProgramIRCompiledStage> stage;
  if (!definition.writes_output) {
    if (!core || !definition.inputs.empty() || !definition.stage.expressions.empty() ||
        !definition.stage.conditional_returns.empty()) {
      return invalid;
    }
  } else {
    if (definition.stage.expressions.empty() ||
        definition.stage.expressions.back().result_type != ((definition.function == Function::kUsedCoreNum || perf)
                                                                ? Type::kFloat64
                                                            : core ? Type::kInt64
                                                                   : Type::kBoolean)) {
      return invalid;
    }
    for (size_t i = 0; i < definition.inputs.size(); ++i) {
      const auto &source = definition.inputs[i];
      if (definition.stage.input_types[i] != GetAxesModelInputType(source.kind)) {
        return invalid;
      }
      switch (source.kind) {
        case AxesModelInputKind::kInputShape:
          if (source.index >= definition.input_count) {
            return invalid;
          }
          break;
        case AxesModelInputKind::kPureMcVariable:
          if ((!core && !perf) || source.index >= definition.pure_mc_count) {
            return invalid;
          }
          break;
        case AxesModelInputKind::kLocalBufferVariable:
          if (cache || source.index >= definition.local_buffer_count) {
            return invalid;
          }
          break;
        case AxesModelInputKind::kCacheLocalValue:
          if (!cache || source.index >= definition.local_buffer_count) {
            return invalid;
          }
          break;
        case AxesModelInputKind::kUbSize:
        case AxesModelInputKind::kUbThreshold:
          if (!threshold || source.index != 0) {
            return invalid;
          }
          break;
        default:
          return invalid;
      }
    }
    const auto error = BuildTilingProgramIRStage(definition.stage, &stage);
    if (error.status != TilingExpressionStatus::kOk) {
      return error;
    }
  }
  auto evaluation = std::make_shared<AxesModelEvaluation>(AxesModelEvaluation::ConstructionKey{});
  evaluation->function_ = definition.function;
  evaluation->input_count_ = definition.input_count;
  evaluation->pure_mc_count_ = definition.pure_mc_count;
  evaluation->local_buffer_count_ = definition.local_buffer_count;
  evaluation->writes_output_ = definition.writes_output;
  evaluation->inputs_ = definition.inputs;
  evaluation->stage_ = std::move(stage);
  *output = std::move(evaluation);
  return {};
}
}  // namespace att
