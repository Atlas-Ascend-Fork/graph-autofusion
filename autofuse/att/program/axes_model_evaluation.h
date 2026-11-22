/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_MODEL_EVALUATION_H_
#define ATT_PROGRAM_AXES_MODEL_EVALUATION_H_
#include "axes_program_evaluation.h"
namespace att {
enum class AxesModelFunction { kUsedCoreNum, kRealUsedCoreNum, kUbThreshold, kUbCacheLine, kPerf };
enum class AxesModelInputKind {
  kInputShape,
  kPureMcVariable,
  kLocalBufferVariable,
  kUbSize,
  kUbThreshold,
  kCacheLocalValue
};
TilingProgramIRValueType GetAxesModelInputType(AxesModelInputKind kind);
struct AxesModelInput {
  AxesModelInputKind kind = AxesModelInputKind::kInputShape;
  uint32_t index = 0;
};
struct AxesModelEvaluationDefinition {
  AxesModelFunction function = AxesModelFunction::kUsedCoreNum;
  uint32_t input_count = 0, pure_mc_count = 0, local_buffer_count = 0;
  bool writes_output = true;
  std::vector<AxesModelInput> inputs;
  TilingProgramIRStageDefinition stage;
};
class AxesModelEvaluation;
TilingStageError BuildAxesModelEvaluation(const AxesModelEvaluationDefinition &definition,
                                          std::shared_ptr<const AxesModelEvaluation> *output);
// Internal immutable model-method definition. Runtime arrays and scratch belong to the current solve.
class AxesModelEvaluation final {
 public:
  AxesModelFunction Function() const {
    return function_;
  }
  uint32_t InputCount() const {
    return input_count_;
  }
  uint32_t PureMcCount() const {
    return pure_mc_count_;
  }
  uint32_t LocalBufferCount() const {
    return local_buffer_count_;
  }
  bool WritesOutput() const {
    return writes_output_;
  }
  const std::vector<AxesModelInput> &Inputs() const {
    return inputs_;
  }
  const TilingProgramIRCompiledStage *Stage() const {
    return stage_.get();
  }

 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit AxesModelEvaluation(ConstructionKey) {}

 private:
  AxesModelFunction function_ = AxesModelFunction::kUsedCoreNum;
  uint32_t input_count_ = 0, pure_mc_count_ = 0, local_buffer_count_ = 0;
  bool writes_output_ = true;
  std::vector<AxesModelInput> inputs_;
  std::shared_ptr<const TilingProgramIRCompiledStage> stage_;
  friend TilingStageError BuildAxesModelEvaluation(const AxesModelEvaluationDefinition &,
                                                   std::shared_ptr<const AxesModelEvaluation> *);
};
struct AxesCoreNumEvaluations {
  std::shared_ptr<const AxesModelEvaluation> used;
  std::shared_ptr<const AxesModelEvaluation> real;
};
namespace axes_model_detail {
template <class Input>
struct Reader {
  const Input &input;
  const AxesModelEvaluation &evaluation;
  uint32_t cache_index;
  static TilingProgramValueStatus Read(void *context, uint32_t slot, TilingProgramIRValue *output) {
    const auto &reader = *static_cast<Reader *>(context);
    if (slot >= reader.evaluation.Inputs().size()) {
      return TilingProgramValueStatus::kInvalidArgument;
    }
    const auto &source = reader.evaluation.Inputs()[slot];
    const auto &input = reader.input;
    using Kind = AxesModelInputKind;
    using Type = TilingProgramIRValueType;
    int64_t value;
    switch (source.kind) {
      case Kind::kInputShape:
        if (input.input_vars == nullptr || input.input_vars[source.index] == nullptr) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = input.input_vars[source.index]->value;
        break;
      case Kind::kPureMcVariable:
        if (input.pure_mc_vars == nullptr || input.pure_mc_vars[source.index] == nullptr) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = input.pure_mc_vars[source.index]->value;
        break;
      case Kind::kLocalBufferVariable:
        if (input.local_buffer_vars == nullptr || input.local_buffer_vars[source.index] == nullptr) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = input.local_buffer_vars[source.index]->value;
        break;
      case Kind::kCacheLocalValue:
        if (reader.cache_index >= reader.evaluation.LocalBufferCount()) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = UINT16_MAX;
        if (source.index == reader.cache_index) {
          if (input.local_buffer_vars == nullptr || input.local_buffer_vars[reader.cache_index] == nullptr) {
            return TilingProgramValueStatus::kInvalidArgument;
          }
          value = input.local_buffer_vars[reader.cache_index]->value;
        }
        break;
      case Kind::kUbSize:
        *output = {Type::kUint32, input.ub_size};
        return TilingProgramValueStatus::kOk;
      case Kind::kUbThreshold:
        output->type = Type::kFloat64;
        output->raw_bits = CopyTilingScalarBits<uint64_t>(input.ub_threshold);
        return TilingProgramValueStatus::kOk;
      default:
        return TilingProgramValueStatus::kInvalidArgument;
    }
    *output = {Type::kInt64, static_cast<uint64_t>(value)};
    return TilingProgramValueStatus::kOk;
  }
};
}  // namespace axes_model_detail
// Shared checked model-method entry. The caller supplies the exact expected role.
// No-output success preserves the caller's value; errors also preserve it and record the first failure.
template <class Context, class Output>
auto EvaluateAxesModel(Context &context, const AxesModelEvaluation &evaluation, AxesModelFunction expected,
                       uint32_t expression_id, uint32_t cache_index, Output &output) -> decltype(context.first_error) {
  static_assert(
      std::is_same<Output, double>::value || std::is_same<Output, int64_t>::value || std::is_same<Output, bool>::value,
      "Model output must retain its native type");
  using Status = decltype(context.first_error);
  using Type = TilingProgramIRValueType;
  if (context.first_error != Status::kSuccess) {
    return context.first_error;
  }
  const auto fail = [&](Status status, uint32_t step = UINT32_MAX, uint32_t instruction = UINT32_MAX) {
    context.RecordEvaluationError(status, expression_id, step, instruction);
    if (expected == AxesModelFunction::kUbCacheLine) {
      context.error_cache_index = cache_index;
    }
    return status;
  };
  const Type result_type = std::is_same<Output, double>::value    ? Type::kFloat64
                           : std::is_same<Output, int64_t>::value ? Type::kInt64
                                                                  : Type::kBoolean;
  const Type expected_type = (expected == AxesModelFunction::kUsedCoreNum || expected == AxesModelFunction::kPerf)
                                 ? Type::kFloat64
                             : expected == AxesModelFunction::kRealUsedCoreNum ? Type::kInt64
                                                                               : Type::kBoolean;
  if (evaluation.Function() != expected || result_type != expected_type) {
    return fail(Status::kTypeMismatch);
  }
  const auto *input = context.variables;
  if (input == nullptr || input->input_vars_size != evaluation.InputCount() ||
      input->pure_mc_vars_size != evaluation.PureMcCount() ||
      input->local_buffer_vars_size != evaluation.LocalBufferCount()) {
    return fail(Status::kInvalidBinding);
  }
  if (!evaluation.WritesOutput()) {
    return Status::kSuccess;
  }
  if (context.evaluation_state == nullptr) {
    return fail(Status::kInvalidBinding);
  }
  const auto &scratch = *static_cast<const AxesProgramScratch *>(context.evaluation_state);
  using Input = typename std::remove_reference<decltype(*input)>::type;
  axes_model_detail::Reader<Input> reader{*input, evaluation, cache_index};
  TilingProgramIRValue result;
  const auto error = EvaluateTilingProgramIRStage(*evaluation.Stage(), &reader, reader.Read, scratch.values,
                                                  scratch.elements, &result);
  if (error.status != TilingExpressionStatus::kOk) {
    return fail(
        error.status == TilingExpressionStatus::kEvaluationError ? Status::kArithmeticError : Status::kInvalidBinding,
        error.step, error.expression_error.instruction);
  }
  if (result.type != result_type) {
    return fail(Status::kTypeMismatch);
  }
  if (expected == AxesModelFunction::kRealUsedCoreNum) {
    const auto real = CopyTilingScalarBits<int64_t>(result.raw_bits);
    if (real < 1 || real > INT32_MAX) {
      return fail(Status::kArithmeticError);
    }
  }
  if constexpr (std::is_same<Output, bool>::value) {
    output = result.raw_bits != 0;
  } else {
    output = CopyTilingScalarBits<Output>(result.raw_bits);
  }
  return Status::kSuccess;
}
template <class Context, class Output>
auto EvaluateAxesCoreNum(Context &context, const AxesModelEvaluation &evaluation, uint32_t expression_id,
                         Output &output) -> decltype(context.first_error) {
  static_assert(std::is_same<Output, double>::value || std::is_same<Output, int64_t>::value,
                "Core number output must retain its native type");
  const auto expected =
      std::is_same<Output, double>::value ? AxesModelFunction::kUsedCoreNum : AxesModelFunction::kRealUsedCoreNum;
  return EvaluateAxesModel(context, evaluation, expected, expression_id, 0, output);
}
template <class Context>
auto EvaluateAxesPerf(Context &context, const AxesModelEvaluation &evaluation, uint32_t expression_id,
                      double &output) -> decltype(context.first_error) {
  return EvaluateAxesModel(context, evaluation, AxesModelFunction::kPerf, expression_id, 0, output);
}
template <class Context>
auto EvaluateAxesUbThreshold(Context &context, const AxesModelEvaluation &evaluation, uint32_t expression_id,
                             bool &output) -> decltype(context.first_error) {
  return EvaluateAxesModel(context, evaluation, AxesModelFunction::kUbThreshold, expression_id, 0, output);
}
template <class Context>
auto EvaluateAxesUbCacheLine(Context &context, const AxesModelEvaluation &evaluation, uint32_t expression_id,
                             uint32_t index, bool &output) -> decltype(context.first_error) {
  return EvaluateAxesModel(context, evaluation, AxesModelFunction::kUbCacheLine, expression_id, index, output);
}
}  // namespace att
#endif  // ATT_PROGRAM_AXES_MODEL_EVALUATION_H_
