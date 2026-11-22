/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_PROGRAM_EVALUATION_H_
#define ATT_PROGRAM_AXES_PROGRAM_EVALUATION_H_
#include <type_traits>
#include <utility>
#include "tiling_stage.h"
namespace att {
enum class AxesProgramEvaluationRole { kConstraint, kUpperBound };
enum class AxesProgramInputKind { kRelatedTilingVariable, kRelatedInputShape, kResourceValue, kParentVariable };
struct AxesProgramInput {
  AxesProgramInputKind kind = AxesProgramInputKind::kRelatedTilingVariable;
  uint32_t index = 0;
};
struct AxesProgramEvaluationDefinition {
  AxesProgramEvaluationRole role = AxesProgramEvaluationRole::kConstraint;
  uint32_t related_tiling_count = 0, related_input_count = 0, parent_count = 0;
  std::vector<AxesProgramInput> inputs;
  TilingProgramIRStageDefinition stage;
};
struct AxesProgramScratch {
  TilingProgramIRValue *values = nullptr;
  size_t elements = 0;
};
class AxesProgramEvaluation;
TilingStageError BuildAxesProgramEvaluation(const AxesProgramEvaluationDefinition &definition,
                                            std::shared_ptr<const AxesProgramEvaluation> *output);
// Immutable and internal to same-source AF components. No addresses from these objects are serialized.
class AxesProgramEvaluation final {
 public:
  AxesProgramEvaluationRole Role() const {
    return role_;
  }
  uint32_t RelatedTilingCount() const {
    return related_tiling_count_;
  }
  uint32_t RelatedInputCount() const {
    return related_input_count_;
  }
  uint32_t ParentCount() const {
    return parent_count_;
  }
  const std::vector<AxesProgramInput> &Inputs() const {
    return inputs_;
  }
  const TilingProgramIRCompiledStage &Stage() const {
    return *stage_;
  }

 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit AxesProgramEvaluation(ConstructionKey) {}

 private:
  AxesProgramEvaluationRole role_ = AxesProgramEvaluationRole::kConstraint;
  uint32_t related_tiling_count_ = 0, related_input_count_ = 0, parent_count_ = 0;
  std::vector<AxesProgramInput> inputs_;
  std::shared_ptr<const TilingProgramIRCompiledStage> stage_;
  friend TilingStageError BuildAxesProgramEvaluation(const AxesProgramEvaluationDefinition &,
                                                     std::shared_ptr<const AxesProgramEvaluation> *);
};
namespace axes_program_detail {
// Templates bind to either owning namespace of the one shared normal/equal-order core.
// They only read declared inputs and invoke the typed VM; no solver algorithm is reimplemented here.
template <class Context, class Binding>
struct Reader {
  using Constraint = typename std::remove_pointer<
      typename std::remove_reference<decltype(std::declval<Context>().variables->all_cons[0])>::type>::type;
  using TilingVariable = typename std::remove_pointer<
      typename std::remove_reference<decltype(std::declval<Context>().variables->tiling_vars[0])>::type>::type;
  const Binding &binding;
  const AxesProgramEvaluation &evaluation;
  bool Valid() const {
    if (binding.native_data == nullptr) {
      return false;
    }
    if (evaluation.Role() == AxesProgramEvaluationRole::kConstraint) {
      const auto &constraint = *static_cast<const Constraint *>(binding.native_data);
      return constraint.rel_tiling_vars_size == evaluation.RelatedTilingCount() &&
             constraint.rel_in_shapes_size == evaluation.RelatedInputCount() &&
             (constraint.rel_tiling_vars_size == 0 || constraint.rel_tiling_vars != nullptr) &&
             (constraint.rel_in_shapes_size == 0 || constraint.rel_in_shapes != nullptr);
    }
    const auto &variable = *static_cast<const TilingVariable *>(binding.native_data);
    return variable.upper_bound_vars_size == evaluation.ParentCount() &&
           (variable.upper_bound_vars_size == 0 || variable.upper_bound_vars != nullptr);
  }
  static TilingProgramValueStatus Read(void *context, uint32_t slot, TilingProgramIRValue *output) {
    const auto &reader = *static_cast<Reader *>(context);
    if (slot >= reader.evaluation.Inputs().size()) {
      return TilingProgramValueStatus::kInvalidArgument;
    }
    const auto &source = reader.evaluation.Inputs()[slot];
    int64_t value = 0;
    if (source.kind == AxesProgramInputKind::kResourceValue) {
      value = reader.binding.resource_value;
    } else if (source.kind == AxesProgramInputKind::kParentVariable) {
      const auto &variable = *static_cast<const TilingVariable *>(reader.binding.native_data);
      const auto *parent = variable.upper_bound_vars[source.index];
      if (parent == nullptr) {
        return TilingProgramValueStatus::kInvalidArgument;
      }
      value = parent->value;
    } else {
      const auto &constraint = *static_cast<const Constraint *>(reader.binding.native_data);
      if (source.kind == AxesProgramInputKind::kRelatedTilingVariable) {
        const auto *variable = constraint.rel_tiling_vars[source.index];
        if (variable == nullptr) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = variable->value;
      } else {
        const auto *variable = constraint.rel_in_shapes[source.index];
        if (variable == nullptr) {
          return TilingProgramValueStatus::kInvalidArgument;
        }
        value = variable->value;
      }
    }
    *output = {TilingProgramIRValueType::kInt64, static_cast<uint64_t>(value)};
    return TilingProgramValueStatus::kOk;
  }
};
}  // namespace axes_program_detail
// Bind through the shared core's EvalEntry. The caller owns the descriptor and one exclusive scratch state per solve.
// Constraint/upper-bound construction must match the descriptor role; serialized input never supplies these pointers.
template <AxesProgramEvaluationRole Role, class Context, class Binding, class Output>
auto AxesProgramEvaluationEntry(Context &context, const Binding &binding,
                                Output &output) -> decltype(context.first_error) {
  using Status = decltype(context.first_error);
  using ValueType = decltype(binding.expected_type);
  if (context.first_error != Status::kSuccess) {
    return context.first_error;
  }
  const auto fail = [&](Status status, uint32_t step = UINT32_MAX, uint32_t instruction = UINT32_MAX) {
    context.RecordEvaluationError(status, binding.expression_id, step, instruction);
    return status;
  };
  if (binding.expected_type != ValueType::kInt64) {
    return fail(Status::kTypeMismatch);
  }
  if (context.variables == nullptr || binding.definition == nullptr || context.evaluation_state == nullptr) {
    return fail(Status::kInvalidBinding);
  }
  const auto &evaluation = *static_cast<const AxesProgramEvaluation *>(binding.definition);
  if (evaluation.Role() != Role) {
    return fail(Status::kInvalidBinding);
  }
  const auto &scratch = *static_cast<const AxesProgramScratch *>(context.evaluation_state);
  axes_program_detail::Reader<Context, Binding> reader{binding, evaluation};
  if (!reader.Valid()) {
    return fail(Status::kInvalidBinding);
  }
  TilingProgramIRValue value;
  const auto error =
      EvaluateTilingProgramIRStage(evaluation.Stage(), &reader, reader.Read, scratch.values, scratch.elements, &value);
  if (error.status != TilingExpressionStatus::kOk) {
    return fail(
        error.status == TilingExpressionStatus::kEvaluationError ? Status::kArithmeticError : Status::kInvalidBinding,
        error.step, error.expression_error.instruction);
  }
  if (value.type != TilingProgramIRValueType::kInt64) {
    return fail(Status::kTypeMismatch);
  }
  const auto signed_value = CopyTilingScalarBits<int64_t>(value.raw_bits);
  output.signed_value = signed_value;
  output.type = ValueType::kInt64;
  return Status::kSuccess;
}
template <class Context, class Binding, class Output>
auto AxesProgramConstraintEntry(Context &context, const Binding &binding,
                                Output &output) -> decltype(context.first_error) {
  return AxesProgramEvaluationEntry<AxesProgramEvaluationRole::kConstraint>(context, binding, output);
}
template <class Context, class Binding, class Output>
auto AxesProgramUpperBoundEntry(Context &context, const Binding &binding,
                                Output &output) -> decltype(context.first_error) {
  return AxesProgramEvaluationEntry<AxesProgramEvaluationRole::kUpperBound>(context, binding, output);
}
}  // namespace att
#endif  // ATT_PROGRAM_AXES_PROGRAM_EVALUATION_H_
