/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>
#include <cstring>
#include "program/axes_program_evaluation.h"
#define OP_LOGD(...) ((void)0)
#define OP_LOGI(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#define OP_NAME "EvaluationContract"
#include "axes_eval_core.h"

namespace {
using Type = att::TilingProgramIRValueType;
using Value = att::TilingProgramIRValue;
using Kind = att::TilingProgramIRInstructionKind;
using InputKind = att::AxesProgramInputKind;
using Definition = att::AxesProgramEvaluationDefinition;
using Evaluation = att::AxesProgramEvaluation;
using Role = att::AxesProgramEvaluationRole;
using Status = att::TilingExpressionStatus;
#define CHECK(condition)                               \
  do {                                                 \
    if (!(condition)) {                                \
      std::cerr << "failed line " << __LINE__ << "\n"; \
      return false;                                    \
    }                                                  \
  } while (0)
att::TilingProgramIRInstruction Symbol(uint32_t slot) {
  att::TilingProgramIRInstruction instruction;
  instruction.kind = Kind::kSymbol;
  instruction.symbol_id = slot;
  return instruction;
}
att::TilingProgramIRInstruction Op(att::TilingProgramIROpcode opcode) {
  att::TilingProgramIRInstruction instruction;
  instruction.kind = Kind::kOperation;
  instruction.opcode = opcode;
  instruction.arg_count = 2;
  return instruction;
}
Definition ConstraintDefinition() {
  Definition definition;
  definition.related_tiling_count = 1;
  definition.related_input_count = 1;
  definition.inputs = {{InputKind::kRelatedTilingVariable, 0}, {InputKind::kResourceValue, 0}};
  definition.stage.input_types = {Type::kInt64, Type::kInt64};
  att::TilingProgramIRExpressionCode code;
  code.input_types = definition.stage.input_types;
  code.constants = {{Type::kInt64, 4}};
  code.instructions = {
      Symbol(0), {}, Op(att::TilingProgramIROpcode::kMul), Symbol(1), Op(att::TilingProgramIROpcode::kSub)};
  definition.stage.expressions = {code};
  return definition;
}
Definition UpperDefinition() {
  Definition definition;
  definition.role = Role::kUpperBound;
  definition.parent_count = 1;
  definition.inputs = {{InputKind::kParentVariable, 0}};
  definition.stage.input_types = {Type::kInt64};
  att::TilingProgramIRExpressionCode code;
  code.input_types = definition.stage.input_types;
  code.instructions = {Symbol(0)};
  definition.stage.expressions = {code};
  return definition;
}
struct Programs {
  std::shared_ptr<const Evaluation> constraint, upper;
  bool Build() {
    CHECK(att::BuildAxesProgramEvaluation(ConstraintDefinition(), &constraint).status == Status::kOk);
    CHECK(att::BuildAxesProgramEvaluation(UpperDefinition(), &upper).status == Status::kOk);
    return true;
  }
};
int64_t NativeConstraint(TilingVariable **tiles, Variable **, int64_t limit) {
  return tiles[0]->value * 4 - limit;
}
int64_t NativeUpper(Variable **parents) {
  return parents[0]->value;
}
struct Fixture {
  Variable size;
  Variable *sizes[1] = {&size};
  TilingVariable tile;
  TilingVariable *tiles[1] = {&tile};
  Constraint constraint;
  Constraint *constraints[1] = {&constraint};
  uint32_t order[1] = {0};
  AxesReorderSolverInput input;
  std::vector<Value> values;
  att::AxesProgramScratch scratch;
  explicit Fixture(const Programs &programs, bool program, int64_t shape = 32) {
    size.value = shape;
    tile.upper_bound = NativeUpper;
    tile.upper_bound_vars = sizes;
    tile.upper_bound_vars_size = 1;
    tile.mc_related = true;
    tile.rel_cons = constraints;
    tile.rel_cons_size = 1;
    constraint.rel_tiling_vars = tiles;
    constraint.rel_tiling_vars_size = 1;
    constraint.rel_in_shapes = sizes;
    constraint.rel_in_shapes_size = 1;
    constraint.rel_hw_spec = 64;
    constraint.type = ConstraintType::LOCAL_BUFFER;
    constraint.eval = NativeConstraint;
    if (program) {
      tile.upper_bound_entry = att::AxesProgramUpperBoundEntry<EvalContext, EvalBinding, TypedValue>;
      tile.upper_bound_binding.definition = programs.upper.get();
      tile.upper_bound_binding.expression_id = 11;
      constraint.eval_entry = att::AxesProgramConstraintEntry<EvalContext, EvalBinding, TypedValue>;
      constraint.eval_binding.definition = programs.constraint.get();
      constraint.eval_binding.expression_id = 23;
    }
    input.input_vars = sizes;
    input.input_vars_size = 1;
    input.core_num = 4;
    input.ub_size = 64;
    input.tiling_vars = tiles;
    input.tiling_vars_size = 1;
    input.local_buffer_vars = tiles;
    input.local_buffer_vars_size = 1;
    input.ordered_local_buffer_vars = tiles;
    input.ordered_to_canonical = order;
    input.all_cons = constraints;
    input.all_cons_size = 1;
    values.resize(std::max(att::GetTilingStageScratchElements(programs.constraint->Stage()),
                           att::GetTilingStageScratchElements(programs.upper->Stage())));
    scratch = {values.data(), values.size()};
  }
};
template <class Base>
class Solver : public Base {
 public:
  explicit Solver(Fixture &fixture) : Base(fixture.input) {
    this->eval_context_.evaluation_state = &fixture.scratch;
  }
  uint32_t CoreNum() const {
    return this->input_.core_num;
  }

 protected:
  bool CalUsedCoreNum(double &value) override {
    value = static_cast<double>(this->input_.input_vars[0]->value) / this->input_.tiling_vars[0]->value;
    return true;
  }
  bool CalRealUsedCoreNum(int64_t &value) override {
    value = CeilDiv(this->input_.input_vars[0]->value, this->input_.tiling_vars[0]->value);
    return true;
  }
  bool SatisfyThresholdUBSize() override {
    return this->input_.tiling_vars[0]->value * 4 >= this->input_.ub_threshold * this->input_.ub_size;
  }
  double GetPerf() override {
    return CeilDiv(this->input_.input_vars[0]->value, this->input_.tiling_vars[0]->value) * 10.0 +
           this->input_.tiling_vars[0]->value / 100.0;
  }
  bool SatisfyUBSizeCacheLine(uint32_t) override {
    return true;
  }
};
bool InvalidDefinitionsAndOwnership() {
  auto definition = ConstraintDefinition();
  std::shared_ptr<const Evaluation> output;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kOk);
  const auto original = output;
  definition.inputs[0].index = 1;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  CHECK(output == original && output->Inputs()[0].index == 0);
  definition = ConstraintDefinition();
  definition.inputs[1].index = 1;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  definition = ConstraintDefinition();
  definition.inputs[0].kind = InputKind::kParentVariable;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  definition = ConstraintDefinition();
  definition.stage.input_types[0] = Type::kFloat64;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  definition = ConstraintDefinition();
  definition.stage.expressions[0].result_type = Type::kUint64;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  definition = UpperDefinition();
  definition.related_input_count = 1;
  CHECK(att::BuildAxesProgramEvaluation(definition, &output).status == Status::kInvalidProgram);
  CHECK(att::BuildAxesProgramEvaluation({}, &output).status == Status::kInvalidProgram);
  CHECK(att::BuildAxesProgramEvaluation(UpperDefinition(), nullptr).status == Status::kInvalidArgument);
  CHECK(output == original);
  return true;
}
bool BindingErrorsAndFirstError(const Programs &programs) {
  Fixture fixture(programs, true);
  EvalContext context;
  context.variables = &fixture.input;
  context.evaluation_state = &fixture.scratch;
  auto binding = fixture.constraint.eval_binding;
  binding.native_data = &fixture.constraint;
  binding.resource_value = 5;  // Must override the stored resource value 64.
  fixture.tile.value = 3;
  TypedValue output;
  const EvalEntry entry = att::AxesProgramConstraintEntry<EvalContext, EvalBinding, TypedValue>;
  CHECK(entry(context, binding, output) == EvalStatus::kSuccess && output.signed_value == 7);
  fixture.scratch.elements = 0;
  output.signed_value = 99;
  CHECK(entry(context, binding, output) == EvalStatus::kInvalidBinding && output.signed_value == 99);
  CHECK(context.error_expression_id == 23 && context.error_stage_step == UINT32_MAX);
  fixture.scratch.elements = fixture.values.size();
  context.ResetError();
  fixture.tiles[0] = nullptr;
  CHECK(entry(context, binding, output) == EvalStatus::kInvalidBinding && output.signed_value == 99);
  CHECK(context.error_stage_step == 0 && context.error_instruction == 0);
  fixture.tiles[0] = &fixture.tile;
  fixture.tile.value = INT64_MAX;
  CHECK(entry(context, binding, output) == EvalStatus::kInvalidBinding);  // Must preserve the earlier input error.
  CHECK(context.error_instruction == 0);
  context.ResetError();
  CHECK(entry(context, binding, output) == EvalStatus::kArithmeticError && output.signed_value == 99);
  CHECK(context.error_stage_step == 0 && context.error_instruction == 2);
  context.ResetError();
  binding.definition = programs.upper.get();
  CHECK(entry(context, binding, output) == EvalStatus::kInvalidBinding);  // Reject role before casting native_data.
  context.ResetError();
  binding = fixture.tile.upper_bound_binding;
  binding.native_data = &fixture.tile;
  fixture.size.value = -1;
  const EvalEntry upper = att::AxesProgramUpperBoundEntry<EvalContext, EvalBinding, TypedValue>;
  CHECK(upper(context, binding, output) == EvalStatus::kSuccess && output.signed_value == -1);
  fixture.tile.upper_bound_vars_size = 0;
  CHECK(upper(context, binding, output) == EvalStatus::kInvalidBinding);
  context.ResetError();
  fixture.tile.upper_bound_vars_size = 1;
  binding.expected_type = EvalValueType::kFloat64;
  CHECK(upper(context, binding, output) == EvalStatus::kTypeMismatch);
  context.ResetError();
  CHECK(context.error_expression_id == UINT32_MAX && context.error_stage_step == UINT32_MAX &&
        context.error_instruction == UINT32_MAX);
  return true;
}
bool CompareSearchAndPgo(const Programs &programs) {
  for (int64_t shape : {0, 1, 16, 32, 65}) {
    for (bool tradeoff : {false, true}) {
      for (bool block_tune : {false, true}) {
        for (bool tune : {false, true}) {
          Fixture native(programs, false, shape), program(programs, true, shape);
          Solver<AxesReorderSolver> left(native), right(program);
          CHECK(left.Run(tradeoff, block_tune, tune, AF_AXES_EQUAL_ORDER) ==
                right.Run(tradeoff, block_tune, tune, AF_AXES_EQUAL_ORDER));
          CHECK(left.IsEmptyTensor() == right.IsEmptyTensor() && native.tile.value == program.tile.value &&
                left.CoreNum() == right.CoreNum());
          CHECK(right.GetEvaluationContext().first_error == EvalStatus::kSuccess);
        }
      }
    }
  }
  Fixture native(programs, false, 16), program(programs, true, 16);
  Solver<AxesReorderPgoSolver> left(native), right(program);
  CHECK(left.PgoSolverGenerateAllTilingData() == right.PgoSolverGenerateAllTilingData());
  CHECK(left.GetTilingDataList() == right.GetTilingDataList() && !right.GetTilingDataList().empty());
  program.sizes[0] = nullptr;
  CHECK(!right.PgoSolverGenerateAllTilingData() && right.GetTilingDataList().empty());
  CHECK(right.GetEvaluationContext().first_error == EvalStatus::kInvalidBinding);
  program.sizes[0] = &program.size;
  program.tile.value = -1;
  CHECK(right.PgoSolverGenerateAllTilingData());
  CHECK(right.GetEvaluationContext().first_error == EvalStatus::kSuccess &&
        right.GetEvaluationContext().error_stage_step == UINT32_MAX);
  return true;
}
bool ConcurrentContexts(const Programs &programs) {
  bool left = false, right = false;
  std::thread a([&] { left = CompareSearchAndPgo(programs); });
  std::thread b([&] { right = CompareSearchAndPgo(programs); });
  a.join();
  b.join();
  return left && right;
}
}  // namespace
int main() {
  Programs programs;
  if (!programs.Build() || !InvalidDefinitionsAndOwnership() || !BindingErrorsAndFirstError(programs) ||
      !CompareSearchAndPgo(programs) || !ConcurrentContexts(programs)) {
    return 1;
  }
  std::cout << "program_bindings=ok search_and_pgo=ok concurrent_contexts=ok\n";
  return 0;
}
