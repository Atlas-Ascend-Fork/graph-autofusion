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
#define OP_LOGD(...) ((void)0)
#define OP_LOGI(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#define OP_NAME "EvaluationContract"
#define private public
#include "axes_eval_core.h"
#undef private

namespace {
struct Probe {
  size_t calls = 0;
  size_t fail_at = 0;
  int64_t value_at_failure = 0;
  int64_t bound = 16;
  int64_t limit = 8;
  bool fail(EvalContext &context) {
    ++calls;
    if (calls != fail_at) {
      return false;
    }
    value_at_failure = context.variables->local_buffer_vars[0]->value;
    context.RecordError(EvalStatus::kArithmeticError, 17);
    return true;
  }
};

EvalStatus Bound(EvalContext &context, const EvalBinding &binding, TypedValue &output) {
  auto &probe = *static_cast<Probe *>(context.sdk_context);
  if (probe.fail(context)) {
    return EvalStatus::kSdkError;  // Must not overwrite the earlier arithmetic error.
  }
  output.type = EvalValueType::kInt64;
  output.signed_value = *static_cast<const int64_t *>(binding.definition);
  return EvalStatus::kSuccess;
}

EvalStatus Limit(EvalContext &context, const EvalBinding &binding, TypedValue &output) {
  auto &probe = *static_cast<Probe *>(context.sdk_context);
  if (probe.fail(context)) {
    return EvalStatus::kSdkError;
  }
  output.type = EvalValueType::kInt64;
  output.signed_value =
      context.variables->local_buffer_vars[0]->value - *static_cast<const int64_t *>(binding.definition);
  return EvalStatus::kSuccess;
}

template <typename Base>
class ProbeSolverT : public Base {
 public:
  ProbeSolverT(const AxesReorderSolverInput &input, Probe &probe) : Base(input) {
    this->eval_context_.sdk_context = &probe;
  }
  using Base::ApplyPromptAlign;
  using Base::SatisfyCons;

 protected:
  bool CalUsedCoreNum(double &value) override {
    value = 1.0;
    return !static_cast<Probe *>(this->eval_context_.sdk_context)->fail(this->eval_context_);
  }
  bool CalRealUsedCoreNum(int64_t &value) override {
    value = 1;
    return !static_cast<Probe *>(this->eval_context_.sdk_context)->fail(this->eval_context_);
  }
  bool SatisfyThresholdUBSize() override {
    return !static_cast<Probe *>(this->eval_context_.sdk_context)->fail(this->eval_context_);
  }
  bool SatisfyUBSizeCacheLine(uint32_t) override {
    return !static_cast<Probe *>(this->eval_context_.sdk_context)->fail(this->eval_context_);
  }
  double GetPerf() override {
    (void)static_cast<Probe *>(this->eval_context_.sdk_context)->fail(this->eval_context_);
    return 1.0;
  }
};

using ProbeSolver = ProbeSolverT<AxesReorderSolver>;

struct Fixture {
  Probe probe;
  TilingVariable var;
  TilingVariable *vars[1] = {&var};
  Constraint constraint;
  Constraint *constraints[1] = {&constraint};
  uint32_t order[1] = {0};
  AxesReorderSolverInput input;
  explicit Fixture(bool prompt_alignment = false) {
    if (prompt_alignment) {
      probe.bound = 64;
      probe.limit = 31;
      var.prompt_align = 16;
    }
    var.value = 1;
    var.upper_bound_entry = Bound;
    var.upper_bound_binding.definition = &probe.bound;
    var.upper_bound_binding.expression_id = 5;
    var.rel_cons = constraints;
    var.rel_cons_size = 1;
    constraint.type = ConstraintType::LOCAL_BUFFER;
    constraint.eval_entry = Limit;
    constraint.eval_binding.definition = &probe.limit;
    constraint.eval_binding.expression_id = 9;
    input.core_num = 1;
    input.ub_size = 16;
    input.tiling_vars = vars;
    input.tiling_vars_size = 1;
    input.local_buffer_vars = vars;
    input.local_buffer_vars_size = 1;
    input.ordered_local_buffer_vars = vars;
    input.ordered_to_canonical = order;
    input.all_cons = constraints;
    input.all_cons_size = 1;
  }
};

bool InjectAtEveryEvaluation(bool trade_off, bool tune, bool block_tune, bool prompt_alignment = false) {
  Fixture healthy(prompt_alignment);
  ProbeSolver original(healthy.input, healthy.probe);
  if (!original.Run(trade_off, block_tune, tune, AF_AXES_EQUAL_ORDER)) {
    std::cerr << "Healthy probe failed\n";
    return false;
  }
  const auto count = healthy.probe.calls;
  if (count == 0 || healthy.var.value > healthy.probe.limit) {
    return false;
  }
  for (size_t failure = 1; failure <= count; ++failure) {
    Fixture fixture(prompt_alignment);
    fixture.probe.fail_at = failure;
    ProbeSolver solver(fixture.input, fixture.probe);
    if (solver.Run(trade_off, block_tune, tune, AF_AXES_EQUAL_ORDER) || solver.IsEmptyTensor() ||
        solver.GetEvaluationContext().first_error != EvalStatus::kArithmeticError ||
        solver.GetEvaluationContext().error_expression_id != 17 || fixture.probe.calls != failure ||
        fixture.var.value != fixture.probe.value_at_failure) {
      std::cerr << "Failed termination contract at evaluation " << failure << "/" << count << "\n";
      return false;
    }
    // A later invocation has a new error state; the caller reinitializes its own input values.
    fixture.probe.calls = 0;
    fixture.probe.fail_at = 0;
    fixture.var.value = 1;
    if (!solver.Run(trade_off, block_tune, tune, AF_AXES_EQUAL_ORDER) ||
        solver.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
      return false;
    }
  }
  std::cout << "fault_points=" << count << "\n";
  return true;
}

bool CheckBindingsAndIsolation() {
  Fixture fixture;
  ProbeSolver solver(fixture.input, fixture.probe);
  fixture.var.value = 6;
  const int64_t smaller = 5;
  const int64_t larger = 7;
  fixture.constraint.eval_binding.definition = &smaller;
  if (solver.SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return false;
  }
  fixture.constraint.eval_binding.definition = &larger;
  if (!solver.SatisfyCons(ConstraintType::LOCAL_BUFFER) ||
      solver.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
    return false;
  }
  bool left = false, right = false;
  std::thread a([&] { left = InjectAtEveryEvaluation(false, true, false); });
  std::thread b([&] { right = InjectAtEveryEvaluation(true, true, true); });
  a.join();
  b.join();
  return left && right;
}

bool CheckPgoFailure() {
  Fixture healthy;
  ProbeSolverT<AxesReorderPgoSolver> original(healthy.input, healthy.probe);
  const std::vector<std::vector<uint32_t>> expected{{1}, {2}, {4}, {8}};
  if (!original.PgoSolverGenerateAllTilingData() || original.GetTilingDataList() != expected) {
    return false;
  }
  const auto count = healthy.probe.calls;
  healthy.probe.calls = 0;
  healthy.probe.fail_at = count;
  if (original.PgoSolverGenerateAllTilingData() || !original.GetTilingDataList().empty() ||
      original.GetEvaluationContext().first_error != EvalStatus::kArithmeticError) {
    return false;
  }
  healthy.probe.calls = 0;
  healthy.probe.fail_at = 0;
  if (!original.PgoSolverGenerateAllTilingData() || original.GetTilingDataList() != expected ||
      original.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
    return false;
  }
  for (size_t failure = 1; failure <= count; ++failure) {
    Fixture fixture;
    fixture.probe.fail_at = failure;
    ProbeSolverT<AxesReorderPgoSolver> solver(fixture.input, fixture.probe);
    if (solver.PgoSolverGenerateAllTilingData() || !solver.GetTilingDataList().empty() ||
        solver.GetEvaluationContext().first_error != EvalStatus::kArithmeticError ||
        solver.GetEvaluationContext().error_expression_id != 17 || fixture.probe.calls != failure ||
        fixture.var.value != fixture.probe.value_at_failure) {
      return false;
    }
  }
  return true;
}

bool CheckPromptAlignmentAdapters() {
  for (const bool native : {false, true}) {
    Fixture fixture(true);
    fixture.var.value = 31;
    if (native) {
      fixture.var.upper_bound_entry = NativeUpperBoundEntry;
      fixture.var.upper_bound = [](Variable **) -> int64_t { return 64; };
    }
    ProbeSolver solver(fixture.input, fixture.probe);
    solver.ApplyPromptAlign(&fixture.var);
    if (fixture.var.value != 16 || solver.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
      return false;
    }
  }
  Fixture optional;
  optional.var.upper_bound_entry = NativeUpperBoundEntry;
  optional.var.value = 31;
  optional.var.prompt_align = 16;
  ProbeSolver solver(optional.input, optional.probe);
  solver.ApplyPromptAlign(&optional.var);
  return optional.var.value == 31 && solver.GetEvaluationContext().first_error == EvalStatus::kSuccess;
}

bool CheckInvalidBindings() {
  Fixture missing;
  missing.var.upper_bound_entry = nullptr;
  ProbeSolver missing_solver(missing.input, missing.probe);
  if (missing_solver.Run(true, false, false) || missing.probe.calls != 0 ||
      missing_solver.GetEvaluationContext().first_error != EvalStatus::kInvalidBinding) {
    return false;
  }
  Fixture wrong_type;
  wrong_type.var.upper_bound_entry = [](EvalContext &, const EvalBinding &, TypedValue &output) {
    output.type = EvalValueType::kFloat64;
    output.double_value = 0.0;
    return EvalStatus::kSuccess;
  };
  ProbeSolver wrong_solver(wrong_type.input, wrong_type.probe);
  if (wrong_solver.Run(true, false, false) || wrong_solver.IsEmptyTensor() ||
      wrong_solver.GetEvaluationContext().first_error != EvalStatus::kTypeMismatch) {
    return false;
  }
  Fixture native;
  native.var.upper_bound_entry = NativeUpperBoundEntry;
  native.var.upper_bound = [](Variable **) -> int64_t { return 0; };
  ProbeSolver native_solver(native.input, native.probe);
  if (!native_solver.Run(true, false, false) || !native_solver.IsEmptyTensor() ||
      native_solver.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
    return false;
  }
  native.var.upper_bound_entry = nullptr;
  if (native_solver.Run(false, false, false) || native_solver.IsEmptyTensor() ||
      native_solver.GetEvaluationContext().first_error != EvalStatus::kInvalidBinding) {
    return false;
  }
  native.var.upper_bound_entry = NativeUpperBoundEntry;
  native.var.upper_bound = [](Variable **) -> int64_t { return 16; };
  return native_solver.Run(false, false, false) && !native_solver.IsEmptyTensor() && native.var.value == 8 &&
         native_solver.GetEvaluationContext().first_error == EvalStatus::kSuccess;
}
}  // namespace

int main() {
  for (const bool trade_off : {false, true}) {
    for (const bool tune : {false, true}) {
      for (const bool block_tune : {false, true}) {
        if (!InjectAtEveryEvaluation(trade_off, tune, block_tune)) {
          return 1;
        }
      }
    }
  }
  return InjectAtEveryEvaluation(false, false, false, true) && InjectAtEveryEvaluation(true, true, true, true) &&
                 CheckBindingsAndIsolation() && CheckInvalidBindings() && CheckPromptAlignmentAdapters() &&
                 CheckPgoFailure()
             ? 0
             : 2;
}
