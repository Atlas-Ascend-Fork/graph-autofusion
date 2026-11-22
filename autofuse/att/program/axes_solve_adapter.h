/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_SOLVE_ADAPTER_H_
#define ATT_PROGRAM_AXES_SOLVE_ADAPTER_H_
#include <algorithm>
#include "program/axes_solve_runtime.h"

namespace att {
// The caller provides solve-exclusive native input objects and topology.
template <class Solver, class Input, class Context, class Binding, class Output>
class AxesSolveAdapter : private Solver {
 public:
  explicit AxesSolveAdapter(const Input &input) : Solver(input) {}

  TilingExpressionStatus Prepare(const AxesEvaluationRecords &records, const AxesProgramScratch *provided = nullptr) {
    using Status = TilingExpressionStatus;
    using Function = AxesModelFunction;
    constexpr Function order[] = {Function::kUsedCoreNum, Function::kRealUsedCoreNum, Function::kUbThreshold,
                                  Function::kUbCacheLine, Function::kPerf};
    const auto &input = this->input_;
    if (ready_ || records.tiling_count != input.tiling_vars_size) {
      return Status::kInvalidArgument;
    }
    for (size_t i = 0; i < records.methods.size(); ++i) {
      const auto &method = records.methods[i];
      if (!method || method->Function() != order[i] || method->InputCount() != input.input_vars_size ||
          method->PureMcCount() != input.pure_mc_vars_size ||
          method->LocalBufferCount() != input.local_buffer_vars_size) {
        return Status::kInvalidProgram;
      }
    }
    size_t capacity;
    const auto requirements = GetAxesEvaluationScratchElements(records, &capacity);
    if (requirements != Status::kOk) {
      return requirements;
    }
    if (provided != nullptr && provided->elements < capacity) {
      return Status::kBufferTooSmall;
    }
    if (provided != nullptr && capacity != 0 && provided->values == nullptr) {
      return Status::kInvalidArgument;
    }
    std::vector<TilingProgramIRValue> scratch(provided == nullptr ? capacity : 0);
    const auto status = bindings_.Bind(records.bindings, this->input_);
    if (status != Status::kOk) {
      return status;
    }
    methods_ = records.methods;
    scratch_ = std::move(scratch);
    scratch_view_ = provided == nullptr ? AxesProgramScratch{scratch_.data(), scratch_.size()} : *provided;
    this->eval_context_.evaluation_state = &scratch_view_;
    ready_ = true;
    return Status::kOk;
  }

  bool Run(bool trade_off, bool tune_block, bool tune, bool equal_order) {
    return ready_ && Solver::Run(trade_off, tune_block, tune, equal_order);
  }
  bool GetSolvedBlockDim(uint32_t &block_dim) {
    int64_t used = 0;
    if (!CalRealUsedCoreNum(used) || used < 1 || used > INT32_MAX) {
      return false;
    }
    block_dim = static_cast<uint32_t>(used);
    return true;
  }

  using Solver::GetEvaluationContext;
  using Solver::IsEmptyTensor;

 protected:
  bool CalUsedCoreNum(double &value) override {
    return Success(EvaluateAxesCoreNum(this->eval_context_, Method(AxesModelFunction::kUsedCoreNum),
                                       MethodId(AxesModelFunction::kUsedCoreNum), value));
  }
  bool CalRealUsedCoreNum(int64_t &value) override {
    return Success(EvaluateAxesCoreNum(this->eval_context_, Method(AxesModelFunction::kRealUsedCoreNum),
                                       MethodId(AxesModelFunction::kRealUsedCoreNum), value));
  }
  bool SatisfyThresholdUBSize() override {
    bool value = false;
    return Success(EvaluateAxesUbThreshold(this->eval_context_, Method(AxesModelFunction::kUbThreshold),
                                           MethodId(AxesModelFunction::kUbThreshold), value)) &&
           value;
  }
  bool SatisfyUBSizeCacheLine(uint32_t index) override {
    bool value = false;
    return Success(EvaluateAxesUbCacheLine(this->eval_context_, Method(AxesModelFunction::kUbCacheLine),
                                           MethodId(AxesModelFunction::kUbCacheLine), index, value)) &&
           value;
  }
  double GetPerf() override {
    double value = 0;
    EvaluateAxesPerf(this->eval_context_, Method(AxesModelFunction::kPerf), MethodId(AxesModelFunction::kPerf), value);
    return value;
  }

 private:
  static constexpr uint32_t MethodSlot(AxesModelFunction function) {
    return static_cast<uint32_t>(function);
  }

  const AxesModelEvaluation &Method(AxesModelFunction function) const {
    return *methods_[MethodSlot(function)];
  }

  static constexpr uint32_t MethodId(AxesModelFunction function) {
    return UINT32_MAX - 1 - MethodSlot(function);
  }

  template <class Status>
  static bool Success(Status status) {
    return status == Status::kSuccess;
  }

  bool ready_ = false;
  AxesModelMethods methods_{};
  AxesSolveCallbackStorage<Input, Context, Binding, Output> bindings_;
  std::vector<TilingProgramIRValue> scratch_;
  AxesProgramScratch scratch_view_;
};
inline TilingExpressionStatus WriteAxesDerivedResult(const AxesDerivedWriteback &writeback,
                                                     const std::vector<int64_t> &shapes,
                                                     const std::vector<int64_t> &tiles, void *destination,
                                                     size_t capacity, uint32_t block_dim = 0, uint32_t ub_size = 0,
                                                     const AxesProgramScratch *solve_scratch = nullptr,
                                                     AxesSolveEvaluationFailure *failure = nullptr) {
  using Status = TilingExpressionStatus;
  constexpr auto type = TilingProgramIRValueType::kUint32;
  if (!writeback.layout || !writeback.layout->HasAllValues() ||
      writeback.layout->ValueCount() != tiles.size() + writeback.stages.size()) {
    return Status::kInvalidProgram;
  }
  if (shapes.size() != writeback.shape_count) {
    return Status::kInputError;
  }
  std::vector<TilingProgramIRValue> inputs;
  constexpr size_t kResourceInputCount = 2;  // blockDim and effective UB
  inputs.reserve(shapes.size() + tiles.size() + kResourceInputCount);
  for (const auto *group : {&shapes, &tiles}) {
    for (const auto value : *group) {
      if (value < 0 || static_cast<uint64_t>(value) > UINT32_MAX) {
        return Status::kInputError;
      }
      inputs.push_back({type, static_cast<uint32_t>(value)});
    }
  }
  std::vector<TilingProgramIRValue> values;
  values.reserve(tiles.size() + writeback.stages.size());
  values.insert(values.end(), inputs.begin() + shapes.size(), inputs.end());
  inputs.push_back({type, block_dim});
  inputs.push_back({type, ub_size});
  const auto read = [](void *context, uint32_t slot, TilingProgramIRValue *value) {
    const auto &bound = *static_cast<const std::vector<TilingProgramIRValue> *>(context);
    if (slot >= bound.size()) {
      return TilingProgramValueStatus::kInvalidArgument;
    }
    *value = bound[slot];
    return TilingProgramValueStatus::kOk;
  };
  size_t required_scratch = 0;
  for (const auto &stage : writeback.stages) {
    if (!stage) {
      return Status::kInvalidProgram;
    }
    required_scratch = std::max(required_scratch, GetTilingStageScratchElements(*stage));
  }
  std::vector<TilingProgramIRValue> fallback;
  auto *scratch = solve_scratch == nullptr ? nullptr : solve_scratch->values;
  if (required_scratch != 0 && (scratch == nullptr || solve_scratch->elements < required_scratch)) {
    fallback.resize(required_scratch);
    scratch = fallback.data();
  }
  for (size_t i = 0; i < writeback.stages.size(); ++i) {
    const auto &stage = writeback.stages[i];
    TilingProgramIRValue value{};
    const auto result = EvaluateTilingProgramIRStage(*stage, &inputs, read, scratch, required_scratch, &value);
    if (result.status != Status::kOk) {
      if (failure != nullptr && failure->status == 0 && result.status == Status::kEvaluationError) {
        const auto kind = result.expression_error.value_status == TilingProgramValueStatus::kArithmeticError
                              ? AxesSolveFailureStatus::kArithmeticError
                              : AxesSolveFailureStatus::kOther;
        *failure = {static_cast<uint32_t>(kind),         UINT32_MAX, result.step,
                    result.expression_error.instruction, true,       static_cast<uint32_t>(i)};
      }
      return result.status;
    }
    values.push_back(value);
  }
  return writeback.layout->Write(values.data(), values.size(), destination, capacity);
}

template <class EvalStatus>
AxesSolveFailureStatus MapAxesSolveFailureStatus(EvalStatus status) {
  switch (status) {
    case EvalStatus::kSuccess:
      return AxesSolveFailureStatus::kSuccess;
    case EvalStatus::kInvalidBinding:
      return AxesSolveFailureStatus::kInvalidBinding;
    case EvalStatus::kTypeMismatch:
      return AxesSolveFailureStatus::kTypeMismatch;
    case EvalStatus::kArithmeticError:
      return AxesSolveFailureStatus::kArithmeticError;
    case EvalStatus::kSdkError:
      return AxesSolveFailureStatus::kSdkError;
    default:
      return AxesSolveFailureStatus::kOther;
  }
}

// Mutable solver state belongs to the call. Only a solved result is written.
template <class Solver, class Input, class Context, class Binding, class Output>
TilingExpressionStatus SolveAxesRecord(const AxesSolveRecords &records, const std::vector<int64_t> &shapes,
                                       const Input &input, const AxesSolveOptions &options, void *destination,
                                       size_t capacity, AxesSolveOutcome *outcome,
                                       const AxesProgramScratch *scratch = nullptr, uint32_t *block_dim = nullptr,
                                       AxesSolveEvaluationFailure *failure = nullptr) {
  using Status = TilingExpressionStatus;
  using EvalStatus = decltype(std::declval<Context>().first_error);
  using Storage = AxesSolveTopologyStorage<Input>;
  if (outcome == nullptr) {
    return Status::kInvalidArgument;
  }
  if (!records.layout || records.layout->ValueCount() != records.topology.variables.size()) {
    return Status::kInvalidProgram;
  }
  const auto payload_size = records.layout->PayloadSize();
  if (capacity < payload_size) {
    return Status::kBufferTooSmall;
  }
  if (payload_size != 0 && destination == nullptr) {
    return Status::kInvalidArgument;
  }
  std::unique_ptr<Storage> storage;
  auto status = Storage::Build(records.topology, shapes, input, &storage);
  if (status != Status::kOk) {
    return status;
  }
  auto &native = storage->GetInput();
  AxesSolveAdapter<Solver, Input, Context, Binding, Output> adapter(native);
  status = adapter.Prepare(records.evaluation, scratch);
  if (status != Status::kOk) {
    return status;
  }
  const auto capture_failure = [&adapter, failure]() {
    const auto &context = adapter.GetEvaluationContext();
    if (failure != nullptr && context.first_error != EvalStatus::kSuccess) {
      *failure = {static_cast<uint32_t>(MapAxesSolveFailureStatus(context.first_error)),
                  context.error_expression_id,
                  context.error_stage_step,
                  context.error_instruction,
                  false,
                  UINT32_MAX,
                  context.error_cache_index};
    }
  };
  const bool solved = adapter.Run(options.trade_off, options.tune_block, options.tune, options.equal_order);
  if (adapter.GetEvaluationContext().first_error != EvalStatus::kSuccess) {
    capture_failure();
    return Status::kEvaluationError;
  }
  if (!solved || adapter.IsEmptyTensor()) {
    *outcome = solved ? AxesSolveOutcome::kEmpty : AxesSolveOutcome::kNoSolution;
    return Status::kOk;
  }
  uint32_t solved_block_dim = 0;
  if ((block_dim != nullptr || records.derived.layout != nullptr) && !adapter.GetSolvedBlockDim(solved_block_dim)) {
    capture_failure();
    return Status::kEvaluationError;
  }
  std::vector<int64_t> values(native.tiling_vars_size);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = native.tiling_vars[i]->value;
  }
  status = records.derived.layout == nullptr
               ? records.layout->WriteTiles(values.data(), values.size(), destination, capacity)
               : WriteAxesDerivedResult(records.derived, shapes, values, destination, capacity, solved_block_dim,
                                        input.ub_size, scratch, failure);
  if (status == Status::kOk) {
    if (block_dim != nullptr) {
      *block_dim = solved_block_dim;
    }
    *outcome = AxesSolveOutcome::kSolved;
  }
  return status;
}
}  // namespace att
#endif
