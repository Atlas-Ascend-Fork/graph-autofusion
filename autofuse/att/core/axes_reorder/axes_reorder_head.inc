/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included once per owning implementation namespace, after scalar helpers and log definitions.
#ifndef AF_AXES_EQUAL_ORDER
#error "Define AF_AXES_EQUAL_ORDER as 0 or 1 before including the shared core"
#endif
#ifndef AF_AXES_EXPLICIT_EVAL
#define AF_AXES_EXPLICIT_EVAL 0
#endif
// AF_AXES_RESOURCE_BEGIN
/*
ConstraintType:约束类型
  LOCAL_BUFFER:仅与内存占用相关的约束, 如s1t * s2t < UB
  LB_MIXED:与内存占用相关的约束
  MC_MIXED:纯多核相关的约束
*/
enum class ConstraintType {
  LOCAL_BUFFER = 0,
  LB_MIXED = 1,
  MC_MIXED = 2,
};

#if AF_AXES_EXPLICIT_EVAL
// Internal, per-invocation evaluation contract. No pointers in this contract are serialized.
struct AxesReorderSolverInput;
enum class EvalStatus : uint32_t {
  kSuccess = 0,
  kInvalidBinding,
  kTypeMismatch,
  kArithmeticError,
  kSdkError,
};
enum class EvalValueType : uint32_t {
  kInvalid = 0,
  kInt64,
  kUint64,
  kFloat32,
  kFloat64,
  kBool,
};
struct TypedValue {
  EvalValueType type = EvalValueType::kInvalid;
  union {
    int64_t signed_value = 0;
    uint64_t unsigned_value;
    float float_value;
    double double_value;
    bool bool_value;
  };
};
struct EvalBinding {
  uint32_t expression_id = UINT32_MAX;
  EvalValueType expected_type = EvalValueType::kInt64;
  const void *definition = nullptr;
  const void *native_data = nullptr;
  int64_t resource_value = 0;
};
struct EvalContext {
  const AxesReorderSolverInput *variables = nullptr;
  const void *program = nullptr;
  void *sdk_context = nullptr;
  EvalStatus first_error = EvalStatus::kSuccess;
  uint32_t error_expression_id = UINT32_MAX;
  void ResetError() {
    first_error = EvalStatus::kSuccess;
    error_expression_id = UINT32_MAX;
  }
  void RecordError(EvalStatus status, uint32_t expression_id) {
    if (first_error == EvalStatus::kSuccess && status != EvalStatus::kSuccess) {
      first_error = status;
      error_expression_id = expression_id;
    }
  }
};
using EvalEntry = EvalStatus (*)(EvalContext &, const EvalBinding &, TypedValue &);
inline EvalStatus NativeConstraintEntry(EvalContext &, const EvalBinding &, TypedValue &);
inline EvalStatus NativeUpperBoundEntry(EvalContext &, const EvalBinding &, TypedValue &);

#endif
struct Variable;
struct TilingVariable;
struct Constraint;
using ConsEvalFuncPtr = int64_t (*)(TilingVariable **rel_tiling_vars, Variable **rel_input_shapes, int64_t rel_hw_spec);
using GetUpperBoundFuncPtr = int64_t (*)(Variable **rel_ori_dims);

struct Variable {
  int64_t value = -1;
};

struct Constraint {
  int64_t rel_hw_spec = 0;
  uint32_t rel_tiling_vars_size = 0u;
  uint32_t rel_in_shapes_size = 0u;
  TilingVariable **rel_tiling_vars = nullptr;
  Variable **rel_in_shapes = nullptr;
  ConsEvalFuncPtr eval = nullptr;
#if AF_AXES_EXPLICIT_EVAL
  EvalBinding eval_binding;
  EvalEntry eval_entry = NativeConstraintEntry;
#endif
  ConstraintType type;
};

struct TilingVariable : public Variable {
  int64_t align = 1;
  int64_t prompt_align = 1;
  int64_t data_type_size = 4;
  bool notail = false;
  bool mc_related = false;
  TilingVariable *notail_var = nullptr;
  uint32_t rel_cons_size = 0u;
  uint32_t upper_bound_vars_size = 0u;
  Variable **upper_bound_vars = nullptr;
  Constraint **rel_cons = nullptr;
  GetUpperBoundFuncPtr upper_bound = nullptr;
#if AF_AXES_EXPLICIT_EVAL
  EvalBinding upper_bound_binding;
  EvalEntry upper_bound_entry = NativeUpperBoundEntry;
#endif
  size_t order{0UL};
  __attribute__((always_inline)) bool SetValue(int64_t val) noexcept{
    return (val > 0) ? (value = val, true) : false;
  }
};

#if AF_AXES_EXPLICIT_EVAL
inline EvalStatus NativeConstraintEntry(EvalContext &, const EvalBinding &binding, TypedValue &output) {
  const auto *constraint = static_cast<const Constraint *>(binding.native_data);
  if (constraint == nullptr || constraint->eval == nullptr) {
    return EvalStatus::kInvalidBinding;
  }
  output.signed_value =
      constraint->eval(constraint->rel_tiling_vars, constraint->rel_in_shapes, binding.resource_value);
  output.type = EvalValueType::kInt64;
  return EvalStatus::kSuccess;
}

inline EvalStatus NativeUpperBoundEntry(EvalContext &, const EvalBinding &binding, TypedValue &output) {
  const auto *variable = static_cast<const TilingVariable *>(binding.native_data);
  if (variable == nullptr || variable->upper_bound == nullptr) {
    return EvalStatus::kInvalidBinding;
  }
  output.signed_value = variable->upper_bound(variable->upper_bound_vars);
  output.type = EvalValueType::kInt64;
  return EvalStatus::kSuccess;
}

#endif
struct AxesReorderSolverInput {
  uint32_t result_id = 0u;
  uint32_t group_id = 0u;
  uint32_t case_id = 0u;
  uint32_t sub_case_id = 0u;
  uint32_t core_num = 0u;
  uint32_t ub_size = 0u;
  uint32_t input_vars_size = 0u;
  uint32_t tiling_vars_size = 0u;
  uint32_t pure_mc_vars_size = 0u;
  uint32_t local_buffer_vars_size = 0u;
  uint32_t all_cons_size = 0u;
  double ub_threshold = 0.2f;
  double corenum_threshold = 0.4f;
  double perf_threshold = 0000.0f;
  Variable **input_vars = nullptr;
  TilingVariable **tiling_vars = nullptr;
  TilingVariable **pure_mc_vars = nullptr;
  TilingVariable **local_buffer_vars = nullptr;
  TilingVariable **ordered_local_buffer_vars = nullptr;
  uint32_t *ordered_to_canonical = nullptr;
  Constraint **all_cons = nullptr;

std::string DebugString() const {
  std::stringstream ss;
  ss << "result_id: " << result_id << ", group_id: " << group_id << ", case_id: " << case_id << ", sub_case_id: " << sub_case_id;
  ss << "core_num: " << core_num;
  ss << ", ub_size: " << ub_size;
  if (input_vars != nullptr) {
    for (uint32_t i = 0; i < input_vars_size; i++) {
      if (input_vars[i] != nullptr) {
        ss << ", input_vars[" << i << "]: " << input_vars[i]->value;
      }
    }
  }
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0; i < tiling_vars_size; i++) {
      if (tiling_vars[i] != nullptr) {
        ss << ", tiling_vars[" << i << "]: " << tiling_vars[i]->value;
      }
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0; i < pure_mc_vars_size; i++) {
      if (pure_mc_vars[i] != nullptr) {
        ss << ", pure_mc_vars[" << i << "]: " << pure_mc_vars[i]->value;
      }
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0; i < local_buffer_vars_size; i++) {
      if (local_buffer_vars[i] != nullptr) {
        ss << ", local_buffer_vars[" << i << "]: " << local_buffer_vars[i]->value;
      }
    }
  }
  ss << ", all_cons_size: " << all_cons_size;
  ss << ", ub_threshold: " << ub_threshold;
  ss << ", corenum_threshold: " << corenum_threshold;
  ss << ", perf_threshold: " << perf_threshold;
  return ss.str();
}
};

class AxesReorderSolver {
public:
#if AF_AXES_EXPLICIT_EVAL
  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {
    eval_context_.variables = &input_;
  }
  AxesReorderSolver(const AxesReorderSolver &) = delete;
  AxesReorderSolver &operator=(const AxesReorderSolver &) = delete;
#else
  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {}
#endif
  ~AxesReorderSolver() {}
  bool Run(const bool is_trade_off, const bool block_loop_auto_tune, const bool enable_auto_tune,
           const bool enable_equal_order = false);
  bool IsEmptyTensor() const { return is_empty_tensor_; }
#if AF_AXES_EXPLICIT_EVAL
  const EvalContext &GetEvaluationContext() const { return eval_context_; }
#endif
protected:
  virtual bool CalUsedCoreNum(double &used_core_num) = 0;
  virtual bool CalRealUsedCoreNum(int64_t &used_corenum) = 0;
  virtual bool SatisfyThresholdUBSize() = 0;
  virtual double GetPerf() = 0;
  virtual bool SatisfyUBSizeCacheLine(uint32_t idx) = 0;
  AxesReorderSolverInput input_;
  bool is_empty_tensor_{false};
#if AF_AXES_EXPLICIT_EVAL
  static size_t BackupStorageSize(size_t size) { return std::max<size_t>(1U, size); }
  mutable EvalContext eval_context_;
  // Only evaluation failures use this private unwinding path. Ordinary false/-1 results keep their original meaning.
  struct EvalAbort {};
  void CheckEvaluation() const {
    if (eval_context_.first_error != EvalStatus::kSuccess) {
      throw EvalAbort{};
    }
  }
  int64_t EvaluateInt64(EvalEntry entry, const EvalBinding &binding) const {
    CheckEvaluation();
    if (entry == nullptr || binding.expected_type != EvalValueType::kInt64) {
      eval_context_.RecordError(EvalStatus::kInvalidBinding, binding.expression_id);
      CheckEvaluation();
    }
    TypedValue output;
    eval_context_.RecordError(entry(eval_context_, binding, output), binding.expression_id);
    CheckEvaluation();
    if (output.type != binding.expected_type) {
      eval_context_.RecordError(EvalStatus::kTypeMismatch, binding.expression_id);
      CheckEvaluation();
    }
    return output.signed_value;
  }
  int64_t EvaluateConstraint(const Constraint *constraint, int64_t resource_value) const {
    auto binding = constraint->eval_binding;
    binding.native_data = constraint;
    binding.resource_value = resource_value;
    return EvaluateInt64(constraint->eval_entry, binding);
  }
  int64_t EvaluateUpperBound(const TilingVariable *variable) const {
    auto binding = variable->upper_bound_binding;
    binding.native_data = variable;
    return EvaluateInt64(variable->upper_bound_entry, binding);
  }
  bool CheckedCalUsedCoreNum(double &value) {
    CheckEvaluation();
    const bool result = CalUsedCoreNum(value);
    CheckEvaluation();
    return result;
  }
  bool CheckedCalRealUsedCoreNum(int64_t &value) {
    CheckEvaluation();
    const bool result = CalRealUsedCoreNum(value);
    CheckEvaluation();
    return result;
  }
  double CheckedGetPerf() {
    CheckEvaluation();
    const double result = GetPerf();
    CheckEvaluation();
    return result;
  }
  bool CheckedSatisfyUBSizeCacheLine(uint32_t idx) {
    CheckEvaluation();
    const bool result = SatisfyUBSizeCacheLine(idx);
    CheckEvaluation();
    return result;
  }
  bool HasUpperBoundBinding(const TilingVariable *variable) const {
    return variable->upper_bound_entry != NativeUpperBoundEntry || variable->upper_bound != nullptr;
  }
#else
  // The generated PGO extension still calls this inherited adapter.
  int64_t EvaluateUpperBound(const TilingVariable *variable) const {
    return variable->upper_bound(variable->upper_bound_vars);
  }
#endif

private:
#if AF_AXES_EXPLICIT_EVAL
  bool RunImpl(bool is_trade_off, bool block_loop_auto_tune, bool enable_auto_tune, bool enable_equal_order);
#endif
  inline bool FindNextUpperBlockDim(const uint32_t block_dim, uint32_t &next_lower_block_dim) const;
  inline bool FindNextLowerBlockDim(const uint32_t block_dim, uint32_t &next_upper_block_dim) const;
  inline void SaveInputTilingVars(TilingVariable *tiling_vars, TilingVariable *pure_mc_vars,
                                  TilingVariable *local_buffer_vars) const;
  inline void RestoreInputTilingVars(const TilingVariable *tiling_vars, const TilingVariable *pure_mc_vars,
                                     const TilingVariable *local_buffer_vars) const;
  inline bool WorkloadBalance();
  inline bool IsSatisfyCons(TilingVariable *var, int64_t value, ConstraintType cons_type, double ratio = 1.0);
  inline bool BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable *var, int64_t lower_bound,
                                                   int64_t &result);
  inline bool BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable *var, uint32_t idx, int64_t lower_bound,
                                                           int64_t &result);
  inline std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable *var, uint32_t idx,
                                                                          int64_t lower_bound);
  // Multi-variable versions of binary search functions
  bool BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound,
                                            int64_t &result);
  bool BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable **vars, uint32_t *idxs, size_t count,
                                                    int64_t lower_bound, int64_t &result);
  std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable **vars, uint32_t *idxs, size_t count,
                                                                   int64_t lower_bound);
  // Multi-variable version with per-variable upper bounds
  // Returns: {value_for_vars[0], value_for_vars[1], ...}
  // Each value is the binary search result (left) clipped to that variable's upper bound
  // When variables have different upper bounds, this enables non-symmetric solutions
  // Example: if upper_bounds={32, 128} and left=64, returns {32, 64}
  //          where vars[0] (a-axis) gets 32 (smaller), vars[1] (b-axis) gets 64 (larger)
  std::pair<int64_t, int64_t> BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable **vars, uint32_t *idxs,
                                                                    size_t count, int64_t lower_bound,
                                                                    const int64_t *upper_bounds);
  bool TuneNotailVar(TilingVariable *var);
  bool InitLocalBufferVars();
  bool InitMulticoreVars();
  bool ProcessMCAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num);
  bool ProcessSingleAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num);
  bool MulticoreTiling(bool block_loop_auto_tune, bool enable_workload_balance=false);
  bool MulticoreTilingCore(bool block_loop_auto_tune);
#if AF_AXES_EQUAL_ORDER
  inline bool GetTiling(const bool is_tuning, const bool block_loop_auto_tune, const bool enable_workload_balance, const bool enable_equal_order);
  inline bool GetMaxBlockDimTiling(const uint32_t block_dim, const bool enable_equal_order);
  inline bool AutoTuning(const bool is_trade_off, const bool enable_equal_order);
  inline void FindBetterSolutionByUpperBlockDim(double next_upper_perf, uint32_t next_upper_block_dim, const bool enable_equal_order);
  inline void FindBetterSolutionByLowerBlockDim(double next_lower_perf, uint32_t next_lower_block_dim, const bool enable_equal_order);
  bool NaiveLocalBufTiling(const bool enable_equal_order);
  bool BinaryLocalBufTiling(const bool enable_equal_order);
#else
  inline bool GetTiling(const bool is_tuning, const bool block_loop_auto_tune, const bool enable_workload_balance);
  inline bool GetMaxBlockDimTiling(const uint32_t block_dim);
  inline bool AutoTuning(const bool is_trade_off);
  inline void FindBetterSolutionByUpperBlockDim(double next_upper_perf, uint32_t next_upper_block_dim);
  inline void FindBetterSolutionByLowerBlockDim(double next_lower_perf, uint32_t next_lower_block_dim);
  bool NaiveLocalBufTiling();
  bool BinaryLocalBufTiling();
#endif
  bool BinaryLocalBufTilingCore(const std::vector<bool> &solved_axes);
  void ApplyPromptAlign(TilingVariable *var);
#if AF_AXES_EQUAL_ORDER
  bool LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune, const bool enable_equal_order);
#else
  bool LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune);
#endif
  bool DecreaseUntilSatisfied(TilingVariable *var, ConstraintType cons_type,     std::function<bool(TilingVariable *)> tune_func);
  bool ShrinkBoundaryUntilSatisfied(TilingVariable *var, int64_t boundary, ConstraintType cons_type);
#if AF_AXES_EQUAL_ORDER
  // Equal priority axes support functions
  bool IdentifyEqualPriorityAxes(const uint32_t axes_num, uint32_t *axis_idx, int64_t &min_upper_bound,     int64_t &max_aligned);
  bool BinarySearchEqualPriorityAxes(uint32_t axis_num, uint32_t *axis_idx, int64_t lower_bound,     int64_t upper_bound);
  bool IterativeSolveEqualPriorityAxes(const uint32_t axis_num, uint32_t *axis_idx, int64_t lower_bound,     int64_t upper_bound);
  bool BinarySearchWithAlignment(TilingVariable **vars, uint32_t var_num, int64_t lower_bound,     int64_t upper_bound, int64_t align, const char *log_prefix);
  bool TuneNoTail(TilingVariable **vars, uint32_t id);
  // NaiveLocalBufTiling dual threshold support functions
  bool SolveEqualPriorityAxesWithDualThreshold(uint32_t *axis_idx, int64_t low_bound,     int64_t upper_bound, std::vector<bool> &solved_axes);
  bool BinarySearchEqualPriorityAxesWithDualThreshold(TilingVariable **vars, int64_t lower_bound,     int64_t upper_bound, int64_t &upper_ub_a, int64_t &upper_ub_b);
  bool ProcessNonMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes);
  bool ProcessSingleMCAxis(TilingVariable **vars, bool first_is_mc, uint32_t *axis_idx,     std::vector<bool> &solved_axes);
  bool ProcessDualMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes);
  bool FinalizeEqualPriorityAxes(uint32_t *axis_idx, std::vector<bool> &solved_axes) const;
  // ProcessDualMCAxes helper functions
  struct DualAxesInfo {
    TilingVariable *var_a{nullptr};
    TilingVariable *var_b{nullptr};
    uint32_t idx_a{0U};
    uint32_t idx_b{1U};
    int64_t upper_bound_a{0L};
    int64_t upper_bound_b{0L};
    int64_t low_bound_both{0L};
    int64_t upper_bound_both{0L};
  };
  // Three-phase algorithm framework
  DualAxesInfo InitializeDualAxesInfo(TilingVariable **vars, uint32_t *axis_idx) const;
  bool ExecuteStep1_FindUBLimit(const DualAxesInfo &info, int64_t &low_bound_both_ub,
    int64_t &low_bound_b_ub, int64_t &range_a_high, int64_t &range_b_high);
  bool ExecuteStep2_FindUBThreshold(const DualAxesInfo &info, int64_t range_a_low,
    int64_t range_a_high, int64_t &range_a_low_out, int64_t &range_b_low_out);
  bool ExecuteStep3_FindCoreNumTileSize(const DualAxesInfo &info, int64_t range_a_low,
    int64_t range_a_high, int64_t range_b_high, TilingVariable **vars_final);
#else
#endif
protected:
  bool SatisfyCons(ConstraintType cons_type);
  bool SatisfyCons(TilingVariable *var, ConstraintType cons_type, double ratio = 1.0);
  bool SatisfyMCCons();
};

// AF_AXES_RESOURCE_END
