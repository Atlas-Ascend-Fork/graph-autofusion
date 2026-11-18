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
// AF_AXES_RESOURCE_BEGIN
// Callback policy only; the search and selection algorithm below is shared.
// Native expands to the original calls. Explicit evaluation retains checked adapters.
// Pointer arguments are local variables, never expressions with side effects.
#if AF_AXES_EXPLICIT_EVAL
#define AF_AXES_BACKUP_SIZE BackupStorageSize
#define AF_AXES_EVAL_CONSTRAINT EvaluateConstraint
#define AF_AXES_EVAL_UPPER_BOUND EvaluateUpperBound
#define AF_AXES_HAS_UPPER_BOUND HasUpperBoundBinding
#define AF_AXES_CAL_USED_CORE_NUM CheckedCalUsedCoreNum
#define AF_AXES_CAL_REAL_USED_CORE_NUM CheckedCalRealUsedCoreNum
#define AF_AXES_GET_PERF CheckedGetPerf
#define AF_AXES_SATISFY_UB_CACHE_LINE CheckedSatisfyUBSizeCacheLine
#else
#define AF_AXES_BACKUP_SIZE(size) (size)
#define AF_AXES_EVAL_CONSTRAINT(c, r) ((c)->eval((c)->rel_tiling_vars, (c)->rel_in_shapes, (r)))
#define AF_AXES_EVAL_UPPER_BOUND(v) ((v)->upper_bound((v)->upper_bound_vars))
#define AF_AXES_HAS_UPPER_BOUND(v) ((v)->upper_bound != nullptr)
#define AF_AXES_CAL_USED_CORE_NUM CalUsedCoreNum
#define AF_AXES_CAL_REAL_USED_CORE_NUM CalRealUsedCoreNum
#define AF_AXES_GET_PERF GetPerf
#define AF_AXES_SATISFY_UB_CACHE_LINE SatisfyUBSizeCacheLine
#endif

constexpr uint32_t kDualAxesCount = 2U;
constexpr int64_t kMidpointDivisor = 2;
constexpr int64_t kWorkloadBalanceCoreNum = 2;

bool AxesReorderSolver::InitLocalBufferVars() {
  auto *vars = input_.local_buffer_vars;
  const auto size = input_.local_buffer_vars_size;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      if (!vars[i+k]->SetValue(vars[i+k]->align)) {
        OP_LOGW(OP_NAME, "Failed to init local buffer value.");
        return false;
      }
    }
  }
  return true;
}

bool AxesReorderSolver::InitMulticoreVars() {
  uint32_t size = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (uint32_t i = 0u; i < size; i++) {
    auto &var = vars[i];
    auto upper_bound_val = AF_AXES_EVAL_UPPER_BOUND(var);
    if (upper_bound_val == -1) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;
    if (!var->SetValue(upper_bound_val)) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(ConstraintType cons_type) {
  uint32_t size = input_.all_cons_size;
  auto *cons_list = input_.all_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      auto &cons = cons_list[i+k];
    if (cons->type != cons_type) {
      continue;
    }
    if (AF_AXES_EVAL_CONSTRAINT(cons, static_cast<double>(cons->rel_hw_spec)) > 0) {
      OP_LOGD(OP_NAME, "Check eval failed, rel_hw_spec:%ld, input:%s", cons->rel_hw_spec, input_.DebugString().c_str());
      return false;
    }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(TilingVariable *var, ConstraintType cons_type, double ratio) {
  uint32_t size = var->rel_cons_size;
  auto *cons_list = var->rel_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      auto &cons = cons_list[i+k];
    if (cons->type != cons_type) {
      continue;
    }
    if (AF_AXES_EVAL_CONSTRAINT(cons, Ceiling(ratio * static_cast<double>(cons->rel_hw_spec))) > 0) {
      OP_LOGD(OP_NAME, "Check eval failed, rel_hw_spec:%ld, input:%s, ratio:%lf", cons->rel_hw_spec,
              input_.DebugString().c_str(), ratio);
      return false;
    }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyMCCons() {
  int64_t used_core_num = 0;
  AF_AXES_CAL_REAL_USED_CORE_NUM(used_core_num);
  return used_core_num <= static_cast<int64_t>(input_.core_num);
}

bool AxesReorderSolver::TuneNotailVar(TilingVariable* var) {
  if (!var->notail) {
    return true;
  }
  if (var->notail_var->value % var->value == 0) {
    return true;
  }
  for (; var->value > 0; var->value -= var->align) {
    if (var->notail_var->value % var->value != 0) {
      continue;
    }
    break;
  }
  return var->value != 0;
}

bool AxesReorderSolver::MulticoreTilingCore(bool block_loop_auto_tune) {
  const int32_t num_vars = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (int32_t i = num_vars - 1; i >= 0; --i) {
    auto &var = vars[i];
    int64_t boundary = var->align;
    auto init_val = var->value;
    int64_t last_boundary = -1;
    int64_t last_val = -1;
    double pre_obj = -1.0;
    double cur_obj = -1.0;
    double next_obj = -1.0;
    if (block_loop_auto_tune) {
      pre_obj = AF_AXES_GET_PERF();
    }
    while (!(last_boundary == boundary && last_val == var->value)) {
      last_boundary = boundary;
      last_val = var->value;
      var->value = CeilDiv((boundary + var->value) / kMidpointDivisor, var->align);
      bool not_satisfied_mc_cons = !SatisfyMCCons();
      if (block_loop_auto_tune) {
        cur_obj = AF_AXES_GET_PERF();
        var->value += 1;
        next_obj = AF_AXES_GET_PERF();
        var->value -= 1;
        not_satisfied_mc_cons = not_satisfied_mc_cons || cur_obj > pre_obj || cur_obj > next_obj;
      }
      if (not_satisfied_mc_cons) {
        boundary = var->value;
        var->value = last_val;
      } else {
        if (block_loop_auto_tune) {
          pre_obj = cur_obj;
        }
      }
    }

    if (!SatisfyMCCons()) {
      var->value = init_val;
    }
    while (!SatisfyCons(var, ConstraintType::MC_MIXED) && var->value != init_val) {
      var->value += var->align;
    }
    }
  if (!SatisfyCons(ConstraintType::MC_MIXED)) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the final check, input: %s", input_.DebugString().c_str());
    return false;
  }
  return true;
}

bool AxesReorderSolver::MulticoreTiling(bool block_loop_auto_tune, bool enable_workload_balance) {
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyMCCons()) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the first check, input: %s.",
            input_.DebugString().c_str());
    return false;
  }
  if (!MulticoreTilingCore(block_loop_auto_tune)) {
    OP_LOGW(OP_NAME, "Multicore tiling core calculation failed in the core check, block_loop_auto_tune: %d, input: %s.",
            block_loop_auto_tune, input_.DebugString().c_str());
    return false;
  }
  if (enable_workload_balance) {
  int64_t cur_corenum = 0;
  AF_AXES_CAL_REAL_USED_CORE_NUM(cur_corenum);
  if (cur_corenum != kWorkloadBalanceCoreNum) {
    return true;
  }
  const int32_t num_vars = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  double cur_corenum_fp = 0.0;
  AF_AXES_CAL_USED_CORE_NUM(cur_corenum_fp);
  TilingVariable optimal_mc_vars[AF_AXES_BACKUP_SIZE(input_.pure_mc_vars_size)];

  for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
    optimal_mc_vars[i] = *vars[i];
  }
  double max_balance = std::fmod(cur_corenum_fp, 1.0);
  OP_LOGD(OP_NAME, "max_balance initialized: %f, current corenum is %ld", max_balance, cur_corenum);
  if (fabs(max_balance) < 0.00000001f) {
    OP_LOGI(OP_NAME, "max_balance already satisfied");
    return true;
  }
  double balance = max_balance;
  for (int32_t i=num_vars-1; i >= 0; i--) {
    auto &var = vars[i];
    int64_t corenum = 0;
    auto upper_bound_val = AF_AXES_EVAL_UPPER_BOUND(var);
    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;
    while (SatisfyCons(ConstraintType::MC_MIXED) && (var->value < upper_bound_val)) {
      var->value += var->align;
      AF_AXES_CAL_REAL_USED_CORE_NUM(corenum);
      if ((corenum < std::max(1L, cur_corenum - 1))) {
        break;
      }
      double corenum_fp;
      AF_AXES_CAL_USED_CORE_NUM(corenum_fp);
      balance = std::fmod(corenum_fp, 1.0);
      if ((fabs(balance) < 0.00000001f) || (balance > max_balance)) {
        max_balance = balance;

  for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
    optimal_mc_vars[i] = *vars[i];
  }
        OP_LOGD(OP_NAME, "max_balance updated: %f, corenum updated: %ld", max_balance, corenum);
      }
    }
  }
    for (int32_t i=0; i < num_vars; i++) {
      input_.pure_mc_vars[i]->value = optimal_mc_vars[i].value;
    }
  }
  return true;
}

  void AxesReorderSolver::ApplyPromptAlign(TilingVariable *var) {
    const auto original_val = var->value;
    auto aligned_val = original_val;
    bool found_prompt_aligned = false;
    while (aligned_val > 0) {
      if ((aligned_val % var->prompt_align == 0) && (aligned_val % var->align == 0)) {
        found_prompt_aligned = true;
        break;
      }
      aligned_val -= var->align;
    }
    bool is_applied = found_prompt_aligned && (aligned_val != original_val) && (aligned_val > 0);
    if (is_applied) {
      if (!AF_AXES_HAS_UPPER_BOUND(var)) {
        OP_LOGI(OP_NAME, "Var upper bound func is not set.");
        return;
      }
      const auto upper_bound_val = AF_AXES_EVAL_UPPER_BOUND(var);
      const auto loop_size = upper_bound_val / original_val;
      const auto tail_size = upper_bound_val % original_val;
      const auto tile_data_size = original_val * var->data_type_size;
      // if tile data size is less than 512B, no need to update prompt align
      if ((loop_size == 1) && (tail_size == 0) && (tile_data_size <= 512)) {
        OP_LOGI(OP_NAME, "No need to update prompt align, as loop size is 1 and tail size is 0, tile data size is %ld",
                tile_data_size);
        return;
      }
      // 当block_len > 64B 对性能影响较大
      if ((original_val * var->data_type_size) <= 64) {
        OP_LOGI(OP_NAME, "No need to update prompt align, as block len is less than 64B");
        return;
      }
      OP_LOGI(OP_NAME, "Update prompt align from %ld to %ld", original_val, aligned_val);
      var->value = aligned_val;
    }
  }

bool AxesReorderSolver::IsSatisfyCons(TilingVariable *var, int64_t val, ConstraintType cons_type, double ratio) {
  auto old = var->value;
  var->SetValue(val);
  bool satisfied = SatisfyCons(var, cons_type, ratio);
  var->SetValue(old);
  return satisfied;
}
// Single-variable version delegates to multi-variable version
bool AxesReorderSolver::BinaryFindUpperBoundSatisfiedUBLimit(TilingVariable *var, int64_t lower_bound,
                                                             int64_t &result) {
  // Delegate to multi-variable version
  TilingVariable* vars[] = {var};
  uint32_t idxs[] = {0};
  return BinaryFindUpperBoundSatisfiedUBLimit(vars, idxs, 1, lower_bound, result);
}

// Multi-variable version of BinaryFindUpperBoundSatisfiedUBLimit
// Simultaneously sets all variables to the same value and checks UB constraint
bool AxesReorderSolver::BinaryFindUpperBoundSatisfiedUBLimit(
    TilingVariable **vars, uint32_t *idxs, size_t count,
    int64_t lower_bound, int64_t &result) {
  (void)idxs;  // Suppress unused parameter warning

  if (count == 0) return false;

  const int64_t align = vars[0]->align;
  if (align <= 0) {
    OP_LOGW(OP_NAME, "MultiVarUBLimit: invalid align %ld", align);
    return false;
  }
  const int64_t raw_upper = vars[0]->value;
  const int64_t aligned_lower = CeilDiv(lower_bound, align) * align;
  const int64_t aligned_upper = (raw_upper / align) * align;
  if (aligned_lower > aligned_upper) {
    OP_LOGW(OP_NAME, "MultiVarUBLimit: aligned_lower > aligned_upper, lower:%ld, upper:%ld, align:%ld",
            aligned_lower, aligned_upper, align);
    return false;
  }

  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }

  // Check if aligned_upper satisfies constraint when all variables are set to it
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(aligned_upper);
  }
  bool all_satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);
  if (all_satisfied) {
    result = aligned_upper;
    return true;
  }

  // Binary search on align-grid for maximum value satisfying UB constraint
  int64_t left_idx = aligned_lower / align;
  int64_t right_idx = aligned_upper / align;
  result = aligned_lower;

  while (left_idx <= right_idx) {
    int64_t mid_idx = left_idx + (right_idx - left_idx) / kMidpointDivisor;
    int64_t candidate = mid_idx * align;

    // Set all variables to candidate and check constraint
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(candidate);
    }
    bool mid_satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);
    if (mid_satisfied) {
      result = candidate;
      left_idx = mid_idx + 1;
    } else {
      right_idx = mid_idx - 1;
    }
  }

  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }
  return true;
}

// Single-variable version delegates to multi-variable version
bool AxesReorderSolver::BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable *var, uint32_t idx,
                                                                     int64_t lower_bound, int64_t &result) {
  // Delegate to multi-variable version
  OP_LOGD(OP_NAME, "Begin to search ub threshold limit bound, upper_bound:%ld, lower_bound:%ld, id:%u",
          var->value, lower_bound, idx);
  TilingVariable* vars[] = {var};
  uint32_t idxs[] = {idx};
  return BinaryFindLowerBoundSatisfiedUBThresholdCond(vars, idxs, 1, lower_bound, result);
}

// Multi-variable version of BinaryFindLowerBoundSatisfiedUBThresholdCond
// Simultaneously sets all variables to the same value and checks UB threshold
bool AxesReorderSolver::BinaryFindLowerBoundSatisfiedUBThresholdCond(TilingVariable **vars, uint32_t *idxs,
                                                                     size_t count, int64_t lower_bound,
                                                                     int64_t &result) {
  if (count == 0) return false;
  int64_t upper_bound = vars[0]->value;
  if (upper_bound < lower_bound) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: upper_bound < lower_bound, return false");
    return false;
  }

  const int64_t align = vars[0]->align;
  if (align <= 0) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: invalid align %ld", align);
    return false;
  }
  const int64_t aligned_lower = CeilDiv(lower_bound, align) * align;
  const int64_t aligned_upper = (upper_bound / align) * align;
  if (aligned_lower > aligned_upper) {
    OP_LOGW(OP_NAME, "MultiVarUBThr: aligned_lower > aligned_upper, lower:%ld, upper:%ld, align:%ld",
            aligned_lower, aligned_upper, align);
    return false;
  }

  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }

  // Check if aligned lower_bound satisfies conditions for all variables
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(aligned_lower);
  }
  bool bound_ok = true;
  for (size_t i = 0; i < count; ++i) {
    if (!AF_AXES_SATISFY_UB_CACHE_LINE(idxs[i])) {
      bound_ok = false;
      break;
    }
  }
  // Check if UB usage is ABOVE threshold (i.e., !SatisfyCons with threshold ratio)
  if (bound_ok && IsSatisfyCons(vars[0], vars[0]->value, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
    bound_ok = false;
  }
  result = aligned_lower;
  if (bound_ok) {
    return true;
  }

  // Binary search on align-grid for minimum value that satisfies conditions for all variables
  int64_t left_idx = aligned_lower / align;
  int64_t right_idx = aligned_upper / align;
  result = aligned_upper;
  while (left_idx <= right_idx) {
    int64_t mid_idx = left_idx + (right_idx - left_idx) / kMidpointDivisor;
    int64_t candidate = mid_idx * align;
    // Set all variables to candidate and check conditions
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(candidate);
    }
    bool mid_satisfied = true;
    for (size_t i = 0; i < count; ++i) {
      if (!AF_AXES_SATISFY_UB_CACHE_LINE(idxs[i])) {
        mid_satisfied = false;
        break;
      }
    }
    // Check if UB usage is ABOVE threshold
    if (mid_satisfied && IsSatisfyCons(vars[0], candidate, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
      mid_satisfied = false;
    }
    if (mid_satisfied) {
      result = candidate;
      right_idx = mid_idx - 1;
    } else {
      left_idx = mid_idx + 1;
    }
  }

  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }

  return true;
}
// Single-variable version delegates to multi-variable version
inline std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(TilingVariable *var,
                                                                                           uint32_t idx,
                                                                                           int64_t lower_bound) {
  // Delegate to multi-variable version
  TilingVariable* vars[] = {var};
  uint32_t idxs[] = {idx};
  return BinaryFindLowerBoundSatisfiedCoreNum(vars, idxs, 1, lower_bound);
}

// Multi-variable version of BinaryFindLowerBoundSatisfiedCoreNum
// Simultaneously sets all variables to the same value and checks core number constraint
std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(
    TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound) {
  if (count == 0) return {lower_bound, lower_bound};
  int64_t upper_bound = vars[0]->value;
  if (lower_bound >= upper_bound) {
    OP_LOGW(OP_NAME, "MultiVarCoreNum: lower_bound >= upper_bound, return lower_bound");
    return {lower_bound, lower_bound};
  }
  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }
  // Calculate max core number threshold
  int64_t max_core_num = static_cast<int64_t>(
      input_.corenum_threshold * static_cast<double>(input_.core_num));
  // Check if upper_bound already satisfies condition
  (void)MulticoreTilingCore(false);
  int64_t available_core_num = 0L;
  (void)AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num);
  if (available_core_num >= max_core_num) {
    return {upper_bound, upper_bound};
  }

  // Binary search for value that satisfies core number constraint
  int64_t left = lower_bound;
  int64_t right = upper_bound - vars[0]->align;
  while (left <= right) {
    int64_t mid = left + (right - left) / kMidpointDivisor;
    // Set all variables to mid and check condition
    for (size_t i = 0; i < count; ++i) {
      vars[i]->SetValue(mid);
    }
    // Check if all variables satisfy cache line constraint
    bool all_cache_line_ok = true;
    for (size_t i = 0; i < count; ++i) {
      if (!AF_AXES_SATISFY_UB_CACHE_LINE(idxs[i])) {
        all_cache_line_ok = false;
        break;
      }
    }
    if (!all_cache_line_ok || (AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num) && (available_core_num >= max_core_num))) {
      left = mid + vars[0]->align;
    } else {
      right = mid - vars[0]->align;
    }
    OP_LOGD(OP_NAME, "MultiVarCoreNum: left=%ld, right=%ld, mid=%ld, core_num=%ld, max=%ld",
            left, right, mid, available_core_num, max_core_num);
  }

  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }

  return {right, left};
}

// ============================================================================
// Helper functions for BinaryFindLowerBoundSatisfiedCoreNum (advanced version)
// ============================================================================


bool AxesReorderSolver::ShrinkBoundaryUntilSatisfied(TilingVariable *var, int64_t boundary, ConstraintType cons_type) {
  int64_t last_boundary = -1;
  int64_t last_val = -1;
  while (!(last_boundary == boundary && last_val == var->value)) {
    last_boundary = boundary;
    last_val = var->value;
    var->value = CeilDiv((boundary + var->value) / kMidpointDivisor, var->align) * var->align;
    if (!SatisfyCons(var, cons_type)) {
      boundary = var->value;
      var->value = last_val;
    }
  }
  return true;
}

bool AxesReorderSolver::DecreaseUntilSatisfied(TilingVariable *var, ConstraintType cons_type,
                                               std::function<bool(TilingVariable *)> tune_func) {
  while (!SatisfyCons(var, cons_type) && var->value != var->align) {
    var->value -= var->align;
    if (tune_func != nullptr && !tune_func(var)) {
      OP_LOGW(OP_NAME, "Tune function failed");
      return false;
    }
  }
  return true;
}

#if AF_AXES_EQUAL_ORDER
// Multi-variable version of BinaryFindLowerBoundSatisfiedCoreNum with per-variable upper bounds
// This version allows different variables to have different upper bounds, enabling non-symmetric solutions
// When the binary search mid value exceeds a variable's upper bound, that variable stays at its upper bound
std::pair<int64_t, int64_t> AxesReorderSolver::BinaryFindLowerBoundSatisfiedCoreNum(
    TilingVariable **vars, uint32_t *idxs, size_t count, int64_t lower_bound, const int64_t *upper_bounds) {
  if (count == 0) return {lower_bound, lower_bound};

  // Find the maximum upper bound for binary search range
  int64_t max_upper_bound = upper_bounds[0];
  for (size_t i = 1; i < count; ++i) {
    if (upper_bounds[i] > max_upper_bound) {
      max_upper_bound = upper_bounds[i];
    }
  }

  if (lower_bound >= max_upper_bound) {
    OP_LOGW(OP_NAME, "MultiVarCoreNumWithUpperBounds: lower_bound >= max_upper_bound, return lower_bound");
    return {lower_bound, lower_bound};
  }

  // Save original values
  std::vector<int64_t> old_values(count);
  for (size_t i = 0; i < count; ++i) {
    old_values[i] = vars[i]->value;
  }

  // Calculate max core number threshold
  int64_t max_core_num = static_cast<int64_t>(
      input_.corenum_threshold * static_cast<double>(input_.core_num));

  // Check if max_upper_bound already satisfies condition
  for (size_t i = 0; i < count; ++i) {
    int64_t val = (max_upper_bound <= upper_bounds[i]) ? max_upper_bound : upper_bounds[i];
    vars[i]->SetValue(val);
  }
  (void)MulticoreTilingCore(false);
  int64_t available_core_num = 0L;
  (void)AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num);

  if (available_core_num >= max_core_num) {
    // All variables at their respective bounds satisfy the condition
    std::pair<int64_t, int64_t> result = {upper_bounds[0], upper_bounds[0]};
    for (size_t i = 1; i < count; ++i) {
      result.first = std::min(result.first, upper_bounds[i]);
      result.second = std::max(result.second, upper_bounds[i]);
    }
    return result;
  }

  // Binary search for value that satisfies core number constraint
  int64_t left = lower_bound;
  int64_t right = max_upper_bound - vars[0]->align;

  while (left <= right) {
    int64_t mid = left + (right - left) / kMidpointDivisor;

    // Set each variable to mid, but capped at its individual upper bound
    for (size_t i = 0; i < count; ++i) {
      int64_t val = (mid <= upper_bounds[i]) ? mid : upper_bounds[i];
      vars[i]->SetValue(val);
    }

    // Check if all variables satisfy cache line constraint
    bool all_cache_line_ok = true;
    for (size_t i = 0; i < count; ++i) {
      if (!AF_AXES_SATISFY_UB_CACHE_LINE(idxs[i])) {
        all_cache_line_ok = false;
        break;
      }
    }

    // Check if UB usage is ABOVE threshold
    if (all_cache_line_ok && IsSatisfyCons(vars[0], mid, ConstraintType::LOCAL_BUFFER, input_.ub_threshold)) {
      all_cache_line_ok = false;
    }

    if (!all_cache_line_ok || (AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num) && (available_core_num >= max_core_num))) {
      left = mid + vars[0]->align;
    } else {
      right = mid - vars[0]->align;
    }

    OP_LOGD(OP_NAME, "MultiVarCoreNumWithUpperBounds: left=%ld, right=%ld, mid=%ld, core_num=%ld, max=%ld",
            left, right, mid, available_core_num, max_core_num);
  }

  // Restore original values
  for (size_t i = 0; i < count; ++i) {
    vars[i]->SetValue(old_values[i]);
  }

  // Return values corresponding to each variable in order
  // For each variable: use left (converged value) clipped to its upper bound
  // This ensures return[i] corresponds to vars[i]
  //
  // For dual-axis non-symmetric case:
  // - vars[0] (a-axis): smaller upper bound (e.g., 32) → gets smaller value
  // - vars[1] (b-axis): larger upper bound (e.g., 128) → gets larger value
  // - Returns {value_for_vars[0], value_for_vars[1]}
  //
  // Example: if upper_bounds = {32, 128} and left = 64
  // - results[0] = min(64, 32) = 32
  // - results[1] = min(64, 128) = 64
  // - Returns {32, 64} where a-axis=32 (smaller), b-axis=64 (larger)
  std::vector<int64_t> results(count);
  for (size_t i = 0; i < count; ++i) {
    results[i] = std::min(left, upper_bounds[i]);
  }
  return {results[0], results[1]};
}

// Custom gcd implementation for C++11 compatibility
template<typename T>
inline T gcd(T a, T b) {
  while (b != 0) {
    T temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

// Custom lcm implementation for C++11 compatibility
template<typename T>
inline T lcm(T a, T b) {
  if (a == 0 || b == 0) return 0;
  return (a / gcd(a, b)) * b;
}

// Helper function to align value to upper bound
inline int64_t AlignToUpperBound(int64_t value, int64_t align) {
  return CeilDiv(value, align) * align;
}

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

// Initialize dual axes information
inline AxesReorderSolver::DualAxesInfo AxesReorderSolver::InitializeDualAxesInfo(
    TilingVariable **vars, uint32_t *axis_idx) const {
  int64_t upper_a = AF_AXES_EVAL_UPPER_BOUND(vars[0]);
  int64_t upper_b = AF_AXES_EVAL_UPPER_BOUND(vars[1]);
  bool a_is_first = (upper_a < upper_b);
  DualAxesInfo info{};
  info.var_a = a_is_first ? vars[0] : vars[1];
  info.var_b = a_is_first ? vars[1] : vars[0];
  info.idx_a = a_is_first ? axis_idx[0] : axis_idx[1];
  info.idx_b = a_is_first ? axis_idx[1] : axis_idx[0];
  info.upper_bound_a = AF_AXES_EVAL_UPPER_BOUND(info.var_a);
  info.upper_bound_b = AF_AXES_EVAL_UPPER_BOUND(info.var_b);
  info.upper_bound_both = std::min(info.upper_bound_a, info.upper_bound_b);
  info.low_bound_both = lcm(info.var_a->align, info.var_b->align);
  OP_LOGD(OP_NAME, "[DFX] Axes: a(ub=%ld, align=%ld), b(ub=%ld, align=%ld), low_bound_both=%ld",
          info.upper_bound_a, info.var_a->align, info.upper_bound_b, info.var_b->align, info.low_bound_both);
  return info;
}
// Execute Step 1: Find UB limit
inline bool AxesReorderSolver::ExecuteStep1_FindUBLimit(const DualAxesInfo &info,
    int64_t &low_bound_both_ub, int64_t &low_bound_b_ub, int64_t &range_a_high, int64_t &range_b_high) {
  info.var_a->SetValue(info.upper_bound_a);
  info.var_b->SetValue(info.upper_bound_b);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  int64_t low_bound_both = info.low_bound_both;
  bool step1_success = BinaryFindUpperBoundSatisfiedUBLimit(both_vars, both_idxs, kDualAxesCount, low_bound_both, low_bound_both_ub);
  if (!step1_success) {
    if (info.var_a->align == info.var_b->align) {
      OP_LOGW(OP_NAME, "[DFX] Step 1 failed, same align, return false");
      return false;
    }
    low_bound_both = std::min(info.var_a->align, info.var_b->align);
    step1_success = BinaryFindUpperBoundSatisfiedUBLimit(both_vars, both_idxs, kDualAxesCount, low_bound_both, low_bound_both_ub);
    if (!step1_success) {
      OP_LOGW(OP_NAME, "[DFX] Step 1 failed even with smaller lower_bound, return false");
      return false;
    }
  }

  OP_LOGD(OP_NAME, "[DFX] Step 1: low_bound_both_ub=%ld, upper_bound_both=%ld", low_bound_both_ub,
          info.upper_bound_both);
  low_bound_b_ub = info.upper_bound_b;
  if (low_bound_both_ub == info.upper_bound_both) {
    low_bound_b_ub = info.var_b->value;
    info.var_b->SetValue(info.upper_bound_b);
    TilingVariable* var_b_arr[] = {info.var_b};
    uint32_t idx_b_arr[] = {info.idx_b};
    BinaryFindUpperBoundSatisfiedUBLimit(var_b_arr, idx_b_arr, 1, info.upper_bound_a, low_bound_b_ub);
    OP_LOGD(OP_NAME, "[DFX] Step 1: UB sufficient, low_bound_b_ub=%ld", low_bound_b_ub);
  }
  range_a_high = low_bound_both_ub;
  range_b_high = (low_bound_both_ub == info.upper_bound_both) ? low_bound_b_ub : low_bound_both_ub;
  OP_LOGD(OP_NAME, "[DFX] Step 1 ranges: a_high=%ld, b_high=%ld", range_a_high, range_b_high);

  return true;
}
// Execute Step 2: Find UB threshold
inline bool AxesReorderSolver::ExecuteStep2_FindUBThreshold(const DualAxesInfo &info,
    int64_t range_a_low, int64_t range_a_high, int64_t &range_a_low_out, int64_t &range_b_low_out) {
  info.var_a->SetValue(range_a_high);
  info.var_b->SetValue(range_a_high);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  int64_t upper_bound_both_ub_thr = -1;
  bool step2_success =
      BinaryFindLowerBoundSatisfiedUBThresholdCond(both_vars, both_idxs, kDualAxesCount, range_a_low, upper_bound_both_ub_thr);
  if (!step2_success) {
    OP_LOGD(OP_NAME, "[DFX] Step 2 failed");
    return false;
  }

  range_a_low_out = range_a_low;
  range_b_low_out = range_a_low;
  if (upper_bound_both_ub_thr > range_a_low) {
    range_a_low_out = range_b_low_out = upper_bound_both_ub_thr;
  }
  OP_LOGD(OP_NAME, "[DFX] Step 2: ub_thr=%ld, ranges: a_low=%ld, b_low=%ld",
          upper_bound_both_ub_thr, range_a_low_out, range_b_low_out);
  return true;
}
// Execute Step 3: Find tile size satisfying core number constraint
inline bool AxesReorderSolver::ExecuteStep3_FindCoreNumTileSize(const DualAxesInfo &info,
    int64_t range_a_low, int64_t range_a_high, int64_t range_b_high, TilingVariable **vars_final) {
  info.var_a->SetValue(range_a_high);
  info.var_b->SetValue(range_a_high);
  TilingVariable* both_vars[] = {info.var_a, info.var_b};
  uint32_t both_idxs[] = {info.idx_a, info.idx_b};
  // Define separate upper bounds for each variable to enable non-symmetric solutions
  int64_t upper_bounds[kDualAxesCount] = {range_a_high, range_b_high};
  OP_LOGD(OP_NAME, "[DFX] Step 3: Using per-variable upper bounds: a=%ld, b=%ld",
          upper_bounds[0], upper_bounds[1]);

  auto lower_left = BinaryFindLowerBoundSatisfiedCoreNum(both_vars, both_idxs, kDualAxesCount, range_a_low, upper_bounds);
  auto lower_left_1 = lower_left.first;
  auto lower_left_2 = lower_left.second;
  OP_LOGD(OP_NAME, "[DFX] Step 3: lower_left_1=%ld, lower_left_2=%ld", lower_left_1, lower_left_2);

  if (lower_left_1 < 0) {
    info.var_a->SetValue(range_a_high);
    TilingVariable *var_b_arr[] = {info.var_b};
    uint32_t idx_b_arr[] = {info.idx_b};
    auto lower_left = BinaryFindLowerBoundSatisfiedCoreNum(var_b_arr, idx_b_arr, 1, range_b_high);
    auto lower_left_b1 = lower_left.first;
    auto lower_left_b2 = lower_left.second;
    if (lower_left_b1 < 0) {
      OP_LOGW(OP_NAME, "[DFX] Step 3 failed for b axis, return false");
      return false;
    }
    info.var_b->SetValue((lower_left_b1 >= lower_left_b2) ? lower_left_b1 : lower_left_b2);
    OP_LOGD(OP_NAME, "[DFX] Step 3 fallback: a=%ld, b=%ld", range_a_high, info.var_b->value);
    vars_final[0] = info.var_a;
    vars_final[1] = info.var_b;
    return true;
  }
  info.var_a->SetValue(lower_left_1);
  info.var_b->SetValue(lower_left_2);
  OP_LOGD(OP_NAME, "[DFX] Step 3 success: a=%ld, b=%ld", lower_left_1, lower_left_2);
  vars_final[0] = info.var_a;
  vars_final[1] = info.var_b;
  return true;
}

// Orchestration function for dual multi-core axes processing
bool AxesReorderSolver::ProcessDualMCAxes(TilingVariable **vars, uint32_t *axis_idx, std::vector<bool> &solved_axes) {
  OP_LOGD(OP_NAME, "[DFX] ProcessDualMCAxes begin, input=%s", input_.DebugString().c_str());
  // Step 0: Initialize dual axes information
  DualAxesInfo info = InitializeDualAxesInfo(vars, axis_idx);
  // Step 1: Find UB limit
  int64_t low_bound_both_ub, low_bound_b_ub, range_a_high, range_b_high;
  if (!ExecuteStep1_FindUBLimit(info, low_bound_both_ub, low_bound_b_ub, range_a_high, range_b_high)) {
    return false;
  }
  // Step 2: Find UB threshold
  int64_t range_a_low = info.low_bound_both;
  int64_t range_b_low = info.low_bound_both;
  if (!ExecuteStep2_FindUBThreshold(info, range_a_low, range_a_high, range_a_low, range_b_low)) {
    info.var_a->SetValue(range_a_high);
    info.var_b->SetValue(range_a_high);
    return FinalizeEqualPriorityAxes(axis_idx, solved_axes);
  }
  // Step 3: Find tile size satisfying core number constraint
  TilingVariable* vars_final[kDualAxesCount];
  if (!ExecuteStep3_FindCoreNumTileSize(info, range_a_low, range_a_high, range_b_high, vars_final)) {
    return false;
  }
  return FinalizeEqualPriorityAxes(axis_idx, solved_axes);
}
// Helper method to process multi-core related axis
bool AxesReorderSolver::ProcessMCAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num) {
  int64_t upper_bound_satisfied_ub_threshold = var->value;
  int64_t lower_bound_satisfied_ub_threshold = var->align;
  // If no solution satisfies UB utilization, no further processing needed
  if (BinaryFindLowerBoundSatisfiedUBThresholdCond(var, var_idx, var->align, lower_bound_satisfied_ub_threshold)) {
    OP_LOGD(OP_NAME, "Found lower_bound_satisfied_ub_threshold:%ld, upper:%ld, lower:%ld, i:%u, input: %s",
            lower_bound_satisfied_ub_threshold, upper_bound_satisfied_ub_threshold, var->align, var_idx,
            input_.DebugString().c_str());
    var->SetValue(upper_bound_satisfied_ub_threshold);
    // 3.1) Binary search within [var->align, upper_bound_satisfied_ub_threshold] for UB utilization boundary
    auto satisfied_core_threshold = BinaryFindLowerBoundSatisfiedCoreNum(var, var_idx,
        lower_bound_satisfied_ub_threshold);
    auto satisfied_core_threshold_left = satisfied_core_threshold.first;
    auto satisfied_core_threshold_right = satisfied_core_threshold.second;
    OP_LOGD(OP_NAME, "Found lower bound satisfied core num:%ld, %ld, var upper:%ld, lower:%ld, i:%u, input: %s",
            satisfied_core_threshold_left, satisfied_core_threshold_right, upper_bound_satisfied_ub_threshold,
            var->align, var_idx, input_.DebugString().c_str());

    // 3.2) Try larger tile size first, update var if solution exists
    int64_t available_core_num_right = 0L;
    var->SetValue(satisfied_core_threshold_right);
    if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_right)) {
      OP_LOGD(OP_NAME, "Found larger tile size:%ld, available_core_num:%ld, max_core_num:%ld, i:%u",
              satisfied_core_threshold_right, available_core_num_right, max_core_num, var_idx);
    }

    int64_t available_core_num_left = 0L;
    if ((satisfied_core_threshold_left != satisfied_core_threshold_right) && (satisfied_core_threshold_left > 0L)) {
      var->SetValue(satisfied_core_threshold_left);
      if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_left) &&
          (available_core_num_left > available_core_num_right)) {
        OP_LOGD(OP_NAME, "Found smaller tile size:%ld, available_core_num:%ld, max_core_num:%ld, i:%u",
                satisfied_core_threshold_left, available_core_num_left, max_core_num, var_idx);
      } else {
        var->SetValue(satisfied_core_threshold_right);
      }
    }
  }
  return true;
}
// Helper method to process single axis
bool AxesReorderSolver::ProcessSingleAxisNaive(TilingVariable *var, uint32_t var_idx, int64_t max_core_num) {
  auto upper_bound = AF_AXES_EVAL_UPPER_BOUND(var);
  int64_t boundary = (upper_bound / var->align) * var->align;
  if (boundary < var->align) {
    OP_LOGW(OP_NAME, "Invalid aligned upper bound:%ld, raw upper:%ld, align:%ld, i:%u, input: %s.",
            boundary, upper_bound, var->align, var_idx, input_.DebugString().c_str());
    return false;
  }
  var->SetValue(boundary);

  int64_t upper_bound_satisfied_ub = -1L;
  if (!BinaryFindUpperBoundSatisfiedUBLimit(var, var->align, upper_bound_satisfied_ub)) {
    OP_LOGW(OP_NAME, "BinaryFindUpperBoundSatisfiedUBLimit failed, upper:%ld, lower:%ld, i:%u, input: %s.",
            upper_bound, var->align, var_idx, input_.DebugString().c_str());
    return false;
  }

  OP_LOGD(OP_NAME, "Found upper_bound_satisfied_ub:%ld, upper_bound:%ld, lower_bound:%ld, i:%u, input: %s",
          upper_bound_satisfied_ub, boundary, var->align, var_idx, input_.DebugString().c_str());
  var->SetValue(upper_bound_satisfied_ub);

  // Binary search for UB utilization values for multi-core related axis
  if (var->mc_related) {
    if (!ProcessMCAxisNaive(var, var_idx, max_core_num)) {
      return false;
    }
  }

  OP_LOGD(OP_NAME, "After local buffer tiling, input:%s", input_.DebugString().c_str());
  if (!TuneNotailVar(var)) {
    OP_LOGW(OP_NAME, "Tune notail var failed");
    return false;
  }

  while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value != var->align) {
    var->value -= var->align;
    if (!TuneNotailVar(var)) {
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
  }

  ApplyPromptAlign(var);
  return true;
}

bool AxesReorderSolver::BinarySearchWithAlignment(TilingVariable **vars, uint32_t var_num, int64_t lower_bound,
                                                  int64_t upper_bound, int64_t align, const char *log_prefix) {
  if (lower_bound > upper_bound) {
    OP_LOGW(OP_NAME, "Invalid search range: lower_bound=%ld > upper_bound=%ld", lower_bound, upper_bound);
    return false;
  }
  int64_t left = lower_bound;
  int64_t right = upper_bound;
  int64_t result = -1;
  while (left <= right) {
    int64_t mid = left + (right - left) / kMidpointDivisor;
    mid = CeilDiv(mid, align) * align;
    for (uint32_t i = 0; i < var_num; ++i) {
      vars[i]->SetValue(mid);
    }

    bool satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);

    OP_LOGD(OP_NAME, "%s: left=%ld, right=%ld, mid=%ld, satisfied=%d",
            log_prefix, left, right, mid, satisfied);

    if (satisfied) {
      result = mid;
      left = mid + align;
    } else {
      right = mid - align;
    }
  }
  if (result != -1) {
    for (uint32_t i = 0; i < var_num; ++i) {
      vars[i]->SetValue(result);
    }
  }
  return result != -1;
}

bool AxesReorderSolver::TuneNoTail(TilingVariable **vars, uint32_t id) {
  if (!TuneNotailVar(vars[id])) {
    OP_LOGW(OP_NAME, "Tune notail var failed for axis_%d, falling back to original algorithm", id);
    return false;
  }
  return DecreaseUntilSatisfied(vars[id], ConstraintType::LB_MIXED, [this](TilingVariable* v) { return TuneNotailVar(v); });
}

bool AxesReorderSolver::IdentifyEqualPriorityAxes(const uint32_t axes_num, uint32_t *axis_idx, int64_t &min_upper_bound,
                                                  int64_t &max_aligned) {
  std::map<size_t, std::vector<uint32_t>> order_to_axis_indices;

  for (uint32_t i = 0; i < input_.local_buffer_vars_size; ++i) {
    auto &var = input_.local_buffer_vars[i];
    order_to_axis_indices[var->order].push_back(i);
  }
  bool need_equal_priority = false;
  uint32_t index = 0;
  constexpr uint32_t kSupportMaxEqualPriorityAxes = 2;
  for (const auto &pair : order_to_axis_indices) {
    if (pair.second.size() < kSupportMaxEqualPriorityAxes) {
      continue;
    }
    for (const auto &id : pair.second) {
      if (index >= axes_num) {
        OP_LOGI(OP_NAME, "Axes num %u is not enough to identify equal priority axes, index=%u", axes_num, index);
        break;
      }
      axis_idx[index++] = id;
      auto &var = input_.local_buffer_vars[id];
      int64_t upper = AF_AXES_EVAL_UPPER_BOUND(var);
      upper = CeilDiv(upper, var->align) * var->align;
      min_upper_bound = std::min(upper, min_upper_bound);
      max_aligned = std::max(var->align, max_aligned);
      OP_LOGI(OP_NAME, "Identified equal priority axes: axis_idx=%u, min_upper_bound=%ld, max_aligned=%ld",
              id, min_upper_bound, max_aligned);
      if (index >= kSupportMaxEqualPriorityAxes) {
        need_equal_priority = true;
        break;
      }
    }
    return need_equal_priority;
  }
  return false;
}

bool AxesReorderSolver::BinarySearchEqualPriorityAxes(uint32_t axis_num, uint32_t *axis_idx, int64_t lower_bound,
                                                      int64_t upper_bound) {
  TilingVariable *var[axis_num];
  for (uint32_t i = 0; i < axis_num; ++i) {
    var[i] = input_.local_buffer_vars[axis_idx[i]];
  }

  bool found = BinarySearchWithAlignment(var, axis_num, lower_bound, upper_bound, lower_bound,
                                         "Binary search");
  if (found) {
    OP_LOGI(OP_NAME, "Found solution for equal priority axes in range [%ld, %ld]", lower_bound, upper_bound);
  } else {
    OP_LOGW(OP_NAME, "No solution found for equal priority axes in range [%ld, %ld]", lower_bound, upper_bound);
  }

  return found;
}

bool AxesReorderSolver::IterativeSolveEqualPriorityAxes(const uint32_t axis_num, uint32_t *axis_idx,
                                                        int64_t lower_bound, int64_t upper_bound) {
  if (!BinarySearchEqualPriorityAxes(axis_num, axis_idx, lower_bound, upper_bound)) {
    return false;
  }
  for (uint32_t i = 0u; i < axis_num; ++i) {
    auto &var = input_.local_buffer_vars[axis_idx[i]];
    int64_t limited_upper_bound = var->value;
    if (AF_AXES_EVAL_UPPER_BOUND(var) <= limited_upper_bound) {
      continue;
    }
    var->SetValue(limited_upper_bound);
    int64_t var_upper_bound = AF_AXES_EVAL_UPPER_BOUND(var);
    var_upper_bound = CeilDiv(var_upper_bound, var->align) * var->align;
    TilingVariable *var_ptr[] = {var};
    if (BinarySearchWithAlignment(var_ptr, 1, limited_upper_bound, var_upper_bound, var->align,
                                  "Iterative solve")) {
      OP_LOGI(OP_NAME, "Successfully solved equal priority axes: limited_upper_bound=%ld, var_upper_bound=%ld", limited_upper_bound, var_upper_bound);
    } else {
      OP_LOGW(OP_NAME, "Failed to solve axis_b in range [%ld, %ld]", limited_upper_bound, var_upper_bound);
      return false;
    }
  }
  return true;
}

bool AxesReorderSolver::SolveEqualPriorityAxesWithDualThreshold(
    uint32_t *axis_idx, int64_t low_bound, int64_t upper_bound,
    std::vector<bool> &solved_axes) {
  constexpr uint32_t kSupportMaxEqualPriorityAxes = 2;
  TilingVariable *vars[kSupportMaxEqualPriorityAxes];
  for (uint32_t i = 0; i < kSupportMaxEqualPriorityAxes; ++i) {
    vars[i] = input_.local_buffer_vars[axis_idx[i]];
  }

  bool mc_related_a = vars[0]->mc_related;
  bool mc_related_b = vars[1]->mc_related;
  // 如果两个轴都是 mc_related，直接使用 ProcessDualMCAxes 进行对称切分
  if (mc_related_a && mc_related_b) {
    return ProcessDualMCAxes(vars, axis_idx, solved_axes);
  }
  // 其他情况：先尝试使用 BinarySearchEqualPriorityAxesWithDualThreshold
  int64_t upper_ub_a;
  int64_t upper_ub_b;
  if (!BinarySearchEqualPriorityAxesWithDualThreshold(vars, low_bound,
                                                      upper_bound, upper_ub_a, upper_ub_b)) {
    OP_LOGW(OP_NAME, "BinarySearchEqualPriorityAxesWithDualThreshold failed");
    return false;
  }
  vars[0]->SetValue(upper_ub_a);
  vars[1]->SetValue(upper_ub_b);
  if (!mc_related_a && !mc_related_b) {
    return ProcessNonMCAxes(vars, axis_idx, solved_axes);
  }
  return ProcessSingleMCAxis(vars, mc_related_a, axis_idx, solved_axes);
}

bool AxesReorderSolver::BinarySearchEqualPriorityAxesWithDualThreshold(
    TilingVariable **vars, int64_t lower_bound, int64_t upper_bound,
    int64_t &upper_ub_a, int64_t &upper_ub_b) {
  int64_t left = lower_bound;
  int64_t right = upper_bound;
  int64_t result = -1;
  int64_t max_core_num = static_cast<int64_t>(input_.corenum_threshold * static_cast<double>(input_.core_num));

  OP_LOGD(OP_NAME, "[DFX] BinarySearchEqualPriorityAxesWithDualThreshold: range=[%ld, %ld], min_ub_threshold=%ld",
          lower_bound, upper_bound, static_cast<int64_t>(input_.ub_size * input_.ub_threshold));

  while (left <= right) {
    int64_t mid = left + (right - left) / kMidpointDivisor;
    mid = CeilDiv(mid, vars[0]->align) * vars[0]->align;
    vars[0]->SetValue(mid);
    vars[1]->SetValue(mid);
    // 计算block_dim（使用CalRealUsedCoreNum）
    int64_t block_dim = 0;
    if (!AF_AXES_CAL_REAL_USED_CORE_NUM(block_dim)) {
      right = mid - vars[0]->align;
      continue;
    }
    // 检查UB是否满足（使用IsSatisfyCons）
    bool satisfied_ub = SatisfyCons(ConstraintType::LOCAL_BUFFER);
    // 1. block_dim >= max_core_num：核数利用率满足阈值
    // 2. UB满足：UB利用率满足要求
    bool satisfies_test = (block_dim >= max_core_num) || satisfied_ub;
    bool satisfied = SatisfyCons(ConstraintType::LOCAL_BUFFER);
    if (satisfied && satisfies_test) {
      result = mid;
      left = mid + vars[0]->align;
    } else {
      right = mid - vars[0]->align;
    }
  }
  if (result == -1) {
    return false;
  }
  vars[0]->SetValue(result);
  vars[1]->SetValue(result);
  upper_ub_a = result;
  upper_ub_b = result;

  int64_t var_b_upper_bound = AF_AXES_EVAL_UPPER_BOUND(vars[1]);
  var_b_upper_bound = CeilDiv(var_b_upper_bound, vars[1]->align) * vars[1]->align;
  if (var_b_upper_bound > result) {
    TilingVariable *var_ptr[] = {vars[1]};
    if (!BinarySearchWithAlignment(var_ptr, 1, result, var_b_upper_bound,
                                   vars[1]->align, "B axis search")) {
      return false;
    }
    upper_ub_b = vars[1]->value;
  }
  return true;
}

bool AxesReorderSolver::ProcessNonMCAxes(
    TilingVariable **vars, uint32_t *axis_idx,
    std::vector<bool> &solved_axes) {
  for (uint32_t i = 0; i < kDualAxesCount; ++i) {
    if (!TuneNotailVar(vars[i])) {
      OP_LOGW(OP_NAME, "Tune notail var failed for axis %u", axis_idx[i]);
      return false;
    }
    if (!DecreaseUntilSatisfied(vars[i], ConstraintType::LB_MIXED,
        [this](TilingVariable* v) { return TuneNotailVar(v); })) {
      return false;
    }
    ApplyPromptAlign(vars[i]);
    solved_axes[axis_idx[i]] = true;
  }
  return true;
}

bool AxesReorderSolver::ProcessSingleMCAxis(
    TilingVariable **vars, bool first_is_mc, uint32_t *axis_idx,
    std::vector<bool> &solved_axes) {
  TilingVariable *mc_var = first_is_mc ? vars[0] : vars[1];
  TilingVariable *non_mc_var = first_is_mc ? vars[1] : vars[0];
  uint32_t mc_idx = first_is_mc ? axis_idx[0] : axis_idx[1];
  uint32_t non_mc_idx = first_is_mc ? axis_idx[1] : axis_idx[0];
  // 非多核相关轴直接处理
  if (!TuneNotailVar(non_mc_var)) {
    return false;
  }
  if (!DecreaseUntilSatisfied(non_mc_var, ConstraintType::LB_MIXED,
                              [this](TilingVariable* v) { return TuneNotailVar(v); })) {
    return false;
  }
  ApplyPromptAlign(non_mc_var);
  solved_axes[non_mc_idx] = true;
  // 多核相关轴使用双阈值调优
  int64_t upper_bound_satisfied_ub_threshold = mc_var->value;
  int64_t lower_bound_satisfied_ub_threshold = mc_var->align;
  if (BinaryFindLowerBoundSatisfiedUBThresholdCond(mc_var, mc_idx, mc_var->align, lower_bound_satisfied_ub_threshold)) {
    mc_var->SetValue(upper_bound_satisfied_ub_threshold);
    auto satisfied_core_threshold = BinaryFindLowerBoundSatisfiedCoreNum(
        mc_var, mc_idx, lower_bound_satisfied_ub_threshold);
    int64_t available_core_num_right = 0L;
    mc_var->SetValue(satisfied_core_threshold.second);
    if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_right)) {
      // 成功
    }
    int64_t available_core_num_left = 0L;
    if ((satisfied_core_threshold.first != satisfied_core_threshold.second) && (satisfied_core_threshold.first > 0L)) {
      mc_var->SetValue(satisfied_core_threshold.first);
      if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_left) &&
          (available_core_num_left > available_core_num_right)) {
        // 保留较小值
      } else {
        mc_var->SetValue(satisfied_core_threshold.second);
      }
    }
  }
  if (!TuneNotailVar(mc_var)) {
    return false;
  }
  if (!DecreaseUntilSatisfied(mc_var, ConstraintType::LB_MIXED,
                              [this](TilingVariable* v) { return TuneNotailVar(v); })) {
    return false;
  }
  ApplyPromptAlign(mc_var);
  solved_axes[mc_idx] = true;
  return true;
}

bool AxesReorderSolver::FinalizeEqualPriorityAxes(uint32_t *axis_idx, std::vector<bool> &solved_axes) const {
  constexpr uint32_t kSupportMaxEqualPriorityAxes = 2;
  // 对于同优先级轴，需要保持对称性
  // 不进行 TuneNotailVar 和 DecreaseUntilSatisfied 操作，直接使用 ProcessDualMCAxes 中设置的对称值
  for (uint32_t i = 0; i < kSupportMaxEqualPriorityAxes; ++i) {
    solved_axes[axis_idx[i]] = true;
  }
  // 检查两个轴的值是否接近（对称性）
  auto &var0 = input_.local_buffer_vars[axis_idx[0]];
  auto &var1 = input_.local_buffer_vars[axis_idx[1]];
  OP_LOGD(OP_NAME, "Equal priority axes final values: axis[%u]=%ld, axis[%u]=%ld, diff=%ld", axis_idx[0], var0->value,
          axis_idx[1], var1->value,
          (var0->value > var1->value) ? (var0->value - var1->value) : (var1->value - var0->value));
  return true;
}

#else
#endif
bool AxesReorderSolver::BinaryLocalBufTilingCore(const std::vector<bool> &solved_axes) {
  for (uint32_t ordered_idx = 0U; ordered_idx < input_.local_buffer_vars_size; ++ordered_idx) {
    const uint32_t canonical_idx = input_.ordered_to_canonical[ordered_idx];
    if (solved_axes[canonical_idx]) {
      continue;
    }
    auto &var = input_.ordered_local_buffer_vars[ordered_idx];
    auto upper_bound = AF_AXES_EVAL_UPPER_BOUND(var);
    int64_t boundary = (upper_bound / var->align) * var->align;
    int64_t init_val = var->value;
    var->SetValue(boundary);
    if (!SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
      var->SetValue(init_val);
      ShrinkBoundaryUntilSatisfied(var, boundary, ConstraintType::LOCAL_BUFFER);
    }
    if (!TuneNotailVar(var)) {
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
    if (!DecreaseUntilSatisfied(var, ConstraintType::LB_MIXED, [this](TilingVariable* v) { return TuneNotailVar(v); })) {
      return false;
    }
    ApplyPromptAlign(var);
  }
  return true;
}
#if AF_AXES_EQUAL_ORDER
bool AxesReorderSolver::NaiveLocalBufTiling(const bool enable_equal_order) {
  // 初始化和验证
#else
bool AxesReorderSolver::NaiveLocalBufTiling() {
#endif
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }

#if AF_AXES_EQUAL_ORDER
  // 处理同优先级轴
  std::vector<bool> solved_axes(input_.local_buffer_vars_size, false);
  if (enable_equal_order) {
    constexpr uint32_t kMaxAxes = 2;
    uint32_t axis_idx[kMaxAxes]{};
    int64_t min_ub = std::numeric_limits<int64_t>::max();
    int64_t max_align = 1L;
#else
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.ordered_local_buffer_vars;
#endif

#if AF_AXES_EQUAL_ORDER
    if (IdentifyEqualPriorityAxes(kMaxAxes, axis_idx, min_ub, max_align)) {
      if (!SolveEqualPriorityAxesWithDualThreshold(
              axis_idx, max_align, min_ub, solved_axes)) {
        return NaiveLocalBufTiling(false);  // 回退到非同优先级模式
      }
    }
  }
  // 处理剩余的轴
  int64_t max_core_num = static_cast<int64_t>(input_.corenum_threshold * input_.core_num);
  for (uint32_t ordered_idx = 0U; ordered_idx < input_.local_buffer_vars_size; ++ordered_idx) {
#else
  for (uint32_t ordered_idx = 0U; ordered_idx < num_vars; ++ordered_idx) {
    auto &var = vars[ordered_idx];
#endif
    const uint32_t canonical_idx = input_.ordered_to_canonical[ordered_idx];
#if AF_AXES_EQUAL_ORDER
    if (solved_axes[canonical_idx]) {
      continue;
    }
    auto *var = input_.ordered_local_buffer_vars[ordered_idx];
    if (!ProcessSingleAxisNaive(var, canonical_idx, max_core_num)) {
#else
    auto upper_bound = AF_AXES_EVAL_UPPER_BOUND(var);
    int64_t boundary = (upper_bound / var->align) * var->align;
    if (boundary < var->align) {
      OP_LOGW(OP_NAME, "Invalid aligned upper bound:%ld, raw upper:%ld, align:%ld, i:%u, input: %s.",
              boundary, upper_bound, var->align, canonical_idx, input_.DebugString().c_str());
#endif
      return false;
    }
#if AF_AXES_EQUAL_ORDER
#else
    var->SetValue(boundary);
    int64_t upper_bound_satisfied_ub = -1L;
    if (!BinaryFindUpperBoundSatisfiedUBLimit(var, var->align, upper_bound_satisfied_ub)) {
      OP_LOGW(OP_NAME, "BinaryFindUpperBoundSatisfiedUBLimit failed, upper:%ld, lower:%ld, i:%u, input: %s.",
              upper_bound, var->align, canonical_idx, input_.DebugString().c_str());
      return false;
    }
    OP_LOGD(OP_NAME, "Found upper_bound_satisfied_ub:%ld, upper_bound:%ld, lower_bound:%ld, i:%u, input: %s",
            upper_bound_satisfied_ub, boundary, var->align, canonical_idx, input_.DebugString().c_str());
    var->SetValue(upper_bound_satisfied_ub);

    // 2.1)针对多核相关轴进行二分查找满足UB利用率的值，找到可以取的最小Tile切分大小
    // 2.2)若非多核相关轴，则按照能够满足UB的最大约束返回(upper_bound_satisfied_ub)
    if (var->mc_related) {
      int64_t upper_bound_satisfied_ub_threshold = var->value;
      int64_t lower_bound_satisfied_ub_threshold = var->align;
      // 若未找到满足UB利用率的解，则不需要进一步处理
      if (BinaryFindLowerBoundSatisfiedUBThresholdCond(var, canonical_idx, var->align,
                                                       lower_bound_satisfied_ub_threshold)) {
        OP_LOGD(OP_NAME, "Found lower_bound_satisfied_ub_threshold:%ld, upper:%ld, lower:%ld, i:%u, input: %s",
                lower_bound_satisfied_ub_threshold, upper_bound_satisfied_ub_threshold, var->align, canonical_idx,
                input_.DebugString().c_str());
        var->SetValue(upper_bound_satisfied_ub_threshold);
        // 3.1) 在[var->align,upper_bound_satisfied_ub_threshold]范围内，二分查找满足UB利用率的边界
        auto satisfied_core_threshold = BinaryFindLowerBoundSatisfiedCoreNum(var, canonical_idx,
            lower_bound_satisfied_ub_threshold);
        auto satisfied_core_threshold_left = satisfied_core_threshold.first;
        auto satisfied_core_threshold_right = satisfied_core_threshold.second;
        OP_LOGD(OP_NAME, "Found lower bound satisfied core num:%ld, %ld, var upper:%ld, lower:%ld, i:%u, input: %s",
                satisfied_core_threshold_left, satisfied_core_threshold_right, upper_bound_satisfied_ub_threshold,
                var->align, canonical_idx, input_.DebugString().c_str());
        // 3.2)先尝试Tile块更大的值，若有解，则更新var
        int64_t available_core_num_right = 0L;
        var->SetValue(satisfied_core_threshold_right);
        if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_right)) {
          OP_LOGD(OP_NAME, "Found larger tile size:%ld, available_core_num:%ld, i:%u",
                  satisfied_core_threshold_right, available_core_num_right, canonical_idx);
        }
        int64_t available_core_num_left = 0L;
        if ((satisfied_core_threshold_left != satisfied_core_threshold_right) && (satisfied_core_threshold_left > 0L)) {
          var->SetValue(satisfied_core_threshold_left);
          if (InitMulticoreVars() && MulticoreTilingCore(false) && AF_AXES_CAL_REAL_USED_CORE_NUM(available_core_num_left) &&
              (available_core_num_left > available_core_num_right)) {
            OP_LOGD(OP_NAME, "Found smaller tile size:%ld, available_core_num:%ld, i:%u",
                    satisfied_core_threshold_left, available_core_num_left, canonical_idx);
          } else {
            var->SetValue(satisfied_core_threshold_right);
          }
        }
      }
    }

    OP_LOGD(OP_NAME, "After local buffer tiling, input:%s", input_.DebugString().c_str());
    if (!TuneNotailVar(var)) {
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
    while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value!= var->align) {
      var->value -= var->align;
      if (!TuneNotailVar(var)) {
        OP_LOGW(OP_NAME, "Tune notail var failed");
        return false;
      }
    }
    ApplyPromptAlign(var);
#endif
  }
#if AF_AXES_EQUAL_ORDER
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return enable_equal_order && NaiveLocalBufTiling(false);
  }
  return SatisfyCons(ConstraintType::LB_MIXED);
#else
  return SatisfyCons(ConstraintType::LOCAL_BUFFER) && SatisfyCons(ConstraintType::LB_MIXED);
#endif
}
#if AF_AXES_EQUAL_ORDER
bool AxesReorderSolver::BinaryLocalBufTiling(const bool enable_equal_order) {
#else
bool AxesReorderSolver::BinaryLocalBufTiling() {
#endif
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }

  std::vector<bool> solved_axes(input_.local_buffer_vars_size, false);
#if AF_AXES_EQUAL_ORDER
  // 相等优先级轴的逻辑，仅开关使能后生效
  constexpr uint32_t kSupportMaxEqualPriorityAxes = 2;
  uint32_t axis_idx[kSupportMaxEqualPriorityAxes]{0};
  int64_t min_upper_bound = std::numeric_limits<int64_t>::max();
  int64_t max_aligned = 1L;
  bool equal_axes_tiling_success = false;
  // 可在编译时确定axis_id，待优化
  if (enable_equal_order &&
      IdentifyEqualPriorityAxes(kSupportMaxEqualPriorityAxes, axis_idx, min_upper_bound, max_aligned)) {
    OP_LOGD(OP_NAME,
            "Attempting to solve equal priority axes with new algorithm, input=[%s], "
            "min_upper_bound=%ld, max_aligned=%ld",
            input_.DebugString().c_str(), min_upper_bound, max_aligned);
    TilingVariable *vars[kSupportMaxEqualPriorityAxes]{};
    for (uint32_t id = 0U; id < kSupportMaxEqualPriorityAxes; ++id) {
      vars[id] = input_.local_buffer_vars[axis_idx[id]];
    }
    if (IterativeSolveEqualPriorityAxes(kSupportMaxEqualPriorityAxes, axis_idx, max_aligned, min_upper_bound)) {
      // 检查是否需要fail back
      for (uint32_t id = 0U; id < kSupportMaxEqualPriorityAxes; id++) {
        if (!TuneNoTail(vars, id)) {
          return BinaryLocalBufTiling(false);
        }
        ApplyPromptAlign(vars[id]);
      }
      if (!SatisfyCons(ConstraintType::LB_MIXED)) {
        OP_LOGI(OP_NAME, "Equal priority axes solution failed final check, falling back to original algorithm");
        return BinaryLocalBufTiling(false);
      }
      OP_LOGI(OP_NAME,
              "Successfully solved equal priority axes with new algorithm, "
              "input=[%s], min_upper_bound=%ld, max_aligned=%ld",
              input_.DebugString().c_str(), min_upper_bound, max_aligned);
      for (uint32_t id = 0U; id < kSupportMaxEqualPriorityAxes; ++id) {
        solved_axes[axis_idx[id]] = true;
      }
      equal_axes_tiling_success = true;
    }
  }
#else

#endif
  if (!BinaryLocalBufTilingCore(solved_axes)) {
    OP_LOGW(OP_NAME, "Binary local Tiling Calculation failed in final check.");
    return false;
  }

  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
#if AF_AXES_EQUAL_ORDER
    return enable_equal_order && BinaryLocalBufTiling(false);
#else
    return false;
#endif
  }

  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Binary local Tiling Calculation failed in final check.");
    return false;
  }
  return true;
}

#if AF_AXES_EQUAL_ORDER
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune,
                                       const bool enable_equal_order) {
#else
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune) {
#endif
  (void)block_loop_auto_tune;
  if (is_tuning) {
#if AF_AXES_EQUAL_ORDER
    return NaiveLocalBufTiling(enable_equal_order);
#else
    return NaiveLocalBufTiling();
#endif
  } else {
#if AF_AXES_EQUAL_ORDER
    return BinaryLocalBufTiling(enable_equal_order);
#else
    return BinaryLocalBufTiling();
#endif
  }
}

inline bool AxesReorderSolver::GetTiling(const bool is_tuning,
                                         const bool block_loop_auto_tune,
#if AF_AXES_EQUAL_ORDER
                                         const bool enable_workload_balance,
                                         const bool enable_equal_order) {
  if (!LocalBufTiling(is_tuning, block_loop_auto_tune, enable_equal_order)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed, is_tuning: %d, block_loop_auto_tune: %d, enable_equal_order: %d,"
    "input: %s", is_tuning, block_loop_auto_tune, enable_equal_order, input_.DebugString().c_str());
#else
                                         const bool enable_workload_balance) {
  if (!LocalBufTiling(is_tuning, block_loop_auto_tune)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed, is_tuning: %d, block_loop_auto_tune: %d,"
    "input: %s", is_tuning, block_loop_auto_tune, input_.DebugString().c_str());
#endif
    return false;
  }

  OP_LOGI(OP_NAME, "local buffer tiling success, input: %s", input_.DebugString().c_str());
  if (!MulticoreTiling(block_loop_auto_tune, enable_workload_balance)) {
    OP_LOGW(OP_NAME, "multicore tiling failed, input: %s", input_.DebugString().c_str());
    return false;
  }
  OP_LOGI(OP_NAME, "multicore tiling success, input: %s", input_.DebugString().c_str());
  return true;
  }

#if AF_AXES_EQUAL_ORDER
inline bool AxesReorderSolver::GetMaxBlockDimTiling(const uint32_t block_dim, const bool enable_equal_order) {
#else
inline bool AxesReorderSolver::GetMaxBlockDimTiling(const uint32_t block_dim) {
#endif
  auto save_block_dim = input_.core_num;
  input_.core_num = block_dim;
  input_.ub_threshold = 0.0;
  input_.corenum_threshold = 1;
  // 按照约束规则求解不需要开启负载均衡
#if AF_AXES_EQUAL_ORDER
  auto ret = GetTiling(true, true, false, enable_equal_order);
#else
  auto ret = GetTiling(true, true, false);
#endif
  input_.core_num = save_block_dim;
  return ret;
}

inline bool AxesReorderSolver::FindNextUpperBlockDim(const uint32_t block_dim, uint32_t &next_lower_block_dim) const {
  // 当前核数已达到或超过上限，无更大档位
  if (block_dim >= input_.core_num) {
    return false;
  }
  // 确定当前区间的间隔
  uint32_t step;
  if (block_dim <= 4U) {
    step = 1U;
  } else if (block_dim < 16U) {
    step = 2U;
  } else {
    step = 4U;
  }
  // 计算理论上下一个档位
  uint32_t candidate = block_dim + step;
  // 若候选值超过上限，则直接取上限作为下一档位
  if (candidate > input_.core_num) {
    next_lower_block_dim = input_.core_num;
  } else {
    next_lower_block_dim = candidate;
  }
  return true;
}

inline bool AxesReorderSolver::FindNextLowerBlockDim(const uint32_t block_dim, uint32_t &next_upper_block_dim) const {
  // 核数必须大于1才可能有上一个档位（至少为1）
  if (block_dim <= 1U) {
    return false;
  }
  // 确定当前区间的间隔（与上限判断逻辑一致）
  uint32_t step;
  if (block_dim <= 4U) {
    step = 1;
  } else if (block_dim < 16U) {
    step = 2U;
  } else {
    step = 4U;
  }
  // 计算上一个档位（确保结果至少为1）
  uint32_t candidate = block_dim - step;
  next_upper_block_dim = std::max(candidate, 1u); // 避免核数为0
  return true;
}

inline void AxesReorderSolver::SaveInputTilingVars(TilingVariable *tiling_vars,
                                                   TilingVariable *pure_mc_vars,
                                                   TilingVariable *local_buffer_vars) const {
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
      tiling_vars[i] = *(input_.tiling_vars[i]);
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
      pure_mc_vars[i] = *(input_.pure_mc_vars[i]);
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
      local_buffer_vars[i] = *(input_.local_buffer_vars[i]);
    }
  }
}

inline void AxesReorderSolver::RestoreInputTilingVars(const TilingVariable *tiling_vars,
                                                      const TilingVariable *pure_mc_vars,
                                                      const TilingVariable *local_buffer_vars) const {
  if (tiling_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.tiling_vars_size; i++) {
      *(input_.tiling_vars[i]) = tiling_vars[i];
    }
  }
  if (pure_mc_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.pure_mc_vars_size; i++) {
      *(input_.pure_mc_vars[i]) = pure_mc_vars[i];
    }
  }
  if (local_buffer_vars != nullptr) {
    for (uint32_t i = 0U; i < input_.local_buffer_vars_size; i++) {
      *(input_.local_buffer_vars[i]) = local_buffer_vars[i];
    }
  }
}

inline void AxesReorderSolver::FindBetterSolutionByLowerBlockDim(double next_lower_perf,
                                                                 uint32_t next_lower_block_dim
#if AF_AXES_EQUAL_ORDER
,
                                                                 const bool enable_equal_order) {
#else
) {
#endif
double current_perf;
uint32_t current_block_dim;
TilingVariable tiling_vars[AF_AXES_BACKUP_SIZE(input_.tiling_vars_size)];
TilingVariable pure_mc_vars[AF_AXES_BACKUP_SIZE(input_.pure_mc_vars_size)];
TilingVariable local_buffer_vars[AF_AXES_BACKUP_SIZE(input_.local_buffer_vars_size)];

do {
  // 更新当前的性能和block_dim
  current_perf = next_lower_perf;
  current_block_dim = next_lower_block_dim;
  if (!FindNextLowerBlockDim(current_block_dim, next_lower_block_dim)) {
  OP_LOGD(OP_NAME,
          "Found better solution by lower block dim, no lower block dim remains, current_perf: %f, "
          "current_block_dim: %u, input:%s",
          current_perf, current_block_dim, input_.DebugString().c_str());
    // 无更低档位，当前input_就是最优解，直接返回下档位的最优解
    OP_LOGD(OP_NAME, "current_perf: %f, next_lower_perf: %f, current_dim %u, next_lower_block_dim %u, input:%s",
            current_perf, next_lower_perf, current_block_dim, next_lower_block_dim, input_.DebugString().c_str());
    return;
  }
  // 由于GetMaxBlockDimTiling会修改input_，所以这里备份
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  // 根据不限制ub占用率，占满核数的策略更新tiling_data，并返回性能
#if AF_AXES_EQUAL_ORDER
  if (GetMaxBlockDimTiling(next_lower_block_dim, enable_equal_order)) {
#else
  if (GetMaxBlockDimTiling(next_lower_block_dim)) {
#endif
    next_lower_perf = AF_AXES_GET_PERF();
    } else {
      // 继续求解失败，则跳出使用上一次成功求解的值
      break;
    }
  } while ((next_lower_perf - current_perf) < 0.0);  // 若当前性能差于下档位性能，继续下档位

  OP_LOGD(OP_NAME,
          "Found better solution by lower block dim, current_perf: %f, next_lower_perf: %f, "
          "use current_dim %u as best block dim[next_lower_block_dim %u]",
          current_perf, next_lower_perf, current_block_dim, next_lower_block_dim);
  // 下一档位性能差于当前档位，恢复当前档位为最优解返回(维测点：打印上档位最优解的perf,核数)
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
}

inline void AxesReorderSolver::FindBetterSolutionByUpperBlockDim(double next_upper_perf,
#if AF_AXES_EQUAL_ORDER
                                                                 uint32_t next_upper_block_dim,
                                                                 const bool enable_equal_order) {
#else
                                                                 uint32_t next_upper_block_dim) {
#endif

  double current_perf;
  uint32_t current_block_dim;
  // 备份input_
  TilingVariable tiling_vars[AF_AXES_BACKUP_SIZE(input_.tiling_vars_size)];
  TilingVariable pure_mc_vars[AF_AXES_BACKUP_SIZE(input_.pure_mc_vars_size)];
  TilingVariable local_buffer_vars[AF_AXES_BACKUP_SIZE(input_.local_buffer_vars_size)];
  do {
    // 更新当前的性能和block_dim
    current_perf = next_upper_perf;
    current_block_dim = next_upper_block_dim;
    if (!FindNextUpperBlockDim(current_block_dim, next_upper_block_dim)) {
      OP_LOGD(OP_NAME,
              "Found better solution by upper block dim, no upper block dim remains, current_perf: %f, "
              "current_block_dim: %u, input:%s",
              current_perf, current_block_dim, input_.DebugString().c_str());
      // 无更高档位，当前input_就是最优解，直接返回上档位的最优解
      return;
    }
    // 由于GetMaxBlockDimTiling会修改input_，所以这里备份
    SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    // 根据不限制ub占用率，占满核数的策略更新tiling_data，并返回性能
#if AF_AXES_EQUAL_ORDER
    if (GetMaxBlockDimTiling(next_upper_block_dim, enable_equal_order)) {
#else
    if (GetMaxBlockDimTiling(next_upper_block_dim)) {
#endif
      next_upper_perf = AF_AXES_GET_PERF();
    } else {
      // 继续求解失败，则跳出使用上一次成功求解的值
      break;
    }
    OP_LOGD(OP_NAME, "current_perf: %f, next_upper_perf: %f, current_dim %u, next_upper_block_dim %u, input:%s",
            current_perf, next_upper_perf, current_block_dim, next_upper_block_dim, input_.DebugString().c_str());
  } while ((next_upper_perf - current_perf) < 0.0);  // 若当前性能差于下档位性能，继续下档位
  // 返回上档位的最优解(维测点：打印上档位最优解的perf,核数)
  OP_LOGD(OP_NAME,
          "Found better solution by upper block dim, current_perf: %f, next_upper_perf: %f, "
          "use current_dim %u as best block dim[next_upper_block_dim %u], input:%s",
          current_perf, next_upper_perf, current_block_dim, next_upper_block_dim, input_.DebugString().c_str());
  // 上档位性能差于当前档位，恢复当前档位为最优解返回(维测点：打印上档位最优解的perf,核数)
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
}

#if AF_AXES_EQUAL_ORDER
bool AxesReorderSolver::AutoTuning(const bool is_trade_off, const bool enable_equal_order) {
  OP_LOGI(OP_NAME, "Start auto tuning, is_trade_off:%d, enable_equal_order:%d, input:%s", is_trade_off, enable_equal_order,
    input_.DebugString().c_str());
#else
bool AxesReorderSolver::AutoTuning(const bool is_trade_off) {
  OP_LOGI(OP_NAME, "Start auto tuning, is_trade_off:%d, input:%s", is_trade_off, input_.DebugString().c_str());
#endif

  if (is_trade_off) {
    OP_LOGD(OP_NAME,"Do not auto tuning, as is_trade_off is %d.", is_trade_off);
    return true;
  }
  if (input_.corenum_threshold > 1.0f || input_.ub_threshold < 0.0f) {
    OP_LOGD(OP_NAME,"Do not auto tuning, as corenum_threshold is invalid:%f.", input_.corenum_threshold);
    return true;
  }
  auto current_perf = AF_AXES_GET_PERF();
  if (current_perf < input_.perf_threshold) {
    OP_LOGD(OP_NAME, "Do not auto tuning, as current_perf:%lf is less than threshold:%lf, input:%s.", current_perf,
            input_.perf_threshold, input_.DebugString().c_str());
    return true;
  }

  // 使能自动融合
  double got_block_dim = 0.0;
  (void)AF_AXES_CAL_USED_CORE_NUM(got_block_dim);
  const auto block_dim = static_cast<uint32_t>(Ceiling(got_block_dim));
  uint32_t next_upper_block_dim = block_dim;
  uint32_t next_lower_block_dim = block_dim;
  bool got_upper_block_dim = FindNextUpperBlockDim(block_dim, next_upper_block_dim);
  bool got_lower_block_dim = FindNextLowerBlockDim(block_dim, next_lower_block_dim);

  // 1.上档位找不到，则向下档位寻找
  if (!got_upper_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by lower block dim, no upper block dim, current_perf: %f, "
            "current_block_dim: %u, next_lower_block_dim: %u, input:%s",
            current_perf, block_dim, next_lower_block_dim, input_.DebugString().c_str());
#if AF_AXES_EQUAL_ORDER
    FindBetterSolutionByLowerBlockDim(current_perf, block_dim, enable_equal_order);
#else
    FindBetterSolutionByLowerBlockDim(current_perf, block_dim);
#endif

    return true;
  }

  // 2.下档位找不到，则向上档位寻找
  if (!got_lower_block_dim) {
    OP_LOGD(OP_NAME,
            "Ready to find better solution by upper block dim, no lower block dim, current_perf: %f, "
            "current_block_dim: %u, next_upper_block_dim: %u, input:%s",
            current_perf, block_dim, next_upper_block_dim, input_.DebugString().c_str());
#if AF_AXES_EQUAL_ORDER
    FindBetterSolutionByUpperBlockDim(current_perf, block_dim, enable_equal_order);
#else
    FindBetterSolutionByUpperBlockDim(current_perf, block_dim);
#endif

    return true;
  }

  TilingVariable tiling_vars[AF_AXES_BACKUP_SIZE(input_.tiling_vars_size)];
  TilingVariable pure_mc_vars[AF_AXES_BACKUP_SIZE(input_.pure_mc_vars_size)];
  TilingVariable local_buffer_vars[AF_AXES_BACKUP_SIZE(input_.local_buffer_vars_size)];
  // 3.找下1个档位，比较当前档位与下档位性能
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
#if AF_AXES_EQUAL_ORDER
  if (!GetMaxBlockDimTiling(next_lower_block_dim, enable_equal_order)) {
#else
  if (!GetMaxBlockDimTiling(next_lower_block_dim)) {
#endif

    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }

  double next_lower_perf = AF_AXES_GET_PERF();
  // 4.当前档位差于下档位，向下找更优解(考虑多核头开销对小Shape场景的影响和同地址冲突对多核的影响，当前更倾向于下档位)
  if (current_perf > next_lower_perf) {
    OP_LOGD(OP_NAME, "Find lower block dim, as next_lower_perf: %f(block_dim=%u) is better than "
            "current_perf: %f(block_dim=%u), input: %s", next_lower_perf, next_lower_block_dim, current_perf,
            block_dim, input_.DebugString().c_str());
#if AF_AXES_EQUAL_ORDER
    FindBetterSolutionByLowerBlockDim(next_lower_perf, next_lower_block_dim, enable_equal_order);
#else
    FindBetterSolutionByLowerBlockDim(next_lower_perf, next_lower_block_dim);
#endif

    return true;
  }

  // 因为档位档位性能好于下档位，所以还原当前档位的性能
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  SaveInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
#if AF_AXES_EQUAL_ORDER
  if (!GetMaxBlockDimTiling(next_upper_block_dim, enable_equal_order)) {
#else
  if (!GetMaxBlockDimTiling(next_upper_block_dim)) {
#endif

    RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
    return true;
  }
  double next_upper_perf = AF_AXES_GET_PERF();
  // 5.当前档位差于上档位，向上找更优解
  if (current_perf > next_upper_perf) {
    OP_LOGD(OP_NAME,
        "Find upper block dim, as next_upper_perf: %f(block_dim=%u) is better than current_perf: %f(block_dim=%u).",
        next_upper_perf, next_upper_block_dim, current_perf, block_dim);
#if AF_AXES_EQUAL_ORDER
    FindBetterSolutionByUpperBlockDim(next_upper_perf, next_upper_block_dim, enable_equal_order);
#else
    FindBetterSolutionByUpperBlockDim(next_upper_perf, next_upper_block_dim);
#endif

    return true;
  }

  // 6.当前档位好于上档位也好于下档位，当前档位为最优解
  RestoreInputTilingVars(tiling_vars, pure_mc_vars, local_buffer_vars);
  OP_LOGD(
      OP_NAME,
      "Current block dim %u(perf:%f) is best, next upper block dim: %u(perf:%f), next lower block dim: %u(perf:%f).",
      block_dim, current_perf, next_upper_block_dim, next_upper_perf, next_lower_block_dim, next_lower_perf);
  return true;
}

inline bool AxesReorderSolver::WorkloadBalance() {
  OP_LOGD(OP_NAME, "Begin to calculate core num for workload balance, input:%s.", input_.DebugString().c_str());
  int64_t used_corenum = 0;
  (void)AF_AXES_CAL_REAL_USED_CORE_NUM(used_corenum);
  int64_t used_corenum_updated = used_corenum;
  double tmp_corenum = 0.0;
  (void)AF_AXES_CAL_USED_CORE_NUM(tmp_corenum);
  double diff = static_cast<double>(used_corenum) - tmp_corenum;
  double initial_diff = static_cast<double>(used_corenum) - tmp_corenum;
  constexpr double EPS = 1e-6;
  bool related = false;
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.ordered_local_buffer_vars;
  uint32_t index = num_vars - 1;
  for (uint32_t i = 0; i < num_vars; i++) {
    if (vars[i]->mc_related) {
      related = true;
      index = i;
      break;
    }
  }
  if (!related) {
    return true;
  }

  auto &var = vars[index];
  double tmp_vars = var->value;
  OP_LOGD(OP_NAME, "Tmp_vars %f Diff is: %lf, input: %s.", tmp_vars, diff, input_.DebugString().c_str());
  int64_t left = var->prompt_align;
  int64_t right = static_cast<double>(tmp_vars);
  double result = tmp_vars;
  if (right > left) {
    int64_t last_mid = -1;
    while (left <= right) {
      int64_t mid = left + (right - left) / kMidpointDivisor;
      if (mid == last_mid) break;
      last_mid = mid;
      // 对齐到prompt_align的倍数
      int64_t aligned_mid = mid - Mod(mid, var->prompt_align);
      if (aligned_mid < var->prompt_align) {
        left = mid + 1;
        continue;
      }
      // 测试对齐后的值
      var->SetValue(static_cast<double>(aligned_mid));
      (void)AF_AXES_CAL_REAL_USED_CORE_NUM(used_corenum_updated);
      if (used_corenum_updated == used_corenum) {
        result = aligned_mid;   // 满足条件，尝试更小的值
        right = aligned_mid - 1;
      } else {
        left = aligned_mid + 1;  // 不满足条件，尝试更大的值
      }
    }
  }
  var->SetValue(result);
  (void)AF_AXES_CAL_USED_CORE_NUM(tmp_corenum);
  double fine_diff = static_cast<double>(used_corenum) - tmp_corenum;
  OP_LOGD(OP_NAME, "Balance work flag = %d, Initial Diff is : %lf, Fine Diff is: %lf, Opt Ratio is: %lf, input: %s.",
          fabs(initial_diff - fine_diff) >= EPS, initial_diff, fine_diff, initial_diff < EPS ?
          (initial_diff - fine_diff): ((initial_diff - fine_diff) / initial_diff), input_.DebugString().c_str());
  return true;
}
// is_trade_off 描述:表示是否使能根据ub占用率/多核占用率双阈值调优的策略
//              使用说明：默认打开，关闭场景：1)Vector重型算子 2)att_enable_multicore_ub_tradeoff开启
// is_block_loop_auto_tune 描述：表示多核调优时是否根据性能公式权衡核数和核内循环
//                         使用说明：默认关闭，通过att_enable_multicore_ub_tradeoff=true打开
// enable_auto_tune 描述：表示是否使能自动核数调优
//                  使用说明：默认开启，通过att_accuracy_level=0关闭
// enable_equal_order 描述：表示是否使能同优先级切分
#if AF_AXES_EXPLICIT_EVAL
bool AxesReorderSolver::Run(const bool is_trade_off, const bool is_block_loop_auto_tune, const bool enable_auto_tune,
                            const bool enable_equal_order) {
  eval_context_.ResetError();
  is_empty_tensor_ = false;
  try {
    return RunImpl(is_trade_off, is_block_loop_auto_tune, enable_auto_tune, enable_equal_order);
  } catch (const EvalAbort &) {
    return false;
  }
}

bool AxesReorderSolver::RunImpl(const bool is_trade_off, const bool is_block_loop_auto_tune,
                                const bool enable_auto_tune, const bool enable_equal_order)
#else
bool AxesReorderSolver::Run(const bool is_trade_off, const bool is_block_loop_auto_tune,
                            const bool enable_auto_tune, const bool enable_equal_order)
#endif
{
  // Check empty tensor case: when the upper_bound of any axis is 0, return success directly
  auto is_var_empty = [this](const TilingVariable *var) -> bool { return AF_AXES_EVAL_UPPER_BOUND(var) == 0; };
  bool has_empty = std::any_of(input_.pure_mc_vars, input_.pure_mc_vars + input_.pure_mc_vars_size, is_var_empty) ||
                   std::any_of(input_.local_buffer_vars, input_.local_buffer_vars + input_.local_buffer_vars_size,
                               is_var_empty);
  if (has_empty) {
    OP_LOGW(OP_NAME, "Got empty tensor, input[%s]", input_.DebugString().c_str());
    is_empty_tensor_ = true;
    return true;
  }

  // 初始解默认会占满UB，核数使用较少
  // 双阈值策略(NaiveLocalBufTiling)求解失败时，以is_trade_off=false回退到BinaryLocalBufTiling
  // 的简单二分求解重试一次；原本就走Binary的失败路径不重试，第二次仍失败则返回失败
#if AF_AXES_EQUAL_ORDER
  bool solved = GetTiling(is_trade_off, is_block_loop_auto_tune, false, enable_equal_order);
#else
  bool solved = GetTiling(is_trade_off, is_block_loop_auto_tune, false);
#endif
  if (!solved && is_trade_off) {
#if AF_AXES_EQUAL_ORDER
    solved = GetTiling(false, is_block_loop_auto_tune, false, enable_equal_order);
#else
    solved = GetTiling(false, is_block_loop_auto_tune, false);
#endif
  }
  if (!solved) {
#if AF_AXES_EQUAL_ORDER
    OP_LOGW(OP_NAME, "Get default tiling failed, is_trade_off:%d, is_block_loop_tune:%d, enable_equal_order:%d,"
            " input: %s.", is_trade_off, is_block_loop_auto_tune, enable_equal_order, input_.DebugString().c_str());
#else
    OP_LOGW(OP_NAME, "Get default tiling failed, is_trade_off:%d, is_block_loop_tune:%d,"
            " input: %s.", is_trade_off, is_block_loop_auto_tune, input_.DebugString().c_str());
#endif
    return false;
  }
  if (!enable_auto_tune) {
    // hi-perf level设置为0，不支持自动调优，直接返回默认值
    OP_LOGD(OP_NAME, "Do not need auto tuning, enable_auto_tune: %d, input: %s.", enable_auto_tune,
            input_.DebugString().c_str());
    return true;
  }
  // hi-perf level设置为1，支持自动调优
  auto save_core_num = input_.core_num;
#if AF_AXES_EQUAL_ORDER
  if (!AutoTuning(is_trade_off, enable_equal_order)) {
    OP_LOGD(OP_NAME, "Do not need auto tuning, is_trade_off: %d, enable_equal_order: %d, input: %s.", is_trade_off,
            enable_equal_order, input_.DebugString().c_str());
#else
  if (!AutoTuning(is_trade_off)) {
    OP_LOGD(OP_NAME, "Do not need auto tuning, is_trade_off: %d, input: %s.", is_trade_off,
            input_.DebugString().c_str());
#endif
    return false;
  }
  // 负载均衡的逻辑，当前仅考虑单核场景，后续优化
  input_.core_num = save_core_num;
  if (!enable_equal_order && !WorkloadBalance()) {
    return false;
  }
  return true;
}

#undef AF_AXES_BACKUP_SIZE
#undef AF_AXES_EVAL_CONSTRAINT
#undef AF_AXES_EVAL_UPPER_BOUND
#undef AF_AXES_HAS_UPPER_BOUND
#undef AF_AXES_CAL_USED_CORE_NUM
#undef AF_AXES_CAL_REAL_USED_CORE_NUM
#undef AF_AXES_GET_PERF
#undef AF_AXES_SATISFY_UB_CACHE_LINE
// AF_AXES_RESOURCE_END
