/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axes_reorder_solver_code_common.h"

namespace att {

// ============================================================================
// Section 6: Equal Priority Axes Support
// ============================================================================

// Helper function: Binary search with alignment
std::string GenBinarySearchWithAlignment() {
  return R"(
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
    int64_t mid = left + (right - left) / 2;
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
)";
}

std::string GenDecreaseUntilSatisfied() {
  return R"(
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
)";
}

std::string GenShrinkBoundaryUntilSatisfied() {
  return R"(
bool AxesReorderSolver::ShrinkBoundaryUntilSatisfied(TilingVariable *var, int64_t boundary, ConstraintType cons_type) {
  int64_t last_boundary = -1;
  int64_t last_val = -1;
  while (!(last_boundary == boundary && last_val == var->value)) {
    last_boundary = boundary;
    last_val = var->value;
    var->value = CeilDiv((boundary + var->value) / 2, var->align) * var->align;
    if (!SatisfyCons(var, cons_type)) {
      boundary = var->value;
      var->value = last_val;
    }
  }
  return true;
}
)";
}

// Helper function: Tune no tail
std::string GenTuneNoTail() {
  return R"(
bool AxesReorderSolver::TuneNoTail(TilingVariable **vars, uint32_t id) {
  if (!TuneNotailVar(vars[id])) {
    OP_LOGW(OP_NAME, "Tune notail var failed for axis_%d, falling back to original algorithm", id);
    return false;
  }
  return DecreaseUntilSatisfied(vars[id], ConstraintType::LB_MIXED, [this](TilingVariable* v) { return TuneNotailVar(v); });
}
)";
}

std::string GenIdentifyEqualPriorityAxes() {
  return R"(
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
      int64_t upper = var->upper_bound(var->upper_bound_vars);
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
)";
}

std::string GenBinarySearchEqualPriorityAxes() {
  return R"(
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
)";
}

std::string GenIterativeSolveEqualPriorityAxes() {
  return R"(
bool AxesReorderSolver::IterativeSolveEqualPriorityAxes(const uint32_t axis_num, uint32_t *axis_idx,
                                                        int64_t lower_bound, int64_t upper_bound) {
  if (!BinarySearchEqualPriorityAxes(axis_num, axis_idx, lower_bound, upper_bound)) {
    return false;
  }
  for (uint32_t i = 0u; i < axis_num; ++i) {
    auto &var = input_.local_buffer_vars[axis_idx[i]];
    int64_t limited_upper_bound = var->value;
    if (var->upper_bound(var->upper_bound_vars) <= limited_upper_bound) {
      continue;
    }
    var->SetValue(limited_upper_bound);
    int64_t var_upper_bound = var->upper_bound(var->upper_bound_vars);
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
)";
}

std::string GenSolveEqualPriorityAxesWithDualThreshold() {
  return R"(
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
)";
}

// Helper function: Generate function signature for binary search with dual threshold
std::string GenBinarySearchDualThresholdSignature() {
  return R"(
bool AxesReorderSolver::BinarySearchEqualPriorityAxesWithDualThreshold(
    TilingVariable **vars, int64_t lower_bound, int64_t upper_bound,
    int64_t &upper_ub_a, int64_t &upper_ub_b) {
)";
}

// Helper function: Generate initialization code for binary search
std::string GenBinarySearchDualThresholdInitialization() {
  return R"(
  int64_t left = lower_bound;
  int64_t right = upper_bound;
  int64_t result = -1;
  int64_t max_core_num = static_cast<int64_t>(input_.corenum_threshold * static_cast<double>(input_.core_num));

  OP_LOGD(OP_NAME, "[DFX] BinarySearchEqualPriorityAxesWithDualThreshold: range=[%ld, %ld], min_ub_threshold=%ld",
          lower_bound, upper_bound, static_cast<int64_t>(input_.ub_size * input_.ub_threshold));
)";
}

// Helper function: Generate binary search loop code
std::string GenBinarySearchDualThresholdLoop() {
  return R"(
  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    mid = CeilDiv(mid, vars[0]->align) * vars[0]->align;
    vars[0]->SetValue(mid);
    vars[1]->SetValue(mid);
    // 计算block_dim（使用CalRealUsedCoreNum）
    int64_t block_dim = 0;
    if (!CalRealUsedCoreNum(block_dim)) {
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
)";
}

// Helper function: Generate post-processing code for B axis
std::string GenBinarySearchDualThresholdPostProcess() {
  return R"(
  int64_t var_b_upper_bound = vars[1]->upper_bound(vars[1]->upper_bound_vars);
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
)";
}

std::string GenBinarySearchEqualPriorityAxesWithDualThreshold() {
  std::string codes;
  codes += GenBinarySearchDualThresholdSignature();
  codes += GenBinarySearchDualThresholdInitialization();
  codes += GenBinarySearchDualThresholdLoop();
  codes += GenBinarySearchDualThresholdPostProcess();
  return codes;
}

std::string GenProcessNonMCAxes() {
  return R"(
bool AxesReorderSolver::ProcessNonMCAxes(
    TilingVariable **vars, uint32_t *axis_idx,
    std::vector<bool> &solved_axes) {
  for (uint32_t i = 0; i < 2; ++i) {
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
)";
}

static std::string GenProcessSingleMCAxisPre() {
  return R"(
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
  int64_t lower_bound_satisfied_ub_threshold = mc_var->align;)";
}

std::string GenProcessSingleMCAxis() {
  std::string codes = GenProcessSingleMCAxisPre();
  return codes.append(R"(
  if (BinaryFindLowerBoundSatisfiedUBThresholdCond(mc_var, mc_idx, mc_var->align, lower_bound_satisfied_ub_threshold)) {
    mc_var->SetValue(upper_bound_satisfied_ub_threshold);
    auto satisfied_core_threshold = BinaryFindLowerBoundSatisfiedCoreNum(
        mc_var, mc_idx, lower_bound_satisfied_ub_threshold);
    int64_t available_core_num_right = 0L;
    mc_var->SetValue(satisfied_core_threshold.second);
    if (InitMulticoreVars() && MulticoreTilingCore(false) && CalRealUsedCoreNum(available_core_num_right)) {
      // 成功
    }
    int64_t available_core_num_left = 0L;
    if ((satisfied_core_threshold.first != satisfied_core_threshold.second) && (satisfied_core_threshold.first > 0L)) {
      mc_var->SetValue(satisfied_core_threshold.first);
      if (InitMulticoreVars() && MulticoreTilingCore(false) &&
          CalRealUsedCoreNum(available_core_num_left) &&
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
)");
}

// Process dual multicore axes - updated to use three-phase algorithm
std::string GenProcessDualMCAxes() {
  return GenProcessDualMCAxesOrchestration();
}

// ============================================================================
// Helper functions for GenTuneDualMCAxesScenario1
// ============================================================================

std::string GenTuneDualMCAxesScenario1_SearchA() {
  return R"(
  // 固定a轴为upper_ub，在[lower_bound, upper_ub]中寻找满足核数利用率的值
  var_a->SetValue(upper_ub);
  int64_t left = lower_bound;
  int64_t right = upper_ub;
  int64_t result_a = upper_ub;
  int64_t max_core_num = static_cast<int64_t>(input_.corenum_threshold * static_cast<double>(input_.core_num));
  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    var_a->SetValue(mid);
    if (!InitMulticoreVars() || !MulticoreTilingCore(false)) {
      left = mid + var_a->align;
      continue;
    }
    int64_t available_core_num = 0L;
    if (!CalRealUsedCoreNum(available_core_num)) {
      left = mid + var_a->align;
      continue;
    }
    bool satisfied_cache_line = SatisfyUBSizeCacheLine(0);
    if (satisfied_cache_line && available_core_num >= max_core_num) {
      result_a = mid;
      right = mid - var_a->align;
    } else if (satisfied_cache_line) {
      // 尝试减小a轴以增加核数
      result_a = mid;
      right = mid - var_a->align;
    } else {
      left = mid + var_a->align;
    }
  }
  var_a->SetValue(result_a;
)";
}

std::string GenTuneDualMCAxesScenario1_InitB() {
  return R"(
  // 如果仍无法满足核数利用率，尝试调节b轴
  if (!InitMulticoreVars() || !MulticoreTilingCore(false)) {
    return false;
  }
  int64_t available_core_num = 0L;
  if (!CalRealUsedCoreNum(available_core_num) || available_core_num < max_core_num) {
)";
}

std::string GenTuneDualMCAxesScenario1_SearchB() {
  return R"(
    int64_t upper_ub_b = var_b->value;
    int64_t lower_bound_b = var_b->align;
    int64_t left_b = lower_bound_b;
    int64_t right_b = upper_ub_b;
    int64_t result_b = lower_bound;
    while (left_b <= right_b) {
      int64_t mid = left_b + (right_b - left) / 2;
      var_b->SetValue(mid);
      if (!InitMulticoreVars() || !MulticoreTilingCore(false)) {
        left_b = mid + var_b->align;
        continue;
      }
      int64_t available_core_num_b = 0L;
      if (!CalRealUsedCoreNum(available_core_num_b)) {
        left_b = mid + var_b->align;
        continue;
      }
      bool satisfied_cache_line = SatisfyUBSizeCacheLine(1);
      if (satisfied_cache_line && available_core_num_b >= max_core_num) {
        result_b = mid;
        right_b = mid - var_b->align;
      } else {
        left_b = mid + var_b->align;
      }
    }
    var_b->SetValue(result_b);
  }
)";
}

std::string GenTuneDualMCAxesScenario1_Finalize() {
  return R"(
  TilingVariable *vars_final[] = {var_a, var_b};
  return FinalizeEqualPriorityAxes(vars_final, axis_idx, solved_axes);
)";
}

// Tune dual MC axes scenario 1: a-axis can satisfy UB threshold
std::string GenTuneDualMCAxesScenario1() {
  std::string codes;
  codes += "bool AxesReorderSolver::TuneDualMCAxesScenario1(TilingVariable *var_a, TilingVariable *var_b,\n";
  codes += "                                                int64_t lower_bound, int64_t upper_ub,\n";
  codes += "                                                uint32_t *axis_idx, std::vector<bool> &solved_axes) {\n";
  codes += GenTuneDualMCAxesScenario1_SearchA();
  codes += GenTuneDualMCAxesScenario1_InitB();
  codes += GenTuneDualMCAxesScenario1_SearchB();
  codes += GenTuneDualMCAxesScenario1_Finalize();
  codes += "}\n";

  return codes;
}

std::string GenTuneDualMCAxesScenario2() {
  return R"(
bool AxesReorderSolver::TuneDualMCAxesScenario2(
    TilingVariable *var_a, TilingVariable *var_b,
    uint32_t *axis_idx, std::vector<bool> &solved_axes) {
  // 固定a轴为upper_ub，继续调节b轴
  int64_t upper_ub_b = var_b->value;
  int64_t lower_bound_b = var_b->align;
  // 寻找b轴满足UB利用率的下限
  int64_t lower_bound_satisfied_ub_b = lower_bound_b;
  if (!BinaryFindLowerBoundSatisfiedUBThresholdCond(var_b, 1, lower_bound_b, lower_bound_satisfied_ub_b)) {
    return false;
  }
  // 在 [lower_bound_satisfied_ub_b, upper_ub_b] 中寻找满足核数利用率的值
  int64_t left = lower_bound_satisfied_ub_b;
  int64_t right = upper_ub_b;
  int64_t result_b = lower_bound_satisfied_ub_b;
  int64_t max_core_num = static_cast<int64_t>(input_.corenum_threshold * static_cast<double>(input_.core_num));
  while (left <= right) {
    int64_t mid = left + (right - left) / 2;
    var_b->SetValue(mid);
    if (!InitMulticoreVars() || !MulticoreTilingCore(false)) {
      left = mid + var_b->align;
      continue;
    }
    int64_t available_core_num = 0L;
    if (!CalRealUsedCoreNum(available_core_num)) {
      left = mid + var_b->align;
      continue;
    }
    bool satisfied_cache_line = SatisfyUBSizeCacheLine(1);
    if (satisfied_cache_line && available_core_num >= max_core_num) {
      result_b = mid;
      right = mid - var_b->align;
    } else {
      left = mid + var_b->align;
    }
  }
  var_b->SetValue(result_b);
  TilingVariable *vars[] = {var_a, var_b};
  return FinalizeEqualPriorityAxes(vars, axis_idx, solved_axes);
}
)";
}

// Finalize equal priority axes - updated version
std::string GenFinalizeEqualPriorityAxes() {
  return R"(
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
)";
}

}  // namespace att
