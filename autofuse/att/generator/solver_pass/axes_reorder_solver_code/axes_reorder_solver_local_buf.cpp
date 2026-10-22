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
// Section 5: Local Buffer Tiling
// ============================================================================

std::string GenInitLocalMCVars() {
  std::string codes;
  codes += "  if (!InitLocalBufferVars()) {\n";
  codes += "    OP_LOGW(OP_NAME, \"init local buffer failed\");\n";
  codes += "    return false;\n";
  codes += "  }\n";
  codes += "  if (!InitMulticoreVars()) {\n";
  codes += "    OP_LOGW(OP_NAME, \"multicore tiling failed\");\n";
  codes += "    return false;\n";
  codes += "  }\n";
  codes += "  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {\n";
  codes += "    OP_LOGW(OP_NAME, \"local buffer tiling failed in the initial check\");\n";
  codes += "    return false;\n";
  codes += "  }\n";
  return codes;
}

std::string GenMcRelatedNaiveTiling() {
  return R"(
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
        if (InitMulticoreVars() && MulticoreTilingCore(false) && CalRealUsedCoreNum(available_core_num_right)) {
          OP_LOGD(OP_NAME, "Found larger tile size:%ld, available_core_num:%ld, i:%u",
                  satisfied_core_threshold_right, available_core_num_right, canonical_idx);
        }
        int64_t available_core_num_left = 0L;
        if ((satisfied_core_threshold_left != satisfied_core_threshold_right) && (satisfied_core_threshold_left > 0L)) {
          var->SetValue(satisfied_core_threshold_left);
          if (InitMulticoreVars() && MulticoreTilingCore(false) && CalRealUsedCoreNum(available_core_num_left) &&
              (available_core_num_left > available_core_num_right)) {
            OP_LOGD(OP_NAME, "Found smaller tile size:%ld, available_core_num:%ld, i:%u",
                    satisfied_core_threshold_left, available_core_num_left, canonical_idx);
          } else {
            var->SetValue(satisfied_core_threshold_right);
          }
        }
      }
    }
)";
}

std::string GenNaiveLocalBufTilingImpl() {
  std::string kNaiveLocalBufTilingImpl = R"(
  for (uint32_t ordered_idx = 0U; ordered_idx < num_vars; ++ordered_idx) {
    auto &var = vars[ordered_idx];
    const uint32_t canonical_idx = input_.ordered_to_canonical[ordered_idx];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
    int64_t boundary = (upper_bound / var->align) * var->align;
    if (boundary < var->align) {
      OP_LOGW(OP_NAME, "Invalid aligned upper bound:%ld, raw upper:%ld, align:%ld, i:%u, input: %s.",
              boundary, upper_bound, var->align, canonical_idx, input_.DebugString().c_str());
      return false;
    }
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
)";
  std::string kNaiveLocalBufTilingImplPostProcess = R"(
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
  }
  return SatisfyCons(ConstraintType::LOCAL_BUFFER) && SatisfyCons(ConstraintType::LB_MIXED);
}
)";
  return kNaiveLocalBufTilingImpl + GenMcRelatedNaiveTiling() + kNaiveLocalBufTilingImplPostProcess;
}

std::string GenNaiveLocalBufTilingWithEqualOrderImpl() {
  std::string codes = "bool AxesReorderSolver::NaiveLocalBufTiling(const bool enable_equal_order) {\n";
  codes.append(R"(
  // 初始化和验证
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

  // 处理同优先级轴
  std::vector<bool> solved_axes(input_.local_buffer_vars_size, false);
  if (enable_equal_order) {
    constexpr uint32_t kMaxAxes = 2;
    uint32_t axis_idx[kMaxAxes]{};
    int64_t min_ub = std::numeric_limits<int64_t>::max();
    int64_t max_align = 1L;

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
    const uint32_t canonical_idx = input_.ordered_to_canonical[ordered_idx];
    if (solved_axes[canonical_idx]) {
      continue;
    }
    auto *var = input_.ordered_local_buffer_vars[ordered_idx];
    if (!ProcessSingleAxisNaive(var, canonical_idx, max_core_num)) {
      return false;
    }
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return enable_equal_order && NaiveLocalBufTiling(false);
  }
  return SatisfyCons(ConstraintType::LB_MIXED);
}
)");
  return codes;
}

// 通过双阈值平衡多核占用和ub利用，双阈值分别是ub和多核的占比，默认ub为0.2，多核为0.8，满足如下两条规则，需要停止增大ub
// 1.ub至少达到总UB*ub_threshold
// 2.最大使用核数不大于总CORENUM*corenum_threshold
std::string GenNaiveLocalBufTiling(bool enable_equal_order_tiling) {
  std::string codes;
  if (enable_equal_order_tiling) {
    codes.append(GenNaiveLocalBufTilingWithEqualOrderImpl());
  } else {
    // 使能为 false 时：函数签名无参数
    codes.append("bool AxesReorderSolver::NaiveLocalBufTiling() {\n");
    codes.append(GenInitLocalMCVars());
    codes.append(R"(
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.ordered_local_buffer_vars;
)");
    codes.append(GenNaiveLocalBufTilingImpl());
  }
  return codes;
}

std::string GenBinaryLocalBufTilingCore() {
  return R"(
bool AxesReorderSolver::BinaryLocalBufTilingCore(const std::vector<bool> &solved_axes) {
  for (uint32_t ordered_idx = 0U; ordered_idx < input_.local_buffer_vars_size; ++ordered_idx) {
    const uint32_t canonical_idx = input_.ordered_to_canonical[ordered_idx];
    if (solved_axes[canonical_idx]) {
      continue;
    }
    auto &var = input_.ordered_local_buffer_vars[ordered_idx];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
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
)";
}

static std::string GenBinaryLocalBufTilingFinalCheck(bool enable_equal_order_tiling) {
  std::string codes = R"(
  if (!BinaryLocalBufTilingCore(solved_axes)) {
    OP_LOGW(OP_NAME, "Binary local Tiling Calculation failed in final check.");
    return false;
  }
)";
  if (enable_equal_order_tiling) {
    codes += R"(
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return enable_equal_order && BinaryLocalBufTiling(false);
  }
)";
  } else {
    codes += R"(
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return false;
  }
)";
  }
  codes += R"(
  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Binary local Tiling Calculation failed in final check.");
    return false;
  }
  return true;
)";
  return codes;
}

namespace {
inline std::string GenEqualPriorityAxesTiling() {
  return R"(
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
  })";
}
}  // namespace

std::string GenBinaryLoadBufTilingEqualOrder() {
  std::string codes = R"(
  std::vector<bool> solved_axes(input_.local_buffer_vars_size, false);
  // 相等优先级轴的逻辑，仅开关使能后生效
  constexpr uint32_t kSupportMaxEqualPriorityAxes = 2;
  uint32_t axis_idx[kSupportMaxEqualPriorityAxes]{0};
  int64_t min_upper_bound = std::numeric_limits<int64_t>::max();
  int64_t max_aligned = 1L;
  bool equal_axes_tiling_success = false;)";
  codes.append(GenEqualPriorityAxesTiling());
  codes.append(GenBinaryLocalBufTilingFinalCheck(true));
  return codes;
}

std::string GenBinaryLocalBufTiling(bool enable_equal_order_tiling) {
  std::string codes;
  if (enable_equal_order_tiling) {
    // 使能为 true 时：函数签名有参数，调用时传写死的 true
    codes += "bool AxesReorderSolver::BinaryLocalBufTiling(const bool enable_equal_order) {\n";
    codes += GenInitLocalMCVars();
    codes += GenBinaryLoadBufTilingEqualOrder();
  } else {
    // 使能为 false 时：函数签名无参数，直接处理所有轴
    codes += "bool AxesReorderSolver::BinaryLocalBufTiling() {\n";
    codes += GenInitLocalMCVars();
    codes += "\n  std::vector<bool> solved_axes(input_.local_buffer_vars_size, false);\n";
    codes += GenBinaryLocalBufTilingFinalCheck(false);
  }
  return codes.append("}\n");
}

std::string GenLocalBufTiling(bool enable_equal_order_tiling) {
  if (enable_equal_order_tiling) {
    // 使能为 true 时：调用时传写死的 true
    return R"(
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune,
                                       const bool enable_equal_order) {
  (void)block_loop_auto_tune;
  if (is_tuning) {
    return NaiveLocalBufTiling(enable_equal_order);
  } else {
    return BinaryLocalBufTiling(enable_equal_order);
  }
}
)";
  } else {
    // 使能为 false 时：无参数调用
    return R"(
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning, const bool block_loop_auto_tune) {
  (void)block_loop_auto_tune;
  if (is_tuning) {
    return NaiveLocalBufTiling();
  } else {
    return BinaryLocalBufTiling();
  }
}
)";
  }
}

}  // namespace att
