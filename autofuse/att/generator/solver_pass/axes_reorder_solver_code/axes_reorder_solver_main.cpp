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
// Section 9: Main Entry Points
// ============================================================================

std::string GenWorkloadBalancePrepare() {
  return R"(
inline bool AxesReorderSolver::WorkloadBalance() {
  OP_LOGD(OP_NAME, "Begin to calculate core num for workload balance, input:%s.", input_.DebugString().c_str());
  int64_t used_corenum = 0;
  (void)CalRealUsedCoreNum(used_corenum);
  int64_t used_corenum_updated = used_corenum;
  double tmp_corenum = 0.0;
  (void)CalUsedCoreNum(tmp_corenum);
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
)";
}

std::string GenWorkloadBalance() {
  std::string kWorkloadBalanceImpl = R"(
  auto &var = vars[index];
  double tmp_vars = var->value;
  OP_LOGD(OP_NAME, "Tmp_vars %f Diff is: %lf, input: %s.", tmp_vars, diff, input_.DebugString().c_str());
  int64_t left = var->prompt_align;
  int64_t right = static_cast<double>(tmp_vars);
  double result = tmp_vars;
  if (right > left) {
    int64_t last_mid = -1;
    while (left <= right) {
      int64_t mid = left + (right - left) / 2;
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
      (void)CalRealUsedCoreNum(used_corenum_updated);
      if (used_corenum_updated == used_corenum) {
        result = aligned_mid;   // 满足条件，尝试更小的值
        right = aligned_mid - 1;
      } else {
        left = aligned_mid + 1;  // 不满足条件，尝试更大的值
      }
    }
  }
  var->SetValue(result);
  (void)CalUsedCoreNum(tmp_corenum);
  double fine_diff = static_cast<double>(used_corenum) - tmp_corenum;
  OP_LOGD(OP_NAME, "Balance work flag = %d, Initial Diff is : %lf, Fine Diff is: %lf, Opt Ratio is: %lf, input: %s.",
          fabs(initial_diff - fine_diff) >= EPS, initial_diff, fine_diff, initial_diff < EPS ?
          (initial_diff - fine_diff): ((initial_diff - fine_diff) / initial_diff), input_.DebugString().c_str());
  return true;
}
)";
  return GenWorkloadBalancePrepare() + kWorkloadBalanceImpl;
}

std::string GenObjDrivenOptimize(bool enable_equal_order_tiling) {
  std::string codes;
  codes += "  if (!enable_auto_tune) {\n";
  codes += "    // hi-perf level设置为0，不支持自动调优，直接返回默认值\n";
  codes += "    OP_LOGD(OP_NAME, \"Do not need auto tuning, enable_auto_tune: %d, input: %s.\", enable_auto_tune,\n";
  codes += "            input_.DebugString().c_str());\n";
  codes += "    return true;\n";
  codes += "  }\n";
  codes += "  // hi-perf level设置为1，支持自动调优\n";
  codes += "  auto save_core_num = input_.core_num;\n";
  if (enable_equal_order_tiling) {
    codes += "  if (!AutoTuning(is_trade_off, enable_equal_order)) {\n";
    codes +=
        "    OP_LOGD(OP_NAME, \"Do not need auto tuning, is_trade_off: %d, enable_equal_order: %d, input: %s.\", "
        "is_trade_off,\n";
    codes += "            enable_equal_order, input_.DebugString().c_str());\n";
    codes += "    return false;\n";
    codes += "  }\n";
  } else {
    codes += "  if (!AutoTuning(is_trade_off)) {\n";
    codes += "    OP_LOGD(OP_NAME, \"Do not need auto tuning, is_trade_off: %d, input: %s.\", is_trade_off,\n";
    codes += "            input_.DebugString().c_str());\n";
    codes += "    return false;\n";
    codes += "  }\n";
  }
  codes += "  // 负载均衡的逻辑，当前仅考虑单核场景，后续优化\n";
  codes += "  input_.core_num = save_core_num;\n";
  codes += "  if (!enable_equal_order && !WorkloadBalance()) {\n";
  codes += "    return false;\n";
  codes += "  }\n";
  return codes;
}

std::string GenEmptyTensorCheck() {
  std::string codes;
  codes += "  // Check empty tensor case: when the upper_bound of any axis is 0, return success directly\n";
  codes += "  auto is_var_empty = [](const TilingVariable *var) -> bool {\n";
  codes += "    return var->upper_bound(var->upper_bound_vars) == 0;\n";
  codes += "  };\n";
  codes +=
      "  bool has_empty = std::any_of(input_.pure_mc_vars, input_.pure_mc_vars + input_.pure_mc_vars_size, "
      "is_var_empty) ||\n";
  codes +=
      "                   std::any_of(input_.local_buffer_vars, input_.local_buffer_vars + "
      "input_.local_buffer_vars_size,\n";
  codes += "                               is_var_empty);\n";
  codes += "  if (has_empty) {\n";
  codes += "    OP_LOGW(OP_NAME, \"Got empty tensor, input[%s]\", input_.DebugString().c_str());\n";
  codes += "    is_empty_tensor_ = true;\n";
  codes += "    return true;\n";
  codes += "  }\n";
  codes += "\n";
  return codes;
}

std::string GenAxesReorderRun(bool enable_equal_order_tiling) {
  std::string codes;
  std::string run_impl_start =
      "bool AxesReorderSolver::Run(const bool is_trade_off, const bool is_block_loop_auto_tune, "
      "const bool enable_auto_tune, const bool enable_equal_order) {\n";
  codes += "// is_trade_off 描述:表示是否使能根据ub占用率/多核占用率双阈值调优的策略\n";
  codes += "//              使用说明：默认打开，关闭场景：1)Vector重型算子 2)att_enable_multicore_ub_tradeoff开启\n";
  codes += "// is_block_loop_auto_tune 描述：表示多核调优时是否根据性能公式权衡核数和核内循环\n";
  codes += "//                         使用说明：默认关闭，通过att_enable_multicore_ub_tradeoff=true打开\n";
  codes += "// enable_auto_tune 描述：表示是否使能自动核数调优\n";
  codes += "//                  使用说明：默认开启，通过att_accuracy_level=0关闭\n";
  codes += "// enable_equal_order 描述：表示是否使能同优先级切分\n";
  codes += run_impl_start;

  // 在GetTiling之前添加空tensor检测
  codes += GenEmptyTensorCheck();

  // 根据是否使能equal_order生成不同的GetTiling调用
  std::string equal_order_param = enable_equal_order_tiling ? ", enable_equal_order" : "";
  codes += "  // 初始解默认会占满UB，核数使用较少\n";
  codes += "  // 双阈值策略(NaiveLocalBufTiling)求解失败时，以is_trade_off=false回退到BinaryLocalBufTiling\n";
  codes += "  // 的简单二分求解重试一次；原本就走Binary的失败路径不重试，第二次仍失败则返回失败\n";
  codes += "  bool solved = GetTiling(is_trade_off, is_block_loop_auto_tune, false" + equal_order_param + ");\n";
  codes += "  if (!solved && is_trade_off) {\n";
  codes += "    solved = GetTiling(false, is_block_loop_auto_tune, false" + equal_order_param + ");\n";
  codes += "  }\n";
  std::string equal_order_param_str = enable_equal_order_tiling ? ", enable_equal_order:%d" : "";
  codes += "  if (!solved) {\n";
  codes += "    OP_LOGW(OP_NAME, \"Get default tiling failed, is_trade_off:%d, is_block_loop_tune:%d" +
           equal_order_param_str + ",\"\n";
  codes += "            \" input: %s.\", is_trade_off, is_block_loop_auto_tune" + equal_order_param +
           ", input_.DebugString().c_str());\n";
  codes += "    return false;\n";
  codes += "  }\n";
  codes += GenObjDrivenOptimize(enable_equal_order_tiling);
  codes += "  return true;\n";
  codes += "}\n";
  return codes;
}

std::string GetAxesSolverSolverHead(bool enable_equal_order_tiling) {
  std::string general_solver;
  general_solver += GenConstraintType();
  general_solver += GenStructDef();
  general_solver += GenVariable();
  general_solver += GenConstraint();
  general_solver += GenTilingVariable();
  general_solver += GenAxesReorderSolverInput();
  general_solver += GenAxesReorderSolver(enable_equal_order_tiling);
  return general_solver;
}

std::string GetAxesSolverPgoSolverHead(int64_t pgo_step_max) {
  std::string codes;
  codes += "class AxesReorderPgoSolver : public AxesReorderSolver {\n";
  codes += "public:\n";
  codes += "  explicit AxesReorderPgoSolver(const AxesReorderSolverInput &input) : AxesReorderSolver(input) {}\n";
  codes += "  ~AxesReorderPgoSolver() = default;\n";
  codes += "  bool PgoSolverGenerateAllTilingData();\n";
  codes += "  std::vector<std::vector<uint32_t>> GetTilingDataList() { return availiable_tiling_data_list_; }\n";
  codes += "private:\n";
  codes += "  std::vector<std::vector<uint32_t>> availiable_tiling_data_list_;\n";
  codes += "  int64_t pgo_step_max_{" + std::to_string(pgo_step_max) + "};\n";
  codes += "  void PgoSolverGenerateAllTilingDataInner(const uint32_t index, std::vector<uint32_t> &ans_item,\n";
  codes +=
      "                                           std::vector<std::vector<uint32_t>> &ans, int64_t step_max = 16);\n";
  codes += "};\n";
  return codes;
}

// Helper function: Generate core solver functions
std::string GenCoreSolverFunctions() {
  std::string codes;
  codes += GenInitAllVars();
  codes += GenSatisfyCons();
  codes += GenTuneNoTailVar();
  codes += GenMutiCoreTilingCore();
  codes += GenMulticoreTiling();
  codes += GenApplyPromptAlign();
  return codes;
}

// Helper function: Generate binary search functions
std::string GenBinarySearchFunctions() {
  std::string codes;
  codes += GenIsSatisfyCons();
  codes += GenBinaryFindUpperBoundSatisfiedUBLimit();
  codes += GenBinaryFindLowerBoundSatisfiedUBThresholdCond();
  codes += GenBinaryFindLowerBoundSatisfiedCoreNum();
  codes += GenShrinkBoundaryUntilSatisfied();
  codes += GenDecreaseUntilSatisfied();
  return codes;
}

// Helper function: Generate equal order solver functions
std::string GenEqualOrderSolverFunctions(bool enable_equal_order_tiling) {
  if (!enable_equal_order_tiling) {
    return "";
  }
  std::string codes;
  // Add advanced version of BinaryFindLowerBoundSatisfiedCoreNum with per-variable upper bounds
  codes += GenBinaryFindLowerBoundSatisfiedCoreNum_Advanced();
  // Add math utility functions
  codes += GenGcd();
  codes += GenLcm();
  codes += GenAlignToUpperBound();
  codes += GenDualAxesInfo();
  // Add three-phase algorithm framework
  codes += GenInitializeDualAxesInfo();
  codes += GenExecuteStep1_FindUBLimit();
  codes += GenExecuteStep2_FindUBThreshold();
  codes += GenExecuteStep3_FindCoreNumTileSize();
  codes += GenProcessDualMCAxesOrchestration();
  // Add NaiveTiling helper functions
  codes += GenProcessMCAxisNaive();
  codes += GenProcessSingleAxisNaive();
  // Add existing equal priority functions
  codes += GenBinarySearchWithAlignment();
  codes += GenTuneNoTail();
  codes += GenIdentifyEqualPriorityAxes();
  codes += GenBinarySearchEqualPriorityAxes();
  codes += GenIterativeSolveEqualPriorityAxes();
  codes += GenSolveEqualPriorityAxesWithDualThreshold();
  codes += GenBinarySearchEqualPriorityAxesWithDualThreshold();
  codes += GenProcessNonMCAxes();
  codes += GenProcessSingleMCAxis();
  codes += GenFinalizeEqualPriorityAxes();
  return codes;
}

// Helper function: Generate local buffer tiling functions
std::string GenLocalBufferTilingFunctions(bool enable_equal_order_tiling) {
  std::string codes;
  codes += GenBinaryLocalBufTilingCore();
  codes += GenNaiveLocalBufTiling(enable_equal_order_tiling);
  codes += GenBinaryLocalBufTiling(enable_equal_order_tiling);
  codes += GenLocalBufTiling(enable_equal_order_tiling);
  return codes;
}

// Helper function: Generate main solver functions
std::string GenMainSolverFunctions(bool enable_equal_order_tiling) {
  std::string codes;
  codes += GenGetTiling(enable_equal_order_tiling);
  codes += GenGetMaxBlockDimTiling(enable_equal_order_tiling);
  codes += GenFindNextUpperBlockDim();
  codes += GenFindNextLowerBlockDim();
  codes += GenSaveInputTilingVars();
  codes += GenRestoreInputTilingVars();
  codes += GenFindBetterSolutionByLowerBlockDim(enable_equal_order_tiling);
  codes += GenFindBetterSolutionByUpperBlockDim(enable_equal_order_tiling);
  codes += GenAutoTuning(enable_equal_order_tiling);
  codes += GenWorkloadBalance();
  codes += GenAxesReorderRun(enable_equal_order_tiling);
  return codes;
}

std::string GetAxesSolverSolverFunc(bool enable_equal_order_tiling) {
  std::string general_solver;
  general_solver += GenCoreSolverFunctions();
  general_solver += GenBinarySearchFunctions();
  general_solver += GenEqualOrderSolverFunctions(enable_equal_order_tiling);
  general_solver += GenLocalBufferTilingFunctions(enable_equal_order_tiling);
  general_solver += GenMainSolverFunctions(enable_equal_order_tiling);
  return general_solver;
}

std::string GetAxesSolverPgoSolverFunc() {
  std::string pgo_solver;
  pgo_solver += GenPgoSolverGenerateAllTilingData();
  return pgo_solver;
}

const std::string AXES_SOLVER_CODE_HEAD = GetAxesSolverSolverHead(true);
const std::string AXES_SOLVER_CODE_FUNC = GetAxesSolverSolverFunc(true);
const std::string AXES_SOLVER_PGO_CODE_FUNC = GetAxesSolverPgoSolverFunc();

}  // namespace att
