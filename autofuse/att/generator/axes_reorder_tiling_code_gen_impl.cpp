/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axes_reorder_tiling_code_gen_impl.h"
#include "axes_solver_resources.h"
#include "args_manager.h"
#include "solver_pass_manager.h"
#include "common/checker.h"
#include <map>
#include <vector>
#include "base/model_info.h"
#include "att_utils.h"

namespace att {
namespace {
constexpr int32_t kMaxEqualOrderAxesCount = 2;
bool IsEnableEqualOrderTiling(const ModelInfo &model_info) {
  std::map<size_t, std::vector<std::string>> order_to_axes;
  for (const auto &arg : model_info.arg_list) {
    if (AttUtils::IsTileSplitAxis(arg)) {
      order_to_axes[arg->order].push_back(arg->name);
    }
  }

  for (const auto &pair : order_to_axes) {
    const std::vector<std::string> &axes = pair.second;
    size_t count = axes.size();
    GE_WARN_ASSERT(
        count <= kMaxEqualOrderAxesCount,
        "[DFX]Equal order tiling algorithm does not support more than %d split axes with same order value, count[%zu]",
        kMaxEqualOrderAxesCount, count);
    if (count >= kMaxEqualOrderAxesCount) {
      GELOGI("[DFX]Equal order tiling algorithm enabled for axes: %s, model: %s", DebugString(axes).c_str(),
             model_info.graph_name.c_str());
      return true;
    }
  }
  GELOGI("[DFX]Equal order tiling algorithm disabled for model: %s", model_info.graph_name.c_str());
  return false;
}

bool IsAnyModelEnableEqualOrderTiling(const TilingModelInfo &model_info) {
  for (const auto &info : model_info) {
    if (IsEnableEqualOrderTiling(info)) {
      return true;
    }
  }
  return false;
}

// 辅助函数：获取Group个数并设置到SolverPassManager
size_t GetGroupNumAndSetToSolver(const ModelInfo &model_info,
                                 const std::map<std::pair<size_t, size_t>, size_t> &schedule_result_group_nums,
                                 SolverPassManager &solver_pass_manager) {
  const auto key =
      std::make_pair(model_info.schedule_group_ident.asc_graph_id, model_info.schedule_group_ident.impl_graph_id);
  size_t group_num = 1UL;  // 默认值为1
  if (const auto it = schedule_result_group_nums.find(key); it != schedule_result_group_nums.end()) {
    group_num = it->second;
  }
  solver_pass_manager.SetGroupNum(group_num);
  return group_num;
}
}  // namespace

void AxesReorderTilingCodeGenImpl::ConfigureSolverPassManagerCommon(SolverPassManager &solver_pass_manager) {
  solver_pass_manager.SetUBThreshold(config_.ub_threshold);
  solver_pass_manager.SetCoreNumThreshold(config_.corenum_threshold);
  solver_pass_manager.SetEnableMulticoreUBTradeoff(config_.enable_multicore_ub_tradeoff);
  solver_pass_manager.SetEnableAutofusePGO(config_.enable_autofuse_pgo);
  solver_pass_manager.SetAutofusePGOStepMax(config_.pgo_step_max);
  solver_pass_manager.SetVariableReplace(config_.do_variable_replace);
  solver_pass_manager.SetHighPerfTiling(config_.high_precision);
}

af::Status AxesReorderTilingCodeGenImpl::GenSolverBaseClass() {
  auto &solver_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kSolver].dependencies;
  for (const auto &header : {"cstddef", "sstream", "utility", "vector"}) {
    autofuse::RequireSystemHeader(solver_dependencies, header);
  }
  const bool is_enable_equal_order_tiling = IsAnyModelEnableEqualOrderTiling(tiling_model_info_);
  if (is_enable_equal_order_tiling) {
    RequireTranslationUnitSystemHeader("limits");
    RequireTranslationUnitSystemHeader("map");
  }
  std::string basic_solvers_head = SolverPassManager::GenAxesReorderBaseClassesHead(is_enable_equal_order_tiling);
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver, basic_solvers_head);
  std::string basic_solvers_func = SolverPassManager::GenAxesReorderBaseClassesFunc(is_enable_equal_order_tiling);
  tiling_func_.AddLine(basic_solvers_func);
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    std::string pgo_solver_head = SolverPassManager::GenAxesReorderPgoClassesHead(config_.pgo_step_max);
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver, pgo_solver_head);
    std::string pgo_solver_func = SolverPassManager::GenAxesReorderPgoClassesFunc();
    tiling_func_.AddLine(pgo_solver_func);
  }
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenSolverTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId(), model_info.sub_case_tag},
                                        config_.tiling_data_type_name);
  ConfigureSolverPassManagerCommon(solver_pass_manager);
  solver_pass_manager.SetReservedUbSize(model_info.reserved_ub_size);

  // 获取同一ScheduleResult中的Group个数并设置到SolverPassManager
  size_t group_num = GetGroupNumAndSetToSolver(model_info, schedule_result_group_nums_, solver_pass_manager);
  GELOGI("[DFX] GenSolverTiling: asc_graph_id=%zu, impl_graph_id=%zu, group_num=%zu, group_ids_in_map=%zu",
         model_info.schedule_group_ident.asc_graph_id, model_info.schedule_group_ident.impl_graph_id, group_num,
         schedule_result_group_nums_.size());

  tiling_func_.AddLine(solver_pass_manager.GenAxesReorderClass());
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenDoTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  SolverPassManager solver_pass_manager(args_manager, {args_manager.GetTilingCaseId(), model_info.sub_case_tag},
                                        config_.tiling_data_type_name);
  ConfigureSolverPassManagerCommon(solver_pass_manager);

  // 获取同一ScheduleResult中的Group个数并设置到SolverPassManager
  GetGroupNumAndSetToSolver(model_info, schedule_result_group_nums_, solver_pass_manager);

  solver_pass_manager.SetEnableEqualOrder(IsEnableEqualOrderTiling(model_info));
  solver_pass_manager.SetIsInductorScene(config_.is_inductor_scene);

  GenGetSetTilingImpl(model_info);
  solver_pass_manager.SetIsUniGroup(is_uniq_group_);
  solver_pass_manager.SetTilingDataSubGroupItemName(model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data");
  return GenDoTilingCommon(model_info, solver_pass_manager.GenAxesReorderFunc(arrange_code_));
}

af::Status AxesReorderTilingCodeGenImpl::GenToolFuncs() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenToolFuncs(), "GenToolFuncs failed!");
  std::istringstream source(axes_resource::kAxesCeilDiv);
  std::string line;
  while (std::getline(source, line)) {
    tiling_func_.AddLine(line);
  }
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenStructCopyDef(), "Generate struct copy.");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenCacheHashMapDef(), "Generate cache hash map.");
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenTilingImplPublicFunc() {
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenTilingImplPublicFunc(), "Generate tiling public func failed.");
  GE_ASSERT_SUCCESS(GenVirtualDataTransferFuncs(), "Generate virtual data transfer funcs failed.");
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenHardwareCons(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  for (const auto &pair : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
    auto iter = kHardwareNameMap.find(pair.first);
    if (iter == kHardwareNameMap.end()) {
      continue;
    }
    tiling_func_.AddLine("  int Get" + iter->second + "(" + config_.tiling_data_type_name + "& tiling_data) {");
    if (iter->second == "ub_size") {
      tiling_func_.AddLine(std::string("    return AxesReorderSolvercase") + model_info.sub_case_tag +
                           std::to_string(model_info.tiling_case_id) + "::GetTilingDataUbSizeStatic(tiling_data);");
    } else if (iter->second == "block_dim") {
      tiling_func_.AddLine(std::string("    return AxesReorderSolvercase") + model_info.sub_case_tag +
                           std::to_string(model_info.tiling_case_id) + "::GetTilingDataBlockDimStatic(tiling_data);");
    } else {
      tiling_func_.AddLine(GenBufRelatedVars(pair.second, args_manager.GetContainerMap()));
    }
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("");
  }
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenPipeTypeObj(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  const std::string solver_name =
      "AxesReorderSolvercase" + model_info.sub_case_tag + std::to_string(model_info.tiling_case_id);
  tiling_func_.AddLine("  void CollectPipeEstimates(" + config_.tiling_data_type_name +
                       " &tiling_data, std::vector<FinalTilingPipeEstimate> &out) override {");
  for (const auto &pair : args_manager.GetObjectFunc()) {
    const auto iter = kPipetypeNameMap.find(pair.first);
    if (iter == kPipetypeNameMap.end()) {
      continue;
    }
    tiling_func_.AddLine("    out.push_back(FinalTilingPipeEstimate{\"" + iter->second + "\", " + solver_name +
                         "::GetTilingDataPerfStatic(PipeType::" + iter->second + ", tiling_data), true});");
  }
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenGetObj(const ModelInfo &model_info) {
  Expr expression;
  std::vector<Expr> funcs;
  Expr expr;
  std::string codes;
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  Expr head_cost = args_manager.GetHeadCost();
  tiling_func_.AddLine("  double GetPerf(" + config_.tiling_data_type_name + "& tiling_data) override {");
  tiling_func_.AddLine("    return AxesReorderSolvercase" + model_info.sub_case_tag +
                       std::to_string(model_info.tiling_case_id) +
                       "::GetTilingDataPerfStatic(PipeType::ALL, tiling_data);");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status AxesReorderTilingCodeGenImpl::GenExtraSummaryInfo(const ModelInfo &model_info,
                                                             const ArgsManager &args_manager,
                                                             std::string &case_info_str) {
  for (const auto &pair : args_manager.GetObjectFunc()) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter != kPipetypeNameMap.end()) {
      tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + iter->second + " is %f" + case_info_str +
                           ".\", AxesReorderSolvercase" + model_info.sub_case_tag +
                           std::to_string(model_info.tiling_case_id) +
                           "::GetTilingDataPerfStatic(PipeType::" + iter->second + ", tiling_data));");
    }
  }
  tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The objective value of the tiling data is %f" + case_info_str +
                       ".\", AxesReorderSolvercase" + model_info.sub_case_tag +
                       std::to_string(model_info.tiling_case_id) +
                       "::GetTilingDataPerfStatic(PipeType::ALL, tiling_data));");
  return af::SUCCESS;
}
}  // namespace att
