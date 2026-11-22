/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "solver_pass_manager.h"
#include "common/platform_context.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace att {
template <typename SolverGenType>
auto SolverPassManager::GenerateSolverGen() -> SolverGenType {
  ExprExprMap max_value;
  ExprExprMap min_value;
  ExprExprMap init_value;
  std::vector<Expr> search_args = args_manager_.GetSearchableVars();

  for (const auto &arg : search_args) {
    min_value[arg] = args_manager_.GetMinValue(arg);
    max_value[arg] = args_manager_.GetMaxValue(arg);
  }

  for (const auto &arg : search_args) {
    init_value[arg] = args_manager_.GetDefaultInitValue(arg);
  }

  SolverGenType solver_gen("case" + std::to_string(case_id_), tiling_data_type_);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetExprRelation(args_manager_.GetExprRelations(), args_manager_.GetVarsRelations());
  solver_gen.SetHeadCost(args_manager_.GetHeadCost());
  solver_gen.SetConstArgs(args_manager_.GetConstVars());
  solver_gen.SetSolvedArgs(args_manager_.GetSolvedVars());
  solver_gen.SetObj(args_manager_.GetObjectFunc());
  solver_gen.SetMinValue(min_value);
  solver_gen.SetMaxValue(max_value);
  solver_gen.SetBufferCons(args_manager_.GetTotalHardwareCons());
  solver_gen.SetInputArgs(args_manager_.GetInputVars());
  solver_gen.SetCutCons(args_manager_.GetTotalCutCons());
  solver_gen.SetInputAlign(GetInputsAlign(true));
  solver_gen.SetReplaceVars(args_manager_.GetTernaryOpReplaceVars());
  solver_gen.SetExeTimeMap(args_manager_.GetTernaryOpRelatedVars());
  solver_gen.SetInnestDim(args_manager_.GetNodeInnerestDimSizes());
  solver_gen.SetInitValue(init_value);

  return solver_gen;
}

template <typename SpecificSolverGen>
std::pair<std::string, std::string> SolverPassManager::GenerateSolverPassFunc(SpecificSolverGen solver_gen) {
  std::string impl_codes;
  std::string invoke_codes;
  std::pair<std::string, std::string> codes;
  std::vector<Expr> all_args = args_manager_.GetSearchableVars();
  if (all_args.size() == 0) {
    return codes;
  }
  impl_codes = solver_gen.GenSolverFuncImpl();
  invoke_codes = solver_gen.GenSolverFuncInvoke();
  codes = std::make_pair(impl_codes, invoke_codes);
  args_manager_.SetSolvedVars(args_manager_.GetSearchableVars());
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::SolverPassFuncGen(SolverType type) {
  std::pair<std::string, std::string> codes;
  if (type == SolverType::SEARCH_TILE) {
    args_manager_.DoVarsReplace();
    codes = GenerateSolverPassFunc(GenerateSolverGen<GeneralSolverGen>());
  }
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::GenFuncPass() {
  std::string impl_codes;
  std::string invoke_codes;
  std::pair<std::string, std::string> pass_codes;
  args_manager_.Process(false);
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    auto codes = SolverPassFuncGen(type);
    impl_codes += codes.first;
    invoke_codes += codes.second;
  }
  pass_codes = std::make_pair(impl_codes, invoke_codes);
  return pass_codes;
}

std::pair<std::string, std::string> SolverPassManager::GeneralSolverDtFuncGen() {
  GeneralSolverGen solver_gen = GenerateSolverGen<GeneralSolverGen>();
  std::string impl_codes = solver_gen.GenSolverDTImpl();
  std::string invoke_codes = solver_gen.GenSolverDTInvoke();
  return std::make_pair(impl_codes, invoke_codes);
}

std::pair<std::string, std::string> SolverPassManager::SolverDtFuncGen(SolverType type) {
  std::pair<std::string, std::string> codes;
  if (type == SolverType::SEARCH_TILE) {
    args_manager_.DoVarsReplace();
    codes = GeneralSolverDtFuncGen();
  }
  return codes;
}

std::string SolverPassManager::GeneralSolverPassClassGen() {
  GeneralSolverGen solver_gen = GenerateSolverGen<GeneralSolverGen>();
  std::string code = solver_gen.GenSolverClassImpl();
  args_manager_.SetSolvedVars(args_manager_.GetSearchableVars());
  return code;
}

std::string SolverPassManager::SolverPassClassGen(SolverType type) {
  if (type == SolverType::SEARCH_TILE) {
    args_manager_.DoVarsReplace();
    return GeneralSolverPassClassGen();
  }
  return "";
}

std::string SolverPassManager::GenClassPass() {
  std::string codes;
  args_manager_.Process(false);
  for (uint32_t i = 0; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    codes += SolverPassClassGen(type);
  }
  return codes;
}

bool SolverPassManager::IsNeedSolver(std::vector<ArgsManager> args_managers, SolverType type) {
  if (type == SolverType::SEARCH_TILE) {
    for (auto &args_manager : args_managers) {
      if (args_manager.GetSearchableVars().size() > 0) {
        return true;
      }
    }
    return false;
  }
  return false;
}

ExprExprMap SolverPassManager::GetInputsAlign(bool do_replace) {
  Expr align_value;
  std::vector<Expr> ancestors;
  ExprExprMap input_align;
  std::vector<Expr> input_args = args_manager_.GetInputVars();
  std::vector<Expr> search_args;
  if (do_replace) {
    for (const auto &pair : args_manager_.GetExprRelations()) {
      search_args.push_back(pair.first);
    }
  } else {
    search_args = args_manager_.GetSearchableVars();
  }
  for (const auto &arg : input_args) {
    input_align[arg] = af::Symbol(1);
  }
  for (const auto &arg : search_args) {
    ancestors = args_manager_.GetAncestor(arg);
    align_value = args_manager_.GetVarAlignValue(arg);
    for (const auto &ancestor : ancestors) {
      if (input_align.find(ancestor) != input_align.end()) {
        if (!align_value.IsConstExpr() ||
            (align_value.IsConstExpr() &&
             af::SymbolicUtils::StaticCheckGt(align_value, input_align[ancestor]) == af::TriBool::kTrue)) {
          input_align[ancestor] = align_value;
        }
      }
    }
  }
  return input_align;
}

ExprExprMap SolverPassManager::GetOriginalInputAlign() const {
  ExprExprMap input_align;
  for (const auto &arg : args_manager_.GetInputVars()) {
    input_align[arg] = af::Symbol(1);
  }
  return input_align;
}

std::string SolverPassManager::GenCommonBaseClassesHead(std::vector<ArgsManager> args_managers) {
  return GenCommonBaseClassesHeader(std::move(args_managers)).body;
}

autofuse::GeneratedCode SolverPassManager::GenCommonBaseClassesHeader(std::vector<ArgsManager> args_managers) {
  autofuse::GeneratedCode base_classes;
  for (uint32_t i = 0U; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    if (IsNeedSolver(args_managers, type)) {
      base_classes.body += GetSolverHead(type);
      if (type == SolverType::SEARCH_TILE) {
        autofuse::RequireSystemHeader(base_classes.dependencies, "cstddef");
      }
    }
  }
  return base_classes;
}

std::string SolverPassManager::GenCommonBaseClassesFunc(std::vector<ArgsManager> args_managers) {
  std::string base_classes;
  for (uint32_t i = 0U; i < static_cast<std::uint32_t>(SolverType::ERROR); i++) {
    SolverType type = static_cast<SolverType>(i);
    if (IsNeedSolver(args_managers, type)) {
      base_classes += GetSolverFunc(type);
    }
  }
  return base_classes;
}

std::string SolverPassManager::GenAxesReorderBaseClassesHead(bool enable_equal_order_tiling) {
  return GetAxesReorderSolverHead(enable_equal_order_tiling);
}

std::string SolverPassManager::GenAxesReorderBaseClassesFunc(bool enable_equal_order_tiling) {
  return GetAxesReorderSolverFunc(enable_equal_order_tiling);
}

std::string SolverPassManager::GenAxesReorderPgoClassesHead(int64_t pgo_step_max) {
  return GetAxesReorderPgoSolverHead(pgo_step_max);
}

std::string SolverPassManager::GenAxesReorderPgoClassesFunc() {
  return GetAxesReorderPgoSolverFunc();
}

void SolverPassManager::AddConcatInnerDims(const Expr &arg, std::vector<Expr> &concat_inner_dims) {
  if (args_manager_.IsConcatInnerDim(arg)) {
    concat_inner_dims.emplace_back(arg);
  }
}

void SolverPassManager::InitSolverGen(AxesReorderSolverGen &solver_gen) {
  solver_gen.SetTotalCutCons(args_manager_.GetTotalCutCons());
  solver_gen.SetBufferUseAlg(args_manager_.GetTotalHardwareCons(do_variable_replace_));
  solver_gen.SetInputArgs(args_manager_.GetInputVars());
  solver_gen.SetSearchArgs(args_manager_.GetSearchableVars());
  solver_gen.SetContainerExpr(args_manager_.GetContainerMap());
  solver_gen.SetContainerNames(args_manager_.GetContainerNames());
  solver_gen.SetReplaceVars(args_manager_.GetTernaryOpReplaceVars());
  solver_gen.SetTernaryOps(args_manager_.GetTernaryOps());
  solver_gen.SetExeTimeMap(args_manager_.GetTernaryOpRelatedVars());
  solver_gen.SetInputAlign(GetOriginalInputAlign());
  solver_gen.SetVarPriority(args_manager_.GetAxesPriority());
  solver_gen.SetAxesOrder(args_manager_.GetAxesOrder());
  solver_gen.SetVarsRelations(args_manager_.GetVarsRelations());
  solver_gen.SetObjFunc(args_manager_.GetHeadCost(), args_manager_.GetObjectFunc());
  solver_gen.SetUBThreshold(ub_threshold_);
  solver_gen.SetReservedUbSize(reserved_ub_size_);
  solver_gen.SetCoreNumThreshold(corenum_threshold_);
  solver_gen.SetEnableMulticoreUBTradeoff(enable_multicore_ub_tradeoff_);
  solver_gen.SetEnableAutofusePGO(enable_autofuse_pgo_);
  solver_gen.SetIsInductorScene(is_inductor_scene_);
  solver_gen.SetAutofusePGOStepMax(pgo_step_max_);
  solver_gen.SetHighPerfTiling(enable_high_perf_);
  solver_gen.SetEnableEqualOrder(enable_equal_order_);
  solver_gen.Arrange();
  solver_gen.SetInputOutputDef(GetInputOutputDef());
  solver_gen.SetInputOutputCall(GetInputOutputCall());
  solver_gen.SetTilingDataSubGroupItemName(GetTilingDataSubGroupItemName());
  solver_gen.SetIsUniGroup(GetIsUniGroup());
  solver_gen.SetTilingScheduleConfigTable(args_manager_.GetModelInfo().tiling_schedule_config_table);
  solver_gen.SetTilingScheduleConfig(args_manager_.GetModelInfo().tiling_schedule_config);
  solver_gen.SetRuntimeReorderRules(args_manager_.GetModelInfo().runtime_reorder_rules);
  solver_gen.SetCacheLineConfig(&args_manager_.GetModelInfo().cache_line_config);
  // 门控输入：核预算取平台 AIV 核数（编译期常量）。整张张量字节数由每个 CacheLineConfig 自带，
  // 避免把全图 size 变量乘积误当成单节点张量规模。核预算不可用时置 0，生成器退回 develop 判据。
  ge::PlatformInfo platform_info;
  uint32_t core_num_budget = 0U;
  if (ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(platform_info) && platform_info.aiv_num > 0) {
    core_num_budget = static_cast<uint32_t>(platform_info.aiv_num);
  }
  solver_gen.SetCoreNumBudget(core_num_budget);
  solver_gen.SetEnableParallel(args_manager_.GetModelInfo().enable_group_parallel);
  solver_gen.SetGroupNum(group_num_);
  solver_gen.SetTilingCaseIdent({args_manager_.GetModelInfo().schedule_group_ident,
                                 args_manager_.GetModelInfo().tiling_case_id,
                                 args_manager_.GetModelInfo().sub_case_tag});
  GELOGD("[DFX]Set %s and tiling schedule %s for axes reorder solver gen", DebugString().c_str(),
         args_manager_.GetModelInfo().tiling_schedule_config.DebugString().c_str());
}

AxesReorderSolverGen SolverPassManager::GenAxesReorderGen() {
  ExprExprMap arg_align_map;
  ExprUintMap arg_prompt_align_map;
  ExprUintMap const_vars_map;
  ExprUintMap is_concat_outer_map;
  ExprUintMap arg_data_type_size_map;
  std::vector<Expr> concat_inner_dims;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  for (const auto &arg : args_manager_.GetSearchableVars()) {
    from_axes_map[arg] = args_manager_.GetParentVars(arg);
    arg_align_map[arg] = args_manager_.GetVarAlignValue(arg);
    arg_prompt_align_map[arg] = args_manager_.GetVarPromptAlignValue(arg);
    is_concat_outer_map[arg] = args_manager_.IsConcatOuterDim(arg);
    AddConcatInnerDims(arg, concat_inner_dims);
    arg_data_type_size_map[arg] = args_manager_.GetDataTypeSizeVar(arg);
  }
  for (const auto &pair : args_manager_.GetConstVars()) {
    const_vars_map[pair.first] = pair.second;
    arg_prompt_align_map[pair.first] = args_manager_.GetVarPromptAlignValue(pair.first);
    ;
    AddConcatInnerDims(pair.first, concat_inner_dims);
    arg_data_type_size_map[pair.first] = args_manager_.GetDataTypeSizeVar(pair.first);
  }
  for (const auto &input_var : args_manager_.GetInputVars()) {
    arg_prompt_align_map[input_var] = args_manager_.GetVarPromptAlignValue(input_var);
    AddConcatInnerDims(input_var, concat_inner_dims);
    arg_data_type_size_map[input_var] = args_manager_.GetDataTypeSizeVar(input_var);
  }
  AxesReorderSolverGen solver_gen("case" + sub_case_tag_ + std::to_string(case_id_), tiling_data_type_);
  InitSolverGen(solver_gen);
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetArgPromptAlignMap(arg_prompt_align_map);
  solver_gen.SetArgDataTypeSizeMap(arg_data_type_size_map);
  solver_gen.SetConstArgs(const_vars_map);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetIsConcatOuterMap(is_concat_outer_map);
  solver_gen.SetConcatInnerDims(concat_inner_dims);
  return solver_gen;
}

std::string SolverPassManager::GenAxesReorderClass() {
  args_manager_.Process(false);
  AxesReorderSolverGen solver_gen = GenAxesReorderGen();
  std::string codes = solver_gen.GenSolverClassImpl();
  return codes;
}

std::pair<std::string, std::string> SolverPassManager::GenAxesReorderFunc(const std::string &arrange_code) {
  args_manager_.Process(false);
  AxesReorderSolverGen solver_gen = GenAxesReorderGen();
  solver_gen.SetArrangeCode(arrange_code);
  std::string impl_codes = solver_gen.GenSolverFuncImpl();
  std::string invoke_codes = solver_gen.GenSolverFuncInvoke();
  return std::make_pair(impl_codes, invoke_codes);
}
}  // namespace att
