/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_AXES_REORDER_SOLVER_GEN_H_
#define ATT_AXES_REORDER_SOLVER_GEN_H_
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "base/base_types.h"
#include "code_printer.h"
#include "util/base_types_printer.h"
#include "generator/solver_pass_gen/solver_gen.h"
#include "gen_model_info/api_perf_register/perf_param.h"
#include "autofuse_config/auto_fuse_config.h"
#include "generator/solver_pass_gen/input_output_setters.h"
#include "generator/solver_pass_gen/input_output_setters_mixin.h"
#include "generator/solver_pass_gen/pgo_config_setters_mixin.h"

namespace att {
enum class ConsType {
  BUFFER = 0,
  CUT = 1,
  MCMIXED = 2,
  ALL = 3,
};

enum class InputType {
  INPUT = 0,
  TILING = 1,
};

enum class VarsType {
  PUREMC = 0,
  LOCALBUFFER = 1,
};

class AxesReorderSolverGen : public SolverGen,
                             public InputOutputSettersMixin<AxesReorderSolverGen>,
                             public PgoConfigSettersMixin<AxesReorderSolverGen> {
 public:
  explicit AxesReorderSolverGen(const std::string &tiling_case_id, const std::string &type_name)
      : SolverGen(tiling_case_id, type_name) {}
  ~AxesReorderSolverGen() override = default;
  std::string GenSolverClassImpl() override;
  std::string GenSolverFuncImpl() override;
  std::string GenPGOSolverFilter();
  std::string GenSolverFuncInvoke() override;
  std::string GenPGOSolverClassImpl();
  std::string GenPGOSolverFuncImpl();

  void SetInputArgs(const std::vector<Expr> &input_args) {
    input_args_ = input_args;
  }
  void SetConstArgs(const ExprUintMap &const_vars) {
    std::vector<Expr> const_args;
    const_vars_map_ = const_vars;
    for (const auto &pair : const_vars_map_) {
      const_args.push_back(pair.first);
    }
    const_args_ = const_args;
  }
  void SetBufferUseAlg(const std::map<HardwareDef, Expr> &hardware_use_map) {
    hardware_use_map_ = hardware_use_map;
  }
  void SetArgAlignMap(const ExprExprMap &arg_align_map) {
    arg_align_map_ = arg_align_map;
  }
  void SetArgPromptAlignMap(const ExprUintMap &arg_prompt_align_map) {
    arg_prompt_align_map_ = arg_prompt_align_map;
  }
  void SetArgDataTypeSizeMap(const ExprUintMap &data_type_size_map) {
    data_type_size_map_ = data_type_size_map;
  }
  void SetInputAlign(const ExprExprMap &input_align);
  void SetTotalCutCons(const std::vector<Expr> &total_cut_cons) {
    total_cut_cons_ = total_cut_cons;
  }
  void SetFromAxesMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &from_axes_map) {
    from_axes_map_ = from_axes_map;
  }
  void SetVarPriority(const ExprUintMap &priority) {
    priority_map_ = priority;
  }
  void SetAxesOrder(const ExprUintMap &axes_order) {
    axes_order_ = axes_order;
  }
  void SetVarsRelations(const ExprExprMap &vars_relations) {
    vars_relations_ = vars_relations;
  }
  void SetContainerExpr(const ExprExprMap &container_expr) {
    container_expr_ = container_expr;
  }
  void SetContainerNames(const std::map<Expr, std::string, ExprCmp> &container_names) {
    container_names_ = container_names;
  }
  void SetReplaceVars(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
    for (const auto &var : replace_vars) {
      replace_vars_.emplace_back(var);
    }
  }
  void SetTernaryOps(const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
    ternary_ops_ = ternary_ops;
  }

  void SetExeTimeMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &exe_time_map) {
    for (const auto &pair : exe_time_map) {
      exe_time_map_[pair.first] = pair.second;
    }
  }
  void Arrange();
  void SetObjFunc(const Expr &head_cost, const std::map<PipeType, Expr> pipe_2_obj_map) {
    head_cost_ = head_cost;
    pipe_2_obj_map_ = pipe_2_obj_map;
  }
  void SetIsConcatOuterMap(const ExprUintMap &is_concat_outer_map) {
    is_concat_outer_map_ = is_concat_outer_map;
  }
  void SetConcatInnerDims(const std::vector<Expr> &concat_inner_dims) {
    concat_inner_dims_ = concat_inner_dims;
  }
  void SetUBThreshold(const double &ub_threshold) {
    ub_threshold_ = ub_threshold;
  }
  void SetCoreNumThreshold(const double &corenum_threshold) {
    corenum_threshold_ = corenum_threshold;
  };
  void SetReservedUbSize(const Expr &reserved_ub_size) {
    reserved_ub_size_ = reserved_ub_size;
  };
  void SetHighPerfTiling(const bool enable_high_perf) {
    enable_high_perf_ = enable_high_perf;
  }
  void SetEnableEqualOrder(const bool enable_equal_order) {
    enable_equal_order_ = enable_equal_order;
  }
  void SetSearchArgs(const std::vector<Expr> &search_args) {
    search_args_ = search_args;
  }
  void SetArrangeCode(const std::string &arrange_code) {
    arrange_code_ = arrange_code;
  }
  void SetTilingScheduleConfigTable(const TilingScheduleConfigTable *tiling_schedule_config_table) {
    tiling_schedule_config_table_ = tiling_schedule_config_table;
  }
  void SetTilingScheduleConfig(const TilingScheduleConfig &tiling_schedule_config) {
    tiling_schedule_config_ = tiling_schedule_config;
  }
  void SetRuntimeReorderRules(const std::vector<RuntimeReorderRule> &runtime_reorder_rules) {
    runtime_reorder_rules_ = runtime_reorder_rules;
  }
  void SetCacheLineConfig(const vector<CacheLineConfig> *cache_line_config) {
    cache_line_config_ = cache_line_config;
  }
  // 核预算（平台 AIV 核数），用于判断"整张量是否可能让多个核共享同一条 cache line"。
  // 必须取编译期常量：求解器自动调优时 input_.core_num 会被改写为正在试探的档位，
  // 若门控依赖它会随档位翻转，破坏各候选档位的可比性。
  void SetCoreNumBudget(uint32_t core_num_budget) {
    core_num_budget_ = core_num_budget;
  }
  void SetEnableParallel(bool enable_parallel) {
    enable_group_parallel_ = enable_parallel;
  }
  void SetTilingCaseIdent(TilingCaseIdent tiling_case_ident) {
    tiling_case_ident_ = tiling_case_ident;
  }
  void SetGroupNum(size_t group_num) {
    group_num_ = group_num;
  }

 private:
  static bool VarCmp(Expr &a, Expr &b);
  void ReorderVars();
  void GetMCArgs();
  void GetLocalBufferTilingVars();
  void GetRelatedArgs(const Expr &expr, std::vector<Expr> &related_args) const;
  bool NeedUBMultiCoreBalance();
  std::string GenGetStaticInputParam(const HardwareDef &hardware_type, bool no_type = false) const;
  std::string GenGetObjStaticInputParam(bool no_type = false);
  std::string GenGetObjStaticFunc();
  void CollectInitialWorkList(std::vector<Expr> &work_list) const;
  void CollectNeededTenaryVarsClosure(std::vector<Expr> &work_list, std::set<std::string> &needed_vars) const;
  std::string GenTenaryVarDecls(const std::set<std::string> &needed_vars);
  std::string GenSingleTenaryVar(const TernaryOp &op, const std::string &var_name, std::set<std::string> &declared_vars,
                                 std::map<std::string, std::string> &content_to_first_var);
  std::string GenGetTilingDataObjStaticFunc();
  std::string GenObjFunc();
  std::string GenGetUbSizeStaticFunc();
  std::string GenGetTilingDataUbSizeStaticFunc();
  std::string GenGetBlockDimStatic(Expr &corenum_cons);
  std::string GenGetTilingDataBlockDimStatic(Expr &corenum_cons);
  std::string GenUBThresholdFunc();
  std::string GenUBSizeCacheLineFunc();
  // 静态 shape：在编译期直接判定 "whole_tensor_bytes < k * core_budget * cache_line"，
  // 使门控折叠为常量判据，运行期不引入分支。无法求值（动态 shape 或缺核预算）时返回 false。
  bool TryEvalConflictPossible(const Expr &whole_tensor_bytes, uint32_t cache_line_size, bool &conflict_possible) const;
  // 生成门控条件文本（只依赖 shape 输入变量，不含 input_.core_num）。不可用时返回空串。
  std::string GenConflictPossibleCond(const Expr &whole_tensor_bytes, uint32_t cache_line_size) const;
  std::string GenCoreNumFunc();
  std::pair<std::vector<Expr>, std::vector<Expr>> SortConsArgs(const Expr &expr, bool &is_mc_mixed);
  std::string ObtainRelatedVars(Expr &expr);
  std::string InitiateArgs();
  std::string InitiateBufferConsArgs(uint32_t cons_idx, HardwareDef hardware, const Expr &cons);
  std::string InitiateCutConsArgs(uint32_t cons_idx, const Expr &cons, bool &is_mc_mixed);
  std::string GenConsUbFunc(uint32_t cons_idx, const std::vector<Expr> &rel_tiling_vars,
                            const std::vector<Expr> &rel_cons_vars) const;
  std::string GenConsFunc(uint32_t cons_idx, ConsType cons_type, const Expr &cons,
                          const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars) const;
  std::string SetVarCons(const Expr &arg, const std::vector<Expr> &all_cons) const;
  std::string GenUpperBoundFunc(const Expr &var);
  std::string GenUpperBoundInfo(const Expr &var);
  std::string SetInputVars(InputType input_type);
  std::string SetInputCons(std::vector<Expr> cons) const;
  std::string SetTilingVars(VarsType var_type);
  std::string GenRuntimeReorderRules();
  std::string GenRuntimeReorderRule(const RuntimeReorderRule &rule);
  std::string GenRuntimeExprValue(const Expr &expr) const;
  std::string GenRuntimeCompoundExprValue(const Expr &expr) const;
  int32_t GetLocalBufferVarIndex(const Expr &expr) const;
  int32_t ResolveLocalBufferVarIndex(const Expr &expr) const;
  std::vector<int32_t> GetRuntimePreferredIndices(const RuntimeReorderRule &rule) const;
  void InitConcatPromptAlign(const Expr &local_var, const uint32_t prompt_align, std::string &strs);
  std::string GenInputInfo(std::vector<Expr> &all_cons, std::vector<Expr> &local_buffer_cons,
                           std::vector<Expr> &mc_mixed_cons);
  std::string GenInput(const TradeOffConfig &trade_off_config, std::vector<Expr> &all_cons);
  std::string GenSetTiling();
  std::string GenSolverRunInvoke(const std::string &class_name);
  std::string GenEmptyTensorCheckInSolver();
  std::string GenOriginExpr(const std::vector<Expr> &exprs, const std::string &indent) const;
  std::pair<std::string, std::string> GenOriginBufExpr(const Expr &expr, const std::string &indent) const;
  std::pair<std::string, std::string> GenNamedOriginBufExpr(const Expr &expr, const std::string &indent) const;
  std::string GenPgoSetTiling();
  std::string GenPgoSetMaxBlockDim() const;
  std::vector<uint32_t> GetArgRelateCons(const Expr &arg, const std::vector<Expr> &all_cons) const;
  std::string IsEnableBlockLoopTradeOffByPerf() const;
  std::string GenPendingSearchConfigOverride();
  std::vector<Expr> mc_args_;
  std::vector<Expr> input_args_;
  std::vector<Expr> const_args_;
  std::vector<Expr> total_cut_cons_;
  std::vector<Expr> local_buffer_tiling_vars_;
  ExprExprMap input_align_;
  ExprUintMap const_vars_map_;
  ExprExprMap arg_align_map_;
  ExprUintMap arg_prompt_align_map_;
  ExprUintMap data_type_size_map_;
  ExprExprMap container_expr_;
  std::vector<std::pair<Expr, Expr>> replace_vars_;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops_;
  std::map<Expr, std::vector<Expr>, ExprCmp> exe_time_map_;
  std::map<Expr, std::string, ExprCmp> container_names_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map_;
  static ExprUintMap priority_map_;
  ExprUintMap axes_order_;
  ExprExprMap vars_relations_;
  std::map<PipeType, Expr> pipe_2_obj_map_;
  Expr head_cost_;
  ExprUintMap is_concat_outer_map_;
  std::vector<Expr> concat_inner_dims_;
  ExprUintMap mc_related_ub_args_map_;
  std::vector<Expr> search_args_;
  double ub_threshold_{0.2};
  Expr reserved_ub_size_{CreateExpr(0)};
  double corenum_threshold_{0.4};
  bool enable_high_perf_{false};
  bool enable_equal_order_{false};
  std::string arrange_code_;
  const TilingScheduleConfigTable *tiling_schedule_config_table_{nullptr};
  TilingScheduleConfig tiling_schedule_config_;  // Model 级别的 Tiling 调度配置
  std::vector<RuntimeReorderRule> runtime_reorder_rules_;
  const vector<CacheLineConfig> *cache_line_config_{nullptr};
  uint32_t core_num_budget_{0U};
  bool enable_group_parallel_{false};
  size_t group_num_{1UL};
  TilingCaseIdent tiling_case_ident_{ScheduleGroupIdent{}, 0U, ""};
};
bool CheckExist(const std::vector<Expr> &args, const Expr &check_arg);
std::string SetRelatedVars(const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars);
std::string GenRelatedVars(uint32_t cons_idx, const std::vector<Expr> &rel_tiling_vars,
                           const std::vector<Expr> &rel_cons_vars);
}  // namespace att
#endif
