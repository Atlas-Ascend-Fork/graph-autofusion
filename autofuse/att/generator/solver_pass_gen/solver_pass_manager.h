/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_SOLVER_PASS_MANAGER_H_
#define ATT_SOLVER_PASS_MANAGER_H_
#include <string>
#include <utility>
#include <algorithm>
#include "common/tiling_source_dependencies.h"
#include "base/base_types.h"
#include "generator/preprocess/args_manager.h"
#include "generator/solver_pass/solver.h"
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
#include "util/base_types_printer.h"
#include "autofuse_config/auto_fuse_config.h"
#include "generator/solver_pass_gen/input_output_setters.h"
#include "generator/solver_pass_gen/input_output_setters_mixin.h"
#include "generator/solver_pass_gen/pgo_config_setters_mixin.h"

namespace att {
struct CaseIdInfo {
  uint32_t case_id;
  std::string sub_case_tag = "";
};

class SolverPassManager : public InputOutputSettersMixin<SolverPassManager>,
                          public PgoConfigSettersMixin<SolverPassManager> {
 public:
  SolverPassManager(ArgsManager args_manager, CaseIdInfo case_id_info, const std::string &type_name)
      : args_manager_(args_manager),
        case_id_(case_id_info.case_id),
        sub_case_tag_(case_id_info.sub_case_tag),
        tiling_data_type_(type_name) {}
  static std::string GenCommonBaseClassesHead(std::vector<ArgsManager> args_managers);
  static autofuse::GeneratedCode GenCommonBaseClassesHeader(std::vector<ArgsManager> args_managers);
  static std::string GenCommonBaseClassesFunc(std::vector<ArgsManager> args_managers);
  std::string GenClassPass();
  std::pair<std::string, std::string> GenFuncPass();

  static std::string GenAxesReorderBaseClassesHead(bool enable_equal_order_tiling);
  static std::string GenAxesReorderBaseClassesFunc(bool enable_equal_order_tiling);
  static std::string GenAxesReorderPgoClassesHead(int64_t pgo_step_max);
  static std::string GenAxesReorderPgoClassesFunc();
  std::string GenAxesReorderClass();
  std::pair<std::string, std::string> GenAxesReorderFunc(const std::string &arrange_code);
  void SetUBThreshold(double &ub_threshold) {
    ub_threshold_ = ub_threshold;
  }
  void SetReservedUbSize(const Expr &reserved_ub_size) {
    reserved_ub_size_ = reserved_ub_size;
  };
  void SetCoreNumThreshold(double &corenum_threshold) {
    corenum_threshold_ = corenum_threshold;
  }
  void SetVariableReplace(bool &do_variable_replace) {
    do_variable_replace_ = do_variable_replace;
  }
  void SetHighPerfTiling(bool enable_high_perf) {
    enable_high_perf_ = enable_high_perf;
  }
  void SetEnableEqualOrder(bool enable_equal_order) {
    enable_equal_order_ = enable_equal_order;
  }
  void SetGroupNum(size_t group_num) {
    group_num_ = group_num;
  }

 private:
  // solver pass
  static bool IsNeedSolver(std::vector<ArgsManager> args_managers, SolverType type);
  static std::string GenBaseClass(SolverType type);

  ExprExprMap GetInputsAlign(bool do_replace);
  ExprExprMap GetOriginalInputAlign() const;

  void InitSolverGen(AxesReorderSolverGen &solver_gen);
  AxesReorderSolverGen GenAxesReorderGen();
  template <typename SolverGenType>
  SolverGenType GenerateSolverGen();

  std::string SolverPassClassGen(SolverType type);
  std::string GeneralSolverPassClassGen();

  template <typename SpecificSolverGen>
  std::pair<std::string, std::string> GenerateSolverPassFunc(SpecificSolverGen solver_gen);
  std::pair<std::string, std::string> SolverPassFuncGen(SolverType type);

  std::pair<std::string, std::string> SolverDtFuncGen(SolverType type);
  std::pair<std::string, std::string> GeneralSolverDtFuncGen();

  void AddConcatInnerDims(const Expr &arg, std::vector<Expr> &concat_inner_dims);
  std::string DebugString() const {
    std::stringstream ss;
    ss << "EnableTradeOff:" << enable_multicore_ub_tradeoff_ << ",EnableAutofusePGO:" << enable_autofuse_pgo_
       << ",PGO Step Max:" << pgo_step_max_ << ",HighPerfTiling:" << enable_high_perf_
       << ",EnableEqualOrder:" << enable_equal_order_ << ",ReservedUbSize:" << reserved_ub_size_.Str().get()
       << ",CoreNumThreshold:" << corenum_threshold_ << ",UBThreshold:" << ub_threshold_
       << ",TilingDataSubName:" << GetTilingDataSubGroupItemName() << ",CaseId:" << case_id_
       << ",SubCaseTag:" << sub_case_tag_ << std::endl;
    return ss.str();
  }

  ArgsManager args_manager_;
  uint32_t case_id_;
  std::string sub_case_tag_;
  std::string tiling_data_type_;
  bool do_variable_replace_{false};
  bool enable_high_perf_{false};
  bool enable_equal_order_{false};
  double ub_threshold_{0.5};
  Expr reserved_ub_size_{CreateExpr(0)};
  double corenum_threshold_{0.4};
  size_t group_num_{1UL};
};
}  // namespace att
#endif
