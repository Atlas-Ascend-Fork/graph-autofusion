/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "general_solver_gen.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace att {

bool GeneralSolverGen::IsRelated(const Expr &expr) {
  bool related = false;
  for (const auto &arg : search_args_) {
    if (expr.ContainVar(arg)) {
      related = true;
    }
  }
  return related;
}

void GeneralSolverGen::SetSearchArgs(const std::vector<Expr> &search_args) {
  for (const auto &arg : search_args) {
    search_args_.emplace_back(arg);
  }
}

void GeneralSolverGen::SetInputArgs(const std::vector<Expr> &input_args) {
  for (const auto &arg : input_args) {
    input_args_.emplace_back(arg);
    input_align_[arg] = af::Symbol(1);
  }
}

void GeneralSolverGen::SetSolvedArgs(const std::vector<Expr> &solved_args) {
  for (const auto &arg : solved_args) {
    solved_args_.emplace_back(arg);
  }
}

void GeneralSolverGen::SetConstArgs(const ExprUintMap &const_args) {
  for (const auto &pair : const_args) {
    const_args_[pair.first] = pair.second;
  }
}

void GeneralSolverGen::SetInputAlign(const ExprExprMap &input_align) {
  for (const auto &pair : input_align) {
    input_align_[pair.first] = pair.second;
  }
}

void GeneralSolverGen::SetObj(const std::map<PipeType, Expr> &obj) {
  for (const auto &pair : obj) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter != kPipetypeNameMap.end()) {
      std::string pipetype = iter->second;
      obj_[pipetype] = pair.second;
    }
  }
}

void GeneralSolverGen::SetHeadCost(const Expr &head_cost) {
  head_cost_ = head_cost;
}

void GeneralSolverGen::SetMaxValue(const ExprExprMap &max_value) {
  for (const auto &pair : max_value) {
    max_value_[pair.first] = pair.second;
  }
}

void GeneralSolverGen::SetReplaceVars(const std::vector<std::pair<Expr, Expr>> &replace_vars) {
  for (const auto &var : replace_vars) {
    replace_vars_.emplace_back(var);
  }
}

void GeneralSolverGen::SetExeTimeMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &exe_time_map) {
  for (const auto &pair : exe_time_map) {
    exe_time_map_[pair.first] = pair.second;
  }
}

void GeneralSolverGen::SetMinValue(const ExprExprMap &min_value) {
  for (const auto &pair : min_value) {
    min_value_[pair.first] = pair.second;
  }
}
void GeneralSolverGen::SetInitValue(const ExprExprMap &init_value) {
  for (const auto &pair : init_value) {
    init_value_[pair.first] = pair.second;
  }
}

void GeneralSolverGen::SetBufferCons(const std::map<HardwareDef, Expr> &buffer_cons) {
  Expr penalty;
  Expr remain;
  Expr cons_expr;
  Expr hardware_expr;
  std::string hardware_cost;
  for (const auto &pair : buffer_cons) {
    auto iter = kHardwareNameMap.find(pair.first);
    if (iter != kHardwareNameMap.end()) {
      std::string hardware = iter->second;
      hardware_expr = CreateExpr(hardware.c_str());
      cons_expr = af::sym::Sub(pair.second, hardware_expr);
      // CORENUM is the launch/core budget and must remain an explicit solver
      // constraint even when the expression only depends on input/container
      // variables (and therefore is not searchable by this solver).
      if (IsRelated(cons_expr) || pair.first == HardwareDef::CORENUM) {
        hardware_cost = hardware;
        remain = af::sym::Min(cons_expr, af::sym::kSymbolZero);
        penalty = af::sym::Max(cons_expr, af::sym::kSymbolZero);

        buffer_cost_[hardware_cost] = cons_expr;
        leq_map_[hardware_cost] = leqs_.size();
        leqs_.emplace_back(cons_expr);
        hardware_args_.emplace_back(hardware_expr);
      }
    }
  }
}

void GeneralSolverGen::SetCutCons(const std::vector<Expr> &cut_cons) {
  uint16_t leq_idx = 1;
  std::string leq_cost;
  for (const auto &cons_expr : cut_cons) {
    if (IsRelated(cons_expr)) {
      leq_cost = "leq" + std::to_string(leq_idx++) + "_cost";
      leq_cost_[leq_cost] = cons_expr;
      leq_map_[leq_cost] = leqs_.size();
      leqs_.emplace_back(cons_expr);
    }
  }
}

void GeneralSolverGen::SetExprRelation(const ExprExprMap &expr_relation, const ExprExprMap &vars_relation) {
  for (const auto &pair : expr_relation) {
    expr_relation_[pair.first] = pair.second;
  }
  for (const auto &arg : search_args_) {
    if (!IsValid(arg)) {
      continue;
    }
    if (vars_relation.find(arg) == vars_relation.end()) {
      expr_relation_[arg] = arg;
    }
  }
}

void GeneralSolverGen::SetInnestDim(const std::vector<Expr> &innest_dim) {
  bool has_found;
  for (const auto &arg : search_args_) {
    if (!IsValid(arg)) {
      continue;
    }
    has_found = false;
    for (const auto &innest_arg : innest_dim) {
      if (IsValid(innest_arg) && arg == innest_arg) {
        has_found = true;
        break;
      }
    }
    innest_dim_.push_back(has_found);
  }
}

/*
函数名:FixRange
功能描述:
  定制变量的求解范围,即确定求解过程中变量的上下界
注意:
  目前通用求解器仅支持固定待求解变量而非tiling_data字段的上下界
输入参数:
  var_idx:需要固定的待求解变量的下标
  lower_bound:待求解变量的下界,即待求解变量x>=lower_bound
  upper_bound:待求解变量的上界,即待求解变量x<=upper_bound
*/
void GeneralSolverGen::FixRange(uint64_t var_idx, uint16_t lower_bound, uint16_t upper_bound) {
  if (var_idx < search_args_.size()) {
    max_value_[search_args_[var_idx]] = CreateExpr(upper_bound);
    min_value_[search_args_[var_idx]] = CreateExpr(lower_bound);
  }
}

/*
函数名:FixVar
功能描述:
  将求解变量锁定至固定值
注意:
  目前通用求解器仅支持固定待求解变量而非tiling_data字段
  例如存在tiling_data字段a,待求解变量x0,且有a = pow(2, x0)
  若需要固定a=2,则需要将x固定为1
输入参数:
  var_idx:需要固定的待求解变量的下标
  value:待求解变量的固定值
*/
void GeneralSolverGen::FixVar(uint64_t var_idx, uint32_t value) {
  if (var_idx < search_args_.size()) {
    fixed_args_[var_idx] = value;
    expr_relation_[search_args_[var_idx]] = CreateExpr(value);
  }
}

std::string GeneralSolverGen::GetDoubleVars(const std::string &indent, const std::vector<Expr> &related_expr) {
  std::string codes;
  std::set<std::string> rec;
  std::set<std::string> used_args;
  for (const auto &expr : related_expr) {
    for (const auto &arg : expr.FreeSymbols()) {
      auto iter = exe_time_map_.find(arg);
      if (iter != exe_time_map_.end()) {
        for (const auto &item : iter->second) {
          rec.insert(Str(item));
        }
      }
    }
  }
  for (size_t i = 0u; i < search_args_.size(); i++) {
    std::string arg_name = Str(search_args_[i]);
    if (IsValid(search_args_[i])) {
      if (fixed_args_.find(i) != fixed_args_.end()) {
        continue;
      }
      if (rec.find(arg_name) != rec.end()) {
        used_args.insert(arg_name);
        continue;
      }
      for (const auto &expr : related_expr) {
        if (expr.ContainVar(search_args_[i])) {
          used_args.insert(arg_name);
          break;
        }
      }
    }
  }
  for (const auto &arg : used_args) {
    codes += indent + "double " + arg + " = static_cast<double>(vars[" + arg + "_idx]);\n";
  }
  return codes;
}

std::string GeneralSolverGen::GetUIntVars(const std::string &indent) {
  std::string arg;
  std::string codes;
  for (size_t i = 0u; i < search_args_.size(); i++) {
    if (IsValid(search_args_[i])) {
      if (fixed_args_.find(i) == fixed_args_.end()) {
        arg = Str(search_args_[i]);
        codes += indent + "uint64_t " + arg + " = vars[" + arg + "_idx];\n";
      }
    }
  }
  return codes;
}

std::string GeneralSolverGen::GenArgDef(std::vector<Expr> args) const {
  std::string codes = "";
  std::string key_arg = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  for (const auto &arg : args) {
    if (IsValid(arg)) {
      if (key_arg != Str(arg)) {
        codes += "        uint64_t " + Str(arg) + ";\n";
      } else {
        codes += "        uint64_t " + Str(arg) + "{0};\n";
      }
    }
  }
  return codes;
}

bool GeneralSolverGen::GenVarDef() {
  uint16_t idx = 0;
  for (size_t i = 0u; i < search_args_.size(); i++) {
    if (IsValid(search_args_[i])) {
      if (fixed_args_.find(i) != fixed_args_.end()) {
        impl_codes_ += "        uint64_t " + Str(search_args_[i]) + "{" + std::to_string(fixed_args_[i]) + "};\n";
      } else {
        impl_codes_ += "        const int64_t " + Str(search_args_[i]) + "_idx = " + std::to_string(idx++) + ";\n";
      }
    }
  }
  for (const auto &pair : const_args_) {
    if (IsValid(pair.first)) {
      impl_codes_ += "        uint64_t " + GetSymbolName(pair.first) + "{" + std::to_string(pair.second) + "};\n";
    }
  }
  impl_codes_ += GenArgDef(input_args_);
  impl_codes_ += GenArgDef(hardware_args_);
  impl_codes_ += GenArgDef(solved_args_);
  return true;
}

bool GeneralSolverGen::GenFuncDef() {
  impl_codes_ += "        double GetObj(uint64_t* vars);\n";
  impl_codes_ += "        double GetSmoothObj(uint64_t* vars);\n";
  impl_codes_ += "        double GetBuffCost(uint64_t* vars);\n";
  impl_codes_ += "        bool CheckLocalValid(double* leqs, int32_t idx);\n";
  impl_codes_ += "        void DisplayVarVal(uint64_t* vars);\n";
  impl_codes_ += "        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs);\n";
  impl_codes_ += "        double GetBuffDiff(uint64_t* vars, double* weight);\n";
  impl_codes_ += "        double GetLeqDiff(uint64_t* vars, double* weight);\n";

  for (const auto &pair : buffer_cost_) {
    impl_codes_ += "        double Get" + pair.first + "Cost(uint64_t* vars);\n";
    impl_codes_ += "        double GetSmooth" + pair.first + "Cost(uint64_t* vars);\n";
  }
  impl_codes_ += "        void MapVarVal(uint64_t* vars, " + type_name_ + "& tiling_data);\n";
  impl_codes_ +=
      ("        void GetResult(int32_t solution_num, uint64_t* solution, " + type_name_ + "& tiling_data);\n");

  impl_codes_ += "        bool Init(const SolverInput &input);\n";
  return true;
}

std::string GeneralSolverGen::InitiateArgs(std::vector<Expr> args) const {
  std::string codes = "";
  std::string key_arg = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  for (const auto &arg : args) {
    if (IsValid(arg)) {
      if (key_arg != Str(arg)) {
        codes += "            " + Str(arg) + " = tiling_data.get_" + Str(arg) + "();\n";
      }
    }
  }
  return codes;
}

std::string GeneralSolverGen::GenAlignInput(const Expr arg, const std::string indent) {
  std::string codes = "";
  std::string input_align;
  auto input_align_expr = input_align_[arg];

  if (IsValid(arg) && (!input_align_expr.IsConstExpr() ||
                       (input_align_expr.IsConstExpr() &&
                        af::SymbolicUtils::StaticCheckNe(input_align_expr, af::Symbol(1)) == af::TriBool::kTrue))) {
    input_align = ascgen_utils::RuntimeAlignExprToCode(Str(input_align_expr));
    codes += indent + Str(arg) + " = ((" + Str(arg) + " + " + input_align + " - 1) / " + input_align + ") * " +
             input_align + ";\n";
  }
  return codes;
}

std::string GeneralSolverGen::GenClassAnotataion() {
  uint16_t idx = 0;
  std::string strs = "";
  strs += "Users can override the Run function in a derived class to construct a custom solving algorithm:\n";
  strs += "  void bool Run(int32_t &solution_num, uint64_t *solutions) override;\n";
  strs += "where:\n";
  strs += "  solution_num: an int32_t parameter for the actual number of solutions found\n";
  strs +=
      "  solutions: a uint64_t array pointing to num_var * top_num elements where the algorithm stores feasible "
      "solutions\n";
  strs += "The Run function can use the following helper functions:\n";
  strs += "  bool CheckValid()\n";
  strs += "    Checks whether the current solution is feasible\n";
  strs += "  bool UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  strs += "    Sets the variable at index idx to value and updates cons_info_->leqs\n";
  strs += "  bool RecordBestVarVal()\n";
  strs += "    Optimizes the objective function for the current variable values\n";
  strs += "The Run function can use the following parameters:\n";
  strs +=
      "  cons_info_->leqs, a double array storing inequality constraint values; its indices are defined as follows:\n";
  for (size_t j = 0u; j < leqs_.size(); j++) {
    strs += "    cons_info_->leqs[" + std::to_string(j) + "] = " + Str(leqs_[j]) + "\n";
  }
  strs +=
      "  var_info_->cur_vars, a uint64_t array storing current variable values; its indices are defined as follows:\n";
  for (size_t j = 0u; j < search_args_.size(); j++) {
    if (fixed_args_.find(j) != fixed_args_.end()) {
      strs += "    var_info_->cur_vars[" + std::to_string(idx++) + "] = " + Str(search_args_[j]) + "\n";
    }
  }
  strs += "  var_info_->upper_bound, a uint64_t array storing upper bounds for the variables\n";
  strs += "  var_info_->lower_bound, a uint64_t array storing lower bounds for the variables\n";
  return AddAnotationBlock(strs);
}

bool GeneralSolverGen::GenClassDef() {
  std::string str_arg;
  std::string construct_func;
  impl_codes_ += GenClassAnotataion();
  impl_codes_ += "class GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += " : public GeneralSolver<";
  impl_codes_ += "GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += ">\n";
  impl_codes_ += "{\n";

  impl_codes_ += "    public:\n";
  construct_func = "explicit GeneralSolver";
  construct_func += tiling_case_id_;
  construct_func += "(SolverConfig& config, " + type_name_ + "& tiling_data) {\n";
  construct_func += "            case_id_ = \"" + tiling_case_id_ + "\";\n";
  construct_func += "            solver_config_ = config;\n";
  construct_func += InitiateArgs(input_args_);
  construct_func += InitiateArgs(hardware_args_);
  construct_func += InitiateArgs(solved_args_);
  for (const auto &arg : input_args_) {
    construct_func += GenAlignInput(arg, "            ");
  }
  construct_func += "        }\n";
  impl_codes_ += "        " + construct_func + "\n";
  GenFuncDef();
  impl_codes_ += "    private:\n";
  GenVarDef();
  impl_codes_ += "};\n";
  return true;
}

Expr GeneralSolverGen::GenFuncExpr(FuncType func_type) {
  Expr penalty;
  Expr expression;
  Expr expr;
  if (func_type == FuncType::OBJ) {
    for (const auto &pair : obj_) {
      expression = CreateExpr(pair.first.c_str());
      if (IsValid(expr)) {
        expr = af::sym::Max(expr, expression);
      } else {
        expr = expression;
      }
    }
    expr = af::sym::Add(expr, head_cost_);
  } else if (func_type == FuncType::BUFFER) {
    expr = af::sym::kSymbolZero;
    for (const auto &pair : buffer_cost_) {
      expression = CreateExpr((pair.first + "_cost").c_str());
      penalty = af::sym::Min(expression, af::sym::kSymbolZero);
      expr = af::sym::Add(expr, af::sym::Mul(penalty, penalty));
    }
  }
  return expr;
}

bool GeneralSolverGen::GenBuffFunc() {
  std::string strs;
  for (const auto &pair : buffer_cost_) {
    strs = "";
    strs += "Function: Get" + pair.first + "Cost(important)\n";
    strs += "Description:\n";
    strs += "  Gets cache occupancy information (occupy-buff) from " + pair.first + "\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
    impl_codes_ += AddAnotationBlock(strs);
    impl_codes_ += "inline double GeneralSolver";
    impl_codes_ += tiling_case_id_;
    impl_codes_ += "::Get" + pair.first + "Cost(uint64_t* vars)\n";
    impl_codes_ += "{\n";
    impl_codes_ += GetDoubleVars("    ", {pair.second});
    impl_codes_ += "    return " + Str(pair.second) + ";\n";
    impl_codes_ += "}\n";
    impl_codes_ += "\n";

    strs = "";
    strs += "Function: GetSmooth" + pair.first + "Cost(important)\n";
    strs += "Description:\n";
    strs += "  Gets smoothed cache occupancy information from " + pair.first + "\n";
    strs += "  Compared with Get" + pair.first + "Cost, integer division is replaced with floating-point division\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
    impl_codes_ += AddAnotationBlock(strs);
    impl_codes_ += "inline double GeneralSolver";
    impl_codes_ += tiling_case_id_;
    impl_codes_ += "::GetSmooth" + pair.first + "Cost(uint64_t* vars)\n";
    impl_codes_ += "{\n";
    impl_codes_ += GetDoubleVars("    ", {pair.second});
    impl_codes_ += "    return " + GetSmoothString(Str(pair.second)) + ";\n";
    impl_codes_ += "}\n";
    impl_codes_ += "\n";
  }
  return true;
}

bool GeneralSolverGen::GenSubFunc(const std::map<std::string, Expr> &funcs) {
  for (const auto &pair : funcs) {
    if (!IsValid(pair.second)) {
      continue;
    }
    Expr expr = pair.second.Replace(replace_vars_);
    impl_codes_ += "    double " + pair.first + " = " + Str(expr) + ";\n";
    impl_codes_ += "    OP_LOGD(OP_NAME, \"" + pair.first + " = %f\", " + pair.first + ");\n";
    impl_codes_ += "    OP_LOGD(OP_NAME, \"The expression of " + pair.first + " is " + Str(expr) + "\");\n";
  }
  return true;
}

bool GeneralSolverGen::GenBuffExpr() {
  for (const auto &pair : buffer_cost_) {
    impl_codes_ += "    double " + pair.first + "_cost = Get" + pair.first + "Cost(vars);\n";
  }
  return true;
}

std::string GeneralSolverGen::GenAnnotation(FuncType func_type) const {
  std::string strs = "";
  if (func_type == FuncType::OBJ) {
    strs += "Function: GetObj(important)\n";
    strs += "Description:\n";
    strs += "  Outputs the objective function for the variable values\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
  } else if (func_type == FuncType::BUFFER) {
    strs += "Function: GetBuffCost(important)\n";
    strs += "Description:\n";
    strs += "  Outputs the cache occupancy penalty function (sigma(min(0, occupy-buff)^2))\n";
    strs += "  Quantifies solution quality in terms of cache occupancy\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
  }
  return AddAnotationBlock(strs);
}

bool GeneralSolverGen::GenGetFunc(FuncType func_type) {
  std::string func_name;
  Expr expr;
  std::vector<Expr> funcs;
  if (func_type == FuncType::OBJ) {
    func_name = "GetObj";
  } else if (func_type == FuncType::BUFFER) {
    func_name = "GetBuffCost";
  }
  expr = GenFuncExpr(func_type);
  impl_codes_ += GenAnnotation(func_type);
  impl_codes_ += "inline double GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::";
  impl_codes_ += func_name;
  impl_codes_ += "(uint64_t* vars)\n";
  impl_codes_ += "{\n";
  if (IsValid(expr)) {
    if (func_type == FuncType::OBJ) {
      // 不可能没有
      if (buffer_cost_.find("block_dim") != buffer_cost_.end()) {
        impl_codes_ += "    double block_dim = Getblock_dimCost(vars) + this->block_dim;\n";
      } else {
        impl_codes_ += "    double block_dim = 1;\n";
      }
      for (const auto &pair : obj_) {
        if (!IsValid(pair.second)) {
          continue;
        }
        funcs.emplace_back(pair.second);
      }
      funcs.emplace_back(head_cost_);
      impl_codes_ += GetDoubleVars("    ", funcs);
      GenSubFunc(obj_);
    } else if (func_type == FuncType::BUFFER) {
      GenBuffExpr();
    }
    impl_codes_ += "    return " + Str(expr) + ";\n";
  } else {
    impl_codes_ += "    return 0;\n";
  }
  impl_codes_ += "}\n";
  return true;
}

bool GeneralSolverGen::GenGetSmoothObj() {
  Expr expr = GenFuncExpr(FuncType::OBJ);
  std::string strs = "";
  std::string pipe_strs = "";
  std::vector<Expr> related_expr;
  strs += "Function: GetSmoothObj(important)\n";
  strs += "Description:\n";
  strs += "  Outputs the smoothed objective function for the variable values\n";
  strs += "  Compared with GetObj, integer division is replaced with floating-point division\n";
  impl_codes_ += AddAnotationBlock(strs);
  impl_codes_ += "inline double GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::GetSmoothObj(uint64_t* vars)\n";
  impl_codes_ += "{\n";
  if (IsValid(expr)) {
    if (buffer_cost_.find("block_dim") != buffer_cost_.end()) {
      impl_codes_ += "    double block_dim = Getblock_dimCost(vars) + this->block_dim;\n";
    } else {
      impl_codes_ += "    double block_dim = 1;\n";
    }
    for (const auto &pair : obj_) {
      if (!IsValid(pair.second)) {
        continue;
      }
      related_expr.emplace_back(pair.second);
      pipe_strs +=
          "    double " + pair.first + " = " + GetSmoothString(Str(pair.second.Replace(replace_vars_))) + ";\n";
    }
    related_expr.emplace_back(head_cost_);
    impl_codes_ += GetDoubleVars("    ", related_expr);
    impl_codes_ += pipe_strs;
    impl_codes_ += "    return " + Str(expr) + ";\n";
  } else {
    impl_codes_ += "    return 0;\n";
  }
  impl_codes_ += "}\n";
  return true;
}

std::string GeneralSolverGen::GenDiffAnnotation(FuncType func_type) const {
  std::string strs = "";
  if (func_type == FuncType::BUFFER) {
    strs += "Function: GetBuffDiff(important)\n";
    strs += "Description:\n";
    strs += "  Gets the weighted cache occupancy difference for smooth cache occupancy\n";
    strs += "  The formula is sigma_j(delta_{var_i}(g_j(var))) * g_j(var))\n";
    strs +=
        "  where g_j is the j-th cache occupancy inequality, and delta_{var_i}(g_j(var)) is the change in g_j(var) "
        "when var_i increases by one unit\n";
    strs += "  Determines the update direction that increases cache occupancy\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
    strs += "  weight:an array of length num_leq representing the weight of each cache occupancy\n";
  } else if (func_type == FuncType::LEQ) {
    strs += "Function: GetLeqDiff(important)\n";
    strs += "Description:\n";
    strs +=
        "  Gets the weighted difference of inequality constraints; the weight is the actual inequality function "
        "value\n";
    strs += "  The formula is sigma_j(delta_{var_i}(f_j(var))) * f_j(var))\n";
    strs +=
        "  where f_j is the j-th inequality constraint, and delta_{var_i}(f_j(var)) is the change in f_j(var) when "
        "var_i increases by one unit\n";
    strs += "  Determines the update direction from outside the feasible region toward the inequality boundary\n";
    strs += "Input parameters:\n";
    strs += "  vars:an array of length num_var corresponding to the variables\n";
    strs += "  weight:an array of length num_leq representing the weight of each cache occupancy\n";
  }
  return AddAnotationBlock(strs);
}

std::string GeneralSolverGen::GetWeightedDiff(FuncType func_type) {
  std::string idx;
  std::string ret_expr = "";
  std::vector<Expr> funcs;
  std::vector<std::string> ret_exprs;
  if (func_type == FuncType::LEQ) {
    for (const auto &pair : leq_cost_) {
      if (!IsValid(pair.second)) {
        continue;
      }
      funcs.emplace_back(pair.second);
    }
    impl_codes_ += GetDoubleVars("    ", funcs);
  }
  for (const auto &pair : buffer_cost_) {
    idx = std::to_string(leq_map_[pair.first]);
    if (func_type == FuncType::LEQ) {
      impl_codes_ += "    double " + pair.first + "_cost = weight[" + idx + "] > 0 ? " + "GetSmooth" + pair.first +
                     "Cost(vars) * weight[" + idx + "] : 0;\n";
    } else {
      impl_codes_ += "    double " + pair.first + "_cost = weight[" + idx + "] < 0 ? " + "GetSmooth" + pair.first +
                     "Cost(vars) * weight[" + idx + "] : 0;\n";
    }
    ret_exprs.emplace_back(pair.first + "_cost");
  }
  if (func_type == FuncType::LEQ) {
    for (const auto &pair : leq_cost_) {
      idx = std::to_string(leq_map_[pair.first]);
      impl_codes_ += "    double " + pair.first + " = weight[" + idx + "] > 0 ? " + GetSmoothString(Str(pair.second)) +
                     " * weight[" + idx + "] : 0;\n";
      ret_exprs.emplace_back(pair.first);
    }
  }
  if (ret_exprs.size() == 0) {
    ret_expr = "0";
  } else {
    for (size_t i = 0; i < ret_exprs.size(); ++i) {
      ret_expr += ((i != 0) ? " + " : "") + ret_exprs[i];
    }
  }
  return ret_expr;
}

bool GeneralSolverGen::GenGetWeightedDiff(FuncType func_type) {
  std::string func_name;
  if (func_type == FuncType::BUFFER) {
    func_name = "GetBuffDiff";
  } else if (func_type == FuncType::LEQ) {
    func_name = "GetLeqDiff";
  } else {
    return false;
  }
  impl_codes_ += GenDiffAnnotation(func_type);
  impl_codes_ += "inline double GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::";
  impl_codes_ += func_name;
  impl_codes_ += "(uint64_t* vars, double* weight)\n";
  impl_codes_ += "{\n";
  std::string ret_expr = GetWeightedDiff(func_type);
  impl_codes_ += "    return " + ret_expr + ";\n";
  impl_codes_ += "}\n";
  return true;
}

bool GeneralSolverGen::GetLeqInfo(int32_t i, std::string &local_valid_func, std::string &update_func) {
  std::string temp;
  if (i == 0) {
    temp = "    if (idx == " + Str(search_args_[i]) + "_idx) {\n";
  } else {
    temp = "    } else if (idx == " + Str(search_args_[i]) + "_idx) {\n";
  }
  local_valid_func += temp;
  update_func += temp;
  temp = "";
  for (size_t j = 0u; j < leqs_.size(); j++) {
    if (IsValid(leqs_[j])) {
      if (leqs_[j].ContainVar(search_args_[i])) {
        temp += temp.empty() ? "" : " && ";
        temp += "leqs[" + std::to_string(j) + "] <= 0";
        update_func += "        leqs[" + std::to_string(j) + "] = " + Str(leqs_[j]) + ";\n";
      }
    }
  }
  if (temp.size() > 0) {
    local_valid_func += "        return " + temp + ";\n";
  } else {
    local_valid_func += "        return true;\n";
  }
  return true;
}

bool GeneralSolverGen::GenLeqInfo() {
  std::string local_valid_func;
  std::string update_func;
  std::string update_strs;
  std::vector<Expr> funcs;
  local_valid_func = "inline bool GeneralSolver";
  local_valid_func += tiling_case_id_;
  local_valid_func += "::CheckLocalValid(double* leqs, int32_t idx)\n{\n";
  update_func = "inline void GeneralSolver";
  update_func += tiling_case_id_;
  update_func += "::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)\n{\n";
  for (size_t i = 0u; i < search_args_.size(); i++) {
    if (fixed_args_.find(i) == fixed_args_.end()) {
      GetLeqInfo(i, local_valid_func, update_strs);
    }
  }
  update_strs += "    } else if (idx == -1) {\n";
  for (size_t j = 0u; j < leqs_.size(); j++) {
    if (!IsValid(leqs_[j])) {
      continue;
    }
    funcs.emplace_back(leqs_[j]);
    update_strs += "        leqs[" + std::to_string(j) + "] = " + Str(leqs_[j]) + ";\n";
  }
  update_func += GetDoubleVars("    ", funcs);
  update_func += update_strs;
  local_valid_func += "    }\n";
  local_valid_func += "    return true;\n}\n\n";
  update_func += "    }\n}\n\n";
  impl_codes_ += local_valid_func;
  impl_codes_ += update_func;
  return true;
}

bool GeneralSolverGen::GenMapVarVal() {
  impl_codes_ += "inline void GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::MapVarVal(uint64_t* vars, " + type_name_ + "& tiling_data)\n{\n";
  impl_codes_ += GetUIntVars("    ");
  impl_codes_ += "    OP_LOGD(OP_NAME, \"The output of the solver for tilingCaseId " + tiling_case_id_ + " is:\");\n";
  for (const auto &pair : expr_relation_) {
    if (!IsValid(pair.first) || !IsValid(pair.second)) {
      continue;
    }
    impl_codes_ += "    tiling_data.set_" + Str(pair.first) + "(static_cast<uint64_t>(" + Str(pair.second) + "));\n";
    impl_codes_ += "    OP_LOGD(OP_NAME, \"" + Str(pair.first) +
                   " = %llu\", static_cast<unsigned long long>(tiling_data.get_" + Str(pair.first) + "()));\n";
  }
  impl_codes_ += "}\n\n";
  return true;
}

bool GeneralSolverGen::GenInit() {
  impl_codes_ += "inline bool GeneralSolver" + tiling_case_id_ + "::Init(const SolverInput &input) {\n";
  std::set<std::string> arg_list;
  for (const auto &arg : hardware_args_) {
    arg_list.insert(Str(arg));
  }
  for (const auto &arg : input_args_) {
    arg_list.insert(Str(arg));
  }
  std::string block_arg = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  if (arg_list.find(block_arg) != arg_list.end()) {
    impl_codes_ += "    " + block_arg + " = input.corenum;\n";
  }
  impl_codes_ += "    return GeneralSolver::Init(input);\n";
  impl_codes_ += "}\n";
  return true;
}

bool GeneralSolverGen::GenDisplayVarVal() {
  impl_codes_ += "inline void GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::DisplayVarVal(uint64_t* vars)\n{\n";
  impl_codes_ += GetUIntVars("    ");
  for (const auto &pair : expr_relation_) {
    if (!IsValid(pair.first) || !IsValid(pair.second)) {
      continue;
    }
    impl_codes_ +=
        "    OP_LOGD(OP_NAME, \"" + Str(pair.first) + " = %lu\", static_cast<uint64_t>(" + Str(pair.second) + "));\n";
  }
  impl_codes_ += "}\n\n";
  return true;
}

bool GeneralSolverGen::GenGetResult() {
  impl_codes_ += "inline void GeneralSolver";
  impl_codes_ += tiling_case_id_;
  impl_codes_ += "::GetResult(int32_t solution_num, uint64_t* solution, " + type_name_ + "& tiling_data)\n{\n";
  impl_codes_ += "    if (solution_num > 0) {\n";
  impl_codes_ += "        OP_LOGD(OP_NAME, \"Filling tilingdata for " + tiling_case_id_ + ".\");\n";
  impl_codes_ += "        OP_LOGD(OP_NAME, \"Estimate the occupy.\");\n";
  for (const auto &pair : buffer_cost_) {
    impl_codes_ += "        OP_LOGD(OP_NAME, \"" + pair.first + " = %ld\", static_cast<uint64_t>(Get" + pair.first +
                   "Cost(solution) + " + pair.first + "));\n";
  }
  impl_codes_ += "        OP_LOGD(OP_NAME, \"Simulate the cost.\");\n";
  impl_codes_ +=
      "        OP_LOGD(OP_NAME, \"Objective value for " + tiling_case_id_ + " is %f.\", GetObj(solution));\n";
  impl_codes_ += "        MapVarVal(solution, tiling_data);\n";
  impl_codes_ += "    }\n";
  impl_codes_ += "}\n\n";
  return true;
}

std::string GeneralSolverGen::GenDTInit() {
  std::string codes;
  // codes += "  #ifndef DISABLE_DT\n";
  uint32_t input_num = 0u;
  for (const auto &arg : input_args_) {
    if (IsValid(arg)) {
      input_num += 1u;
    }
  }
  codes += "    std::array<uint64_t, " + std::to_string(input_num) + "> feature_vector;\n";
  codes += "    std::array<uint64_t, " + std::to_string(search_args_.size()) + "> dt_outputs;\n";
  uint32_t idx = 0u;
  for (const auto &arg : input_args_) {
    if (IsValid(arg)) {
      codes += "    feature_vector[" + std::to_string(idx) + "] = tiling_data.get_" + Str(arg) + "();\n";
      idx++;
    }
  }
  codes += "    tiling" + tiling_case_id_ + "::AttDTInit(feature_vector, dt_outputs);\n";
  for (uint32_t i = 0u; i < search_args_.size(); i++) {
    codes += "    init_vars[" + std::to_string(i) + "] = std::min(std::max(dt_outputs[" + std::to_string(i) +
             "], lower_bound[" + std::to_string(i) + "]), upper_bound[" + std::to_string(i) + "]);\n";
  }
  // codes += "  #endif\n";
  return codes;
}

std::string GeneralSolverGen::InitiateValue() {
  const uint32_t init_offset = 4;
  std::vector<std::string> upper_expr;
  std::vector<std::string> lower_expr;
  std::vector<std::string> init_expr;
  std::vector<std::string> update_last;
  std::string codes;
  size_t num_var = search_args_.size();
  for (size_t i = 0u; i < num_var; i++) {
    if (!IsValid(min_value_[search_args_[i]]) || !IsValid(max_value_[search_args_[i]])) {
      continue;
    }
    if (fixed_args_.find(i) == fixed_args_.end()) {
      std::string max_value_str = "static_cast<uint64_t>(" + Str(max_value_[search_args_[i]]) + ")";
      upper_expr.emplace_back(max_value_str);
      lower_expr.emplace_back("static_cast<uint64_t>(" + Str(min_value_[search_args_[i]]) + ")");
      init_expr.emplace_back(
          init_value_.empty() ? max_value_str : ("static_cast<uint64_t>(" + Str(init_value_[search_args_[i]]) + ")"));
      update_last.emplace_back(innest_dim_[i] ? "true" : "false");
    }
  }
  codes += AddAnotationLine(
      "Configurable:Variable upper bounds; overly large bounds increase search range and time, while overly small "
      "bounds may produce a worse local optimum\n",
      "    ");
  for (size_t i = 0u; i < upper_expr.size(); i++) {
    codes += "    uint_space[" + std::to_string(i) + "] = " + upper_expr[i] + ";\n";
  }
  codes += AddAnotationLine(
      "Configurable:Variable lower bounds; overly small bounds increase search range and time, while overly large "
      "bounds may produce a worse local optimum\n",
      "    ");
  for (size_t i = 0u; i < lower_expr.size(); i++) {
    codes += "    uint_space[" + std::to_string(i + upper_expr.size()) + "] = " + lower_expr[i] + ";\n";
    if (!min_value_[search_args_[i]].IsConstExpr() || !max_value_[search_args_[i]].IsConstExpr()) {
      codes += "    if (" + lower_expr[i] + " > " + upper_expr[i] + ") {\n";
      codes += "        OP_LOGW(OP_NAME, \"Lower_bound[" + std::to_string(i) + "] is larger than upper_bound[" +
               std::to_string(i) + "].\");\n";
      codes += "        return false;\n";
      codes += "    }\n";
    }
  }
  codes += AddAnotationLine(
      "Configurable:Initial variable values; the algorithm tends to find a local optimum near them\n", "    ");
  for (size_t i = 0u; i < init_expr.size(); i++) {
    codes += "    uint_space[" + std::to_string(i + init_offset * upper_expr.size()) + "] = " + init_expr[i] + ";\n";
  }
  codes += "    uint64_t* upper_bound = uint_space;\n";
  codes += "    uint64_t* lower_bound = uint_space + " + std::to_string(num_var) + ";\n";
  codes += "    uint64_t* init_vars = uint_space + " + std::to_string(init_offset * num_var) + ";\n";
  if (open_dt_ && !training_) {
    codes += GenDTInit();
  }
  codes += AddAnotationLine(
      "Configurable:Last updated variables; variables set to true stay closer to their initial values\n", "    ");
  for (size_t i = 0u; i < update_last.size(); i++) {
    codes += "    bool_space[" + std::to_string(i) + "] = " + update_last[i] + ";\n";
  }
  return codes;
}

bool GeneralSolverGen::CheckUseful(Expr arg) {
  bool used = false;
  for (const auto &pair : min_value_) {
    if (pair.second.ContainVar(arg)) {
      used = true;
    }
  }
  for (const auto &pair : max_value_) {
    if (pair.second.ContainVar(arg)) {
      used = true;
    }
  }
  for (const auto &pair : init_value_) {
    if (pair.second.ContainVar(arg)) {
      used = true;
    }
  }
  return used;
}

std::string GeneralSolverGen::InitiateDefArgs(std::vector<Expr> args) {
  std::string codes = "";
  for (const auto &arg : args) {
    if (CheckUseful(arg)) {
      codes += "    uint64_t " + Str(arg) + " = tiling_data.get_" + Str(arg) + "();\n";
    }
  }
  return codes;
}

std::string GeneralSolverGen::InitiateDefInputs() {
  std::string codes = "";
  for (const auto &arg : input_args_) {
    if (IsValid(arg) && CheckUseful(arg)) {
      codes += "    uint64_t " + Str(arg) + " = tiling_data.get_" + Str(arg) + "();\n";
      codes += GenAlignInput(arg, "    ");
    }
  }
  return codes;
}

std::string GeneralSolverGen::InitiateMemoryPool() const {
  std::string codes;
  codes += "    size_t uint_size = 6 * static_cast<size_t>(num_var) * sizeof(uint64_t);\n";
  codes += "    size_t double_size = 2 * static_cast<size_t>(num_leq + num_var) * sizeof(double);\n";
  codes += "    size_t bool_size = 2 * static_cast<size_t>(num_var) * sizeof(bool);\n";
  codes += "    size_t VarVal_size = sizeof(VarVal) + (sizeof(uint64_t) * static_cast<size_t>(num_var));\n";
  codes += "    size_t total_VarVal_size = static_cast<size_t>(2 * cfg_top_num + 1) * VarVal_size;\n";
  codes += "    size_t ret_size = static_cast<size_t>(num_var * cfg_top_num) * sizeof(uint64_t);\n";
  codes += "    size_t visited_size = static_cast<size_t>(num_var * cfg_iterations) * sizeof(uint64_t);\n";
  codes +=
      "    void* memory_pool = calloc(1, uint_size + double_size + bool_size"
      " + sizeof(VarInfo) + sizeof(ConsInfo)"
      " + sizeof(Momentum) + total_VarVal_size + sizeof(Result)"
      " + ret_size + visited_size + sizeof(VisitedNode));\n";
  codes += "    if (memory_pool == nullptr) {\n";
  codes += "        OP_LOGE(OP_NAME, \"Failed to allocate memory pool for solver.\");\n";
  codes += "        return false;\n";
  codes += "    }\n";
  codes += "    size_t offset_uint = 0;\n";
  codes += "    size_t offset_double = offset_uint + uint_size;\n";
  codes += "    size_t offset_bool = offset_double + double_size;\n";
  codes += "    size_t offset_var_info = offset_bool + bool_size;\n";
  codes += "    size_t offset_cons_info = offset_var_info + sizeof(VarInfo);\n";
  codes += "    size_t offset_momentum = offset_cons_info + sizeof(ConsInfo);\n";
  codes += "    size_t offset_varVal = offset_momentum + sizeof(Momentum);\n";
  codes += "    size_t offset_temp = offset_varVal + VarVal_size;\n";
  codes += "    size_t offset_solution = offset_temp + cfg_top_num * VarVal_size;\n";
  codes += "    size_t offset_result = offset_solution + cfg_top_num * VarVal_size;\n";
  codes += "    size_t offset_ret = offset_result + sizeof(Result);\n";
  codes += "    size_t offset_visited = offset_ret + ret_size;\n";
  codes += "    size_t offset_node = offset_visited + visited_size;\n";
  return codes;
}

std::string GeneralSolverGen::GenMemoryPool() {
  std::string codes;
  codes += InitiateMemoryPool();
  codes += "    uint64_t* uint_space = (uint64_t*)((char*)memory_pool + offset_uint);\n";
  codes += "    double* double_space = (double*)((char*)memory_pool + offset_double);\n";
  codes += "    bool* bool_space = (bool*)((char*)memory_pool + offset_bool);\n";
  codes += InitiateValue();
  codes += "    VarInfo* var_info = (VarInfo*)((char*)memory_pool + offset_var_info);\n";
  codes += "    ConsInfo* cons_info = (ConsInfo*)((char*)memory_pool + offset_cons_info);\n";
  codes += "    Momentum* momentum = (Momentum*)((char*)memory_pool + offset_momentum);\n";
  codes += "    VarVal* varval;\n";
  codes += "    size_t offset;\n";
  codes += "    for (uint64_t i = 0u; i < 2 * cfg_top_num + 1; i++) {\n";
  codes += "        offset = offset_varVal + i * VarVal_size;\n";
  codes += "        varval = (VarVal*)((char*)memory_pool + offset);\n";
  codes += "        varval->var_num = num_var;\n";
  codes += "        varval->vars = (uint64_t*)((char*)memory_pool + offset + sizeof(VarVal));\n";
  codes += "    }\n";
  codes += "    Result* result = (Result*)((char*)memory_pool + offset_result);\n";
  codes += "    uint64_t* solution = (uint64_t*)((char*)memory_pool + offset_ret);\n";
  codes += "    uint64_t* visited_head = (uint64_t*)((char*)memory_pool + offset_visited);\n";
  codes += "    VisitedNode* visited_node = (VisitedNode*)((char*)memory_pool + offset_node);\n";

  codes += "    var_info->SetVarInfo(num_var, uint_space, bool_space);\n";
  codes += "    cons_info->SetConsInfo(num_leq, double_space);\n";
  codes += "    momentum->SetMomentum(num_var, num_leq, double_space, bool_space);\n";
  codes +=
      "    result->SetResult(cfg_top_num, num_var, (VarVal*)((char*)memory_pool + offset_varVal),"
      "((char*)memory_pool + offset_temp), ((char*)memory_pool + offset_solution));\n";
  codes += "    visited_node->SetVisitedNode(num_var, visited_head);\n";
  return codes;
}

bool GeneralSolverGen::CreateInput() {
  uint32_t idx = 0;
  std::string add_log;
  std::string arg_name;
  std::string search_arg_str;
  invoke_codes_ += AddAnotationLine("Do not modify parameters unless marked as configurable\n", "    ");
  invoke_codes_ += InitiateDefInputs();
  invoke_codes_ += InitiateDefArgs(hardware_args_);
  invoke_codes_ += InitiateDefArgs(solved_args_);
  invoke_codes_ += AddAnotationLine("Number of variables passed from modelinfo\n", "    ");
  invoke_codes_ += "    int32_t num_var = " + std::to_string(search_args_.size() - fixed_args_.size()) + ";\n";
  invoke_codes_ += AddAnotationLine("Number of inequality constraints passed from modelinfo\n", "    ");
  invoke_codes_ += "    int32_t num_leq = " + std::to_string(leqs_.size()) + ";\n";
  for (size_t i = 0u; i < search_args_.size(); i++) {
    if (IsValid(search_args_[i])) {
      if (fixed_args_.find(i) == fixed_args_.end()) {
        arg_name = Str(search_args_[i]);
        search_arg_str += (idx == 0 ? "" : ", ") + arg_name;
        add_log += "    OP_LOGD(OP_NAME, \"" + arg_name + "->init value: %lu, range: [%lu, %lu].\", init_vars[" +
                   std::to_string(idx) + "], lower_bound[" + std::to_string(idx) + "], upper_bound[" +
                   std::to_string(idx) + "]);\n";
        ++idx;
      }
    }
  }
  invoke_codes_ += "    OP_LOGD(OP_NAME, \"The number of variable is %d(" + search_arg_str +
                   "), the number of constraints is %d.\", num_var, num_leq);\n";
  invoke_codes_ += AddAnotationLine("Initialize the number of solutions to 0\n", "    ");
  invoke_codes_ += "    int32_t solution_num = 0;\n";
  invoke_codes_ += GenMemoryPool();
  invoke_codes_ += AddAnotationLine("Generic solver input parameters\n", "    ");
  invoke_codes_ += "    SolverInput input;\n";
  invoke_codes_ += "    input.corenum = corenum_;\n";
  invoke_codes_ += "    input.var_info = var_info;\n";
  invoke_codes_ += "    input.cons_info = cons_info;\n";
  invoke_codes_ += "    input.momentum = momentum;\n";
  invoke_codes_ += "    input.result = result;\n";
  invoke_codes_ += "    input.visited_node = visited_node;\n";
  invoke_codes_ += add_log;
  invoke_codes_ += "\n";
  return true;
}

bool GeneralSolverGen::RunSolver(bool is_dt) {
  std::string class_name = "GeneralSolver" + tiling_case_id_;
  invoke_codes_ +=
      "    std::shared_ptr<" + class_name + "> solver = std::make_shared<" + class_name + ">(cfg, tiling_data);\n";

  invoke_codes_ += "    if (solver != nullptr) {\n";
  invoke_codes_ += AddAnotationLine("Import and initialize generic solver input parameters\n", "        ");
  invoke_codes_ += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  invoke_codes_ += "        if (solver -> Init(input)) {\n";
  invoke_codes_ += AddAnotationLine("Run the generic solver and obtain algorithm solutions\n", "            ");
  invoke_codes_ += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  invoke_codes_ += "            if (solver -> Run(solution_num, solution)) {\n";
  invoke_codes_ += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  if (is_dt) {
    for (uint32_t i = 0; i < search_args_.size(); i++) {
      invoke_codes_ += "                output_tilings.emplace_back(solution[" + std::to_string(i) + "]);\n";
    }
  }
  invoke_codes_ += "                free(memory_pool);\n";
  invoke_codes_ += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  invoke_codes_ += "                return true;\n";
  invoke_codes_ += "            }\n";
  invoke_codes_ += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  invoke_codes_ += "        }\n";
  invoke_codes_ += "    }\n";
  invoke_codes_ += "    free(memory_pool);\n";
  return true;
}

bool GeneralSolverGen::CreateConfig() {
  invoke_codes_ += "    SolverConfig cfg;\n";
  invoke_codes_ += "    cfg.top_num = cfg_top_num;\n";
  invoke_codes_ += "    cfg.search_length = cfg_search_length;\n";
  invoke_codes_ += "    cfg.iterations = cfg_iterations;\n";
  invoke_codes_ += "    cfg.simple_ver = cfg_simple_ver;\n";
  invoke_codes_ +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  invoke_codes_ += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  invoke_codes_ += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  invoke_codes_ += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  invoke_codes_ += "    if (cfg.simple_ver) {\n";
  invoke_codes_ += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  invoke_codes_ += "    } else {\n";
  invoke_codes_ += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  invoke_codes_ += "    }\n";
  invoke_codes_ += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  invoke_codes_ += "\n";
  return true;
}

std::string GeneralSolverGen::GenSolverClassImpl() {
  GenClassDef();
  GenBuffFunc();
  GenGetFunc(FuncType::OBJ);
  GenGetSmoothObj();
  GenGetFunc(FuncType::BUFFER);
  GenGetWeightedDiff(FuncType::BUFFER);
  GenGetWeightedDiff(FuncType::LEQ);
  GenLeqInfo();
  GenDisplayVarVal();
  GenMapVarVal();
  GenInit();
  GenGetResult();
  return impl_codes_;
}

std::string GeneralSolverGen::GenSolverFuncInvoke() {
  std::string strs = "";
  strs += "    if (!ExecuteGeneralSolver(tiling_data)) {\n";
  strs += "      OP_LOGW(OP_NAME, \"Failed to execute general solver for tilingCaseId " + tiling_case_id_ + ".\");\n";
  strs += "      return false;\n";
  strs += "    }\n";
  strs += "    OP_LOGD(OP_NAME, \"Execute general solver for tilingCaseId " + tiling_case_id_ + " successfully.\");\n";
  return strs;
}

std::string GeneralSolverGen::GenSolverDTInvoke() {
  std::string strs = "";
  strs += "    if (!ExecuteGeneralSolverForDT(tiling_data, output_tilings)) {\n";
  strs +=
      "        OP_LOGW(OP_NAME, \"Failed to execute dt general solver for tilingCaseId " + tiling_case_id_ + "!\");\n";
  strs += "        return false;\n";
  strs += "    }\n";
  strs += "    OP_LOGD(OP_NAME, \"Execute general solver for tilingCaseId " + tiling_case_id_ + " successfully.\");\n";
  return strs;
}

std::string GeneralSolverGen::GenSolverFuncImpl() {
  invoke_codes_ += "  bool ExecuteGeneralSolver(" + type_name_ + "& tiling_data) {\n";
  CreateConfig();
  CreateInput();
  RunSolver();
  invoke_codes_ += "    OP_LOGW(OP_NAME, \"The solver execution failed.\");\n";
  invoke_codes_ += "    return false;\n";
  invoke_codes_ += "  }\n";
  return invoke_codes_;
}

std::string GeneralSolverGen::GenSolverDTImpl() {
  invoke_codes_ +=
      "  bool ExecuteGeneralSolverForDT(" + type_name_ + " &tiling_data, std::vector<uint64_t> &output_tilings) {\n";
  CreateConfig();
  CreateInput();
  RunSolver(true);
  invoke_codes_ += "    OP_LOGW(OP_NAME, \"The solver for decision tree execution failed.\");\n";
  invoke_codes_ += "    return false;\n";
  invoke_codes_ += "  }\n";
  return invoke_codes_;
}
}  // namespace att
