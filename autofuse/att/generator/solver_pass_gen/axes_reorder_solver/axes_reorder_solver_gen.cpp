/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_reorder_solver_gen.h"
#include <algorithm>
#include <set>
#include "autofuse_config/auto_fuse_config.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common_utils.h"
#include "named_origin_buf_expr_generator.h"
#include "ub_named_expr_builder.h"

using namespace ascgen_utils;

namespace att {
namespace {
// AxesReorderSolverGen 默认阈值
constexpr double kDefaultSolverUbThreshold = 0.2;
constexpr double kDefaultSolverCoreNumThreshold = 0.4;

void CollectContainerWorkList(const std::vector<Expr> &exprs, const ExprExprMap &container_expr,
                              std::vector<Expr> &work_list) {
  for (const auto &expr : exprs) {
    for (const auto &sym : expr.FreeSymbols()) {
      auto container_it = container_expr.find(sym);
      if (container_it == container_expr.end()) {
        continue;
      }
      for (const auto &container_sym : container_it->second.FreeSymbols()) {
        work_list.push_back(container_sym);
      }
    }
  }
}

struct TernaryVarDeclOrderContext {
  const std::set<std::string> &needed_vars;
  const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops;
  std::set<std::string> visiting_vars;
  std::set<std::string> ordered_vars;
  std::vector<Expr> decl_order;
};

void CollectTernaryVarDeclOrder(const Expr &var, TernaryVarDeclOrderContext &context) {
  const std::string var_name = Str(var);
  if ((context.needed_vars.count(var_name) == 0U) || (context.ordered_vars.count(var_name) > 0U)) {
    return;
  }
  const auto ternary_it = context.ternary_ops.find(var);
  if (ternary_it == context.ternary_ops.end()) {
    return;
  }
  if (context.visiting_vars.count(var_name) > 0U) {
    GELOGW("Circular dependency detected when generating ternary variable [%s].", var_name.c_str());
    return;
  }
  context.visiting_vars.insert(var_name);
  for (const auto &dep : ternary_it->second.GetUsedArgs()) {
    CollectTernaryVarDeclOrder(dep, context);
  }
  context.visiting_vars.erase(var_name);
  context.ordered_vars.insert(var_name);
  context.decl_order.emplace_back(var);
}
}  // namespace
ExprUintMap AxesReorderSolverGen::priority_map_;

bool CheckExist(const std::vector<Expr> &args, const Expr &check_arg) {
  for (const auto &arg : args) {
    if (arg == check_arg) {
      return true;
    }
  }
  return false;
}

void AxesReorderSolverGen::SetInputAlign(const ExprExprMap &input_align) {
  for (const auto &pair : input_align) {
    input_align_[pair.first] = pair.second;
  }
}

std::string AxesReorderSolverGen::ObtainRelatedVars(Expr &expr) {
  std::string codes;
  std::vector<Expr> all_args;
  GetRelatedArgs(expr, all_args);
  for (uint32_t i = 0u; i < input_args_.size(); ++i) {
    if (CheckExist(all_args, input_args_[i])) {
      codes += "  double " + Str(input_args_[i]) + " = static_cast<double>(input_.input_vars[" + std::to_string(i) +
               "]->value);\n";
    }
  }
  for (uint32_t i = 0u; i < mc_args_.size(); ++i) {
    if (CheckExist(all_args, mc_args_[i])) {
      codes += "  double " + Str(mc_args_[i]) + " = static_cast<double>(input_.pure_mc_vars[" + std::to_string(i) +
               "]->value);\n";
    }
  }
  for (uint32_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    if (CheckExist(all_args, local_buffer_tiling_vars_[i])) {
      codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = static_cast<double>(input_.local_buffer_vars[" +
               std::to_string(i) + "]->value);\n";
    }
  }
  return codes;
}

std::string SetRelatedVars(const std::vector<Expr> &rel_tiling_vars, const std::vector<Expr> &rel_cons_vars) {
  std::string strs;
  for (uint32_t i = 0u; i < rel_tiling_vars.size(); ++i) {
    strs += "      double " + Str(rel_tiling_vars[i]) + " = rel_tiling_vars[" + std::to_string(i) + "]->value;\n";
  }
  for (uint32_t i = 0u; i < rel_cons_vars.size(); ++i) {
    strs += "      double " + Str(rel_cons_vars[i]) + " = rel_in_shapes[" + std::to_string(i) + "]->value;\n";
  }
  return strs;
}

std::string GenRelatedVars(uint32_t cons_idx, const std::vector<Expr> &rel_tiling_vars,
                           const std::vector<Expr> &rel_cons_vars) {
  std::string strs;
  if (!rel_tiling_vars.empty()) {
    strs += "    TilingVariable* cons_" + std::to_string(cons_idx) + "rel_tiling_vars[" +
            std::to_string(rel_tiling_vars.size()) + "] = {";
    for (uint32_t i = 0u; i < rel_tiling_vars.size(); ++i) {
      strs += "&" + Str(rel_tiling_vars[i]) + ", ";
    }
    strs += "};\n";
    strs += "    cons" + std::to_string(cons_idx) + ".rel_tiling_vars = cons_" + std::to_string(cons_idx) +
            "rel_tiling_vars;\n";
    strs += "    cons" + std::to_string(cons_idx) +
            ".rel_tiling_vars_size = " + std::to_string(rel_tiling_vars.size()) + "u;\n";
  }
  if (!rel_cons_vars.empty()) {
    strs += "    Variable* cons_" + std::to_string(cons_idx) + "rel_in_shapes[" + std::to_string(rel_cons_vars.size()) +
            "] = {";
    for (uint32_t i = 0u; i < rel_cons_vars.size(); ++i) {
      strs += "&" + Str(rel_cons_vars[i]) + ", ";
    }
    strs += "};\n";
    strs += "    cons" + std::to_string(cons_idx) + ".rel_in_shapes = cons_" + std::to_string(cons_idx) +
            "rel_in_shapes;\n";
    strs += "    cons" + std::to_string(cons_idx) + ".rel_in_shapes_size = " + std::to_string(rel_cons_vars.size()) +
            "u;\n";
  }
  return strs;
}

std::string AxesReorderSolverGen::SetVarCons(const Expr &arg, const std::vector<Expr> &all_cons) const {
  std::string strs;
  auto related_cons = GetArgRelateCons(arg, all_cons);
  if (!related_cons.empty()) {
    strs += "    Constraint*" + Str(arg) + "_rel_cons[" + std::to_string(related_cons.size()) + "] = {";
    for (uint32_t i = 0u; i < related_cons.size(); ++i) {
      strs += "&cons" + std::to_string(related_cons[i]) + ", ";
    }
    strs += "};\n";
    strs += "    " + Str(arg) + ".rel_cons = " + Str(arg) + "_rel_cons;\n";
    strs += "    " + Str(arg) + ".rel_cons_size = " + std::to_string(related_cons.size()) + "u;\n";
  }
  return strs;
}

std::pair<std::vector<Expr>, std::vector<Expr>> AxesReorderSolverGen::SortConsArgs(const Expr &expr,
                                                                                   bool &is_mc_mixed) {
  std::vector<Expr> rel_tiling_vars;
  std::vector<Expr> rel_cons_vars;
  std::vector<Expr> arg_list;
  GetRelatedArgs(expr, arg_list);
  for (const auto &arg : arg_list) {
    if (arg.IsConstExpr()) {
      continue;
    }
    if (CheckExist(mc_args_, arg)) {
      is_mc_mixed = true;
    }
    // rel_tiling_vars表示tile切分相关的变量
    // rel_cons_vars表示和多核相关的变量，输入变量，固定值等
    if (!CheckExist(input_args_, arg) && !CheckExist(mc_args_, arg) && !CheckExist(const_args_, arg)) {
      if (!CheckExist(rel_tiling_vars, arg)) {
        rel_tiling_vars.emplace_back(arg);
      }
    } else {
      if (!CheckExist(rel_cons_vars, arg)) {
        rel_cons_vars.emplace_back(arg);
      }
    }
  }
  return std::make_pair(rel_tiling_vars, rel_cons_vars);
}

void AxesReorderSolverGen::GetRelatedArgs(const Expr &expr, std::vector<Expr> &related_args) const {
  for (const auto &arg : expr.FreeSymbols()) {
    if (arg.IsConstExpr()) {
      continue;
    }
    auto iter = container_expr_.find(arg);
    if (iter != container_expr_.end()) {
      for (const auto &var : iter->second.FreeSymbols()) {
        related_args.emplace_back(var);
      }
    } else {
      related_args.emplace_back(arg);
    }
  }
}

void AxesReorderSolverGen::GetMCArgs() {
  std::vector<Expr> other_args;
  std::vector<Expr> related_args;
  for (const auto &pair : hardware_use_map_) {
    if (pair.first == HardwareDef::CORENUM) {
      continue;
    }
    GetRelatedArgs(pair.second, other_args);
  }
  for (const auto &pair : hardware_use_map_) {
    if (pair.first != HardwareDef::CORENUM) {
      continue;
    }
    GetRelatedArgs(pair.second, related_args);
  }
  for (const auto &arg : related_args) {
    if ((!CheckExist(mc_args_, arg)) && (!CheckExist(const_args_, arg)) && (!CheckExist(input_args_, arg)) &&
        (!CheckExist(other_args, arg))) {
      mc_args_.push_back(arg);
    }
  }
  GELOGD("Got mc args: %s", GetVecString(mc_args_).c_str());
}

void AxesReorderSolverGen::GetLocalBufferTilingVars() {
  std::vector<Expr> related_args;
  for (const auto &pair : hardware_use_map_) {
    if (pair.first == HardwareDef::CORENUM) {
      continue;
    }
    GetRelatedArgs(pair.second, related_args);
  }
  for (const auto &arg : related_args) {
    if (arg.IsConstExpr()) {
      continue;
    }
    if (!CheckExist(input_args_, arg) && !CheckExist(search_args_, arg) && !CheckExist(const_args_, arg)) {
      input_args_.emplace_back(arg);
      continue;
    }
    // 若args已加入mc_args(多核)，则不加入local_buffer_tiling_vars_
    if ((!CheckExist(local_buffer_tiling_vars_, arg)) && (!CheckExist(mc_args_, arg)) &&
        (!CheckExist(const_args_, arg)) && (!CheckExist(input_args_, arg))) {
      local_buffer_tiling_vars_.push_back(arg);
    }
  }
  GELOGD("Got local buffer tiling vars: %s input args: %s", GetVecString(local_buffer_tiling_vars_).c_str(),
         GetVecString(input_args_).c_str());
}

bool AxesReorderSolverGen::VarCmp(Expr &a, Expr &b) {
  if (priority_map_.find(a) != priority_map_.end() && priority_map_.find(b) != priority_map_.end()) {
    return priority_map_[a] < priority_map_[b];
  }
  return af::SymbolicUtils::ToString(a) < af::SymbolicUtils::ToString(b);
}

std::vector<uint32_t> AxesReorderSolverGen::GetArgRelateCons(const Expr &arg, const std::vector<Expr> &all_cons) const {
  std::vector<uint32_t> relate_cons;
  std::vector<Expr> used_args;
  for (uint32_t i = 0u; i < all_cons.size(); ++i) {
    GetRelatedArgs(all_cons[i], used_args);
    if (CheckExist(used_args, arg)) {
      relate_cons.emplace_back(i);
    }
  }
  return relate_cons;
}

void AxesReorderSolverGen::ReorderVars() {
  std::sort(local_buffer_tiling_vars_.begin(), local_buffer_tiling_vars_.end(), VarCmp);
  std::sort(mc_args_.begin(), mc_args_.end(), VarCmp);
}

void AxesReorderSolverGen::Arrange() {
  GetMCArgs();
  GetLocalBufferTilingVars();
  ReorderVars();
}

std::string AxesReorderSolverGen::GenOriginExpr(const std::vector<Expr> &exprs, const std::string &indent) const {
  std::string res;
  ExprExprMap container_args;
  for (const auto &expr : exprs) {
    for (const auto &arg : expr.FreeSymbols()) {
      auto it = container_expr_.find(arg);
      if (it != container_expr_.end()) {
        container_args[arg] = it->second;
      }
    }
  }
  for (const auto &pair : container_args) {
    res += indent + "double " + Str(pair.first) + " = " + Str(pair.second) + ";\n";
  }
  return res;
}

std::pair<std::string, std::string> AxesReorderSolverGen::GenOriginBufExpr(const Expr &expr,
                                                                           const std::string &indent) const {
  std::string tmp_def;
  ExprExprMap container_args;
  Optimizer ast_optimizer;
  std::map<std::string, ASTNode> ast_expr_map;
  for (const auto &arg : expr.FreeSymbols()) {
    auto it = container_expr_.find(arg);
    if (it != container_expr_.end()) {
      Parser parser(Str(it->second));
      ASTPtr ast = parser.Parse();
      ast_optimizer.Optimize(ast);
      container_args[arg] = it->second;
      ast_expr_map.emplace(Str(arg), *ast.get());
    }
  }
  std::string tmp_vars = ast_optimizer.GenerateCode(indent);
  tmp_def += tmp_vars;
  for (const auto &pair : ast_expr_map) {
    std::string return_expr = ast_optimizer.RebuildExpr(pair.second, 1);
    tmp_def += indent + "double " + pair.first + " = " + return_expr + ";\n";
  }
  Parser parser(Str(expr));
  ASTPtr ast = parser.Parse();
  ast_optimizer.Optimize(ast);
  std::string func_tmp_vars = ast_optimizer.GenerateCode(indent);
  tmp_def += func_tmp_vars;
  std::string func_return_expr = ast_optimizer.RebuildExpr(*ast.get(), 1);
  return std::make_pair(tmp_def, func_return_expr);
}

std::pair<std::string, std::string> AxesReorderSolverGen::GenNamedOriginBufExpr(const Expr &expr,
                                                                                const std::string &indent) const {
  return NamedOriginBufExprGenerator(container_expr_, container_names_).Generate(expr, indent);
}

std::string AxesReorderSolverGen::GenObjFunc() {
  std::string codes;
  std::string pipe_obj;
  std::vector<Expr> funcs;
  codes += "double AxesReorderSolver" + tiling_case_id_ + "::GetPerf() {\n";
  bool input_contain_block_dim = false;
  for (uint32_t i = 0u; i < input_args_.size(); ++i) {
    if (Str(input_args_[i]) == "block_dim") {
      input_contain_block_dim = true;
    }
    codes += "  double " + Str(input_args_[i]) + " = static_cast<double>(input_.input_vars[" + std::to_string(i) +
             "]->value);\n";
  }
  if (!input_contain_block_dim) {
    codes += "  double block_dim = 1;\n";
    codes += "  CalUsedCoreNum(block_dim);\n";
  }
  for (uint32_t i = 0u; i < mc_args_.size(); ++i) {
    codes += "  double " + Str(mc_args_[i]) + " = static_cast<double>(input_.pure_mc_vars[" + std::to_string(i) +
             "]->value);\n";
  }
  for (uint32_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = static_cast<double>(input_.local_buffer_vars[" +
             std::to_string(i) + "]->value);\n";
  }
  codes += "  return GetPerfStatic(PipeType::ALL, " + GenGetObjStaticInputParam(true) + ");\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

// 两种schedule可能生效
bool AxesReorderSolverGen::NeedUBMultiCoreBalance() {
  bool need_ub_mc_balance = false;
  // 1.如果先切多核，再去分ub的schedule也生效，z->zt zt->ztt
  for (const auto &local_arg : local_buffer_tiling_vars_) {
    const auto &from_axes = from_axes_map_[local_arg];
    for (const auto &from_axis : from_axes) {
      for (const auto &mc_arg : mc_args_) {
        if (Str(mc_arg) == Str(from_axis)) {
          mc_related_ub_args_map_[local_arg] = 1u;
          need_ub_mc_balance = true;
        }
      }
    }
  }
  // 2.假设两根轴[z0,z1], z1.tilesplit->[z1T, z1t] z0.blocksplit->[z0B, z0b]，这种schedule多核和ub参数没有关系，
  // ub的大小不会影响多核的切分，这种schedule无需做平衡策略
  for (const auto &pair : hardware_use_map_) {
    if (pair.first != HardwareDef::CORENUM) {
      continue;
    }
    auto &mc_cons = pair.second;
    for (const auto &mc_arg : mc_cons.FreeSymbols()) {
      for (const auto &local_arg : local_buffer_tiling_vars_) {
        if (Str(mc_arg) == Str(local_arg)) {
          mc_related_ub_args_map_[local_arg] = 1u;
          need_ub_mc_balance = true;
        }
      }
    }
  }
  return need_ub_mc_balance;
}

inline void GetExprFromParamList(const std::set<std::string> &param_set, std::string &codes) {
  size_t index = 0UL;
  for (const auto &param : param_set) {
    if (index > 0UL) {
      codes += ", ";
    }
    codes += param;
    ++index;
  }
}

std::string AxesReorderSolverGen::GenGetStaticInputParam(const HardwareDef &hardware_type, bool no_type) const {
  std::string codes;
  std::set<std::string> param_set;
  std::string type_prefix = "double ";
  if (no_type) {
    type_prefix = "";
  }
  auto iter = hardware_use_map_.find(hardware_type);
  if (iter != hardware_use_map_.end()) {
    std::vector<Expr> all_args;
    GetRelatedArgs(iter->second, all_args);
    for (const auto &arg : all_args) {
      if (arg.IsConstExpr()) {
        continue;
      }
      param_set.insert(type_prefix + Str(arg));
    }
  }
  GetExprFromParamList(param_set, codes);
  return codes;
}

std::string AxesReorderSolverGen::GenGetUbSizeStaticFunc() {
  std::string codes;
  codes += "int64_t AxesReorderSolver" + tiling_case_id_ + "::GetUbSizeStatic(" +
           GenGetStaticInputParam(HardwareDef::UB) + ") {\n";
  bool ub_exist = false;
  auto ub_iter = hardware_use_map_.find(HardwareDef::UB);
  if (ub_iter != hardware_use_map_.end()) {
    ascir::UbExprContext context;
    context.ub_expr = ub_iter->second;
    for (const auto &item : container_expr_) {
      context.container_expr[item.first] = item.second;
    }
    for (const auto &item : container_names_) {
      context.container_names[item.first] = item.second;
    }
    auto tmp_func_pair = BuildNamedUbExpr(context, "  ");
    std::string tmp_def = tmp_func_pair.first;
    std::string func_return_expr = tmp_func_pair.second;
    codes += tmp_def;
    codes += "  int64_t ub_size = " + func_return_expr + ";\n";
    ub_exist = true;
  }
  if (!ub_exist) {
    codes += "  return 0;\n";
  } else {
    codes += "  return ub_size;\n";
  }
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenGetTilingDataUbSizeStaticFunc() {
  std::string codes;
  codes +=
      "int AxesReorderSolver" + tiling_case_id_ + "::GetTilingDataUbSizeStatic(" + type_name_ + "& tiling_data) {\n";
  for (size_t i = 0UL; i < input_args_.size(); ++i) {
    codes += "  double " + Str(input_args_[i]) + " = tiling_data.get_" + Str(input_args_[i]) + "();\n";
  }
  for (size_t i = 0UL; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = tiling_data.get_" +
             Str(local_buffer_tiling_vars_[i]) + "();\n";
  }
  codes += "  return GetUbSizeStatic(" + GenGetStaticInputParam(HardwareDef::UB, true) + ");\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenGetBlockDimStatic(Expr &corenum_cons) {
  std::string codes;
  codes += "int AxesReorderSolver" + tiling_case_id_ + "::GetBlockDimStatic(" +
           GenGetStaticInputParam(HardwareDef::CORENUM) + ") {\n";
  codes += "  return std::max(1, static_cast<int32_t>(" + Str(corenum_cons) + "));\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenGetTilingDataBlockDimStatic(Expr &corenum_cons) {
  std::string codes;
  codes +=
      "int AxesReorderSolver" + tiling_case_id_ + "::GetTilingDataBlockDimStatic(" + type_name_ + "& tiling_data) {\n";
  std::vector<Expr> all_args;
  GetRelatedArgs(corenum_cons, all_args);
  for (size_t i = 0UL; i < input_args_.size(); ++i) {
    if (CheckExist(all_args, input_args_[i])) {
      codes += "  double " + Str(input_args_[i]) + " = tiling_data.get_" + Str(input_args_[i]) + "();\n";
    }
  }
  for (size_t i = 0UL; i < mc_args_.size(); ++i) {
    if (CheckExist(all_args, mc_args_[i])) {
      codes += "  double " + Str(mc_args_[i]) + " = tiling_data.get_" + Str(mc_args_[i]) + "();\n";
    }
  }
  for (size_t i = 0UL; i < local_buffer_tiling_vars_.size(); ++i) {
    if (CheckExist(all_args, local_buffer_tiling_vars_[i])) {
      codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = tiling_data.get_" +
               Str(local_buffer_tiling_vars_[i]) + "();\n";
    }
  }
  codes += "  return GetBlockDimStatic(" + GenGetStaticInputParam(HardwareDef::CORENUM, true) + ");\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenGetObjStaticInputParam(bool no_type) {
  std::string codes;
  std::set<std::string> param_set;
  if (no_type) {
    param_set.insert("block_dim");
  } else {
    param_set.insert("double block_dim");
  }
  std::string type_prefix = "double ";
  if (no_type) {
    type_prefix = "";
  }
  for (const auto &arg : input_args_) {
    std::string arg_str = Str(arg);
    if (arg_str == "block_dim") {
      continue;
    }
    std::string param = type_prefix + arg_str;
    param_set.insert(param);
  }
  for (const auto &arg : mc_args_) {
    std::string param = type_prefix + Str(arg);
    param_set.insert(param);
  }
  for (const auto &var : local_buffer_tiling_vars_) {
    std::string param = type_prefix + Str(var);
    param_set.insert(param);
  }
  GetExprFromParamList(param_set, codes);
  return codes;
}

void AxesReorderSolverGen::CollectInitialWorkList(std::vector<Expr> &work_list) const {
  for (const auto &pair : pipe_2_obj_map_) {
    for (const auto &sym : pair.second.FreeSymbols()) {
      work_list.push_back(sym);
    }
  }
  if (IsValid(head_cost_)) {
    for (const auto &sym : head_cost_.FreeSymbols()) {
      work_list.push_back(sym);
    }
  }
}

void AxesReorderSolverGen::CollectNeededTenaryVarsClosure(std::vector<Expr> &work_list,
                                                          std::set<std::string> &needed_vars) const {
  std::set<std::string> visited_vars;
  while (!work_list.empty()) {
    Expr sym = work_list.back();
    work_list.pop_back();
    std::string sym_name = Str(sym);
    if (visited_vars.count(sym_name) > 0U) {
      continue;
    }
    visited_vars.insert(sym_name);
    auto container_it = container_expr_.find(sym);
    if (container_it != container_expr_.end()) {
      for (const auto &dep : container_it->second.FreeSymbols()) {
        work_list.push_back(dep);
      }
    }
    auto ternary_it = ternary_ops_.find(sym);
    if (ternary_it == ternary_ops_.end()) {
      continue;
    }
    needed_vars.insert(sym_name);
    for (const auto &dep : ternary_it->second.GetUsedArgs()) {
      work_list.push_back(dep);
    }
  }
}

std::string AxesReorderSolverGen::GenSingleTenaryVar(const TernaryOp &op, const std::string &var_name,
                                                     std::set<std::string> &declared_vars,
                                                     std::map<std::string, std::string> &content_to_first_var) {
  if (declared_vars.count(var_name) > 0U) {
    return "";
  }
  declared_vars.insert(var_name);
  std::string var_preamble;
  std::string tenary_expr_str;
  op.DecomposeNamedVars(var_name, var_preamble, tenary_expr_str);
  std::string content_key = var_preamble + tenary_expr_str;
  auto content_it = content_to_first_var.find(content_key);
  if (content_it != content_to_first_var.end()) {
    return "  double " + var_name + " = " + content_it->second + ";\n";
  }
  content_to_first_var[content_key] = var_name;
  return var_preamble + "  double " + var_name + " = " + tenary_expr_str + ";\n";
}

std::string AxesReorderSolverGen::GenTenaryVarDecls(const std::set<std::string> &needed_vars) {
  std::string perf_decls;
  std::set<std::string> declared_vars;
  std::map<std::string, std::string> content_to_first_var;
  TernaryVarDeclOrderContext context{needed_vars, ternary_ops_, {}, {}, {}};

  for (const auto &replace_pair : replace_vars_) {
    CollectTernaryVarDeclOrder(replace_pair.first, context);
  }
  for (const auto &var : context.decl_order) {
    const std::string var_name = Str(var);
    auto it = ternary_ops_.find(var);
    if (it == ternary_ops_.end()) {
      continue;
    }
    perf_decls += GenSingleTenaryVar(it->second, var_name, declared_vars, content_to_first_var);
  }
  return perf_decls;
}

std::string AxesReorderSolverGen::GenGetObjStaticFunc() {
  Expr expr;
  std::string codes;
  std::string pipe_obj;
  std::vector<Expr> funcs;
  codes += "double AxesReorderSolver" + tiling_case_id_ + "::GetPerfStatic(const PipeType &pipe_type, " +
           GenGetObjStaticInputParam() + ") {\n";

  for (const auto &pair : pipe_2_obj_map_) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter != kPipetypeNameMap.end()) {
      funcs.emplace_back(pair.second);
      Expr expression = CreateExpr(iter->second.c_str());
      pipe_obj += "  double " + Str(expression) + " = " + GetSmoothString(Str(pair.second)) + ";\n";
      expr = (!IsValid(expr)) ? expression : af::sym::Max(expr, expression);
      pipe_obj += "  if (pipe_type == PipeType::" + Str(expression) + ") {\n    return " + Str(expression) + ";\n  }\n";
    }
  }
  funcs.emplace_back(head_cost_);

  std::set<std::string> needed_vars;
  std::vector<Expr> work_list;
  CollectInitialWorkList(work_list);
  CollectNeededTenaryVarsClosure(work_list, needed_vars);
  std::set<std::string> pre_origin_vars;
  CollectContainerWorkList(funcs, container_expr_, work_list);
  CollectNeededTenaryVarsClosure(work_list, pre_origin_vars);
  for (const auto &var : pre_origin_vars) {
    needed_vars.erase(var);
  }

  codes += GenTenaryVarDecls(pre_origin_vars);
  codes += GenOriginExpr(funcs, "  ");
  codes += GenTenaryVarDecls(needed_vars);
  codes += pipe_obj;
  if (!IsValid(expr)) {
    codes += "  return 0;\n";
  } else {
    expr = af::sym::Add(expr, head_cost_);
    codes += "  return " + GetSmoothString(Str(expr)) + ";\n";
  }
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenGetTilingDataObjStaticFunc() {
  std::string codes;
  codes += "double AxesReorderSolver" + tiling_case_id_ + "::GetTilingDataPerfStatic(const PipeType &pipe_type, " +
           type_name_ + "& tiling_data) {\n";
  bool input_contain_block_dim = false;
  for (size_t i = 0UL; i < input_args_.size(); ++i) {
    if (Str(input_args_[i]) == "block_dim") {
      input_contain_block_dim = true;
    }
    codes += "  double " + Str(input_args_[i]) + " = tiling_data.get_" + Str(input_args_[i]) + "();\n";
  }
  if (!input_contain_block_dim) {
    codes += "  double block_dim = tiling_data.get_block_dim();\n";
  }
  for (size_t i = 0UL; i < mc_args_.size(); ++i) {
    codes += "  double " + Str(mc_args_[i]) + " = tiling_data.get_" + Str(mc_args_[i]) + "();\n";
  }
  for (size_t i = 0UL; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = tiling_data.get_" +
             Str(local_buffer_tiling_vars_[i]) + "();\n";
  }
  codes += "  return GetPerfStatic(pipe_type, " + GenGetObjStaticInputParam(true) + ");\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenUBThresholdFunc() {
  std::string codes;
  codes += "bool AxesReorderSolver" + tiling_case_id_ + "::SatisfyThresholdUBSize() {\n";
  if (!NeedUBMultiCoreBalance()) {
    codes += "  return false;\n";
    codes += "}\n";
    return codes;
  }
  for (uint32_t i = 0u; i < input_args_.size(); ++i) {
    codes += "  double " + Str(input_args_[i]) + " = static_cast<double>(input_.input_vars[" + std::to_string(i) +
             "]->value);\n";
  }
  for (uint32_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = static_cast<double>(input_.local_buffer_vars[" +
             std::to_string(i) + "]->value);\n";
  }
  codes += "  uint32_t ub_size = GetUbSizeStatic(" + GenGetStaticInputParam(HardwareDef::UB, true) + ");\n";
  codes += "  return (ub_size - " + Str(reserved_ub_size_) +
           ") > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenUBSizeCacheLineFunc() {
  std::string codes;
  codes += "bool AxesReorderSolver" + tiling_case_id_ + "::SatisfyUBSizeCacheLine(uint32_t idx) {\n";
  if (!NeedUBMultiCoreBalance()) {
    codes += "  return false;\n";
    codes += "}\n";
    return codes;
  }
  if (enable_multicore_ub_tradeoff_) {
    codes += "  // enable multicore ub tradeoff by config\n";
    codes += "  return true;\n";
    codes += "}\n";
    return codes;
  }
  for (size_t i = 0u; i < input_args_.size(); ++i) {
    codes += "  double " + Str(input_args_[i]) + " = static_cast<double>(input_.input_vars[" + std::to_string(i) +
             "]->value);\n";
  }
  codes += "  double* sizes[" + std::to_string(local_buffer_tiling_vars_.size()) + "];\n";
  for (size_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "  double " + Str(local_buffer_tiling_vars_[i]) + " = static_cast<double>(0xffff);\n";
    codes += "  sizes[" + std::to_string(i) + "] = &" + Str(local_buffer_tiling_vars_[i]) + ";\n";
  }
  codes += "  *sizes[idx] = static_cast<double>(input_.local_buffer_vars[idx]->value);\n\n";
  if (cache_line_config_ != nullptr) {
    for (const auto &c : *cache_line_config_) {
      if (c.cache_line_size > 0 && IsValid(c.solver_cache_line_expr)) {
        GELOGD("GetCacheLineCont for %s", c.ToString().c_str());
        codes += "  // check node " + c.node_name + "\n";
        codes += "  if (" + Str(c.solver_cache_line_expr) + " < " + std::to_string(c.cache_line_size) + ") {\n";
        codes += "      OP_LOGD(OP_NAME, \"" + c.node_name + " condition does not satisfy UB size cache line\");\n";
        codes += "      return false;\n";
        codes += "  }\n";
      }
    }
  }

  codes += "  OP_LOGD(OP_NAME, \"condition satisfies UB size cache line\");\n";
  codes += "  return true;\n";
  codes += "}\n";
  codes += "\n";
  return codes;
}

std::string AxesReorderSolverGen::GenCoreNumFunc() {
  std::string codes;
  const std::string solver_class_name = "AxesReorderSolver" + tiling_case_id_;
  const bool has_core_num = (hardware_use_map_.find(HardwareDef::CORENUM) != hardware_use_map_.end());
  Expr corenum_cons;
  std::string related_vars_code;
  if (has_core_num) {
    corenum_cons = hardware_use_map_[HardwareDef::CORENUM];
    related_vars_code = ObtainRelatedVars(corenum_cons);
  }
  codes += "bool " + solver_class_name + "::CalUsedCoreNum(double &used_core_num) {\n";
  if (has_core_num) {
    codes += related_vars_code;
    codes += "  used_core_num = " + GetSmoothString(Str(corenum_cons)) + ";\n";
  }
  codes += "  return true;\n";
  codes += "}\n";
  codes += "\n";
  codes += "bool " + solver_class_name + "::CalRealUsedCoreNum(int64_t &used_core_num) {\n";
  if (has_core_num) {
    codes += related_vars_code;
    codes += "  used_core_num = GetBlockDimStatic(" + GenGetStaticInputParam(HardwareDef::CORENUM, true) + ");\n";
    codes += "  return true;\n";
    codes += "};\n";
    codes += "\n";
    codes += GenGetBlockDimStatic(corenum_cons);
    codes += GenGetTilingDataBlockDimStatic(corenum_cons);
  } else {
    codes += "  return true;\n";
    codes += "};\n";
    codes += "\n";
  }
  return codes;
}

std::string AxesReorderSolverGen::GenSolverClassImpl() {
  std::string codes;
  codes += "class AxesReorderSolver" + tiling_case_id_ + " : public AxesReorderSolver {\n";
  codes += " public:\n";
  codes += "  explicit AxesReorderSolver" + tiling_case_id_ +
           "(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}\n";
  codes += "  ~AxesReorderSolver" + tiling_case_id_ + "() = default;\n";
  codes += "  bool CalUsedCoreNum(double &used_core_num) override;\n";
  codes += "  bool CalRealUsedCoreNum(int64_t &used_corenum) override;\n";
  codes += "  bool SatisfyThresholdUBSize() override;\n";
  codes += "  bool SatisfyUBSizeCacheLine(uint32_t idx) override;\n";
  codes += "  double GetPerf() override;\n";
  if (hardware_use_map_.find(HardwareDef::CORENUM) != hardware_use_map_.end()) {
    Expr corenum_cons = hardware_use_map_[HardwareDef::CORENUM];
    codes += "  static int GetBlockDimStatic(" + GenGetStaticInputParam(HardwareDef::CORENUM) + ");\n";
    codes += "  static int GetTilingDataBlockDimStatic(" + type_name_ + "& tiling_data);\n";
  }
  codes += "  static double GetPerfStatic(const PipeType &pipe_type, " + GenGetObjStaticInputParam() + ");\n";
  codes += "  static double GetTilingDataPerfStatic(const PipeType &pipe_type, " + type_name_ + "& tiling_data);\n";
  codes += "  static int64_t GetUbSizeStatic(" + GenGetStaticInputParam(HardwareDef::UB) + ");\n";
  codes += "  static int GetTilingDataUbSizeStatic(" + type_name_ + "& tiling_data);\n";
  codes += "};\n";
  codes += "\n";
  codes += GenGetObjStaticFunc();
  codes += GenGetTilingDataObjStaticFunc();
  codes += GenObjFunc();
  codes += GenGetUbSizeStaticFunc();
  codes += GenGetTilingDataUbSizeStaticFunc();
  codes += GenUBThresholdFunc();
  codes += GenUBSizeCacheLineFunc();
  codes += GenCoreNumFunc();
  if (enable_autofuse_pgo_) {
    codes += GenPGOSolverClassImpl();
  }
  return codes;
}

std::string AxesReorderSolverGen::GenPGOSolverClassImpl() {
  std::string codes;
  codes += "class PGOSolver" + tiling_case_id_ + " : public AxesReorderPgoSolver {\n";
  codes += " public:\n";
  codes += "  explicit PGOSolver" + tiling_case_id_ +
           "(const AxesReorderSolverInput input) : AxesReorderPgoSolver(input) {}\n";
  codes += "  ~PGOSolver" + tiling_case_id_ + "() = default;\n";
  codes += "  bool CalUsedCoreNum(double &used_core_num) override;\n";
  codes += "  bool CalRealUsedCoreNum(int64_t &used_corenum) override;\n";
  codes += "  bool SatisfyThresholdUBSize() override {return false;};\n";
  codes += "  bool SatisfyUBSizeCacheLine(uint32_t) override {return false;};\n";
  codes += "  double GetPerf() override {return 0;};\n";
  codes += "};\n";
  codes += "\n";
  codes += "bool PGOSolver" + tiling_case_id_ + "::CalUsedCoreNum(double &used_core_num) {\n";
  if (hardware_use_map_.find(HardwareDef::CORENUM) != hardware_use_map_.end()) {
    Expr corenum_cons = hardware_use_map_[HardwareDef::CORENUM];
    codes += ObtainRelatedVars(corenum_cons);
    codes += "  used_core_num = " + GetSmoothString(Str(corenum_cons)) + ";\n";
  }
  codes += "  return true;\n";
  codes += "}\n";
  codes += "bool PGOSolver" + tiling_case_id_ + "::CalRealUsedCoreNum(int64_t &used_core_num) {\n";
  if (hardware_use_map_.find(HardwareDef::CORENUM) != hardware_use_map_.end()) {
    Expr corenum_cons = hardware_use_map_[HardwareDef::CORENUM];
    codes += ObtainRelatedVars(corenum_cons);
    codes += "  used_core_num = " + Str(corenum_cons) + ";\n";
  }
  codes += "  return true;\n";
  codes += "};\n";
  return codes;
}

std::string AxesReorderSolverGen::InitiateArgs() {
  std::string strs = "";
  for (const auto &arg : input_args_) {
    std::string arg_string = arg.Str().get();
    strs += "    Variable " + arg_string + ";\n";
    auto input_align_expr = input_align_[arg];
    if (IsValid(arg) && (!input_align_expr.IsConstExpr() ||
                         (input_align_expr.IsConstExpr() &&
                          af::SymbolicUtils::StaticCheckNe(input_align_expr, af::Symbol(1)) == af::TriBool::kTrue))) {
      std::string input_align = "std::max(1, " + RuntimeAlignExprToCode(Str(input_align_expr)) + ")";
      strs += "    " + arg_string + ".value = (tiling_data.get_" + arg_string + "() + " + input_align + " - 1) / " +
              input_align + " * " + input_align + ";\n";
    } else {
      strs += "    " + arg_string + ".value = tiling_data.get_" + arg_string + "();\n";
    }
  }
  for (const auto &mc_arg : mc_args_) {
    strs += "    TilingVariable " + Str(mc_arg) + ";\n";
  }
  for (const auto &local_arg : local_buffer_tiling_vars_) {
    strs += "    TilingVariable " + Str(local_arg) + ";\n";
    const auto order_iter = axes_order_.find(local_arg);
    if (order_iter != axes_order_.end()) {
      strs += "    " + Str(local_arg) + ".order = " + std::to_string(order_iter->second) + "UL;\n";
    }
  }
  return strs;
}

std::string AxesReorderSolverGen::GenConsUbFunc(uint32_t cons_idx, const std::vector<Expr> &rel_tiling_vars,
                                                const std::vector<Expr> &rel_cons_vars) const {
  std::string strs = "";
  strs += "    auto cons" + std::to_string(cons_idx) +
          "Eval = [](TilingVariable **rel_tiling_vars, "
          "Variable **rel_in_shapes, int64_t rel_hw_spec) {\n";
  strs += SetRelatedVars(rel_tiling_vars, rel_cons_vars);
  strs += "      (void)rel_tiling_vars;\n";
  strs += "      (void)rel_in_shapes;\n";
  strs += "      int64_t value = AxesReorderSolver" + tiling_case_id_ + "::GetUbSizeStatic(" +
          GenGetStaticInputParam(HardwareDef::UB, true) + ") - rel_hw_spec;\n";
  strs += "      return value;\n";
  strs += "    };\n";
  return strs;
}

std::string AxesReorderSolverGen::GenConsFunc(uint32_t cons_idx, ConsType cons_type, const Expr &cons,
                                              const std::vector<Expr> &rel_tiling_vars,
                                              const std::vector<Expr> &rel_cons_vars) const {
  std::string strs = "";
  strs += "    auto cons" + std::to_string(cons_idx) +
          "Eval = [](TilingVariable **rel_tiling_vars, "
          "Variable **rel_in_shapes, int64_t rel_hw_spec) {\n";
  strs += SetRelatedVars(rel_tiling_vars, rel_cons_vars);
  strs += "      (void)rel_tiling_vars;\n";
  strs += "      (void)rel_in_shapes;\n";

  if (cons_type == ConsType::BUFFER) {
    auto tmp_func_pair = GenOriginBufExpr(cons, "      ");
    auto tmp_def = tmp_func_pair.first;
    auto func_return_expr = tmp_func_pair.second;
    strs += tmp_def;
    strs += "      int64_t value = " + func_return_expr + "- rel_hw_spec;\n";
  } else if (cons_type == ConsType::CUT) {
    strs += "      (void)rel_hw_spec;\n";
    strs += GenOriginExpr({cons}, "      ");
    strs += "      int64_t value = " + Str(cons) + ";\n";
  }
  strs += "      return value;\n";
  strs += "    };\n";
  return strs;
}

std::string AxesReorderSolverGen::InitiateBufferConsArgs(uint32_t cons_idx, HardwareDef hardware, const Expr &cons) {
  bool is_mc_mixed;
  std::string strs;
  std::string hardware_name = BaseTypeUtils::DumpHardware(hardware);
  auto rel_vars = SortConsArgs(cons, is_mc_mixed);
  strs += "    int64_t " + hardware_name + " = tiling_data.get_" + hardware_name + "();\n";
  strs += "    Constraint cons" + std::to_string(cons_idx) + ";\n";
  if (hardware == HardwareDef::UB) {
    strs += GenConsUbFunc(cons_idx, rel_vars.first, rel_vars.second);
  } else {
    strs += GenConsFunc(cons_idx, ConsType::BUFFER, cons, rel_vars.first, rel_vars.second);
  }
  strs += GenRelatedVars(cons_idx, rel_vars.first, rel_vars.second);
  strs += "    cons" + std::to_string(cons_idx) + ".rel_hw_spec = " + hardware_name + ";\n";
  strs += "    cons" + std::to_string(cons_idx) + ".type = ConstraintType::LOCAL_BUFFER;\n";
  strs += "    cons" + std::to_string(cons_idx) + ".eval = cons" + std::to_string(cons_idx) + "Eval;\n";
  return strs;
}

std::string AxesReorderSolverGen::InitiateCutConsArgs(uint32_t cons_idx, const Expr &cons, bool &is_mc_mixed) {
  std::string strs;
  std::vector<Expr> rel_cons_vars;
  auto rel_vars = SortConsArgs(cons, is_mc_mixed);
  strs += "    Constraint cons" + std::to_string(cons_idx) + ";\n";
  strs += GenConsFunc(cons_idx, ConsType::CUT, cons, rel_vars.first, rel_vars.second);
  strs += GenRelatedVars(cons_idx, rel_vars.first, rel_vars.second);
  // 与多核切分无关的变量
  if (!is_mc_mixed) {
    strs += "    cons" + std::to_string(cons_idx) + ".type = ConstraintType::LB_MIXED;\n";
  } else {
    strs += "    cons" + std::to_string(cons_idx) + ".type = ConstraintType::MC_MIXED;\n";
  }
  strs += "    cons" + std::to_string(cons_idx) + ".eval = cons" + std::to_string(cons_idx) + "Eval;\n";
  return strs;
}

std::string AxesReorderSolverGen::GenUpperBoundFunc(const Expr &var) {
  std::string strs;
  std::string valid_cond;
  auto from_axes = from_axes_map_[var];
  strs += "    GetUpperBoundFuncPtr " + Str(var) + "_upper_bound = [](Variable **parent_vars) {\n";
  strs += "      (void)parent_vars;\n";
  strs += "      int64_t upper_bound = 1;\n";
  for (uint32_t i = 0u; i < from_axes.size(); ++i) {
    auto primary_args = from_axes[i].FreeSymbols();
    if (primary_args.empty()) {
      auto from_axis = from_axes[i];
      if (from_axis.IsConstExpr()) {
        strs += "      upper_bound *= " + Str(from_axes[i]) + ";\n";
        continue;
      }
    } else {
      for (uint32_t j = 0u; j < primary_args.size(); ++j) {
        auto pri_arg = primary_args[j];
        std::string cond = "parent_vars[" + std::to_string(j) + "]->value == -1";
        valid_cond += ((valid_cond.size() > 0) ? " || " : "") + cond;
        strs += "      double " + Str(pri_arg) + " = parent_vars[" + std::to_string(j) + "]->value;\n";
      }
    }
    strs += "      if (" + valid_cond + ") {\n";
    strs += "        return static_cast<int64_t>(-1);\n";
    strs += "      }\n";
    strs += "      upper_bound *= " + Str(from_axes[i]) + ";\n";
  }
  strs += "      return upper_bound;\n";
  strs += "    };\n";
  strs += "    " + Str(var) + ".upper_bound = " + Str(var) + "_upper_bound;\n";
  return strs;
}

std::string AxesReorderSolverGen::GenUpperBoundInfo(const Expr &var) {
  auto from_axes = from_axes_map_[var];
  uint32_t from_axes_size = 0u;
  std::string strs;
  std::string set_vars;
  for (uint32_t i = 0u; i < from_axes.size(); ++i) {
    auto primary_args = from_axes[i].FreeSymbols();
    if (primary_args.empty()) {
      if (from_axes[i].IsConstExpr()) {
        continue;
      }
    } else {
      for (uint32_t j = 0u; j < primary_args.size(); ++j) {
        set_vars += "&" + Str(primary_args[j]) + ", ";
        ++from_axes_size;
      }
    }
  }
  if (from_axes_size == 0u) {
    return "";
  }
  strs += "    Variable* " + Str(var) + "_upper_bound_vars[" + std::to_string(from_axes_size) + "] = {";
  strs += set_vars + "};\n";
  strs += "    " + Str(var) + ".upper_bound_vars = " + Str(var) + "_upper_bound_vars;\n";
  strs += "    " + Str(var) + ".upper_bound_vars_size = " + std::to_string(from_axes_size) + "u;\n";
  return strs;
}

std::string AxesReorderSolverGen::SetInputVars(InputType input_type) {
  std::string strs;
  std::string var_name;
  std::string var_type;
  std::vector<std::vector<Expr>> vars;
  if (input_type == InputType::INPUT) {
    var_name = "input_vars";
    var_type = "Variable";
    vars = {input_args_, const_args_};
  } else if (input_type == InputType::TILING) {
    var_name = "tiling_vars";
    var_type = "TilingVariable";
    vars = {mc_args_, local_buffer_tiling_vars_};
  }
  if (vars[0].size() + vars[1].size() > 0) {
    strs += "    " + var_type + "* " + var_name + "[" + std::to_string(vars[0].size() + vars[1].size()) + "] = {";
    for (uint32_t i = 0u; i < vars[0].size(); ++i) {
      strs += "&" + Str(vars[0][i]) + ", ";
    }
    for (uint32_t i = 0u; i < vars[1].size(); ++i) {
      if (!CheckExist(const_args_, vars[1][i])) {
        strs += "&" + Str(vars[1][i]) + ", ";
      }
    }
    strs += "};\n";
    strs += "    input." + var_name + " = " + var_name + ";\n";
    strs += "    input." + var_name + "_size = " + std::to_string(vars[0].size() + vars[1].size()) + "u;\n";
  }
  return strs;
}

std::string AxesReorderSolverGen::SetInputCons(std::vector<Expr> cons) const {
  std::string strs;
  if (!cons.empty()) {
    strs += "    Constraint* all_cons[" + std::to_string(cons.size()) + "] = {";
    for (uint32_t i = 0u; i < cons.size(); ++i) {
      strs += "&cons" + std::to_string(i) + ", ";
    }
    strs += "};\n";
    strs += "    input.all_cons_size = " + std::to_string(cons.size()) + "u;\n";
    strs += "    input.all_cons = all_cons;\n";
  }
  return strs;
}

std::string AxesReorderSolverGen::SetTilingVars(VarsType var_type) {
  std::string strs;
  std::string var_name;
  std::vector<Expr> vars;
  if (var_type == VarsType::PUREMC) {
    vars = mc_args_;
    var_name = "pure_mc_vars";
  } else if (var_type == VarsType::LOCALBUFFER) {
    vars = local_buffer_tiling_vars_;
    var_name = "local_buffer_vars";
  }
  if (!vars.empty()) {
    strs += "    TilingVariable* " + var_name + "[" + std::to_string(vars.size()) + "] = {";
    for (uint32_t i = 0u; i < vars.size(); ++i) {
      strs += "&" + Str(vars[i]) + ", ";
    }
    strs += "};\n";
    strs += "    input." + var_name + "_size = " + std::to_string(vars.size()) + "u;\n";
    strs += "    input." + var_name + " = " + var_name + ";\n";
    if (var_type == VarsType::LOCALBUFFER) {
      strs += "    TilingVariable* ordered_local_buffer_vars[" + std::to_string(vars.size()) + "] = {";
      for (uint32_t i = 0U; i < vars.size(); ++i) {
        strs += "&" + Str(vars[i]) + ", ";
      }
      strs += "};\n";
      strs += "    uint32_t ordered_to_canonical[" + std::to_string(vars.size()) + "] = {";
      for (uint32_t i = 0U; i < vars.size(); ++i) {
        strs += std::to_string(i) + "u, ";
      }
      strs += "};\n";
      strs += "    input.ordered_local_buffer_vars = ordered_local_buffer_vars;\n";
      strs += "    input.ordered_to_canonical = ordered_to_canonical;\n";
    }
  }
  return strs;
}

int32_t AxesReorderSolverGen::GetLocalBufferVarIndex(const Expr &expr) const {
  for (size_t i = 0U; i < local_buffer_tiling_vars_.size(); ++i) {
    if (local_buffer_tiling_vars_[i] == expr) {
      return static_cast<int32_t>(i);
    }
  }
  return -1;
}

int32_t AxesReorderSolverGen::ResolveLocalBufferVarIndex(const Expr &expr) const {
  int32_t resolved_idx = -1;
  for (size_t i = 0U; i < local_buffer_tiling_vars_.size(); ++i) {
    const Expr &search_var = local_buffer_tiling_vars_[i];
    bool matched = search_var == expr;
    const auto relation_iter = vars_relations_.find(search_var);
    matched = matched || ((relation_iter != vars_relations_.end()) && (relation_iter->second == expr));
    if (!matched) {
      continue;
    }
    if (resolved_idx >= 0) {
      return -1;
    }
    resolved_idx = static_cast<int32_t>(i);
  }
  return resolved_idx;
}

std::vector<int32_t> AxesReorderSolverGen::GetRuntimePreferredIndices(const RuntimeReorderRule &rule) const {
  std::vector<int32_t> indices;
  std::set<int32_t> unique_indices;
  for (const Expr &expr : rule.preferred_order) {
    const int32_t index = ResolveLocalBufferVarIndex(expr);
    if (index < 0) {
      continue;
    }
    if (!unique_indices.emplace(index).second) {
      return {};
    }
    indices.emplace_back(index);
  }
  if (indices.size() != local_buffer_tiling_vars_.size()) {
    return {};
  }
  return indices;
}

std::string AxesReorderSolverGen::GenRuntimeExprValue(const Expr &expr) const {
  if (expr.IsConstExpr()) {
    return Str(expr);
  }
  const int32_t local_idx = GetLocalBufferVarIndex(expr);
  if (local_idx >= 0) {
    return Str(expr) + ".upper_bound(" + Str(expr) + ".upper_bound_vars)";
  }
  if (expr.IsVariableExpr()) {
    return Str(expr) + ".value";
  }
  return GenRuntimeCompoundExprValue(expr);
}

std::string AxesReorderSolverGen::GenRuntimeCompoundExprValue(const Expr &expr) const {
  Expr runtime_expr = expr;
  std::vector<std::pair<Expr, Expr>> subs_vars;
  std::vector<std::pair<std::string, std::string>> runtime_values;
  size_t runtime_idx = 0U;
  for (const auto &arg : expr.FreeSymbols()) {
    const std::string runtime_name = "__runtime_expr_" + std::to_string(runtime_idx++) + "__";
    subs_vars.emplace_back(arg, CreateExpr(runtime_name.c_str()));
    runtime_values.emplace_back(runtime_name, GenRuntimeExprValue(arg));
  }
  runtime_expr = runtime_expr.Subs(subs_vars);
  std::string runtime_expr_str = Str(runtime_expr);
  for (const auto &runtime_value : runtime_values) {
    size_t pos = runtime_expr_str.find(runtime_value.first);
    while (pos != std::string::npos) {
      runtime_expr_str.replace(pos, runtime_value.first.size(), runtime_value.second);
      pos = runtime_expr_str.find(runtime_value.first, pos + runtime_value.second.size());
    }
  }
  return runtime_expr_str;
}

std::string AxesReorderSolverGen::GenRuntimeReorderRule(const RuntimeReorderRule &rule) {
  const std::vector<int32_t> preferred_indices = GetRuntimePreferredIndices(rule);
  if (preferred_indices.empty()) {
    return "";
  }
  std::string code;
  code += "    if ((" + GenRuntimeExprValue(rule.condition_axis) + " < " + std::to_string(rule.condition_threshold) +
          ") && (" + GenRuntimeExprValue(rule.compare_axis) + " > " + std::to_string(rule.compare_threshold) + ")) {\n";
  for (size_t i = 0U; i < preferred_indices.size(); ++i) {
    code += "      input.ordered_local_buffer_vars[" + std::to_string(i) + "] = input.local_buffer_vars[" +
            std::to_string(preferred_indices[i]) + "];\n";
    code += "      input.ordered_to_canonical[" + std::to_string(i) + "] = " + std::to_string(preferred_indices[i]) +
            "u;\n";
  }
  code += "    }\n";
  return code;
}

std::string AxesReorderSolverGen::GenRuntimeReorderRules() {
  std::string code;
  for (const auto &rule : runtime_reorder_rules_) {
    code += GenRuntimeReorderRule(rule);
  }
  return code;
}

std::string AxesReorderSolverGen::GenInput(const TradeOffConfig &trade_off_config, std::vector<Expr> &all_cons) {
  std::string strs;
  strs += "    AxesReorderSolverInput input;\n";
  strs += SetInputVars(InputType::INPUT);
  strs += SetInputVars(InputType::TILING);
  strs += SetInputCons(all_cons);
  strs += SetTilingVars(VarsType::PUREMC);
  strs += SetTilingVars(VarsType::LOCALBUFFER);
  strs += "    input.core_num = corenum_;\n";
  strs += "    input.result_id = " + std::to_string(tiling_case_ident_.schedule_group_ident.impl_graph_id) + ";\n";
  strs += "    input.group_id = " + std::to_string(tiling_case_ident_.schedule_group_ident.group_id) + ";\n";
  strs += "    input.case_id = " + std::to_string(tiling_case_ident_.tiling_case_id) + ";\n";
  std::string sub_case_str = tiling_case_ident_.sub_case_tag.empty() ? "0" : "1";
  strs += "    input.sub_case_id = " + sub_case_str + ";\n";
  std::string ub_threshold_str;
  std::string core_num_threshold_str;

  // 三种场景（优先级从高到低）：
  // 1. 用户配置使能（enable_multicore_ub_tradeoff_ == true）
  // 2. Group并行场景（enable_group_parallel_ == true && group_num_ > 1）
  // 3. ModelInfo 惩罚配置（trade_off_config.is_enable == true）
  // 4. 默认配置
  GELOGI(
      "[DFX] GenInput: enable_multicore_ub_tradeoff=%d, enable_group_parallel_=%d, group_num_=%zu, "
      "trade_off_config.is_enable=%d, ub_ratio=%s, core_num_ratio=%s",
      enable_multicore_ub_tradeoff_, enable_group_parallel_, group_num_, trade_off_config.is_enable,
      Str(trade_off_config.ub_ratio).c_str(), Str(trade_off_config.core_num_ratio).c_str());
  if (enable_multicore_ub_tradeoff_) {
    // 场景1: 用户配置使能（最高优先级）
    ub_threshold_str = std::to_string(ub_threshold_);
    core_num_threshold_str = std::to_string(corenum_threshold_);
  } else if (trade_off_config.is_enable) {
    if (enable_group_parallel_ && group_num_ > 1) {
      // 场景2: Group并行场景，按照Group数均分核数
      double group_ratio = 1.0 / static_cast<double>(group_num_);
      ub_threshold_str = Str(trade_off_config.ub_ratio);
      core_num_threshold_str = std::to_string(group_ratio);
    } else {
      // 场景3: ModelInfo 级别的 TilingScheduleConfig（惩罚配置）
      ub_threshold_str = Str(trade_off_config.ub_ratio);
      core_num_threshold_str = Str(trade_off_config.core_num_ratio);
    }
  } else {
    // 场景4: 默认配置
    ub_threshold_str = std::to_string(kDefaultSolverUbThreshold);
    core_num_threshold_str = std::to_string(kDefaultSolverCoreNumThreshold);
  }

  strs += "    input.ub_threshold = " + ub_threshold_str + ";\n";
  strs += "    input.corenum_threshold = " + core_num_threshold_str + ";\n";

  for (const auto &pair : hardware_use_map_) {
    if (pair.first == HardwareDef::UB) {
      std::string ub_name = BaseTypeUtils::DumpHardware(pair.first);
      strs += "    input.ub_size = tiling_data.get_" + ub_name + "();\n";
    }
  }
  return strs;
}

void AxesReorderSolverGen::InitConcatPromptAlign(const Expr &local_var, const uint32_t prompt_align,
                                                 std::string &strs) {
  const Expr concat_innner_dim = concat_inner_dims_[0];
  const auto concat_align = arg_prompt_align_map_[concat_innner_dim];
  int32_t const_val = 0;
  const auto is_const = concat_innner_dim.GetConstValue(const_val);
  GELOGD(
      "[DFX] InitConcatPromptAlign (is_const %d): const_val=%d, concat_align=%u, local_var=%s, prompt_align=%u,"
      " concat_inner_dim=%s",
      is_const, const_val, concat_align, Str(local_var).c_str(), prompt_align, Str(concat_innner_dim).c_str());
  if (is_const) {
    if ((const_val != 0) && (const_val % concat_align != 0)) {
      strs += "    " + Str(local_var) + ".prompt_align = " + std::to_string(prompt_align) + ";\n";
    }
  } else {
    // 非const表达式：如果concat_align <= 1，条件无意义（任何数模1都是0），直接赋值
    const bool not_check = (concat_align <= 1);
    strs += not_check ? ""
                      : "    if (" + Str(concat_innner_dim) + ".value % " + std::to_string(concat_align) + " != 0) {\n";
    strs += "      " + Str(local_var) + ".prompt_align = " + std::to_string(prompt_align) + ";\n";
    strs += not_check ? "" : "    }\n";
  }
}

std::string AxesReorderSolverGen::GenInputInfo(std::vector<Expr> &all_cons, std::vector<Expr> &local_buffer_cons,
                                               std::vector<Expr> &mc_mixed_cons) {
  std::string strs;
  uint32_t cons_idx = 0u;
  for (const auto &pair : hardware_use_map_) {
    if (pair.first == HardwareDef::CORENUM) {
      continue;
    }
    strs += InitiateBufferConsArgs(cons_idx++, pair.first, pair.second);
    all_cons.emplace_back(pair.second);
    local_buffer_cons.emplace_back(pair.second);
  }
  for (const auto &cut_cons : total_cut_cons_) {
    bool is_mc_mixed = false;
    strs += InitiateCutConsArgs(cons_idx++, cut_cons, is_mc_mixed);
    all_cons.emplace_back(cut_cons);
    if (is_mc_mixed) {
      mc_mixed_cons.emplace_back(cut_cons);
    }
  }
  GELOGD("Got all cut cons: %s mc_mixed_cons: %s all_cons: %s", GetVecString(total_cut_cons_).c_str(),
         GetVecString(mc_mixed_cons).c_str(), GetVecString(all_cons).c_str());
  for (const auto &mc_var : mc_args_) {
    strs += GenUpperBoundFunc(mc_var);
    strs += GenUpperBoundInfo(mc_var);
    strs += SetVarCons(mc_var, all_cons);
  }
  for (const auto &local_var : local_buffer_tiling_vars_) {
    const auto align = arg_align_map_[local_var];
    const auto prompt_align = arg_prompt_align_map_[local_var];
    const auto data_type_size = data_type_size_map_[local_var];
    strs += "    " + Str(local_var) + ".align = std::max(1, " + RuntimeAlignExprToCode(Str(align)) + ");\n";
    if (data_type_size > 0U) {
      strs += "    " + Str(local_var) + ".data_type_size = " + std::to_string(data_type_size) + ";\n";
    }
    if ((is_concat_outer_map_[local_var] == 1) && (!concat_inner_dims_.empty())) {
      InitConcatPromptAlign(local_var, prompt_align, strs);
    } else {
      if (prompt_align > 1U) {
        strs += "    " + Str(local_var) + ".prompt_align = " + std::to_string(std::max(1u, prompt_align)) + ";\n";
      }
    }
    NeedUBMultiCoreBalance();
    if (mc_related_ub_args_map_.find(local_var) != mc_related_ub_args_map_.end()) {
      strs += "    " + Str(local_var) + ".mc_related = true;\n";
    }
    strs += GenUpperBoundFunc(local_var);
    strs += GenUpperBoundInfo(local_var);
    strs += SetVarCons(local_var, all_cons);
  }
  return strs;
}

std::string AxesReorderSolverGen::GenSetTiling() {
  std::string strs;
  for (uint32_t i = 0u; i < mc_args_.size(); ++i) {
    auto &mc_arg = mc_args_[i];
    strs += "    tiling_data.set_" + Str(mc_arg) + "(input.pure_mc_vars[" + std::to_string(i) + "]->value);\n";
  }
  for (uint32_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    auto &local_var = local_buffer_tiling_vars_[i];
    strs += "    tiling_data.set_" + Str(local_var) + "(input.local_buffer_vars[" + std::to_string(i) + "]->value);\n";
  }
  return strs;
}

// Helper function to generate PGO solver run code
static std::string GenPGOSolverRunCode(const std::string &class_name, const char *high_perf_val,
                                       const std::string &enable_equal_order_arg) {
  std::string codes;
  codes += "    if (PgoConfig::Instance().need_change_solver_run == 1) {\n";
  codes +=
      "      input.ub_threshold = "
      "PgoConfig::Instance().pgo_ub_threshold_list[PgoConfig::Instance().pgo_threshold_index];\n";
  codes +=
      "      input.corenum_threshold = "
      "PgoConfig::Instance().pgo_corenum_threshold_list[PgoConfig::Instance().pgo_threshold_index];\n";
  codes += "      " + class_name + " solver(input);\n";
  codes += "      if (!solver.Run(true, false, " + std::string(high_perf_val) + enable_equal_order_arg + ")) { \n";
  codes += "        return false;\n";
  codes += "      }\n";
  codes += "    } else {\n";
  return codes;
}

std::string AxesReorderSolverGen::GenSolverRunInvoke(const std::string &class_name) {
  std::string codes;
  const af::char_t *high_perf_val = (enable_high_perf_ && (!enable_group_parallel_)) ? "true" : "false";
  bool hit_pattern = NeedUBMultiCoreBalance();
  const auto enable_block_loop_trade_off_by_perf = IsEnableBlockLoopTradeOffByPerf();
  const bool model_tradeoff_enable = tiling_schedule_config_.trade_off_config.is_enable;
  const std::string enable_multicore_ub_tradeoff =
      ((enable_multicore_ub_tradeoff_ || model_tradeoff_enable) && hit_pattern) ? "true" : "false";
  std::string enable_equal_order_arg = enable_equal_order_ ? ", true" : ", false";

  if (enable_autofuse_pgo_) {
    codes += GenPGOSolverRunCode(class_name, high_perf_val, enable_equal_order_arg);
  }

  std::string run_multicore_ub_tradeoff = enable_multicore_ub_tradeoff;
  std::string run_block_loop_trade_off_by_perf = enable_block_loop_trade_off_by_perf;
  if (enable_autofuse_pgo_ || is_inductor_scene_) {
    run_multicore_ub_tradeoff = "runtime_enable_multicore_ub_tradeoff";
    run_block_loop_trade_off_by_perf = "runtime_enable_block_loop_trade_off_by_perf";
  }
  std::string run_args = run_multicore_ub_tradeoff + ", " + run_block_loop_trade_off_by_perf + ", " + high_perf_val +
                         enable_equal_order_arg;
  codes += "    if (!solver.Run(" + run_args + ")) {\n";
  GELOGI(
      "[DFX] Gen solver func, high_perf_val: %s, hit_pattern: %d, high_perf_: %d, multicore_ub_tradeoff:%d, "
      "model_tradeoff_enable:%d, trade_off_config:%s, parallel enable flag:%d, "
      "enable_block_loop_trade_off_by_perf:%s, enable_equal_order_:%d",
      high_perf_val, hit_pattern, enable_high_perf_, enable_multicore_ub_tradeoff_, model_tradeoff_enable,
      tiling_schedule_config_.trade_off_config.DebugString().c_str(), enable_group_parallel_,
      enable_block_loop_trade_off_by_perf.c_str(), enable_equal_order_);
  codes += "      return false;\n";
  codes += "    }\n";
  if (enable_autofuse_pgo_) {
    codes += "    }\n";
  }
  return codes;
}

std::string AxesReorderSolverGen::GenEmptyTensorCheckInSolver() {
  std::string codes;
  codes += "    if (solver.IsEmptyTensor()) {\n";
  codes += "      tiling_data.set_block_dim(1);\n";
  for (uint32_t i = 0u; i < mc_args_.size(); ++i) {
    auto &mc_arg = mc_args_[i];
    codes += "      tiling_data.set_" + Str(mc_arg) + "(0);\n";
  }
  codes += "      is_empty_tensor_ = true;\n";
  codes += "      return true;\n";
  codes += "    }\n";
  return codes;
}

std::string AxesReorderSolverGen::GenPgoSetTiling() {
  std::string code;
  std::string filed_name = GetTilingDataSubGroupItemName();
  code += "    auto tiling_data_tmp = solver.GetTilingDataList();\n";
  code += "    OP_LOGD(OP_NAME, \"[PGO]before filter solver: %u\", tiling_data_tmp.size());\n";
  code += "    uint32_t solver_count = 0;\n";
  code += "    for (const auto &item : tiling_data_tmp) {\n";
  code += "      " + type_name_ + " new_auto_tiling = tiling_data;\n";
  for (size_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    code += "      new_auto_tiling.set_" + Str(local_buffer_tiling_vars_[i]) + "(item[" + std::to_string(i) + "]);\n";
  }
  for (size_t i = 0u; i < mc_args_.size(); ++i) {
    code += "      new_auto_tiling.set_" + Str(mc_args_[i]) + "(item[" +
            std::to_string(i + local_buffer_tiling_vars_.size()) + "]);\n";
  }
  code += "      if (!SetWorkspaceSize(new_auto_tiling, workspace_map)) {\n";
  code += "        continue;\n";
  code += "      }\n";
  code += "      DoApiTiling(new_auto_tiling);\n";
  code += "      GeneralTiling(new_auto_tiling);\n";
  code += GenPgoSetMaxBlockDim();
  code += "      if (!is_filter(new_auto_tiling)) { \n";
  code += "          continue;\n";
  code += "      }\n";
  if (!GetIsUniGroup()) {
    code += "      autofuse_tiling_data->" + filed_name + " = new_auto_tiling;\n";
  } else {
    code += "      *autofuse_tiling_data = new_auto_tiling;\n";
  }
  code += "      AutofuseTilingDataPerf tiling_perf;\n";
  code += "      tiling_perf.tiling_data = *autofuse_tiling_data;\n";
  code += "      tiling_perf.best_perf = DBL_MAX;\n";
  code += "      tiling_data_list.push_back(tiling_perf);\n";
  code += "      solver_count++;\n";
  code += "    }\n";
  code += "    OP_LOGD(OP_NAME, \"[PGO]after filter solver: %u\", solver_count);\n";
  return code;
}

std::string AxesReorderSolverGen::GenPgoSetMaxBlockDim() const {
  std::string code;
  code += "      if (!block_dim_vec.empty()) {\n";
  code += "        uint32_t temp_block_dim = new_auto_tiling.get_block_dim();\n";
  code += "        for (auto block_dim : block_dim_vec) {\n";
  code += "          if (block_dim != nullptr) {\n";
  code += "            if (*block_dim > temp_block_dim) {\n";
  code += "              temp_block_dim = *block_dim;\n";
  code += "            }\n";
  code += "          }\n";
  code += "        }\n";
  code += "        autofuse_tiling_data->set_block_dim(temp_block_dim);\n";
  code += "      }\n";
  return code;
}

std::string AxesReorderSolverGen::IsEnableBlockLoopTradeOffByPerf() const {
  std::string res = "true";
  if (tiling_schedule_config_table_ != nullptr) {
    bool enable_trade_off = (tiling_schedule_config_.trade_off_config.is_enable) || (enable_multicore_ub_tradeoff_);
    // 开启trade off后需要禁用自动核内循环调节
    if (enable_trade_off || !tiling_schedule_config_table_->IsEnableBlockLoopAutoTune()) {
      res = "false";
    }
  }
  return res;
}

std::string AxesReorderSolverGen::GenPendingSearchConfigOverride() {
  std::string codes;
  if (!(enable_autofuse_pgo_ || is_inductor_scene_)) {
    return codes;
  }
  const ge::char_t *runtime_enable_multicore_ub_tradeoff =
      (NeedUBMultiCoreBalance() &&
       (enable_multicore_ub_tradeoff_ || tiling_schedule_config_.trade_off_config.is_enable))
          ? "true"
          : "false";
  codes += "    bool runtime_enable_multicore_ub_tradeoff = ";
  codes += runtime_enable_multicore_ub_tradeoff;
  codes += ";\n";
  codes += "    if (pending_search_cfg_ != nullptr) {\n";
  codes += "      if (pending_search_cfg_->ub_threshold_enabled) {\n";
  codes += "        input.ub_threshold = pending_search_cfg_->ub_threshold;\n";
  codes += "      }\n";
  codes += "      if (pending_search_cfg_->corenum_threshold_enabled) {\n";
  codes += "        input.corenum_threshold = pending_search_cfg_->corenum_threshold;\n";
  codes += "      }\n";
  codes += "      runtime_enable_multicore_ub_tradeoff = pending_search_cfg_->enable_multicore_ub_tradeoff;\n";
  codes += "    }\n";
  codes += "    const bool runtime_enable_block_loop_trade_off_by_perf = ";
  codes += IsEnableBlockLoopTradeOffByPerf();
  codes += " && !runtime_enable_multicore_ub_tradeoff;\n";
  return codes;
}

std::string AxesReorderSolverGen::GenSolverFuncImpl() {
  std::string codes;
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;
  std::string class_name = "AxesReorderSolver" + tiling_case_id_;
  codes += "  bool ExecuteAxesReorderSolver(" + type_name_ + "& tiling_data) {\n";
  codes += InitiateArgs();
  codes += GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);
  // 直接使用 ModelInfo 中的 TilingScheduleConfig
  codes += GenInput(tiling_schedule_config_.trade_off_config, all_cons);
  codes += GenRuntimeReorderRules();
  // 支持二次Tiling：使用调整后的核数比例（需在创建solver之前设置）
  // 注意：g_secondary_tiling_ratio 只在 enable_group_parallel_ 为 true 时才声明
  if (enable_group_parallel_ && group_num_ > 1) {
    codes += "    // 支持二次Tiling：使用调整后的核数比例（需在创建solver之前设置）\n";
    codes += "    if (g_secondary_tiling_ratio > 0.0) {\n";
    codes +=
        "      OP_LOGI(OP_NAME, \"CorenumThreshold update from %lf to %lf\", input.corenum_threshold, "
        "g_secondary_tiling_ratio);\n";
    codes += "      input.corenum_threshold = g_secondary_tiling_ratio;\n";
    codes += "    }\n";
  }
  codes += GenPendingSearchConfigOverride();
  codes += "    " + class_name + " solver(input);\n";
  codes += GenSolverRunInvoke(class_name);
  codes += GenEmptyTensorCheckInSolver();
  codes += GenSetTiling();
  codes += "    return true;\n";
  codes += "  }\n";
  if (enable_autofuse_pgo_ && !is_inductor_scene_) {
    codes += GenPGOSolverFuncImpl();
    codes += GenPGOSolverFilter();
  }
  return codes;
}

std::string AxesReorderSolverGen::GenPGOSolverFilter() {
  std::string codes;
  codes += "  bool is_filter( " + type_name_ + " tiling_data) {\n";
  if (mc_args_.size() == 0) {
    codes += "      return true;\n";
    codes += "  }\n";
    return codes;
  }
  codes += "      std::string hashkey;\n";
  codes += "      std::vector<uint32_t> key_value;\n";
  for (size_t i = 0u; i < local_buffer_tiling_vars_.size(); ++i) {
    codes += "    key_value.push_back(tiling_data.get_" + Str(local_buffer_tiling_vars_[i]) + "());\n";
  }

  codes += "    for (auto it : key_value) {\n";
  codes += "        hashkey += (\"_\" + std::to_string(it));\n";
  codes += "    }\n";
  codes += "    hashkey += (\"_\" + std::to_string(tiling_data.get_block_dim()));\n";
  codes += "    if (filter_map.find(hashkey)!=filter_map.end()) {\n";
  codes += "        auto& value_list = filter_map[hashkey];\n";
  codes += "        if (value_list.size() >= 5) {\n";
  codes += "             auto maxIt = std::max_element(value_list.begin(), value_list.end(),\n";
  codes += "             [](" + type_name_ + "& a, " + type_name_ + "& b) {\n";
  codes += "                 return a.get_" + Str(mc_args_[0]) + "() < b.get_" + Str(mc_args_[0]) + "();\n";
  codes += "             });\n";
  codes += "             if (maxIt->get_" + Str(mc_args_[0]) + "() <= tiling_data.get_" + Str(mc_args_[0]) + "()) {\n";
  codes += "                 return false;\n";
  codes += "             } else { \n";
  codes += "                value_list.erase(maxIt);\n";
  codes += "                value_list.push_back(tiling_data);\n";
  codes += "                return true;\n";
  codes += "            }\n";
  codes += "        } else {\n";
  codes += "            value_list.push_back(tiling_data);\n";
  codes += "            return true;\n";
  codes += "        }\n";
  codes += "    }\n";
  codes += "    filter_map[hashkey] = std::vector<" + type_name_ + ">{tiling_data};\n";
  codes += "    return true;\n";
  codes += "   }\n";
  return codes;
}

std::string AxesReorderSolverGen::GenPGOSolverFuncImpl() {
  std::string codes;
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;
  std::string class_name = "PGOSolver" + tiling_case_id_;
  codes += "  bool ExecutePGOSolver(" + type_name_ +
           "& tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* "
           "autofuse_tiling_data, " +
           GetInputOutputDef() + "void* stream, std::unordered_map<int64_t, uint64_t> &workspace_map, " +
           "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {\n";
  codes += "    " + GetInputOutputVoidCasts() + "(void)stream; (void)workspace_map; (void)block_dim_vec;\n";
  codes += InitiateArgs();
  codes += GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);
  codes += GenInput(TradeOffConfig(), all_cons);
  codes += GenRuntimeReorderRules();
  // topn search_cfg override: directly apply ub_threshold/corenum_threshold to solver input
  codes += "    if (search_cfg != nullptr) {\n";
  codes += "      if (search_cfg->ub_threshold_enabled) {\n";
  codes += "        input.ub_threshold = search_cfg->ub_threshold;\n";
  codes += "      }\n";
  codes += "      if (search_cfg->corenum_threshold_enabled) {\n";
  codes += "        input.corenum_threshold = search_cfg->corenum_threshold;\n";
  codes += "      }\n";
  codes += "    }\n";
  codes += "    " + class_name + " solver(input);\n";
  // PGO path enumerates all tiling via PgoSolverGenerateAllTilingData (no Run call).
  // enable_multicore_ub_tradeoff is a Run-arg only meaningful for single-solve path;
  // log it here for diagnostics but do not pass to enumeration.
  codes += "    if (search_cfg != nullptr) {\n";
  codes +=
      "      OP_LOGI(OP_NAME, \"[PGO] search_cfg: ub_threshold=%f, corenum_threshold=%f, "
      "enable_multicore_ub_tradeoff=%d\", search_cfg->ub_threshold, "
      "search_cfg->corenum_threshold, search_cfg->enable_multicore_ub_tradeoff);\n";
  codes += "    }\n";
  codes += "    if (!solver.PgoSolverGenerateAllTilingData()) {\n";
  codes += "      return false;\n";
  codes += "    }\n";
  codes += GenPgoSetTiling();
  codes += "    return true;\n";
  codes += "  }\n";
  return codes;
}

std::string AxesReorderSolverGen::GenSolverFuncInvoke() {
  std::string strs;
  strs += "    if (!ExecuteAxesReorderSolver(tiling_data)) {\n";
  strs +=
      "      OP_LOGW(OP_NAME, \"Failed to execute axes reorder solver for tilingCaseId " + tiling_case_id_ + ".\");\n";
  strs += "      return false;\n";
  strs += "    }\n";
  strs +=
      "    OP_LOGD(OP_NAME, \"Execute axes reorder solver for tilingCaseId " + tiling_case_id_ + " successfully.\");\n";
  return strs;
}
}  // namespace att
