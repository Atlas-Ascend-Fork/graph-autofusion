/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generator/preprocess/args_manager.h"

#include <string>
#include <queue>
#include <array>
#include <fstream>
#include <iostream>

#include "common/checker.h"
#include "util/base_types_printer.h"
#include "base/att_const_values.h"
#include "generator/preprocess/args_replace.h"
namespace att {
const int32_t kDefaultAlign = 1;

// 变量替换的约束条件：1.只支持变量整除约束以及对齐约束。例如支持x%y,不支持x%(y+1)。
// 2.多级连场景(例如x%y, y%z),要求底层变量对齐值为2的幂次，且变量替换以后z的取值为2的幂次。例如要求z，y都是2的幂次
//   但是允许x为非2的幂次项。
// 替换规则：x = max(m_align, y)*n_x, y = max(y_align, z)*2^n_y, z = z_align * 2^n_z
bool ArgsManager::ReplaceVars(ExprExprMap &replaced_vars, ExprExprMap &replacements,
                              ExprExprMap &new_expr_replacements) {
  ArgsReplacer replacer(vars_infos_);
  if (!replacer.DoReplace(model_info_.eq_exprs)) {
    GELOGW("Replace failed.");
    return false;
  }
  replacer.GetReplaceResult(replaced_vars, replacements, new_expr_replacements);
  replacer.GetNewExprInitValue(replaced_var_init_values_);
  // 变量替换后的结果用于替换约束和目标表达式
  // 对所有表达式做变量替换
  std::vector<std::pair<Expr, Expr>> old_to_new_expr_replacement;
  for (auto &expr_pair : replacements) {
    old_to_new_expr_replacement.emplace_back(expr_pair);
  }
  for (auto &hardware_cons : hardware_cons_) {
    GELOGD("hardware cons before: %s", hardware_cons.second.Str().get());
    hardware_cons.second = hardware_cons.second.Replace(old_to_new_expr_replacement);
    GELOGD("hardware cons after: %s", hardware_cons.second.Str().get());
  }
  for (auto &pipe_cost : objs_) {
    GELOGD("obj before: %s", pipe_cost.second.Str().get());
    pipe_cost.second = pipe_cost.second.Replace(old_to_new_expr_replacement);
    GELOGD("obj after: %s", pipe_cost.second.Str().get());
  }
  for (auto &pair : ternary_op_) {
    GELOGD("ternary op before: %s", pair.second.GetTernaryOpStr().c_str());
    pair.second.Replace(old_to_new_expr_replacement);
    GELOGD("ternary op after: %s", pair.second.GetTernaryOpStr().c_str());
  }
  for (auto &group : perf_breakdowns_) {
    for (auto &item : group.items) {
      item.expr = item.expr.Replace(old_to_new_expr_replacement);
    }
  }
  for (auto &leq_expr : cut_leq_cons_) {
    leq_expr = leq_expr.Replace(old_to_new_expr_replacement);
  }
  for (auto &eq_expr : cut_eq_cons_) {
    eq_expr.first = eq_expr.first.Replace(old_to_new_expr_replacement);
    eq_expr.second = eq_expr.second.Replace(old_to_new_expr_replacement);
  }
  return true;
}

VarInfo &ArgsManager::SetSizeInfo(VarInfo &info, const SymVarInfoPtr &var_info, const AttAxis *arg_axis) {
  info.align = var_info->align;
  info.prompt_align = var_info->prompt_align;
  info.data_type_size = var_info->data_type_size;
  info.is_concat_inner_dim = arg_axis->is_concat_inner_dim;
  info.is_concat_outer_dim = arg_axis->is_concat_outer_dim;
  info.scopes = var_info->related_scope;
  GELOGD(
      "[PROMPT_ALIGN] SetSizeInfo for axis[%s]: prompt_align=%u, align=%s, data_type_size=%u, "
      "is_concat_inner_dim=%d, is_concat_outer_dim=%d",
      arg_axis->name.c_str(), info.prompt_align, Str(info.align).c_str(), info.data_type_size, info.is_concat_inner_dim,
      info.is_concat_outer_dim);
  if (IsValid(var_info->max_value)) {
    info.max_value = var_info->max_value;
  } else {
    Expr max_value;
    for (size_t i = 0u; i < arg_axis->orig_axis.size(); i++) {
      if ((arg_axis->orig_axis[i] == nullptr) || (arg_axis->orig_axis[i]->size == nullptr) ||
          !IsValid(arg_axis->orig_axis[i]->size->symbol_expr)) {
        GELOGW("Axis [%s] ori axis or ori axis size is null.", arg_axis->name.c_str());
        continue;
      }
      if (i > 0u) {
        max_value = af::sym::Mul(max_value, arg_axis->orig_axis[i]->size->symbol_expr);
        continue;
      }
      max_value = arg_axis->orig_axis[i]->size->symbol_expr;
    }
    if (!IsValid(max_value)) {
      max_value = info.align;
    }
    info.max_value = max_value;
  }
  return info;
}

VarInfo &ArgsManager::SetInitSize(VarInfo &info, const bool is_last) {
  // const 初始值就是变量的值本身
  if (info.is_const_var) {
    info.init_value = CreateExpr(static_cast<int32_t>(info.const_value));
    info.min_value = CreateExpr(static_cast<int32_t>(info.const_value));
    return info;
  }
  if (is_last) {
    info.init_value = info.max_value;
  } else {
    info.init_value = info.align;
  }
  info.min_value = info.align;
  if (!IsValid(info.init_value)) {
    GELOGW("null init value.");
  }
  return info;
}

bool SplitVars(const AttAxisPtr &arg_axis, ExprInfoMap &var_infos) {
  GE_ASSERT_TRUE(IsValid(arg_axis->size->symbol_expr), "Arg size is nullptr.");
  if (arg_axis->axis_pos == AxisPosition::ORIGIN) {
    if (arg_axis->size->symbol_expr.GetExprType() == af::ExprType::kExprOperation) {
      auto args = arg_axis->size->symbol_expr.FreeSymbols();
      GELOGD(
          "[PROMPT_ALIGN] SplitVars: axis[%s] has %zu free variables, is_concat_inner_dim=%d, is_concat_outer_dim=%d, "
          "prompt_align=%u, data_type_size=%u",
          arg_axis->name.c_str(), args.size(), arg_axis->is_concat_inner_dim, arg_axis->is_concat_outer_dim,
          arg_axis->size->prompt_align, arg_axis->size->data_type_size);
      for (auto &arg : args) {
        if (arg.GetExprType() == af::ExprType::kExprVariable) {
          VarInfo info;
          info.is_input_var = true;
          // 修复：继承原始轴的prompt_align等属性
          info.prompt_align = arg_axis->size->prompt_align;
          info.data_type_size = arg_axis->size->data_type_size;
          info.is_concat_inner_dim = arg_axis->is_concat_inner_dim;
          info.is_concat_outer_dim = arg_axis->is_concat_outer_dim;
          var_infos.emplace(arg, info);
          GELOGD(
              "[PROMPT_ALIGN] SplitVars: created free variable[%s] with prompt_align=%u, data_type_size=%u, "
              "is_concat_inner_dim=%d, is_concat_outer_dim=%d",
              Str(arg).c_str(), info.prompt_align, info.data_type_size, info.is_concat_inner_dim,
              info.is_concat_outer_dim);
        }
      }
      return true;
    }
  }
  return false;
}

VarInfo ArgsManager::GetNaiveVarInfo(const AttAxis *arg_axis) {
  // 内部接口，生成变量替换前的varInfo。arg_axis是否为空由外部判断。
  VarInfo info;
  GE_ASSERT_NOTNULL(arg_axis->size);
  // 判断轴的size类型: variable 还是 const类型
  SymVarInfoPtr size_var_info = std::dynamic_pointer_cast<SymVarInfo>(arg_axis->size);
  SymConstInfoPtr size_const_info = std::dynamic_pointer_cast<SymConstInfo>(arg_axis->size);
  if (size_var_info != nullptr) {
    if (arg_axis->axis_pos == AxisPosition::ORIGIN) {
      info.is_input_var = true;
    } else if (arg_axis->axis_pos == AxisPosition::INNER) {
      info.do_search = true;
    }
    // 设置轴的size信息
    info = SetSizeInfo(info, size_var_info, arg_axis);
    info = SetInitSize(info, arg_axis->is_last);
    info.value_range = size_var_info->value_range;
  } else if (size_const_info != nullptr) {
    info.is_const_var = true;
    info.prompt_align = size_const_info->prompt_align;
    info.is_concat_inner_dim = arg_axis->is_concat_inner_dim;
    info.is_concat_outer_dim = arg_axis->is_concat_outer_dim;
    info.const_value = size_const_info->const_value;
    info.data_type_size = size_const_info->data_type_size;
    info = SetInitSize(info, arg_axis->is_last);
    info.value_range = size_const_info->value_range;
    GELOGD(
        "[PROMPT_ALIGN] GetNaiveVarInfo (const) axis[%s]: prompt_align=%u, data_type_size=%u, "
        "is_concat_inner_dim=%d, is_concat_outer_dim=%d",
        arg_axis->name.c_str(), info.prompt_align, info.data_type_size, info.is_concat_inner_dim,
        info.is_concat_outer_dim);
  } else {
    GELOGE(af::FAILED, "Arg [%s] size type is not defined.", arg_axis->name.c_str());
  }
  info.is_node_innerest_dim_size = arg_axis->is_node_innerest_dim;
  for (const auto &from_axis : arg_axis->from_axis) {
    if ((from_axis != nullptr) && (from_axis->size != nullptr)) {
      info.from_axis_size.emplace_back(from_axis->size->symbol_expr);
    }
  }
  // 获取原始轴的名称
  for (const auto orig_axis : arg_axis->orig_axis) {
    if ((orig_axis != nullptr) && (orig_axis->size != nullptr)) {
      info.orig_axis_size.emplace_back(orig_axis->size->symbol_expr);
      info.orig_axis_name.emplace_back(orig_axis->name);
    }
  }
  if ((info.orig_axis_size.empty()) && (arg_axis->axis_pos == AxisPosition::ORIGIN)) {
    info.orig_axis_size.emplace_back(arg_axis->size->symbol_expr);
    info.orig_axis_name.emplace_back(arg_axis->name);
  }
  return info;
}

ExprInfoMap ArgsManager::GetOrigVarInfos(const ModelInfo &model_info) {
  ExprInfoMap expr_var_map;
  for (const auto &arg_axis : model_info.arg_list) {
    GE_ASSERT_NOTNULL(arg_axis);
    GE_ASSERT_NOTNULL(arg_axis->size);
    if (SplitVars(arg_axis, expr_var_map)) {
      continue;
    }
    auto var_info = GetNaiveVarInfo(arg_axis.get());
    expr_var_map.emplace(arg_axis->size->symbol_expr, var_info);
  }
  for (const auto &var : model_info.sizes) {
    if (expr_var_map.find(var) == expr_var_map.end()) {
      VarInfo info;
      info.is_input_var = true;
      info.do_search = false;
      GELOGD("append extra graph %s size var %s", model_info.graph_name.c_str(), var.Serialize().get());
      expr_var_map.emplace(var, info);
    }
  }
  // 设置变量相关的等式与不等式
  for (auto &expr_info : expr_var_map) {
    for (const auto &eq_exprs : model_info.eq_exprs) {
      for (const auto &eq_expr : eq_exprs.second) {
        if (eq_expr.first.ContainVar(expr_info.first) || eq_expr.second.ContainVar(expr_info.first)) {
          expr_info.second.cut_eq_cons.emplace_back(eq_expr);
        }
      }
    }
    for (const auto &leq_exprs : model_info.leq_exprs) {
      for (const auto &leq_expr : leq_exprs.second) {
        if (leq_expr.ContainVar(expr_info.first)) {
          expr_info.second.cut_leq_cons.emplace_back(leq_expr);
        }
      }
    }
  }
  return expr_var_map;
}

bool GetNewVarsInExpr(const Expr &expr, const ExprExprMap &new_expr_replacements, std::vector<Expr> &expr_args) {
  std::vector<Expr> expr_args_local;
  expr_args.swap(expr_args_local);
  auto args = expr.FreeSymbols();
  for (const auto &arg : args) {
    if (new_expr_replacements.find(arg) != new_expr_replacements.end()) {
      expr_args.emplace_back(arg);
    }
  }
  return (!expr_args.empty());
}

// 把表达式中的替换后变量替换成替换前的变量。例如new_B = B / max(16, 128*2^new_A) -> new_B = B / max(16, A)
void ArgsManager::ReplaceNewExpr(ExprExprMap &new_expr_replacements) {
  for (auto &new_expr_replace : new_expr_replacements) {
    auto &new_var_expr = new_expr_replace.second;
    std::vector<Expr> expr_args;
    while (GetNewVarsInExpr(new_var_expr, new_expr_replacements, expr_args)) {
      std::vector<std::pair<Expr, Expr>> replace_vars;
      for (const auto &arg : expr_args) {
        replace_vars.emplace_back(std::make_pair(arg, new_expr_replacements[arg]));
      }
      new_var_expr = new_var_expr.Replace(replace_vars);
    }
  }
  for (auto &new_expr_replace : new_expr_replacements) {
    GELOGD("new_var: %s, expr: %s", new_expr_replace.first.Str().get(), new_expr_replace.second.Str().get());
  }
}

// 对于形如B / max(16, A)的表达式，其最大值为max_B/max(16, min_A), min_A = A_align
Expr ArgsManager::GetNewExprMaxValueReplaced(const Expr &ori_expr, const Expr &max_value) {
  std::vector<std::pair<Expr, Expr>> replace_vars;
  replace_vars.emplace_back(ori_expr, ori_var_max_values_[ori_expr]);
  for (const auto &ori_var_align : ori_var_align_values_) {
    if (!vars_infos_[ori_var_align.first].do_search) {
      continue;
    }
    if (ori_var_align.first == ori_expr) {
      continue;
    }
    replace_vars.emplace_back(ori_var_align);
  }
  auto output = max_value.Replace(replace_vars);
  return output;
}

Expr ArgsManager::GetNewExprInitValueReplaced(const Expr &new_var) {
  if (replaced_var_init_values_.find(new_var) != replaced_var_init_values_.end()) {
    return replaced_var_init_values_[new_var];
  }
  return af::sym::kSymbolOne;
}

bool ArgsManager::SetNewVarInfoAttrs(const Expr &old_var, const ExprExprMap &replacement,
                                     const ExprExprMap ori_to_new_vars_map,
                                     const ExprExprMap local_new_expr_replacements, VarInfo &new_var_info) {
  GE_ASSERT_TRUE(ori_to_new_vars_map.find(old_var) != ori_to_new_vars_map.cend(), "CreateExpr replacement loss");
  new_var_info.align = af::Symbol(kDefaultAlign);
  auto new_var = ori_to_new_vars_map.at(old_var);
  new_var_info.replacement.orig_expr = old_var;
  new_var_info.cut_leq_cons.clear();
  new_var_info.cut_eq_cons.clear();
  for (const auto &leq_expr : cut_leq_cons_) {
    if (leq_expr.ContainVar(new_var)) {
      new_var_info.cut_leq_cons.emplace_back(leq_expr);
    }
  }
  for (const auto &eq_expr : cut_eq_cons_) {
    if (eq_expr.first.ContainVar(new_var) || eq_expr.second.ContainVar(new_var)) {
      new_var_info.cut_eq_cons.emplace_back(eq_expr);
    }
  }
  for (auto &init_value : ori_var_init_values_) {
    GELOGD("ori_var: %s, ori_init_value: %s", init_value.first.Str().get(), init_value.second.Str().get());
  }
  auto it = local_new_expr_replacements.find(new_var);
  if (it != local_new_expr_replacements.end()) {
    new_var_info.max_value = GetNewExprMaxValueReplaced(old_var, it->second);
  } else {
    new_var_info.max_value = GetMaxValue(old_var);
  }
  if (vars_infos_[old_var].init_value == vars_infos_[old_var].max_value) {
    new_var_info.init_value = new_var_info.max_value;
  } else {
    new_var_info.init_value = GetNewExprInitValueReplaced(new_var);
  }
  new_var_info.min_value = GetNewExprInitValueReplaced(new_var);
  for (auto &from_axis : new_var_info.from_axis_size) {
    if (IsValid(from_axis) and (replacement.find(from_axis) != replacement.end())) {
      from_axis = replacement.at(from_axis);
    }
  }
  return true;
}

bool ArgsManager::UpdateVarInfos(const ExprExprMap &replaced_vars, const ExprExprMap &replacement,
                                 const ExprExprMap &new_expr_replacements) {
  ExprExprMap ori_to_new_vars_map;
  for (const auto &new_to_ori_var_pair : replaced_vars) {
    ori_to_new_vars_map.emplace(new_to_ori_var_pair.second, new_to_ori_var_pair.first);
  }
  ExprExprMap local_new_expr_replacements{new_expr_replacements};
  ReplaceNewExpr(local_new_expr_replacements);
  ExprInfoMap replaced_var_infos;
  for (auto &ori_var_info : vars_infos_) {
    auto ori_var = ori_var_info.first;
    if (replacement.find(ori_var) != replacement.end()) {
      VarInfo new_var_info(ori_var_info.second);
      ori_var_info.second.replacement.new_replaced_expr = replacement.at(ori_var);

      auto set_new_var_status =
          SetNewVarInfoAttrs(ori_var, replacement, ori_to_new_vars_map, local_new_expr_replacements, new_var_info);
      GE_ASSERT_TRUE(set_new_var_status, "Set new var info failed.");
      // 变量替换以后，原变量移出待求解变量
      ori_var_info.second.do_search = false;
      auto new_var = ori_to_new_vars_map.at(ori_var);
      replaced_var_infos.emplace(new_var, new_var_info);
    }
  }
  vars_infos_.insert(replaced_var_infos.begin(), replaced_var_infos.end());
  return true;
}

bool ArgsManager::DoVarsReplace() {
  if (replacement_done_) {
    return true;
  }
  ExprExprMap replaced_vars;
  ExprExprMap replacement;
  ExprExprMap new_expr_replacement;
  if (!ReplaceVars(replaced_vars, replacement, new_expr_replacement)) {
    GELOGW("Replace vars failed.");
    return false;
  }
  replacement_done_ = true;
  GE_ASSERT_TRUE(UpdateVarInfos(replaced_vars, replacement, new_expr_replacement),
                 "Create var after replacement failed.");
  return true;
}

void ArgsManager::SetOrigExprs() {
  hardware_cons_ = model_info_.hardware_cons;
  objs_ = model_info_.objects;
  ternary_op_.clear();
  for (const auto &pair : model_info_.ternary_op_map) {
    ternary_op_[pair.first] = pair.second.DeepCopy();
  }
  perf_breakdowns_ = model_info_.perf_breakdowns;
  for (const auto &var_info : vars_infos_) {
    if (!var_info.second.is_input_var) {
      ori_var_init_values_[var_info.first] = GetDefaultInitValue(var_info.first);
      ori_var_max_values_[var_info.first] = GetMaxValue(var_info.first);
      ori_var_align_values_[var_info.first] = var_info.second.align;
    }
  }
  for (const auto &leq_exprs : model_info_.leq_exprs) {
    cut_leq_cons_.insert(cut_leq_cons_.end(), leq_exprs.second.begin(), leq_exprs.second.end());
  }
}

bool ArgsManager::Process(bool do_var_replace) {
  Reset();
  vars_infos_ = GetOrigVarInfos(model_info_);
  SetOrigExprs();
  ExprExprMap replaced_vars;
  ExprExprMap replacement;
  ExprExprMap new_expr_replacement;
  if (do_var_replace) {
    if (!ReplaceVars(replaced_vars, replacement, new_expr_replacement)) {
      GELOGW("Replace vars failed.");
      return false;
    }
    replacement_done_ = true;
  }
  // 根据变量替换的结果以及原始变量的信息，新增替换后的变量信息
  GE_ASSERT_TRUE(UpdateVarInfos(replaced_vars, replacement, new_expr_replacement),
                 "Create var after replacement failed.");
  return true;
}

std::vector<Expr> ArgsManager::GetSearchableVars() const {
  std::vector<Expr> searchable_vars;
  for (const auto &var_info : vars_infos_) {
    if (var_info.second.do_search) {
      searchable_vars.emplace_back(var_info.first);
    }
  }
  return searchable_vars;
}

std::vector<Expr> ArgsManager::GetSearchableVars(const HardwareDef scope) const {
  std::vector<Expr> searchable_vars;
  for (const auto &var_info : vars_infos_) {
    if (var_info.second.do_search) {
      for (const auto &related_scope : var_info.second.scopes) {
        if (related_scope == scope) {
          searchable_vars.emplace_back(var_info.first);
        }
      }
    }
  }
  return searchable_vars;
}

ExprExprMap ArgsManager::GetVarsRelations() const {
  ExprExprMap var_relations;
  for (const auto &var_info : vars_infos_) {
    if (IsValid(var_info.second.replacement.orig_expr)) {
      var_relations.emplace(var_info.first, var_info.second.replacement.orig_expr);
    }
  }
  return var_relations;
}

ExprExprMap ArgsManager::GetExprRelations() const {
  ExprExprMap expr_relations;
  for (const auto &var_info : vars_infos_) {
    if (IsValid(var_info.second.replacement.new_replaced_expr)) {
      expr_relations.emplace(var_info.first, var_info.second.replacement.new_replaced_expr);
    }
  }
  return expr_relations;
}

std::vector<Expr> ArgsManager::GetInputVars() const {
  std::vector<Expr> input_vars;
  for (const auto &var_info : vars_infos_) {
    if (var_info.second.is_input_var) {
      input_vars.emplace_back(var_info.first);
    }
  }
  return input_vars;
}

std::vector<std::pair<Expr, std::pair<int64_t, int64_t>>> ArgsManager::GetInputVarsRange() const {
  std::vector<std::pair<Expr, std::pair<int64_t, int64_t>>> input_vars_range;
  for (const auto &var_info : vars_infos_) {
    if (var_info.second.is_input_var) {
      input_vars_range.emplace_back(std::make_pair(var_info.first, var_info.second.value_range));
    }
  }
  return input_vars_range;
}

ExprUintMap ArgsManager::GetConstVars() const {
  ExprUintMap const_vars;
  for (const auto &var_info : vars_infos_) {
    if (var_info.second.is_const_var) {
      const_vars.emplace(var_info.first, var_info.second.const_value);
    }
  }
  return const_vars;
}

std::vector<HardwareDef> ArgsManager::GetRelatedHardware(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    const auto &var_info = vars_infos_.at(var);
    return var_info.scopes;
  }
  std::vector<HardwareDef> related_hardware;
  return related_hardware;
}

std::map<HardwareDef, Expr> ArgsManager::GetTotalHardwareCons(bool do_container_replace) const {
  if (!do_container_replace) {
    std::map<HardwareDef, Expr> hardware_cons;
    std::vector<std::pair<Expr, Expr>> replace_map;
    for (const auto &pair : GetContainerMap()) {
      replace_map.emplace_back(std::make_pair(pair.first, pair.second));
    }
    for (const auto &pair : hardware_cons_) {
      if (pair.first == HardwareDef::GM) {
        GELOGD("HBM(GM) constraint is ignored because hbm_size is removed from tiling data: %s",
               pair.second.Str().get());
        continue;
      }
      hardware_cons[pair.first] = pair.second.Replace(replace_map);
    }
    return hardware_cons;
  }
  std::map<HardwareDef, Expr> filtered_hardware_cons;
  for (const auto &pair : hardware_cons_) {
    if (pair.first == HardwareDef::GM) {
      GELOGD("HBM(GM) constraint is ignored because hbm_size is removed from tiling data: %s", pair.second.Str().get());
      continue;
    }
    filtered_hardware_cons[pair.first] = pair.second;
  }
  return filtered_hardware_cons;
}

Expr ArgsManager::GetUsedHardwareInfo(const HardwareDef scope) const {
  if (hardware_cons_.find(scope) != hardware_cons_.end()) {
    return hardware_cons_.at(scope);
  }
  GELOGW("Scope : [%s] is not used.", BaseTypeUtils::DumpHardware(scope).c_str());
  Expr res;
  return res;
}

std::map<PipeType, Expr> ArgsManager::GetObjectFunc() const {
  std::map<PipeType, Expr> res;
  for (auto &obj : objs_) {
    res[obj.first] = obj.second;
  }
  return res;
}

std::vector<Expr> ArgsManager::GetAncestor(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    return vars_infos_.at(var).orig_axis_size;
  }
  GELOGW("CreateExpr : [%s] has no ancestor", var.Str().get());
  std::vector<Expr> res;
  return res;
}

std::vector<std::string> ArgsManager::GetAncestorNames(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    return vars_infos_.at(var).orig_axis_name;
  }
  GELOGW("CreateExpr : [%s] has no ancestor", var.Str().get());
  std::vector<std::string> res;
  return res;
}

Expr ArgsManager::GetMaxValue(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    return vars_infos_.at(var).max_value;
  }
  GELOGW("CreateExpr : [%s] has no max value, set default value 1.", var.Str().get());
  return af::sym::kSymbolOne;
}

Expr ArgsManager::GetMinValue(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    return vars_infos_.at(var).min_value;
  }
  GELOGW("CreateExpr : [%s] has no min value, set default value 1.", var.Str().get());
  return af::sym::kSymbolOne;
}

Expr ArgsManager::GetDefaultInitValue(const Expr &var) const {
  if (vars_infos_.find(var) != vars_infos_.end()) {
    return vars_infos_.at(var).init_value;
  }
  GELOGW("CreateExpr : [%s] has no init value, set default value 1.", var.Str().get());
  return af::sym::kSymbolOne;
}

Expr ArgsManager::GetVarAlignValue(const Expr &var) const {
  if (vars_infos_.find(var) == vars_infos_.end()) {
    GELOGE(af::FAILED, "CreateExpr : [%s] is not defined", var.Str().get());
    return af::Symbol(0U);
  }
  return vars_infos_.at(var).align;
}

uint32_t ArgsManager::GetVarPromptAlignValue(const Expr &var) const {
  if (vars_infos_.find(var) == vars_infos_.end()) {
    GELOGE(af::FAILED, "CreateExpr : [%s] is not defined", var.Str().get());
    return 0u;
  }
  return vars_infos_.at(var).prompt_align;
}

uint32_t ArgsManager::GetDataTypeSizeVar(const Expr &var) const {
  const auto iter = vars_infos_.find(var);
  GE_ASSERT_TRUE(iter != vars_infos_.end(), "CreateExpr : [%s] is not defined", var.Str().get());
  return iter->second.data_type_size;
}

bool ArgsManager::IsConcatOuterDim(const Expr &var) const {
  if (vars_infos_.find(var) == vars_infos_.end()) {
    GELOGE(af::FAILED, "CreateExpr : [%s] is not defined", var.Str().get());
    return false;
  }
  return vars_infos_.at(var).is_concat_outer_dim;
}

bool ArgsManager::IsConcatInnerDim(const Expr &var) const {
  if (vars_infos_.find(var) == vars_infos_.end()) {
    GELOGE(af::FAILED, "CreateExpr : [%s] is not defined", var.Str().get());
    return false;
  }
  return vars_infos_.at(var).is_concat_inner_dim;
}

bool ArgsManager::SetSolvedVars(const std::vector<Expr> &vars) {
  for (const auto &var : vars) {
    auto &var_info = vars_infos_[var];
    var_info.do_search = false;
    solved_vars_.emplace_back(var);
  }
  return true;
}

std::vector<Expr> ArgsManager::GetSolvedVars() const {
  return solved_vars_;
}

std::vector<Expr> ArgsManager::GetTotalCutCons() const {
  std::vector<Expr> res(cut_leq_cons_);
  return res;
}

std::vector<Expr> ArgsManager::GetParentVars(const Expr &var) const {
  std::vector<Expr> parent_vars;
  if (vars_infos_.find(var) == vars_infos_.end()) {
    return parent_vars;
  }
  return vars_infos_.at(var).from_axis_size;
}

std::vector<Expr> ArgsManager::GetNodeInnerestDimSizes() const {
  std::vector<Expr> res;
  for (const auto &var_info : vars_infos_) {
    if ((var_info.second.do_search) && (var_info.second.is_node_innerest_dim_size)) {
      res.emplace_back(var_info.first);
    }
  }
  return res;
}

uint32_t ArgsManager::GetTilingCaseId() const {
  uint32_t tiling_case_id = model_info_.tiling_case_id;
  return tiling_case_id;
}

const ExprExprMap &ArgsManager::GetContainerMap() const {
  return model_info_.variable_expr_map;
}

const std::map<Expr, std::string, ExprCmp> &ArgsManager::GetContainerNames() const {
  return model_info_.variable_name_map;
}

std::vector<std::pair<Expr, Expr>> ArgsManager::GetTernaryOpReplaceVars() const {
  return ConcursiveReplaceVars(ternary_op_);
}

std::map<Expr, std::vector<Expr>, ExprCmp> ArgsManager::GetTernaryOpRelatedVars() const {
  return ConcursiveRelatedVars(ternary_op_);
}

const std::map<Expr, TernaryOp, ExprCmp> &ArgsManager::GetTernaryOps() const {
  return ternary_op_;
}

const std::vector<PerfBreakdownGroup> &ArgsManager::GetPerfBreakdowns() const {
  return perf_breakdowns_;
}

const ModelInfo &ArgsManager::GetModelInfo() const {
  return model_info_;
}

Expr ArgsManager::GetHeadCost() const {
  const auto iter = hardware_cons_.find(HardwareDef::CORENUM);
  if (iter != hardware_cons_.end()) {
    Expr res = iter->second;
    res = res * model_info_.head_cost;
    return res;
  }
  GELOGW("CoreNum is not found, HeadCost is zero.");
  return CreateExpr(0);
}

ExprUintMap ArgsManager::GetAxesPriority() const {
  ExprUintMap axes_pirority;
  uint32_t priority = 0u;
  for (const auto &arg_axis : model_info_.arg_list) {
    GE_ASSERT_NOTNULL(arg_axis);
    GE_ASSERT_NOTNULL(arg_axis->size);
    axes_pirority[arg_axis->size->symbol_expr] = priority++;
  }
  return axes_pirority;
}

ExprUintMap ArgsManager::GetAxesOrder() const {
  ExprUintMap original_orders;
  std::set<Expr, ExprCmp> ambiguous_originals;
  uint32_t next_order = 1U;
  for (const auto &arg_axis : model_info_.arg_list) {
    GE_ASSERT_NOTNULL(arg_axis);
    GE_ASSERT_NOTNULL(arg_axis->size);
    const uint32_t order = static_cast<uint32_t>(arg_axis->order);
    next_order = std::max(next_order, order + 1U);
    const auto result = original_orders.emplace(arg_axis->size->symbol_expr, order);
    if (!result.second) {
      ambiguous_originals.emplace(arg_axis->size->symbol_expr);
    }
  }
  const ExprExprMap var_relations = GetVarsRelations();
  const std::vector<Expr> searchable_vars = GetSearchableVars();
  std::map<Expr, size_t, ExprCmp> original_counts;
  for (const auto &var : searchable_vars) {
    const auto relation_iter = var_relations.find(var);
    const Expr &original_var = relation_iter == var_relations.end() ? var : relation_iter->second;
    ++original_counts[original_var];
  }
  ExprUintMap axes_order;
  for (const auto &var : searchable_vars) {
    Expr original_var = var;
    const auto relation_iter = var_relations.find(var);
    if (relation_iter != var_relations.end()) {
      original_var = relation_iter->second;
    }
    const auto order_iter = original_orders.find(original_var);
    if ((order_iter != original_orders.end()) && (ambiguous_originals.count(original_var) == 0U) &&
        (original_counts.at(original_var) == 1U)) {
      axes_order[var] = order_iter->second;
    } else {
      axes_order[var] = next_order++;
    }
  }
  return axes_order;
}

void ArgsManager::Reset() {
  vars_infos_.clear();
  hardware_cons_.clear();
  cut_leq_cons_.clear();
  objs_.clear();
  ori_var_init_values_.clear();
  ori_var_max_values_.clear();
  ori_var_align_values_.clear();
  solved_vars_.clear();
  perf_breakdowns_.clear();
}

}  // namespace att
