/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attribute_group/attr_group_shape_env.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/detail/attributes_holder.h"
#include "proto/af_ir.pb.h"
#include "graph/expression/expression_impl.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace {
static thread_local ShapeEnvAttr *shape_env_context{nullptr};
static std::map<ge::DataType, std::string> kGeDType2CppDtype = {
    {ge::DT_INT32, "int32_t"},
    {ge::DT_INT64, "int64_t"},
    {ge::DT_UINT32, "uint32_t"},
    {ge::DT_UINT64, "uint64_t"},
};
}  // namespace

thread_local std::string ShapeEnvAttr::guard_dfx_info_ = "";
ShapeEnvAttr *GetCurShapeEnvContext() {
  return shape_env_context;
}

void SetCurShapeEnvContext(ShapeEnvAttr *shape_env) {
  shape_env_context = shape_env;
}

std::string Source::GetGlobalIndexStr() const {
  return "context->GetInputPointer<int64_t>(" + std::to_string(global_index_) + ")";
}

graphStatus ShapeEnvAttr::SerializeSymbolInfo(proto::ShapeEnvAttrGroupsDef *shape_env_group) {
  GE_ASSERT_NOTNULL(shape_env_group);
  shape_env_group->clear_symbol_to_value();
  auto symbol_to_value_def = shape_env_group->mutable_symbol_to_value();
  GE_ASSERT_NOTNULL(symbol_to_value_def);
  GELOGI("symbol_to_value size: %zu", symbol_to_value_.size());
  for (const auto &iter : symbol_to_value_) {
    GE_ASSERT_TRUE(!iter.first.IsConstExpr(),
                   "Symbol in symbol_to_value of shape env attr should be a variable, but get: %s",
                   iter.first.Str().get());
    symbol_to_value_def->insert({iter.first.Str().get(), iter.second});
  }
  auto value_to_symbol_def = shape_env_group->mutable_value_to_symbol();
  GE_ASSERT_NOTNULL(value_to_symbol_def);
  for (const auto &iter : value_to_symbol_) {
    GE_ASSERT_TRUE(!iter.second.empty());
    proto::SymbolInfoDef symbol_infos;
    for (const auto &sym_iter : iter.second) {
      GE_ASSERT_TRUE(!sym_iter.IsConstExpr(),
                     "Symbol in value_to_symbol of shape env attr should be a variable, but get: %s",
                     sym_iter.Str().get());
      symbol_infos.add_symbols(sym_iter.Str().get());
    }
    value_to_symbol_def->insert({iter.first, symbol_infos});
  }

  auto symbol_to_source_def = shape_env_group->mutable_symbol_to_source();
  GE_ASSERT_NOTNULL(symbol_to_source_def);
  // todoo: symbol_to_source_实现序列化
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::SerializeSymbolCheckInfos(proto::ShapeEnvAttrGroupsDef *shape_env_group) {
  GE_ASSERT_NOTNULL(shape_env_group);
  auto replacements_def = shape_env_group->mutable_replacements();
  for (const auto &iter : replacements_) {
    proto::ReplacementDef rep_def;
    rep_def.set_replace_expr(iter.second.replace_expr.Str().get());
    rep_def.set_rank(iter.second.rank);
    replacements_def->insert({iter.first.Str().get(), rep_def});
  }
  shape_env_group->clear_symbol_check_infos();
  for (const auto &iter : symbol_check_infos_) {
    proto::SymbolCheckInfoDef *symbol_check_info_def = shape_env_group->add_symbol_check_infos();
    symbol_check_info_def->set_expr(iter.expr.Str().get());
    symbol_check_info_def->set_file(iter.file);
    symbol_check_info_def->set_line(iter.line);
    symbol_check_info_def->set_dfx(iter.dfx_info);
  }
  shape_env_group->clear_symbol_assert_infos();
  for (const auto &iter : symbol_assert_infos_) {
    proto::SymbolCheckInfoDef *symbol_assert_info_def = shape_env_group->add_symbol_assert_infos();
    symbol_assert_info_def->set_expr(iter.expr.Str().get());
    symbol_assert_info_def->set_file(iter.file);
    symbol_assert_info_def->set_line(iter.line);
    symbol_assert_info_def->set_dfx(iter.dfx_info);
  }
  return GRAPH_SUCCESS;
}

ShapeEnvAttr::ShapeEnvAttr(const ShapeEnvAttr &other) {
  shape_env_setting_ = other.shape_env_setting_;
  symbol_to_value_ = other.symbol_to_value_;
  value_to_symbol_ = other.value_to_symbol_;
  symbol_to_source_ = other.symbol_to_source_;
  replacements_ = other.replacements_;
  symbol_check_infos_ = other.symbol_check_infos_;
  symbol_assert_infos_ = other.symbol_assert_infos_;
  unique_sym_id_ = other.unique_sym_id_;
}

ShapeEnvAttr &ShapeEnvAttr::operator=(const ShapeEnvAttr &other) {
  if (this != &other) {
    shape_env_setting_ = other.shape_env_setting_;
    symbol_to_value_ = other.symbol_to_value_;
    value_to_symbol_ = other.value_to_symbol_;
    symbol_to_source_ = other.symbol_to_source_;
    replacements_ = other.replacements_;
    symbol_check_infos_ = other.symbol_check_infos_;
    symbol_assert_infos_ = other.symbol_assert_infos_;
    unique_sym_id_ = other.unique_sym_id_;
  }
  return *this;
}

graphStatus ShapeEnvAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto shape_env_group = attr_group_def.mutable_shape_env_attr_group();
  GE_ASSERT_SUCCESS(SerializeSymbolInfo(shape_env_group));
  GE_ASSERT_SUCCESS(SerializeSymbolCheckInfos(shape_env_group));
  proto::ShapeEnvSettingDef *shape_env_setting_def = shape_env_group->mutable_shape_setting();
  shape_env_setting_def->set_specialize_zero_one(shape_env_setting_.specialize_zero_one);
  shape_env_setting_def->set_dynamic_mode(static_cast<int32_t>(shape_env_setting_.dynamic_mode));
  shape_env_group->set_unique_sym_id(unique_sym_id_);
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::DeserializeSymbolInfo(const proto::ShapeEnvAttrGroupsDef &shape_env_group) {
  symbol_to_value_.clear();
  GELOGI("symbol_to_value size: %zu", shape_env_group.symbol_to_value_size());
  for (const auto &iter : shape_env_group.symbol_to_value()) {
    Expression sym = Expression::Deserialize(iter.first.c_str());
    GE_ASSERT_TRUE(!sym.IsConstExpr(), "Symbol in symbol_to_value of shape env attr should be a variable, but get: %s",
                   iter.first.c_str());
    symbol_to_value_.emplace(std::make_pair(sym, iter.second));
  }
  value_to_symbol_.clear();
  for (const auto &iter : shape_env_group.value_to_symbol()) {
    std::vector<Expression> symbol_infos;
    for (const auto &sym_iter : iter.second.symbols()) {
      Expression sym = Expression::Deserialize(sym_iter.c_str());
      GE_ASSERT_TRUE(!sym.IsConstExpr(),
                     "Symbol in value_to_symbol of shape env attr should be a variable, but get: %s", sym_iter.c_str());
      symbol_infos.emplace_back(sym);
    }
    value_to_symbol_.emplace(std::make_pair(iter.first, symbol_infos));
  }
  symbol_to_source_.clear();
  // todoo: symbol_to_source_实现反序列化
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::DeserializeSymbolCheckInfos(const proto::ShapeEnvAttrGroupsDef &shape_env_group) {
  replacements_.clear();
  for (const auto &iter : shape_env_group.replacements()) {
    Expression expr = Expression::Deserialize(iter.first.c_str());
    Expression replace_expr = Expression::Deserialize(iter.second.replace_expr().c_str());
    replacements_.emplace(std::make_pair(expr, Replacement(replace_expr, iter.second.rank())));
  }
  symbol_check_infos_.clear();
  for (const auto &iter : shape_env_group.symbol_check_infos()) {
    Expression expr = Expression::Deserialize(iter.expr().c_str());
    symbol_check_infos_.emplace(SymbolCheckInfo(expr, iter.file(), iter.line(), iter.dfx()));
  }
  symbol_assert_infos_.clear();
  for (const auto &iter : shape_env_group.symbol_assert_infos()) {
    Expression expr = Expression::Deserialize(iter.expr().c_str());
    symbol_assert_infos_.emplace(SymbolCheckInfo(expr, iter.file(), iter.line(), iter.dfx()));
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  (void)attr_holder;
  const auto &shape_env_group = attr_group_def.shape_env_attr_group();
  DeserializeSymbolInfo(shape_env_group);
  DeserializeSymbolCheckInfos(shape_env_group);
  shape_env_setting_ = ShapeEnvSetting(shape_env_group.shape_setting().specialize_zero_one(),
                                       static_cast<DynamicMode>(shape_env_group.shape_setting().dynamic_mode()));
  unique_sym_id_ = shape_env_group.unique_sym_id();
  return GRAPH_SUCCESS;
}

std::unique_ptr<AttrGroupsBase> ShapeEnvAttr::CloneAf() {
  std::unique_ptr<AttrGroupsBase> new_attr = ComGraphMakeUnique<ShapeEnvAttr>(*this);
  GE_ASSERT_NOTNULL(new_attr);
  return new_attr;
}

bool ShapeEnvAttr::HasSymbolCheckInfo(const Expression &e) const {
  auto expr = e.CanonicalizeBoolExpr();
  if (symbol_check_infos_.find(SymbolCheckInfo(expr)) != symbol_check_infos_.end()) {
    return true;
  }
  return false;
}

bool ShapeEnvAttr::HasSymbolAssertInfo(const Expression &e) const {
  auto expr = e.CanonicalizeBoolExpr();
  if (symbol_assert_infos_.find(SymbolCheckInfo(expr)) != symbol_assert_infos_.end()) {
    return true;
  }
  return false;
}

Expression ShapeEnvAttr::FindReplacements(const Expression &expr) {
  auto iter = replacements_.find(expr);
  if (iter == replacements_.end()) {
    return expr;
  }
  if (iter->second.has_replace) {
    GELOGD("Found replacement expr: %s for expr: %s", iter->second.replace_expr.Str().get(), expr.Str().get());
    return expr;
  }
  auto replace_expr = iter->second.replace_expr;
  GELOGD("Find replace expr: %s of expr: %s", replace_expr.Str().get(), expr.Str().get());
  if (replace_expr == expr) {
    return expr;
  }
  std::vector<std::pair<Expression, Expression>> var_replacements;
  iter->second.has_replace = true;
  for (auto &sym : replace_expr.FreeSymbols()) {
    auto replace_sym = FindReplacements(sym);
    var_replacements.emplace_back(std::make_pair(sym, replace_sym));
  }
  iter->second.has_replace = false;
  return replace_expr.Replace(var_replacements);
}

const std::vector<SymbolCheckInfo> ShapeEnvAttr::GetAllSymbolCheckInfos() const {
  std::vector<SymbolCheckInfo> results;
  for (const auto &iter : symbol_check_infos_) {
    results.emplace_back(iter);
  }
  return results;
}

const std::vector<SymbolCheckInfo> ShapeEnvAttr::GetAllSymbolAssertInfos() const {
  std::vector<SymbolCheckInfo> results;
  for (const auto &iter : symbol_assert_infos_) {
    results.emplace_back(iter);
  }
  return results;
};

void ShapeEnvAttr::SimplifySymbolCheckInfo(
    std::set<SymbolCheckInfo, SymbolCheckInfoKeyLess> &symbol_check_infos) const {
  std::vector<SymbolCheckInfo> simplify_symbol_check_info;
  for (auto &iter : symbol_check_infos) {
    const auto simple_expr = iter.expr.Simplify().CanonicalizeBoolExpr();
    if (simple_expr.IsConstExpr()) {
      continue;
    }
    (void)simplify_symbol_check_info.emplace_back(SymbolCheckInfo(simple_expr));
  }
  (void)symbol_check_infos.insert(simplify_symbol_check_info.begin(), simplify_symbol_check_info.end());
}

void ShapeEnvAttr::SimplifySymbolCheckInfo() {
  GELOGD("Start simplify guard");
  SimplifySymbolCheckInfo(symbol_check_infos_);
  SimplifySymbolCheckInfo(symbol_assert_infos_);
}

Expression ShapeEnvAttr::Simplify(const Expression &expr) {
  std::vector<std::pair<Expression, Expression>> var_replacements;
  // 初始化replacements遍历状态
  for (auto &iter : replacements_) {
    iter.second.has_replace = false;
  }
  for (const auto &sym : expr.FreeSymbols()) {
    auto replace_expr = FindReplacements(sym);
    if ((!replace_expr.IsVariableExpr()) || (replace_expr != sym)) {
      var_replacements.emplace_back(std::make_pair(sym, replace_expr));
    }
  }
  if (!var_replacements.empty()) {
    auto result_expr = expr.Replace(var_replacements);
    GELOGI("Simplify expr: %s to expr: %s", SymbolicUtils::ToString(expr).c_str(),
           SymbolicUtils::ToString(result_expr).c_str());
    GE_ASSERT_NOTNULL(result_expr.impl_);
    return Expression(result_expr.impl_->Simplify());
  }
  return Expression(expr.impl_->Simplify());
}

Expression ShapeEnvAttr::EvaluateExpr(const Expression &expr) {
  std::vector<std::pair<Expression, Expression>> var_to_val;
  auto free_symbols = expr.FreeSymbols();
  for (const auto &free_sym : free_symbols) {
    const auto &iter = symbol_to_value_.find(free_sym);
    if (iter != symbol_to_value_.end()) {
      var_to_val.emplace_back(std::make_pair(free_sym, Symbol(iter->second)));
    }
  }
  return expr.Subs(var_to_val);
}

TriBool ShapeEnvAttr::HasSymbolInfo(const Expression &expr) const {
  Expression e = expr.CanonicalizeBoolExpr();
  if (HasSymbolCheckInfo(e) || HasSymbolAssertInfo(e)) {
    return TriBool::kTrue;
  }
  return TriBool::kUnknown;
}

void ShapeEnvAttr::AppendInitReplacement(const Expression &expr) {
  if (replacements_.find(expr) == replacements_.end()) {
    (void)replacements_.emplace(std::make_pair(expr, Replacement(expr, 1)));
  }
}

graphStatus ShapeEnvAttr::FindRootExpr(const Expression &expr, Expression &root_expr) const {
  const auto &iter = replacements_.find(expr);
  GE_ASSERT_TRUE(iter != replacements_.end(), "Can not find replacement of expr: %s",
                 SymbolicUtils::ToString(expr).c_str());
  if (iter->second.replace_expr == expr) {
    root_expr = expr;
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_SUCCESS(FindRootExpr(iter->second.replace_expr, root_expr));
  return GRAPH_SUCCESS;
}

std::vector<std::pair<Expression, SourcePtr>> ShapeEnvAttr::GetAllSym2Src() {
  std::vector<std::pair<Expression, SourcePtr>> result;
  for (const auto &iter : symbol_to_source_) {
    result.emplace_back(iter.first, iter.second);
  }
  return result;
}

bool Replacement::operator<=(const Replacement &other) {
  // 并查集的根节点优先级： 常数 > 表达式 > 变量
  if (replace_expr.IsConstExpr()) {
    if (other.replace_expr.IsConstExpr()) {
      return rank <= other.rank;
    }
    return false;
  }
  if (replace_expr.IsVariableExpr()) {
    if (other.replace_expr.IsVariableExpr()) {
      return rank <= other.rank;
    }
    return true;
  }
  if (other.replace_expr.IsConstExpr()) {
    return true;
  }
  if (other.replace_expr.IsVariableExpr()) {
    return false;
  }
  return rank <= other.rank;
}

graphStatus ShapeEnvAttr::MergeReplacement(const Expression &expr1, const Expression &expr2) {
  Expression father_expr1;
  GE_ASSERT_SUCCESS(FindRootExpr(expr1, father_expr1));
  Expression father_expr2;
  GE_ASSERT_SUCCESS(FindRootExpr(expr2, father_expr2));
  auto &replacement_1 = replacements_[father_expr1];
  auto &replacement_2 = replacements_[father_expr2];
  if (replacement_1 <= replacement_2) {
    replacement_1.replace_expr = father_expr2;
    if (replacement_2.rank <= replacement_1.rank) {
      replacement_2.rank = replacement_1.rank + 1;
    }
  } else {
    replacement_2.replace_expr = father_expr1;
    if (replacement_1.rank <= replacement_2.rank) {
      replacement_1.rank = replacement_2.rank + 1;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::MergePath() {
  for (auto &iter : replacements_) {
    Expression root_expr;
    GE_ASSERT_SUCCESS(FindRootExpr(iter.first, root_expr));
    iter.second.replace_expr = root_expr;
    iter.second.rank = 1;
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendReplacement(const Expression &target, const Expression &replacement) {
  if (target == replacement) {
    return GRAPH_SUCCESS;
  }
  Expression expr1 = target;
  Expression expr2 = replacement;
  auto expr = sym::Eq(target, replacement).CanonicalizeBoolExpr();
  std::vector<Expression> args = expr.GetArgs();
  if (args.size() == kSizeTwo) {
    expr1 = args[0];
    expr2 = args[1];
    GELOGD("expr1 %s->%s, expr2 %s->%s", SymbolicUtils::ToString(target).c_str(),
           SymbolicUtils::ToString(expr1).c_str(), SymbolicUtils::ToString(replacement).c_str(),
           SymbolicUtils::ToString(expr2).c_str());
  }

  // 仅支持 符号->常量，符号->表达式，符号->符号 映射
  if (expr1.IsConstExpr()) {
    if (!expr2.IsVariableExpr()) {
      GELOGW("Unsupported append replacement %s to %s", SymbolicUtils::ToString(expr1).c_str(),
             SymbolicUtils::ToString(expr2).c_str());
      return GRAPH_SUCCESS;
    }
  } else if (!expr1.IsVariableExpr()) {
    if (!expr2.IsVariableExpr()) {
      GELOGW("Unsupported append replacement %s to %s", SymbolicUtils::ToString(expr1).c_str(),
             SymbolicUtils::ToString(expr2).c_str());
      return GRAPH_SUCCESS;
    }
  }
  // 判断replacement是否成环：成环时跳过该条替换（保留既有替换集合不变），
  // 不返回失败——等式两侧共享符号的合法自引用约束（如 s1*s1 == s3）同样会命中
  // 该检测，若拒绝会让推导整体失败并污染 guard 判定结果，静默跳过语义等价于
  // 不建立该替换（guard 仍正常记录，仅替换关系不进并查集）
  if (CheckReplacementCycle(expr1, expr2)) {
    GELOGW("Unsupported append replacement %s to %s, replacement contains the other.",
           SymbolicUtils::ToString(expr1).c_str(), SymbolicUtils::ToString(expr2).c_str());
    return GRAPH_SUCCESS;
  }
  AppendInitReplacement(expr1);
  AppendInitReplacement(expr2);
  GE_ASSERT_SUCCESS(MergeReplacement(expr1, expr2));
  // 路径压缩
  GE_ASSERT_SUCCESS(MergePath());
  // replace插入后全量化简已有的guard
  SimplifySymbolCheckInfo();
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendSymbolAssertInfo(const Expression &expr, const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Assert expr: %s should be boolean", SymbolicUtils::ToString(expr).c_str());
  if (!expr.IsConstExpr()) {
    (void)symbol_assert_infos_.emplace(
        SymbolCheckInfo(expr.CanonicalizeBoolExpr(), file, line, GetGuardDfxContextInfo()));
  }
  return GRAPH_SUCCESS;
}

graphStatus ShapeEnvAttr::AppendSymbolCheckInfo(const Expression &expr, const std::string &file, const int64_t line) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Check expr: %s should be boolean", SymbolicUtils::ToString(expr).c_str());
  if (!expr.IsConstExpr()) {
    (void)symbol_check_infos_.emplace(
        SymbolCheckInfo(expr.CanonicalizeBoolExpr(), file, line, GetGuardDfxContextInfo()));
  }
  return GRAPH_SUCCESS;
}

void ShapeEnvAttr::SetGuardDfxContextInfo(const std::string &guard_dfx_info) const {
  guard_dfx_info_ = guard_dfx_info;
}

void ShapeEnvAttr::ClearGuardDfxContextInfo() const {
  guard_dfx_info_.clear();
}

std::string ShapeEnvAttr::GetGuardDfxContextInfo() const {
  return guard_dfx_info_;
}

bool ShapeEnvAttr::CheckReplacementCycle(const Expression &expr1, const Expression &expr2) const {
  Expression root_expr1;
  if (replacements_.find(expr1) == replacements_.end()) {
    root_expr1 = expr1;
  } else {
    GE_ASSERT_SUCCESS(FindRootExpr(expr1, root_expr1));
  }
  Expression root_expr2;
  if (replacements_.find(expr2) == replacements_.end()) {
    root_expr2 = expr2;
  } else {
    GE_ASSERT_SUCCESS(FindRootExpr(expr2, root_expr2));
  }

  if (root_expr1 == root_expr2) {
    return false;
  }

  /*
   * 判断expr1与expr2是否互相包含：
   * 旧实现仅在某一侧根为单符号时用 ContainVar+Simplify 判断，两根均为复合表达式时
   * 直接漏判（例如已有 replacement: s0 == s2，再 Append(s0 + s1, s2)：s0 的根是 s2，
   * expr1 经符号根解析后包含 s2，构成环，旧实现检测不到）。
   * 新实现逐表达式枚举自由符号并解析其替换根，与对方根比较即可覆盖复合表达式对，
   * 且无需对根表达式做化简展开（化简在替换链较长时代价高）。
   */
  const auto contains_root = [this](const Expression &expr, const Expression &target_root) {
    for (const auto &symbol : expr.FreeSymbols()) {
      Expression symbol_root = symbol;
      if (replacements_.find(symbol) != replacements_.end() && FindRootExpr(symbol, symbol_root) != GRAPH_SUCCESS) {
        return true;
      }
      if (symbol_root == target_root) {
        return true;
      }
      // 根表达式包含对方根的自由符号时构成间接环：如已有 s3 == Min(2, s2) 时再建立
      // s2 == (s0+s3)*Ceil(s1)，替换链闭合出 s2 -> 复合 -> s3 -> Min(2, s2) 的环
      // （ContainVar 仅接受无参表达式，复合根需逐自由符号检查）
      for (const auto &target_sym : target_root.FreeSymbols()) {
        if (symbol_root.ContainVar(target_sym)) {
          return true;
        }
      }
    }
    return false;
  };
  return contains_root(root_expr1, root_expr2) || contains_root(root_expr2, root_expr1);
}

}  // namespace af
