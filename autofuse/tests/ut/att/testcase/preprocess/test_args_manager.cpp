/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/preprocess/var_info.h"
#include "common/util/mem_utils.h"
#include <iostream>
#include <fstream>
#define private public
#include "generator/preprocess/args_replace.h"
#include "generator/preprocess/args_manager.h"
#define private public
#define protected public
#include "generator/tiling_code_gen_impl.h"
#include "generator/axes_reorder_tiling_code_gen_impl.h"
using namespace att;
using namespace testing;

namespace {
SymVarInfoPtr MakeSymVar(const Expr &expr) {
  return std::make_shared<SymVarInfo>(expr);
}

AttAxisPtr MakeAxis(const std::string &name, AxisPosition pos, bool bind_mc, bool is_last, bool is_innerest,
                    const SymVarInfoPtr &size, const std::vector<AttAxis *> &from = {}) {
  auto axis = std::make_shared<AttAxis>();
  axis->name = name;
  axis->axis_pos = pos;
  axis->bind_multicore = bind_mc;
  axis->is_last = is_last;
  axis->is_node_innerest_dim = is_innerest;
  axis->size = size;
  for (auto parent : from) {
    axis->orig_axis.push_back(parent);
  }
  axis->from_axis = from;
  return axis;
}

void SetSymVarScope(const SymVarInfoPtr &sym, const std::initializer_list<HardwareDef> &scopes) {
  sym->align = af::Symbol(16);
  sym->related_scope = scopes;
}
}  // namespace

class ArgsManagerUtest : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}

  void TearDown() override {}
  void dumps(const std::string &res) {
    std::cout << res << std::endl;
  }
};

TEST_F(ArgsManagerUtest, test_set_size_info) {
  VarInfo info;
  SymVarInfoPtr sym_var = std::make_shared<SymVarInfo>(CreateExpr("basem_size"));
  SymVarInfoPtr sym_var1 = std::make_shared<SymVarInfo>(CreateExpr("stepm_size"));
  SymVarInfoPtr sym_var2 = std::make_shared<SymVarInfo>(CreateExpr("tilem_size"));
  AttAxisPtr axis = std::make_shared<AttAxis>();
  axis->name = "basem";
  AttAxisPtr ori_axis1 = std::make_shared<AttAxis>();
  ori_axis1->name = "stepm";
  AttAxisPtr ori_axis2 = std::make_shared<AttAxis>();
  ori_axis2->name = "tilem";
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(info.max_value == 1);
  axis->orig_axis.emplace_back(ori_axis1.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(info.max_value == 1);
  axis->orig_axis.clear();
  ori_axis1->size = sym_var1;
  axis->orig_axis.emplace_back(ori_axis1.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(IsValid(info.max_value));
  EXPECT_TRUE(info.max_value == CreateExpr("stepm_size"));
  ori_axis2->size = sym_var2;
  axis->orig_axis.emplace_back(ori_axis2.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(IsValid(info.max_value));
  EXPECT_TRUE(info.max_value == (CreateExpr("stepm_size") * CreateExpr("tilem_size")));
}

TEST_F(ArgsManagerUtest, test_set_init_info) {
  VarInfo info;
  info.is_const_var = true;
  info.const_value = 1u;
  info = ArgsManager::SetInitSize(info, false);
  EXPECT_TRUE(info.init_value == 1);
  info = ArgsManager::SetInitSize(info, true);
  EXPECT_TRUE(info.init_value == 1);
  info.is_const_var = false;
  info.max_value = CreateExpr("m");
  info.align = af::Symbol(16u);
  info = ArgsManager::SetInitSize(info, false);
  EXPECT_TRUE(info.init_value == 16);
  info = ArgsManager::SetInitSize(info, true);
  EXPECT_TRUE(info.init_value == CreateExpr("m"));
}

ModelInfo StubModelInfo() {
  ModelInfo model_info;
  Expr expr_m = CreateExpr("m_size");
  Expr expr_tilem = CreateExpr("tilem_size");
  Expr expr_stepm = CreateExpr("stepm_size");
  Expr expr_basem = CreateExpr("basem_size");

  auto sym_m = MakeSymVar(expr_m);
  auto sym_tilem = MakeSymVar(expr_tilem);
  SetSymVarScope(sym_tilem, {HardwareDef::L2});
  auto sym_stepm = MakeSymVar(expr_stepm);
  SetSymVarScope(sym_stepm, {HardwareDef::L1, HardwareDef::CORENUM});
  auto sym_basem = MakeSymVar(expr_basem);
  SetSymVarScope(sym_basem, {HardwareDef::L0A, HardwareDef::L0C});

  auto m = MakeAxis("m", AxisPosition::ORIGIN, false, false, false, sym_m);
  auto tilem = MakeAxis("tilem", AxisPosition::INNER, false, false, true, sym_tilem, {m.get()});
  auto stepm = MakeAxis("stepm", AxisPosition::INNER, true, false, true, sym_stepm, {tilem.get()});
  auto basem = MakeAxis("basem", AxisPosition::INNER, false, true, true, sym_basem, {stepm.get()});
  model_info.arg_list = {m, tilem, stepm, basem};

  model_info.hardware_cons[HardwareDef::L0A] = expr_basem * CreateExpr(4);
  model_info.hardware_cons[HardwareDef::L0C] = expr_basem * CreateExpr(4);
  model_info.hardware_cons[HardwareDef::L1] = expr_stepm * CreateExpr(4);
  model_info.hardware_cons[HardwareDef::L2] = expr_tilem * CreateExpr(2);
  model_info.hardware_cons[HardwareDef::CORENUM] = expr_tilem / expr_stepm;
  model_info.hardware_cons[HardwareDef::UB] = CreateExpr(0L);
  model_info.objects[PipeType::AIC_MAC] = expr_basem / CreateExpr(256);
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepm, expr_basem));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_stepm));
  return model_info;
}

TEST_F(ArgsManagerUtest, test_set_ori_exprs) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.vars_infos_ = ArgsManager::GetOrigVarInfos(info);
  args_manager.SetOrigExprs();
  EXPECT_EQ(args_manager.hardware_cons_.size(), 6u);
  EXPECT_EQ(args_manager.cut_leq_cons_.size(), 1u);
  EXPECT_EQ(args_manager.cut_eq_cons_.size(), 0);
}

TEST_F(ArgsManagerUtest, test_replace_vars) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.vars_infos_ = ArgsManager::GetOrigVarInfos(info);
  args_manager.SetOrigExprs();
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  auto res = args_manager.ReplaceVars(replaced_vars, replacements, new_expr_replacements);
  EXPECT_TRUE(res);
  EXPECT_EQ(replaced_vars.size(), 3u);
  EXPECT_EQ(replacements.size(), 3u);
  EXPECT_TRUE(replacements.find(CreateExpr("basem_size")) != replacements.end());
  EXPECT_TRUE(replacements[CreateExpr("basem_size")] ==
              (CreateExpr(16) * af::sym::Pow(CreateExpr(2), CreateExpr("basem_size_base"))));
  EXPECT_TRUE(replacements.find(CreateExpr("stepm_size")) != replacements.end());
  EXPECT_TRUE(
      replacements[CreateExpr("stepm_size")] ==
      ((af::sym::Max((CreateExpr(16) * af::sym::Pow(CreateExpr(2), CreateExpr("basem_size_base"))), CreateExpr(16)) *
        CreateExpr("stepm_size_div_align"))));
  for (auto hard_cons : args_manager.hardware_cons_) {
    dumps("scope: " + std::to_string(static_cast<int32_t>(hard_cons.first)) + " --- " + Str(hard_cons.second));
  }
  for (auto obj : args_manager.objs_) {
    dumps("Pipe: " + std::to_string(static_cast<int32_t>(obj.first)) + " --- " + Str(obj.second));
  }
  for (auto leq_cons : args_manager.cut_leq_cons_) {
    dumps("Leq: " + Str(leq_cons));
  }
  for (auto eq_cons : args_manager.cut_eq_cons_) {
    dumps("Eq: " + Str(eq_cons.first / eq_cons.second));
  }
}

TEST_F(ArgsManagerUtest, test_update_vars) {
  ModelInfo model_info;
  ArgsManager args_manager(model_info);
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  replaced_vars.emplace(CreateExpr("basem_base"), CreateExpr("basem"));
  replacements.emplace(CreateExpr("basem"), (CreateExpr(16) * af::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))));
  replaced_vars.emplace(CreateExpr("stepm_align"), CreateExpr("stepm"));
  replacements.emplace(
      CreateExpr("stepm"),
      (af::sym::Max((CreateExpr(16) * af::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))), CreateExpr(16)) *
       CreateExpr("stepm_align")));
  new_expr_replacements.emplace(CreateExpr("basem_base"),
                                af::sym::Log(CreateExpr(2), (CreateExpr("basem") / CreateExpr(16))));
  new_expr_replacements.emplace(
      CreateExpr("stepm_align"),
      (CreateExpr("stepm") /
       af::sym::Max((CreateExpr(16) * af::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))), CreateExpr(16))));
  args_manager.ori_var_init_values_ = {{CreateExpr("basem"), CreateExpr(16)}, {CreateExpr("stepm"), CreateExpr(16)}};
  args_manager.ori_var_max_values_ = {{CreateExpr("basem"), CreateExpr("m")}, {CreateExpr("stepm"), CreateExpr("m")}};
  VarInfo basem_info;
  VarInfo stepm_info;
  VarInfo m_info;
  basem_info.do_search = true;
  basem_info.align = af::Symbol(16u);
  basem_info.init_value = CreateExpr("m");
  basem_info.max_value = CreateExpr("m");
  stepm_info.do_search = true;
  stepm_info.align = af::Symbol(16u);
  stepm_info.max_value = CreateExpr("m");
  stepm_info.init_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("basem"), basem_info);
  args_manager.vars_infos_.emplace(CreateExpr("stepm"), stepm_info);
  args_manager.vars_infos_.emplace(CreateExpr("m"), m_info);
  auto res = args_manager.UpdateVarInfos(replaced_vars, replacements, new_expr_replacements);
  EXPECT_TRUE(res);
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("basem")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("basem_base")].do_search);
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("stepm")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("stepm_align")].do_search);
  EXPECT_TRUE(IsValid(args_manager.vars_infos_[CreateExpr("stepm_align")].init_value));
  EXPECT_FALSE(!IsValid(args_manager.vars_infos_[CreateExpr("basem_base")].init_value));
  EXPECT_FALSE(!IsValid(args_manager.vars_infos_[CreateExpr("stepm_align")].max_value));
  EXPECT_TRUE(IsValid(args_manager.vars_infos_[CreateExpr("basem_base")].max_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("stepm_align")].init_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("basem_base")].max_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("basem_base")].init_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("stepm_align")].max_value));
}

TEST_F(ArgsManagerUtest, test_process_true) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.Process(true);
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_manager.replacement_done_);
  auto var_info_str = MakeJson(args_manager.vars_infos_);
  std::ofstream out_file("./vars_info_ut.json");
  if (out_file.is_open()) {
    out_file << var_info_str;
    out_file.close();
  }
}

TEST_F(ArgsManagerUtest, test_process_false) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.Process(false);
  EXPECT_TRUE(res);
  EXPECT_FALSE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_do_var_replace) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.DoVarsReplace();
  EXPECT_FALSE(res);
  EXPECT_FALSE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_do_var_replace_success) {
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.Process(false);
  auto res = args_manager.DoVarsReplace();
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_get_orig_var_info) {
  ModelInfo info = StubModelInfo();
  auto expr_infos = ArgsManager::GetOrigVarInfos(info);
  EXPECT_EQ(expr_infos.size(), 4);
  auto m_info = expr_infos[CreateExpr("m_size")];
  EXPECT_EQ(m_info.cut_eq_cons.size(), 0);
  EXPECT_EQ(m_info.cut_leq_cons.size(), 0);
  auto basem_info = expr_infos[CreateExpr("basem_size")];
  EXPECT_EQ(basem_info.cut_eq_cons.size(), 1);
  EXPECT_EQ(basem_info.cut_leq_cons.size(), 0);
  auto stepm_info = expr_infos[CreateExpr("stepm_size")];
  EXPECT_EQ(stepm_info.cut_eq_cons.size(), 1);
  EXPECT_EQ(stepm_info.cut_leq_cons.size(), 1);
  auto tilem_info = expr_infos[CreateExpr("tilem_size")];
  EXPECT_EQ(tilem_info.cut_eq_cons.size(), 0);
  EXPECT_EQ(tilem_info.cut_leq_cons.size(), 1);
}

TEST_F(ArgsManagerUtest, test_naive_var_info) {
  ModelInfo model_info;
  ArgsManager args_manager(model_info);
  AttAxisPtr axis = std::make_shared<AttAxis>();
  auto info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.scopes.empty());
  SymVarInfoPtr var_size = std::make_shared<SymVarInfo>(CreateExpr("m"));
  SymConstInfoPtr const_size = std::make_shared<SymConstInfo>(CreateExpr("m"));
  const_size->const_value = 16u;
  axis->size = var_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_FALSE(info.do_search);
  axis->axis_pos = AxisPosition::ORIGIN;
  axis->size = var_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_FALSE(info.do_search);
  EXPECT_TRUE(info.is_input_var);
  axis->axis_pos = AxisPosition::INNER;
  axis->size = var_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.do_search);
  EXPECT_TRUE(info.from_axis_size.empty());
  axis->size = const_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.is_const_var);
  EXPECT_EQ(info.const_value, 16u);
  EXPECT_FALSE(info.do_search);

  auto parent_axis = std::make_unique<AttAxis>();
  parent_axis->name = "ori_m";
  parent_axis->axis_pos = AxisPosition::ORIGIN;
  axis->from_axis.emplace_back(parent_axis.get());
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.from_axis_size.empty());
  axis->from_axis[0]->size = std::make_shared<SymVarInfo>(CreateExpr("ori_m_size"));
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_EQ(Str(info.from_axis_size[0]), "ori_m_size");
  EXPECT_TRUE(info.orig_axis_size.empty());
  axis->orig_axis = axis->from_axis;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_EQ(Str(info.orig_axis_size[0]), "ori_m_size");
  info = ArgsManager::GetNaiveVarInfo(axis->from_axis[0]);
  EXPECT_EQ(Str(info.orig_axis_size[0]), "ori_m_size");
}

TEST_F(ArgsManagerUtest, test_get_searchable_vars) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetSearchableVars();
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_searchable_vars2) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  var_info1.scopes.push_back(HardwareDef::L0A);
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.scopes.push_back(HardwareDef::L0A);
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetSearchableVars(HardwareDef::L0A);
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
  exprs = args_manager.GetSearchableVars(HardwareDef::L0B);
  EXPECT_EQ(exprs.size(), 0);
}

TEST_F(ArgsManagerUtest, test_get_axes_order_uses_unique_fallback_for_missing_or_ambiguous_axes) {
  const Expr valid = CreateExpr("valid");
  const Expr ambiguous = CreateExpr("ambiguous");
  const Expr missing = CreateExpr("missing");
  const Expr shared = CreateExpr("shared");
  const Expr shared_search0 = CreateExpr("shared_search0");
  const Expr shared_search1 = CreateExpr("shared_search1");
  ModelInfo model_info;
  auto valid_axis = MakeAxis("valid", AxisPosition::INNER, false, false, false, MakeSymVar(valid));
  auto ambiguous_axis0 = MakeAxis("ambiguous0", AxisPosition::INNER, false, false, false, MakeSymVar(ambiguous));
  auto ambiguous_axis1 = MakeAxis("ambiguous1", AxisPosition::INNER, false, false, false, MakeSymVar(ambiguous));
  auto shared_axis = MakeAxis("shared", AxisPosition::INNER, false, false, false, MakeSymVar(shared));
  valid_axis->order = 3U;
  ambiguous_axis0->order = 1U;
  ambiguous_axis1->order = 1U;
  shared_axis->order = 4U;
  model_info.arg_list = {valid_axis, ambiguous_axis0, ambiguous_axis1, shared_axis};

  ArgsManager args_manager(model_info);
  VarInfo searchable_info;
  searchable_info.do_search = true;
  VarInfo shared_info = searchable_info;
  shared_info.replacement.orig_expr = shared;
  args_manager.vars_infos_ = {{valid, searchable_info},
                              {ambiguous, searchable_info},
                              {missing, searchable_info},
                              {shared_search0, shared_info},
                              {shared_search1, shared_info}};

  const ExprUintMap axes_order = args_manager.GetAxesOrder();
  ASSERT_EQ(axes_order.size(), 5U);
  EXPECT_EQ(axes_order.at(valid), 3U);
  const std::set<uint32_t> fallback_orders = {axes_order.at(ambiguous), axes_order.at(missing),
                                              axes_order.at(shared_search0), axes_order.at(shared_search1)};
  EXPECT_EQ(fallback_orders.size(), 4U);
  EXPECT_GE(*fallback_orders.begin(), 5U);
}

TEST_F(ArgsManagerUtest, test_get_replaced_vars) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.replacement.orig_expr = CreateExpr("n");
  args_manager.vars_infos_.emplace(CreateExpr("new_m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("new_n"), var_info2);
  auto exprs_map = args_manager.GetVarsRelations();
  EXPECT_EQ(exprs_map.size(), 1u);
  for (auto expr_pair : exprs_map) {
    EXPECT_EQ(Str(expr_pair.first), "new_n");
    EXPECT_EQ(Str(expr_pair.second), "n");
  }
}

TEST_F(ArgsManagerUtest, test_get_replaced_exprs) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.replacement.new_replaced_expr = af::sym::Pow(CreateExpr(2), CreateExpr("new_n"));
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs_map = args_manager.GetExprRelations();
  EXPECT_EQ(exprs_map.size(), 1u);
  for (auto expr_pair : exprs_map) {
    EXPECT_EQ(Str(expr_pair.first), "n");
    EXPECT_TRUE(expr_pair.second == af::sym::Pow(CreateExpr(2), CreateExpr("new_n")));
  }
}

TEST_F(ArgsManagerUtest, test_get_input_vars) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.is_input_var = true;
  VarInfo var_info2;
  var_info2.is_input_var = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetInputVars();
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_const_vars) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.is_const_var = true;
  VarInfo var_info2;
  var_info2.is_const_var = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetConstVars();
  EXPECT_EQ(exprs.size(), 1u);
  for (auto expr_pair : exprs) {
    EXPECT_EQ(Str(expr_pair.first), "m");
    EXPECT_EQ(expr_pair.second, 0);
  }
}

TEST_F(ArgsManagerUtest, test_get_related_scope) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.scopes = {HardwareDef::L0A};
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  auto scopes = args_manager.GetRelatedHardware(CreateExpr("m"));
  EXPECT_EQ(scopes.size(), 1u);
  EXPECT_TRUE(scopes[0] == HardwareDef::L0A);
  scopes = args_manager.GetRelatedHardware(CreateExpr("n"));
  EXPECT_TRUE(scopes.empty());
}

TEST_F(ArgsManagerUtest, test_split_args) {
  ModelInfo model_info;
  Expr s0 = CreateExpr("s0");
  Expr s1 = CreateExpr("s1");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(s0 * s1);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m;
  model_info.arg_list.emplace_back(z0);
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  auto input_args = args_manager.GetInputVars();
  std::set<std::string> target_search_args = {"s0", "s1"};
  for (auto input_arg : input_args) {
    EXPECT_TRUE(target_search_args.find(Str(input_arg)) != target_search_args.end());
  }
}

TEST_F(ArgsManagerUtest, test_total_scope) {
  ModelInfo info;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetTotalHardwareCons();
  EXPECT_TRUE(scopes.empty());
  args_manager.hardware_cons_[HardwareDef::L0A] = (CreateExpr("m") * CreateExpr(2));
  scopes = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(scopes.size(), 1u);
}

TEST_F(ArgsManagerUtest, test_total_scope_with_replace) {
  ModelInfo info;
  Expr m = CreateExpr("m");
  info.variable_expr_map[m] = CreateExpr("n") + af::sym::kSymbolOne;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetTotalHardwareCons(false);
  EXPECT_TRUE(scopes.empty());
  args_manager.hardware_cons_[HardwareDef::L0A] = (m * CreateExpr(2));
  scopes = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(scopes.size(), 1u);
}

TEST_F(ArgsManagerUtest, test_used_scope) {
  ModelInfo info;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  EXPECT_TRUE(!IsValid(scopes));
  args_manager.hardware_cons_[HardwareDef::L0A] = (CreateExpr("m") * CreateExpr(2));
  scopes = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  EXPECT_TRUE(IsValid(scopes));
}

TEST_F(ArgsManagerUtest, test_get_obj) {
  ModelInfo info;
  ArgsManager args_manager(info);
  Expr mac = (CreateExpr("m") / CreateExpr(16));
  Expr mte = (CreateExpr("n") / CreateExpr(256));
  args_manager.objs_.emplace(PipeType::AIC_MAC, mac);
  args_manager.objs_.emplace(PipeType::AIC_MTE2, mte);
  auto obj = args_manager.GetObjectFunc();
  EXPECT_TRUE(obj[PipeType::AIC_MAC] == mac);
  EXPECT_TRUE(obj[PipeType::AIC_MTE2] == mte);
  dumps(Str(obj[PipeType::AIC_MAC]));
  dumps(Str(obj[PipeType::AIC_MTE2]));
}

TEST_F(ArgsManagerUtest, test_get_ancestor) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.orig_axis_size = {CreateExpr("m")};
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  auto m_ori_name = Str(args_manager.GetAncestor(CreateExpr("m"))[0]);
  EXPECT_EQ(m_ori_name, "m");
  auto n_ori_size = args_manager.GetAncestor(CreateExpr("n"));
  EXPECT_TRUE(n_ori_size.empty());
}

TEST_F(ArgsManagerUtest, test_get_max_value) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.max_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto m_ori_name = args_manager.GetMaxValue(CreateExpr("tilem"));
  EXPECT_TRUE(m_ori_name == CreateExpr("m"));
  m_ori_name = args_manager.GetMaxValue(CreateExpr("m"));
  EXPECT_TRUE(m_ori_name == 1);
}

TEST_F(ArgsManagerUtest, test_get_min_value) {
  ModelInfo info;
  ArgsManager args_manager(info);
  auto test = args_manager.GetMaxValue(CreateExpr("test"));
  EXPECT_TRUE(test == 1);
}

TEST_F(ArgsManagerUtest, test_get_init_value) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.init_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto m_ori_name = args_manager.GetDefaultInitValue(CreateExpr("tilem"));
  EXPECT_TRUE(m_ori_name == CreateExpr("m"));
  m_ori_name = args_manager.GetDefaultInitValue(CreateExpr("m"));
  EXPECT_TRUE(m_ori_name == af::sym::kSymbolOne);
}

TEST_F(ArgsManagerUtest, test_get_align_value) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.align = af::Symbol(16u);
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto align_value = args_manager.GetVarAlignValue(CreateExpr("tilem"));
  EXPECT_EQ(align_value, af::Symbol(16u));
  align_value = args_manager.GetVarAlignValue(CreateExpr("m"));
  EXPECT_EQ(align_value, af::Symbol(0));
}

TEST_F(ArgsManagerUtest, test_set_solved_vars) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.align = af::Symbol(16u);
  info1.do_search = true;
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  args_manager.vars_infos_.emplace(CreateExpr("basem"), info1);
  std::vector<Expr> solved_vars{CreateExpr("basem")};
  args_manager.SetSolvedVars(solved_vars);
  EXPECT_TRUE(args_manager.solved_vars_.size() == 1u);
  EXPECT_TRUE(args_manager.solved_vars_[0] == CreateExpr("basem"));
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("basem")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("tilem")].do_search);
  auto vars = args_manager.GetSolvedVars();
  EXPECT_TRUE(vars.size() == 1u);
  EXPECT_TRUE(vars[0] == CreateExpr("basem"));
}

TEST_F(ArgsManagerUtest, test_get_total_cut_cons) {
  ModelInfo info;
  ArgsManager args_manager(info);
  args_manager.cut_leq_cons_.push_back(CreateExpr("m"));
  auto leq_exprs = args_manager.GetTotalCutCons();
  EXPECT_TRUE(leq_exprs.size() == 1u);
  EXPECT_TRUE(leq_exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_parent_var) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  auto parent_m = args_manager.GetParentVars(CreateExpr("m"));
  EXPECT_TRUE(parent_m.empty());
  auto parent_n = args_manager.GetParentVars(CreateExpr("n"));
  EXPECT_TRUE(parent_n.empty());
  VarInfo info2;
  info2.from_axis_size.emplace_back(CreateExpr("k"));
  args_manager.vars_infos_.emplace(CreateExpr("k"), info2);
  auto parent_k = args_manager.GetParentVars(CreateExpr("k"));
  EXPECT_FALSE(parent_k.empty());
}

TEST_F(ArgsManagerUtest, test_get_innerest_var) {
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.is_node_innerest_dim_size = true;
  VarInfo info2;
  info2.is_node_innerest_dim_size = true;
  info2.do_search = true;
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  args_manager.vars_infos_.emplace(CreateExpr("k"), info2);
  auto innerest_vars = args_manager.GetNodeInnerestDimSizes();
  EXPECT_EQ(innerest_vars.size(), 1u);
  EXPECT_TRUE(innerest_vars[0] == CreateExpr("k"));
}

TEST_F(ArgsManagerUtest, test_replace_new_expr) {
  ExprExprMap new_expr_replacements;
  Expr new_m = CreateExpr("new_m");
  Expr m = CreateExpr("m");
  Expr stepm = CreateExpr("stepm");
  Expr new_stepm = CreateExpr("new_stepm");
  Expr new_m_expr = m / CreateExpr(16);
  Expr new_stepm_expr = af::sym::Log(CreateExpr(2), (stepm / new_m));
  new_expr_replacements.emplace(new_m, new_m_expr);
  new_expr_replacements.emplace(new_stepm, new_stepm_expr);
  ArgsManager::ReplaceNewExpr(new_expr_replacements);
  EXPECT_FALSE(new_expr_replacements[new_stepm].ContainVar(new_m));
  for (auto expr : new_expr_replacements) {
    dumps(Str(expr.first) + "---------" + Str(expr.second));
  }
}

TEST_F(ArgsManagerUtest, vars_not_find) {
  ModelInfo info;
  ArgsManager args_manager(info);
  auto test = CreateExpr("test");
  EXPECT_TRUE(args_manager.GetAncestor(test).empty());
  EXPECT_TRUE(args_manager.GetMaxValue(test) == 1);
  EXPECT_TRUE(args_manager.GetMinValue(test) == 1);
  EXPECT_TRUE(args_manager.GetDefaultInitValue(test) == 1);
  EXPECT_TRUE(args_manager.GetVarAlignValue(test) == 0u);
  EXPECT_TRUE(args_manager.GetVarPromptAlignValue(test) == 0u);
}

TEST_F(ArgsManagerUtest, IsConcatOuterDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");
  ModelInfo info;
  ArgsManager manager(info);

  VarInfo info1;
  info1.is_concat_outer_dim = true;
  manager.vars_infos_[var1] = info1;

  VarInfo info2;
  info2.is_concat_outer_dim = false;
  manager.vars_infos_[var2] = info2;
  EXPECT_TRUE(manager.IsConcatOuterDim(var1));
  EXPECT_FALSE(manager.IsConcatOuterDim(var2));
  EXPECT_FALSE(manager.IsConcatOuterDim(undefined_var));
}

TEST_F(ArgsManagerUtest, IsConcatInnerDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");

  ModelInfo info;
  ArgsManager manager(info);
  VarInfo info1;
  info1.is_concat_inner_dim = true;
  manager.vars_infos_[var1] = info1;

  VarInfo info2;
  info2.is_concat_inner_dim = false;
  manager.vars_infos_[var2] = info2;
  EXPECT_TRUE(manager.IsConcatInnerDim(var1));
  EXPECT_FALSE(manager.IsConcatInnerDim(var2));
  EXPECT_FALSE(manager.IsConcatInnerDim(undefined_var));
}

// 辅助函数：构建small_shape_pattern测试的通用ModelInfo（常量z0 + 变量z0t）
ModelInfo BuildSmallShapePatternModelInfo(std::map<HardwareDef, Expr> &hardware_cons) {
  ModelInfo model_info;
  Expr s0 = CreateExpr(128);
  Expr s1 = CreateExpr("s1_size");
  SymConstInfoPtr sym_m = std::make_shared<SymConstInfo>(s0);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m;
  z0->name = "z0";

  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(s1);
  AttAxisPtr z0t = std::make_shared<AttAxis>();
  z0t->axis_pos = AxisPosition::INNER;
  z0t->size = sym_n;
  z0t->name = "z0t";
  z0t->orig_axis.emplace_back(z0.get());

  model_info.arg_list.emplace_back(z0);
  model_info.arg_list.emplace_back(z0t);
  model_info.hardware_cons = hardware_cons;
  return model_info;
}

// 辅助函数：构建AxesReorder impl并测试HitSmallShapePattern
bool TestHitSmallShapePattern(ModelInfo &model_info) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos = {model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(
      af::MakeShared<AxesReorderTilingCodeGenImpl>("op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  return impl->HitSmallShapePattern(args_manager);
}

TEST_F(ArgsManagerUtest, test_small_shape_pattern) {
  ModelInfo model_info;
  Expr s0 = CreateExpr("s0");
  Expr s1 = CreateExpr("s1");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(s0);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m;
  z0->name = "z0";

  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(s1);
  AttAxisPtr z1 = std::make_shared<AttAxis>();
  z1->axis_pos = AxisPosition::ORIGIN;
  z1->size = sym_n;
  z1->name = "z1";

  model_info.arg_list.emplace_back(z0);
  model_info.arg_list.emplace_back(z1);
  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::UB] = CreateExpr(1024) * s1;
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * s0;
  model_info.hardware_cons = hardware_cons;
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(
      af::MakeShared<AxesReorderTilingCodeGenImpl>("op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_FALSE(impl->HitSmallShapePattern(args_manager));
}

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case2) {
  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::UB] = CreateExpr(1024) * CreateExpr("s1_size");
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * CreateExpr(128);
  auto model_info = BuildSmallShapePatternModelInfo(hardware_cons);
  EXPECT_TRUE(TestHitSmallShapePattern(model_info));
}

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case3) {
  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::L1] = CreateExpr(1024) * CreateExpr("s1_size");
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * CreateExpr(128);
  auto model_info = BuildSmallShapePatternModelInfo(hardware_cons);
  EXPECT_FALSE(TestHitSmallShapePattern(model_info));
}

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case4) {
  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::UB] = CreateExpr(1024);
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024);
  auto model_info = BuildSmallShapePatternModelInfo(hardware_cons);
  EXPECT_FALSE(TestHitSmallShapePattern(model_info));
}

TEST_F(ArgsManagerUtest, ProductionSolverClassUsesSameLocalRefinementConfiguration) {
  for (const bool equal_order : {false, true}) {
    for (const size_t groups : {1UL, 2UL}) {
      for (const bool dynamic : {false, true}) {
        ModelInfo model;
        auto outer_m = std::make_shared<AttAxis>();
        outer_m->name = "m";
        outer_m->axis_pos = AxisPosition::ORIGIN;
        outer_m->size = std::make_shared<SymConstInfo>(CreateExpr(64));
        auto outer_n = std::make_shared<AttAxis>();
        outer_n->name = "n";
        outer_n->axis_pos = AxisPosition::ORIGIN;
        outer_n->size = std::make_shared<SymConstInfo>(CreateExpr(128));
        if (dynamic) {
          outer_n->size = MakeSymVar(CreateExpr("n_size"));
        }
        auto tile_m = MakeSymVar(CreateExpr("mt_size"));
        auto tile_n = MakeSymVar(CreateExpr("nt_size"));
        SetSymVarScope(tile_m, {HardwareDef::UB});
        SetSymVarScope(tile_n, {HardwareDef::UB});
        auto inner_m = MakeAxis("mt", AxisPosition::INNER, false, true, true, tile_m, {outer_m.get()});
        auto inner_n = MakeAxis("nt", AxisPosition::INNER, false, true, true, tile_n, {outer_n.get()});
        inner_m->order = 1U;
        inner_n->order = equal_order ? 1U : 2U;
        model.arg_list = {outer_m, outer_n, inner_m, inner_n};
        model.hardware_cons[HardwareDef::UB] = tile_m->symbol_expr * tile_n->symbol_expr;
        model.hardware_cons[HardwareDef::CORENUM] = CreateExpr(1);
        model.objects[PipeType::AIV_VEC] = tile_m->symbol_expr + tile_n->symbol_expr;
        TilingCodeGenConfig config;
        ScoreFuncs score_funcs;
        AxesReorderTilingCodeGenImpl impl("op", config, {model}, score_funcs, false);
        const auto &ident = model.schedule_group_ident;
        impl.SetScheduleResultGroupNums({{{ident.asc_graph_id, ident.impl_graph_id}, groups}});
        ASSERT_EQ(impl.GenSolverTiling(model), af::SUCCESS);
        const auto class_code = impl.tiling_func_.GetOutputStr();
        const bool expected = equal_order && groups > 1UL && !dynamic;
        EXPECT_EQ(class_code.find("objective_keys_") != std::string::npos, expected);
        ASSERT_EQ(impl.GenDoTiling(model), af::SUCCESS);
        const auto all_code = impl.tiling_func_.GetOutputStr();
        EXPECT_EQ(all_code.find("input.enable_local_core_refinement = true;") != std::string::npos, expected);
      }
    }
  }
}
