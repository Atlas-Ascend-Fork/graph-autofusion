/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "common/ub_expr/ub_expr_types.h"
#define private public
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "generator/solver_pass/axes_reorder_solver_code.h"
#include "generator/solver_pass_gen/axes_reorder_solver/ub_named_expr_builder.h"
#include "gen_model_info/api_perf_register/v1/perf_param_v1.h"
using namespace att;

size_t CountSubstr(const std::string &text, const std::string &pattern) {
  size_t count = 0U;
  size_t pos = text.find(pattern);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(pattern, pos + pattern.size());
  }
  return count;
}

void ExpectNoTempVars(const std::string &actual) {
  EXPECT_EQ(actual.find("temp0"), std::string::npos);
  EXPECT_EQ(actual.find("temp1"), std::string::npos);
  EXPECT_EQ(actual.find("auto temp"), std::string::npos);
}

class TestAxesReorderSolverGen : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0, x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x0");
}

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE_SIZE_VAR_AS_INPUT) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::UB] = x0 + x1 + x2;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();

  auto vars = solver_gen.input_args_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x2");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0, x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_EQ(Str(vars[0]), "x1");

  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, GenGetPerfStaticDeclaresReduceTreeLeafDependencyBeforeUse) {
  Expr dim_r = CreateExpr("s22");
  Expr dim_a = CreateExpr("z3z4t_size");
  Expr dim_a_tail = CreateExpr("z3z4t_size_tail");
  Expr tree_cost = CreateExpr("reduce_ra_normal_unaligned_tree_cost_tail");

  TernaryOp tail_op(CondType::K_EQ, dim_a, CreateExpr(0), dim_a, dim_a + CreateExpr(1));
  tail_op.SetVariable(dim_a_tail);
  auto small_case = std::make_shared<IfCase>(dim_a_tail + dim_r);
  auto large_case = std::make_shared<IfCase>(dim_a_tail * CreateExpr(2));
  auto tree_case =
      std::make_shared<IfCase>(CondType::K_LT, dim_r, CreateExpr(2), std::move(small_case), std::move(large_case));
  // Mimic SelectSymbolicTreeCost: cached dependencies only contain dim_r, while leaf expressions use dim_a_tail.
  TernaryOp tree_op(tree_cost, std::move(tree_case), {dim_r});
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops{{dim_a_tail, tail_op}, {tree_cost, tree_op}};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetInputArgs({dim_r, dim_a});
  solver_gen.SetTernaryOps(ternary_ops);
  solver_gen.SetReplaceVars(ConcursiveReplaceVars(ternary_ops));
  solver_gen.SetObjFunc(CreateExpr(0), {{PipeType::AIV_VEC, tree_cost}});

  const std::string code = solver_gen.GenGetObjStaticFunc();
  const size_t tail_decl_pos = code.find("double z3z4t_size_tail =");
  const size_t tree_decl_pos = code.find("double reduce_ra_normal_unaligned_tree_cost_tail =");
  ASSERT_NE(tail_decl_pos, std::string::npos) << code;
  ASSERT_NE(tree_decl_pos, std::string::npos) << code;
  EXPECT_LT(tail_decl_pos, tree_decl_pos) << code;
}

TEST_F(TestAxesReorderSolverGen, GenRuntimeReorderRuleForDynamicReduceTile) {
  Expr reduce = CreateExpr("reduce");
  Expr tail = CreateExpr("tail");
  RuntimeReorderRule rule;
  rule.preferred_order = {tail, reduce};
  rule.condition_axis = tail;
  rule.compare_axis = reduce;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {reduce, tail};
  solver_gen.from_axes_map_[tail] = {tail};
  solver_gen.from_axes_map_[reduce] = {reduce};
  solver_gen.SetRuntimeReorderRules({rule});

  std::string code = solver_gen.GenRuntimeReorderRules();
  EXPECT_TRUE(code.find("(tail.upper_bound(tail.upper_bound_vars) < 64)") != std::string::npos);
  EXPECT_TRUE(code.find("(reduce.upper_bound(reduce.upper_bound_vars) > 128)") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_local_buffer_vars[0] = input.local_buffer_vars[1]") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_to_canonical[0] = 1u") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_local_buffer_vars[1] = input.local_buffer_vars[0]") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_to_canonical[1] = 0u") != std::string::npos);
  EXPECT_TRUE(code.find("input.local_buffer_vars[0] =") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenRuntimeReorderRuleUsesOriginalAxesForDynamicReduceTile) {
  Expr reduce = CreateExpr("reduce");
  Expr tail = CreateExpr("tail");
  Expr origin_reduce = CreateExpr("origin_reduce");
  Expr origin_tail = CreateExpr("origin_tail");
  RuntimeReorderRule rule;
  rule.preferred_order = {tail, reduce};
  rule.condition_axis = origin_tail;
  rule.compare_axis = origin_reduce;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.input_args_ = {origin_reduce, origin_tail};
  solver_gen.local_buffer_tiling_vars_ = {reduce, tail};
  solver_gen.from_axes_map_[tail] = {origin_tail};
  solver_gen.from_axes_map_[reduce] = {origin_reduce};
  solver_gen.SetRuntimeReorderRules({rule});

  std::string code = solver_gen.GenRuntimeReorderRules();
  EXPECT_TRUE(code.find("(origin_tail.value < 64)") != std::string::npos);
  EXPECT_TRUE(code.find("(origin_reduce.value > 128)") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_local_buffer_vars[0] = input.local_buffer_vars[1]") != std::string::npos);
  EXPECT_TRUE(code.find("input.ordered_local_buffer_vars[1] = input.local_buffer_vars[0]") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenRuntimeReorderRuleUsesOriginalAxisProductForDynamicReduceTile) {
  Expr reduce = CreateExpr("reduce");
  Expr tail = CreateExpr("tail");
  Expr origin_reduce0 = CreateExpr("origin_reduce0");
  Expr origin_reduce1 = CreateExpr("origin_reduce1");
  Expr origin_tail0 = CreateExpr("origin_tail0");
  Expr origin_tail1 = CreateExpr("origin_tail1");
  RuntimeReorderRule rule;
  rule.preferred_order = {tail, reduce};
  rule.condition_axis = origin_tail0 * origin_tail1;
  rule.compare_axis = origin_reduce0 * origin_reduce1;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.input_args_ = {origin_reduce0, origin_reduce1, origin_tail0, origin_tail1};
  solver_gen.local_buffer_tiling_vars_ = {reduce, tail};
  solver_gen.SetRuntimeReorderRules({rule});

  std::string code = solver_gen.GenRuntimeReorderRules();
  EXPECT_TRUE(code.find("origin_tail0.value") != std::string::npos);
  EXPECT_TRUE(code.find("origin_tail1.value") != std::string::npos);
  EXPECT_TRUE(code.find("origin_reduce0.value") != std::string::npos);
  EXPECT_TRUE(code.find("origin_reduce1.value") != std::string::npos);
  EXPECT_TRUE(code.find("origin_tail0*origin_tail1.value") == std::string::npos);
  EXPECT_TRUE(code.find("origin_reduce0*origin_reduce1.value") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImplAppliesRuntimeReorderOnceBeforeSolverRun) {
  Expr reduce = CreateExpr("reduce");
  Expr tail = CreateExpr("tail");
  RuntimeReorderRule rule;
  rule.preferred_order = {tail, reduce};
  rule.condition_axis = tail;
  rule.compare_axis = reduce;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {reduce, tail};
  solver_gen.hardware_use_map_[HardwareDef::UB] = reduce + tail;
  solver_gen.SetRuntimeReorderRules({rule});

  const std::string code = solver_gen.GenSolverFuncImpl();
  const std::string reorder_code = "input.ordered_local_buffer_vars[0] = input.local_buffer_vars[1]";
  const size_t reorder_pos = code.find(reorder_code);
  const size_t solver_pos = code.find("AxesReorderSolvercase_test solver(input);");
  EXPECT_EQ(CountSubstr(code, reorder_code), 1U);
  EXPECT_NE(reorder_pos, std::string::npos);
  EXPECT_NE(solver_pos, std::string::npos);
  EXPECT_LT(reorder_pos, solver_pos);
  EXPECT_NE(code.find("TilingVariable* ordered_local_buffer_vars[2] = {&reduce, &tail"), std::string::npos);
  EXPECT_NE(code.find("uint32_t ordered_to_canonical[2] = {0u, 1u"), std::string::npos);
  EXPECT_NE(code.find("tiling_data.set_reduce(input.local_buffer_vars[0]->value)"), std::string::npos);
  EXPECT_NE(code.find("tiling_data.set_tail(input.local_buffer_vars[1]->value)"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenRuntimeReorderUsesCompletePermutationAndRejectsInvalidOrder) {
  Expr axis0 = CreateExpr("axis0");
  Expr axis1 = CreateExpr("axis1");
  Expr axis2 = CreateExpr("axis2");
  RuntimeReorderRule rule;
  rule.preferred_order = {axis2, axis0, axis1};
  rule.condition_axis = axis0;
  rule.compare_axis = axis1;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {axis0, axis1, axis2};
  EXPECT_NE(
      solver_gen.GenRuntimeReorderRule(rule).find("input.ordered_local_buffer_vars[2] = input.local_buffer_vars[1]"),
      std::string::npos);

  Expr ignored_axis = CreateExpr("ignored_axis");
  rule.preferred_order = {axis2, ignored_axis, axis0, axis1};
  EXPECT_NE(
      solver_gen.GenRuntimeReorderRule(rule).find("input.ordered_local_buffer_vars[2] = input.local_buffer_vars[1]"),
      std::string::npos);

  rule.preferred_order = {axis2, axis2, axis1};
  EXPECT_TRUE(solver_gen.GenRuntimeReorderRule(rule).empty());
}

TEST_F(TestAxesReorderSolverGen, GenRuntimeReorderResolvesReplacedVarsOneToOne) {
  Expr origin_reduce = CreateExpr("origin_reduce");
  Expr origin_tail = CreateExpr("origin_tail");
  Expr search_reduce = CreateExpr("search_reduce");
  Expr search_tail = CreateExpr("search_tail");
  RuntimeReorderRule rule;
  rule.preferred_order = {origin_tail, origin_reduce};
  rule.condition_axis = origin_tail;
  rule.compare_axis = origin_reduce;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {search_reduce, search_tail};
  solver_gen.vars_relations_ = {{search_reduce, origin_reduce}, {search_tail, origin_tail}};
  const std::string code = solver_gen.GenRuntimeReorderRule(rule);
  EXPECT_NE(code.find("input.ordered_local_buffer_vars[0] = input.local_buffer_vars[1]"), std::string::npos);

  Expr duplicate_search = CreateExpr("duplicate_search");
  solver_gen.local_buffer_tiling_vars_ = {search_reduce, duplicate_search, search_tail};
  solver_gen.vars_relations_[duplicate_search] = origin_reduce;
  rule.preferred_order = {origin_reduce, origin_tail, duplicate_search};
  EXPECT_TRUE(solver_gen.GenRuntimeReorderRule(rule).empty());
}

TEST_F(TestAxesReorderSolverGen, GenPGOSolverAppliesRuntimeOrderBeforeSolverConstruction) {
  Expr reduce = CreateExpr("reduce");
  Expr tail = CreateExpr("tail");
  RuntimeReorderRule rule;
  rule.preferred_order = {tail, reduce};
  rule.condition_axis = tail;
  rule.compare_axis = reduce;
  rule.condition_threshold = 64U;
  rule.compare_threshold = 128U;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {reduce, tail};
  solver_gen.hardware_use_map_[HardwareDef::UB] = reduce + tail;
  solver_gen.SetRuntimeReorderRules({rule});

  const std::string code = solver_gen.GenPGOSolverFuncImpl();
  const size_t reorder_pos = code.find("input.ordered_local_buffer_vars[0] = input.local_buffer_vars[1]");
  const size_t solver_pos = code.find("PGOSolvercase_test solver(input);");
  ASSERT_NE(reorder_pos, std::string::npos);
  ASSERT_NE(solver_pos, std::string::npos);
  EXPECT_LT(reorder_pos, solver_pos);
}

TEST_F(TestAxesReorderSolverGen, GenLocalBufferSolversUseOrderedAxesWithCanonicalIndices) {
  const std::string naive_code = GenNaiveLocalBufTiling(true);
  EXPECT_NE(naive_code.find("input_.ordered_local_buffer_vars[ordered_idx]"), std::string::npos);
  EXPECT_NE(naive_code.find("input_.ordered_to_canonical[ordered_idx]"), std::string::npos);
  EXPECT_NE(naive_code.find("solved_axes[canonical_idx]"), std::string::npos);
  EXPECT_NE(naive_code.find("ProcessSingleAxisNaive(var, canonical_idx"), std::string::npos);

  const std::string binary_code = GenBinaryLocalBufTilingCore();
  EXPECT_NE(binary_code.find("input_.ordered_local_buffer_vars[ordered_idx]"), std::string::npos);
  EXPECT_NE(binary_code.find("input_.ordered_to_canonical[ordered_idx]"), std::string::npos);
  EXPECT_NE(binary_code.find("solved_axes[canonical_idx]"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenWorkloadBalanceUsesOrderedAxes) {
  const std::string code = GenWorkloadBalancePrepare();
  EXPECT_NE(code.find("auto *vars = input_.ordered_local_buffer_vars"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenEqualOrderAndAxisOrderRemainCanonical) {
  const std::string equal_order_code = GenNaiveLocalBufTiling(true);
  EXPECT_NE(equal_order_code.find("IdentifyEqualPriorityAxes"), std::string::npos);
  EXPECT_NE(equal_order_code.find("solved_axes[canonical_idx]"), std::string::npos);
  const std::string identify_code = GenIdentifyEqualPriorityAxes();
  EXPECT_NE(identify_code.find("if (pair.second.size() < kSupportMaxEqualPriorityAxes)"), std::string::npos);

  Expr axis0 = CreateExpr("axis0");
  Expr axis1 = CreateExpr("axis1");
  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.local_buffer_tiling_vars_ = {axis0, axis1};
  solver_gen.axes_order_ = {{axis0, 3U}, {axis1, 3U}};
  const std::string init_code = solver_gen.InitiateArgs();
  EXPECT_NE(init_code.find("axis0.order = 3UL"), std::string::npos);
  EXPECT_NE(init_code.find("axis1.order = 3UL"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbBinarySearchCommitsResultBeforeReturn) {
  const std::string code = GenBinarySearchWithAlignment();
  const size_t commit_pos = code.find("vars[i]->SetValue(result);");
  const size_t return_pos = code.find("return result != -1;");
  ASSERT_NE(commit_pos, std::string::npos) << code;
  ASSERT_NE(return_pos, std::string::npos) << code;
  EXPECT_LT(commit_pos, return_pos);
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbMainSearchRestoresBothAxesBeforeBSearch) {
  const std::string code = GenBinarySearchEqualPriorityAxesWithDualThreshold();
  const size_t failed_pos = code.find("if (result == -1) {");
  const size_t restore_a_pos = code.find("vars[0]->SetValue(result);");
  const size_t restore_b_pos = code.find("vars[1]->SetValue(result);");
  const size_t b_search_pos = code.find("\"B axis search\"");
  ASSERT_NE(failed_pos, std::string::npos) << code;
  ASSERT_NE(restore_a_pos, std::string::npos) << code;
  ASSERT_NE(restore_b_pos, std::string::npos) << code;
  ASSERT_NE(b_search_pos, std::string::npos) << code;
  EXPECT_LT(failed_pos, restore_a_pos);
  EXPECT_LT(restore_a_pos, restore_b_pos);
  EXPECT_LT(restore_b_pos, b_search_pos);
}

TEST_F(TestAxesReorderSolverGen, EqualPriorityUbLocalBufTilingGeneratesFinalChecks) {
  const std::string naive_equal_code = GenNaiveLocalBufTiling(true);
  EXPECT_NE(naive_equal_code.find("if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {\n"
                                  "    return enable_equal_order && NaiveLocalBufTiling(false);"),
            std::string::npos);

  const std::string naive_normal_code = GenNaiveLocalBufTiling(false);
  EXPECT_NE(naive_normal_code.find("return SatisfyCons(ConstraintType::LOCAL_BUFFER) && "
                                   "SatisfyCons(ConstraintType::LB_MIXED);"),
            std::string::npos);

  const std::string binary_equal_code = GenBinaryLocalBufTiling(true);
  EXPECT_NE(binary_equal_code.find("return enable_equal_order && BinaryLocalBufTiling(false);"), std::string::npos);

  const std::string binary_normal_code = GenBinaryLocalBufTiling(false);
  EXPECT_NE(binary_normal_code.find("if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {\n"
                                    "    return false;\n"
                                    "  }"),
            std::string::npos);
  EXPECT_EQ(binary_normal_code.find("enable_equal_order"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenTransposeEqualOrderDoesNotGenerateReduceTailBalance) {
  const std::string solver_head = GetAxesSolverSolverHead(true);
  const std::string solver_func = GetAxesSolverSolverFunc(true);
  EXPECT_EQ(solver_head.find("BalanceReduceTailAxes"), std::string::npos);
  EXPECT_EQ(solver_head.find("enable_reduce_tail_balance"), std::string::npos);
  EXPECT_EQ(solver_func.find("BalanceReduceTailAxes"), std::string::npos);
  EXPECT_NE(solver_func.find("IdentifyEqualPriorityAxes"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenPGOEnumeratesOrderedAxesAndStoresCanonicalCandidates) {
  const std::string code = GenPgoSolverGenerateAllTilingData();
  EXPECT_NE(code.find("tilingDataVar = input_.ordered_local_buffer_vars[index]"), std::string::npos);
  EXPECT_NE(code.find("input_.ordered_local_buffer_vars[tmp]"), std::string::npos);
  EXPECT_NE(code.find("ans_item[input_.ordered_to_canonical[index]] = tilingDataVar->value"), std::string::npos);
  EXPECT_NE(code.find("ans_item[index] = tilingDataVar->value"), std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_case2) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args;
  input_args.emplace_back(x1);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_ub_cache_code) {
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args;
  input_args.emplace_back(x1);
  search_args.emplace_back(x2);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = af::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x2;
  hardware_cons[HardwareDef::CORENUM] = x0 + x1;
  from_axes_map[x0] = {x1};
  from_axes_map[x2] = {x0};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;
  std::vector<CacheLineConfig> config;
  CacheLineConfig cache_line_cfg{"test_node", CreateExpr(120), 128};
  cache_line_cfg.solver_cache_line_expr = CreateExpr(120);
  config.push_back(cache_line_cfg);

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
  EXPECT_FALSE(impl_code.find("// check node ") != std::string::npos);
  auto vars = solver_gen.local_buffer_tiling_vars_;

  solver_gen.SetCacheLineConfig(&config);
  solver_gen.Arrange();
  impl_code = solver_gen.GenSolverClassImpl();
  invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
  EXPECT_TRUE(impl_code.find("// check node ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstExpr) {
  Expr var = CreateExpr("var");
  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3)};

  std::string result = solver_gen.GenUpperBoundFunc(var);

  std::string expected = R"(    GetUpperBoundFuncPtr var_upper_bound = [](Variable **parent_vars) {
      (void)parent_vars;
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      return upper_bound;
    };
    var.upper_bound = var_upper_bound;
)";
  EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundInfo_ConstExpr) {
  Expr var = CreateExpr("var");
  AxesReorderSolverGen solver("case_test", "TilingData");
  solver.from_axes_map_[var] = {CreateExpr(5)};  // Constant expression
  EXPECT_EQ(solver.GenUpperBoundInfo(var), "");
}

TEST_F(TestAxesReorderSolverGen, InitiateArgs_MixedValidInvalidArgs) {
  // Setup test data with mixed valid and invalid args
  AxesReorderSolverGen solver_("case_test", "TilingData");
  Expr valid = CreateExpr("valid");
  Expr invalid = CreateExpr("invalid");
  solver_.input_args_ = {valid, invalid};
  solver_.input_align_ = {{valid, af::Symbol(4)}, {invalid, af::Symbol(1)}};  // One needs alignment
  solver_.mc_args_ = {};
  solver_.local_buffer_tiling_vars_ = {};

  // Expected output
  std::string expected =
      "    Variable valid;\n"
      "    valid.value = (tiling_data.get_valid() + std::max(1, 4) - 1) / std::max(1, 4) * std::max(1, 4);\n"
      "    Variable invalid;\n"
      "    invalid.value = tiling_data.get_invalid();\n";

  std::string result = solver_.InitiateArgs();
  EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, NeitherInPriorityMap) {
  // Neither expression in priority map
  // Should use string comparison
  Expr expr1 = CreateExpr("var1");
  Expr expr2 = CreateExpr("var2");
  Expr expr3 = CreateExpr("var3");
  Expr expr4 = CreateExpr("var4");
  AxesReorderSolverGen solver("case_test", "TilingData");
  solver.VarCmp(expr1, expr2);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstantAxes) {
  // Test with constant axes
  Expr var = CreateExpr("test_var");
  AxesReorderSolverGen solver("case_test", "TilingData");
  solver.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3), CreateExpr(5)};

  std::string expected = R"(    GetUpperBoundFuncPtr test_var_upper_bound = [](Variable **parent_vars) {
      (void)parent_vars;
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      upper_bound *= 5;
      return upper_bound;
    };
    test_var.upper_bound = test_var_upper_bound;
)";
  EXPECT_EQ(solver.GenUpperBoundFunc(var), expected);
}

TEST_F(TestAxesReorderSolverGen, HandlesBothVars) {
  std::vector<Expr> tiling_vars = {CreateExpr("t_var1"), CreateExpr("t_var2")};
  std::vector<Expr> cons_vars = {CreateExpr("c_var1"), CreateExpr("c_var2")};
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr cons = CreateExpr("c_var1 * c_var2");
  std::string result = solver.GenConsFunc(0, ConsType::BUFFER, cons, tiling_vars, cons_vars);
  EXPECT_NE(result, "");
}

TEST_F(TestAxesReorderSolverGen, GenInputInfoTest) {
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;

  AxesReorderSolverGen solver("case_test", "TilingData");
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::vector<Expr> total_cut_cons_;
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  ExprExprMap arg_align_map_;
  ExprUintMap arg_prompt_align_map_;
  ExprUintMap is_concat_outer_map_;
  std::vector<Expr> concat_inner_dims_;

  // Initialize test data
  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
  hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
  hardware_use_map_[HardwareDef::UB] = CreateExpr(512);

  total_cut_cons_.push_back(CreateExpr(256));
  total_cut_cons_.push_back(CreateExpr(128));

  mc_args_.push_back(CreateExpr("mc_var1"));
  mc_args_.push_back(CreateExpr("mc_var2"));

  Expr var1 = CreateExpr("local_var1");
  Expr var2 = CreateExpr("local_var2");
  Expr var3 = af::Symbol(255, "local_var3");

  local_buffer_tiling_vars_.push_back(var1);
  local_buffer_tiling_vars_.push_back(var2);

  arg_align_map_[var1] = af::Symbol(64);
  arg_align_map_[var2] = af::Symbol(32);

  arg_prompt_align_map_[var1] = 128;
  arg_prompt_align_map_[var3] = 64;

  is_concat_outer_map_[var1] = 1;
  is_concat_outer_map_[var2] = 0;

  concat_inner_dims_.push_back(var3);

  // Set the test data to the solver
  solver.hardware_use_map_ = hardware_use_map_;
  solver.total_cut_cons_ = total_cut_cons_;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.arg_align_map_ = arg_align_map_;
  solver.arg_prompt_align_map_ = arg_prompt_align_map_;
  solver.is_concat_outer_map_ = is_concat_outer_map_;
  solver.concat_inner_dims_ = concat_inner_dims_;

  std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

  // 4. Check local buffer variables processing
  // Verify alignment strings are generated
  EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
  EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);

  std::cout << result << std::endl;
  // Verify prompt alignment for concat outer case
  EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);

  // Verify no prompt alignment for non-concat outer case
  EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenInputInfoCase2Test) {
  std::vector<Expr> all_cons;
  std::vector<Expr> local_buffer_cons;
  std::vector<Expr> mc_mixed_cons;

  AxesReorderSolverGen solver("case_test", "TilingData");
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::vector<Expr> total_cut_cons_;
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  ExprExprMap arg_align_map_;
  ExprUintMap arg_prompt_align_map_;
  ExprUintMap is_concat_outer_map_;
  std::vector<Expr> concat_inner_dims_;

  // Initialize test data
  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
  hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
  hardware_use_map_[HardwareDef::UB] = CreateExpr(512);

  total_cut_cons_.push_back(CreateExpr(256));
  total_cut_cons_.push_back(CreateExpr(128));

  mc_args_.push_back(CreateExpr("mc_var1"));
  mc_args_.push_back(CreateExpr("mc_var2"));

  Expr var1 = CreateExpr("local_var1");
  Expr var2 = CreateExpr("local_var2");
  Expr var3 = af::Symbol("local_var3");

  local_buffer_tiling_vars_.push_back(var1);
  local_buffer_tiling_vars_.push_back(var2);

  arg_align_map_[var1] = af::Symbol(64);
  arg_align_map_[var2] = af::Symbol(32);

  arg_prompt_align_map_[var1] = 128;
  arg_prompt_align_map_[var3] = 64;

  is_concat_outer_map_[var1] = 1;
  is_concat_outer_map_[var2] = 0;

  concat_inner_dims_.push_back(var3);

  // Set the test data to the solver
  solver.hardware_use_map_ = hardware_use_map_;
  solver.total_cut_cons_ = total_cut_cons_;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.arg_align_map_ = arg_align_map_;
  solver.arg_prompt_align_map_ = arg_prompt_align_map_;
  solver.is_concat_outer_map_ = is_concat_outer_map_;
  solver.concat_inner_dims_ = concat_inner_dims_;

  std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

  // 4. Check local buffer variables processing
  // Verify alignment strings are generated
  EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
  EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);

  std::cout << result << std::endl;
  // Verify prompt alignment for concat outer case
  EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);

  // Verify no prompt alignment for non-concat outer case
  EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return (ub_size - 0) > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") !=
              std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariablesNotFound) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;

  solver.input_args_ = {var1, var2, var3, var4};
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return ub_size > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") ==
              std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.SetEnableMulticoreUBTradeoff(true);
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, test_not_contain_heavy_op) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  TilingScheduleConfigTableV1 tiling_schedule_config_table;
  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.tiling_schedule_config_table_ = &tiling_schedule_config_table;
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(false, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenPgo_SolverwithClassImpl) {
  std::string className = "AxesReorderSolverGen";
  AxesReorderSolverGen solver("GenPgo_test", "TilingData");
  solver.SetEnableAutofusePGO(true);
  const std::string code = solver.GenSolverClassImpl();
  EXPECT_NE(code.find("bool SatisfyUBSizeCacheLine(uint32_t) override {return false;};"), std::string::npos);
  solver.GenSolverFuncImpl();
}

/**
 * @brief 测试：当 enable_multicore_ub_tradeoff 为 true 时，GenInput 使用成员变量而非硬编码常量
 *
 * 原问题背景： GenInput() 函数中
 * 使用硬编码常量 kDefaultSolverUbThreshold(0.2) 和 kDefaultSolverCoreNumThreshold(0.4)
 * 而非成员变量 ub_threshold_ 和 corenum_threshold_。
 *
 * 本测试用例用于看护该修复，防止未来代码回退。
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseMemberVariableWhenTradeOffEnabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  // 设置非默认阈值 (0.0 和 1.0 是边界值，常用于环境变量配置)
  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(true);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = false;

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 验证生成的代码使用成员变量值
  // 注意: std::to_string() 默认生成6位小数格式
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.000000;") != std::string::npos)
      << "GenInput should use ub_threshold_ member variable (0.0), not hardcoded default.\n"
      << "Generated code:\n"
      << gen_code;

  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 1.000000;") != std::string::npos)
      << "GenInput should use corenum_threshold_ member variable (1.0), not hardcoded default.\n"
      << "Generated code:\n"
      << gen_code;

  // 防止回退检查：确保代码中不包含硬编码的默认值 0.200000 和 0.400000
  EXPECT_EQ(gen_code.find("0.200000"), std::string::npos)
      << "Generated code contains hardcoded default ub_threshold 0.2!\n"
      << "This indicates the fix from commit 3d7b57d5 may have regressed.\n"
      << "Generated code:\n"
      << gen_code;

  EXPECT_EQ(gen_code.find("0.400000"), std::string::npos)
      << "Generated code contains hardcoded default corenum_threshold 0.4!\n"
      << "This indicates the fix from commit 3d7b57d5 may have regressed.\n"
      << "Generated code:\n"
      << gen_code;
}

/**
 * @brief 测试：验证 enable_multicore_ub_tradeoff 为 false 时使用默认值
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseDefaultWhenTradeOffDisabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  // 即使设置了非默认值，关闭开关后应使用默认值
  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(false);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = false;

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 应该使用默认值 0.2 和 0.4 (std::to_string 生成6位小数)
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.200000;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 0.400000;") != std::string::npos);
}

/**
 * @brief 测试：验证 trade_off_config 优先级低于 enable_multicore_ub_tradeoff
 */
TEST_F(TestAxesReorderSolverGen, GenInput_TradeOffConfigPriority) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(true);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = true;
  trade_off_config.ub_ratio = CreateExpr(0.3);
  trade_off_config.core_num_ratio = CreateExpr(0.5);

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // enable_multicore_ub_tradeoff 优先级更高，应该使用成员变量值
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.000000;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 1.000000;") != std::string::npos);
}

/**
 * @brief 测试：验证 trade_off_config 在 enable_multicore_ub_tradeoff 为 false 时生效
 */
TEST_F(TestAxesReorderSolverGen, GenInput_UseTradeOffConfigWhenMemberDisabled) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  solver.SetUBThreshold(0.0);
  solver.SetCoreNumThreshold(1.0);
  solver.SetEnableMulticoreUBTradeoff(false);
  solver.SetTilingCaseIdent({{0, 0, 0}, 1, ""});

  std::vector<Expr> all_cons;
  TradeOffConfig trade_off_config;
  trade_off_config.is_enable = true;
  trade_off_config.ub_ratio = CreateExpr(0.3);
  trade_off_config.core_num_ratio = CreateExpr(0.5);

  std::string gen_code = solver.GenInput(trade_off_config, all_cons);

  // 应该使用 trade_off_config 中的值
  EXPECT_TRUE(gen_code.find("input.ub_threshold = 0.3;") != std::string::npos);
  EXPECT_TRUE(gen_code.find("input.corenum_threshold = 0.5;") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_DisableGroupParallel_EnableTradeOff) {
  // 验证当 enable_group_parallel_ 为 false 时，
  // 如果 trade_off_config.is_enable 为 true，multicore_ub_tradeoff 应该是 "true"
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);
  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);
  from_axes_map[var3] = {var1};

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  TilingScheduleConfig tiling_schedule_config;
  tiling_schedule_config.trade_off_config.is_enable = true;
  tiling_schedule_config.trade_off_config.ub_ratio = af::Symbol(0.1);
  tiling_schedule_config.trade_off_config.core_num_ratio = af::Symbol(0.8);

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.from_axes_map_ = from_axes_map;
  solver.SetTilingScheduleConfig(tiling_schedule_config);
  solver.SetEnableParallel(false);  // enable_group_parallel_ = false

  std::string actual = solver.GenSolverFuncImpl();
  // 验证生成的代码中 multicore_ub_tradeoff 为 "true"
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenGetBlockDimStatic_ShouldClampToOneWhenCoreNumExprIsZero) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  Expr corenum_cons = CreateExpr(0);
  std::string actual = solver.GenGetBlockDimStatic(corenum_cons);

  EXPECT_TRUE(actual.find("return 0;") == std::string::npos) << actual;
  EXPECT_TRUE(actual.find("return std::max(1, static_cast<int32_t>(0));") != std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_UsesNamedExprForSemanticContainer) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor = CreateExpr("tensor_0");
  Expr s2 = CreateExpr("S2");
  Expr s1tt_size = CreateExpr("s1tt_size");
  Expr common_size = CreateExpr(4) * s2 * s1tt_size;
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = CreateExpr(32) * af::sym::Ceiling(tensor * af::sym::Rational(1, 32));
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, common_size}});
  solver.SetContainerNames({{tensor, "tensor_0"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("tensor_size_0"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("que_size_"), std::string::npos) << actual;
  EXPECT_EQ(CountSubstr(actual, "auto tensor_size_0 ="), 1U) << actual;
  const size_t tensor_decl = actual.find("auto tensor_size_0 =");
  ASSERT_NE(tensor_decl, std::string::npos) << actual;
  const size_t tensor_decl_end = actual.find('\n', tensor_decl);
  const std::string tensor_decl_line = actual.substr(tensor_decl, tensor_decl_end - tensor_decl);
  EXPECT_EQ(tensor_decl_line.find("common_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("ub_size = (32 * Ceiling((Rational(1,32) * tensor_size_0)))"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, BuildNamedUbExprUsesCommonOriginExpr) {
  Expr tensor = CreateExpr("tensor_size");
  Expr tensor_value = CreateExpr("s0") * CreateExpr("s1");
  ascir::UbExprContext context;
  context.ub_expr = af::sym::Mul(CreateExpr(32), af::sym::Ceiling(af::sym::Div(tensor, CreateExpr(32))));
  context.container_expr[tensor] = tensor_value;
  context.container_names[tensor] = "load0";

  const auto actual = BuildNamedUbExpr(context, "  ");

  EXPECT_NE(actual.first.find("auto tensor_size_0 = (s0 * s1);"), std::string::npos) << actual.first;
  EXPECT_NE(actual.second.find("tensor_size_0"), std::string::npos) << actual.second;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_NamesExpandedUbExpr) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr s2 = CreateExpr("S2");
  Expr s1tt_size = CreateExpr("s1tt_size");
  Expr tensor_size = CreateExpr(4) * s2 * s1tt_size;
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = CreateExpr(32) * af::sym::Ceiling(tensor_size * af::sym::Rational(1, 32));
  solver.SetBufferUseAlg(hardware_use_map);

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(actual.find("auto common_size_"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("auto que_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (32 * Ceiling(((Rational(1,8) * S2) * s1tt_size)));"), std::string::npos)
      << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_ReusesDuplicateSemanticExpr) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor0 = CreateExpr("tensor_0");
  Expr tensor1 = CreateExpr("tensor_1");
  Expr s2 = CreateExpr("S2");
  Expr s1tt_size = CreateExpr("s1tt_size");
  Expr common_size = CreateExpr(4) * s2 * s1tt_size;
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = af::sym::Max(tensor0, tensor1);
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor0, common_size}, {tensor1, common_size}});
  solver.SetContainerNames({{tensor0, "tensor_0"}, {tensor1, "tensor_1"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(CountSubstr(actual, "auto tensor_size_0 ="), 1U) << actual;
  EXPECT_EQ(actual.find("auto tmp_buffer_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("ub_size = Max(tensor_size_0,tensor_size_0);"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_NoUbKeepsReturnZero) {
  AxesReorderSolverGen solver("case_test", "TilingData");

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("return 0;"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenUbIndirectPathsReuseGetUbSizeStatic) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");
  Expr z0 = CreateExpr("z0");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::CORENUM] = z0 * z0t_size;
  hardware_use_map[HardwareDef::UB] =
      CreateExpr(32) * af::sym::Ceiling((z0t_size * z1t_size) * af::sym::Rational(1, 32));
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetInputArgs({z0});
  solver.SetSearchArgs({z0t_size, z1t_size});
  solver.SetFromAxesMap({{z0t_size, {z0}}});
  solver.Arrange();

  std::string tiling_data_func = solver.GenGetTilingDataUbSizeStaticFunc();
  std::string threshold_func = solver.GenUBThresholdFunc();
  std::string cons_func = solver.GenConsUbFunc(0, {z0t_size}, {z0});

  EXPECT_NE(tiling_data_func.find("return GetUbSizeStatic("), std::string::npos) << tiling_data_func;
  EXPECT_NE(threshold_func.find("GetUbSizeStatic("), std::string::npos) << threshold_func;
  EXPECT_NE(cons_func.find("GetUbSizeStatic("), std::string::npos) << cons_func;
  ExpectNoTempVars(tiling_data_func);
  ExpectNoTempVars(threshold_func);
  ExpectNoTempVars(cons_func);
}

TEST_F(TestAxesReorderSolverGen, GenConsFunc_BufferStillUsesOriginTempExpr) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tiling_var = CreateExpr("tiling_var");
  Expr cons_var = CreateExpr("cons_var");
  Expr cons = tiling_var * cons_var;

  std::string actual = solver.GenConsFunc(0, ConsType::BUFFER, cons, {tiling_var}, {cons_var});

  EXPECT_NE(actual.find("auto temp"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("tensor_size_"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("que_size_"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_LeafUbExprDoesNotCreateIntermediate) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr var = CreateExpr("var");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = var;
  solver.SetBufferUseAlg(hardware_use_map);

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("int64_t ub_size = var;"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("auto common_size_"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_AvoidsGeneratedNameCollision) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor_size_0 = CreateExpr("tensor_size_0");
  Expr tensor = CreateExpr("tensor_0");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = tensor + tensor_size_0;
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, tensor_size_0 * CreateExpr(4)}});
  solver.SetContainerNames({{tensor, "tensor_0"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(actual.find("auto tensor_size_0 ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto tensor_size_1 ="), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_QueryTensorNameUsesTensorKind) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor = CreateExpr("tensor_0");
  Expr axis = CreateExpr("axis_size");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = tensor + CreateExpr(1);
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, axis * CreateExpr(4)}});
  solver.SetContainerNames({{tensor, "query"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("auto tensor_size_0 ="), std::string::npos) << actual;
  EXPECT_EQ(actual.find("auto que_size_"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_KeepsFirstKindWhenSemanticExprReused) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor = CreateExpr("a_tensor_0");
  Expr buffer = CreateExpr("z_buffer_size");
  Expr axis = CreateExpr("axis_size");
  Expr shared = axis * CreateExpr(4);
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = tensor + buffer;
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, shared}, {buffer, shared}});
  solver.SetContainerNames({{tensor, "a_tensor_0"}, {buffer, "z_buffer_size"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(CountSubstr(actual, "auto tensor_size_0 ="), 1U) << actual;
  EXPECT_EQ(actual.find("auto tmp_buffer_size_0 ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (tensor_size_0 + tensor_size_0);"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_QueueAlignSupportsRightMultiply) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor = CreateExpr("tensor_0");
  Expr axis = CreateExpr("axis_size");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = af::sym::Ceiling(tensor) * CreateExpr(32);
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, axis * CreateExpr(4)}});
  solver.SetContainerNames({{tensor, "tensor_0"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(actual.find("auto que_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (32 * Ceiling(tensor_size_0));"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_KeepsShortCeilingInline) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor = CreateExpr("tensor_0");
  Expr axis = CreateExpr("axis_size");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = af::sym::Ceiling(tensor) + CreateExpr(1);
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor, axis * CreateExpr(4)}});
  solver.SetContainerNames({{tensor, "tensor_0"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(actual.find("auto que_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (1 + Ceiling(tensor_size_0));"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_MaterializesRepeatedSubExpr) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr tensor0 = CreateExpr("tensor_0");
  Expr tensor1 = CreateExpr("tensor_1");
  Expr axis_size = CreateExpr("axis_size");
  Expr repeated_size = axis_size * CreateExpr(4);
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = tensor0 + tensor1;
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{tensor0, repeated_size}, {tensor1, repeated_size}});
  solver.SetContainerNames({{tensor0, "tensor_0"}, {tensor1, "tensor_1"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_EQ(CountSubstr(actual, "auto tensor_size_0 = (4 * axis_size);"), 1U) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (tensor_size_0 + tensor_size_0);"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_MaterializesLongSubExpr) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr very_long_axis_size_0 = CreateExpr("very_long_axis_size_0");
  Expr very_long_axis_size_1 = CreateExpr("very_long_axis_size_1");
  Expr very_long_axis_size_2 = CreateExpr("very_long_axis_size_2");
  Expr very_long_axis_size_3 = CreateExpr("very_long_axis_size_3");
  Expr long_expr = very_long_axis_size_0 * very_long_axis_size_1 * very_long_axis_size_2 * very_long_axis_size_3;
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = long_expr + CreateExpr(1);
  solver.SetBufferUseAlg(hardware_use_map);

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("auto common_size_"), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (common_size_"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_UsesQueueSemanticName) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr q0 = CreateExpr("q0");
  Expr q1 = CreateExpr("q1");
  Expr axis0 = CreateExpr("axis0_size");
  Expr axis1 = CreateExpr("axis1_size");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = CreateExpr(32) * af::sym::Ceiling(q0 * af::sym::Rational(1, 32)) +
                                      CreateExpr(64) * af::sym::Ceiling(q1 * af::sym::Rational(1, 64));
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{q0, axis0 * CreateExpr(4)}, {q1, axis1 * CreateExpr(8)}});
  solver.SetContainerNames({{q0, "q0"}, {q1, "q1"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("auto q0_size ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto q1_size ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto q0_que_size ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto q1_que_size ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = (q0_que_size + q1_que_size);"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("auto que_size_"), std::string::npos) << actual;
}

TEST_F(TestAxesReorderSolverGen, GenGetUbSizeStaticFunc_UsesBufferSemanticName) {
  AxesReorderSolverGen solver("case_test", "TilingData");
  Expr buffer = CreateExpr("buffer_size_0");
  Expr tmp_buffer = CreateExpr("tmp_buffer");
  Expr axis0 = CreateExpr("axis0_size");
  Expr axis1 = CreateExpr("axis1_size");
  std::map<HardwareDef, Expr> hardware_use_map;
  hardware_use_map[HardwareDef::UB] = af::sym::Max(buffer, tmp_buffer);
  solver.SetBufferUseAlg(hardware_use_map);
  solver.SetContainerExpr({{buffer, axis0 * CreateExpr(4)}, {tmp_buffer, axis1 * CreateExpr(8)}});
  solver.SetContainerNames({{buffer, "buffer_size_0"}, {tmp_buffer, "tmp_buffer"}});

  std::string actual = solver.GenGetUbSizeStaticFunc();

  ExpectNoTempVars(actual);
  EXPECT_NE(actual.find("auto buffer_size_0 ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto tmp_buffer_size ="), std::string::npos) << actual;
  EXPECT_NE(actual.find("auto tmp_buffer_size_max = Max(buffer_size_0"), std::string::npos) << actual;
  EXPECT_NE(actual.find("int64_t ub_size = tmp_buffer_size_max;"), std::string::npos) << actual;
  EXPECT_EQ(actual.find("auto tmp_buffer_size_0 ="), std::string::npos) << actual;
}
