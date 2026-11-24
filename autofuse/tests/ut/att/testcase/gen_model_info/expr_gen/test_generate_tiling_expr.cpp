/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "test_fa_ascir_graph.h"
#include "parser/ascend_graph_parser.h"
#include "expr_gen/arg_list_manager.h"
#include "graph_construct_utils.h"
#define private public
#include "expr_gen/generate_tiling_expr.h"
#undef private
#include "generator/preprocess/args_manager.h"

namespace att {
static TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
class TestGenerateTilingExpr : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    af::AscGraph graph("graph");
    att::FaBeforeAutoFuse(graph);
    att::FaAfterScheduler(graph);
    att::FaAfterQueBufAlloc(graph);
    GraphConstructUtils::UpdateGraphVectorizedStride(graph);

    EXPECT_NE(tuning_space, nullptr);
    att::AscendGraphParser ascend_graph_parser(tuning_space);
    EXPECT_EQ(ascend_graph_parser.GraphParser(graph), af::SUCCESS);
    EXPECT_EQ(ArgListManager::GetInstance().LoadArgList(tuning_space), af::SUCCESS);
  }
  void TearDown() override {}
};

TEST_F(TestGenerateTilingExpr, case1) {
  std::map<HardwareDef, Expr> hardware_cons;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetCoreConstraint(hardware_cons), af::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case2) {
  std::vector<AttAxisPtr> arg_lists;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetSubAxisArgs(arg_lists), af::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case3) {
  std::map<std::string, std::vector<std::pair<Expr, Expr>>> eq_exprs;
  std::map<std::string, std::vector<Expr>> leq_exprs;
  GenerateTilingExpr generate_expr(tuning_space);
  generate_expr.GetAxisConstraints(eq_exprs, leq_exprs);
  EXPECT_TRUE(!eq_exprs.empty());
  EXPECT_TRUE(!leq_exprs.empty());
}

TEST_F(TestGenerateTilingExpr, case4) {
  std::map<HardwareDef, Expr> hardware_cons;
  Expr workspaceSize;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetCoreConstraint(hardware_cons), af::SUCCESS);
}

TEST_F(TestGenerateTilingExpr, case5) {
  std::map<std::string, Expr> tensor_exprs;
  GenerateTilingExpr generate_expr(tuning_space);
  EXPECT_EQ(generate_expr.GetTensorExpr(tensor_exprs), af::SUCCESS);
}

namespace {
void CheckTileSearchBounds(const std::vector<Expr> &extents, const Expr &alignment, bool padding, bool tail,
                           const Expr &expected_max) {
  auto space = std::make_shared<TuningSpace>();
  auto tile = std::make_unique<SubAxis>();
  tile->name = "cv_tile";
  tile->axis_type = AxisPosition::INNER;
  tile->repeat = CreateExpr("cv_tile");
  tile->align = alignment;
  tile->enable_pad = padding;
  tile->enable_tail = tail;
  tile->is_last = true;
  for (size_t i = 0; i < extents.size(); ++i) {
    auto origin = std::make_unique<SubAxis>();
    origin->name = "cv_origin_" + std::to_string(i);
    origin->axis_type = AxisPosition::ORIGIN;
    origin->repeat = extents[i];
    tile->orig_axis.push_back(origin.get());
    tile->parent_axis.push_back(origin.get());
    space->sub_axes.push_back(std::move(origin));
  }
  space->sub_axes.push_back(std::move(tile));
  ASSERT_EQ(ArgListManager::GetInstance().LoadArgList(space), af::SUCCESS);
  GenerateTilingExpr generator(space);
  ModelInfo model;
  ASSERT_EQ(generator.GetSubAxisArgs(model.arg_list), af::SUCCESS);
  ASSERT_EQ(generator.GetAxisConstraints(model.eq_exprs, model.leq_exprs), af::SUCCESS);
  ArgsManager args(model);
  ASSERT_TRUE(args.Process(false));
  const auto variable = CreateExpr("cv_tile");
  EXPECT_TRUE(args.GetMinValue(variable) == alignment);
  EXPECT_TRUE(args.GetMaxValue(variable) == expected_max);
  EXPECT_TRUE(args.GetDefaultInitValue(variable) == expected_max);
  for (size_t i = 0; i < extents.size(); ++i) {
    EXPECT_TRUE(model.arg_list[i]->size->symbol_expr == extents[i]);
  }
  if (padding && tail) {
    EXPECT_TRUE(model.leq_exprs.empty());
    EXPECT_TRUE(model.eq_exprs.empty());
  } else if (!tail) {
    EXPECT_FALSE(model.eq_exprs.empty());
  } else {
    EXPECT_FALSE(model.leq_exprs.empty());
  }
}
}  // namespace

TEST(GeneratePaddedAxisBoundsTest, ShortCvAxesKeepPhysicalAlignment) {
  for (int32_t extent : {1, 8, 16, 17, 31, 32, 33, 64, 128, 256}) {
    SCOPED_TRACE(extent);
    CheckTileSearchBounds({CreateExpr(extent)}, CreateExpr(32), true, true, CreateExpr(((extent + 31) / 32) * 32));
  }
}

TEST(GeneratePaddedAxisBoundsTest, NoPaddingAndNoTailKeepLogicalBounds) {
  CheckTileSearchBounds({CreateExpr(8)}, CreateExpr(32), false, true, CreateExpr(8));
  CheckTileSearchBounds({CreateExpr(16)}, CreateExpr(32), true, false, CreateExpr(16));
  CheckTileSearchBounds({CreateExpr(17)}, CreateExpr(1), false, true, CreateExpr(17));
}

TEST(GeneratePaddedAxisBoundsTest, MergedAndDynamicAxesUseOriginalExtentProduct) {
  CheckTileSearchBounds({CreateExpr(2), CreateExpr(8)}, CreateExpr(32), true, true, CreateExpr(32));
  const auto n = CreateExpr("cv_dynamic_n");
  CheckTileSearchBounds({n}, CreateExpr(32), true, true, af::sym::Align(n, 32U));
  const auto alignment = CreateExpr("cv_runtime_alignment");
  CheckTileSearchBounds({CreateExpr(8)}, alignment, true, true,
                        af::sym::Mul(af::sym::Ceiling(af::sym::Div(CreateExpr(8), alignment)), alignment));
}

TEST(GeneratePaddedAxisBoundsTest, ConstantTileRemainsConstant) {
  auto space = std::make_shared<TuningSpace>();
  auto tile = std::make_unique<SubAxis>();
  tile->name = "cv_constant_tile";
  tile->axis_type = AxisPosition::INNER;
  tile->repeat = CreateExpr(1);
  tile->align = CreateExpr(32);
  tile->enable_pad = true;
  tile->enable_tail = true;
  space->sub_axes.push_back(std::move(tile));
  ASSERT_EQ(ArgListManager::GetInstance().LoadArgList(space), af::SUCCESS);
  GenerateTilingExpr generator(space);
  ModelInfo model;
  ASSERT_EQ(generator.GetSubAxisArgs(model.arg_list), af::SUCCESS);
  ASSERT_EQ(model.arg_list.size(), 1U);
  const auto size = std::dynamic_pointer_cast<SymConstInfo>(model.arg_list[0]->size);
  ASSERT_NE(size, nullptr);
  EXPECT_EQ(size->const_value, 1U);
}

}  // namespace att
