/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/AttCostEvaluator.h"

#include <gtest/gtest.h>

#include "test/ut/schedule/ScheduleTestUtils.h"

namespace {

att::ModelInfo MakeCostModel(int64_t cost) {
  att::ModelInfo model{};
  model.head_cost = att::CreateExpr(cost);
  return model;
}

att::ParsedScheduleResult MakePlanModel(std::initializer_list<std::initializer_list<int64_t>> stage_costs) {
  att::ParsedScheduleResult plan;
  size_t stage_id = 0U;
  for (const auto &costs : stage_costs) {
    for (int64_t cost : costs) {
      plan.groups_tiling_model_info[stage_id].push_back(MakeCostModel(cost));
    }
    ++stage_id;
  }
  return plan;
}

// Regression for the scoring bug: two pipes executing in parallel should be
// combined with max(), not sum(), and head_cost is additive on top of that
// max. A model whose pipes cost {10, 30} and whose head_cost is 5 must score
// 35 (30 + 5), not 45 (10 + 30 + 5) -- summing would have picked the wrong
// winner in ScoreModel whenever schedules trade off pipe count against
// per-pipe cost, which is exactly the case this component exists to choose
// between.
TEST(AttCostEvaluatorTest, ScoreModelTakesMaxOfPipesPlusHeadCost) {
  att::ModelInfo model{};
  model.objects[att::PipeType::AIC_MTE2] = att::CreateExpr(10);
  model.objects[att::PipeType::AIC_MAC] = att::CreateExpr(30);
  model.head_cost = att::CreateExpr(5);

  long double score = 0.0L;
  ASSERT_TRUE(mlir::bridge::detail::ScoreModel(model, {}, score));
  EXPECT_DOUBLE_EQ(static_cast<double>(score), 35.0);
}

TEST(AttCostEvaluatorTest, ScoreModelWithNoPipesIsJustHeadCost) {
  att::ModelInfo model{};
  model.head_cost = att::CreateExpr(7);

  long double score = 0.0L;
  ASSERT_TRUE(mlir::bridge::detail::ScoreModel(model, {}, score));
  EXPECT_DOUBLE_EQ(static_cast<double>(score), 7.0);
}

TEST(AttCostEvaluatorTest, ScoreModelSubstitutesShapeSymbolsBeforeComparing) {
  att::ModelInfo model{};
  model.objects[att::PipeType::AIC_MTE2] = af::Expression::Parse("s0");
  model.objects[att::PipeType::AIC_MAC] = att::CreateExpr(4);
  model.head_cost = att::CreateExpr(1);

  mlir::bridge::ConcreteShape shape{{"s0", 20}};
  long double score = 0.0L;
  ASSERT_TRUE(mlir::bridge::detail::ScoreModel(model, shape, score));
  EXPECT_DOUBLE_EQ(static_cast<double>(score), 21.0);  // max(20, 4) + 1
}

TEST(AttCostEvaluatorTest, ScoreModelParsesFractionalConstantCost) {
  att::ModelInfo model{};
  model.head_cost = af::Expression::Parse("0.25");

  long double score = 0.0L;
  ASSERT_TRUE(mlir::bridge::detail::ScoreModel(model, {}, score));
  EXPECT_DOUBLE_EQ(static_cast<double>(score), 0.25);
}

TEST(AttCostEvaluatorTest, SelectsLowestCostPlanAndTemplateForEveryStageAndSubgraph) {
  auto result = mlir::test::MakeMultiStageFusedResult(2U, 2U, 2U);
  result.node_idx_to_scheduled_results.push_back({mlir::test::MakeScheduledResult(2U)});
  att::FusedParsedScheduleResult models;
  models[0][0] = MakePlanModel({{1, 2}, {100, 101}});  // Total 101.
  models[0][1] = MakePlanModel({{5, 3}, {4, 6}});      // Total 7, despite a more expensive first stage.
  models[1][0] = MakePlanModel({{9, 2}});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, nullptr));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 1U, {{1U}, {0U}}}, {1U, 0U, {{1U}}}}));
}

TEST(AttCostEvaluatorTest, EqualScoresKeepFirstPlanAndFirstTemplate) {
  const auto result = mlir::test::MakeFusedResult(2U, 2U);
  att::FusedParsedScheduleResult models;
  models[0][1] = MakePlanModel({{3, 3}});
  models[0][0] = MakePlanModel({{3, 3}});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, nullptr));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
}

TEST(AttCostEvaluatorTest, ShapeSubstitutionChangesWinningTemplate) {
  const auto result = mlir::test::MakeFusedResult(1U, 2U);
  att::FusedParsedScheduleResult models;
  models[0][0] = MakePlanModel({{0, 10}});
  models[0][0].groups_tiling_model_info[0][0].head_cost = af::Expression::Parse("s0");
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {{"s0", 2}}, selections, nullptr));
  EXPECT_EQ(selections[0].template_ids_per_stage[0], (std::vector<size_t>{0U}));
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {{"s0", 20}}, selections, nullptr));
  EXPECT_EQ(selections[0].template_ids_per_stage[0], (std::vector<size_t>{1U}));
}

TEST(AttCostEvaluatorTest, SkipsPlansWithMissingModelsOrUnresolvedCosts) {
  const auto result = mlir::test::MakeFusedResult(3U);
  att::FusedParsedScheduleResult models;
  // Plan 0 is absent; plan 1 cannot be evaluated with the supplied shape.
  models[0][1] = MakePlanModel({{1}});
  models[0][1].groups_tiling_model_info[0][0].head_cost = af::Expression::Parse("s0");
  models[0][2] = MakePlanModel({{10}});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, nullptr));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 2U, {{0U}}}}));
}

TEST(AttCostEvaluatorTest, RejectsIncompleteStageAndTemplateModels) {
  const auto result = mlir::test::MakeMultiStageFusedResult(1U, 2U, 2U);
  std::vector<att::ParsedScheduleResult> invalid_models{MakePlanModel({{1, 2}}), MakePlanModel({{1, 2}, {3}}),
                                                        MakePlanModel({{1, 2}, {3, 4}, {5, 6}})};
  auto wrong_stage_id = MakePlanModel({{1, 2}, {3, 4}});
  wrong_stage_id.groups_tiling_model_info[2] = wrong_stage_id.groups_tiling_model_info[1];
  wrong_stage_id.groups_tiling_model_info.erase(1);
  invalid_models.push_back(wrong_stage_id);
  for (const auto &model : invalid_models) {
    att::FusedParsedScheduleResult models;
    models[0][0] = model;
    std::vector<mlir::bridge::PlanSelection> selections{{0U, 0U, {{0U}}}};
    std::string diagnostic = "previous error";
    EXPECT_FALSE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, &diagnostic));
    EXPECT_TRUE(selections.empty());
    EXPECT_EQ(diagnostic, "ATT did not provide a complete, constant-cost plan for subgraph 0");
  }
}

TEST(AttCostEvaluatorTest, MissingLaterSubgraphClearsAllWinners) {
  const auto result = mlir::test::MakeMultiSubgraphFusedResult({1U, 1U});
  att::FusedParsedScheduleResult models;
  models[0][0] = MakePlanModel({{1}});
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, &diagnostic));
  EXPECT_TRUE(selections.empty());
  EXPECT_EQ(diagnostic, "ATT returned no model information for subgraph 1");
}

TEST(AttCostEvaluatorTest, RejectsPlanWhenAnyTemplateCostIsUnresolved) {
  const auto result = mlir::test::MakeFusedResult(1U, 2U);
  att::FusedParsedScheduleResult models;
  models[0][0] = MakePlanModel({{1, 2}});
  models[0][0].groups_tiling_model_info[0][1].head_cost = af::Expression::Parse("s0");
  std::vector<mlir::bridge::PlanSelection> selections;
  EXPECT_FALSE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, nullptr));
  EXPECT_TRUE(selections.empty());
}

TEST(AttCostEvaluatorTest, SelectedCoordinatesCanBeNarrowedForCodegen) {
  const auto result = mlir::test::MakeMultiStageFusedResult(2U, 2U, 2U);
  att::FusedParsedScheduleResult models;
  models[0][0] = MakePlanModel({{20, 30}, {40, 50}});
  models[0][1] = MakePlanModel({{2, 1}, {1, 2}});
  std::vector<mlir::bridge::PlanSelection> selections, remapped;
  ASSERT_TRUE(mlir::bridge::detail::SelectWinnersFromModels(result, models, {}, selections, nullptr));
  ascir::FusedScheduledResult output;
  ASSERT_TRUE(mlir::bridge::NarrowFusedResultToSelections(result, selections, output, remapped));
  EXPECT_EQ(remapped, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}, {0U}}}}));
  EXPECT_EQ(output.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetName(),
            result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[1].GetName());
}

}  // namespace
