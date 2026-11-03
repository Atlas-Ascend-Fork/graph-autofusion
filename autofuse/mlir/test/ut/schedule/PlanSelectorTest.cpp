/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelector.h"

#include <gtest/gtest.h>

#include "test/ut/schedule/ScheduleTestUtils.h"

namespace {

TEST(PlanSelectorTest, KeepsEveryDynamicTemplate) {
  const auto result = mlir::test::MakeFusedResult(2U, 3U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(result, selections));
  ASSERT_EQ(selections.size(), 2U);
  for (const auto &selection : selections) {
    ASSERT_EQ(selection.template_ids_per_stage.size(), 1U);
    EXPECT_EQ(selection.template_ids_per_stage.front(), (std::vector<size_t>{0U, 1U, 2U}));
  }
}

TEST(PlanSelectorTest, RejectsDynamicEmptyStageAndMissingPlan) {
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(
      mlir::bridge::SelectDynamicCandidates(mlir::test::MakeFusedResult(1U, 1U, true), selections, &diagnostic));
  EXPECT_EQ(diagnostic, "subgraph 0 plan 0 contains an empty stage");

  diagnostic.clear();
  EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(mlir::test::MakeFusedResult(0U), selections, &diagnostic));
  EXPECT_EQ(diagnostic, "subgraph 0 contains no candidate plan");
}

// Regression: a subgraph with zero plans must fail the whole selection, not be
// silently skipped while other subgraphs still produce selections. Only
// checking selections.empty() after the loop would miss this -- subgraph 0
// contributing plans is enough to make that check pass even though subgraph 1
// contributed nothing, leaving the aggregate entry with no plan for subgraph 1.
TEST(PlanSelectorTest, RejectsSubgraphWithNoPlansAmongOthersWithPlans) {
  const auto result = mlir::test::MakeMultiSubgraphFusedResult({1U, 0U});
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(result, selections, &diagnostic));
  EXPECT_EQ(diagnostic, "subgraph 1 contains no candidate plan");
  EXPECT_TRUE(selections.empty());
}

// Regression guard: PlanSelection::subgraph_id defaults to 0U, so a loop body
// that forgets to assign it would silently mislabel every subgraph past the
// first as subgraph 0 instead of failing loudly.
TEST(PlanSelectorTest, KeepsEveryDynamicTemplateAcrossSubgraphs) {
  const auto result = mlir::test::MakeMultiSubgraphFusedResult({2U, 1U}, 3U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(result, selections));
  ASSERT_EQ(selections.size(), 3U);
  EXPECT_EQ(selections[0].subgraph_id, 0U);
  EXPECT_EQ(selections[0].plan_id, 0U);
  EXPECT_EQ(selections[1].subgraph_id, 0U);
  EXPECT_EQ(selections[1].plan_id, 1U);
  EXPECT_EQ(selections[2].subgraph_id, 1U);
  EXPECT_EQ(selections[2].plan_id, 0U);
  for (const auto &selection : selections) {
    ASSERT_EQ(selection.template_ids_per_stage.size(), 1U);
    EXPECT_EQ(selection.template_ids_per_stage.front(), (std::vector<size_t>{0U, 1U, 2U}));
  }
}

TEST(PlanSelectorTest, SelectsOnlyStaticCandidateWithoutSymbols) {
  std::vector<mlir::bridge::PlanSelection> selections;
  mlir::bridge::AttOptions options;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(mlir::test::MakeFusedResult(), {}, options, selections, &diagnostic));
  ASSERT_EQ(selections.size(), 1U);
  EXPECT_EQ(selections.front().subgraph_id, 0U);
  EXPECT_EQ(selections.front().plan_id, 0U);
  EXPECT_EQ(selections.front().template_ids_per_stage, (std::vector<std::vector<size_t>>{{0U}}));
}

TEST(PlanSelectorTest, RejectsInvalidStaticShapeSymbol) {
  std::vector<mlir::bridge::PlanSelection> selections;
  mlir::bridge::AttOptions options;
  std::string diagnostic;
  EXPECT_FALSE(
      mlir::bridge::SelectStaticWinner(mlir::test::MakeFusedResult(), {{"s-0", 32}}, options, selections, &diagnostic));
  EXPECT_EQ(diagnostic, "static plan selection received an invalid shape symbol");
}

TEST(PlanSelectorTest, SelectsSingleCandidateForEverySubgraphWithoutAtt) {
  const auto result = mlir::test::MakeMultiSubgraphFusedResult({1U, 1U});
  mlir::bridge::AttOptions options;
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(result, {{"s0", 16}}, options, selections, &diagnostic));
  ASSERT_EQ(selections.size(), 2U);
  EXPECT_EQ(selections[0].subgraph_id, 0U);
  EXPECT_EQ(selections[1].subgraph_id, 1U);
  EXPECT_TRUE(diagnostic.empty());
}

TEST(PlanSelectorTest, SelectsUniqueTemplatesAcrossMultipleStages) {
  auto result = mlir::test::MakeMultiStageFusedResult(1U, 3U);
  result.node_idx_to_scheduled_results.push_back({mlir::test::MakeMultiStageScheduledResult(2U)});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(result, {}, {}, selections));
  EXPECT_EQ(selections,
            (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}, {0U}, {0U}}}, {1U, 0U, {{0U}, {0U}}}}));
}

TEST(PlanSelectorTest, DynamicSelectionKeepsTemplatesInEveryStage) {
  const auto result = mlir::test::MakeMultiStageFusedResult(2U, 3U, 2U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(result, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U, 1U}, {0U, 1U}, {0U, 1U}}},
                                                                  {0U, 1U, {{0U, 1U}, {0U, 1U}, {0U, 1U}}}}));
}

TEST(PlanSelectorTest, BothStrategiesRejectMalformedSchedulesAndClearPreviousResults) {
  const std::vector<ascir::FusedScheduledResult> invalid_results{{},
                                                                 mlir::test::MakeMultiSubgraphFusedResult({1U, 0U}),
                                                                 mlir::test::MakeMultiStageFusedResult(1U, 0U),
                                                                 mlir::test::MakeFusedResult(1U, 0U)};
  for (const auto &result : invalid_results) {
    std::vector<mlir::bridge::PlanSelection> selections{{0U, 0U, {{0U}}}};
    std::string dynamic_diagnostic;
    EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(result, selections, &dynamic_diagnostic));
    EXPECT_TRUE(selections.empty());
    selections = {{0U, 0U, {{0U}}}};
    std::string static_diagnostic;
    EXPECT_FALSE(mlir::bridge::SelectStaticWinner(result, {}, {}, selections, &static_diagnostic));
    EXPECT_EQ(static_diagnostic, dynamic_diagnostic);
    EXPECT_FALSE(static_diagnostic.empty());
    EXPECT_TRUE(selections.empty());
  }
}

}  // namespace
