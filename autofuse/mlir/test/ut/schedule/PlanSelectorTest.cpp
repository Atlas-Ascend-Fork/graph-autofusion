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

// Reject the complete selection if any subgraph has no plans. A nonempty result
// from another subgraph does not prove coverage of every subgraph; accepting it
// would leave the aggregate entry without a plan for the empty subgraph.
TEST(PlanSelectorTest, RejectsSubgraphWithNoPlansAmongOthersWithPlans) {
  const auto result = mlir::test::MakeMultiSubgraphFusedResult({1U, 0U});
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(result, selections, &diagnostic));
  EXPECT_EQ(diagnostic, "subgraph 1 contains no candidate plan");
  EXPECT_TRUE(selections.empty());
}

// Check explicit subgraph IDs: the default 0U could otherwise label all later
// subgraphs as subgraph 0 when an assignment is omitted.
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

TEST(PlanSelectorTest, DynamicSelectionKeepsTemplatesInEveryStage) {
  const auto result = mlir::test::MakeMultiStageFusedResult(2U, 3U, 2U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(result, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U, 1U}, {0U, 1U}, {0U, 1U}}},
                                                                  {0U, 1U, {{0U, 1U}, {0U, 1U}, {0U, 1U}}}}));
}

TEST(PlanSelectorTest, SelectionAndPreparationRejectMalformedSchedulesAndClearPreviousResults) {
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
    std::string preparation_diagnostic;
    mlir::bridge::PlanGenerationResult prepared;
    EXPECT_FALSE(
        mlir::bridge::PreparePlanGeneration({result, {32, 196608}, true, {}}, prepared, &preparation_diagnostic));
    selections = prepared.selections;
    EXPECT_EQ(preparation_diagnostic, dynamic_diagnostic);
    EXPECT_FALSE(preparation_diagnostic.empty());
    EXPECT_TRUE(selections.empty());
  }
}

}  // namespace
