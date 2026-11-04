/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelection.h"

#include <gtest/gtest.h>
#include <set>

#include "base/model_info.h"
#include "common/tiling_schema/schedule_group_names.h"
#include "common/tiling_schema/tiling_field_schema.h"

#include "test/ut/schedule/ScheduleTestUtils.h"

namespace {

// Use ATT's naming implementation as the oracle, including the schema's emitted scopes.
void ExpectScheduleGroupNamesMatchAtt(const ascir::FusedScheduledResult &schedule) {
  mlir::common::TilingFieldSchema schema;
  std::string diagnostic;
  ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(schedule, schema, &diagnostic)) << diagnostic;
  std::set<std::string> expectedScopes;
  for (size_t subgraph = 0; subgraph < schedule.node_idx_to_scheduled_results.size(); ++subgraph) {
    const auto &plans = schedule.node_idx_to_scheduled_results[subgraph];
    for (size_t plan = 0; plan < plans.size(); ++plan) {
      for (size_t stage = 0; stage < plans[plan].schedule_groups.size(); ++stage) {
        const att::ScheduleGroupIdent ident{subgraph, plan, stage};
        const std::string field = ident.GetItemPrefix() + "_tiling_data";
        SCOPED_TRACE(field);
        EXPECT_EQ(mlir::common::MakeScheduleGroupTilingFieldName(subgraph, plan, stage), field);
        EXPECT_EQ(mlir::common::MakeScheduleGroupTilingTypeName(subgraph, plan, stage),
                  ident.GetGroupPrefix() + "TilingData");
        EXPECT_TRUE(expectedScopes.insert(field).second);
      }
    }
  }
  std::set<std::string> actualScopes;
  for (const auto &field : schema.fields) {
    if (field.name == "tiling_key") {
      EXPECT_TRUE(actualScopes.insert(field.path).second);
    }
  }
  EXPECT_EQ(actualScopes, expectedScopes);
}

TEST(PlanSelectionTest, ScheduleGroupNamesMatchAttAcrossSubgraphsPlansAndStages) {
  auto schedule = mlir::test::MakeMultiStageFusedResult(3U, 2U, 2U);
  schedule.node_idx_to_scheduled_results.push_back(
      {mlir::test::MakeMultiStageScheduledResult(3U, 2U), mlir::test::MakeMultiStageScheduledResult(2U, 1U)});
  ExpectScheduleGroupNamesMatchAtt(schedule);
}

TEST(PlanSelectionTest, ScheduleGroupNamesMatchAttAfterNarrowing) {
  auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  input.node_idx_to_scheduled_results.push_back(input.node_idx_to_scheduled_results.front());
  const std::vector<mlir::bridge::PlanSelection> selected{
      {0U, 2U, {{2U}, {1U}}}, {1U, 1U, {{1U}, {2U}}}, {0U, 0U, {{0U}, {2U, 0U}}}};
  ascir::FusedScheduledResult narrowed;
  std::vector<mlir::bridge::PlanSelection> remapped;
  ASSERT_TRUE(mlir::bridge::NarrowFusedResultToSelections(input, selected, narrowed, remapped));
  EXPECT_EQ(remapped, (std::vector<mlir::bridge::PlanSelection>{
                          {0U, 0U, {{0U}, {0U}}}, {1U, 0U, {{0U}, {0U}}}, {0U, 1U, {{0U}, {0U, 1U}}}}));
  ASSERT_EQ(narrowed.node_idx_to_scheduled_results.size(), 2U);
  ASSERT_EQ(narrowed.node_idx_to_scheduled_results[0].size(), 2U);
  ASSERT_EQ(narrowed.node_idx_to_scheduled_results[1].size(), 1U);
  ExpectScheduleGroupNamesMatchAtt(narrowed);
}

TEST(PlanSelectionTest, ValidatesSelectionCoordinates) {
  const auto result = mlir::test::MakeFusedResult();
  mlir::bridge::PlanSelection selection;
  std::string diagnostic;

  selection.template_ids_per_stage = {{0U}};
  EXPECT_TRUE(mlir::bridge::ValidatePlanSelection(result, selection, &diagnostic));

  selection.subgraph_id = 1U;
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, selection, &diagnostic));
  selection.subgraph_id = 0U;
  selection.plan_id = 1U;
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, selection, &diagnostic));
  selection.plan_id = 0U;
  selection.template_ids_per_stage.clear();
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, selection, &diagnostic));
  selection.template_ids_per_stage = {{1U}};
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, selection, &diagnostic));
}

TEST(PlanSelectionTest, RejectsDuplicateTemplatesAndPlansWithoutStages) {
  auto result = mlir::test::MakeFusedResult();
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, {0U, 0U, {{0U, 0U}}}, &diagnostic));
  EXPECT_EQ(diagnostic, "duplicate template_id for stage 0");
  result.node_idx_to_scheduled_results[0][0].schedule_groups.clear();
  EXPECT_FALSE(mlir::bridge::ValidatePlanSelection(result, {0U, 0U, {}}, &diagnostic));
  EXPECT_EQ(diagnostic, "subgraph 0 plan 0 contains no stage");
}

TEST(PlanSelectionTest, BuildsCandidatesInSubgraphPlanStageTemplateOrder) {
  auto result = mlir::test::MakeMultiStageFusedResult(2U, 2U, 3U);
  result.node_idx_to_scheduled_results.push_back({mlir::test::MakeScheduledResult(2U)});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(result, selections));
  EXPECT_EQ(selections,
            (std::vector<mlir::bridge::PlanSelection>{
                {0U, 0U, {{0U, 1U, 2U}, {0U, 1U, 2U}}}, {0U, 1U, {{0U, 1U, 2U}, {0U, 1U, 2U}}}, {1U, 0U, {{0U, 1U}}}}));
}

TEST(PlanSelectionTest, NarrowsAndRenumbersWhilePreservingSelectedGraphsAndMetadata) {
  auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  input.node_idx_to_scheduled_results.push_back({mlir::test::MakeScheduledResult(2U)});
  input.fused_graph_name = "selection_test";
  // Give every source graph a distinct identity so a wrong plan/template
  // lookup cannot pass merely because the test factories reuse graph names.
  for (size_t subgraph = 0U; subgraph < input.node_idx_to_scheduled_results.size(); ++subgraph) {
    auto &plans = input.node_idx_to_scheduled_results[subgraph];
    for (size_t plan = 0U; plan < plans.size(); ++plan) {
      for (size_t stage = 0U; stage < plans[plan].schedule_groups.size(); ++stage) {
        auto &graphs = plans[plan].schedule_groups[stage].impl_graphs;
        for (size_t graph = 0U; graph < graphs.size(); ++graph) {
          const std::string name = std::to_string(subgraph) + "_" + std::to_string(plan) + "_" + std::to_string(stage) +
                                   "_" + std::to_string(graph);
          graphs[graph] = af::AscGraph(name.c_str());
        }
      }
    }
  }
  const std::vector<mlir::bridge::PlanSelection> selections{
      {0U, 2U, {{2U, 0U}, {1U}}}, {1U, 0U, {{1U}}}, {0U, 0U, {{1U}, {2U}}}};
  ascir::FusedScheduledResult output;
  std::vector<mlir::bridge::PlanSelection> remapped;
  ASSERT_TRUE(mlir::bridge::NarrowFusedResultToSelections(input, selections, output, remapped));
  EXPECT_EQ(remapped, (std::vector<mlir::bridge::PlanSelection>{
                          {0U, 0U, {{0U, 1U}, {0U}}}, {1U, 0U, {{0U}}}, {0U, 1U, {{0U}, {0U}}}}));
  EXPECT_STREQ(output.fused_graph_name.GetString(), "selection_test");
  ASSERT_EQ(output.node_idx_to_scheduled_results[0].size(), 2U);
  ASSERT_EQ(output.node_idx_to_scheduled_results[1].size(), 1U);
  for (size_t i = 0U; i < selections.size(); ++i) {
    ASSERT_TRUE(mlir::bridge::ValidatePlanSelection(output, remapped[i]));
    const auto &old_plan = input.node_idx_to_scheduled_results[selections[i].subgraph_id][selections[i].plan_id];
    const auto &new_plan = output.node_idx_to_scheduled_results[remapped[i].subgraph_id][remapped[i].plan_id];
    for (size_t stage = 0U; stage < old_plan.schedule_groups.size(); ++stage) {
      for (size_t j = 0U; j < selections[i].template_ids_per_stage[stage].size(); ++j) {
        EXPECT_EQ(
            new_plan.schedule_groups[stage].impl_graphs[j].GetName(),
            old_plan.schedule_groups[stage].impl_graphs[selections[i].template_ids_per_stage[stage][j]].GetName());
      }
    }
  }
  EXPECT_EQ(input.node_idx_to_scheduled_results[0].size(), 3U);
  EXPECT_EQ(input.node_idx_to_scheduled_results[0][2].schedule_groups[0].impl_graphs.size(), 3U);
}

TEST(PlanSelectionTest, NarrowingRejectsInvalidSelectionsWithoutPublishingPartialResult) {
  const auto input = mlir::test::MakeMultiSubgraphFusedResult({2U, 1U}, 2U);
  const std::vector<std::vector<mlir::bridge::PlanSelection>> invalid_selections{
      {},
      {{0U, 0U, {{0U}}}},                                       // Missing subgraph.
      {{0U, 0U, {{0U}}}, {1U, 1U, {{0U}}}},                     // Invalid plan after a valid selection.
      {{0U, 0U, {{0U}}}, {2U, 0U, {{0U}}}},                     // Invalid subgraph.
      {{0U, 0U, {{0U}}}, {1U, 0U, {{2U}}}},                     // Invalid template.
      {{0U, 0U, {{0U}}}, {1U, 0U, {}}},                         // Missing stage.
      {{0U, 0U, {{0U}}}, {1U, 0U, {{}}}},                       // Empty selection for stage.
      {{0U, 0U, {{0U}}}, {1U, 0U, {{0U, 0U}}}},                 // Duplicate template.
      {{0U, 0U, {{0U}}}, {1U, 0U, {{0U}}}, {0U, 0U, {{1U}}}}};  // Duplicate plan.
  for (const auto &selections : invalid_selections) {
    SCOPED_TRACE(selections.size());
    auto output = mlir::test::MakeFusedResult(3U);
    output.fused_graph_name = "unchanged";
    std::vector<mlir::bridge::PlanSelection> remapped{{0U, 0U, {{0U}}}};
    std::string diagnostic;
    EXPECT_FALSE(mlir::bridge::NarrowFusedResultToSelections(input, selections, output, remapped, &diagnostic));
    EXPECT_FALSE(diagnostic.empty());
    EXPECT_TRUE(remapped.empty());
    EXPECT_STREQ(output.fused_graph_name.GetString(), "unchanged");
    ASSERT_EQ(output.node_idx_to_scheduled_results.size(), 1U);
    EXPECT_EQ(output.node_idx_to_scheduled_results[0].size(), 3U);
  }
}

TEST(PlanSelectionTest, NarrowingRejectsEmptyFusion) {
  ascir::FusedScheduledResult input, output;
  std::vector<mlir::bridge::PlanSelection> remapped;
  EXPECT_FALSE(mlir::bridge::NarrowFusedResultToSelections(input, {}, output, remapped));
}

}  // namespace
