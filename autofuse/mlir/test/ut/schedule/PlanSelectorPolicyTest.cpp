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

#include <map>
#include <string>

#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"
#include "test/ut/schedule/ScheduleTestUtils.h"
#include "framework/easy_asc_graph/asc_graph_builder.h"
#include "gen_tiling_impl.h"

namespace {

// Isolate ATT generation from the real ATT integration tests.
struct TilingGenerationStub {
  bool is_inductor_scene{false};
  bool succeed{true};
  std::map<std::string, std::string> generated;
  size_t calls{0U};
};

TilingGenerationStub *active_tiling_stub = nullptr;

class PlanSelectorPolicyTest : public testing::Test {
 protected:
  void SetUp() override {
    active_tiling_stub = &tiling_stub;
    tiling_stub.generated = {{"tiling_func", "// ATT program stub"}};
  }
  void TearDown() override {
    active_tiling_stub = nullptr;
  }
  TilingGenerationStub tiling_stub;
};

af::AscGraph MakeTransferGraph(const std::string &name) {
  using af::testing::AscGraphBuilder;
  AscGraphBuilder builder(name);
  builder.Loops({af::testing::Sym(4), af::testing::Sym(2)})
      .Data("input", 0, {af::testing::Sym(4), af::testing::Sym(2)}, {}, af::DT_FLOAT)
      .Load("load", "input")
      .Store("store", "load")
      .Output("output", "store", 0, af::DT_FLOAT);
  return builder.Build();
}

af::AscGraph MakeMalformedDataMovementGraph() {
  af::AscGraph graph = MakeTransferGraph("unsupported_data_movement");
  auto load = graph.FindNode("load");
  if (!load || load->outputs().empty()) {
    ADD_FAILURE() << "test graph has no load output";
    return graph;
  }
  auto &attr = load->outputs()[0]->attr;
  attr.axis = {0, 1};
  attr.repeats = {af::Symbol(4), af::Symbol(2)};
  attr.strides = {af::Symbol(8), af::Symbol(1)};
  attr.vectorized_axis = {0, 2};  // Axis 2 has no repeat in the descriptor.
  attr.vectorized_strides = {af::Symbol(8), af::Symbol(1)};
  return graph;
}

af::AscGraph MakeUnsupportedDataMovementGraph() {
  af::AscGraph graph = MakeTransferGraph("unsupported_data_movement");
  auto load = graph.FindNode("load");
  if (!load || load->outputs().empty()) {
    ADD_FAILURE() << "test graph has no load output";
    return graph;
  }
  auto &attr = load->outputs()[0]->attr;
  attr.axis = {0, 1};
  attr.repeats = {af::Symbol(4), af::Symbol(2)};
  attr.strides = {af::Symbol(8), af::Symbol(1)};
  attr.vectorized_axis = {0, 2};
  attr.vectorized_strides = {af::Symbol(8), af::Symbol(1)};
  return graph;
}

af::AscGraph MakeElementwiseGraph(const std::string &name) {
  using af::testing::AscGraphBuilder;
  AscGraphBuilder builder(name);
  builder.Loops({af::testing::Sym(8), af::testing::Sym(8)})
      .Data("input", 0, {af::testing::Sym(8), af::testing::Sym(8)}, {}, af::DT_FLOAT)
      .Load("load", "input")
      .Abs("abs", "load")
      .Store("store", "abs")
      .Output("output", "store", 0, af::DT_FLOAT);
  return builder.Build();
}

af::AscGraph MakeUnsupportedElementwiseBroadcastGraph() {
  af::AscGraph graph = MakeElementwiseGraph("unsupported_elementwise_broadcast");
  auto absolute = graph.FindNode("abs");
  if (!absolute || absolute->inputs().empty() || absolute->outputs().empty()) {
    ADD_FAILURE() << "test graph has no elementwise tensor";
    return graph;
  }
  auto &input = absolute->inputs()[0]->attr;
  auto &output = absolute->outputs()[0]->attr;
  input.axis = {0, 1};
  input.repeats = {af::Symbol(8), af::Symbol(8)};
  input.strides = {af::Symbol(8), af::Symbol(1)};
  output.axis = {0, 1};
  output.repeats = {af::Symbol(8), af::Symbol(8)};
  output.strides = {af::Symbol(9), af::Symbol(1)};
  return graph;
}

af::AscGraph MakeWrongOuterPitchElementwiseBroadcastGraph() {
  af::AscGraph graph = MakeElementwiseGraph("wrong_outer_pitch_elementwise_broadcast");
  auto absolute = graph.FindNode("abs");
  if (!absolute || absolute->inputs().empty() || absolute->outputs().empty()) {
    ADD_FAILURE() << "test graph has no elementwise tensor";
    return graph;
  }
  auto &input = absolute->inputs()[0]->attr;
  auto &output = absolute->outputs()[0]->attr;
  input.axis = {0, 1};
  input.repeats = {af::Symbol(1), af::Symbol(8)};
  input.strides = {af::Symbol(1), af::Symbol(0)};
  output.axis = {0, 1};
  output.repeats = {af::Symbol(8), af::Symbol(8)};
  output.strides = {af::Symbol(999), af::Symbol(1)};
  return graph;
}

af::AscGraph MakeQueueGraph(const std::string &name) {
  using af::testing::AscGraphBuilder;
  AscGraphBuilder builder(name);
  builder.Loops({af::testing::Sym(4), af::testing::Sym(2)})
      .Data("input", 0, {af::testing::Sym(4), af::testing::Sym(2)}, {}, af::DT_FLOAT)
      .Load("load0", "input")
      .Load("load1", "input")
      .Add("add", "load0", "load1")
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT);
  return builder.Build();
}

void UseQueueView(af::AscGraph &graph, const std::string &nodeName, int64_t outerStride) {
  auto node = graph.FindNode(nodeName.c_str());
  if (!node || node->outputs().empty()) {
    ADD_FAILURE() << "test graph has no " << nodeName << " output";
    return;
  }
  auto &attr = node->outputs()[0]->attr;
  attr.mem.position = af::Position::kPositionVecIn;
  attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  attr.mem.reuse_id = 0;
  attr.que.id = 0;
  attr.que.buf_num = 1;
  attr.que.depth = 1;
  attr.axis = {0, 1};
  attr.repeats = {af::Symbol(4), af::Symbol(2)};
  attr.strides = {af::Symbol(outerStride), af::Symbol(1)};
  attr.vectorized_axis = {1};
  attr.vectorized_strides = {af::Symbol(1)};
}

// Construct different vectorized layouts sharing the same queue identity.
void UseDifferentVectorizedView(af::AscGraph &graph, const std::string &nodeName) {
  auto node = graph.FindNode(nodeName.c_str());
  if (!node || node->outputs().empty()) {
    ADD_FAILURE() << "test graph has no " << nodeName << " output";
    return;
  }
  auto &attr = node->outputs()[0]->attr;
  attr.strides[0] = af::Symbol(3);
  attr.vectorized_axis = {0, 1};
  attr.vectorized_strides = {af::Symbol(2), af::Symbol(1)};
}

}  // namespace

namespace att {

extern "C" bool GenTilingImplAutoFuseV3(const std::string &, const ascir::FusedScheduledResult &,
                                        std::map<std::string, std::string> &,
                                        std::map<std::string, std::string> &tiling_func, bool is_inductor_scene) {
  if (active_tiling_stub == nullptr) {
    ADD_FAILURE() << "ATT tiling generation called outside its test fixture";
    return false;
  }
  ++active_tiling_stub->calls;
  active_tiling_stub->is_inductor_scene = is_inductor_scene;
  tiling_func = active_tiling_stub->generated;
  return active_tiling_stub->succeed;
}

}  // namespace att

namespace {

TEST_F(PlanSelectorPolicyTest, DynamicCandidatesNeverGenerateAttModels) {
  const auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(input, selections));
  EXPECT_EQ(tiling_stub.calls, 0U);
  ASSERT_EQ(selections.size(), 3U);
  EXPECT_EQ(selections[2].template_ids_per_stage, (std::vector<std::vector<size_t>>{{0U, 1U, 2U}, {0U, 1U, 2U}}));
}

// 代价模型代不出常数时**保留全部候选**，把选择交给生成的 tiling 在运行期做，而不是报失败。
// 拿掉 `PlanSelector.cpp` 里那段回落，这条与下一条都红。

class AttDeclineClassificationTest : public testing::Test {
 protected:
  void SetUp() override {
    active_tiling_stub = &tiling_stub;
    schema.fields.push_back({"block_dim", "fusion", mlir::common::TilingFieldType::kUInt32});
  }
  void TearDown() override {
    active_tiling_stub = nullptr;
  }

  af::Status Run(bool *unsupported) {
    mlir::bridge::PlanSelection selection;
    selection.template_ids_per_stage = {{0U}};
    return mlir::bridge::RunAttForCandidates({"test", mlir::test::MakeFusedResult(), {selection}, schema, {}, false},
                                             bundle, &diagnostic, unsupported);
  }

  TilingGenerationStub tiling_stub;
  mlir::common::TilingFieldSchema schema;
  mlir::bridge::TilingCodeBundle bundle;
  std::string diagnostic;
};

TEST_F(AttDeclineClassificationTest, TilingGenerationDeclineIsCapabilityShortfall) {
  tiling_stub.succeed = false;
  bool unsupported = false;
  EXPECT_EQ(Run(&unsupported), af::FAILED);
  EXPECT_EQ(diagnostic, "ATT tiling generation failed");
  EXPECT_TRUE(unsupported) << diagnostic;
}

TEST_F(AttDeclineClassificationTest, GeneratedFieldMismatchStaysHardFailure) {
  tiling_stub.generated = {{"tiling_func", "  uint32_t k = tiling_data.get_graph0_tiling_key();\n"}};
  bool unsupported = false;
  EXPECT_EQ(Run(&unsupported), af::FAILED);
  EXPECT_NE(diagnostic.find("not present in TilingFieldSchema"), std::string::npos) << diagnostic;
  EXPECT_FALSE(unsupported) << diagnostic;
}

TEST_F(PlanSelectorPolicyTest, UnsupportedTemplateDoesNotChangeTemplateCoordinates) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto &templates = input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  templates.push_back(MakeMalformedDataMovementGraph());
  templates.push_back(MakeUnsupportedElementwiseBroadcastGraph());
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U, 1U, 2U}}}}));
}

TEST_F(PlanSelectorPolicyTest, PaddedRowsRemainCandidatesForTransferLowering) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto graph = MakeMalformedDataMovementGraph();
  graph.FindNode("load")->outputs()[0]->attr.vectorized_axis = {0, 1};
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = std::move(graph);
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, PaddedMultiAxisTemplateIsPreserved) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto graph = MakeMalformedDataMovementGraph();
  auto &attr = graph.FindNode("load")->outputs()[0]->attr;
  attr.axis = attr.vectorized_axis = {0, 1, 2};
  attr.repeats = {af::Symbol(2), af::Symbol(4), af::Symbol(5)};
  attr.strides = {af::Symbol(20), af::Symbol(5), af::Symbol(1)};
  attr.vectorized_strides = {af::Symbol(32), af::Symbol(8), af::Symbol(1)};
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = std::move(graph);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, UnsupportedPlanIsPreservedAlongsideOtherPlans) {
  auto input = mlir::test::MakeFusedResult(2U);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = MakeMalformedDataMovementGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, AllUnsupportedPlansArePreserved) {
  auto input = mlir::test::MakeFusedResult(2U);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = MakeMalformedDataMovementGraph();
  input.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0] = MakeMalformedDataMovementGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(input, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, DynamicCandidatesDoNotPrejudgeBackendCapabilities) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto queue = MakeQueueGraph("different_views");
  UseQueueView(queue, "load0", 2);
  UseQueueView(queue, "load1", 2);
  UseDifferentVectorizedView(queue, "load1");
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs = {
      MakeUnsupportedDataMovementGraph(), MakeUnsupportedElementwiseBroadcastGraph(), queue};
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(input, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U, 1U, 2U}}}}));
  input.node_idx_to_scheduled_results[0][0].schedule_groups.clear();
  EXPECT_FALSE(mlir::bridge::SelectDynamicCandidates(input, selections));
  EXPECT_TRUE(selections.empty());
}

TEST_F(PlanSelectorPolicyTest, PlanWithOneUnsupportedStageIsPreserved) {
  auto input = mlir::test::MakeMultiStageFusedResult(2U, 2U);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[1].impl_graphs[0] = MakeMalformedDataMovementGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}, {0U}}}, {0U, 1U, {{0U}, {0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, ByValueScalarDoesNotRequireTensorBroadcastCompatibility) {
  using af::testing::AscGraphBuilder;
  AscGraphBuilder builder("scalar_elementwise");
  builder.Loops({af::testing::Sym(8), af::testing::Sym(8)})
      .Data("input", 0, {af::testing::Sym(8), af::testing::Sym(8)}, {}, af::DT_FLOAT)
      .Load("load", "input")
      .Scalar("scalar", "2.5", af::DT_FLOAT)
      .Op<af::ascir_op::Add>("add", {"scalar", "load"}, 1U)
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT);
  auto graph = builder.Build();
  auto &output = graph.FindNode("add")->outputs()[0]->attr;
  output.vectorized_axis = {0, 1};
  output.vectorized_strides = {af::Symbol(8), af::Symbol(1)};

  auto input = mlir::test::MakeFusedResult();
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = graph;
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, CompatibleBroadcastViewsEmitNoRejection) {
  auto input = mlir::test::MakeFusedResult();
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = MakeElementwiseGraph("compatible");
  std::vector<mlir::bridge::PlanSelection> selections;
  testing::internal::CaptureStderr();
  const bool result = mlir::bridge::BuildAllCandidateSelections(input, selections);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(result);
  EXPECT_EQ(log.find("AF_MLIR_BROADCAST_REJECT"), std::string::npos);
}

TEST_F(PlanSelectorPolicyTest, UnsupportedElementwiseBroadcastTemplateIsPreserved) {
  auto input = mlir::test::MakeFusedResult(2U);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] =
      MakeUnsupportedElementwiseBroadcastGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, WrongOuterPitchElementwiseBroadcastTemplateIsPreserved) {
  auto input = mlir::test::MakeFusedResult(2U);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] =
      MakeWrongOuterPitchElementwiseBroadcastGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, QueuePitchDoesNotFilterTemplates) {
  for (bool mixedDtype : {false, true}) {
    auto input = mlir::test::MakeFusedResult(1U);
    auto graph = MakeQueueGraph("dtype_queue_pitch");
    UseQueueView(graph, "load0", 2);
    UseQueueView(graph, "load1", 2);
    auto &first = graph.FindNode("load0")->outputs()[0]->attr;
    auto &second = graph.FindNode("load1")->outputs()[0]->attr;
    first.vectorized_axis = second.vectorized_axis = {0, 1};
    first.vectorized_strides = {af::Symbol(8), af::Symbol(1)};
    second.vectorized_strides = {af::Symbol(4), af::Symbol(1)};
    second.dtype = mixedDtype ? af::DT_INT64 : first.dtype;
    input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = std::move(graph);
    std::vector<mlir::bridge::PlanSelection> selections;
    ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections));
    EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
  }
}

TEST_F(PlanSelectorPolicyTest, ConflictingQueueResourceTemplateIsPreserved) {
  auto input = mlir::test::MakeFusedResult(2U);
  auto conflicting = MakeQueueGraph("conflicting_queue_resources");
  UseQueueView(conflicting, "load0", 2);
  UseQueueView(conflicting, "load1", 3);
  auto supported = MakeQueueGraph("supported_queue_resources");
  UseQueueView(supported, "load0", 2);
  UseQueueView(supported, "load1", 2);
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] = std::move(conflicting);
  input.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0] = std::move(supported);
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic)) << diagnostic;
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, InvalidStructureAndShapeFailBeforeModelGeneration) {
  mlir::bridge::PlanGenerationResult result;
  const auto no_stage = mlir::test::MakeMultiStageFusedResult(1U, 0U);
  EXPECT_FALSE(mlir::bridge::PreparePlanGeneration(
      {no_stage, mlir::bridge::PlanGenerationMode::Static, {32, 196608}, true, {}, {}}, result));
  const auto input = mlir::test::MakeFusedResult(2U);
  EXPECT_FALSE(mlir::bridge::PreparePlanGeneration(
      {input, mlir::bridge::PlanGenerationMode::Static, {32, 196608}, true, {{"s-0", 1}}, {}}, result));
  EXPECT_EQ(tiling_stub.calls, 0U);
}

}  // namespace

TEST_F(PlanSelectorPolicyTest, LogicalSingletonBroadcastReachesImplementationSelection) {
  auto graph = MakeElementwiseGraph("singleton_broadcast");
  auto &source = graph.FindNode("abs")->inputs()[0]->attr;
  auto &output = graph.FindNode("abs")->outputs()[0]->attr;
  source.axis = output.axis = {0};
  source.repeats = {af::Symbol(1)};
  source.strides = source.vectorized_strides = {af::Symbol(0)};
  output.repeats = {af::Symbol(32)};
  output.strides = output.vectorized_strides = {af::Symbol(1)};
  source.vectorized_axis = output.vectorized_axis = {0};
  auto input = mlir::test::MakeFusedResult();
  input.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs = {graph};
  std::vector<mlir::bridge::PlanSelection> selections;
  EXPECT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, selections));
  ASSERT_EQ(selections.size(), 1U);
}

TEST_F(PlanSelectorPolicyTest, StaticPreparationKeepsEveryPlanStageAndTemplateWithoutExecutingSolver) {
  const auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  mlir::bridge::PlanGenerationResult result;
  ASSERT_TRUE(mlir::bridge::PreparePlanGeneration(
      {input, mlir::bridge::PlanGenerationMode::Static, {32, 196608}, true, {}, {}}, result));
  EXPECT_EQ(tiling_stub.calls, 1U);
  EXPECT_TRUE(tiling_stub.is_inductor_scene);
  std::vector<mlir::bridge::PlanSelection> expected;
  ASSERT_TRUE(mlir::bridge::BuildAllCandidateSelections(input, expected));
  EXPECT_EQ(result.selections, expected);
  ASSERT_TRUE(std::holds_alternative<mlir::bridge::TilingCodeBundle>(result.tiling));
  EXPECT_EQ(std::get<mlir::bridge::TilingCodeBundle>(result.tiling).tiling_functions, tiling_stub.generated);
}

TEST_F(PlanSelectorPolicyTest, UniqueCandidateDefersTilingToRuntime) {
  const auto input = mlir::test::MakeFusedResult();
  mlir::bridge::PlanGenerationResult result;
  ASSERT_TRUE(mlir::bridge::PreparePlanGeneration(
      {input, mlir::bridge::PlanGenerationMode::Static, {32, 196608}, true, {}, {}}, result));
  EXPECT_EQ(tiling_stub.calls, 1U);
  EXPECT_EQ(result.selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}}));
  EXPECT_TRUE(std::holds_alternative<mlir::bridge::TilingCodeBundle>(result.tiling));
}

TEST_F(PlanSelectorPolicyTest, DynamicPreparationGeneratesOnceAndDoesNotExecuteSolver) {
  const auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  mlir::bridge::PlanGenerationResult result;
  ASSERT_TRUE(mlir::bridge::PreparePlanGeneration(
      {input, mlir::bridge::PlanGenerationMode::Dynamic, {32, 196608}, false, {}, {}}, result));
  EXPECT_EQ(tiling_stub.calls, 1U);
  EXPECT_FALSE(tiling_stub.is_inductor_scene);
  ASSERT_EQ(result.selections.size(), 3U);
  EXPECT_EQ(result.selections[2].template_ids_per_stage, (std::vector<std::vector<size_t>>{{0, 1, 2}, {0, 1, 2}}));
  EXPECT_TRUE(std::holds_alternative<mlir::bridge::TilingCodeBundle>(result.tiling));
}
TEST_F(PlanSelectorPolicyTest, InvalidResourceBudgetsNeverReachAtt) {
  const auto input = mlir::test::MakeFusedResult();
  for (const auto budget : {mlir::common::ResourceBudget{0, 196608},
                            {32, 256},
                            {32, 0},
                            {uint64_t(UINT32_MAX) + 1, 196608},
                            {32, uint64_t(UINT32_MAX) + 1}}) {
    mlir::bridge::PlanGenerationResult result;
    EXPECT_FALSE(mlir::bridge::PreparePlanGeneration(
        {input, mlir::bridge::PlanGenerationMode::Static, budget, true, {}, {}}, result));
  }
  EXPECT_EQ(tiling_stub.calls, 0U);
  EXPECT_EQ((mlir::common::ResourceBudget{1, 257}).availableUb(), 1U);
}
