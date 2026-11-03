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
#include "gen_model_info/gen_model_info.h"
#include "gen_tiling_impl.h"

namespace {

// This executable supplies its own ATT model-generation symbol. Keep it out
// of ScheduleSpaceBridgeTest so those tests continue linking against real ATT.
struct ModelGenerationStub {
  size_t calls{0U};
  af::Status status{af::SUCCESS};
  att::FusedParsedScheduleResult models;
  mlir::bridge::AttOptions options;
  const ascir::FusedScheduledResult *input{nullptr};
};

ModelGenerationStub *active_stub = nullptr;

struct TilingGenerationStub {
  bool succeed{true};
  std::map<std::string, std::string> generated;
  size_t calls{0U};
};

TilingGenerationStub *active_tiling_stub = nullptr;

class PlanSelectorPolicyTest : public testing::Test {
 protected:
  void SetUp() override {
    active_stub = &stub;
  }
  void TearDown() override {
    active_stub = nullptr;
  }
  ModelGenerationStub stub;
};

att::ParsedScheduleResult PlanWithTemplateCosts(std::initializer_list<int64_t> costs) {
  att::ParsedScheduleResult plan;
  for (int64_t cost : costs) {
    att::ModelInfo model{};
    model.head_cost = att::CreateExpr(cost);
    plan.groups_tiling_model_info[0].push_back(model);
  }
  return plan;
}

// 代价里带一个**不是 shape 符号**的量。真实模型里这种量是 ATT 求解器自己的（分核数
// `block_dim`、逐轴切分尺寸），要等 tiling 解完才有值，而计划选择跑在 tiling 之前。
att::ParsedScheduleResult PlanWithSolverBoundCosts(size_t template_count) {
  att::ParsedScheduleResult plan;
  for (size_t index = 0U; index < template_count; ++index) {
    att::ModelInfo model{};
    model.head_cost = af::Symbol("block_dim") + att::CreateExpr(static_cast<int64_t>(index));
    plan.groups_tiling_model_info[0].push_back(model);
  }
  return plan;
}

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

// 改变队列视图的实际步长，让它与同一个 que id / reuse id 上的另一份不相等。
// `InputBroadcastsToOutput` 只看 elementwise 输入输出视图，所以这个反例只违反队列资源判据。
//
// 外轴步长 2 == 内轴 repeat 2 × 内轴步长 1，所以它照旧收成一段，data-movement 判据也不碰。
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

// 构造一个错误的向量化轴引用，让 `TemplateSupportsMlirDataMovement` 拒掉这个模板——用来造
// 「一个模板同时违反两条判据」的场合。
void SplitLocalTileIntoTwoSegments(af::AscGraph &graph, const std::string &nodeName) {
  auto node = graph.FindNode(nodeName.c_str());
  if (!node || node->outputs().empty()) {
    ADD_FAILURE() << "test graph has no " << nodeName << " output";
    return;
  }
  auto &attr = node->outputs()[0]->attr;
  attr.vectorized_axis = {0, 2};
  attr.vectorized_strides = {af::Symbol(5), af::Symbol(1)};
}

}  // namespace

namespace att {

af::Status GetModelInfoMap(const ascir::FusedScheduledResult &input, const std::map<std::string, std::string> &options,
                           FusedParsedScheduleResult &models) {
  if (active_stub == nullptr) {
    ADD_FAILURE() << "ATT model generation called outside its test fixture";
    return af::FAILED;
  }
  ++active_stub->calls;
  active_stub->input = &input;
  active_stub->options = options;
  models = active_stub->models;
  return active_stub->status;
}

extern "C" bool GenTilingImplAutoFuseV3(const std::string &, const ascir::FusedScheduledResult &,
                                        std::map<std::string, std::string> &,
                                        std::map<std::string, std::string> &tiling_func, bool) {
  if (active_tiling_stub == nullptr) {
    ADD_FAILURE() << "ATT tiling generation called outside its test fixture";
    return false;
  }
  ++active_tiling_stub->calls;
  tiling_func = active_tiling_stub->generated;
  return active_tiling_stub->succeed;
}

}  // namespace att

namespace {

TEST_F(PlanSelectorPolicyTest, UniqueCandidatesNeverGenerateAttModels) {
  auto input = mlir::test::MakeMultiStageFusedResult(1U, 3U);
  input.node_idx_to_scheduled_results.push_back({mlir::test::MakeMultiStageScheduledResult(2U)});
  stub.status = af::FAILED;
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections));
  EXPECT_EQ(stub.calls, 0U);
  EXPECT_EQ(selections,
            (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}, {0U}, {0U}}}, {1U, 0U, {{0U}, {0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, DynamicCandidatesNeverGenerateAttModels) {
  const auto input = mlir::test::MakeMultiStageFusedResult(3U, 2U, 3U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectDynamicCandidates(input, selections));
  EXPECT_EQ(stub.calls, 0U);
  ASSERT_EQ(selections.size(), 3U);
  EXPECT_EQ(selections[2].template_ids_per_stage, (std::vector<std::vector<size_t>>{{0U, 1U, 2U}, {0U, 1U, 2U}}));
}

TEST_F(PlanSelectorPolicyTest, MultiplePlansGenerateModelsOnceAndRetainOriginalWinnerId) {
  const auto input = mlir::test::MakeMultiSubgraphFusedResult({2U, 1U});
  stub.models[0][0] = PlanWithTemplateCosts({20});
  stub.models[0][1] = PlanWithTemplateCosts({3});
  stub.models[1][0] = PlanWithTemplateCosts({5});
  const mlir::bridge::AttOptions options{{"test_option", "test_value"}};
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(input, {}, options, selections));
  EXPECT_EQ(stub.calls, 1U);
  EXPECT_EQ(stub.input, &input);
  EXPECT_EQ(stub.options, options);
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 1U, {{0U}}}, {1U, 0U, {{0U}}}}));
}

// 这条同时是回落的反向断言：**能代出常数就必须还在编译期定下来**，
// 选出来的只有一个模板，回落不许把这条路吃掉。
TEST_F(PlanSelectorPolicyTest, MultipleTemplatesRequireModelsEvenWithOnePlan) {
  const auto input = mlir::test::MakeFusedResult(1U, 2U);
  stub.models[0][0] = PlanWithTemplateCosts({20, 3});
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections));
  EXPECT_EQ(stub.calls, 1U);
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{1U}}}}));
}

// 代价模型代不出常数时**保留全部候选**，把选择交给生成的 tiling 在运行期做，而不是报失败。
// 拿掉 `PlanSelector.cpp` 里那段回落，这条与下一条都红。
TEST_F(PlanSelectorPolicyTest, SolverBoundCostsDeferSelectionToRuntime) {
  const auto input = mlir::test::MakeFusedResult(1U, 2U);
  stub.models[0][0] = PlanWithSolverBoundCosts(2U);
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections, &diagnostic));
  EXPECT_EQ(stub.calls, 1U);
  // 两个模板一个都没丢——与 `SelectDynamicCandidates` 在同一张图上的产出一致。
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U, 1U}}}}));
  // 诊断留着，记的是"为什么没在编译期定下来"。
  EXPECT_EQ(diagnostic, "ATT did not provide a complete, constant-cost plan for subgraph 0");
}

// 多个 plan 时同理：plan 与模板都不丢，交给运行期。
TEST_F(PlanSelectorPolicyTest, SolverBoundCostsRetainEveryPlan) {
  const auto input = mlir::test::MakeFusedResult(2U);
  stub.models[0][0] = PlanWithSolverBoundCosts(1U);
  stub.models[0][1] = PlanWithSolverBoundCosts(1U);
  std::vector<mlir::bridge::PlanSelection> selections;
  ASSERT_TRUE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections));
  EXPECT_EQ(selections, (std::vector<mlir::bridge::PlanSelection>{{0U, 0U, {{0U}}}, {0U, 1U, {{0U}}}}));
}

TEST_F(PlanSelectorPolicyTest, AttFailureDoesNotReturnUnscoredCandidates) {
  const auto input = mlir::test::MakeFusedResult(2U);
  stub.status = af::FAILED;
  std::vector<mlir::bridge::PlanSelection> selections{{0U, 0U, {{0U}}}};
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections, &diagnostic));
  EXPECT_EQ(stub.calls, 1U);
  EXPECT_EQ(diagnostic, "ATT model information generation failed");
  EXPECT_TRUE(selections.empty());
}

TEST_F(PlanSelectorPolicyTest, AttModelGenerationFailureIsCapabilityShortfallNotBreakage) {
  const auto input = mlir::test::MakeFusedResult(2U);
  stub.status = af::FAILED;
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  bool unsupported = false;
  EXPECT_FALSE(mlir::bridge::SelectStaticWinner(input, {}, {}, selections, &diagnostic, &unsupported));
  EXPECT_TRUE(unsupported) << diagnostic;
}

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
  std::vector<mlir::bridge::PlanSelection> selections{{0U, 0U, {{0U}}}};
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic));
  // 诊断要带拒收归属：是哪一条判据拒的、拒了几个模板。整片 `candidate-preparation` 在接管表
  // 里没有首因就是因为这句话只说「没有支持的模板」（2026-10-26 全量表 55 格无诊断）。
  EXPECT_EQ(diagnostic,
            "subgraph 0 contains no plan with an MLIR-supported template "
            "(plan 0 stage 0: 1 template(s), all rejected by: data-movement 1)");
  EXPECT_TRUE(selections.empty());
}

// 「能力不足」这一位只在模板判据那一条路上置真。另一侧（调度结构坏了）必须留 false——它一旦
// 被标成能力不足就会静默回落给 legacy，再也没人发现。
TEST_F(PlanSelectorPolicyTest, OnlyUnsupportedTemplatesReportCapabilityShortfall) {
  auto allTemplatesRejected = mlir::test::MakeFusedResult(2U);
  allTemplatesRejected.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0] =
      MakeUnsupportedDataMovementGraph();
  allTemplatesRejected.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0] =
      MakeUnsupportedDataMovementGraph();
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  bool unsupported = false;
  EXPECT_FALSE(mlir::bridge::BuildAllCandidateSelections(allTemplatesRejected, selections, &diagnostic, &unsupported));
  EXPECT_TRUE(unsupported) << diagnostic;

  auto brokenStructure = mlir::test::MakeFusedResult(1U);
  brokenStructure.node_idx_to_scheduled_results[0][0].schedule_groups.clear();
  selections.clear();
  diagnostic.clear();
  bool brokenUnsupported = false;
  EXPECT_FALSE(mlir::bridge::BuildAllCandidateSelections(brokenStructure, selections, &diagnostic, &brokenUnsupported));
  EXPECT_FALSE(brokenUnsupported) << diagnostic;
}

// 两个模板各被不同的一条判据拒掉时，覆盖是一个组合，没有单一首因。这一格会红的变异是退回挑
// 一条报（无论挑哪条都在把组合谎称成单条）。`af-capability-probe` 的 T3 两格就是这个形状。
TEST_F(PlanSelectorPolicyTest, RejectionAttributionSaysSoWhenNoSinglePredicateRejectsAll) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto queueOnly = MakeQueueGraph("queue_conflict_only");
  UseQueueView(queueOnly, "load0", 2);
  UseQueueView(queueOnly, "load1", 2);
  UseDifferentVectorizedView(queueOnly, "load1");

  auto &stage = input.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  stage.impl_graphs.clear();
  stage.impl_graphs.push_back(MakeUnsupportedDataMovementGraph());
  stage.impl_graphs.push_back(std::move(queueOnly));

  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic));
  EXPECT_EQ(diagnostic,
            "subgraph 0 contains no plan with an MLIR-supported template "
            "(plan 0 stage 0: 2 template(s), no single predicate rejects all; "
            "counts: data-movement 1, elementwise-broadcast 1, queue-resources 1)");
}

// 首因必须是「拒掉全部模板」的那一条。这一格会红的变异是把归属改成按判据顺序报第一条非零的：
// 那样这里会答成 data-movement，而换掉 data-movement 那个模板这个 stage 仍然进不来——真正挡住
// 它的是 queue-resources。实测有 3 格栽在这个区别上（见 PlanSelection.cpp 的 `Str()`）。
TEST_F(PlanSelectorPolicyTest, RejectionAttributionNamesThePredicateRejectingEveryTemplate) {
  auto input = mlir::test::MakeFusedResult(1U);
  auto queueOnly = MakeQueueGraph("queue_conflict_only");
  UseQueueView(queueOnly, "load0", 2);
  UseQueueView(queueOnly, "load1", 2);
  UseDifferentVectorizedView(queueOnly, "load1");
  auto queueAndDataMovement = MakeQueueGraph("queue_conflict_and_multi_segment");
  UseQueueView(queueAndDataMovement, "load0", 2);
  UseQueueView(queueAndDataMovement, "load1", 2);
  UseDifferentVectorizedView(queueAndDataMovement, "load1");
  SplitLocalTileIntoTwoSegments(queueAndDataMovement, "load0");

  auto &stage = input.node_idx_to_scheduled_results[0][0].schedule_groups[0];
  stage.impl_graphs.clear();
  stage.impl_graphs.push_back(std::move(queueOnly));
  stage.impl_graphs.push_back(std::move(queueAndDataMovement));

  std::vector<mlir::bridge::PlanSelection> selections;
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::BuildAllCandidateSelections(input, selections, &diagnostic));
  EXPECT_EQ(diagnostic,
            "subgraph 0 contains no plan with an MLIR-supported template "
            "(plan 0 stage 0: 2 template(s), all rejected by: elementwise-broadcast 2, queue-resources 2; "
            "partially: data-movement 1)");
}

TEST_F(PlanSelectorPolicyTest, PlanWithOneUnsupportedStageIsSkipped) {
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
  std::vector<mlir::bridge::PlanSelection> selections;
  const auto no_stage = mlir::test::MakeMultiStageFusedResult(1U, 0U);
  EXPECT_FALSE(mlir::bridge::SelectStaticWinner(no_stage, {}, {}, selections));
  const auto multiple_plans = mlir::test::MakeFusedResult(2U);
  EXPECT_FALSE(mlir::bridge::SelectStaticWinner(multiple_plans, {{"s-0", 1}}, {}, selections));
  EXPECT_EQ(stub.calls, 0U);
}

}  // namespace
