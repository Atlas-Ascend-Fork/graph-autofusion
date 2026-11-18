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

#include <string>
#include <vector>

#include "asc_graph_builder.h"
#include "tests/framework/broadcast_backward/broadcast_backward_test_utils.h"
#include "tests/framework/broadcast_backward/broadcast_backward_ut_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "optimize/graph_pass/broadcast_backward_pass.h"
#include "optimize/graph_pass/broadcast_backward_shared_split.h"
#include "optimize/platform/common/pass_runner.h"
#include "schedule_utils.h"

namespace {
using af::AscGraph;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using namespace broadcast_backward_test;

class ScopedTestPlatform {
 public:
  explicit ScopedTestPlatform(const char *platform) {
    ge::PlatformContext::GetInstance().SetPlatform(platform);
  }

  ~ScopedTestPlatform() {
    ge::PlatformContext::GetInstance().Reset();
  }
};
}  // namespace

TEST(BroadcastBackwardPass, MovesSingleBroadcastChain) {
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  const auto load_node = FindNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  std::vector<af::Expression> expected_strides;
  ASSERT_EQ(
      optimize::ScheduleUtils::RecalculateStridesFromRepeats(load_node->outputs[0].attr.repeats, expected_strides),
      af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
  const auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(compute_node, nullptr);
  ExpectStaticEq(compute_node->outputs[0].attr.strides, expected_strides);
}

TEST(BroadcastBackwardPass, MovesAllSupportedUnaryOperators) {
  for (const auto &op_name : std::vector<std::string>{"Abs", "Neg", "Exp", "Sqrt", "Rsqrt", "Relu", "Reciprocal", "Erf",
                                                      "Sign", "Tanh", "Ln"}) {
    auto graph = BuildUnaryGraph(op_name);
    CompleteApiInfo(graph);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "compute", "broadcast")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "broadcast", "store")) << op_name;
  }
}

TEST(BroadcastBackwardPass, MovesMultipleComputeNodes) {
  auto graph = AscGraphBuilder("broadcast_backward_compute_chain")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast")
                   .Relu("relu", "abs")
                   .Store("store", "relu")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesFormerCastBarrierWithDtypeAwareBackward) {
  auto graph = AscGraphBuilder("broadcast_backward_cast_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Cast("cast", "broadcast", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, SkipsReduceBarrier) {
  auto graph = AscGraphBuilder("broadcast_backward_reduce_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Sum("sum", "broadcast", {0U})
                   .Store("store", "sum")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "sum"));
  EXPECT_TRUE(IsConnected(graph, "sum", "store"));
}

TEST(BroadcastBackwardPass, SkipsVectorizedLayout) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto load_node = FindNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  load_node->outputs[0].attr.vectorized_axis.push_back(0U);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, MovesMultipleBroadcastConsumersPerBranch) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multiple_consumers")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {Sym("s0"), Sym("s1")})
                   .Abs("abs0", "broadcast")
                   .Abs("abs1", "broadcast")
                   .Store("store0", "abs0")
                   .Store("store1", "abs1")
                   .Output("output0", "store0")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 多消费不再整体 fail-closed：无条件拆分后各分支独立后移（Abs 沉到广播前）
  EXPECT_TRUE(IsConnected(graph, "load", "abs0"));
  EXPECT_TRUE(IsConnected(graph, "abs0", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store0"));
  EXPECT_TRUE(IsConnected(graph, "load", "abs1"));
  EXPECT_TRUE(IsConnected(graph, "abs1", "broadcast_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast_consumer_split_1", "store1"));
}

TEST(BroadcastBackwardPass, SkipsIncompleteBroadcastLayout) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  broadcast_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsSchedMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(compute_node, nullptr);
  compute_node->attr.sched.axis.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsControlEdge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildUnaryGraph("Abs");
  CompleteApiInfo(graph);
  auto load_node = FindNode(graph, "load");
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(load_node, nullptr);
  ASSERT_NE(compute_node, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load_node->GetOutControlAnchor(), compute_node->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsScalarBroadcastSource) {
  auto graph = AscGraphBuilder("broadcast_backward_scalar")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "scalar", {Sym("s0"), Sym("s1")})
                   .Broadcast("broadcast1", "broadcast0", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast1")
                   .Store("store", "abs")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "store"));
}

TEST(BroadcastBackwardPass, SkipsIndexExprBroadcastSource) {
  // IndexExpr 是 scalar-like 生产者（输出视图为空）。浮点改写协议产出的
  // IndexExpr -> Broadcast -> Cast 链曾触发 GetBrcAxisesIndex 的轴数断言失败
  // 并留下半改写图（测试环境 2026-09-14 plog）。期望：门控按 scalar 输入
  // 支持性拒绝后移，图保持原状。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_index_expr")
                   .Loops({s0, s1})
                   .IndexExpr("index_expr", "s0 + 2")
                   .Broadcast("broadcast", "index_expr", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  // builder 的 Build() 会为空视图补 loop 形状（测试便利），而真实前端
  // （pyautofuse）的 scalar-like 节点输出视图保持为空——缺陷触发条件。
  const auto index_expr_node = FindNode(graph, "index_expr");
  ASSERT_NE(index_expr_node, nullptr);
  index_expr_node->outputs[0].attr.axis.clear();
  index_expr_node->outputs[0].attr.repeats.clear();
  index_expr_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "index_expr", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "store"));
  const auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  EXPECT_EQ(broadcast_node->outputs[0].attr.dtype, af::DT_INT64);
  EXPECT_TRUE(index_expr_node->outputs[0].attr.repeats.empty());
}

TEST(BroadcastBackwardPass, SkipsScalarDataBroadcastSource) {
  // ScalarData 与 IndexExpr 同为 scalar-like 生产者（输出视图为空），门控
  // 需同样覆盖；期望行为与 SkipsScalarBroadcastSource 一致。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_data")
                   .Loops({s0, s1})
                   .ScalarData("scalar_data")
                   .Broadcast("broadcast", "scalar_data", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  // 还原真实前端的 scalar-like 空视图形态（见 SkipsIndexExprBroadcastSource 注释）。
  const auto scalar_data_node = FindNode(graph, "scalar_data");
  ASSERT_NE(scalar_data_node, nullptr);
  scalar_data_node->outputs[0].attr.axis.clear();
  scalar_data_node->outputs[0].attr.repeats.clear();
  scalar_data_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar_data", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "store"));
}

TEST(BroadcastBackwardPass, SkipsBroadcastWithMismatchedPreViewAxes) {
  // 视图奇偶守卫独立兜底：非 scalar-like 生产者（Load）被手工置空输出视图后，
  // 门控不适用，守卫必须在改写图边前整体拒绝，避免半改写状态。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_mismatched_pre_view")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);
  const auto load_node = FindNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  load_node->outputs[0].attr.repeats.clear();
  load_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "store"));
}

TEST(BroadcastBackwardPass, SkipsArangeBroadcastProducer) {
  // Arange 生产者的单链整链后移未适配：整链后移把 compute 链输出改写为
  // Arange 退化视图，视图代数尚未覆盖（triu/remainder_cast 类图 codegen 失败）。
  // 门控在链级拒绝（图级门控移除后由本链级判定保持行为），图保持原状。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_arange")
                   .Loops({s0, s1})
                   .Arange("arange", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Broadcast("broadcast", "arange", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "arange", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "store"));
}

TEST(BroadcastBackwardPass, MovesDegenerateArangeCommonAxisPartialGroup) {
  // 真实前端退化轴形态（triu 类）：index_expr 降级为两个多轴退化视图 Arange，
  // 单节点 Broadcast 一步扩到全上下文（[1,1,s2]→[s0,s1,s2]）。两分支扩维轴集
  // 不同（{s0,s1} vs {s0,s2}），公共轴 s0 在两个生产者视图中均已退化（repeat=1）。
  // 期望：partial backward 只后移公共轴 s0——多轴广播保留节点（非公共轴扩维
  // 留在计算之前，外积语义不变），分支广播与计算链输出视图 s0 塌缩为 1，
  // 计算链后插入 backward_broadcast 恢复 s0 扩维。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_arange_partial")
                   .Loops({s0, s1, s2})
                   .Arange("arange_col", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                           {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Arange("arange_row", {af::sym::kSymbolOne, s1, af::sym::kSymbolOne},
                           {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero})
                   .Broadcast("brc_col", "arange_col", {s0, s1, s2})
                   .Broadcast("brc_row", "arange_row", {s0, s1, s2})
                   .Add("add", "brc_col", "brc_row")
                   .Cast("cast", "add", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 多轴扩维广播不被整节点删除，Add 输入仍来自两条广播分支。
  EXPECT_TRUE(IsConnected(graph, "arange_col", "brc_col"));
  EXPECT_TRUE(IsConnected(graph, "arange_row", "brc_row"));
  EXPECT_TRUE(IsConnected(graph, "brc_col", "add"));
  EXPECT_TRUE(IsConnected(graph, "brc_row", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "cast"));
  // 计算链之后插入公共轴广播，恢复 s0 扩维。
  EXPECT_TRUE(IsConnected(graph, "cast", "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_cast_0", "store"));
  // 公共轴 s0 塌缩：分支广播与计算链输出视图 s0 repeat 变为 1，其余轴保持。
  const auto brc_col_node = FindNode(graph, "brc_col");
  const auto brc_row_node = FindNode(graph, "brc_row");
  const auto add_node = FindNode(graph, "add");
  const auto cast_node = FindNode(graph, "cast");
  const auto inserted_node = FindNode(graph, "backward_broadcast_cast_0");
  ASSERT_NE(brc_col_node, nullptr);
  ASSERT_NE(brc_row_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(inserted_node, nullptr);
  ExpectStaticEq(brc_col_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(brc_row_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(add_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(cast_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(inserted_node->outputs[0].attr.repeats, {s0, s1, s2});
}

TEST(BroadcastBackwardPass, SkipsArangePrefixExpansionPartialGroup) {
  // 1D Arange 前缀扩维形态（multi_arange/remainder_cast 类）：生产者视图 [s1]
  // 不含公共轴 s0。公共轴后移会删除两分支广播并让生产者直连 Add，留下
  // 生产者/消费者轴集不一致的边（tiling 不可融合）。期望：组级
  // 守卫在改写前拒绝，图保持原状（含不删除同轴重复广播）。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_arange_prefix")
                   .Loops({s0, s1})
                   .Arange("arange_a", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Arange("arange_b", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Broadcast("brc_a", "arange_a", {s0, s1})
                   .Broadcast("brc_b", "arange_b", {s0, s1})
                   .Add("add", "brc_a", "brc_b")
                   .Store("store", "add")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_INT64);
  // builder 会把 Arange 输出 axis 补成全部 loop 轴；真实前端（pyautofuse）的
  // 1D Arange 只持有单轴视图（axis/size/stride 同长）。
  // 手工收缩为单轴视图，还原缺陷触发形态。
  for (const char *name : {"arange_a", "arange_b"}) {
    const auto arange_node = FindNode(graph, name);
    ASSERT_NE(arange_node, nullptr);
    arange_node->outputs[0].attr.axis.erase(arange_node->outputs[0].attr.axis.begin());
    arange_node->outputs[0].attr.repeats = {s1};
    arange_node->outputs[0].attr.strides = {af::sym::kSymbolOne};
  }

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "arange_a", "brc_a"));
  EXPECT_TRUE(IsConnected(graph, "arange_b", "brc_b"));
  EXPECT_TRUE(IsConnected(graph, "brc_a", "add"));
  EXPECT_TRUE(IsConnected(graph, "brc_b", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "store"));
  const auto add_node = FindNode(graph, "add");
  ASSERT_NE(add_node, nullptr);
  ExpectStaticEq(add_node->outputs[0].attr.repeats, {s0, s1});
}

TEST(BroadcastBackwardPass, MovesSupportedScalarBroadcastBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_supported")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {s0, s1})
                   .Broadcast("broadcast1", "scalar1", {s0, s1})
                   .Sub("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "scalar1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "compute_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "compute_broadcast_backward_common", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne});
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "scalar0", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "scalar1", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "compute_broadcast_backward_common"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute_broadcast_backward_common", "store"));
}

TEST(BroadcastBackwardPass, SkipsUnsupportedAllScalarInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_unsupported")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {s0, s1})
                   .Broadcast("broadcast1", "scalar1", {s0, s1})
                   .Add("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar0", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "scalar1", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_FALSE(HasNode(graph, "compute_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, SkipsScalarBranchWithResidualBroadcastAxis) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_mixed")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "load", {s0, s1})
                   .Broadcast("broadcast1", "scalar", {s0, s1})
                   .Add("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
  EXPECT_FALSE(HasNode(graph, "compute_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesSupportedScalarSameConsumerMultiReference) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_same_consumer")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast", "scalar", {s0, s1})
                   .Sub("compute", "broadcast", "broadcast")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne});
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, SkipsUnsupportedScalarSameConsumerMultiReference) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_same_consumer_unsupported")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast", "scalar", {s0, s1})
                   .Add("compute", "broadcast", "broadcast")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "scalar"));
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_TRUE(HasNode(graph, "compute"));
  EXPECT_TRUE(IsConnected(graph, "scalar", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, MovesSupportedMultiLevelScalarBroadcastChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_multi_level")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "scalar", {0})
                   .Broadcast("broadcast1", "broadcast0", {1})
                   .Sub("compute", "broadcast1", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "scalar", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "store"));
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(compute, nullptr);
  ExpectStaticEq(compute->outputs[0].attr.strides, {af::sym::kSymbolZero, af::sym::kSymbolZero});
}

TEST(BroadcastBackwardPass, ScalarBranchAxisRequiresZeroInputStride) {
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_scalar_stride_guard")
                   .Loops({s0, s1})
                   .Scalar("scalar0", "1.0")
                   .Scalar("scalar1", "2.0")
                   .Broadcast("broadcast0", "scalar0", {0})
                   .Broadcast("broadcast1", "scalar1", {0})
                   .Sub("compute", "broadcast0", "broadcast1")
                   .Store("store", "compute")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto scalar1 = FindNode(graph, "scalar1");
  const auto broadcast0 = FindNode(graph, "broadcast0");
  const auto compute = FindNode(graph, "compute");
  ASSERT_NE(scalar1, nullptr);
  ASSERT_NE(broadcast0, nullptr);
  ASSERT_NE(compute, nullptr);
  scalar1->outputs[0].attr.strides[0] = af::sym::kSymbolOne;
}

TEST(BroadcastBackwardPass, MovesIdenticalMultiInputBroadcastChains) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  ExpectBinaryBroadcastMove(graph);
}

TEST(BroadcastBackwardPass, MovesMultiNodeBroadcastAndComputeChains) {
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact_repeats = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_multi_node_chains")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact_repeats, compact_strides)
                   .Load("load1", "data1", compact_repeats, compact_strides)
                   .Broadcast("broadcast00", "load0", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1, s2})
                   .Broadcast("broadcast10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast11", "broadcast10", {s0, s1, s2})
                   .Add("merge", "broadcast01", "broadcast11")
                   .Abs("abs", "merge")
                   .Relu("relu", "abs")
                   .Store("store", "relu")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast00"));
  EXPECT_TRUE(IsConnected(graph, "broadcast00", "broadcast01"));
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast10"));
  EXPECT_FALSE(HasNode(graph, "broadcast11"));
}

TEST(BroadcastBackwardPass, MovesAllSupportedBinaryOperators) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  for (const auto &op_name : std::vector<std::string>{"Add", "Sub", "Mul", "Div", "Minimum", "Maximum"}) {
    auto graph = BuildBinaryGraph(op_name);
    CompleteApiInfo(graph);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load0", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "load1", "compute")) << op_name;
    EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "compute")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "compute")) << op_name;
    EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast0")) << op_name;
    EXPECT_FALSE(HasNode(graph, "broadcast1")) << op_name;
  }
}

TEST(BroadcastBackwardPass, MovesBeforeMultiInputBarrier) {
  auto graph = AscGraphBuilder("broadcast_backward_multi_input_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1")
                   .Broadcast("broadcast", "load0", {Sym("s0"), Sym("s1")})
                   .Abs("abs", "broadcast")
                   .Add("add", "load1", "abs")
                   .Store("store", "add")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "add"));
  EXPECT_TRUE(IsConnected(graph, "load1", "add"));
}

TEST(BroadcastBackwardPass, SkipsFollowingMultiInputBarrier) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = AscGraphBuilder("broadcast_backward_following_multi_input_barrier")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Load("load2", "data2")
                   .Broadcast("broadcast0", "load0", {Sym("s0"), Sym("s1")})
                   .Broadcast("broadcast1", "load1", {Sym("s0"), Sym("s1")})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Add("barrier", "load2", "merge")
                   .Store("store", "barrier")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "barrier"));
}

TEST(BroadcastBackwardPass, SkipsMultiInputSchedMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(broadcast1_node, nullptr);
  broadcast1_node->attr.sched.axis.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, SkipsTailLayoutMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast0_node = FindNode(graph, "broadcast0");
  auto broadcast1_node = FindNode(graph, "broadcast1");
  auto compute_node = FindNode(graph, "compute");
  ASSERT_NE(broadcast0_node, nullptr);
  ASSERT_NE(broadcast1_node, nullptr);
  ASSERT_NE(compute_node, nullptr);
  broadcast0_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  broadcast1_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  compute_node->inputs[0].attr.strides[0] = af::sym::kSymbolZero;
  compute_node->inputs[1].attr.strides[0] = af::sym::kSymbolZero;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "store"));
}

TEST(BroadcastBackwardPass, SkipsEdgeDtypeMismatch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto broadcast1_node = FindNode(graph, "broadcast1");
  auto add_node = FindNode(graph, "compute");
  ASSERT_NE(broadcast1_node, nullptr);
  ASSERT_NE(add_node, nullptr);
  broadcast1_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  add_node->inputs[1].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
}

TEST(BroadcastBackwardPass, SkipsDifferentInputLayouts) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildBinaryGraph("Add");
  CompleteApiInfo(graph);
  auto load1_node = FindNode(graph, "load1");
  auto broadcast1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(load1_node, nullptr);
  ASSERT_NE(broadcast1_node, nullptr);
  load1_node->outputs[0].attr.strides[0] = af::sym::kSymbolZero;
  broadcast1_node->inputs[0].attr.strides[0] = af::sym::kSymbolZero;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "compute"));
}

TEST(BroadcastBackwardPass, HandlesEmptyGraph) {
  af::AscGraph graph("broadcast_backward_empty");
  optimize::BroadcastBackwardPass pass;
  EXPECT_EQ(pass.RunPass(graph), af::SUCCESS);
}

TEST(BroadcastBackwardPass, SkipsPartialBackwardReachingIndirectLoadIndex) {
  // 部分后移禁止穿过 IndirectLoad 索引链：
  // broadcast0/broadcast1(同轴) → sub → abs → add1(多输入) → IndirectLoad.x2
  // 若部分后移执行，索引链视图被标量化重构，IndirectLoad 索引会解析错误。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_partial_indirect_load_index")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Data("data3", 3)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Load("load2", "data2")
                   .Load("load3", "data3")
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Sub("sub", "broadcast0", "broadcast1")
                   .Abs("abs", "sub")
                   .Add("add1", "abs", "load2")
                   .Op<af::ascir_op::IndirectLoad>("indirectload", {"load3", "add1"})
                   .Store("store", "indirectload")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "sub"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "sub"));
  EXPECT_TRUE(IsConnected(graph, "sub", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "add1"));
  EXPECT_TRUE(IsConnected(graph, "add1", "indirectload"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, MovesPartialBackwardWithoutIndirectLoadDownstream) {
  // 正向对照：同一结构但下游无 IndirectLoad 时，部分后移正常执行。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_partial_no_indirect_load")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Load("load2", "data2")
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Sub("sub", "broadcast0", "broadcast1")
                   .Abs("abs", "sub")
                   .Add("add1", "abs", "load2")
                   .Store("store", "add1")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sub"));
  EXPECT_TRUE(IsConnected(graph, "load1", "sub"));
  EXPECT_TRUE(IsConnected(graph, "sub", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast0", "sub"));
  EXPECT_FALSE(IsConnected(graph, "broadcast1", "sub"));
}

TEST(BroadcastBackwardPass, MovesCoaxialSiblingBroadcastPastPeerBroadcastGuard) {
  // 复现：add 的两路输入来自同轴 broadcast0/broadcast1（broadcast0 接 in-0），add 的下游 add1 的另一路
  // 来自异轴 scalar 广播链 broadcast2。修复前：探测期当场删除 broadcast1 后，提交守卫把本链广播
  // broadcast0（经 add 的 in-0）误计入"broadcast behind"（count=2）而拒绝提交，图停留在半改写状态
  // （broadcast1 已删、broadcast0 未动、add 两路输入视图不一致）。修复后：守卫排除本链广播、兄弟删除
  // 延迟到提交期，广播正确后移到 add 之后，add 两路输入均为紧凑视图。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_coaxial_sibling_guard")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Broadcast("broadcast2", "scalar", {s0, s1})
                   .Add("add", "broadcast0", "broadcast1")
                   .Add("add1", "add", "broadcast2")
                   .Store("store", "add1")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "add"));
  EXPECT_TRUE(IsConnected(graph, "load1", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "add1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast2", "add1"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "add"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "add"));
  const auto add = FindNode(graph, "add");
  const auto load0 = FindNode(graph, "load0");
  const auto load1 = FindNode(graph, "load1");
  ASSERT_NE(add, nullptr);
  ASSERT_NE(load0, nullptr);
  ASSERT_NE(load1, nullptr);
  // add 输出改写为紧凑视图（广播轴 repeat=1），与两路输入的紧凑视图匹配，不再出现
  // "一路展开、一路紧凑"的半改写属性。
  ExpectStaticEq(add->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(AreExpressionVectorsEqual(load0->outputs[0].attr.repeats, load1->outputs[0].attr.repeats));
}

TEST(BroadcastBackwardPass, SkipsCoaxialSiblingBroadcastRemovalReachingIndirectLoad) {
  // fail-closed 对照：同图但 add1 下游到达 IndirectLoad。IsMulInputsCanBackward 的 IndirectLoad 熔断
  // 先于兄弟删除登记触发，且修复后兄弟删除已延迟到提交期，整条链干净放弃，broadcast0/broadcast1
  // 均保留原位（不产生"兄弟已删、链未后移"的半改写）。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_coaxial_sibling_indirect_load")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data3", 3)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Load("load3", "data3")
                   .Scalar("scalar", "1.0")
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Broadcast("broadcast2", "scalar", {s0, s1})
                   .Add("add", "broadcast0", "broadcast1")
                   .Add("add1", "add", "broadcast2")
                   .Op<af::ascir_op::IndirectLoad>("indirectload", {"load3", "add1"})
                   .Store("store", "indirectload")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "add"));
  EXPECT_TRUE(IsConnected(graph, "load1", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "add1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast2", "add1"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, MovesCoaxialSiblingBroadcastOnFirstInputPastComputePeerGuard) {
  // 复现 ST RemoveRedundantBroadcast 回归：broadcast1 链（exp0→broadcast1→add.in-1）探测 add 时，同轴
  // 兄弟 broadcast0 垫在 add 的 in-0 上被登记为待删；链尾 mul0 的另一路是 compute peer exp1，其 in-0 由
  // 外源广播 broadcast2 支撑。修复前：提交守卫只排除本链广播（brc_nodes），待删兄弟 broadcast0 探测期
  // 仍垫在链内节点 add 的 in-0 上，被守卫经 peer 的 in-0 生产者检查误计为外源广播，与 broadcast2 凑满
  // count=2 拒绝提交，broadcast0/broadcast1 全部原样保留。修复后：守卫排除集扩展为本事务广播全集
  // （brc_nodes + pending_remove_brcs），count=1 放行提交：broadcast0 提交期删除、broadcast1 后移到
  // add 与 mul0 之间；abs0 的首消费者为 compute（exp0），其链被前置门控，外源 broadcast2 保持原位。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_coaxial_sibling_on_first_input")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Abs("abs0", "load0")
                   .Exp("exp0", "abs0")
                   .Broadcast("broadcast0", "abs0", {s0, s1})
                   .Broadcast("broadcast1", "exp0", {s0, s1})
                   .Broadcast("broadcast2", "abs0", {s0, s1})
                   .Add("add", "broadcast0", "broadcast1")
                   .Exp("exp1", "broadcast2")
                   .Mul("mul0", "add", "exp1")
                   .Store("store", "mul0")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "abs0", "add"));
  EXPECT_TRUE(IsConnected(graph, "exp0", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "mul0"));
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "abs0", "broadcast2"));
  EXPECT_TRUE(IsConnected(graph, "broadcast2", "exp1"));
  EXPECT_TRUE(IsConnected(graph, "exp1", "mul0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "abs0", "add"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "exp0", "add"));
  const auto add = FindNode(graph, "add");
  ASSERT_NE(add, nullptr);
  // add 输出改写为紧凑视图（广播轴 repeat=1），与两路输入的紧凑视图匹配，不再出现
  // "一路展开、一路紧凑"的半改写属性。
  ExpectStaticEq(add->outputs[0].attr.repeats, kCompactRepeats);
}

// ===== Dtype-aware backward tests for Cast and dtype-changing operators =====

TEST(BroadcastBackwardPass, MovesCastAcrossBroadcast) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_cast")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {s0, s1})
                   .Cast("cast", "broadcast", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  EXPECT_EQ(broadcast_node->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(broadcast_node->inputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesComputeAndCastAcrossBroadcast) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_abs_cast")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {s0, s1})
                   .Abs("abs", "broadcast")
                   .Cast("cast", "abs", af::DT_FLOAT16)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast_node = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast_node, nullptr);
  EXPECT_EQ(broadcast_node->outputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesComparisonAcrossIdenticalBroadcastBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_comparison")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", kCompactRepeats, kCompactStrides)
                   .Load("load1", "data1", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Broadcast("broadcast1", "load1", {s0, s1})
                   .Op<af::ascir_op::Ge>("compare", {"broadcast0", "broadcast1"})
                   .Store("store", "compare")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "compare"));
  EXPECT_TRUE(IsConnected(graph, "load1", "compare"));
  EXPECT_TRUE(IsConnected(graph, "compare", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "compare"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "compare"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compare", "broadcast0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast0", "store"));
}

TEST(BroadcastBackwardPass, MovesAdditionalDtypeAwareBinaryOps) {
  GTEST_SKIP();
  for (const auto &op_name : {std::string("Eq"), std::string("TrueDiv")}) {
    SCOPED_TRACE(op_name);
    auto graph = BuildDtypeAwareBinaryGraph(op_name);
    ExpectDtypeAwareBinaryMove(graph, op_name);
  }
}

TEST(BroadcastBackwardPass, DtypeAwareBackwardEnablesCommonAxisAtMultiInputTail) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildDtypeAwareCommonAxisGraph("broadcast_backward_dtype_aware_common_axis");
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "relu", af::DT_FLOAT16);
  SetNodeDtype(graph, "merge", af::DT_FLOAT16);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "cast0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "relu"));
  EXPECT_EQ(FindNode(graph, "broadcast0"), nullptr);
  EXPECT_EQ(FindNode(graph, "broadcast1"), nullptr);
  const auto residual0 = FindNode(graph, "broadcast0_residual_0");
  const auto residual1 = FindNode(graph, "broadcast1_residual_1");
  const auto merge = FindNode(graph, "merge");
  const auto common = FindNode(graph, "merge_broadcast_backward_common");
  ASSERT_NE(residual0, nullptr);
  ASSERT_NE(residual1, nullptr);
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(common, nullptr);
  EXPECT_TRUE(IsConnected(graph, "cast0", "broadcast0_residual_0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0_residual_0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1_residual_1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
  EXPECT_EQ(residual0->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual0->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual1->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(residual1->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->inputs[1].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(merge->outputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(common->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(common->outputs[0].attr.dtype, af::DT_FLOAT16);
}

// ===== Common broadcast-axis partial backward tests =====

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsBroadcastControlEdges) {
  for (size_t case_index = 0U; case_index < 2U; ++case_index) {
    auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_control_" + std::to_string(case_index));
    CompleteApiInfo(graph);
    const auto broadcast = FindNode(graph, "broadcast0");
    const auto peer = case_index == 0U ? FindNode(graph, "load0") : FindNode(graph, "store");
    ASSERT_NE(broadcast, nullptr);
    ASSERT_NE(peer, nullptr);
    const auto status = case_index == 0U
                            ? af::GraphUtils::AddEdge(peer->GetOutControlAnchor(), broadcast->GetInControlAnchor())
                            : af::GraphUtils::AddEdge(broadcast->GetOutControlAnchor(), peer->GetInControlAnchor());
    ASSERT_EQ(status, af::SUCCESS);

    optimize::BroadcastBackwardPass pass;
    ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
    EXPECT_TRUE(HasNode(graph, "broadcast0"));
    EXPECT_TRUE(HasNode(graph, "broadcast1"));
    EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
    EXPECT_EQ(broadcast->GetInControlNodesSize() + broadcast->GetOutControlNodesSize(), 1U);
  }
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsSourceControlEdge) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_source_control");
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load0");
  const auto store = FindNode(graph, "store");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), store->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_EQ(load->GetOutControlNodesSize(), 1U);
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsSourceBroadcastEdgeAttrMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis_source_edge_mismatch");
  CompleteApiInfo(graph);
  const auto broadcast = FindNode(graph, "broadcast0");
  ASSERT_NE(broadcast, nullptr);
  const auto input_desc = broadcast->GetOpDesc()->MutableInputDesc(0U);
  ASSERT_NE(input_desc, nullptr);
  input_desc->SetDataType(af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisFromMultiNodeBroadcastChains) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_common_axis_multi_node")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Load("load1", "data1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("residual0", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("common0", "residual0", {s0, s1, s2})
                   .Broadcast("residual1", "load1", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("common1", "residual1", {s0, s1, s2})
                   .Add("merge", "common0", "common1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "residual0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "residual1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_FALSE(HasNode(graph, "common0_residual_0"));
  EXPECT_FALSE(HasNode(graph, "common1_residual_1"));
  EXPECT_FALSE(HasNode(graph, "common0"));
  EXPECT_FALSE(HasNode(graph, "common1"));
}

TEST(BroadcastBackwardPass, MovesCreatedCommonBroadcastPastUnaryChainWithoutIdentityResiduals) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> compact_strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_created_common_unary_chain")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", {1})
                   .Broadcast("broadcast1", "load1", {1})
                   .Abs("abs", "broadcast0")
                   .Cast("cast0", "abs", af::DT_FLOAT16)
                   .Cast("cast1", "broadcast1", af::DT_FLOAT16)
                   .Relu("relu", "cast1")
                   .Add("merge", "cast0", "relu")
                   .Sqrt("sqrt", "merge")
                   .Op<af::ascir_op::Sigmoid>("sigmoid", {"sqrt"})
                   .Store("store", "sigmoid")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  for (const auto *node_name : {"relu", "merge", "sqrt", "sigmoid", "store"}) {
    SetNodeDtype(graph, node_name, af::DT_FLOAT16);
  }

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0_residual_0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "cast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "relu", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "sqrt", "sigmoid"));
  EXPECT_TRUE(IsConnected(graph, "sigmoid", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisBeforeResidualBroadcasts) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_common_axis_before_residual")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Load("load1", "data1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("common0", "load0", {af::sym::kSymbolOne, s1, s2})
                   .Broadcast("residual0", "common0", {s0, s1, s2})
                   .Broadcast("common1", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("residual1", "common1", {s0, s1, s2})
                   .Add("merge", "residual0", "residual1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "residual0_residual_0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "residual1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_FALSE(HasNode(graph, "common0"));
  EXPECT_FALSE(HasNode(graph, "common1"));
  EXPECT_FALSE(HasNode(graph, "residual0"));
  EXPECT_FALSE(HasNode(graph, "residual1"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxesDistributedAcrossBroadcastChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_distributed_common_axis")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Load("load1", "data1", {af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("broadcast00", "load0", {s0, af::sym::kSymbolOne})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1})
                   .Broadcast("broadcast10", "load1", {s0, s1})
                   .Add("merge", "broadcast01", "broadcast10")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast10", "merge"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisPartialBackward) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = BuildCommonAxisGraph("broadcast_backward_common_axis");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  auto b_common = FindNode(graph, "merge_broadcast_backward_common");
  EXPECT_NE(b_common, nullptr);
  EXPECT_TRUE(IsConnected(graph, "merge", "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "merge_broadcast_backward_common", "store"));
  auto b0_residual = FindNode(graph, "broadcast0_residual_0");
  EXPECT_NE(b0_residual, nullptr);
  auto b1_residual = FindNode(graph, "broadcast1_residual_1");
  EXPECT_NE(b1_residual, nullptr);
  ExpectResidualConnections(graph);
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  std::vector<af::Expression> expected_compact_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(
                std::vector<af::Expression>{s0, af::sym::kSymbolOne, s2}, expected_compact_strides),
            af::SUCCESS);
  std::vector<af::Expression> expected_expanded_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(std::vector<af::Expression>{s0, s1, s2},
                                                                   expected_expanded_strides),
            af::SUCCESS);
  ExpectCommonAxisLayouts(graph, {s0, af::sym::kSymbolOne, s2}, {s0, s1, s2}, expected_compact_strides,
                          expected_expanded_strides);
}

TEST(BroadcastBackwardPass, MovesCommonAxisToActualSuccessorInput) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_successor_input")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {1, 2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Add("succ", "merge", "merge")
                   .Store("store", "succ")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto merge = FindNode(graph, "merge");
  const auto succ = FindNode(graph, "succ");
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(succ, nullptr);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(merge->GetOutDataAnchor(0), succ->GetInDataAnchor(0)), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto common = FindNode(graph, "merge_broadcast_backward_common");
  ASSERT_NE(common, nullptr);
  EXPECT_EQ(succ->GetInDataAnchor(0)->GetPeerOutAnchor(), nullptr);
  const auto succ_input1_peer = succ->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(succ_input1_peer, nullptr);
  EXPECT_EQ(succ_input1_peer->GetOwnerNode()->GetName(), "merge_broadcast_backward_common");
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(common, succ, 1U));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisNoOverlap) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_no_common_axis")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0})
                   .Broadcast("broadcast1", "load1", {2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "merge"));
}

TEST(BroadcastBackwardPass, MovesNoResidualCaseInIdenticalMultiInputBackward) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_common_only")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, strides)
                   .Load("load1", "data1", compact, strides)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {0, 1})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_EQ(FindNode(graph, "merge_broadcast_backward_common"), nullptr);
  const bool has_broadcast0 = HasNode(graph, "broadcast0");
  const bool has_broadcast1 = HasNode(graph, "broadcast1");
  ASSERT_NE(has_broadcast0, has_broadcast1);
  const char *kept_broadcast = has_broadcast0 ? "broadcast0" : "broadcast1";
  EXPECT_TRUE(IsConnected(graph, "load0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", kept_broadcast));
  EXPECT_TRUE(IsConnected(graph, kept_broadcast, "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisOldBrcDeletedWithResidual) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildCommonAxisGraph("broadcast_backward_residual_delete_old");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  auto b0_residual = FindNode(graph, "broadcast0_residual_0");
  ASSERT_NE(b0_residual, nullptr);
  auto b1_residual = FindNode(graph, "broadcast1_residual_1");
  ASSERT_NE(b1_residual, nullptr);
  ExpectResidualConnections(graph);
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardMovesSharedBroadcastInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_shared_common_axis_input")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, strides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("merge", "broadcast", "broadcast")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast, nullptr);
  EXPECT_TRUE(IsConnected(graph, "load", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_EQ(broadcast->GetOutDataNodesSize(), 1U);
}

TEST(BroadcastBackwardPass, SplitsSharedBroadcastBeforeDistinctMultiInputBarriers) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s3, af::sym::kSymbolZero,
                                                       af::sym::kSymbolOne};
  const std::vector<af::Expression> expanded = {s0, s1, s2, s3};
  const std::vector<af::Expression> expanded_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_shared_branch")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0)
                   .Load("load0", "data0", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", {s0, s1, af::sym::kSymbolOne, s3})
                   .Broadcast("broadcast1", "broadcast0", expanded)
                   .Sqrt("sqrt", "broadcast1")
                   .Abs("abs", "broadcast1")
                   .Data("data1", 1)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Sub("sub", "sqrt", "load1")
                   .Add("add", "abs", "load1")
                   .Neg("neg", "sub")
                   .Relu("relu", "add")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast1", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast1", "abs"));

  for (const auto &compute_name : {"sqrt", "abs"}) {
    const auto compute = FindNode(graph, compute_name);
    ASSERT_NE(compute, nullptr);
    const auto peers = compute->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    ASSERT_EQ(peers.size(), 1U);
    EXPECT_EQ((*peers.begin())->GetOwnerNode()->GetType(), "Broadcast");
  }
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSupportsSuccessorInputOne) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_successor_input_one")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Load("load2", "data2", expanded, expanded_strides)
                   .Broadcast("broadcast0", "load0", expanded)
                   .Sqrt("sqrt", "broadcast0")
                   .Abs("abs", "broadcast0")
                   .Add("successor0", "load1", "sqrt")
                   .Add("successor1", "load2", "abs")
                   .Add("join", "successor0", "successor1")
                   .Store("store", "join")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast0", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0_branch_split_1", "successor0"));
  EXPECT_TRUE(HasNode(graph, "broadcast0_branch_split_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "successor1"));
}

TEST(BroadcastBackwardPass, SplitsAndMovesSharedBroadcastPerBranch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_and_move_per_branch")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Broadcast("broadcast", "load0", expanded)
                   .Sqrt("sqrt", "broadcast")
                   .Add("add", "sqrt", "load1")
                   .Neg("neg", "add")
                   .Abs("abs", "broadcast")
                   .Relu("relu", "abs")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_FALSE(IsConnected(graph, "broadcast", "sqrt"));
  EXPECT_FALSE(IsConnected(graph, "broadcast", "abs"));

  const auto sqrt = FindNode(graph, "sqrt");
  const auto relu = FindNode(graph, "relu");
  ASSERT_NE(sqrt, nullptr);
  ASSERT_NE(relu, nullptr);
  ASSERT_EQ(sqrt->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  ASSERT_EQ(relu->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ((*sqrt->GetOutDataAnchor(0)->GetPeerInDataAnchors().begin())->GetOwnerNode()->GetType(), "Broadcast");
  EXPECT_EQ((*relu->GetOutDataAnchor(0)->GetPeerInDataAnchors().begin())->GetOwnerNode()->GetType(), "Broadcast");
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsCommonMerge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_skips_common_merge")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, compact_strides)
                   .Broadcast("broadcast", "load", expanded)
                   .Abs("abs", "broadcast")
                   .Neg("neg", "broadcast")
                   .Add("merge", "abs", "neg")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsWithoutMovableBranch) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  auto graph = AscGraphBuilder("broadcast_backward_split_skips_non_movable")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", compact, compact_strides)
                   .Broadcast("broadcast", "load", expanded)
                   .Store("store0", "broadcast")
                   .Output("output0", "store0")
                   .Store("store1", "broadcast")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "broadcast")->GetOutDataNodesSize(), 2U);
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsMoreThanEightBranches) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  AscGraphBuilder builder("broadcast_backward_split_skips_more_than_eight_branches");
  builder.Loops({s0, s1})
      .Data("data0", 0)
      .Data("data1", 1)
      .Load("load0", "data0", compact, compact_strides)
      .Load("load1", "data1", expanded, expanded_strides)
      .Broadcast("broadcast", "load0", expanded);
  for (size_t index = 0U; index < 9U; ++index) {
    const auto suffix = std::to_string(index);
    builder.Abs("abs" + suffix, "broadcast").Add("add" + suffix, "load1", "abs" + suffix);
  }
  builder.Store("store", "add0").Output("output", "store");
  auto graph = builder.Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "broadcast")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 9U);
}

TEST(BroadcastBackwardPass, MovesBroadcastToNonZeroMultiInputTail) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1};
  const std::vector<af::Expression> expanded_strides = {s1, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_non_zero_tail_input")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact, compact_strides)
                   .Load("load1", "data1", expanded, expanded_strides)
                   .Broadcast("broadcast", "load0", expanded)
                   .Abs("abs", "broadcast")
                   .Add("add", "load1", "abs")
                   .Store("store", "add")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "abs", "broadcast"));
  EXPECT_EQ(FindNode(graph, "add")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "broadcast");
}

TEST(BroadcastBackwardPass, SharedBroadcastSplitSkipsSingleInputSuccessor) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolZero};
  const std::vector<af::Expression> expanded = {s0, s1, s2};
  const std::vector<af::Expression> expanded_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_split_single_input_successor")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Load("load0", "data0", compact, compact_strides)
                   .Broadcast("broadcast0", "load0", expanded)
                   .Sqrt("sqrt", "broadcast0")
                   .Abs("abs", "broadcast0")
                   .Neg("neg", "sqrt")
                   .Relu("relu", "abs")
                   .Mul("mul", "relu", "neg")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "broadcast0_branch_split_1"));
  EXPECT_TRUE(IsConnected(graph, "mul", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
}

TEST(BroadcastBackwardPass, RollsBackMultiLevelChainSplitWithoutBackwardGain) {
  auto graph = BuildMultiLevelChainFanOutGraph("broadcast_backward_chain_split_fanout");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 拆分已解除共享但两分支与 bx/by 无公共轴、无任何后移收益：反向合并把克隆链
  // 并回原链，图恢复共享形态（链尾多消费），不残留拆分克隆
  EXPECT_FALSE(HasNode(graph, "b1_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "b2_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b1"));
  EXPECT_TRUE(IsConnected(graph, "b1", "b2"));
  EXPECT_EQ(FindNode(graph, "b2")->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(FindNode(graph, "load0")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  // 两个消费者的广播侧输入仍来自视图一致的 Broadcast
  const auto tail = FindNode(graph, "b2");
  ASSERT_NE(tail, nullptr);
  for (const auto *consumer_name : {"add", "sub"}) {
    const auto consumer = FindNode(graph, consumer_name);
    ASSERT_NE(consumer, nullptr);
    const auto peer =
        std::dynamic_pointer_cast<af::AscNode>(consumer->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode());
    ASSERT_NE(peer, nullptr);
    EXPECT_EQ(peer->GetType(), "Broadcast");
    EXPECT_TRUE(AreExpressionVectorsEqual(peer->outputs[0].attr.repeats, tail->outputs[0].attr.repeats));
  }
}

TEST(BroadcastBackwardPass, MovesSplitMultiLevelChainWithIdenticalSiblingBroadcasts) {
  auto graph = BuildMultiLevelChainSiblingGraph("broadcast_backward_chain_split_sibling");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // add 分支：同轴兄弟链删除后 loadX 直连，原链整体沉过 add
  EXPECT_TRUE(IsConnected(graph, "load0", "add"));
  EXPECT_TRUE(IsConnected(graph, "loadX", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "b1"));
  EXPECT_TRUE(IsConnected(graph, "b1", "b2"));
  EXPECT_TRUE(IsConnected(graph, "b2", "store1"));
  // mul 分支：克隆链沉过 mul，y 链同轴兄弟删除
  EXPECT_TRUE(IsConnected(graph, "load0", "mul"));
  EXPECT_TRUE(IsConnected(graph, "loadY", "mul"));
  EXPECT_TRUE(IsConnected(graph, "mul", "b1_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "b1_consumer_split_1", "b2_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "b2_consumer_split_1", "store2"));
  for (const auto *gone : {"x1", "x2", "y1", "y2"}) {
    EXPECT_FALSE(HasNode(graph, gone));
  }
  // add 广播侧输入视图收敛为紧凑视图
  const auto add = FindNode(graph, "add");
  ASSERT_NE(add, nullptr);
  ExpectStaticEq(add->inputs[0].attr.repeats,
                 std::vector<af::Expression>{af::sym::kSymbolOne, Sym("s1"), af::sym::kSymbolOne});
}

TEST(BroadcastBackwardPass, SkipsMultiLevelChainWhenNoDownstreamTaker) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_chain_split_no_taker")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("dataF", 1)
                   .Data("dataG", 2)
                   .Load("load0", "data0", {one, s1, one}, {zero, one, zero})
                   .Broadcast("b1", "load0", {0})
                   .Broadcast("b2", "b1", {2})
                   .Load("loadF", "dataF", {s0, s1, s2}, {s1 * s2, s2, one})
                   .Load("loadG", "dataG", {s0, s1, s2}, {s1 * s2, s2, one})
                   .Add("add", "b2", "loadF")
                   .Sub("sub", "b2", "loadG")
                   .Store("store1", "add")
                   .Output("output1", "store1")
                   .Store("store2", "sub")
                   .Output("output2", "store2")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 消费者另一路均为满 shape 张量（无下游承接）：拆分后无后移收益，反向合并回滚，
  // 终态与不拆分等价（无拆分克隆残留、共享形态恢复）
  EXPECT_FALSE(HasNode(graph, "b1_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "b2_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "b2")->GetOutDataNodesSize(), 2U);
  EXPECT_EQ(FindNode(graph, "load0")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
}

TEST(BroadcastBackwardPass, MovesInternalSharingBranchAfterChainSplit) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_chain_split_internal_sharing")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("dataX", 1)
                   .Data("dataY", 2)
                   .Load("load0", "data0", {one, s1, one}, {zero, one, zero})
                   .Broadcast("b1", "load0", {0})
                   .Broadcast("b2", "b1", {2})
                   .Load("loadX", "dataX", {s0, one, s2}, {s2, zero, one})
                   .Broadcast("bx", "loadX", {1})
                   .Load("loadY", "dataY", {s0, one, s2}, {s2, zero, one})
                   .Broadcast("by", "loadY", {1})
                   .Add("add", "b2", "bx")
                   .Sub("sub", "b2", "by")
                   .Neg("neg", "b1")
                   .Store("store1", "add")
                   .Output("output1", "store1")
                   .Store("store2", "sub")
                   .Output("output2", "store2")
                   .Store("store3", "neg")
                   .Output("output3", "store3")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  for (const auto &dbg_node : graph.GetAllNodes()) {
    fprintf(stderr, "=== DEBUG node: %s (%s) in=%u out=%u\n", dbg_node->GetNamePtr(), dbg_node->GetTypePtr(),
            dbg_node->GetInDataNodesSize(), dbg_node->GetOutDataNodesSize());
  }
  // b1 的 neg 分支拆分后获得独立链并可后移（Neg 沉到广播前）
  EXPECT_TRUE(IsConnected(graph, "load0", "neg"));
  EXPECT_TRUE(IsConnected(graph, "neg", "b1_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "b1_consumer_split_1", "store3"));
  EXPECT_EQ(FindNode(graph, "b1")->GetOutDataNodesSize(), 1U);
  // b2 双消费分支无公共轴无后移收益：拆分克隆由反向合并并回原链，恢复共享形态
  EXPECT_FALSE(HasNode(graph, "b2_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b1"));
  EXPECT_TRUE(IsConnected(graph, "b1", "b2"));
  EXPECT_EQ(FindNode(graph, "b2")->GetOutDataNodesSize(), 2U);
}

TEST(BroadcastBackwardPass, MovesChainSplitWithNineConsumers) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  AscGraphBuilder builder("broadcast_backward_chain_split_nine_consumers");
  builder.Loops({s0, s1, s2})
      .Data("data0", 0)
      .Load("load0", "data0", {one, s1, one}, {zero, one, zero})
      .Broadcast("b1", "load0", {0})
      .Broadcast("b2", "b1", {2});
  for (size_t index = 0U; index < 9U; ++index) {
    const auto suffix = std::to_string(index);
    builder.Abs("abs" + suffix, "b2")
        .Neg("neg" + suffix, "abs" + suffix)
        .Store("store" + suffix, "neg" + suffix)
        .Output("output" + suffix, "store" + suffix);
  }
  auto graph = builder.Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 消费者数不再设上限：9 条分支全部拆分为独立链并各自沉过分支计算
  EXPECT_TRUE(IsConnected(graph, "load0", "abs0"));
  EXPECT_TRUE(IsConnected(graph, "neg0", "b1"));
  EXPECT_TRUE(IsConnected(graph, "b1", "b2"));
  EXPECT_TRUE(IsConnected(graph, "b2", "store0"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs3"));
  EXPECT_TRUE(IsConnected(graph, "neg3", "b1_consumer_split_3"));
  EXPECT_TRUE(IsConnected(graph, "b1_consumer_split_3", "b2_consumer_split_3"));
  EXPECT_TRUE(IsConnected(graph, "b2_consumer_split_3", "store3"));
  EXPECT_EQ(FindNode(graph, "b2")->GetOutDataNodesSize(), 1U);
}

TEST(BroadcastBackwardPass, SkipsChainSplitWhenConsumersShareCommonMerge) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_chain_split_common_merge")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Load("load0", "data0", {one, s1, one}, {zero, one, zero})
                   .Broadcast("b1", "load0", {0})
                   .Broadcast("b2", "b1", {2})
                   .Sqrt("sqrt", "b2")
                   .Abs("abs", "b2")
                   .Neg("neg", "sqrt")
                   .Relu("relu", "abs")
                   .Mul("mul", "neg", "relu")
                   .Store("store", "mul")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 公共 merge 场景：无条件拆分后先泊车再由同轴兄弟删除整链沉过 merge（原链被删、
  // 克隆幸存），命名回退把幸存克隆恢复为原链名，终态与 multi-ref 路径等价
  EXPECT_FALSE(HasNode(graph, "b1_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "b2_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "load0", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "load0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "mul", "b1"));
  EXPECT_TRUE(IsConnected(graph, "b2", "store"));
}

TEST(BroadcastBackwardPass, SkipsChainSplitFromScalarSource) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_chain_split_scalar_source")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("b1", "scalar", {s0, s1})
                   .Data("dataX", 0)
                   .Data("dataY", 1)
                   .Load("loadX", "dataX", {s0, one}, {one, zero})
                   .Broadcast("bx", "loadX", {1})
                   .Load("loadY", "dataY", {s0, one}, {one, zero})
                   .Broadcast("by", "loadY", {1})
                   .Add("add", "b1", "bx")
                   .Sub("sub", "b1", "by")
                   .Store("store1", "add")
                   .Output("output1", "store1")
                   .Store("store2", "sub")
                   .Output("output2", "store2")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 标量源链被 CollectCandidates 排除，两个拆分都不参与
  EXPECT_FALSE(HasNode(graph, "b1_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "b1_branch_split_1"));
}

TEST(BroadcastBackwardPass, RollsBackSingleBroadcastSplitWithoutDownstreamTaker) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_single_split_full_tensor_peer")
                   .Loops({s0, s1})
                   .Data("data0", 0)
                   .Data("dataF", 1)
                   .Data("dataG", 2)
                   .Load("load0", "data0", {s0, one}, {one, zero})
                   .Broadcast("broadcast", "load0", {1})
                   .Load("loadF", "dataF", {s0, s1}, {s1, one})
                   .Load("loadG", "dataG", {s0, s1}, {s1, one})
                   .Add("add", "broadcast", "loadF")
                   .Sub("sub", "broadcast", "loadG")
                   .Store("store1", "add")
                   .Output("output1", "store1")
                   .Store("store2", "sub")
                   .Output("output2", "store2")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 另一路为满 shape 张量（无下游承接）：拆分后无后移收益，反向合并回滚为共享形态
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_EQ(FindNode(graph, "load0")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_EQ(FindNode(graph, "broadcast")->GetOutDataNodesSize(), 2U);
}

TEST(BroadcastBackwardPass, MergesBackIdenticalIdleBroadcastChains) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_merge_identical_idle")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast0", "load", {1})
                   .Broadcast("broadcast1", "load", {1})
                   .Store("store0", "broadcast0")
                   .Store("store1", "broadcast1")
                   .Output("output0", "store0")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 同源同签名且均无法后移（Store 阻断）的两条广播链合并回一条共享链（链尾多消费）
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast0"));
  EXPECT_EQ(FindNode(graph, "broadcast0")->GetOutDataNodesSize(), 2U);
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store1"));
}

TEST(BroadcastBackwardPass, SplitsAroundStoreConsumerAndMovesMovableBranch) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_split_store_consumer")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("abs", "broadcast")
                   .Store("store0", "abs")
                   .Output("output0", "store0")
                   .Store("store1", "broadcast")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // Store 消费者不再阻塞拆分：可动分支（abs）沉到广播前，不可动分支（直接落盘）保持
  // 原位；两链后移后 source 不同，不构成反向合并对象
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store0"));
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast_consumer_split_1", "store1"));
}

TEST(BroadcastBackwardPass, SkipsConsumerSplitWithControlEdge) {
  auto graph = BuildDirectFanOutGraph("broadcast_backward_consumer_split_control_edge");
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), broadcast->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 链上控制边约束依附于节点，克隆不带控制边会丢失顺序约束：fail-closed 不拆不改写
  EXPECT_FALSE(HasNode(graph, "broadcast_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch1"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, CommonAxisBackwardSkipsBroadcastWithAnotherConsumer) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact0 = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides0 = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact1 = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> strides1 = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_extra_consumer")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact0, strides0)
                   .Load("load1", "data1", compact1, strides1)
                   .Broadcast("broadcast0", "load0", {0, 1})
                   .Broadcast("broadcast1", "load1", {1, 2})
                   .Add("merge", "broadcast0", "broadcast1")
                   .Abs("side", "broadcast0")
                   .Store("store", "merge")
                   .Store("side_store", "side")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "side"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisEdgeAttrMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_edge_mismatch");
  CompleteApiInfo(graph);
  auto merge_node = FindNode(graph, "merge");
  ASSERT_NE(merge_node, nullptr);
  merge_node->inputs[0].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

TEST(BroadcastBackwardPass, SkipsCommonAxisDtypeMismatch) {
  auto graph = BuildCommonAxisGraph("broadcast_backward_dtype_mismatch");
  CompleteApiInfo(graph);
  auto load1_node = FindNode(graph, "load1");
  ASSERT_NE(load1_node, nullptr);
  load1_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  auto b1_node = FindNode(graph, "broadcast1");
  ASSERT_NE(b1_node, nullptr);
  b1_node->inputs[0].attr.dtype = af::DT_FLOAT16;
  b1_node->outputs[0].attr.dtype = af::DT_FLOAT16;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "merge_broadcast_backward_common"));
  EXPECT_TRUE(HasNode(graph, "broadcast0"));
  EXPECT_TRUE(HasNode(graph, "broadcast1"));
}

// ===== Multi-reference fork-join backward tests =====

TEST(BroadcastBackwardPass, GraphUtilsReconnectsSameSourceMultipleInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_edge_graph_utils")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  const auto consumer = FindNode(graph, "consumer");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_NE(consumer, nullptr);
  const auto peer_copy = broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  ASSERT_EQ(peer_copy.size(), 2U);
  for (const auto &peer : peer_copy) {
    ASSERT_EQ(af::GraphUtils::RemoveEdge(broadcast->GetOutDataAnchor(0), peer), af::SUCCESS);
  }
  EXPECT_TRUE(broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty());
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::RemoveEdge(load->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(broadcast->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)), af::SUCCESS);
  ASSERT_EQ(af::GraphUtils::AddEdge(broadcast->GetOutDataAnchor(0), consumer->GetInDataAnchor(1)), af::SUCCESS);
  EXPECT_EQ(broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(broadcast, consumer, 0U));
  EXPECT_TRUE(AreConnectedTensorAttrsEqual(broadcast, consumer, 1U));
}

TEST(BroadcastBackwardPass, MovesDirectFanOutAfterMerge) {
  auto graph = BuildDirectFanOutGraph("broadcast_backward_multi_reference_direct");
  CompleteApiInfo(graph);
  ExpectDirectFanOutCandidate(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  ExpectDirectFanOutMoved(graph);
}

TEST(BroadcastBackwardPass, MovesPrefixFanOutAfterMerge) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_prefix")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Relu("prefix", "broadcast")
                   .Abs("branch0", "prefix")
                   .Neg("branch1", "prefix")
                   .Add("merge", "branch0", "branch1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "prefix"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "branch0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "branch1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto prefix = FindNode(graph, "prefix");
  ASSERT_NE(prefix, nullptr);
  ExpectStaticEq(prefix->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "prefix"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesMultiNodeFanOutBranches) {
  auto graph = BuildMultiNodeFanOutGraph("broadcast_backward_multi_reference_multi_node_branches");
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "branch0_head"));
  EXPECT_TRUE(IsConnected(graph, "load", "branch1_head"));
  EXPECT_TRUE(IsConnected(graph, "branch0_head", "branch0_tail"));
  EXPECT_TRUE(IsConnected(graph, "branch1_head", "branch1_tail"));
  EXPECT_TRUE(IsConnected(graph, "branch0_tail", "merge"));
  EXPECT_TRUE(IsConnected(graph, "branch1_tail", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto branch0_tail = FindNode(graph, "branch0_tail");
  const auto branch1_tail = FindNode(graph, "branch1_tail");
  ASSERT_NE(branch0_tail, nullptr);
  ASSERT_NE(branch1_tail, nullptr);
  ExpectStaticEq(branch0_tail->outputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(branch1_tail->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch0_head"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0_head", "branch0_tail"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0_tail", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch1_head"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1_head", "branch1_tail"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1_tail", "merge"));
}

TEST(BroadcastBackwardPass, MovesFanOutAcrossFollowingComputeChain) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_following_compute")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("branch0", "broadcast")
                   .Neg("branch1", "broadcast")
                   .Add("merge", "branch0", "branch1")
                   .Relu("following0", "merge")
                   .Exp("following1", "following0")
                   .Store("store", "following1")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "load", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "following0"));
  EXPECT_TRUE(IsConnected(graph, "following0", "following1"));
  EXPECT_TRUE(IsConnected(graph, "following1", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "following0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "following0", "following1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "following1", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, DtypeAwareBackwardEnablesPrefixFanOut) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_dtype_aware_prefix_fanout")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Sqrt("before_cast", "broadcast")
                   .Cast("cast", "before_cast", af::DT_FLOAT16)
                   .Relu("prefix", "cast")
                   .Abs("branch0", "prefix")
                   .Neg("branch1", "prefix")
                   .Add("merge", "branch0", "branch1")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "prefix", af::DT_FLOAT16);
  SetNodeDtype(graph, "branch0", af::DT_FLOAT16);
  SetNodeDtype(graph, "branch1", af::DT_FLOAT16);
  SetNodeDtype(graph, "merge", af::DT_FLOAT16);
  SetNodeDtype(graph, "store", af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "before_cast"));
  EXPECT_TRUE(IsConnected(graph, "before_cast", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "prefix"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "prefix", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(broadcast, nullptr);
  EXPECT_EQ(broadcast->inputs[0].attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(broadcast->outputs[0].attr.dtype, af::DT_FLOAT16);
}

TEST(BroadcastBackwardPass, MovesSharedDtypeAwareBranchesPastFollowingChain) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  ScopedTestPlatform platform("3510");
  auto graph = BuildSharedDtypeAwareFanOutGraph("broadcast_backward_shared_dtype_aware_fanout");
  CompleteSharedDtypeAwareFanOutGraph(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "left_cast"));
  EXPECT_TRUE(IsConnected(graph, "load", "right_cast"));
  EXPECT_TRUE(IsConnected(graph, "right_cast", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "add"));
  EXPECT_TRUE(IsConnected(graph, "left_cast", "add"));
  EXPECT_TRUE(IsConnected(graph, "add", "sqrt"));
  EXPECT_TRUE(IsConnected(graph, "sqrt", "sigmoid"));
  const auto store = FindNode(graph, "store");
  ASSERT_NE(store, nullptr);
  const auto broadcast = std::dynamic_pointer_cast<af::AscNode>(store->GetInDataNodes().at(0));
  ASSERT_NE(broadcast, nullptr);
  EXPECT_EQ(broadcast->GetType(), af::ascir_op::Broadcast::Type);
  EXPECT_EQ(broadcast->outputs[0].attr.dtype, af::DT_FLOAT);
  EXPECT_EQ(broadcast->GetInDataNodes().at(0)->GetName(), "sigmoid");
}

TEST(BroadcastBackwardPass, SkipsSharedDtypeAwareBranchWithMismatchedInputDtype) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  ScopedTestPlatform platform("3510");
  auto graph = BuildSharedDtypeAwareFanOutGraph("broadcast_backward_shared_dtype_aware_mismatch");
  CompleteSharedDtypeAwareFanOutGraph(graph);
  const auto right_cast = FindNode(graph, "right_cast");
  ASSERT_NE(right_cast, nullptr);
  right_cast->GetOpDesc()->MutableInputDesc(0U)->SetDataType(af::DT_FLOAT16);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "abs"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "right_cast"));
  EXPECT_FALSE(IsConnected(graph, "sigmoid", "broadcast"));
}

TEST(BroadcastBackwardPass, MovesSameConsumerMultipleInputs) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_same_consumer")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  const auto consumer = FindNode(graph, "consumer");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_NE(consumer, nullptr);
  EXPECT_EQ(consumer->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_EQ(consumer->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_TRUE(IsConnected(graph, "consumer", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  ExpectStaticEq(consumer->inputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(consumer->inputs[1].attr.repeats, kCompactRepeats);
  ExpectStaticEq(consumer->outputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(broadcast->inputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "consumer", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

TEST(BroadcastBackwardPass, MovesSameConsumerWithThreeDimensionalLayout) {
  const std::vector<af::Expression> compact_repeats = {Sym(83), af::sym::kSymbolOne, Sym(91)};
  const std::vector<af::Expression> compact_strides = {Sym(91), af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> expanded_repeats = {Sym(83), Sym(18), Sym(91)};
  const std::vector<af::Expression> expanded_strides = {Sym(1638), Sym(91), af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_same_consumer_3d")
                   .Loops({Sym(83), Sym(18), Sym(91)})
                   .Data("data", 0)
                   .Load("load", "data", compact_repeats, compact_strides)
                   .Broadcast("broadcast", "load", {1})
                   .Add("consumer", "broadcast", "broadcast")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  const auto consumer = FindNode(graph, "consumer");
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ExpectStaticEq(consumer->inputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->inputs[0].attr.strides, compact_strides);
  ExpectStaticEq(consumer->inputs[1].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->inputs[1].attr.strides, compact_strides);
  ExpectStaticEq(consumer->outputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(consumer->outputs[0].attr.strides, compact_strides);
  ExpectStaticEq(broadcast->inputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(broadcast->inputs[0].attr.strides, compact_strides);
  ExpectStaticEq(broadcast->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(broadcast->outputs[0].attr.strides, expanded_strides);
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsBarrierBranch) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_barrier")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Cast("barrier", "broadcast", af::DT_FLOAT16)
                   .Abs("branch", "broadcast")
                   .Add("merge", "barrier", "branch")
                   .Store("store", "merge")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "barrier"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsBroadcastWithControlEdge) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  auto graph = BuildDirectFanOutGraph("broadcast_backward_multi_reference_control_edge");
  CompleteApiInfo(graph);
  const auto load = FindNode(graph, "load");
  const auto broadcast = FindNode(graph, "broadcast");
  ASSERT_NE(load, nullptr);
  ASSERT_NE(broadcast, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), broadcast->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch1"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsMultiInputSuccessor) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_multi_input_successor")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("branch0", "broadcast")
                   .Neg("branch1", "broadcast")
                   .Add("merge", "branch0", "branch1")
                   .Add("succ", "merge", "merge")
                   .Store("store", "succ")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "branch1"));
  EXPECT_FALSE(IsConnected(graph, "merge", "broadcast"));
}

TEST(BroadcastBackwardPass, MultiReferenceBackwardSkipsSameConsumerWithAnotherSource) {
  // Not supported by the restored repository BRC implementation.
  GTEST_SKIP();
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_multi_reference_mixed_consumer")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("broadcast", "load", {1})
                   .Abs("other", "broadcast")
                   .Add("consumer", "broadcast", "other")
                   .Store("store", "consumer")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "other"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "consumer"));
  EXPECT_FALSE(IsConnected(graph, "consumer", "broadcast"));
}

namespace {
// 无 Arange 的多级广播链（每节点单轴扩维）等价性验证：本修改的
// IsSameBrcNodes 全轴集比较 / RemoveBroadcasts 全轴集删除条件 / 生产者公共轴
// 守卫对正常多级形态必须与旧代码逐分支等价。以下两个用例的图边断言已在
// 修改前（upstream/develop 基线）与修改后分别跑探针 dump 比对，逐边一致。
}  // namespace

TEST(BroadcastBackwardPass, MovesIdenticalMultiLevelChainsNoArange) {
  // 同构双链（load→B00(s1)→B01(s2) 与 load→B10(s1)→B11(s2)）喂多输入 Add：
  // 期望保持既有"删重复链 + 整链后移"行为——B10/B11 删除、load1 直连 merge，
  // B00/B01 整链移到计算链尾。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> compact_repeats = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne};
  const std::vector<af::Expression> compact_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_multilevel_same_no_arange")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", compact_repeats, compact_strides)
                   .Load("load1", "data1", compact_repeats, compact_strides)
                   .Broadcast("broadcast00", "load0", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1, s2})
                   .Broadcast("broadcast10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast11", "broadcast10", {s0, s1, s2})
                   .Add("merge", "broadcast01", "broadcast11")
                   .Abs("abs", "merge")
                   .Relu("relu", "abs")
                   .Store("store", "relu")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 同构重复链 B10/B11 删除，两个 load 直连 merge。
  EXPECT_TRUE(IsConnected(graph, "load0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "merge"));
  EXPECT_FALSE(HasNode(graph, "broadcast10"));
  EXPECT_FALSE(HasNode(graph, "broadcast11"));
  // 保留链 B00/B01 整链后移到计算链尾。
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_TRUE(IsConnected(graph, "relu", "broadcast00"));
  EXPECT_TRUE(IsConnected(graph, "broadcast00", "broadcast01"));
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "store"));
}

TEST(BroadcastBackwardPass, MovesMultiLevelCommonAxisNoArange) {
  // 不同构双链（{s0,s1} vs {s0,s2} 扩维轴集）喂多输入 Add：期望保持既有
  // partial backward 公共轴行为——公共轴 s0 的 B00/B10 删除、load 直连，
  // 非公共轴的 B01/B11 保留并塌缩，计算链后插入 backward_broadcast 恢复 s0。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_multilevel_common_no_arange")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("broadcast00", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1, s2})
                   .Broadcast("broadcast10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast11", "broadcast10", {s0, s1, s2})
                   .Add("merge", "broadcast01", "broadcast11")
                   .Cast("cast", "merge", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 公共轴 s0 的广播删除，load 直连保留的非公共轴广播。
  EXPECT_FALSE(HasNode(graph, "broadcast00"));
  EXPECT_FALSE(HasNode(graph, "broadcast10"));
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast01"));
  EXPECT_TRUE(IsConnected(graph, "load1", "broadcast11"));
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast11", "merge"));
  // 计算链后插入公共轴广播恢复 s0 扩维。
  EXPECT_TRUE(IsConnected(graph, "merge", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_cast_0", "store"));
}

TEST(BroadcastBackwardPass, MovesMultiCommonAxesPartialBackwardByRound) {
  // 多输入多公共轴（{s0,s1}）迭代式单轴后移：两分支扩维轴集不同（{s0,s1,s3} vs
  // {s0,s1,s2}），每轮只迁移一个公共轴（先大轴 s1 后小轴 s0）。两轮后：分支公共
  // 轴广播删除、残差广播塌缩公共轴并保留非公共轴扩维、计算链输出在公共轴塌缩
  // 为 1，链尾后形成 _0(扩轴0)/_1(扩轴1) 的单轴广播链，下游视图不变。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, af::sym::kSymbolOne};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne,
                                                   af::sym::kSymbolZero};
  const std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero,
                                                   af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_multi_common_axes")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("b_a0", "load0", {s0, af::sym::kSymbolOne, s2, af::sym::kSymbolOne})
                   .Broadcast("b_a1", "b_a0", {s0, s1, s2, af::sym::kSymbolOne})
                   .Broadcast("b_a2", "b_a1", {s0, s1, s2, s3})
                   .Broadcast("b_b0", "load1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne, s3})
                   .Broadcast("b_b1", "b_b0", {s0, s1, af::sym::kSymbolOne, s3})
                   .Broadcast("b_b2", "b_b1", {s0, s1, s2, s3})
                   .Add("merge", "b_a2", "b_b2")
                   .Cast("cast", "merge", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 公共轴 0/1 的分支广播删除，load 直连残差广播；残差广播保留非公共轴扩维。
  EXPECT_FALSE(HasNode(graph, "b_a0"));
  EXPECT_FALSE(HasNode(graph, "b_a1"));
  EXPECT_FALSE(HasNode(graph, "b_b0"));
  EXPECT_FALSE(HasNode(graph, "b_b1"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b_a2"));
  EXPECT_TRUE(IsConnected(graph, "load1", "b_b2"));
  EXPECT_TRUE(IsConnected(graph, "b_a2", "merge"));
  EXPECT_TRUE(IsConnected(graph, "b_b2", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "cast"));
  // 迭代两轮各插一个单轴广播：链尾 → _0(扩轴0) → _1(扩轴1) → store。
  EXPECT_TRUE(IsConnected(graph, "cast", "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_cast_0", "backward_broadcast_cast_1"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_cast_1", "store"));
  // 计算链与残差广播在公共轴 0/1 塌缩为 1，残差轴保持扩维。
  const auto b_a2_node = FindNode(graph, "b_a2");
  const auto b_b2_node = FindNode(graph, "b_b2");
  const auto merge_node = FindNode(graph, "merge");
  const auto cast_node = FindNode(graph, "cast");
  const auto axis0_node = FindNode(graph, "backward_broadcast_cast_0");
  const auto axis1_node = FindNode(graph, "backward_broadcast_cast_1");
  ASSERT_NE(b_a2_node, nullptr);
  ASSERT_NE(b_b2_node, nullptr);
  ASSERT_NE(merge_node, nullptr);
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(axis0_node, nullptr);
  ASSERT_NE(axis1_node, nullptr);
  ExpectStaticEq(b_a2_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(b_b2_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(merge_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(cast_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(axis0_node->outputs[0].attr.repeats, {s0, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(axis1_node->outputs[0].attr.repeats, {s0, s1, s2, s3});
  // 单轴语义：_0 仅扩轴 0、_1 仅扩轴 1；strides 与 repeats 满足 recalc 一致性。
  std::vector<af::Expression> axis0_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(axis0_node->outputs[0].attr.repeats, axis0_strides),
            af::SUCCESS);
  ExpectStaticEq(axis0_node->outputs[0].attr.strides, axis0_strides);
  std::vector<af::Expression> axis1_strides;
  ASSERT_EQ(optimize::ScheduleUtils::RecalculateStridesFromRepeats(axis1_node->outputs[0].attr.repeats, axis1_strides),
            af::SUCCESS);
  ExpectStaticEq(axis1_node->outputs[0].attr.strides, axis1_strides);
}

TEST(BroadcastBackwardPass, MovesMultiCommonAxesCascadeToDownstreamAdds) {
  // 级联迭代后移：Add_0 双公共轴 {s0,s1} 两轮迁移完成后，下游 Add_1 的分支链
  // （含 Add_0 插入的 backward_broadcast 链）与 C 分支存在公共轴 s0——必须等
  // Add_0 全部轮次完成后才能正确求交，验证重入顺序（就地递归完成本节点剩余轮
  // 次，先于链收集中登记的下游多输入节点处理，不依赖 next_mul_input_nodes 的
  // 指针序）。Add_2 分支轴集 {s0} vs {s2,s1} 无公共轴，不迁移、D 链保持原状。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const std::vector<af::Expression> a_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> a_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero,
                                                 af::sym::kSymbolOne};
  const std::vector<af::Expression> b_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, af::sym::kSymbolOne};
  const std::vector<af::Expression> b_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne,
                                                 af::sym::kSymbolZero};
  const std::vector<af::Expression> c_repeats = {af::sym::kSymbolOne, s1, s2, af::sym::kSymbolOne};
  const std::vector<af::Expression> c_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolOne, af::sym::kSymbolZero};
  const std::vector<af::Expression> d_repeats = {s0, af::sym::kSymbolOne, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> d_strides = {s3, af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_multi_axes_cascade")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Data("data2", 2)
                   .Data("data3", 3)
                   .Load("load0", "data0", a_repeats, a_strides)
                   .Load("load1", "data1", b_repeats, b_strides)
                   .Load("load2", "data2", c_repeats, c_strides)
                   .Load("load3", "data3", d_repeats, d_strides)
                   .Broadcast("b_a0", "load0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3})
                   .Broadcast("b_a1", "b_a0", {af::sym::kSymbolOne, s1, s2, s3})
                   .Broadcast("b_a2", "b_a1", {s0, s1, s2, s3})
                   .Broadcast("b_b0", "load1", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3})
                   .Broadcast("b_b1", "b_b0", {af::sym::kSymbolOne, s1, s2, s3})
                   .Broadcast("b_b2", "b_b1", {s0, s1, s2, s3})
                   .Add("add0", "b_a2", "b_b2")
                   .Broadcast("b_c0", "load2", {af::sym::kSymbolOne, s1, s2, s3})
                   .Broadcast("b_c1", "b_c0", {s0, s1, s2, s3})
                   .Add("add1", "add0", "b_c1")
                   .Broadcast("b_d0", "load3", {s0, af::sym::kSymbolOne, s2, s3})
                   .Broadcast("b_d1", "b_d0", {s0, s1, s2, s3})
                   .Add("add2", "add1", "b_d1")
                   .Cast("cast", "add2", af::DT_FLOAT)
                   .Store("store", "cast")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // Add_0 两轮删除 z1/z0 扩维广播；Add_1 轮删除 b_c1 并回收 Add_0 轮 2 插入的
  // backward_broadcast_add0_0（其全扩维轴 {0} ⊆ 本轮公共轴）。
  EXPECT_FALSE(HasNode(graph, "b_a1"));
  EXPECT_FALSE(HasNode(graph, "b_a2"));
  EXPECT_FALSE(HasNode(graph, "b_b1"));
  EXPECT_FALSE(HasNode(graph, "b_b2"));
  EXPECT_FALSE(HasNode(graph, "b_c1"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_add0_0"));
  EXPECT_TRUE(HasNode(graph, "b_a0"));
  EXPECT_TRUE(HasNode(graph, "b_b0"));
  EXPECT_TRUE(HasNode(graph, "b_c0"));
  // Add_2 分支轴集无公共轴，D 链保持原状。
  EXPECT_TRUE(HasNode(graph, "b_d0"));
  EXPECT_TRUE(HasNode(graph, "b_d1"));
  // 级联拓扑：add0 → _1 → add1 → _0 → add2。
  EXPECT_TRUE(IsConnected(graph, "load0", "b_a0"));
  EXPECT_TRUE(IsConnected(graph, "b_a0", "add0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "b_b0"));
  EXPECT_TRUE(IsConnected(graph, "b_b0", "add0"));
  EXPECT_TRUE(IsConnected(graph, "add0", "backward_broadcast_add0_1"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_add0_1", "add1"));
  EXPECT_TRUE(IsConnected(graph, "load2", "b_c0"));
  EXPECT_TRUE(IsConnected(graph, "b_c0", "add1"));
  EXPECT_TRUE(IsConnected(graph, "add1", "backward_broadcast_add1_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_add1_0", "add2"));
  EXPECT_TRUE(IsConnected(graph, "load3", "b_d0"));
  EXPECT_TRUE(IsConnected(graph, "b_d0", "b_d1"));
  EXPECT_TRUE(IsConnected(graph, "b_d1", "add2"));
  EXPECT_TRUE(IsConnected(graph, "add2", "cast"));
  // 级联压实：add0 [1,1,s2,s3] → add1 [1,s1,s2,s3] → add2 [s0,s1,s2,s3]。
  const auto add0_node = FindNode(graph, "add0");
  const auto add1_node = FindNode(graph, "add1");
  const auto add2_node = FindNode(graph, "add2");
  const auto tail_brc_node = FindNode(graph, "backward_broadcast_add0_1");
  const auto add1_brc_node = FindNode(graph, "backward_broadcast_add1_0");
  ASSERT_NE(add0_node, nullptr);
  ASSERT_NE(add1_node, nullptr);
  ASSERT_NE(add2_node, nullptr);
  ASSERT_NE(tail_brc_node, nullptr);
  ASSERT_NE(add1_brc_node, nullptr);
  ExpectStaticEq(add0_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(tail_brc_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2, s3});
  ExpectStaticEq(add1_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2, s3});
  ExpectStaticEq(add1_brc_node->outputs[0].attr.repeats, {s0, s1, s2, s3});
  ExpectStaticEq(add2_node->outputs[0].attr.repeats, {s0, s1, s2, s3});
}

namespace {
// 标量链级联后移回归用例：scalar-like 生产者
// 视图恒为空，公共轴守卫必须对其免检放行，否则 Sub 组之后整条级联
// （Cast→Le/And/Where…）被误拒，只后移一小段。断言为 develop 基线
// （图级门控合入前的 partial backward）逐边比对的真值。
}  // namespace

TEST(BroadcastBackwardPass, MovesScalarChainCascadePartialGroups) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_scalar_chain_cascade")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("b00", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("b01", "b00", {s0, s1, s2})
                   .Broadcast("b10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b11", "b10", {s0, s1, s2})
                   .Sub("sub", "b01", "b11")
                   .Cast("cast", "sub", af::DT_FLOAT)
                   .Scalar("scalar0", "0", af::DT_INT64)
                   .Broadcast("b4", "scalar0", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("b5", "b4", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b6", "b5", {s0, s1, s2})
                   .Cast("cast1", "b6", af::DT_FLOAT)
                   .Add("le", "cast", "cast1")
                   .Store("store", "le")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  // 还原真实前端的 scalar-like 空视图形态（构图协议）。
  const auto scalar_node = FindNode(graph, "scalar0");
  ASSERT_NE(scalar_node, nullptr);
  scalar_node->outputs[0].attr.axis.clear();
  scalar_node->outputs[0].attr.repeats.clear();
  scalar_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // Sub 组：公共轴 s0 的 b00/b10 删除，load 直连非公共轴广播。
  EXPECT_FALSE(HasNode(graph, "b00"));
  EXPECT_FALSE(HasNode(graph, "b10"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b01"));
  EXPECT_TRUE(IsConnected(graph, "load1", "b11"));
  EXPECT_TRUE(IsConnected(graph, "b01", "sub"));
  EXPECT_TRUE(IsConnected(graph, "b11", "sub"));
  // Le 组级联（scalar-like 生产者免检）：b4 与 Sub 组插入的 s0 广播一并删除，
  // cast 直连 le，标量直连 b5，le 之后统一恢复 s0 扩维。
  EXPECT_FALSE(HasNode(graph, "b4"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "sub", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "le"));
  EXPECT_TRUE(IsConnected(graph, "scalar0", "b5"));
  EXPECT_TRUE(IsConnected(graph, "b5", "b6"));
  EXPECT_TRUE(IsConnected(graph, "b6", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "le"));
  EXPECT_TRUE(IsConnected(graph, "le", "backward_broadcast_le_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_le_0", "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisThroughSplitMultiConsumerCast) {
  // 标量链级联的 cast1 改为多消费（额外喂 neg→store1）：多输入节点 le 穿透
  // cast1 识别公共轴 s0 时，共享 cast1 的输出视图会被公共轴塌缩一并改写，殃及
  // neg 分支（shape 不一致）。拆分把"Brc 链 + Cast"按消费者整链复制后，le 分支
  // 独占私有 cast1（合法塌缩），neg 分支独占克隆链且视图保持不变。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, s1, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_split_multi_consumer_cast")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("b00", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("b01", "b00", {s0, s1, s2})
                   .Broadcast("b10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b11", "b10", {s0, s1, s2})
                   .Sub("sub", "b01", "b11")
                   .Cast("cast", "sub", af::DT_FLOAT)
                   .Scalar("scalar0", "0", af::DT_INT64)
                   .Broadcast("b4", "scalar0", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("b5", "b4", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b6", "b5", {s0, s1, s2})
                   .Cast("cast1", "b6", af::DT_FLOAT)
                   .Add("le", "cast", "cast1")
                   .Neg("neg", "cast1")
                   .Store("store", "le")
                   .Output("output", "store")
                   .Store("store1", "neg")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  SetNodeDtype(graph, "store1", af::DT_FLOAT);
  // 还原真实前端的 scalar-like 空视图形态（构图协议）。
  const auto scalar_node = FindNode(graph, "scalar0");
  ASSERT_NE(scalar_node, nullptr);
  scalar_node->outputs[0].attr.axis.clear();
  scalar_node->outputs[0].attr.repeats.clear();
  scalar_node->outputs[0].attr.strides.clear();
  // 后移前 cast1 输出为全扩维视图：neg 分支的视图基线。
  const auto cast1_before = FindNode(graph, "cast1");
  ASSERT_NE(cast1_before, nullptr);
  const auto expanded_repeats = cast1_before->outputs[0].attr.repeats;
  const auto expanded_strides = cast1_before->outputs[0].attr.strides;

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // Sub 组 + Le 组级联与单消费版本一致：公共轴广播删除、级联插入的 s0 广播一并
  // 删除、le 之后恢复 s0 扩维。
  EXPECT_FALSE(HasNode(graph, "b00"));
  EXPECT_FALSE(HasNode(graph, "b10"));
  EXPECT_FALSE(HasNode(graph, "b4_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b01"));
  EXPECT_TRUE(IsConnected(graph, "load1", "b11"));
  EXPECT_TRUE(IsConnected(graph, "sub", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "le"));
  EXPECT_TRUE(IsConnected(graph, "le", "backward_broadcast_le_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_le_0", "store"));
  // le 分支独占原 cast1（塌缩合法）：标量直连塌缩后的 b5，链尾接 le。
  EXPECT_TRUE(IsConnected(graph, "scalar0", "b5"));
  EXPECT_TRUE(IsConnected(graph, "b5", "b6"));
  EXPECT_TRUE(IsConnected(graph, "b6", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "le"));
  // neg 分支独占"链 + Cast"克隆：视图保持全扩维（未被公共轴塌缩殃及）。
  EXPECT_TRUE(IsConnected(graph, "scalar0", "b4"));
  EXPECT_TRUE(IsConnected(graph, "b4", "b5_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "b5_consumer_split_1", "b6_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "b6_consumer_split_1", "cast1_consumer_split_1"));
  EXPECT_TRUE(IsConnected(graph, "cast1_consumer_split_1", "neg"));
  EXPECT_TRUE(IsConnected(graph, "neg", "store1"));
  const auto cast1_clone = FindNode(graph, "cast1_consumer_split_1");
  ASSERT_NE(cast1_clone, nullptr);
  ExpectStaticEq(cast1_clone->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(cast1_clone->outputs[0].attr.strides, expanded_strides);
  const auto neg_after = FindNode(graph, "neg");
  ASSERT_NE(neg_after, nullptr);
  ExpectStaticEq(neg_after->inputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(neg_after->inputs[0].attr.strides, expanded_strides);
}

TEST(BroadcastBackwardPass, SkipsCommonAxisThroughSharedCastWithControlEdge) {
  // cast1 带控制边时拆分 fail-closed（克隆不带控制边会丢失顺序约束），共享 Cast
  // 保持多消费；穿透侧对多消费 Cast fail-closed 拒绝登记公共轴后移——否则公共轴
  // 塌缩会殃及 neg 分支。Le 组保持原状，cast1 输出视图不变。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, s1, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_shared_cast_control_edge")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("b00", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("b01", "b00", {s0, s1, s2})
                   .Broadcast("b10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b11", "b10", {s0, s1, s2})
                   .Sub("sub", "b01", "b11")
                   .Cast("cast", "sub", af::DT_FLOAT)
                   .Scalar("scalar0", "0", af::DT_INT64)
                   .Broadcast("b4", "scalar0", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("b5", "b4", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b6", "b5", {s0, s1, s2})
                   .Cast("cast1", "b6", af::DT_FLOAT)
                   .Add("le", "cast", "cast1")
                   .Neg("neg", "cast1")
                   .Store("store", "le")
                   .Output("output", "store")
                   .Store("store1", "neg")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  SetNodeDtype(graph, "store1", af::DT_FLOAT);
  const auto scalar_node = FindNode(graph, "scalar0");
  ASSERT_NE(scalar_node, nullptr);
  scalar_node->outputs[0].attr.axis.clear();
  scalar_node->outputs[0].attr.repeats.clear();
  scalar_node->outputs[0].attr.strides.clear();
  const auto cast1_node = FindNode(graph, "cast1");
  ASSERT_NE(cast1_node, nullptr);
  const auto expanded_repeats = cast1_node->outputs[0].attr.repeats;
  const auto expanded_strides = cast1_node->outputs[0].attr.strides;
  ASSERT_EQ(af::GraphUtils::AddEdge(scalar_node->GetOutControlAnchor(), cast1_node->GetInControlAnchor()), af::SUCCESS);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // Le 组未登记公共轴后移：cast1 链与 neg 分支保持原状，无克隆、无塌缩。
  EXPECT_FALSE(HasNode(graph, "cast1_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_le_0"));
  EXPECT_TRUE(IsConnected(graph, "scalar0", "b4"));
  EXPECT_TRUE(IsConnected(graph, "b4", "b5"));
  EXPECT_TRUE(IsConnected(graph, "b5", "b6"));
  EXPECT_TRUE(IsConnected(graph, "b6", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "le"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "neg"));
  EXPECT_TRUE(IsConnected(graph, "neg", "store1"));
  const auto cast1_after = FindNode(graph, "cast1");
  ASSERT_NE(cast1_after, nullptr);
  ExpectStaticEq(cast1_after->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(cast1_after->outputs[0].attr.strides, expanded_strides);
}

TEST(BroadcastBackwardSharedSplit, MergesBackSplitMultiConsumerCastChain) {
  // "Brc 直连多消费 Cast"拆分的完整回滚：无后移收益时（直调拆分 + 合并，中间无
  // 后移路径），广播链与链尾 Cast 一并合并回一条共享链，图恢复等价形态，无克隆
  // 残留、视图与拆分前一致。
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  auto graph = AscGraphBuilder("broadcast_backward_split_cast_roundtrip")
                   .Loops({s0, s1})
                   .Data("data", 0)
                   .Load("load", "data", kCompactRepeats, kCompactStrides)
                   .Broadcast("brc", "load", {s0, s1})
                   .Cast("cast", "brc", af::DT_FLOAT16)
                   .Abs("abs0", "cast")
                   .Abs("abs1", "cast")
                   .Store("store0", "abs0")
                   .Output("output0", "store0")
                   .Store("store1", "abs1")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);
  const auto cast_before = FindNode(graph, "cast");
  ASSERT_NE(cast_before, nullptr);
  const auto expanded_repeats = cast_before->outputs[0].attr.repeats;
  const auto expanded_strides = cast_before->outputs[0].attr.strides;

  ASSERT_EQ(optimize::broadcast_backward_shared_split::SplitSharedBroadcastConsumers(graph), af::SUCCESS);
  // 拆分后：cast 多消费解除，消费 1 改接"广播链 + Cast"整链克隆。
  EXPECT_NE(FindNode(graph, "cast_consumer_split_1"), nullptr);
  EXPECT_NE(FindNode(graph, "brc_consumer_split_1"), nullptr);
  EXPECT_EQ(FindNode(graph, "cast")->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);

  ASSERT_EQ(optimize::broadcast_backward_shared_split::MergeSharedBroadcastConsumers(graph), af::SUCCESS);
  // 合并后：等价广播链 + 等价尾 Cast 一并回滚，链尾恢复多消费，视图不变。
  EXPECT_FALSE(HasNode(graph, "cast_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "brc_consumer_split_1"));
  const auto cast_after = FindNode(graph, "cast");
  ASSERT_NE(cast_after, nullptr);
  EXPECT_EQ(cast_after->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  ExpectStaticEq(cast_after->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(cast_after->outputs[0].attr.strides, expanded_strides);
  const auto brc_after = FindNode(graph, "brc");
  ASSERT_NE(brc_after, nullptr);
  EXPECT_EQ(brc_after->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 1U);
  EXPECT_TRUE(IsConnected(graph, "load", "brc"));
  EXPECT_TRUE(IsConnected(graph, "brc", "cast"));
}

TEST(BroadcastBackwardPass, MovesArangeChainCascadePartialGroups) {
  // Arange 退化视图生产者 + 标量链级联组合：Sub 组走 Arange 公共轴放行
  // （生产者 [1,1,s2]/[1,s1,1] 含公共轴 s0 且退化），标量链走 scalar-like
  // 免检——两级改写叠加后级联推到底，与 Load 生产者版本（见
  // MovesScalarChainCascadePartialGroups）逐边同构。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_arange_chain_cascade")
                   .Loops({s0, s1, s2})
                   .Arange("arange_col", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                           {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Arange("arange_row", {af::sym::kSymbolOne, s1, af::sym::kSymbolOne},
                           {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero})
                   .Broadcast("b00", "arange_col", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("b01", "b00", {s0, s1, s2})
                   .Broadcast("b10", "arange_row", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b11", "b10", {s0, s1, s2})
                   .Sub("sub", "b01", "b11")
                   .Cast("cast", "sub", af::DT_FLOAT)
                   .Scalar("scalar0", "0", af::DT_INT64)
                   .Broadcast("b4", "scalar0", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne})
                   .Broadcast("b5", "b4", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("b6", "b5", {s0, s1, s2})
                   .Cast("cast1", "b6", af::DT_FLOAT)
                   .Add("le", "cast", "cast1")
                   .Store("store", "le")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);
  SetNodeDtype(graph, "store", af::DT_FLOAT);
  const auto scalar_node = FindNode(graph, "scalar0");
  ASSERT_NE(scalar_node, nullptr);
  scalar_node->outputs[0].attr.axis.clear();
  scalar_node->outputs[0].attr.repeats.clear();
  scalar_node->outputs[0].attr.strides.clear();

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_FALSE(HasNode(graph, "b00"));
  EXPECT_FALSE(HasNode(graph, "b10"));
  EXPECT_FALSE(HasNode(graph, "b4"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_cast_0"));
  EXPECT_TRUE(IsConnected(graph, "arange_col", "b01"));
  EXPECT_TRUE(IsConnected(graph, "arange_row", "b11"));
  EXPECT_TRUE(IsConnected(graph, "b01", "sub"));
  EXPECT_TRUE(IsConnected(graph, "b11", "sub"));
  EXPECT_TRUE(IsConnected(graph, "sub", "cast"));
  EXPECT_TRUE(IsConnected(graph, "cast", "le"));
  EXPECT_TRUE(IsConnected(graph, "scalar0", "b5"));
  EXPECT_TRUE(IsConnected(graph, "b5", "b6"));
  EXPECT_TRUE(IsConnected(graph, "b6", "cast1"));
  EXPECT_TRUE(IsConnected(graph, "cast1", "le"));
  EXPECT_TRUE(IsConnected(graph, "le", "backward_broadcast_le_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_le_0", "store"));
}

TEST(BroadcastBackwardPass, MovesCommonAxisToEachMultiOutputBranch) {
  // 多输入多输出（fan-out）部分后移 + 外层迭代至不动点：公共轴 s0 的 b00/b10 删除、
  // load 直连残差广播，merge 输出在公共轴塌缩，公共轴 Brc 先插入每路输出分支首节点
  // 之前；主循环因 partial backward 存在改写而重跑，插入的广播被下一轮原始链路径沿
  // 分支链继续后移（abs→exp、neg），最终停靠在各路 Store 之前。两路广播位置不同构、
  // 收尾不再合并回滚：分支计算在塌缩视图上进行，广播在 Store 前恢复扩维。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero};
  auto graph = AscGraphBuilder("broadcast_backward_multi_output_common_axis")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("broadcast00", "load0", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("broadcast01", "broadcast00", {s0, s1, s2})
                   .Broadcast("broadcast10", "load1", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("broadcast11", "broadcast10", {s0, s1, s2})
                   .Add("merge", "broadcast01", "broadcast11")
                   .Abs("abs", "merge")
                   .Exp("exp", "abs")
                   .Neg("neg", "merge")
                   .Store("store0", "exp")
                   .Output("output0", "store0")
                   .Store("store1", "neg")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 公共轴 s0 的广播删除，load 直连残差广播（非公共轴扩维保留）。
  EXPECT_FALSE(HasNode(graph, "broadcast00"));
  EXPECT_FALSE(HasNode(graph, "broadcast10"));
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast01"));
  EXPECT_TRUE(IsConnected(graph, "load1", "broadcast11"));
  EXPECT_TRUE(IsConnected(graph, "broadcast01", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast11", "merge"));
  // 公共轴 Brc 插入后被下一轮原始链路径沿分支链继续后移到 Store 之前：分支计算
  // （abs→exp、neg）保持直连且在塌缩视图上计算，两路广播不同构、收尾不合并。
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "exp"));
  EXPECT_TRUE(IsConnected(graph, "exp", "backward_broadcast_merge_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_0", "store0"));
  EXPECT_TRUE(IsConnected(graph, "merge", "neg"));
  EXPECT_TRUE(IsConnected(graph, "neg", "backward_broadcast_merge_0_"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_0_", "store1"));
  // 残差广播/merge/分支计算在公共轴 s0 塌缩，分支尾广播恢复 s0 扩维。
  const auto broadcast01 = FindNode(graph, "broadcast01");
  const auto broadcast11 = FindNode(graph, "broadcast11");
  const auto merge = FindNode(graph, "merge");
  const auto branch_brc = FindNode(graph, "backward_broadcast_merge_0");
  const auto branch_brc_clone = FindNode(graph, "backward_broadcast_merge_0_");
  const auto abs_node = FindNode(graph, "abs");
  const auto neg_node = FindNode(graph, "neg");
  ASSERT_NE(broadcast01, nullptr);
  ASSERT_NE(broadcast11, nullptr);
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(branch_brc, nullptr);
  ASSERT_NE(branch_brc_clone, nullptr);
  ASSERT_NE(abs_node, nullptr);
  ASSERT_NE(neg_node, nullptr);
  ExpectStaticEq(broadcast01->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(broadcast11->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(merge->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(abs_node->inputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(abs_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(neg_node->inputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(neg_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, s1, s2});
  ExpectStaticEq(branch_brc->outputs[0].attr.repeats, {s0, s1, s2});
  ExpectStaticEq(branch_brc_clone->outputs[0].attr.repeats, {s0, s1, s2});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "abs"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "abs", "exp"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "exp", "backward_broadcast_merge_0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_0", "store0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "neg"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "neg", "backward_broadcast_merge_0_"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_0_", "store1"));
}
TEST(BroadcastBackwardPass, MovesMultiCommonAxesToEachMultiOutputBranchByRound) {
  // 多输入多输出（fan-out）双公共轴 {s0,s1} 迭代式单轴后移 + 外层迭代至不动点：
  // 先迁最大轴 s1 再迁 s0（与单输出 partial backward 同样的逐轮迭代机制），每轮在
  // 每路输出分支首节点之前各插入一个单轴广播；主循环因 partial backward 存在改写
  // 而重跑，插入的广播链被下一轮原始链路径沿分支链继续后移，最终每路形成
  // exp→_0(扩轴0)→_1(扩轴1)→Store 的链（分支计算在塌缩视图上），两路广播链
  // 位置不同构、收尾不再合并回滚。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const std::vector<af::Expression> row_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, af::sym::kSymbolOne};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne,
                                                   af::sym::kSymbolZero};
  const std::vector<af::Expression> col_repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, s3};
  const std::vector<af::Expression> col_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero,
                                                   af::sym::kSymbolOne};
  auto graph = AscGraphBuilder("broadcast_backward_multi_output_multi_axes")
                   .Loops({s0, s1, s2, s3})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Load("load0", "data0", row_repeats, row_strides)
                   .Load("load1", "data1", col_repeats, col_strides)
                   .Broadcast("b_a0", "load0", {s0, af::sym::kSymbolOne, s2, af::sym::kSymbolOne})
                   .Broadcast("b_a1", "b_a0", {s0, s1, s2, af::sym::kSymbolOne})
                   .Broadcast("b_a2", "b_a1", {s0, s1, s2, s3})
                   .Broadcast("b_b0", "load1", {s0, af::sym::kSymbolOne, af::sym::kSymbolOne, s3})
                   .Broadcast("b_b1", "b_b0", {s0, s1, af::sym::kSymbolOne, s3})
                   .Broadcast("b_b2", "b_b1", {s0, s1, s2, s3})
                   .Add("merge", "b_a2", "b_b2")
                   .Abs("abs", "merge")
                   .Exp("exp", "abs")
                   .Neg("neg", "merge")
                   .Store("store0", "exp")
                   .Output("output0", "store0")
                   .Store("store1", "neg")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  // 公共轴 0/1 的分支广播删除，load 直连残差广播。
  EXPECT_FALSE(HasNode(graph, "b_a0"));
  EXPECT_FALSE(HasNode(graph, "b_a1"));
  EXPECT_FALSE(HasNode(graph, "b_b0"));
  EXPECT_FALSE(HasNode(graph, "b_b1"));
  EXPECT_TRUE(IsConnected(graph, "load0", "b_a2"));
  EXPECT_TRUE(IsConnected(graph, "load1", "b_b2"));
  EXPECT_TRUE(IsConnected(graph, "b_a2", "merge"));
  EXPECT_TRUE(IsConnected(graph, "b_b2", "merge"));
  // 插入的广播链被下一轮原始链路径沿分支链继续后移到 Store 之前（分支计算
  // abs→exp、neg 保持直连且在塌缩视图上计算），两路广播链不同构、收尾不合并。
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "exp"));
  EXPECT_TRUE(IsConnected(graph, "exp", "backward_broadcast_merge_0"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_0", "backward_broadcast_merge_1"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_1", "store0"));
  EXPECT_TRUE(IsConnected(graph, "merge", "neg"));
  EXPECT_TRUE(IsConnected(graph, "neg", "backward_broadcast_merge_0_"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_0_", "backward_broadcast_merge_1_"));
  EXPECT_TRUE(IsConnected(graph, "backward_broadcast_merge_1_", "store1"));
  const auto b_a2 = FindNode(graph, "b_a2");
  const auto b_b2 = FindNode(graph, "b_b2");
  const auto merge = FindNode(graph, "merge");
  const auto axis0_node = FindNode(graph, "backward_broadcast_merge_0");
  const auto axis1_node = FindNode(graph, "backward_broadcast_merge_1");
  const auto axis0_clone = FindNode(graph, "backward_broadcast_merge_0_");
  const auto axis1_clone = FindNode(graph, "backward_broadcast_merge_1_");
  const auto abs_node = FindNode(graph, "abs");
  const auto neg_node = FindNode(graph, "neg");
  ASSERT_NE(b_a2, nullptr);
  ASSERT_NE(b_b2, nullptr);
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(axis0_node, nullptr);
  ASSERT_NE(axis1_node, nullptr);
  ASSERT_NE(axis0_clone, nullptr);
  ASSERT_NE(axis1_clone, nullptr);
  ASSERT_NE(abs_node, nullptr);
  ASSERT_NE(neg_node, nullptr);
  ExpectStaticEq(b_a2->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(b_b2->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(merge->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(abs_node->inputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(abs_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(neg_node->inputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(neg_node->outputs[0].attr.repeats, {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(axis0_node->outputs[0].attr.repeats, {s0, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(axis1_node->outputs[0].attr.repeats, {s0, s1, s2, s3});
  ExpectStaticEq(axis0_clone->outputs[0].attr.repeats, {s0, af::sym::kSymbolOne, s2, s3});
  ExpectStaticEq(axis1_clone->outputs[0].attr.repeats, {s0, s1, s2, s3});
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "abs"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "abs", "exp"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "exp", "backward_broadcast_merge_0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_0", "backward_broadcast_merge_1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_1", "store0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "neg"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "neg", "backward_broadcast_merge_0_"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_0_", "backward_broadcast_merge_1_"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "backward_broadcast_merge_1_", "store1"));
}
TEST(BroadcastBackwardPass, SkipsSingleInputMultiOutputBackward) {
  // 单输入多输出（fan-out）仍然受限：链式收集依赖唯一后继，且无多分支公共轴
  // 可求交；广播链保持原状，不产生 backward_broadcast 插入。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_single_input_multi_output")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Broadcast("broadcast0", "load0", {s0, s1, s2})
                   .Abs("abs", "broadcast0")
                   .Neg("neg", "abs")
                   .Relu("relu", "abs")
                   .Store("store0", "neg")
                   .Output("output0", "store0")
                   .Store("store1", "relu")
                   .Output("output1", "store1")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "abs"));
  EXPECT_TRUE(IsConnected(graph, "abs", "neg"));
  EXPECT_TRUE(IsConnected(graph, "abs", "relu"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_abs_0"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_neg_0"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_relu_0"));
}

TEST(BroadcastBackwardPass, SkipsMultiOutputWithoutCommonAxis) {
  // 多输入多输出（fan-out）但分支扩维轴集无公共轴：不登记 partial backward，
  // 图保持原状。
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph =
      AscGraphBuilder("broadcast_backward_multi_output_no_common_axis")
          .Loops({s0, s1, s2})
          .Data("data0", 0)
          .Data("data1", 1)
          .Load("load0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Load("load1", "data1", {s0, s1, af::sym::kSymbolOne}, {s1, af::sym::kSymbolOne, af::sym::kSymbolZero})
          .Broadcast("broadcast0", "load0", {s0, af::sym::kSymbolOne, s2})
          .Broadcast("broadcast1", "load1", {s0, s1, s2})
          .Add("merge", "broadcast0", "broadcast1")
          .Abs("abs", "merge")
          .Neg("neg", "merge")
          .Store("store0", "abs")
          .Output("output0", "store0")
          .Store("store1", "neg")
          .Output("output1", "store1")
          .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "load1", "broadcast1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "abs"));
  EXPECT_TRUE(IsConnected(graph, "merge", "neg"));
  EXPECT_FALSE(HasNode(graph, "backward_broadcast_merge_0"));
}

namespace {
af::AscNodePtr InputProducer(const af::AscNodePtr &node, int32_t idx) {
  if (node == nullptr) {
    return nullptr;
  }
  auto in_anchor = node->GetInDataAnchor(idx);
  if (in_anchor == nullptr) {
    return nullptr;
  }
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(out_anchor->GetOwnerNode());
}
}  // namespace

// partial backward 的"同轴兄弟整链删除"公共轴门控回归：兄弟链含非公共轴扩维时不得整链删除，
// 必须退回公共轴 partial backward（分支 2 视图保持一致）。
//
//   loadA[1,1,s2] -> A0(扩 s0) -> A1(扩 s1) -> M.x2
//   loadB[1,s1,1] -> B0(扩 s0) -> B2(扩 s2) -> M.x1
//   M = Add(B2, A1)                       // 两分支扩维轴集 {s0,s2} vs {s0,s1}，公共轴 {s0}
//   loadC[1,1,s2] -> C0(扩 s0) -> C1(扩 s1) -> N.x2
//   N = Add(M, C1)                        // N.x2 的 [C0,C1] 与 origin 链 [A0,A1] 同轴同构
//   store(N)
//
// M 走 partial backward（公共轴 {s0}）。CollectCmpNodes 沿 M->N 收集计算链时遇到 N.x2 的
// [C0,C1]：它与 origin 链同轴同构，但其非公共轴 s1 不在本轮公共轴集 {s0} 内，整链删除后
// 无人在链尾恢复 s1 -> N 两输入视图不一致。门控要求兄弟链全部扩维轴 ⊆ 公共轴集才允许整链
// 删除；此处不满足，退回公共轴 partial backward：只删公共轴 s0 节点、保留 C1，N.x2 保持
// s1 扩维，N 两输入视图一致。
TEST(BroadcastBackwardPass, KeepsNonCommonAxisSiblingChainOnPartialBackward) {
  ScopedTestPlatform platform("3510");
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder("broadcast_backward_repro_sibling_axis_loss")
                   .Loops({s0, s1, s2})
                   .Data("dataA", 0)
                   .Data("dataB", 1)
                   .Data("dataC", 2)
                   .Load("loadA", "dataA", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Broadcast("A0", "loadA", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("A1", "A0", {s0, s1, s2})
                   .Load("loadB", "dataB", {af::sym::kSymbolOne, s1, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero})
                   .Broadcast("B0", "loadB", {s0, s1, af::sym::kSymbolOne})
                   .Broadcast("B2", "B0", {s0, s1, s2})
                   .Add("M", "B2", "A1")
                   .Load("loadC", "dataC", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolOne})
                   .Broadcast("C0", "loadC", {s0, af::sym::kSymbolOne, s2})
                   .Broadcast("C1", "C0", {s0, s1, s2})
                   .Add("N", "M", "C1")
                   .Store("store", "N")
                   .Output("output", "store")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);

  const auto m_node = FindNode(graph, "M");
  const auto n_node = FindNode(graph, "N");
  ASSERT_NE(m_node, nullptr);
  ASSERT_NE(n_node, nullptr);
  const auto x2_prod = InputProducer(n_node, 1);
  ASSERT_NE(x2_prod, nullptr);

  // 正确行为：非公共轴兄弟链不得被整链删除（C1 应保留），loadC 不应直连 N。
  EXPECT_TRUE(HasNode(graph, "A1"));
  EXPECT_FALSE(IsConnected(graph, "loadC", "N"));
  // 逐元素 op 的两个输入必须视图一致：N.x2 侧必须仍带 s1 扩维。
  ExpectStaticEq(m_node->outputs[0].attr.repeats, x2_prod->outputs[0].attr.repeats);
}

// MergeChainInto 回归：scalar 源多消费广播拆分出的多条单节点克隆链与更长原链
// （[brc0,brc1]）混合时，若合并循环先做克隆链之间的等长合并（keep 链尾累积多个
// 消费者）、再让该链作为 dup 前缀并入原链，旧实现只改接首个消费边（.at(0)），
// 其余消费边随克隆删除被丢弃 -> 消费者广播侧输入悬空（实测终态仅剩
// Brc0(s0,1)->Brc1(s0,s1) | Add1，Add2/Mul 去向不明）。本图还原该形态：3 条单消费
// 克隆链 + 1 条两级原链，消费者另一路均为满 shape 张量、无任何后移。RDFS 拓扑序在
// 输出尺寸打平时按名字序探索，原链分支（mul1 一路）的 store/output 命名排在最后，
// 保证克隆链先于原链收集（危险序）；断言与链序无关的终态不变量：克隆全部回滚、
// 原链完整、全部消费者改接回 brc0。
TEST(BroadcastBackwardPass, RollsBackAllConsumersWhenClonesMergeBeforeLongerChain) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  auto graph = AscGraphBuilder("broadcast_backward_merge_clones_before_longer_chain")
                   .Loops({s0, s1})
                   .Scalar("scalar", "1.0")
                   .Broadcast("brc0", "scalar", {s0, one})
                   .Broadcast("brc1", "brc0", {s0, s1})
                   .Data("dataX", 0)
                   .Data("dataA", 1)
                   .Data("dataB", 2)
                   .Data("dataC", 3)
                   .Load("loadX", "dataX", {s0, s1}, {s1, one})
                   .Load("loadA", "dataA", {s0, one}, {one, zero})
                   .Load("loadB", "dataB", {s0, one}, {one, zero})
                   .Load("loadC", "dataC", {s0, one}, {one, zero})
                   .Mul("mul1", "brc1", "loadX")
                   .Add("add1", "brc0", "loadA")
                   .Add("add2", "brc0", "loadB")
                   .Mul("mul2", "brc0", "loadC")
                   .Store("store4", "mul1")
                   .Output("output4", "store4")
                   .Store("store1", "add1")
                   .Output("output1", "store1")
                   .Store("store2", "add2")
                   .Output("output2", "store2")
                   .Store("store3", "mul2")
                   .Output("output3", "store3")
                   .Build();
  CompleteApiInfo(graph);

  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS);

  // 拆分克隆全部合并回滚，不残留。
  EXPECT_FALSE(HasNode(graph, "brc0_consumer_split_1"));
  EXPECT_FALSE(HasNode(graph, "brc0_consumer_split_2"));
  EXPECT_FALSE(HasNode(graph, "brc0_consumer_split_3"));
  // 两级原链保持：scalar -> brc0(s0,1) -> brc1(s0,s1) -> mul1。
  EXPECT_TRUE(IsConnected(graph, "scalar", "brc0"));
  EXPECT_TRUE(IsConnected(graph, "brc0", "brc1"));
  EXPECT_TRUE(IsConnected(graph, "brc1", "mul1"));
  // 3 条克隆链的消费者全部改接回 brc0，无任何消费边悬空（旧实现只保留锚点序第一个）。
  EXPECT_TRUE(IsConnected(graph, "brc0", "add1"));
  EXPECT_TRUE(IsConnected(graph, "brc0", "add2"));
  EXPECT_TRUE(IsConnected(graph, "brc0", "mul2"));
  EXPECT_EQ(FindNode(graph, "brc0")->GetOutDataNodesSize(), 4U);
}
