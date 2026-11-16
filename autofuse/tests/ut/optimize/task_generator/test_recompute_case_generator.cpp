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
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "task_generator/recompute_case_generator.h"
#include "asc_graph_builder.h"
#include "ascgraph_info_complete.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using af::testing::AscGraphBuilder;
using af::testing::Sym;

class RecomputeCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplit) {
  auto s0 = Sym("s0"), s1 = Sym("s1"), s2 = Sym("s2"), s3 = Sym("s3"), s4 = Sym("s4");
  auto graph = AscGraphBuilder("brc_abs")
                   .Loops({s0, s1, s2, s3, s4})
                   .Data("RE_data0", 0, ge::DT_FLOAT16)
                   .Load("RE_load0", "RE_data0", {One, One, One, s3, s4})
                   .Abs("RE_abs", "RE_load0")
                   .Broadcast("brc1", "RE_abs", {s0, s1, s2, s3, s4})
                   .Store("store", "brc1")
                   .Output("out", "store", 0)
                   .Abs("ST_abs2", "RE_abs")
                   .Store("ST_store1", "ST_abs2")
                   .Output("ST_out1", "ST_store1", 1)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 存在首轴 B 语义下 RE_abs->brc1(首轴及其余轴均有 B 扩展)断边无残余连通, 额外产生 broadcast ws 候选:
  // tasks = [broadcast(2), origin(1), recompute(2)]
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[2].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestDynamicGraphSplitTwoLine) {
  auto s0 = Sym("s0"), s1 = Sym("s1"), s2 = Sym("s2"), s3 = Sym("s3"), s4 = Sym("s4");
  auto graph = AscGraphBuilder("brc_abs")
                   .Loops({s0, s1, s2, s3, s4})
                   .Data("RE_data0", 0, ge::DT_FLOAT16)
                   .Load("RE_load0", "RE_data0", {One, One, One, s3, s4})
                   .Abs("RE_abs", "RE_load0")
                   .Broadcast("brc1", "RE_abs", {s0, s1, s2, s3, s4})
                   .Store("store", "brc1")
                   .Output("out", "store", 0)
                   .Abs("ST_abs2", "RE_abs")
                   .Store("ST_store1", "ST_abs2")
                   .Output("ST_out1", "ST_store1", 1)
                   .Abs("ST_abs3", "ST_abs2")
                   .Abs("ST_abs4", "ST_abs3")
                   .Abs("ST_abs5", "ST_abs4")
                   .Abs("ST_abs6", "ST_abs5")
                   .Store("ST_store2", "ST_abs6")
                   .Output("ST_out2", "ST_store2", 2)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 存在首轴 B 语义下 RE_abs->brc1(首轴及其余轴均有 B 扩展)断边无残余连通, 额外产生 broadcast ws 候选:
  // tasks = [broadcast(2), origin(1), recompute(2)]
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[2].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitTwoLine) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_abs")
                   .Loops({s0, s1})
                   .Data("RE_data0", 0, ge::DT_FLOAT16)
                   .Load("RE_load0", "RE_data0", {One, s1})
                   .Abs("RE_abs", "RE_load0")
                   .Broadcast("brc1", "RE_abs", {s0, s1})
                   .Store("store", "brc1")
                   .Output("out", "store", 0)
                   .Abs("ST_abs2", "RE_abs")
                   .Store("ST_store1", "ST_abs2")
                   .Output("ST_out1", "ST_store1", 1)
                   .Abs("ST_abs3", "ST_abs2")
                   .Abs("ST_abs4", "ST_abs3")
                   .Abs("ST_abs5", "ST_abs4")
                   .Abs("ST_abs6", "ST_abs5")
                   .Store("ST_store2", "ST_abs6")
                   .Output("ST_out2", "ST_store2", 2)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 泛化后 RE_abs->brc1 首轴 Brc(非 where 消费者)断边无残余连通, 额外产生 broadcast ws 候选:
  // tasks = [broadcast(2), origin(1), recompute(2)]
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[2].grouped_graphs.size(), 2UL);
}

TEST_F(RecomputeCaseGeneratorTest, TestStaticGraphSplitWithBrc) {
  auto s0 = Sym(1024), s1 = Sym(32), s2 = Sym(32);
  auto graph = AscGraphBuilder("brc_abs")
                   .Loops({s0, s1, s2})
                   .Data("RE_data0", 0, ge::DT_FLOAT16)
                   .Load("RE_load0", "RE_data0", {One, One, s2})
                   .Abs("RE_abs", "RE_load0")
                   .Broadcast("brc1", "RE_abs", {One, s1, s2})
                   .Broadcast("brc2", "brc1", {s0, s1, s2})
                   .Relu("relu0", "brc2")
                   .Relu("relu1", "relu0")
                   .Relu("relu2", "relu1")
                   .Store("store", "relu2")
                   .Output("out", "store", 0)
                   .Abs("ST_abs2", "brc1")
                   .Exp("ST_exp", "brc1")
                   .Add("ST_add", "ST_abs2", "ST_exp")
                   .Store("ST_store1", "ST_add")
                   .Output("ST_out1", "ST_store1", 1)
                   .Abs("ST_abs3", "ST_abs2")
                   .Abs("ST_abs4", "ST_abs3")
                   .Abs("ST_abs5", "ST_abs4")
                   .Abs("ST_abs6", "ST_abs5")
                   .Store("ST_store2", "ST_abs6")
                   .Output("ST_out2", "ST_store2", 2)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 泛化后 brc1->brc2 首轴 Brc(非 where 消费者)断边无残余连通, 额外产生 broadcast ws 候选:
  // tasks = [broadcast(2), origin(1), recompute(2)]
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[2].grouped_graphs.size(), 2UL);
}

namespace {
// 统计候选任务子图中名为 node_name 的节点出现的子图数(调用方需先断言 tasks.size() 与 task_index 有效)
size_t CountSubGraphsWithNode(const ScheduleTask &task, const std::string &type, const std::string &node_name) {
  size_t sub_graph_count = 0UL;
  for (const auto &grouped_graph : task.grouped_graphs) {
    for (const auto &node : grouped_graph.GetAllNodes()) {
      if (node->GetType() == type && node->GetName() == node_name) {
        ++sub_graph_count;
        break;
      }
    }
  }
  return sub_graph_count;
}

// Broadcast workspace 切分模板的打分函数: 本组既有用例非首轴乘积均 ≤ 32K(s1=256), 静态分支
// 生成常量负分(劣后给默认模板; 平分时 result 序号小者胜, 本模板恒为 result0, 返回 0 分无法落选)
const std::string kBroadcastWorkspaceFallbackScoreFunc =
    "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
    "  return -1;\n"
    "}\n";
// 非首轴乘积 > 32K 时静态分支生成常量正分
const std::string kBroadcastWorkspaceSelectedScoreFunc =
    "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
    "  return 1;\n"
    "}\n";
}  // namespace

// where 节点前的首轴 Broadcast 生成 workspace 切分候选: 切分链 abs0 -> store -> ws_pre ⫽
// ws_post -> load -> brc0, ws_pre/ws_post 同名对分属两个子图; 图无可重计算 store, 走 fallback
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseBeforeWhere) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_where")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // broadcast workspace 切分候选 + fallback 默认任务
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Store", "abs0_to_brc0_Store_0"), 1UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Load", "abs0_to_brc0_Load_0"), 1UL);
  // 非首轴乘积(s1=256)未超 32K: broadcast workspace 模板打常量负分劣后, fallback 模板保持空打分(默认0分)
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
  EXPECT_EQ(tasks[1].score_func, "");
}

// Select 与 Where 同为 where 类三输入选择算子, 同样触发切分候选
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseBeforeSelect) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_select")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Select>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
}

// Broadcast 上游为 Load(GM 搬运节点)时不收集(即使输入为 Data 且有其他数据消费者): Load 副本
// 路径不建 ws 链且有输入类型断言与悬置退化子图两类崩溃风险, 上游须为真实计算节点
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForLoadInput) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_load_where")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Broadcast("brc0", "load0", {s0, s1})
                   .Abs("abs0", "load0")  // load0 的其他数据消费者, Load 上游仍不收集
                   .Store("store1", "abs0")
                   .Output("out1", "store1", 1)
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // Load 上游不收集, 无 broadcast ws 候选: tasks = [origin(1), recompute(2)], 均为空打分
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
  EXPECT_EQ(tasks[0].score_func, "");
  EXPECT_EQ(tasks[1].score_func, "");
}

// Broadcast 上游 Load 唯一消费者为本 Brc 时同样不收集(该形态曾因悬置无 Store 退化子图致调度期
// 段错误, Load 上游整体排除后该风险结构性消除)
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForStrandedLoadInput) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_stranded_load_where")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Broadcast("brc0", "load0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 原 Load 会被悬置, 候选被拒, 仅 fallback 默认任务
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// Broadcast 切分候选与重计算候选共存: tasks = [broadcast(2), origin(1), recompute(2)]
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseWithRecomputePath) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_where_recompute")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Exp("exp0", "abs0")
                   .Store("store1", "exp0")
                   .Output("out1", "store1", 1)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[2].grouped_graphs.size(), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  // 非首轴乘积(s1=256)未超 32K: broadcast workspace 模板打常量负分劣后, origin/recompute 模板
  // 保持空打分(默认0分)
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
  EXPECT_EQ(tasks[1].score_func, "");
  EXPECT_EQ(tasks[2].score_func, "");
}

// Broadcast 上个节点是 Scalar 时不识别, 不产生额外候选
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForScalarInput) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_scalar_where")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Scalar("scalar0", "0.5", ge::DT_FLOAT16)
                   .Broadcast("brc0", "scalar0", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "load0", "brc0"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// 非首轴 Broadcast(上游为计算节点)不识别
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForNonFirstAxis) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_tail_where")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {s0, One})
                   .Abs("abs0", "load0")  // 计算节点上游, 确保由非首轴条件拒绝而非 Load 上游条件
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// 泛化: Broadcast 下游为任意 elewise 节点(不限定 where 类)同样识别, 断边后无残余连通的单候选真切
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseForElewiseConsumer) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_abs")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Abs("abs1", "brc0")
                   .Store("store0", "abs1")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // broadcast workspace 切分候选 + fallback 默认任务
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
}

// 独立生产者双 Brc 汇入同一 Where: 两个单候选组各自断边判定均无残余连通, 两处真切,
// 两个独立生产侧子图 + 一个消费侧子图
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseForIndependentDualBrc) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_where_independent_dual")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("data1", 1, ge::DT_FLOAT16)
                   .Load("load1", "data1", {One, s1})
                   .Abs("abs1", "load1")
                   .Broadcast("brc1", "abs1", {s0, s1})
                   .Data("mask_data", 2, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "brc1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 3UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs1_to_brc1_Workspace_0"), 2UL);
}

// 共享生产者双 Brc 汇入同一 Where: 断边后各有经兄弟 Brc 的残余路径(abs0->brc1->where0<-brc0),
// 逐候选双双拒绝不产生切分; 完整流水线中该等价形态已被上游 CSE/合并收敛(060 报告 11.8 实测)
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForSharedProducerDualBrc) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_where_shared_producer")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Broadcast("brc1", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "brc1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 候选全部被残余连通拒绝, 无 broadcast workspace 候选, 仅 fallback 默认任务
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// 残余连通双拒(变体B保守语义钉死): brc_j/brc_k 各自的残余路径穿过对方的候选切边, 逐候选断边判定
// 双双拒绝, 断边恢复原状, 不产生切分任务(候选边一次性全断开再逐个判的激进实现会双双接受并误切)
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForResidualConnectivity) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_residual_connectivity")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load_q", "data0", {One, s1})
                   .Abs("abs_p", "load_q")
                   .Broadcast("brc_j", "abs_p", {s0, s1})
                   .Abs("abs_q", "load_q")  // brc_k 的计算节点上游, 双候选均可达判定阶段
                   .Broadcast("brc_k", "abs_q", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc_j", "brc_k"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  // 候选全部被连通性拒绝, 无 broadcast workspace 候选, 仅 fallback 默认任务
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// Broadcast 上游为 Data 直连无输入叶子时不收集: Data 走 GM 重读副本、不产生 workspace 边界
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForDataDirectInput) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_data_direct")
                   .Loops({s0, s1})
                   .Data("data0", 0, {One, s1}, {Zero, One}, ge::DT_FLOAT16)
                   .Broadcast("brc0", "data0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// Broadcast 上游 Load(输入非 Data/Workspace)同样不收集(该形态曾触发 PartitionLoadNode 断言中止
// Optimize, Load 上游整体排除后该风险结构性消除)
TEST_F(RecomputeCaseGeneratorTest, TestNoBroadcastWorkspaceCaseForIllegalLoadInput) {
  auto s0 = Sym(1024), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_illegal_load_input")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Load("load1", "load0", {One, s1})
                   .Broadcast("brc0", "load1", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load2", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load2"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}

// 首轴 B 轴为小值(<=128, 低于 reduce 侧 workspace 阈值)时同样识别: recompute 场景不限制首轴大小
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseWithSmallFirstAxis) {
  auto s0 = Sym(2), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_small_first_axis")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
}

// 首轴 B 轴为动态符号(无常量值)时同样识别: 结构判定不依赖首轴常量值; 首轴不参与非首轴乘积,
// s1=256 未超 32K 仍走静态分支生成常量负分劣后
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceCaseWithDynamicFirstAxis) {
  auto s0 = Sym("s0"), s1 = Sym(256);
  auto graph = AscGraphBuilder("brc_dynamic_first_axis")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  EXPECT_EQ(CountSubGraphsWithNode(tasks[0U], "Workspace", "abs0_to_brc0_Workspace_0"), 2UL);
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
}

// 非首轴乘积(40960)超过 32K 阈值: 静态分支生成常量正分, broadcast workspace 模板优先
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceScoreOverThreshold) {
  auto s0 = Sym(1024), s1 = Sym(40960);
  auto graph = AscGraphBuilder("brc_score_over")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceSelectedScoreFunc);
  EXPECT_EQ(tasks[1].score_func, "");
}

// 非首轴为动态符号(s1)时生成运行时代码: 从 tiling data 嵌套结构读 s1 实际值与 32768 比较;
// 单候选场景切分后 g0 为生产侧子图(SortSubGraphsByDependency 依赖排序), s1 同时被生产/消费侧
// 引用, 首个包含它的 group 即 g0
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceScoreDynamicNonFirstAxis) {
  auto s0 = Sym(1024), s1 = Sym("s1");
  auto graph = AscGraphBuilder("brc_score_dynamic")
                   .Loops({s0, s1})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One, s1})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0, s1})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0, s1})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  // 动态分支按表达式引擎实际打印形式做子串断言, 避免与常量折叠/括号细节耦合
  EXPECT_NE(tasks[0].score_func.find("tiling_data.graph0_result0_g0_tiling_data.s1"), std::string::npos);
  EXPECT_NE(tasks[0].score_func.find("return non_first_axis_product > 32768 ? 1 : -1;"), std::string::npos);
  EXPECT_EQ(tasks[1].score_func, "");
}

// 一维图(仅首轴 B 轴)非首轴为空集, 空乘积记 1 恒不超过阈值: 生成常量负分劣后
TEST_F(RecomputeCaseGeneratorTest, TestBroadcastWorkspaceScoreSingleAxis) {
  auto s0 = Sym(1024);
  auto graph = AscGraphBuilder("brc_score_single_axis")
                   .Loops({s0})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0", {One})
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0", {s0})
                   .Data("mask_data", 1, ge::DT_UINT8)
                   .Load("mask_load", "mask_data", {s0})
                   .Data("data1", 2, ge::DT_FLOAT16)
                   .Load("load1", "data1", {s0})
                   .Op<af::ascir_op::Where>("where0", {"mask_load", "brc0", "load1"}, 1)
                   .Store("store0", "where0")
                   .Output("out0", "store0", 0)
                   .Build();
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::RecomputeCaseGenerator generator;
  std::vector<ScheduleTask> tasks;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  EXPECT_EQ(tasks[0].score_func, kBroadcastWorkspaceFallbackScoreFunc);
}
}  // namespace schedule
