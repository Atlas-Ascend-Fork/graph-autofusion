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

// 与 test_optimizer.cpp 同款测试侧访问约定：Optimizer::RemoveAllZeroStrideLoopAxis 为
// private static 方法，仓库 UT 依赖本宏放开访问（须先于 optimize.h 等项目头包含）。
#define private public

#include "ascendc_ir.h"
#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "autoschedule/tiling_group.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "optimize.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::ops;
using namespace af::ascir_op;

namespace {

class RemoveZeroStrideAxisTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<ge::RuntimeStub>();
    ge::RuntimeStub::SetInstance(stub_v1);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

// 建图产物的轴信息回传，供断言使用。注意用 af::AxisId 而非 ascir::AxisId：本文件同时
// include ascendc_ir.h（namespace af::ascir）与 ascir.h（顶层 namespace ascir），在
// using namespace af 作用下非限定 ascir 存在二义（test_optimizer.cpp 同场景亦不使用该前缀）。
struct ReduceGraphAxes {
  af::AxisId z0_id;
  af::AxisId z1_id;
  af::Expression s0;
  af::Expression s1;
};

// 按场景生成视图参数：z1_broadcast 时 z1 为端到端广播轴（load 侧 stride=0、reduce 输出
// 全零折叠）；否则 z1 为真实数据轴（reduce 保留 z1 输出）。sum/store 视图同构共用。
struct ReduceViewParams {
  std::vector<af::Expression> load_strides;
  std::vector<af::Expression> out_strides;
  std::vector<af::Expression> out_repeats;
};

ReduceViewParams MakeViewParams(const bool z1_broadcast, const af::Expression &s1) {
  if (z1_broadcast) {
    return {{One, Zero}, {Zero, Zero}, {One, One}};
  }
  return {{s1, One}, {Zero, One}, {One, s1}};
}

// 构造 Data→Load→Sum→Store→Output 归约图。z1_broadcast=true 时 z1 为端到端广播轴
// （全图视图 stride=0，被 RemoveAllZeroStrideLoopAxis 删除后图退化为纯 R）；
// false 时 z1 为真实数据轴（非退化对照）。with_workspace 时附加 Workspace
// （buffer 节点，sched 轴不被删轴更新）。
ReduceGraphAxes BuildReduceGraph(AscGraph &graph, const bool with_workspace, const bool z1_broadcast) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  if (with_workspace) {
    Workspace workspace("workspace");
    graph.AddNode(workspace);
    workspace.attr.sched.axis = {z0.id, z1.id};
  }

  const auto params = MakeViewParams(z1_broadcast, s1);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = params.load_strides;
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = params.out_strides;
  *sum.y.repeats = params.out_repeats;

  Store store("store");
  store.attr.sched.axis = {z0.id, z1.id};
  store.x = sum.y;
  *store.y.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.strides = params.out_strides;
  *store.y.repeats = params.out_repeats;

  Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;
  output.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  return {z0.id, z1.id, s0, s1};
}

// 校验删轴后补偿效果：axis_1d 追加在 sched 轴最前（既有轴 id 不重排）、输出视图头插
// repeat=1 且 stride=repeats[0]*strides[0]、y_group 非空且首轴为 axis_1d
void VerifyExpandDimsResult(AscGraph &graph, const ReduceGraphAxes &axes) {
  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  ASSERT_EQ(load_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(load_node->attr.sched.axis[1UL], axes.z0_id);
  auto expand_axis = graph.FindAxis(load_node->attr.sched.axis[0UL]);
  ASSERT_NE(expand_axis, nullptr);
  EXPECT_EQ(expand_axis->name, "axis_1d");
  EXPECT_EQ(load_node->outputs[0].attr.axis, load_node->attr.sched.axis);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(load_node->outputs[0].attr.repeats[0UL], One), af::TriBool::kTrue);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(load_node->outputs[0].attr.strides[0UL], axes.s0), af::TriBool::kTrue);

  optimize::autoschedule::AxisGroup axes_group;
  EXPECT_EQ(optimize::autoschedule::TilingGroup::GenTilingGroup(graph, axes_group), 0);
  EXPECT_FALSE(axes_group.y_group.empty());
  EXPECT_EQ(axes_group.y_group[0UL], expand_axis->id);
}

// 构造 3 轴部分归约图 Data→Load→Sum→Store→Output：z0 真实归约轴、z1 真实保留轴
// （reduce 输出 stride 非零）、z2 端到端广播轴（全图视图 stride=0）。删 z2 后剩余
// {z0, z1} 中 z1 非归约轴，不满足全归约、不补 axis_1d。
ReduceGraphAxes BuildPartialReduceGraph(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 * s2, s2, Zero};
  *load.y.repeats = {s0, s1, s2};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id, z2.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id, z2.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {Zero, One, Zero};
  *sum.y.repeats = {One, s1, One};

  Store store("store");
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.x = sum.y;
  *store.y.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.strides = {Zero, One, Zero};
  *store.y.repeats = {One, s1, One};

  Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;
  output.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  return {z0.id, z1.id, s0, s1};
}

// 构造计算节点 sched 轴分歧图：主链 Load/Sum 的 sched 为 {z0, z1}（z1 广播），
// Store 的 sched 仅为 {z1}。删 z1 后主链剩 {z0}（全归约命中），但 Store 剩 {z1}
// 与 Reduce 不一致 → ComputeNodesShareSchedAxis false → GELOGW 跳过补偿。
ReduceGraphAxes BuildMismatchGraph(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {One, Zero};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.strides = {Zero, Zero};
  *sum.y.repeats = {One, One};

  Store store("store");
  store.attr.sched.axis = {z1.id};
  store.x = sum.y;
  *store.y.axis = {z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.strides = {Zero};
  *store.y.repeats = {One};

  Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;
  output.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  return {z0.id, z1.id, s0, s1};
}

// 构造含 Gather 的图：RemoveAllZeroStrideLoopAxis 对 Gather 图整体跳过（入口守卫）
void BuildGatherGraph(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.repeats = {s0};
  *(x1.y.axis) = {z0.id};

  Data x2("x2", graph);
  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = ge::DT_INT64;
  *x2.y.repeats = {s0, s1};
  *(x2.y.axis) = {z0.id, z1.id};

  Gather gather("gather");
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z0.id, z1.id};
  gather.ir_attr.SetAxis(0);
  gather.y.dtype = ge::DT_FLOAT;
  *gather.y.axis = {z0.id, z1.id};
  *gather.y.repeats = {s0, s1};
  *gather.y.strides = {One, Zero};
  gather.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  Store store("store");
  store.x = gather.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {One, Zero};

  Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;
  output.ir_attr.SetIndex(0);

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

}  // namespace

// 删轴后退化全归约补偿：z1 被删后剩余 z0 全为归约轴（纯 R 图），就地补 axis_1d 保证
// y_group 非空。修复前：GraphPass 期判定因广播轴 src == dst == 0 不成立不补轴，
// GenTilingCase 枚举空集触发 "No valid tiling cases" 断言失败。
TEST_F(RemoveZeroStrideAxisTest, DegenToAllReduceExpandDims) {
  AscGraph graph("degen_all_reduce_after_remove");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  const auto axes = BuildReduceGraph(graph, false, true);

  auto sum_node = graph.FindNode("sum");
  ASSERT_NE(sum_node, nullptr);
  EXPECT_EQ(sum_node->attr.sched.axis.size(), 2UL);  // 删轴前广播轴不满足全归约判定，不补轴

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  // 删 z1 后未补偿则 sched 轴剩 1 根；补偿后为 {axis_1d, z0}
  EXPECT_EQ(sum_node->attr.sched.axis.size(), 2UL);
  VerifyExpandDimsResult(graph, axes);
}

// 图内含 Workspace（buffer 节点，删轴不更新其 sched 轴）：补偿的比对/同步范围与删轴
// 循环一致（跳过 buffer 节点），Workspace 保持原样且旧轴 id 因追加建轴仍有效。
// 修复前：Workspace 的陈旧 sched 轴被纳入一致性判定 → mismatch → GELOGW 跳过补偿，
// 图仍为纯 R，下游 "No valid tiling cases"。
TEST_F(RemoveZeroStrideAxisTest, DegenWithStaleWorkspace) {
  AscGraph graph("degen_all_reduce_with_workspace");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  const auto axes = BuildReduceGraph(graph, true, true);

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  auto workspace_node = graph.FindNode("workspace");
  ASSERT_NE(workspace_node, nullptr);
  ASSERT_EQ(workspace_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(workspace_node->attr.sched.axis[0UL], axes.z0_id);
  EXPECT_EQ(workspace_node->attr.sched.axis[1UL], axes.z1_id);
  EXPECT_EQ(graph.FindAxis(axes.z0_id)->name, "z0");
  EXPECT_EQ(graph.FindAxis(axes.z1_id)->name, "z1");

  VerifyExpandDimsResult(graph, axes);
}

// 非退化图不受补偿影响：z1 为真实数据轴（stride 非零）时不删轴、不补 axis_1d
TEST_F(RemoveZeroStrideAxisTest, NonDegenKeepsAxes) {
  AscGraph graph("non_degen_remove_zero_stride");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  const auto axes = BuildReduceGraph(graph, false, false);

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(load_node->attr.sched.axis[0UL], axes.z0_id);
  EXPECT_EQ(load_node->attr.sched.axis[1UL], axes.z1_id);
  EXPECT_EQ(graph.FindAxis(load_node->attr.sched.axis[0UL])->name, "z0");

  optimize::autoschedule::AxisGroup axes_group;
  EXPECT_EQ(optimize::autoschedule::TilingGroup::GenTilingGroup(graph, axes_group), 0);
  EXPECT_FALSE(axes_group.y_group.empty());
}

// 删轴发生但 reduce 非全归约（z1 为真实保留轴）：广播轴 z2 被删后剩余 {z0, z1}，
// IsDegeneratedAllReduce 判定不成立，不补 axis_1d、无补偿副作用（覆盖率：扫描
// continue 与 new_axis_ids 空返回分支）
TEST_F(RemoveZeroStrideAxisTest, PartialReduceAfterRemoveNoExpand) {
  AscGraph graph("partial_reduce_after_remove");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  const auto axes = BuildPartialReduceGraph(graph);

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  // 广播轴 z2 被删（3 轴 → 2 轴），但未插入 axis_1d（非全归约不补偿）
  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  ASSERT_EQ(load_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(load_node->attr.sched.axis[0UL], axes.z0_id);
  EXPECT_EQ(load_node->attr.sched.axis[1UL], axes.z1_id);
  EXPECT_EQ(graph.GetAllAxis().size(), 3UL);  // 轴池无新增

  // z1 为真实非归约轴，y_group 天然非空
  optimize::autoschedule::AxisGroup axes_group;
  EXPECT_EQ(optimize::autoschedule::TilingGroup::GenTilingGroup(graph, axes_group), 0);
  EXPECT_FALSE(axes_group.y_group.empty());
}

// 计算节点 sched 轴分歧（Store 为 {z1}，主链为 {z0}）：删轴后虽命中全归约，
// 但 ComputeNodesShareSchedAxis 判定不一致 → GELOGW 跳过补偿，图保持删轴后
// 原样（覆盖率：mismatch 分支）
TEST_F(RemoveZeroStrideAxisTest, ComputeNodeSchedMismatchSkip) {
  AscGraph graph("compute_node_sched_mismatch");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  const auto axes = BuildMismatchGraph(graph);

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  // 广播轴 z1 被删：主链剩 {z0} 且无 axis_1d（mismatch 跳过补偿）
  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  ASSERT_EQ(load_node->attr.sched.axis.size(), 1UL);
  EXPECT_EQ(load_node->attr.sched.axis[0UL], axes.z0_id);
  EXPECT_EQ(graph.GetAllAxis().size(), 2UL);  // 轴池无新增
  // Store 的 sched 轴无被删位置（z1 在其位置 0），保持原样
  auto store_node = graph.FindNode("store");
  ASSERT_NE(store_node, nullptr);
  EXPECT_EQ(store_node->attr.sched.axis.size(), 1UL);
  EXPECT_EQ(store_node->attr.sched.axis[0UL], axes.z1_id);
}

// 含 Gather 的图：RemoveAllZeroStrideLoopAxis 入口守卫整体跳过，sched 轴与
// 轴池完全不变（覆盖率：gather 提前返回分支）
TEST_F(RemoveZeroStrideAxisTest, GatherGraphSkipRemove) {
  AscGraph graph("gather_graph_skip_remove");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  BuildGatherGraph(graph);

  ASSERT_EQ(optimize::Optimizer::RemoveAllZeroStrideLoopAxis(graph), af::SUCCESS);

  auto gather_node = graph.FindNode("gather");
  ASSERT_NE(gather_node, nullptr);
  EXPECT_EQ(gather_node->attr.sched.axis.size(), 2UL);
  EXPECT_EQ(gather_node->outputs[0].attr.repeats.size(), 2UL);
  EXPECT_EQ(graph.GetAllAxis().size(), 2UL);  // 轴池无新增
}
