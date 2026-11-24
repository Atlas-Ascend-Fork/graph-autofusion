/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"

namespace optimize {
namespace indirect_load_detail {
namespace {

using namespace af::ascir_op;

// 构造一个 Data->Load->Broadcast->Store->Output 的最小图，用于取到 Broadcast 节点。
af::AscGraph BuildBroadcastGraph() {
  af::AscGraph graph("indirect_load_view_layout_ut");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::Symbol(1)};

  af::ascir_op::Load load("load");
  load.x = x1.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::Symbol(1)};
  *load.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load.y;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, af::Symbol(1)};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = brc.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::Symbol(1)};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, af::Symbol(1)};
  return graph;
}

class IndirectLoadViewLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

}  // namespace

// broadcast_index 越界时应直接淘汰当前 candidate（fail-closed），不做任何路径改写。
TEST_F(IndirectLoadViewLayoutTest, IsSupportedBroadcastPath_ShouldReject_WhenBroadcastIndexOutOfRange) {
  NodePath path;
  path.emplace_back(nullptr);
  ASSERT_FALSE(IsSupportedBroadcastPath(path, path.size(), ascir::TemplateId::kIndirectLoadSimd));
  ASSERT_FALSE(IsSupportedBroadcastPath(path, path.size() + 1U, ascir::TemplateId::kIndirectLoadSimt));
}

// 空路径下任意索引都越界，同样应被拒绝。
TEST_F(IndirectLoadViewLayoutTest, IsSupportedBroadcastPath_ShouldReject_WhenPathIsEmpty) {
  NodePath path;
  ASSERT_TRUE(path.empty());
  ASSERT_FALSE(IsSupportedBroadcastPath(path, 0U, ascir::TemplateId::kIndirectLoadSimd));
}

// sched rank 与物理布局 rank 不一致时，节点视图应保持原样（Preserve 分支），
// 不得被改写为 layout 视图。
TEST_F(IndirectLoadViewLayoutTest, ApplyPhysicalView_ShouldPreserveNode_WhenSchedRankMismatch) {
  auto graph = BuildBroadcastGraph();
  auto broadcast = graph.FindNode("brc");
  ASSERT_NE(broadcast, nullptr);

  NodePath path{broadcast};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  layout.axis_ids = {0, 1, 2};  // rank 3，与 sched.axis(rank 2) 不一致
  layout.sizes = {af::Symbol(2), af::Symbol(2), af::Symbol(2)};
  layout.physical_repeats = {af::Symbol(2), af::Symbol(2), af::Symbol(2)};
  layout.strides = {af::Symbol(4), af::Symbol(2), af::Symbol(1)};

  const auto sched_axis_before = broadcast->attr.sched.axis;
  const auto repeats_before = broadcast->outputs[0].attr.repeats;

  ASSERT_EQ(ApplyPhysicalView(path, layout), af::SUCCESS);
  EXPECT_EQ(broadcast->attr.sched.axis, sched_axis_before);
  EXPECT_EQ(broadcast->outputs[0].attr.repeats, repeats_before);
}

// rank 一致时按 layout 重写 sched.axis / repeats / strides（对照用例）。
TEST_F(IndirectLoadViewLayoutTest, ApplyPhysicalView_ShouldRewriteView_WhenSchedRankMatched) {
  auto graph = BuildBroadcastGraph();
  auto broadcast = graph.FindNode("brc");
  ASSERT_NE(broadcast, nullptr);

  NodePath path{broadcast};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  layout.axis_ids = {7, 8};  // 与 sched.axis 同 rank
  layout.sizes = {af::Symbol(3), af::Symbol(4)};
  layout.physical_repeats = {af::Symbol(3), af::Symbol(4)};
  layout.strides = {af::Symbol(4), af::Symbol(1)};

  ASSERT_EQ(ApplyPhysicalView(path, layout), af::SUCCESS);
  EXPECT_EQ(broadcast->attr.sched.axis, layout.axis_ids);
  EXPECT_EQ(broadcast->outputs[0].attr.repeats, layout.physical_repeats);
  EXPECT_EQ(broadcast->outputs[0].attr.strides, layout.strides);
}

}  // namespace indirect_load_detail
}  // namespace optimize
