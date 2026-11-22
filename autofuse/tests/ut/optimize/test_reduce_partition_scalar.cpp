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
#include <memory>
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "optimize.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::ops;
using namespace af::ascir_op;

namespace {
class ReducePartitionScalarTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStub>());
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
  optimize::Optimizer optimizer{optimize::OptimizerOptions{}};
};
static void AddReducePartitionScalarTail(af::AxisId z0, af::AxisId z1, const af::Expression &s0,
                                         const af::Expression &s1, const af::AscOpOutput &scalar,
                                         const af::AscOpOutput &load, const af::AscOpOutput &broadcast) {
  Sub sub("sub");
  sub.x1 = load;
  sub.x2 = broadcast;
  sub.attr.sched.axis = {z0, z1};
  *sub.y.axis = {z0, z1};
  sub.y.dtype = ge::DT_FLOAT;
  *sub.y.strides = {s1, af::ops::One};
  *sub.y.repeats = {s0, s1};
  Add add1("add1");
  add1.x1 = sub.y;
  add1.x2 = scalar;
  add1.attr.sched.axis = {z0, z1};
  *add1.y.axis = {z0, z1};
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.strides = {s1, af::ops::One};
  *add1.y.repeats = {s0, s1};
  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0, z1};
  store_op1.x = add1.y;
  *store_op1.y.axis = {z0, z1};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, af::ops::One};
  *store_op1.y.repeats = {s0, s1};
  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}
static af::AscGraph BuildReducePartitionScalarGraph() {
  af::AscGraph graph("reduce_partition_scalar");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Scalar scalar("scalar", graph);
  scalar.y.dtype = ge::DT_FLOAT;
  Workspace workspace("workspace");
  graph.AddNode(workspace);
  workspace.attr.sched.axis = {z0.id, z1.id};
  workspace.y.dtype = ge::DT_FLOAT;
  *workspace.y.axis = {z0.id, z1.id};
  *workspace.y.strides = {s1, af::ops::One};
  *workspace.y.repeats = {s0, s1};
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = workspace.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};
  Add add("add");
  add.x1 = scalar.y;
  add.x2 = load.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, af::ops::One};
  *add.y.repeats = {s0, s1};
  Mean mean("b2_mean");
  mean.x = add.y;
  mean.attr.sched.axis = {z0.id, z1.id};
  mean.y.dtype = ge::DT_FLOAT;
  *mean.y.axis = {z0.id, z1.id};
  *mean.y.strides = {af::ops::Zero, af::ops::Zero};
  *mean.y.repeats = {af::ops::One, af::ops::One};
  Broadcast broadcast("broadcast");
  broadcast.x = mean.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.strides = {s1, af::ops::One};
  *broadcast.y.repeats = {s0, s1};
  AddReducePartitionScalarTail(z0.id, z1.id, s0, s1, scalar.y, load.y, broadcast.y);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
  return graph;
}
TEST_F(ReducePartitionScalarTest, ReducePartitionScalar) {
  af::AscGraph graph = BuildReducePartitionScalarGraph();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  // 通用模板 1 + R 轴分核模板 1 (phase_1 各 tiling case 装入同一 group 作为多 impl) + 全载模板 1
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3);
}
}  // namespace
