/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgen_log.h"
#define private public
#include "optimize.h"
#include "platform_context.h"
#undef private
#include "graph/ascendc_ir/utils/asc_graph_utils.h"

using namespace af;

class OptimizerGatherSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
  }

  optimize::Optimizer optimizer;

  OptimizerGatherSt() : optimizer(optimize::OptimizerOptions{}) {}
};
void AddGatherAbsStoreOutput(af::AscGraph &graph, af::ascir_op::Gather &gather, const std::vector<af::AxisId> &axes,
                             const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.x = gather.y;
  abs.attr.sched.axis = axes;
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = axes;
  *abs.y.repeats = repeats;
  *abs.y.strides = strides;

  af::ascir_op::Store store("store");
  store.attr.api.compute_type = ComputeType::kComputeElewise;
  store.x = abs.y;
  store.attr.sched.axis = axes;
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = axes;
  *store.y.repeats = repeats;
  *store.y.strides = strides;

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = axes;
  y.y.dtype = af::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

af::Expression BuildGatherLast1DimGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, s2};
  *data0.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT;
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z3.id, z4.id};
  *data1.y.axis = {z3.id, z4.id};
  *data1.y.repeats = {s3, s4};
  *data1.y.strides = {s4, af::ops::One};

  af::ascir_op::Gather gather("gather");
  gather.attr.api.compute_type = ComputeType::kComputeGather;
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(2);
  gather.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  gather.y.dtype = af::DT_FLOAT;
  *gather.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s3, s4};
  *gather.y.strides = {s1 * s3 * s4, s3 * s4, s4, af::ops::One};
  AddGatherAbsStoreOutput(graph, gather, {z0.id, z1.id, z3.id, z4.id}, {s0, s1, s3, s4},
                          {s1 * s3 * s4, s3 * s4, s4, af::ops::One});

  auto axis = graph.GetAllAxis();
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  graph_attr->axis = axis;
  return s0 * s1;
}

TEST_F(OptimizerGatherSt, gather_last1dim) {
  af::AscGraph graph("LoadAbsStore");
  const auto merged_size = BuildGatherLast1DimGraph(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  ASSERT_EQ(impl_graphs.size(), 2UL);
  for (auto &impl_graph : impl_graphs) {
    ASSERT_NE(impl_graph.FindNode("gather"), nullptr);
    ASSERT_NE(impl_graph.FindNode("store"), nullptr);
    bool has_merged_batch_axis = false;
    for (const auto &axis : impl_graph.GetAllAxis()) {
      if (axis->name == "z0z1") {
        EXPECT_EQ(axis->size, merged_size);
        has_merged_batch_axis = true;
      }
      if (axis->type != af::Axis::Type::kAxisTypeTileInner) {
        continue;
      }
      ASSERT_EQ(axis->from.size(), 1UL);
      const af::Axis *parent = impl_graph.FindAxis(axis->from.front());
      ASSERT_NE(parent, nullptr);
      EXPECT_EQ(parent->type, af::Axis::Type::kAxisTypeOriginal);
    }
    EXPECT_TRUE(has_merged_batch_axis);
  }
}
