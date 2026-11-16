/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ascendc_ir.h>
#include <ascir_ops.h>
#include <ascir_utils.h>
#include <iostream>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "graph_utils_ex.h"

#define private public
#include "autoschedule/autoschedule.h"
#include "optimize.h"
#undef private
#include "ascir_ops_utils.h"
#include "autoschedule/tiling_group.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace optimize::autoschedule;

void Construct_Argmax_Reduce_ARAR(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  arg4_1.attr.api.type = af::ApiType::kAPITypeBuffer;
  arg4_1.y.dtype = ge::DT_FLOAT;

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  b0_load.attr.api.type = af::ApiType::kAPITypeCompute;
  b0_load.y.dtype = ge::DT_FLOAT;
  *b0_load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_load.y.repeats = {s0, s1, s2, s3};
  *b0_load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  Abs abs("abs");
  abs.x = b0_load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Argmax b0_argmax("b0_argmax");
  b0_argmax.x = abs.y;
  b0_argmax.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b0_argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  b0_argmax.attr.api.type = af::ApiType::kAPITypeCompute;
  b0_argmax.y.dtype = ge::DT_INT32;  // Argmax outputs int32 indices
  *b0_argmax.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b0_argmax.y.repeats = {s0, One, s2, One};
  *b0_argmax.y.strides = {s2, Zero, One, Zero};

  Abs abs1("abs1");
  abs1.x = b0_argmax.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs1.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;
  abs1.y.dtype = ge::DT_INT32;
  *abs1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs1.y.repeats = {s0, One, s2, One};
  *abs1.y.strides = {s2, Zero, One, Zero};

  Store b3_store("b3_store");
  b3_store.x = abs1.y;
  b3_store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  b3_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  b3_store.attr.api.type = af::ApiType::kAPITypeCompute;
  b3_store.y.dtype = ge::DT_INT32;
  *b3_store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *b3_store.y.repeats = {s0, One, s2, One};
  *b3_store.y.strides = {s2, Zero, One, Zero};

  Output buf3("buf3");
  buf3.x = b3_store.y;
  buf3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  buf3.attr.api.type = af::ApiType::kAPITypeBuffer;
  buf3.y.dtype = ge::DT_INT32;
}

namespace optimize {
class AutoSchedulerArgmaxReduceUT : public ::testing::Test {
  void SetUp() override {}

 protected:
  optimize::Optimizer optimizer_;
  AutoSchedulerArgmaxReduceUT() : optimizer_(optimize::OptimizerOptions{}) {};
};

TEST_F(AutoSchedulerArgmaxReduceUT, Autoschedule_argmax_reduce_arar_fusion_rcore) {
  af::AscGraph graph("Argmax_Reduce_ARAR");
  Construct_Argmax_Reduce_ARAR(graph);

  af::AscGraph except_graph("Argmax_Reduce_ARAR_1");
  except_graph.CopyFrom(graph);

  std::vector<autoschedule::AutoScheduleOutput> impl_graphs;
  AutoSchedule autoschedule(graph, impl_graphs, true);
  autoschedule.DoAutoSchedule();
  EXPECT_EQ(impl_graphs.size(), 4);
}

}  // namespace optimize
