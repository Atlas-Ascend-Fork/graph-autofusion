/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "gtest/gtest.h"
#include "ascendc_ir_def.h"
#define private public
#include "autoschedule/alignment_handler.h"
#undef private
#include "ascir_ops_utils.h"
#include "runtime_stub.h"
#include "schedule_utils.h"
#include "platform_context.h"
#include "platformv2.h"
#include "un_alignment_strategy.h"
#include "indirect_load_utils.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace optimize::autoschedule;

namespace optimize {
using namespace ge;

template <typename Op, typename Dtype>
void SetTwoDimNodeAttrV2(Op &op, Dtype dtype, af::ComputeType compute_type, const std::vector<af::AxisId> &axes,
                         const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  op.attr.api.compute_type = compute_type;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
  *op.y.vectorized_axis = axes;
}

void ExpectCompactReduceBranchV2(const af::AscGraph &graph, const char *aligned_input) {
  std::vector<af::Expression> aligned_input_strides = {af::Symbol(24), One};
  std::vector<af::Expression> compact_output_strides = {One, Zero};
  EXPECT_EQ(graph.FindNode(aligned_input)->outputs[0].attr.vectorized_strides, aligned_input_strides);
  EXPECT_EQ(graph.FindNode("max")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("sum")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("scalar_broadcast")->outputs[0].attr.vectorized_strides, compact_output_strides);
  EXPECT_EQ(graph.FindNode("div")->outputs[0].attr.vectorized_strides, compact_output_strides);
}

void AddTailBrcRemovePadBroadcasts(af::ascir_op::Load &load, const std::vector<af::AxisId> &axes) {
  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load.y;
  brc1.attr.sched.axis = axes;
  brc1.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc1.y.dtype = af::DT_FLOAT;
  *brc1.y.axis = axes;
  *brc1.y.repeats = {af::Symbol(2), af::Symbol(3), af::Symbol(5)};
  *brc1.y.strides = {af::Symbol(15), af::Symbol(5), One};
  *brc1.y.vectorized_axis = axes;

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = load.y;
  brc2.attr.sched.axis = axes;
  brc2.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc2.y.dtype = af::DT_FLOAT;
  *brc2.y.axis = axes;
  *brc2.y.repeats = {af::Symbol(2), af::Symbol(3), af::Symbol(4)};
  *brc2.y.strides = {af::Symbol(12), af::Symbol(4), One};
  *brc2.y.vectorized_axis = axes;
}

af::AscGraph BuildTailBrcRemovePadGraph() {
  af::AscGraph graph("tail_brc_remove_pad_unique_name");
  auto z0 = graph.CreateAxis("z0", af::Symbol(2));
  auto z1 = graph.CreateAxis("z1", af::Symbol(3));
  auto z2 = graph.CreateAxis("z2", af::Symbol(5));
  const std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id};
  const std::vector<af::Expression> src_repeats = {af::Symbol(2), af::Symbol(3), One};
  const std::vector<af::Expression> src_strides = {af::Symbol(3), One, Zero};

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  *data.y.axis = axes;
  *data.y.repeats = src_repeats;
  *data.y.strides = src_strides;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.sched.axis = axes;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = axes;
  *load.y.repeats = src_repeats;
  *load.y.strides = src_strides;
  *load.y.vectorized_axis = axes;
  // float 退化尾轴 pad 后的 32B 槽位布局：Align(1, 32/4) == 8
  *load.y.vectorized_strides = {af::Symbol(24), af::Symbol(8), Zero};

  AddTailBrcRemovePadBroadcasts(load, axes);
  return graph;
}

class VectorizedAlignmentUTV2 : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(VectorizedAlignmentUTV2, brc_align_input_not_align) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(999);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, One};
  *load0.y.strides = {One, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Broadcast brc("brc");
  brc.x = load0.y;
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  brc.attr.sched.axis = {z0.id, z1.id};
  brc.y.dtype = af::DT_FLOAT;
  *brc.y.axis = {z0.id, z1.id};
  *brc.y.repeats = {s0, s1};
  *brc.y.strides = {s1, One};
  *brc.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s2};
  *load1.y.strides = {One, Zero};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Concat concat("concat");
  concat.x = {load1.y, brc.y};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = af::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s2};
  *concat.y.strides = {s1 + s2, One};
  *concat.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s2};
  *store.y.strides = {s1 + s2 + s0, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  auto brc_node = graph.FindNode("brc");
  ASSERT_NE(brc_node, nullptr);
  auto concat_node = graph.FindNode("concat");
  ASSERT_NE(concat_node, nullptr);

  std::vector<af::Expression> golden_strides_load1 = {One, Zero};
  std::vector<af::Expression> golden_strides_load0 = {One, Zero};
  std::vector<af::Expression> golden_strides_brc = {s1 + s2, One};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides_load0);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden_strides_load1);
  EXPECT_EQ(concat_node->outputs[0].attr.vectorized_strides, golden_strides_brc);
}

TEST_F(VectorizedAlignmentUTV2, discontinuous_write_not_align) {
  af::AscGraph graph("Concat");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(1);
  auto stride_val = af::Symbol(66);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, Zero};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Store store("store");
  store.x = load0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {stride_val, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto stode_node = graph.FindNode("store");
  ASSERT_NE(stode_node, nullptr);

  std::vector<af::Expression> golden_strides = {s1, Zero};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(stode_node->outputs[0].attr.vectorized_strides, golden_strides);
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = af::ops::One;

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id, z2.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = af::DT_FLOAT16;
  *x.y.axis = {z0.id, z1.id, z2.id};
  *x.y.repeats = {s0, s1, s2};
  *x.y.strides = {s1, One, Zero};

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = af::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1, One, Zero};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1, One, Zero};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1, One, Zero};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1, One, Zero};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  }

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  auto abs_node = graph.FindNode("abs");

  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[0].Str().get()), "s1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
  EXPECT_EQ(std::string(abs_node->outputs[0].attr.vectorized_strides[2].Str().get()), "0");
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides_last_zero) {
  af::AscGraph graph("align_vectorize");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(20);
  auto s3 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2, s2, One, Zero};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto load_node = graph.FindNode("load_i");
  std::vector<af::Expression> golden_stride = {af::Symbol(20), One, Zero};
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, reduce_alignment) {
  af::AscGraph graph("reduce_align");
  auto s0 = af::Symbol(7);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(9);
  auto s3 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.y.dtype = af::DT_FLOAT16;
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, s3};
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Max max("max");
  max.y.dtype = af::DT_FLOAT16;
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, One, s3};
  *max.y.strides = {s1 * s3, s3, Zero, One};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Store store("store");
  store.y.dtype = af::DT_FLOAT16;
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, One, s3};
  *store.y.strides = {s1 * s3, s3, Zero, One};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto s16 = af::Symbol(16);
  std::vector<af::Expression> load_golden_stride = {s2 * s16, s16, One};
  std::vector<af::Expression> golden_stride = {s16, Zero, One};

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, load_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, align_vectorized_strides_by_repeat) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(9);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.y.dtype = af::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, One};

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, Zero};

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.attr.api.compute_type = af::ComputeType::kComputeConcat;
  concat.y.dtype = af::DT_FLOAT;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + One};
  *concat.y.strides = {s1 + One, One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + One};
  *store.y.strides = {s1 + One, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  }

  AlignmentHandler handler;
  EXPECT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);
  ::ascir::utils::DumpImplGraphs({graph}, "xxx");

  auto load1_node = graph.FindNode("load1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load1_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");

  auto load_node = graph.FindNode("load");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "9");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(VectorizedAlignmentUTV2, tail_brc_tail_reduce_alignment) {
  af::AscGraph graph("tail_brc_tail_reduce");
  auto s0 = af::Symbol(7);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(9);
  auto s3 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.y.dtype = af::DT_FLOAT16;
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load.y.repeats = {s0, s1, s2, One};
  *load.y.strides = {s1 * s2, s2, One, Zero};
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *load.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Broadcast brc("brc");
  brc.y.dtype = af::DT_FLOAT16;
  brc.x = load.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.repeats = {s0, s1, s2, s3};
  *brc.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  *brc.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = brc.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *abs.y.repeats = {s0, s1, s2, s3};
  *abs.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  *abs.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Max max("max");
  max.x = brc.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, s2, One};
  *max.y.strides = {s1 * s2, s2, One, Zero};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Max max1("max1");
  max1.x = brc.y;
  max1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max1.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max1.y.repeats = {s0, s1, s2, One};
  *max1.y.strides = {s1 * s2, s2, One, Zero};
  max1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  *max1.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Add add("add");
  add.x1 = max.y;
  add.x2 = max1.y;
  add.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  add.y.dtype = af::DT_FLOAT16;
  *add.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *add.y.repeats = {s0, s1, s2, One};
  *add.y.strides = {s1 * s2, s2, One, Zero};
  *add.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Store store("store");
  store.y.dtype = af::DT_FLOAT16;
  store.x = add.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, One};
  *store.y.strides = {s1 * s2, s2, One, Zero};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto s16 = af::Symbol(16);
  std::vector<af::Expression> load_golden_stride = {s2, One, Zero};
  std::vector<af::Expression> brc_golden_stride = {s2 * s16, s16, One};
  std::vector<af::Expression> max_golden_stride = {s2, One, Zero};

  auto load_node = graph.FindNode("load");  // (9, 1, 0)
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, load_golden_stride);

  auto brc_node = graph.FindNode("brc");  // (9 * 16, 16, 1)
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, brc_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, max_golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, max_golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, scalar_brc_tail_reduce_alignment) {
  af::AscGraph graph("scalar_brc_tail_reduce");
  auto s0 = af::Symbol(7);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(9);
  auto s3 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Scalar scalar0("data0", graph);
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = af::DT_FLOAT16;
  scalar0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast brc("brc");
  brc.y.dtype = af::DT_FLOAT16;
  brc.x = scalar0.y;
  brc.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *brc.y.repeats = {s0, s1, s2, s3};
  *brc.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  brc.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  *brc.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Max max("max");
  max.x = brc.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *max.y.repeats = {s0, s1, s2, One};
  *max.y.strides = {s1 * s2, s2, One, Zero};
  max.attr.api.compute_type = af::ComputeType::kComputeReduce;
  *max.y.vectorized_axis = {z1.id, z2.id, z3.id};

  af::ascir_op::Store store("store");
  store.y.dtype = af::DT_FLOAT16;
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store.y.repeats = {s0, s1, s2, One};
  *store.y.strides = {s1 * s2, s2, One, Zero};
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  *store.y.vectorized_axis = {z1.id, z2.id, z3.id};

  EXPECT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);

  auto s16 = af::Symbol(16);
  std::vector<af::Expression> load_golden_stride = {s2, One, Zero};
  std::vector<af::Expression> brc_golden_stride = {s2 * s16, s16, One};
  std::vector<af::Expression> max_golden_stride = {s2, One, Zero};

  auto brc_node = graph.FindNode("brc");  // (9 * 16, 16, 1)
  EXPECT_EQ(brc_node->outputs[0].attr.vectorized_strides, brc_golden_stride);

  auto reduce_node = graph.FindNode("max");
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, max_golden_stride);

  auto store_node = graph.FindNode("store");
  EXPECT_EQ(store_node->outputs[0].attr.vectorized_strides, max_golden_stride);
}

TEST_F(VectorizedAlignmentUTV2, sibling_reduce_keeps_compact_output_alignment) {
  af::AscGraph graph("sibling_reduce_alignment");
  auto rows = af::Symbol(10);
  auto cols = af::Symbol(22);
  auto row_axis = graph.CreateAxis("row", rows);
  auto col_axis = graph.CreateAxis("col", cols);
  std::vector<af::AxisId> axes = {row_axis.id, col_axis.id};

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  SetTwoDimNodeAttrV2(load, af::DT_FLOAT, af::ComputeType::kComputeLoad, axes, {rows, cols}, {cols, One});

  af::ascir_op::Relu relu("relu");
  relu.x = load.y;
  SetTwoDimNodeAttrV2(relu, af::DT_FLOAT, af::ComputeType::kComputeElewise, axes, {rows, cols}, {cols, One});

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("22.0");
  scalar.y.dtype = af::DT_FLOAT;
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast scalar_broadcast("scalar_broadcast");
  scalar_broadcast.x = scalar.y;
  SetTwoDimNodeAttrV2(scalar_broadcast, af::DT_FLOAT, af::ComputeType::kComputeBroadcast, axes, {rows, One},
                      {One, Zero});

  af::ascir_op::Max max("max");
  max.x = relu.y;
  SetTwoDimNodeAttrV2(max, af::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store max_store("max_store");
  max_store.x = max.y;
  SetTwoDimNodeAttrV2(max_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::Sum sum("sum");
  sum.x = relu.y;
  SetTwoDimNodeAttrV2(sum, af::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store sum_store("sum_store");
  sum_store.x = sum.y;
  SetTwoDimNodeAttrV2(sum_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::TrueDiv div("div");
  div.x1 = sum.y;
  div.x2 = scalar_broadcast.y;
  SetTwoDimNodeAttrV2(div, af::DT_FLOAT, af::ComputeType::kComputeElewise, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store div_store("div_store");
  div_store.x = div.y;
  SetTwoDimNodeAttrV2(div_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  ExpectCompactReduceBranchV2(graph, "relu");
}

TEST_F(VectorizedAlignmentUTV2, tail_broadcast_does_not_propagate_alignment_through_reduce) {
  af::AscGraph graph("tail_broadcast_reduce_alignment");
  auto rows = af::Symbol(10);
  auto cols = af::Symbol(22);
  auto row_axis = graph.CreateAxis("row", rows);
  auto col_axis = graph.CreateAxis("col", cols);
  std::vector<af::AxisId> axes = {row_axis.id, col_axis.id};

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load");
  load.x = data.y;
  SetTwoDimNodeAttrV2(load, af::DT_FLOAT, af::ComputeType::kComputeLoad, axes, {rows, One}, {One, Zero});

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = load.y;
  SetTwoDimNodeAttrV2(broadcast, af::DT_FLOAT, af::ComputeType::kComputeBroadcast, axes, {rows, cols}, {cols, One});

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("22.0");
  scalar.y.dtype = af::DT_FLOAT;
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Broadcast scalar_broadcast("scalar_broadcast");
  scalar_broadcast.x = scalar.y;
  SetTwoDimNodeAttrV2(scalar_broadcast, af::DT_FLOAT, af::ComputeType::kComputeBroadcast, axes, {rows, One},
                      {One, Zero});

  af::ascir_op::Max max("max");
  max.x = broadcast.y;
  SetTwoDimNodeAttrV2(max, af::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store max_store("max_store");
  max_store.x = max.y;
  SetTwoDimNodeAttrV2(max_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::Sum sum("sum");
  sum.x = broadcast.y;
  SetTwoDimNodeAttrV2(sum, af::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store sum_store("sum_store");
  sum_store.x = sum.y;
  SetTwoDimNodeAttrV2(sum_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  af::ascir_op::TrueDiv div("div");
  div.x1 = sum.y;
  div.x2 = scalar_broadcast.y;
  SetTwoDimNodeAttrV2(div, af::DT_FLOAT, af::ComputeType::kComputeElewise, axes, {rows, One}, {One, Zero});

  af::ascir_op::Store div_store("div_store");
  div_store.x = div.y;
  SetTwoDimNodeAttrV2(div_store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {rows, One}, {One, Zero});

  ASSERT_EQ(AlignmentHandler::AlignVectorizedStrides(graph), af::SUCCESS);
  ExpectCompactReduceBranchV2(graph, "broadcast");
  EXPECT_EQ(graph.FindNode("sum_0_pad"), nullptr);
}

TEST_F(VectorizedAlignmentUTV2, tail_brc_remove_pad_unique_name_for_fanout) {
  af::AscGraph graph = BuildTailBrcRemovePadGraph();
  optimize::UnAlignmentStrategy strategy;
  ASSERT_EQ(strategy.ModifyVectorizedStrides(graph), af::SUCCESS);

  auto rp1 = graph.FindNode("load_remove_pad_0_brc_brc1");
  auto rp2 = graph.FindNode("load_remove_pad_0_brc_brc2");
  ASSERT_NE(rp1, nullptr);
  ASSERT_NE(rp2, nullptr);
  EXPECT_NE(rp1.get(), rp2.get());
  EXPECT_EQ(rp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_EQ(rp2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "load");
  EXPECT_EQ(rp1->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "brc1");
  EXPECT_EQ(rp2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "brc2");
}

// 构造链输入 Data+Load：紧凑布局；set_vectorized 同时写 vectorized_strides（用于验证垫整只
// 作用于 post-reduce 链成员、不回溯上游）。Load 移动构造不可用，由调用方构造、此处引用填充。
void CreateChainInputLoad(af::AscGraph &graph, af::ascir_op::Load &load, const char *name, ge::DataType dtype,
                          int32_t index, const std::vector<af::AxisId> &axes,
                          const std::vector<af::Expression> &chain_repeats,
                          const std::vector<af::Expression> &compact_strides, const bool set_vectorized) {
  af::ascir_op::Data data(name, graph);
  data.y.dtype = dtype;
  data.ir_attr.SetIndex(index);
  *data.y.axis = axes;
  *data.y.repeats = chain_repeats;
  *data.y.strides = compact_strides;

  load.x = data.y;
  SetTwoDimNodeAttrV2(load, dtype, af::ComputeType::kComputeLoad, axes, chain_repeats, compact_strides);
  if (set_vectorized) {
    *load.y.vectorized_strides = compact_strides;
  }
}

// 构造链尾输出：with_reduce 时尾轴 reduce（输出尾轴 stride==0 是 kNotAligned 对齐回传的源头，
// Reduce 下游节点不属于存储链，保持自身布局）后 Store；否则直接 Store 保持紧凑布局。
void CreateChainOutput(const af::ascir_op::Asin &asin_op, const std::vector<af::AxisId> &axes,
                       const std::vector<af::Expression> &chain_repeats,
                       const std::vector<af::Expression> &compact_strides, const af::Expression &outer,
                       const bool with_reduce) {
  if (with_reduce) {
    af::ascir_op::Sum sum("sum");
    sum.x = asin_op.y;
    SetTwoDimNodeAttrV2(sum, af::DT_FLOAT, af::ComputeType::kComputeReduce, axes, {outer, One}, {One, Zero});
    *sum.y.vectorized_strides = {One, Zero};

    af::ascir_op::Store store("store");
    store.x = sum.y;
    SetTwoDimNodeAttrV2(store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, {outer, One}, {One, Zero});
    *store.y.vectorized_strides = {One, Zero};
  } else {
    af::ascir_op::Store store("store");
    store.x = asin_op.y;
    SetTwoDimNodeAttrV2(store, af::DT_FLOAT, af::ComputeType::kComputeStore, axes, chain_repeats, compact_strides);
    *store.y.vectorized_strides = compact_strides;
  }
}

// IndirectLoad SIMT 后置 Reduce 链（E2E 精度缺陷形态）：kSimtOp 的 IL value 写 UB local
// target 后由后置 Reduce 消费，Store/Reduce 按 SizeAlign(inner, 32/sizeof(dtype)) 的
// 对齐行距读写；尾轴 reduce 的对齐推导（Reduce 输出尾轴 stride==0 判 kNotAligned）
// 沿链回传生成紧凑 vectorized_strides，buffer 预算按紧凑 span 计算，行数 > 1 时
// Store/Reduce 读写越过 buffer 末端。链上成员（IL 输出至 Reduce 输入）需统一垫整。
af::AscGraph BuildSimtPostReduceChainGraph(const bool with_reduce) {
  af::AscGraph graph("simt_post_reduce_chain");
  const af::Expression outer = af::Symbol(6);
  const af::Expression inner = af::Symbol(10);
  auto z0 = graph.CreateAxis("z0", outer);
  auto z1 = graph.CreateAxis("z1", inner);
  const std::vector<af::AxisId> axes = {z0.id, z1.id};
  const std::vector<af::Expression> chain_repeats = {outer, inner};
  // 紧凑布局：尾轴 stride==1、次内维 stride==尾轴 repeat（10 * 4B = 40B 非 32B 对齐）
  const std::vector<af::Expression> compact_strides = {inner, One};

  // IL 输入侧同设紧凑布局，验证垫整只作用于 post-reduce 链成员（不回溯上游）
  af::ascir_op::Load x_load("x_load");
  CreateChainInputLoad(graph, x_load, "x", af::DT_FLOAT, 0, axes, chain_repeats, compact_strides, true);
  af::ascir_op::Load idx_load("idx_load");
  CreateChainInputLoad(graph, idx_load, "idx", af::DT_INT32, 1, axes, chain_repeats, compact_strides, false);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = x_load.y;
  indirect_load.x2 = idx_load.y;
  indirect_load.ir_attr.SetAxis(0);
  indirect_load.y.dtype = af::DT_FLOAT;
  indirect_load.attr.sched.axis = axes;
  indirect_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  *indirect_load.y.axis = axes;
  *indirect_load.y.repeats = chain_repeats;
  *indirect_load.y.strides = compact_strides;
  *indirect_load.y.vectorized_axis = axes;
  *indirect_load.y.vectorized_strides = compact_strides;

  af::ascir_op::Floor floor_op("floor_op");
  floor_op.x = indirect_load.y;
  SetTwoDimNodeAttrV2(floor_op, af::DT_FLOAT, af::ComputeType::kComputeElewise, axes, chain_repeats, compact_strides);
  *floor_op.y.vectorized_strides = compact_strides;

  af::ascir_op::Asin asin_op("asin_op");
  asin_op.x = floor_op.y;
  SetTwoDimNodeAttrV2(asin_op, af::DT_FLOAT, af::ComputeType::kComputeElewise, axes, chain_repeats, compact_strides);
  *asin_op.y.vectorized_strides = compact_strides;

  CreateChainOutput(asin_op, axes, chain_repeats, compact_strides, outer, with_reduce);
  return graph;
}

TEST_F(VectorizedAlignmentUTV2, simt_post_reduce_chain_pads_static_unaligned_row_stride) {
  af::AscGraph graph = BuildSimtPostReduceChainGraph(true);
  auto il_node = graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  ASSERT_EQ(ascgen_utils::indirect_load::SetTemplateRole(il_node, ascgen_utils::indirect_load::TemplateRole::kSimtOp),
            af::SUCCESS);

  optimize::UnAlignmentStrategy strategy;
  ASSERT_EQ(strategy.ModifyVectorizedStrides(graph), af::SUCCESS);

  // inner=10 float：10 * 4B = 40B 非 32B 对齐，行距垫整到 Align(10, 32/4) = 16
  const std::vector<af::Expression> padded = {af::Symbol(16), One};
  for (const char *name : {"indirect_load", "floor_op", "asin_op"}) {
    SCOPED_TRACE(name);
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->outputs[0].attr.vectorized_strides, padded);
  }
  // Reduce 输出（尾轴 stride==0）与其下游不在存储链上，保持原布局
  EXPECT_EQ(graph.FindNode("sum")->outputs[0].attr.vectorized_strides, (std::vector<af::Expression>{One, Zero}));
  EXPECT_EQ(graph.FindNode("store")->outputs[0].attr.vectorized_strides, (std::vector<af::Expression>{One, Zero}));
  // IL 上游输入侧不受链垫整影响
  EXPECT_EQ(graph.FindNode("x_load")->outputs[0].attr.vectorized_strides,
            (std::vector<af::Expression>{af::Symbol(10), One}));
}

TEST_F(VectorizedAlignmentUTV2, simt_chain_without_reduce_keeps_compact_strides) {
  af::AscGraph graph = BuildSimtPostReduceChainGraph(false);
  auto il_node = graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);
  ASSERT_EQ(ascgen_utils::indirect_load::SetTemplateRole(il_node, ascgen_utils::indirect_load::TemplateRole::kSimtOp),
            af::SUCCESS);

  optimize::UnAlignmentStrategy strategy;
  ASSERT_EQ(strategy.ModifyVectorizedStrides(graph), af::SUCCESS);

  // 无后置 Reduce 消费者时没有行对齐写回契约，链上保持紧凑布局
  const std::vector<af::Expression> compact = {af::Symbol(10), One};
  for (const char *name : {"indirect_load", "floor_op", "asin_op", "store"}) {
    SCOPED_TRACE(name);
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->outputs[0].attr.vectorized_strides, compact);
  }
}

TEST_F(VectorizedAlignmentUTV2, non_simt_indirect_load_keeps_compact_strides) {
  af::AscGraph graph = BuildSimtPostReduceChainGraph(true);
  auto il_node = graph.FindNode("indirect_load");
  ASSERT_NE(il_node, nullptr);

  optimize::UnAlignmentStrategy strategy;
  ASSERT_EQ(strategy.ModifyVectorizedStrides(graph), af::SUCCESS);

  // 非 kSimtOp 角色（默认 kNone）的 IL 不在行对齐写回链上，保持紧凑布局
  const std::vector<af::Expression> compact = {af::Symbol(10), One};
  for (const char *name : {"indirect_load", "floor_op", "asin_op"}) {
    SCOPED_TRACE(name);
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->outputs[0].attr.vectorized_strides, compact);
  }
}
}  // namespace optimize
