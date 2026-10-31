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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "utils/api_call_factory.h"
#include "broadcast/broadcast_api_call.h"
#include "reg_broadcast_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

static void BuildBroadcastRegApiCallPrePosNotFoundGraph(af::AscGraph &graph, af::Expression s0, af::Expression s1,
                                                        af::Expression s2, af::Axis z0, af::Axis z1, af::Axis z2) {
  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);
  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {One, One, s2};
  *load_op.y.strides = {Zero, Zero, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {Zero, Zero, One};
}

static void ConfigureBroadcastRegApiCallPrePosNotFoundGraph(af::AscGraph &graph, af::AxisId z0, af::AxisId z1,
                                                            af::AxisId z2) {
  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0;
  load->outputs[0].attr.vectorized_axis = {z0, z1, z2};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero, One};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0;
  broadcast->outputs[0].attr.vectorized_axis = {z0, z1, z2};
  broadcast->outputs[0].attr.vectorized_strides = {Zero, Zero, One};
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

TEST(CodegenKernel, BroadcastRegApiCallAllCommonAxis) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = af::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  const auto api_param = CodegenApiParam::GetNodeApiParam(broadcast);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "Broadcast");
  EXPECT_EQ(
      result,
      std::string{
          "const uint32_t dst_shape_0_brc_to_1[2] = {static_cast<uint32_t>(t->s0), static_cast<uint32_t>(t->s1)};\n"
          "const uint32_t src_shape_0_brc_to_1[2] = {static_cast<uint32_t>(t->s0), static_cast<uint32_t>(t->s1)};\n"
          "BroadcastExtend<float,2>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n"});
}

TEST(CodegenKernel, BroadcastRegApiCallUBScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {One, One};
  *load_op.y.strides = {Zero, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  const auto api_param = CodegenApiParam::GetNodeApiParam(broadcast);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "Broadcast.Scalar");
  EXPECT_EQ(result, std::string{"Duplicate(local_1[0], local_0_ub_scalar, local_1_actual_size);\n"});
}

TEST(CodegenKernel, BroadcastRegApiCallVectorizedSizeInvalid) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id};
  *broadcast_op.y.repeats = {s0, s1};
  *broadcast_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, s1, Zero};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1, One};
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, "");
}

TEST(CodegenKernel, BroadcastRegApiCall_BAB) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {One, s1, One};
  *load_op.y.strides = {Zero, One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Zero, One, Zero};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * s2, s2, One};
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            "const uint32_t dst_shape_0_brc_to_1[3] = {static_cast<uint32_t>(t->s0), static_cast<uint32_t>(t->s1), "
            "static_cast<uint32_t>(t->s2)};\n"
            "const uint32_t src_shape_0_brc_to_1[3] = {static_cast<uint32_t>(1), static_cast<uint32_t>(t->s1), "
            "static_cast<uint32_t>(1)};\n"
            "BroadcastExtend<int64_t,3>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n");
}

TEST(CodegenKernel, BroadcastRegApiCall_BABAB) {
  af::AscGraph graph("test_graph");

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

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load_op.y.repeats = {One, s1, One, s3, One};
  *load_op.y.strides = {Zero, (s1 * s3), Zero, One, Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *broadcast_op.y.repeats = {s0, s1, s2, s3, s4};
  *broadcast_op.y.strides = {(s1 * s2 * s3 * s4), (s2 * s3 * s4), (s3 * s4), s4, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  load->outputs[0].attr.vectorized_strides = {Zero, s1, Zero, s3, Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  broadcast->outputs[0].attr.dtype = af::DT_FLOAT;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddAxis(z4);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddSizeVar(af::SizeVar(s4));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            "const uint32_t dst_shape_0_brc_to_1[5] = {static_cast<uint32_t>(t->s0), static_cast<uint32_t>(t->s1), "
            "static_cast<uint32_t>(t->s2), static_cast<uint32_t>(t->s3), static_cast<uint32_t>(t->s4)};\n"
            "const uint32_t src_shape_0_brc_to_1[5] = {static_cast<uint32_t>(1), static_cast<uint32_t>(t->s1), "
            "static_cast<uint32_t>(1), static_cast<uint32_t>(t->s3), static_cast<uint32_t>(1)};\n"
            "BroadcastExtend<float,5>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n");
}

TEST(CodegenKernel, ConstantBrcStore) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  Scalar constant_op("constant", graph);
  constant_op.ir_attr.SetValue("1.0");
  Broadcast broadcast("broadcast");
  broadcast.x = constant_op.y;
  broadcast.attr.sched.axis = {z0.id, z1.id};
  *broadcast.y.axis = {z0.id, z1.id};
  *broadcast.y.repeats = {s0, s1};
  *broadcast.y.strides = {s1, One};

  Store store("store");
  store.x = broadcast.y;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  auto broadcast_node = graph.FindNode("broadcast");
  broadcast_node->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast_node->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast_node->attr.sched.loop_axis = z0.id;
  broadcast_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  broadcast_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  broadcast_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast_node->outputs[0].attr.mem.tensor_id = 1;
  broadcast_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast_node->outputs[0].attr.que.id = 1;
  broadcast_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store_node = graph.FindNode("store");
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE3;
  store_node->attr.sched.loop_axis = z0.id;
  store_node->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  store_node->outputs[0].attr.dtype = af::DT_FLOAT16;
  store_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  store_node->outputs[0].attr.mem.tensor_id = 2;
  store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store_node->outputs[0].attr.que.id = 1;
  store_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 0;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(broadcast_node->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;
  x2.id = broadcast_node->outputs[0].attr.mem.tensor_id;

  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast_node), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Duplicate(local_1[0], scalar_0, local_1_actual_size);\n"});
}

TEST(CodegenKernel, BroadcastRegApiCall_BrcAlign_ABAWithTranspose) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(2);
  auto s2 = graph.CreateSizeVar(8);
  auto z0 = graph.CreateAxis("z0", af::Axis::kAxisTypeTileInner, s0, {}, -1);
  auto z1 = graph.CreateAxis("z1", af::Axis::kAxisTypeTileInner, s1, {}, -1);
  auto z2 = graph.CreateAxis("z2", af::Axis::kAxisTypeTileInner, s2, {}, -1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  af::ascir_op::Transpose transpose_op("transpose");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);
  graph.AddNode(transpose_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, One, s2};               // (2, 1, 8)
  *load_op.y.strides = {af::Symbol(8), Zero, One};  // (8, 0, 1)
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {s1 * af::Symbol(8), af::Symbol(8), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), Zero, One};  // (8, 0, 1)
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * af::Symbol(8), af::Symbol(8), One};  // (2 * 8, 8, 1)
  ;
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  auto params = ascir_param::GetAscirNodeParams(broadcast);
  ASSERT_NE(params, nullptr);
  const auto *broadcast_params = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(*params);
  ASSERT_NE(broadcast_params, nullptr);
  ASSERT_EQ(broadcast_params->dst_shape.size(), 3UL);
  ASSERT_EQ(broadcast_params->src_shape.size(), 3UL);
  EXPECT_EQ(broadcast_params->dst_shape.back().role, ascir_param::ParamExprRole::kActualSize);
  EXPECT_EQ(broadcast_params->src_shape.back().role, ascir_param::ParamExprRole::kActualSize);
  EXPECT_EQ(result,
            "const uint32_t dst_shape_0_brc_to_1[3] = {static_cast<uint32_t>(2), static_cast<uint32_t>(2), "
            "static_cast<uint32_t>(8)};\n"
            "const uint32_t src_shape_0_brc_to_1[3] = {static_cast<uint32_t>(2), static_cast<uint32_t>(1), "
            "static_cast<uint32_t>(8)};\n"
            "BroadcastExtend<int64_t,3>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n");
}

TEST(CodegenKernel, BroadcastRegApiCall_BrcAlign_BBA) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(2);
  auto s2 = graph.CreateSizeVar(7);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", af::Symbol(8));

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {One, One, s2};
  *load_op.y.strides = {Zero, Zero, One};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id};
  *broadcast_op.y.repeats = {s0, s1, s2};
  *broadcast_op.y.strides = {s1 * af::Symbol(8), af::Symbol(8), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {Zero, Zero, One};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * af::Symbol(8), af::Symbol(8), One};
  broadcast->outputs[0].attr.dtype = af::DT_INT64;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);

  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            "const uint32_t dst_shape_0_brc_to_1[3] = {static_cast<uint32_t>(2), static_cast<uint32_t>(2), "
            "static_cast<uint32_t>(8)};\n"
            "const uint32_t src_shape_0_brc_to_1[3] = {static_cast<uint32_t>(1), static_cast<uint32_t>(1), "
            "static_cast<uint32_t>(8)};\n"
            "BroadcastExtend<int64_t,3>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n");
}

TEST(CodegenKernel, BroadcastRegApiCall_PrePosNotFound) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar(2);
  auto s1 = graph.CreateSizeVar(2);
  auto s2 = graph.CreateSizeVar(7);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildBroadcastRegApiCallPrePosNotFoundGraph(graph, s0, s1, s2, z0, z1, z2);
  ConfigureBroadcastRegApiCallPrePosNotFoundGraph(graph, z0.id, z1.id, z2.id);
  auto load = graph.FindNode("load");
  auto broadcast = graph.FindNode("broadcast");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(result,
            "const uint32_t dst_shape_0_brc_to_1[3] = {static_cast<uint32_t>(2), static_cast<uint32_t>(2), "
            "static_cast<uint32_t>(7)};\n"
            "const uint32_t src_shape_0_brc_to_1[3] = {static_cast<uint32_t>(1), static_cast<uint32_t>(1), "
            "static_cast<uint32_t>(7)};\n"
            "BroadcastExtend<int64_t,3>(local_1[0], local_0[0], dst_shape_0_brc_to_1, src_shape_0_brc_to_1);\n");
}

// 尾轴广播且广播源内层被对齐成“每元素独占一个 data block”（源内层非零 stride = 32B/dtype）
// 时，需以 srcInnerPad 实例化 BroadcastExtend，使其按 ONE_BLK_SIZE / sizeof(T) 反推源 stride。
TEST(CodegenKernel, BroadcastRegApiCall_TailBrcSrcInnerPad) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(3);
  auto s2 = graph.CreateSizeVar(48);
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, One};
  *load_op.y.strides = {s1 * s2 * af::Symbol(32), s2 * af::Symbol(32), af::Symbol(32), Zero};
  broadcast_op.x = load_op.y;
  *broadcast_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *broadcast_op.y.repeats = {s0, s1, s2, s3};
  *broadcast_op.y.strides = {s1 * s2 * af::Symbol(64), s2 * af::Symbol(64), af::Symbol(64), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {s1 * s2 * af::Symbol(32), s2 * af::Symbol(32), af::Symbol(32), Zero};
  load->outputs[0].attr.dtype = af::DT_INT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto broadcast = graph.FindNode("broadcast");
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  broadcast->attr.sched.loop_axis = z0.id;
  broadcast->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  broadcast->outputs[0].attr.vectorized_strides = {s1 * s2 * af::Symbol(64), s2 * af::Symbol(64), af::Symbol(64), One};
  broadcast->outputs[0].attr.dtype = af::DT_INT8;
  broadcast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  broadcast->outputs[0].attr.que.id = 2;
  broadcast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(broadcast->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::BroadcastRegApiCall call("BroadcastExtend");
  EXPECT_EQ(call.Init(broadcast), 0);
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("BroadcastExtend<int8_t,4,true>("), std::string::npos);
}
