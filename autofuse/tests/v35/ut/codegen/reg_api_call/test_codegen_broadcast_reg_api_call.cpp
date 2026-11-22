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

TEST(CodegenKernel, CvNBroadcastUsesItsAllocatedOutputAndTailSemantic) {
  af::AscGraph graph("cv_explicit_n_broadcast");
  const auto m = graph.CreateAxis("m", af::Symbol(128));
  const auto n = graph.CreateAxis("n", af::Symbol(256));
  Data data("data", graph);
  Load load_op("load");
  Broadcast broadcast_op("broadcast");
  graph.AddNode(load_op);
  graph.AddNode(broadcast_op);
  load_op.x = data.y;
  broadcast_op.x = load_op.y;
  const auto load = graph.FindNode("load");
  const auto broadcast = graph.FindNode("broadcast");
  for (const auto &node : {load, broadcast}) {
    node->attr.sched.axis = {m.id, n.id};
    node->attr.api.type = af::ApiType::kAPITypeCompute;
    auto &attr = node->outputs[0].attr;
    attr.axis = {m.id, n.id};
    attr.vectorized_axis = {m.id, n.id};
    attr.dtype = af::DT_FLOAT16;
    attr.mem.position = af::Position::kPositionVecCalc;
    attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    attr.opt.merge_scope = af::kIdNone;
  }
  load->outputs[0].attr.repeats = {af::Symbol(128), One};
  load->outputs[0].attr.strides = {One, Zero};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(16), Zero};
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.buf.id = 0;
  broadcast->outputs[0].attr.repeats = {af::Symbol(128), af::Symbol(256)};
  broadcast->outputs[0].attr.strides = {af::Symbol(256), One};
  broadcast->outputs[0].attr.vectorized_strides = {af::Symbol(256), One};
  broadcast->outputs[0].attr.mem.tensor_id = 1;
  broadcast->outputs[0].attr.buf.id = 1;
  broadcast->attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast->attr.api.unit = af::ComputeUnit::kUnitVector;
  Tiler tiler;
  ASSERT_EQ(tiler.AddAxis(m), af::SUCCESS);
  ASSERT_EQ(tiler.AddAxis(n), af::SUCCESS);
  TPipe tpipe("tpipe", tiler);
  ASSERT_EQ(tpipe.AddTensor(load->outputs[0]), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(broadcast->outputs[0]), af::SUCCESS);
  tpipe.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  tpipe.cv_output_broadcast = true;
  tpipe.cv_n_axis = n.id;
  ApiTensor input;
  input.id = 0;
  BroadcastRegApiCall call("BroadcastExtend");
  ASSERT_EQ(call.Init(broadcast), af::SUCCESS);
  call.inputs.push_back(&input);
  std::string result;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("BroadcastExtend<half,2>(local_1, local_0, cv_dst_shape, cv_src_shape)"), std::string::npos);
  EXPECT_NE(result.find("cv_src_shape[2] = {1U, 1U}"), std::string::npos);
  const auto api = CodegenApiParam::GetNodeApiParam(broadcast);
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->semantic_feature, "Broadcast.Tail");
  const auto params = ascir_param::GetAscirNodeParams(broadcast);
  ASSERT_NE(params, nullptr);
  const auto specific = std::get_if<ascir_param::BroadcastNodeParams>(&params->specific_params);
  ASSERT_NE(specific, nullptr);
  EXPECT_TRUE(specific->valid);
  EXPECT_FALSE(specific->is_scalar);
  EXPECT_EQ(specific->const_rank, 2);
  ASSERT_EQ(specific->dst_shape.size(), 2UL);
  EXPECT_EQ(specific->dst_shape.front().expr, af::Symbol(1));

  ASSERT_EQ(
      ascir::SetCvBroadcastShape(broadcast, {af::Symbol(128), af::Symbol(256)}, n.id, {af::Symbol(128), One}, m.id),
      af::SUCCESS);
  tpipe.cv_n_batch = true;
  tpipe.cv_m_axis = m.id;
  tpipe.cube_output_tensor_id = 0;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("local_1[cv_row * cv_dst_shape[1]], local_0[cv_row * curAlignN]"), std::string::npos);
  EXPECT_NE(result.find("local_1_actual_size = curAivM * KernelUtils::BlkAlign<half>(curAivN)"), std::string::npos);
  tpipe.cube_output_tensor_id = af::kIdNone;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("local_0[cv_row * KernelUtils::BlkAlign<half>(1)]"), std::string::npos);
  EXPECT_EQ(specific->src_shape.size(), 2UL);
  EXPECT_EQ(specific->dst_shape.size(), 2UL);

  // The scheduler can keep M as an outer loop while vectorizing only N.
  // The device call stays rank two, but ATT binds only the N expression.
  auto *source = tpipe.GetTensor(0);
  auto *target = tpipe.GetTensor(1);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  source->vectorized_axis = {n.id};
  source->vectorized_axis_pos = {1};
  source->vectorized_strides = {Zero};
  target->vectorized_axis = {n.id};
  target->vectorized_axis_pos = {1};
  target->vectorized_strides = {One};
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("BroadcastExtend<half,2>"), std::string::npos);
  EXPECT_EQ(specific->const_rank, 2);
  ASSERT_EQ(specific->src_shape.size(), 1UL);
  ASSERT_EQ(specific->dst_shape.size(), 1UL);
  EXPECT_EQ(specific->src_shape.front().expr, One);
  EXPECT_EQ(specific->dst_shape.front().expr, af::Symbol(256));
  source->vectorized_axis = {m.id, n.id};
  source->vectorized_axis_pos = {0, 1};
  source->vectorized_strides = {af::Symbol(16), Zero};
  target->vectorized_axis = {m.id, n.id};
  target->vectorized_axis_pos = {0, 1};
  target->vectorized_strides = {af::Symbol(256), One};
  tpipe.cv_n_batch = false;

  tpipe.cv_m_broadcast = true;
  tpipe.cv_m_axis = m.id;
  source->axis_size = {One, af::Symbol(256)};
  source->vectorized_strides = {Zero, One};
  source->axis_strides = {Zero, One};
  load->outputs[0].attr.repeats = source->axis_size;
  load->outputs[0].attr.strides = source->axis_strides;
  load->outputs[0].attr.vectorized_strides = source->vectorized_strides;
  EXPECT_EQ(tpipe.CvRows(*source), "1");
  EXPECT_EQ(tpipe.CvColumns(*source), "curAivN");
  EXPECT_EQ(tpipe.CvRows(*tpipe.GetTensor(1)), "curAivM");
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("cv_src_shape[2] = {1U, static_cast<uint32_t>"), std::string::npos);
  EXPECT_NE(result.find("cv_dst_shape[2] = {static_cast<uint32_t>(curAivM)"), std::string::npos);
  EXPECT_EQ(CodegenApiParam::GetNodeApiParam(broadcast)->semantic_feature, "Broadcast.NonTail");
  const auto m_params = ascir_param::GetAscirNodeParams(broadcast);
  ASSERT_NE(m_params, nullptr);
  const auto m_specific = std::get_if<ascir_param::BroadcastNodeParams>(&m_params->specific_params);
  ASSERT_NE(m_specific, nullptr);
  EXPECT_EQ(m_specific->src_shape.front().expr, af::Symbol(1));
  EXPECT_EQ(m_specific->dst_shape.front().expr, af::Symbol(128));

  // MN becomes a scalar Duplicate, but still writes a callback-local full-M
  // tile. A symbolic graph stride must not leak a t pointer into that scope.
  const auto tile_n = af::Symbol("cv_tile_n_size");
  tiler.AddSizeVar(af::SizeVar(tile_n));
  target->vectorized_strides = {tile_n, One};
  source->axis_size = {One, One};
  source->vectorized_strides = {Zero, Zero};
  source->axis_strides = {Zero, Zero};
  source->is_ub_scalar = false;
  source->position = af::Position::kPositionVecCalc;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{m.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("local_1_actual_size = curAivM * KernelUtils::BlkAlign<half>(curAivN);"), std::string::npos);
  EXPECT_NE(result.find("Duplicate(local_1[0], local_0.GetValue(0), local_1_actual_size)"), std::string::npos);
  EXPECT_NE(result.find("WaitFlag<HardEvent::V_S>"), std::string::npos);
  EXPECT_EQ(result.find("t->"), std::string::npos);
  EXPECT_EQ(CodegenApiParam::GetNodeApiParam(broadcast)->semantic_feature, "Broadcast.Scalar");

  source->dtype = af::DT_FLOAT;
  target->dtype = af::DT_FLOAT;
  source->position = af::Position::kPositionVecIn;
  tpipe.cube_output_tensor_id = source->id;
  tpipe.cv_broadcast_n = true;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{m.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("BroadcastExtend<float,1>(local_1, local_0, cv_dst_shape, cv_src_shape)"), std::string::npos);
  EXPECT_EQ(result.find("GetValue"), std::string::npos);
  ASSERT_TRUE(specific->is_scalar);
  EXPECT_EQ(specific->const_rank, 1);
  tpipe.is_inductor = true;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{m.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("local_0.GetValue(0)"), std::string::npos);
  EXPECT_EQ(specific->const_rank, -1);

  tpipe.cv_split_nm_columns = true;
  for (const auto dtype : {af::DT_FLOAT16, af::DT_BF16, af::DT_FLOAT}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    source->dtype = dtype;
    target->dtype = dtype;
    std::string dtype_name;
    ASSERT_EQ(codegen::Tensor::DtypeName(dtype, dtype_name), af::SUCCESS);
    ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{m.id}, result), af::SUCCESS);
    EXPECT_NE(result.find("BroadcastExtend<" + dtype_name + ",1>"), std::string::npos);
    EXPECT_EQ(result.find("GetValue"), std::string::npos);
    EXPECT_EQ(result.find("t->"), std::string::npos);
    EXPECT_TRUE(specific->is_scalar);
    EXPECT_EQ(specific->const_rank, 1);
  }
  tpipe.cv_split_nm_columns = false;
  tpipe.is_inductor = false;

  tpipe.cv_output_broadcast = false;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{m.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("t->cv_tile_n_size"), std::string::npos);

  tpipe.cv_fusion_type = ascir::CubeTemplateType::kCommon;
  source->axis_size = {One, af::Symbol(256)};
  source->axis_strides = {Zero, One};
  source->vectorized_strides = {Zero, One};
  target->vectorized_strides = {af::Symbol(256), One};
  tpipe.common_m_row_broadcast = {target->id, 256};
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_NE(result.find("if constexpr (256 % (AscendC::GetVecLen() / sizeof(float)) == 0)"), std::string::npos);
  const auto row_view = result.find("local_1 = local_0;");
  const auto fallback = result.find("} else {");
  const auto materialize = result.find("BroadcastExtend<float,2>");
  ASSERT_NE(row_view, std::string::npos);
  ASSERT_NE(fallback, std::string::npos);
  ASSERT_NE(materialize, std::string::npos);
  EXPECT_LT(row_view, fallback);
  EXPECT_LT(fallback, materialize);

  tpipe.common_m_row_broadcast.output_id = source->id;
  ASSERT_EQ(call.Generate(tpipe, std::vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(result.find("local_1 = local_0;"), std::string::npos);
  EXPECT_NE(result.find("BroadcastExtend<float,2>"), std::string::npos);
}

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
