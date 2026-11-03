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

#include "graph_utils.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "elewise/unary_api_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, UnaryApicall) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Sqrt sqrt_op("sqrt");
  graph.AddNode(sqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  sqrt_op.x = load_op.y;
  *sqrt_op.y.axis = {z0.id, z1.id};
  *sqrt_op.y.repeats = {s0, s1};
  *sqrt_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sqrt = graph.FindNode("sqrt");
  sqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  sqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  sqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  sqrt->attr.sched.loop_axis = z0.id;
  sqrt->outputs[0].attr.vectorized_axis = {z1.id};
  sqrt->outputs[0].attr.vectorized_strides = {One};
  sqrt->outputs[0].attr.dtype = ge::DT_INT16;
  sqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sqrt->outputs[0].attr.mem.tensor_id = 1;
  sqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sqrt->outputs[0].attr.que.id = 2;
  sqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(sqrt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  auto call = CreateApiCallObject(sqrt);

  delete call;
}

TEST(CodegenKernel, UnaryApicallRsqrt) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Rsqrt rsqrt_op("rsqrt");
  graph.AddNode(rsqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  rsqrt_op.x = load_op.y;
  *rsqrt_op.y.axis = {z0.id, z1.id};
  *rsqrt_op.y.repeats = {s0, s1};
  *rsqrt_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto rsqrt = graph.FindNode("rsqrt");
  rsqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rsqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  rsqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  rsqrt->attr.sched.loop_axis = z0.id;
  rsqrt->outputs[0].attr.vectorized_axis = {z1.id};
  rsqrt->outputs[0].attr.vectorized_strides = {One};
  rsqrt->outputs[0].attr.dtype = ge::DT_INT16;
  rsqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rsqrt->outputs[0].attr.mem.tensor_id = 1;
  rsqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rsqrt->outputs[0].attr.que.id = 2;
  rsqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(rsqrt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  auto call = CreateApiCallObject(rsqrt);

  codegen::UnaryApiCall call_0("Rsqrt");
  EXPECT_EQ(call_0.Init(rsqrt), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Rsqrt(local_1[0], local_0[0], local_0_actual_size);\n"});
  delete call;
}

TEST(CodegenKernel, UnaryApicallReciprocal) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Rsqrt rsqrt_op("Reciprocal");
  graph.AddNode(rsqrt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  rsqrt_op.x = load_op.y;
  *rsqrt_op.y.axis = {z0.id, z1.id};
  *rsqrt_op.y.repeats = {s0, s1};
  *rsqrt_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto rsqrt = graph.FindNode("Reciprocal");
  rsqrt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rsqrt->attr.api.type = af::ApiType::kAPITypeCompute;
  rsqrt->attr.api.unit = af::ComputeUnit::kUnitVector;
  rsqrt->attr.sched.loop_axis = z0.id;
  rsqrt->outputs[0].attr.vectorized_axis = {z1.id};
  rsqrt->outputs[0].attr.vectorized_strides = {One};
  rsqrt->outputs[0].attr.dtype = ge::DT_INT16;
  rsqrt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rsqrt->outputs[0].attr.mem.tensor_id = 1;
  rsqrt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rsqrt->outputs[0].attr.que.id = 2;
  rsqrt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(rsqrt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  auto call = CreateApiCallObject(rsqrt);

  codegen::UnaryApiCall call_0("Reciprocal");
  EXPECT_EQ(call_0.Init(rsqrt), 0);
  call_0.inputs.push_back(&x1);

  std::string result;
  call_0.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Reciprocal(local_1[0], local_0[0], local_0_actual_size);\n"});
  delete call;
}
