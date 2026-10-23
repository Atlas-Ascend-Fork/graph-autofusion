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
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "micro_api_call_factory.h"
#include "micro_load_api_call.h"
#include "micro_where_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, LoadMicroApiCall_Load) {
  af::AscGraph graph("test_graph");

  af::Expression Two = af::Symbol(2);
  af::Expression Three = af::Symbol(3);
  af::Expression Four = af::Symbol(4);

  auto s0 = af::Symbol(16);
  auto s1 = af::Symbol(8);
  auto s2 = af::Symbol(4);
  auto s3 = af::Symbol(2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op.y.repeats = {s0, s1, s2, s3};
  *load_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};
  store_op.x = load_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2, s3};
  *store_op.y.strides = {s1 * s2 * s3 * Four, s2 * s3 * Three, s3 * Two, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id, z2.id, z3.id};
  store->outputs[0].attr.vectorized_strides = {af::Symbol(8), af::Symbol(2), One};
  store->outputs[0].attr.dtype = af::DT_INT16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 2;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

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
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(store->outputs[0], tensor_store);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign(vreg_1, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in8_out16) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_INT8;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT16;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int16_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign<int8_t, "
                                "AscendC::MicroAPI::LoadDist::DIST_UNPACK_B8>(vreg_0, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in16_out32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_FLOAT16;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_FLOAT;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "float";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign<half, "
                                "AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg_0, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in8_out32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_INT8;  // 修复：应该是 INT8
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT32;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int32_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign<int8_t, "
                                "AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(vreg_0, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_bool_max_bool) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_BOOL;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_BOOL;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "bool";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign(vreg_0, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in32_out64) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = af::DT_FLOAT;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT64;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::LoadAlign<float, "
                                "AscendC::MicroAPI::LoadDist::DIST_UNPACK_B32>(vreg_0, local_0 + offset);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in8_out64) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_INT8;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = ge::DT_INT64;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_TRUE(result.find("AscendC::MicroAPI::LoadAlign") != std::string::npos);
  EXPECT_TRUE(result.find("DIST_UNPACK4_B8") != std::string::npos);
  EXPECT_TRUE(result.find("AscendC::Reg::UnPack") != std::string::npos);
}

TEST(CodegenKernel, LoadMicroApiCall_Load_Cast_in16_out64) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(cast_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  cast_op.x = load_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_INT16;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = ge::DT_INT64;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 1;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 2;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->inputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_cast = cast->GetName() + "_" + cast->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(cast->outputs[0], tensor_cast);
  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroLoadApiCall call_0("Load");
  EXPECT_EQ(call_0.Init(load), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_TRUE(result.find("AscendC::MicroAPI::LoadAlign") != std::string::npos);
  EXPECT_TRUE(result.find("DIST_UNPACK_B16") != std::string::npos);
  EXPECT_TRUE(result.find("AscendC::Reg::UnPack") != std::string::npos);
}

// ============================================================================
// MaskReg Load 测试：从 uint8_t UnPack 到 max_dtype_size
// ============================================================================

TEST(CodegenKernel, LoadMicroApiCall_MaskReg_Load_ToUint16) {
  af::AscGraph graph("test_mask_reg_load_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Data t2_op("t2", graph);
  Load load_op("load");
  Where where_op("where");
  graph.AddNode(load_op);
  graph.AddNode(where_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  where_op.x1 = load_op.y;  // mask 输入（第一个输入）
  where_op.x2 = t1_op.y;
  where_op.x3 = t2_op.y;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_UINT8;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t1 = graph.FindNode("t1");
  t1->outputs[0].attr.dtype = ge::DT_UINT16;
  t1->outputs[0].attr.mem.tensor_id = 2;
  t1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t2 = graph.FindNode("t2");
  t2->outputs[0].attr.dtype = ge::DT_UINT16;
  t2->outputs[0].attr.mem.tensor_id = 3;
  t2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_UINT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id};
  where->outputs[0].attr.vectorized_strides = {One};
  where->outputs[0].attr.dtype = ge::DT_UINT16;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  where->outputs[0].attr.mem.tensor_id = 1;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor load_input;
  load_input.id = 0;
  codegen::ApiTensor load_output;
  load_output.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor where_output;
  where_output.id = 1;

  codegen::CallParam cp = {"p_reg", "offset"};
  // Load 输出作为 Where 的 mask 输入，Load 不是 Compare，所以不是 MaskReg
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.init_as_mask_reg_ = false;  // Load 输出不是 MaskReg（生产者不是 Compare）
  load_tensor.dtype_ = ge::DT_UINT8;

  // Where 的 T1 和 T2 输入以及输出（需要添加到 tensor_mng）
  uint64_t t1_id = 2;
  uint64_t t2_id = 3;
  auto tensor_t1 = t1->GetName() + "_" + t1->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_tensor(t1->outputs[0], tensor_t1);
  t1_tensor.id_ = t1_id;
  t1_tensor.init_as_mask_reg_ = false;
  t1_tensor.dtype_ = ge::DT_UINT16;

  auto tensor_t2 = t2->GetName() + "_" + t2->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t2_tensor(t2->outputs[0], tensor_t2);
  t2_tensor.id_ = t2_id;
  t2_tensor.init_as_mask_reg_ = false;
  t2_tensor.dtype_ = ge::DT_UINT16;

  // Where 输出 tensor（使用之前已定义的 where 节点）
  auto tensor_where = where->GetName() + "_" + where->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor where_tensor(where->outputs[0], tensor_where);
  where_tensor.id_ = where_output.id;
  where_tensor.init_as_mask_reg_ = false;
  where_tensor.dtype_ = ge::DT_UINT16;

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_tensor);
  tensor_mng.AddTensor(t2_tensor);
  tensor_mng.AddTensor(where_tensor);

  // 先调用 Load API
  codegen::MicroLoadApiCall load_call("Load");
  EXPECT_EQ(load_call.Init(load), 0);
  load_call.AddInput(load_input.id);
  load_call.AddOutput(load_output.id);

  std::string load_result;
  cp.max_dtype_size = "uint16_t";
  load_call.Generate(tensor_mng, tpipe, cp, load_result);
  // LoadAlign 使用 DIST_UNPACK_B8 模式完成 uint8 → uint16 转换，不需要额外 UnPack
  EXPECT_TRUE(load_result.find("AscendC::MicroAPI::LoadAlign") != std::string::npos);
  size_t pos = 0;
  int unpack_count = 0;
  std::string unpack_str = "UnPack";
  while ((pos = load_result.find(unpack_str, pos)) != std::string::npos) {
    unpack_count++;
    pos += unpack_str.length();
  }
  EXPECT_EQ(unpack_count, 0);

  // 再调用 Where API
  codegen::MicroWhereApiCall where_call("Select");
  EXPECT_EQ(where_call.Init(where), 0);
  where_call.AddInput(load_output.id);  // mask 输入
  where_call.AddInput(t1_id);           // T1 输入
  where_call.AddInput(t2_id);           // T2 输入
  where_call.AddOutput(where_output.id);

  std::string where_result;
  where_call.Generate(tensor_mng, tpipe, cp, where_result);
  // Where 的 mask 输入不是 MaskReg（生产者是 Load），需要 CompareScalar 转换
  EXPECT_TRUE(where_result.find("AscendC::Reg::CompareScalar") != std::string::npos);
  EXPECT_TRUE(where_result.find("AscendC::MicroAPI::Select") != std::string::npos);
}

TEST(CodegenKernel, LoadMicroApiCall_MaskReg_Load_ToUint32) {
  af::AscGraph graph("test_mask_reg_load_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Data t2_op("t2", graph);
  Load load_op("load");
  Where where_op("where");
  graph.AddNode(load_op);
  graph.AddNode(where_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  where_op.x1 = load_op.y;  // mask 输入（第一个输入）
  where_op.x2 = t1_op.y;
  where_op.x3 = t2_op.y;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_UINT8;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t1 = graph.FindNode("t1");
  t1->outputs[0].attr.dtype = ge::DT_UINT32;
  t1->outputs[0].attr.mem.tensor_id = 2;
  t1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t2 = graph.FindNode("t2");
  t2->outputs[0].attr.dtype = ge::DT_UINT32;
  t2->outputs[0].attr.mem.tensor_id = 3;
  t2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_UINT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id};
  where->outputs[0].attr.vectorized_strides = {One};
  where->outputs[0].attr.dtype = ge::DT_UINT32;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  where->outputs[0].attr.mem.tensor_id = 1;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor load_input;
  load_input.id = 0;
  codegen::ApiTensor load_output;
  load_output.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor where_output;
  where_output.id = 1;

  codegen::CallParam cp = {"p_reg", "offset"};
  // Load 输出作为 Where 的 mask 输入，Load 不是 Compare，所以不是 MaskReg
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.init_as_mask_reg_ = false;  // Load 输出不是 MaskReg（生产者不是 Compare）
  load_tensor.dtype_ = ge::DT_UINT8;

  // Where 的 T1 和 T2 输入以及输出（需要添加到 tensor_mng）
  uint64_t t1_id = 2;
  uint64_t t2_id = 3;
  auto tensor_t1 = t1->GetName() + "_" + t1->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_tensor(t1->outputs[0], tensor_t1);
  t1_tensor.id_ = t1_id;
  t1_tensor.init_as_mask_reg_ = false;
  t1_tensor.dtype_ = ge::DT_UINT32;

  auto tensor_t2 = t2->GetName() + "_" + t2->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t2_tensor(t2->outputs[0], tensor_t2);
  t2_tensor.id_ = t2_id;
  t2_tensor.init_as_mask_reg_ = false;
  t2_tensor.dtype_ = ge::DT_UINT32;

  // Where 输出 tensor
  auto tensor_where = where->GetName() + "_" + where->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor where_tensor(where->outputs[0], tensor_where);
  where_tensor.id_ = where_output.id;
  where_tensor.init_as_mask_reg_ = false;
  where_tensor.dtype_ = ge::DT_UINT32;

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_tensor);
  tensor_mng.AddTensor(t2_tensor);
  tensor_mng.AddTensor(where_tensor);

  // 先调用 Load API
  codegen::MicroLoadApiCall load_call("Load");
  EXPECT_EQ(load_call.Init(load), 0);
  load_call.AddInput(load_input.id);
  load_call.AddOutput(load_output.id);

  std::string load_result;
  cp.max_dtype_size = "uint32_t";
  load_call.Generate(tensor_mng, tpipe, cp, load_result);
  // LoadAlign 使用 DIST_UNPACK4_B8 模式完成 uint8 → uint32 转换，不需要额外 UnPack
  EXPECT_TRUE(load_result.find("AscendC::MicroAPI::LoadAlign") != std::string::npos);
  size_t pos = 0;
  int unpack_count = 0;
  std::string unpack_str = "UnPack";
  while ((pos = load_result.find(unpack_str, pos)) != std::string::npos) {
    unpack_count++;
    pos += unpack_str.length();
  }
  EXPECT_EQ(unpack_count, 0);

  // 再调用 Where API
  codegen::MicroWhereApiCall where_call("Select");
  EXPECT_EQ(where_call.Init(where), 0);
  where_call.AddInput(load_output.id);  // mask 输入
  where_call.AddInput(t1_id);           // T1 输入
  where_call.AddInput(t2_id);           // T2 输入
  where_call.AddOutput(where_output.id);

  std::string where_result;
  where_call.Generate(tensor_mng, tpipe, cp, where_result);
  // Where 的 mask 输入不是 MaskReg（生产者是 Load），需要 CompareScalar 转换
  EXPECT_TRUE(where_result.find("AscendC::Reg::CompareScalar") != std::string::npos);
  EXPECT_TRUE(where_result.find("AscendC::MicroAPI::Select") != std::string::npos);
}

TEST(CodegenKernel, LoadMicroApiCall_MaskReg_Load_ToUint64) {
  af::AscGraph graph("test_mask_reg_load_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Data t2_op("t2", graph);
  Load load_op("load");
  Where where_op("where");
  graph.AddNode(load_op);
  graph.AddNode(where_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  where_op.x1 = load_op.y;  // mask 输入（第一个输入）
  where_op.x2 = t1_op.y;
  where_op.x3 = t2_op.y;

  auto x = graph.FindNode("x");
  x->outputs[0].attr.dtype = ge::DT_UINT8;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t1 = graph.FindNode("t1");
  t1->outputs[0].attr.dtype = ge::DT_UINT64;
  t1->outputs[0].attr.mem.tensor_id = 2;
  t1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto t2 = graph.FindNode("t2");
  t2->outputs[0].attr.dtype = ge::DT_UINT64;
  t2->outputs[0].attr.mem.tensor_id = 3;
  t2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_UINT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto where = graph.FindNode("where");
  where->attr.api.compute_type = af::ComputeType::kComputeElewise;
  where->attr.api.type = af::ApiType::kAPITypeCompute;
  where->attr.api.unit = af::ComputeUnit::kUnitVector;
  where->attr.sched.loop_axis = z0.id;
  where->outputs[0].attr.vectorized_axis = {z1.id};
  where->outputs[0].attr.vectorized_strides = {One};
  where->outputs[0].attr.dtype = ge::DT_UINT64;
  where->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  where->outputs[0].attr.mem.tensor_id = 1;
  where->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  where->outputs[0].attr.que.id = 2;
  where->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(where->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor load_input;
  load_input.id = 0;
  codegen::ApiTensor load_output;
  load_output.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor where_output;
  where_output.id = 1;

  codegen::CallParam cp = {"p_reg", "offset"};
  // Load 输出作为 Where 的 mask 输入，Load 不是 Compare，所以不是 MaskReg
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.init_as_mask_reg_ = false;  // Load 输出不是 MaskReg（生产者不是 Compare）
  load_tensor.dtype_ = ge::DT_UINT8;

  // Where 的 T1 和 T2 输入以及输出（需要添加到 tensor_mng）
  uint64_t t1_id = 2;
  uint64_t t2_id = 3;
  auto tensor_t1 = t1->GetName() + "_" + t1->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_tensor(t1->outputs[0], tensor_t1);
  t1_tensor.id_ = t1_id;
  t1_tensor.init_as_mask_reg_ = false;
  t1_tensor.dtype_ = ge::DT_UINT64;

  auto tensor_t2 = t2->GetName() + "_" + t2->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t2_tensor(t2->outputs[0], tensor_t2);
  t2_tensor.id_ = t2_id;
  t2_tensor.init_as_mask_reg_ = false;
  t2_tensor.dtype_ = ge::DT_UINT64;

  // Where 输出 tensor
  auto tensor_where = where->GetName() + "_" + where->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor where_tensor(where->outputs[0], tensor_where);
  where_tensor.id_ = where_output.id;
  where_tensor.init_as_mask_reg_ = false;
  where_tensor.dtype_ = ge::DT_UINT64;

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_tensor);
  tensor_mng.AddTensor(t2_tensor);
  tensor_mng.AddTensor(where_tensor);

  // 先调用 Load API
  codegen::MicroLoadApiCall load_call("Load");
  EXPECT_EQ(load_call.Init(load), 0);
  load_call.AddInput(load_input.id);
  load_call.AddOutput(load_output.id);

  std::string load_result;
  cp.max_dtype_size = "uint64_t";
  load_call.Generate(tensor_mng, tpipe, cp, load_result);
  // LoadAlign 使用 DIST_UNPACK4_B8 完成 uint8 → uint32，再 UnPack uint32 → uint64
  EXPECT_TRUE(load_result.find("AscendC::MicroAPI::LoadAlign") != std::string::npos);
  size_t pos = 0;
  int unpack_count = 0;
  std::string unpack_str = "UnPack";
  while ((pos = load_result.find(unpack_str, pos)) != std::string::npos) {
    unpack_count++;
    pos += unpack_str.length();
  }
  EXPECT_EQ(unpack_count, 1);

  // 再调用 Where API
  codegen::MicroWhereApiCall where_call("Select");
  EXPECT_EQ(where_call.Init(where), 0);
  where_call.AddInput(load_output.id);  // mask 输入
  where_call.AddInput(t1_id);           // T1 输入
  where_call.AddInput(t2_id);           // T2 输入
  where_call.AddOutput(where_output.id);

  std::string where_result;
  where_call.Generate(tensor_mng, tpipe, cp, where_result);
  // Where 的 mask 输入不是 MaskReg（生产者是 Load），需要 CompareScalar 转换
  EXPECT_TRUE(where_result.find("AscendC::Reg::CompareScalar") != std::string::npos);
  EXPECT_TRUE(where_result.find("AscendC::MicroAPI::Select") != std::string::npos);
}
