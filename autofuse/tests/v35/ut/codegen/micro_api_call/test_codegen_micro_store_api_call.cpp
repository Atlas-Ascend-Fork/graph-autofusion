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
#include "micro_store_api_call.h"
#include "micro_compare_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, StoreMicroApiCall_Store) {
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
  *store_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
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
  tpipe.AddTensor(store->outputs[0]);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::StoreAlign(local_1 + offset, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in16_out8) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_INT16;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_INT8;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int16_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(local_3 + offset, "
                "vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in8_out8_max_bool) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

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

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_INT8;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_INT8;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "bool";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::StoreAlign(local_3 + offset, vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in32_out16) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_INT32;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT16;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_INT16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int32_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::StoreAlign<int16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(local_3 + offset, "
                "vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in64_out32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_INT64;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT32;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_INT32;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::StoreAlign<int32_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B64>(local_3 + offset, "
                "vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in32_out8) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = af::DT_INT32;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = af::DT_INT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = af::DT_INT8;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int32_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(local_3 + offset, "
                "vreg_0, p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in64_out16) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = ge::DT_INT64;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = ge::DT_INT16;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_INT16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::MaskReg p_reg_vreg_0_mask_temp;\n"
                "AscendC::Reg::RegTensor<int16_t> vreg_0_pack_tmp;\n"
                "AscendC::Reg::Pack<uint32_t, uint64_t>((AscendC::Reg::RegTensor<uint32_t>&)vreg_0_pack_tmp, "
                "(AscendC::Reg::RegTensor<uint64_t>&)vreg_0);\n"
                "AscendC::Reg::MaskPack(p_reg_vreg_0_mask_temp, p_reg);\n"
                "AscendC::MicroAPI::StoreAlign<int16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(local_3 + offset, "
                "vreg_0_pack_tmp, p_reg_vreg_0_mask_temp);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Cast_in64_out8) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Cast cast_op("cast");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(cast_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  cast_op.x = abs_op.y;
  *cast_op.y.axis = {z0.id, z1.id};
  *cast_op.y.repeats = {s0, s1};
  *cast_op.y.strides = {s1, One};

  store_op.x = cast_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT64;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeLoad;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = ge::DT_INT64;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto cast = graph.FindNode("cast");
  cast->attr.api.compute_type = af::ComputeType::kComputeElewise;
  cast->attr.api.type = af::ApiType::kAPITypeCompute;
  cast->attr.api.unit = af::ComputeUnit::kUnitVector;
  cast->attr.sched.loop_axis = z0.id;
  cast->outputs[0].attr.vectorized_axis = {z1.id};
  cast->outputs[0].attr.vectorized_strides = {One};
  cast->outputs[0].attr.dtype = ge::DT_INT8;
  cast->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  cast->outputs[0].attr.mem.tensor_id = 2;
  cast->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  cast->outputs[0].attr.que.id = 3;
  cast->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeElewise;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitVector;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_INT8;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(cast->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));
  std::vector<ge::AxisId> current_axis;
  current_axis.push_back(z0.id);

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
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result,
            std::string{
                "AscendC::MicroAPI::MaskReg p_reg_vreg_0_mask_temp;\n"
                "AscendC::Reg::RegTensor<int8_t> vreg_0_pack_tmp;\n"
                "AscendC::Reg::Pack<uint32_t, uint64_t>((AscendC::Reg::RegTensor<uint32_t>&)vreg_0_pack_tmp, "
                "(AscendC::Reg::RegTensor<uint64_t>&)vreg_0);\n"
                "AscendC::Reg::MaskPack(p_reg_vreg_0_mask_temp, p_reg);\n"
                "AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(local_3 + offset, "
                "vreg_0_pack_tmp, p_reg_vreg_0_mask_temp);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Out_Int8) {
  af::AscGraph graph("test_graph");

  auto z0 = graph.CreateAxis("z0", One);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {One};
  *load_op.y.strides = {Zero};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id};
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {One};
  *abs_op.y.strides = {Zero};

  store_op.x = abs_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {One};
  *store_op.y.strides = {Zero};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = af::DT_INT8;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {Zero};
  abs->outputs[0].attr.dtype = af::DT_INT8;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {Zero};
  store->outputs[0].attr.dtype = af::DT_INT8;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->inputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  std::vector<af::AxisId> current_axis;

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_abs = abs->GetName() + "_" + abs->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(abs->outputs[0], tensor_abs);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::StoreAlign<int8_t, "
                                "AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(local_2 + offset, vreg_1,"
                                " p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Out_Half) {
  af::AscGraph graph("test_graph");

  auto z0 = graph.CreateAxis("z0", One);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {One};
  *load_op.y.strides = {Zero};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id};
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {One};
  *abs_op.y.strides = {Zero};

  store_op.x = abs_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {One};
  *store_op.y.strides = {Zero};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {Zero};
  abs->outputs[0].attr.dtype = af::DT_FLOAT16;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {Zero};
  store->outputs[0].attr.dtype = af::DT_FLOAT16;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->inputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  std::vector<af::AxisId> current_axis;

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_abs = abs->GetName() + "_" + abs->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(abs->outputs[0], tensor_abs);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::StoreAlign<half, "
                                "AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(local_2 + offset, vreg_1,"
                                " p_reg);\n"});
}

TEST(CodegenKernel, LoadMicroApiCall_Store_Out_Float) {
  af::AscGraph graph("test_graph");

  auto z0 = graph.CreateAxis("z0", One);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {One};
  *load_op.y.strides = {Zero};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id};
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {One};
  *abs_op.y.strides = {Zero};

  store_op.x = abs_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id};
  *store_op.y.repeats = {One};
  *store_op.y.strides = {Zero};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {Zero};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z0.id};
  abs->outputs[0].attr.vectorized_strides = {Zero};
  abs->outputs[0].attr.dtype = af::DT_FLOAT;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {Zero};
  store->outputs[0].attr.dtype = af::DT_FLOAT;
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->inputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  std::vector<af::AxisId> current_axis;

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(load->outputs[0], tensor_load);
  auto tensor_abs = abs->GetName() + "_" + abs->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(abs->outputs[0], tensor_abs);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);
  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_EQ(result, std::string{"AscendC::MicroAPI::StoreAlign<float, "
                                "AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(local_2 + offset, vreg_1,"
                                " p_reg);\n"});
}

// ============================================================================
// MaskReg Store 测试：从 max_dtype_size Pack 到 uint8_t
// ============================================================================

TEST(CodegenKernel, StoreMicroApiCall_MaskReg_Store_FromUint16) {
  af::AscGraph graph("test_mask_reg_store_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Load load_op("load");
  Load t1_load_op("t1_load");
  Ge ge_op("ge");           // Compare 算子
  Where where_op("where");  // Where 算子
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(t1_load_op);
  graph.AddNode(ge_op);
  graph.AddNode(where_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  t1_load_op.x = t1_op.y;
  t1_load_op.attr.sched.axis = {z0.id, z1.id};
  *t1_load_op.y.axis = {z0.id, z1.id};
  *t1_load_op.y.repeats = {s0, s1};
  *t1_load_op.y.strides = {s1, One};

  ge_op.x1 = load_op.y;
  ge_op.x2 = t1_load_op.y;
  ge_op.attr.sched.axis = {z0.id, z1.id};
  *ge_op.y.axis = {z0.id, z1.id};
  *ge_op.y.repeats = {s0, s1};
  *ge_op.y.strides = {s1, One};

  // Compare 输出同时连接到 Where mask 输入和 Store 输入
  // 因为有 Store 消费者，Compare 输出不会是 MaskReg，会触发 Duplicate
  where_op.x1 = ge_op.y;  // mask 输入（Compare 输出）
  where_op.x2 = load_op.y;
  where_op.x3 = t1_load_op.y;
  where_op.attr.sched.axis = {z0.id, z1.id};
  *where_op.y.axis = {z0.id, z1.id};
  *where_op.y.repeats = {s0, s1};
  *where_op.y.strides = {s1, One};

  store_op.x = ge_op.y;  // Compare 输出连接到 Store 输入
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

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

  auto t1_load = graph.FindNode("t1_load");
  t1_load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  t1_load->attr.api.type = af::ApiType::kAPITypeCompute;
  t1_load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  t1_load->attr.sched.loop_axis = z0.id;
  t1_load->outputs[0].attr.vectorized_axis = {z1.id};
  t1_load->outputs[0].attr.vectorized_strides = {One};
  t1_load->outputs[0].attr.dtype = ge::DT_UINT8;
  t1_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  t1_load->outputs[0].attr.mem.tensor_id = 1;
  t1_load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  t1_load->outputs[0].attr.que.id = 2;
  t1_load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id};
  ge->outputs[0].attr.vectorized_strides = {One};
  ge->outputs[0].attr.dtype = ge::DT_UINT16;  // Compare 输出 dtype
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 3;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_UINT8;  // 输出到 uint8_t
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(t1_load->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor compare_input1;
  compare_input1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_input2;
  compare_input2.id = t1_load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_output;
  compare_output.id = ge->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor store_output;
  store_output.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  // Compare 的两个输入 tensor（需要添加到 tensor_mng）
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.id_ = load->outputs[0].attr.mem.tensor_id;
  load_tensor.init_as_mask_reg_ = false;
  load_tensor.dtype_ = ge::DT_UINT8;

  auto tensor_t1_load = t1_load->GetName() + "_" + t1_load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_load_tensor(t1_load->outputs[0], tensor_t1_load);
  t1_load_tensor.id_ = t1_load->outputs[0].attr.mem.tensor_id;
  t1_load_tensor.init_as_mask_reg_ = false;
  t1_load_tensor.dtype_ = ge::DT_UINT8;

  auto tensor_ge = ge->GetName() + "_" + ge->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor ge_tensor(ge->outputs[0], tensor_ge);
  ge_tensor.init_as_mask_reg_ = false;
  ge_tensor.dtype_ = ge::DT_UINT16;

  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor store_tensor(store->outputs[0], tensor_store);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_load_tensor);
  tensor_mng.AddTensor(ge_tensor);
  tensor_mng.AddTensor(store_tensor);

  // 先调用 Compare API（会产生 Duplicate）
  codegen::MicroCompareApiCall compare_call("GE");
  EXPECT_EQ(compare_call.Init(ge), 0);
  compare_call.AddInput(compare_input1.id);
  compare_call.AddInput(compare_input2.id);
  compare_call.AddOutput(compare_output.id);

  std::string compare_result;
  cp.max_dtype_size = "uint16_t";
  compare_call.Generate(tensor_mng, tpipe, cp, compare_result);
  // Compare 输出不是 MaskReg（有 Store 消费者），会产生 Duplicate
  EXPECT_TRUE(compare_result.find("AscendC::Reg::Duplicate") != std::string::npos);

  // 再调用 Store API
  codegen::MicroStoreApiCall store_call("Store");
  EXPECT_EQ(store_call.Init(store), 0);
  store_call.AddInput(compare_output.id);
  store_call.AddOutput(store_output.id);

  std::string store_result;
  store_call.Generate(tensor_mng, tpipe, cp, store_result);
  // StoreAlign 使用 DIST_PACK_B16 完成 uint16 → uint8，不需要额外 Pack
  EXPECT_TRUE(store_result.find("AscendC::MicroAPI::StoreAlign") != std::string::npos);
}

TEST(CodegenKernel, StoreMicroApiCall_MaskReg_Store_FromUint32) {
  af::AscGraph graph("test_mask_reg_store_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Load load_op("load");
  Load t1_load_op("t1_load");
  Ge ge_op("ge");           // Compare 算子
  Where where_op("where");  // Where 算子
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(t1_load_op);
  graph.AddNode(ge_op);
  graph.AddNode(where_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  t1_load_op.x = t1_op.y;
  t1_load_op.attr.sched.axis = {z0.id, z1.id};
  *t1_load_op.y.axis = {z0.id, z1.id};
  *t1_load_op.y.repeats = {s0, s1};
  *t1_load_op.y.strides = {s1, One};

  ge_op.x1 = load_op.y;
  ge_op.x2 = t1_load_op.y;
  ge_op.attr.sched.axis = {z0.id, z1.id};
  *ge_op.y.axis = {z0.id, z1.id};
  *ge_op.y.repeats = {s0, s1};
  *ge_op.y.strides = {s1, One};

  // Compare 输出同时连接到 Where mask 输入和 Store 输入
  where_op.x1 = ge_op.y;  // mask 输入（Compare 输出）
  where_op.x2 = load_op.y;
  where_op.x3 = t1_load_op.y;
  where_op.attr.sched.axis = {z0.id, z1.id};
  *where_op.y.axis = {z0.id, z1.id};
  *where_op.y.repeats = {s0, s1};
  *where_op.y.strides = {s1, One};

  store_op.x = ge_op.y;  // Compare 输出连接到 Store 输入
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

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

  auto t1_load = graph.FindNode("t1_load");
  t1_load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  t1_load->attr.api.type = af::ApiType::kAPITypeCompute;
  t1_load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  t1_load->attr.sched.loop_axis = z0.id;
  t1_load->outputs[0].attr.vectorized_axis = {z1.id};
  t1_load->outputs[0].attr.vectorized_strides = {One};
  t1_load->outputs[0].attr.dtype = ge::DT_UINT8;
  t1_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  t1_load->outputs[0].attr.mem.tensor_id = 1;
  t1_load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  t1_load->outputs[0].attr.que.id = 2;
  t1_load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id};
  ge->outputs[0].attr.vectorized_strides = {One};
  ge->outputs[0].attr.dtype = ge::DT_UINT32;  // Compare 输出 dtype
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 3;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_UINT8;  // 输出到 uint8_t
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(t1_load->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor compare_input1;
  compare_input1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_input2;
  compare_input2.id = t1_load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_output;
  compare_output.id = ge->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor store_output;
  store_output.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_ge = ge->GetName() + "_" + ge->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor ge_tensor(ge->outputs[0], tensor_ge);
  ge_tensor.init_as_mask_reg_ = false;
  ge_tensor.dtype_ = ge::DT_UINT32;

  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor store_tensor(store->outputs[0], tensor_store);

  // Compare 的两个输入 tensor（需要添加到 tensor_mng）
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.id_ = load->outputs[0].attr.mem.tensor_id;
  load_tensor.init_as_mask_reg_ = false;
  load_tensor.dtype_ = ge::DT_UINT8;

  auto tensor_t1_load = t1_load->GetName() + "_" + t1_load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_load_tensor(t1_load->outputs[0], tensor_t1_load);
  t1_load_tensor.id_ = t1_load->outputs[0].attr.mem.tensor_id;
  t1_load_tensor.init_as_mask_reg_ = false;
  t1_load_tensor.dtype_ = ge::DT_UINT8;

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_load_tensor);
  tensor_mng.AddTensor(ge_tensor);
  tensor_mng.AddTensor(store_tensor);

  // 先调用 Compare API（会产生 Duplicate）
  codegen::MicroCompareApiCall compare_call("GE");
  EXPECT_EQ(compare_call.Init(ge), 0);
  compare_call.AddInput(compare_input1.id);
  compare_call.AddInput(compare_input2.id);
  compare_call.AddOutput(compare_output.id);

  std::string compare_result;
  cp.max_dtype_size = "uint32_t";
  compare_call.Generate(tensor_mng, tpipe, cp, compare_result);
  // Compare 输出不是 MaskReg（有 Store 消费者），会产生 Duplicate
  EXPECT_TRUE(compare_result.find("AscendC::Reg::Duplicate") != std::string::npos);

  // 再调用 Store API
  codegen::MicroStoreApiCall store_call("Store");
  EXPECT_EQ(store_call.Init(store), 0);
  store_call.AddInput(compare_output.id);
  store_call.AddOutput(store_output.id);

  std::string store_result;
  store_call.Generate(tensor_mng, tpipe, cp, store_result);
  // StoreAlign 使用 DIST_PACK4_B32 完成 uint32 → uint8，不需要额外 Pack
  EXPECT_TRUE(store_result.find("AscendC::MicroAPI::StoreAlign") != std::string::npos);
  size_t pos = 0;
  int pack_count = 0;
  std::string pack_str = "Pack";
  while ((pos = store_result.find(pack_str, pos)) != std::string::npos) {
    pack_count++;
    pos += pack_str.length();
  }
  EXPECT_EQ(pack_count, 0);
}

TEST(CodegenKernel, StoreMicroApiCall_MaskReg_Store_FromUint64) {
  af::AscGraph graph("test_mask_reg_store_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data t1_op("t1", graph);
  Load load_op("load");
  Load t1_load_op("t1_load");
  Ge ge_op("ge");           // Compare 算子
  Where where_op("where");  // Where 算子
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(t1_load_op);
  graph.AddNode(ge_op);
  graph.AddNode(where_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  t1_load_op.x = t1_op.y;
  t1_load_op.attr.sched.axis = {z0.id, z1.id};
  *t1_load_op.y.axis = {z0.id, z1.id};
  *t1_load_op.y.repeats = {s0, s1};
  *t1_load_op.y.strides = {s1, One};

  ge_op.x1 = load_op.y;
  ge_op.x2 = t1_load_op.y;
  ge_op.attr.sched.axis = {z0.id, z1.id};
  *ge_op.y.axis = {z0.id, z1.id};
  *ge_op.y.repeats = {s0, s1};
  *ge_op.y.strides = {s1, One};

  // Compare 输出同时连接到 Where mask 输入和 Store 输入
  where_op.x1 = ge_op.y;  // mask 输入（Compare 输出）
  where_op.x2 = load_op.y;
  where_op.x3 = t1_load_op.y;
  where_op.attr.sched.axis = {z0.id, z1.id};
  *where_op.y.axis = {z0.id, z1.id};
  *where_op.y.repeats = {s0, s1};
  *where_op.y.strides = {s1, One};

  store_op.x = ge_op.y;  // Compare 输出连接到 Store 输入
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

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

  auto t1_load = graph.FindNode("t1_load");
  t1_load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  t1_load->attr.api.type = af::ApiType::kAPITypeCompute;
  t1_load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  t1_load->attr.sched.loop_axis = z0.id;
  t1_load->outputs[0].attr.vectorized_axis = {z1.id};
  t1_load->outputs[0].attr.vectorized_strides = {One};
  t1_load->outputs[0].attr.dtype = ge::DT_UINT8;
  t1_load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  t1_load->outputs[0].attr.mem.tensor_id = 1;
  t1_load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  t1_load->outputs[0].attr.que.id = 2;
  t1_load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id};
  ge->outputs[0].attr.vectorized_strides = {One};
  ge->outputs[0].attr.dtype = ge::DT_UINT64;  // Compare 输出 dtype
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 3;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_UINT8;  // 输出到 uint8_t
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 4;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(t1_load->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor compare_input1;
  compare_input1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_input2;
  compare_input2.id = t1_load->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor compare_output;
  compare_output.id = ge->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor store_output;
  store_output.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_ge = ge->GetName() + "_" + ge->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor ge_tensor(ge->outputs[0], tensor_ge);
  ge_tensor.init_as_mask_reg_ = false;
  ge_tensor.dtype_ = ge::DT_UINT64;

  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor store_tensor(store->outputs[0], tensor_store);

  // Compare 的两个输入 tensor（需要添加到 tensor_mng）
  auto tensor_load = load->GetName() + "_" + load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor load_tensor(load->outputs[0], tensor_load);
  load_tensor.id_ = load->outputs[0].attr.mem.tensor_id;
  load_tensor.init_as_mask_reg_ = false;
  load_tensor.dtype_ = ge::DT_UINT8;

  auto tensor_t1_load = t1_load->GetName() + "_" + t1_load->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor t1_load_tensor(t1_load->outputs[0], tensor_t1_load);
  t1_load_tensor.id_ = t1_load->outputs[0].attr.mem.tensor_id;
  t1_load_tensor.init_as_mask_reg_ = false;
  t1_load_tensor.dtype_ = ge::DT_UINT8;

  TensorManager tensor_mng;
  tensor_mng.AddTensor(load_tensor);
  tensor_mng.AddTensor(t1_load_tensor);
  tensor_mng.AddTensor(ge_tensor);
  tensor_mng.AddTensor(store_tensor);

  // 先调用 Compare API（会产生 Duplicate）
  codegen::MicroCompareApiCall compare_call("GE");
  EXPECT_EQ(compare_call.Init(ge), 0);
  compare_call.AddInput(compare_input1.id);
  compare_call.AddInput(compare_input2.id);
  compare_call.AddOutput(compare_output.id);

  std::string compare_result;
  cp.max_dtype_size = "uint64_t";
  compare_call.Generate(tensor_mng, tpipe, cp, compare_result);
  // Compare 输出不是 MaskReg（有 Store 消费者），会产生 Duplicate
  EXPECT_TRUE(compare_result.find("AscendC::Reg::Duplicate") != std::string::npos);

  // 再调用 Store API
  codegen::MicroStoreApiCall store_call("Store");
  EXPECT_EQ(store_call.Init(store), 0);
  store_call.AddInput(compare_output.id);
  store_call.AddOutput(store_output.id);

  std::string store_result;
  store_call.Generate(tensor_mng, tpipe, cp, store_result);
  // Pack uint64 → uint32（1次数据Pack + 1次MaskPack），StoreAlign 使用 DIST_PACK4_B32 完成 uint32 → uint8
  EXPECT_TRUE(store_result.find("AscendC::Reg::Pack") != std::string::npos);
  EXPECT_TRUE(store_result.find("AscendC::Reg::MaskPack") != std::string::npos);
  EXPECT_TRUE(store_result.find("AscendC::MicroAPI::StoreAlign") != std::string::npos);
  size_t pos = 0;
  int pack_count = 0;
  std::string pack_str = "Pack";
  while ((pos = store_result.find(pack_str, pos)) != std::string::npos) {
    pack_count++;
    pos += pack_str.length();
  }
  // 包含 Reg::Pack 和 MaskPack，共 2 次
  EXPECT_EQ(pack_count, 2);
}

// ============================================================================
// VF Store dtype 转换测试（使用 StoreDist::DIST_PACK 和 Pack）
// ============================================================================

TEST(CodegenKernel, StoreMicroApiCall_VF_Store_Int64ToInt32) {
  af::AscGraph graph("test_vf_store_dtype_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  store_op.x = abs_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT64;  // 8字节
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = ge::DT_INT64;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_INT32;  // 4字节输出
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(abs->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_abs = abs->GetName() + "_" + abs->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(abs->outputs[0], tensor_abs);
  tensor.dtype_ = ge::DT_INT64;  // RegTensor dtype 是 int64

  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(store->outputs[0], tensor_store);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);

  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int64_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_TRUE(result.find("AscendC::MicroAPI::StoreAlign") != std::string::npos);
}

TEST(CodegenKernel, StoreMicroApiCall_VF_Store_Int32ToInt8) {
  af::AscGraph graph("test_vf_store_dtype_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Abs abs_op("abs");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(abs_op);
  graph.AddNode(store_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  abs_op.x = load_op.y;
  abs_op.attr.sched.axis = {z0.id, z1.id};
  *abs_op.y.axis = {z0.id, z1.id};
  *abs_op.y.repeats = {s0, s1};
  *abs_op.y.strides = {s1, One};

  store_op.x = abs_op.y;
  store_op.ir_attr.SetOffset(af::Symbol(0));
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_INT32;  // 4字节
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto abs = graph.FindNode("abs");
  abs->attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs->attr.api.type = af::ApiType::kAPITypeCompute;
  abs->attr.api.unit = af::ComputeUnit::kUnitVector;
  abs->attr.sched.loop_axis = z0.id;
  abs->outputs[0].attr.vectorized_axis = {z1.id};
  abs->outputs[0].attr.vectorized_strides = {One};
  abs->outputs[0].attr.dtype = ge::DT_INT32;
  abs->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  abs->outputs[0].attr.mem.tensor_id = 1;
  abs->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  abs->outputs[0].attr.que.id = 2;
  abs->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  store->attr.sched.loop_axis = z0.id;
  store->outputs[0].attr.vectorized_axis = {z1.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.dtype = ge::DT_INT8;  // 1字节输出
  store->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  store->outputs[0].attr.que.id = 3;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(abs->outputs[0]);
  tpipe.AddTensor(store->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(ge::SizeVar(s0));
  tiler.AddSizeVar(ge::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = abs->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor y1;
  y1.id = store->outputs[0].attr.mem.tensor_id;

  codegen::CallParam cp = {"p_reg", "offset"};
  auto tensor_abs = abs->GetName() + "_" + abs->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor(abs->outputs[0], tensor_abs);
  tensor.dtype_ = ge::DT_INT32;

  auto tensor_store = store->GetName() + "_" + store->GetOpDesc()->GetOutputNameByIndex(0);
  MicroApiTensor tensor1(store->outputs[0], tensor_store);

  TensorManager tensor_mng;
  tensor_mng.AddTensor(tensor);
  tensor_mng.AddTensor(tensor1);

  codegen::MicroStoreApiCall call_0("Store");
  EXPECT_EQ(call_0.Init(store), 0);
  call_0.AddInput(x1.id);
  call_0.AddOutput(y1.id);

  std::string result;
  cp.max_dtype_size = "int32_t";
  call_0.Generate(tensor_mng, tpipe, cp, result);
  EXPECT_TRUE(result.find("AscendC::MicroAPI::StoreAlign") != std::string::npos);
}
