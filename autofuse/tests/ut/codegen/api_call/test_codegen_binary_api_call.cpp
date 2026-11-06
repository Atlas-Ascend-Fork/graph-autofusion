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
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "elewise/binary_api_call.h"

using namespace std;
using namespace ascir;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(CodegenKernel, Reciprocal_divs) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Div div_op("div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  div_op.x2 = load_op.y;
  div_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto div = graph.FindNode("div");
  div->attr.api.unit = af::ComputeUnit::kUnitVector;
  div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  div->outputs[0].attr.dtype = ge::DT_FLOAT16;
  div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  div->outputs[0].attr.mem.tensor_id = 2;
  div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  div->outputs[0].attr.que.id = 2;
  div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(div->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Div");
  EXPECT_EQ(call.Init(div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"Divs<half, true>(local_2[0], local_0[0], (half)1.0, tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, DivWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Div div_op("div");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(div_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  div_op.x1 = load_op.y;
  div_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto div = graph.FindNode("div");
  div->attr.api.unit = af::ComputeUnit::kUnitVector;
  div->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  div->outputs[0].attr.dtype = ge::DT_FLOAT16;
  div->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  div->outputs[0].attr.mem.tensor_id = 3;
  div->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  div->outputs[0].attr.que.id = 3;
  div->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // div load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // div load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add div tensor
  EXPECT_EQ(tpipe.AddTensor(div->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Div");
  EXPECT_EQ(call.Init(div), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(
      result,
      std::string{
          "Divs<half, true>(local_3[0], local_0[0], (half)local_2_ub_scalar, tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, Subs) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Sub sub_op("sub");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(sub_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  sub_op.x2 = load_op.y;
  sub_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto sub = graph.FindNode("sub");
  sub->attr.api.unit = af::ComputeUnit::kUnitVector;
  sub->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  sub->outputs[0].attr.dtype = ge::DT_FLOAT16;
  sub->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sub->outputs[0].attr.mem.tensor_id = 2;
  sub->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sub->outputs[0].attr.que.id = 2;
  sub->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(sub->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Sub");
  EXPECT_EQ(call.Init(sub), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"Subs<half, true>(local_2[0], local_0[0], (half)1.0, tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, SubWithFirstInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Sub sub_op("sub");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(sub_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  sub_op.x2 = load_op.y;
  sub_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto sub = graph.FindNode("sub");
  sub->attr.api.unit = af::ComputeUnit::kUnitVector;
  sub->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  sub->outputs[0].attr.dtype = ge::DT_FLOAT16;
  sub->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sub->outputs[0].attr.mem.tensor_id = 2;
  sub->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sub->outputs[0].attr.que.id = 2;
  sub->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(sub->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant_node->outputs[0].attr.mem.tensor_id;
  x2.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Sub");
  EXPECT_EQ(call.Init(sub), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"Subs<half, false>(local_2[0], local_0[0], (half)1.0, tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, SubWithDoubleInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Scalar constant_op_0("constant_0");
  constant_op_0.ir_attr.SetValue("1.0");
  af::ascir_op::Sub sub_op("sub");
  Scalar constant_op_1("constant_1");
  constant_op_1.ir_attr.SetValue("1.0");
  graph.AddNode(constant_op_0);
  graph.AddNode(sub_op);
  graph.AddNode(constant_op_1);
  sub_op.x2 = constant_op_0.y;
  sub_op.x1 = constant_op_1.y;

  // graph.SetInputs({x_op});
  auto constant_node_0 = graph.FindNode("constant_0");
  constant_node_0->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node_0->outputs[0].attr.mem.tensor_id = 0;
  constant_node_0->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto constant_node_1 = graph.FindNode("constant_1");
  constant_node_1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node_1->outputs[0].attr.mem.tensor_id = 1;
  constant_node_1->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto sub = graph.FindNode("sub");
  sub->attr.api.unit = af::ComputeUnit::kUnitVector;
  sub->outputs[0].attr.dtype = ge::DT_FLOAT16;
  sub->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sub->outputs[0].attr.mem.tensor_id = 2;
  sub->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sub->outputs[0].attr.que.id = 2;
  sub->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor("1.0", constant_node_0->outputs[0], "const_x");
  tpipe.AddTensor(sub->outputs[0]);
  tpipe.AddTensor("1.0", constant_node_1->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = constant_node_0->outputs[0].attr.mem.tensor_id;
  x2.id = constant_node_1->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Sub");
  EXPECT_EQ(call.Init(sub), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Subs(local_2[0], (float)1.0, (float)1.0);\n"});
}

TEST(CodegenKernel, SubWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Sub sub_op("sub");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(sub_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  sub_op.x1 = load_op.y;
  sub_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sub = graph.FindNode("sub");
  sub->attr.api.unit = af::ComputeUnit::kUnitVector;
  sub->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  sub->outputs[0].attr.dtype = ge::DT_FLOAT16;
  sub->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sub->outputs[0].attr.mem.tensor_id = 3;
  sub->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sub->outputs[0].attr.que.id = 3;
  sub->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // sub load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // sub load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add sub tensor
  EXPECT_EQ(tpipe.AddTensor(sub->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Sub");
  EXPECT_EQ(call.Init(sub), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(
      result,
      std::string{
          "Subs<half, true>(local_3[0], local_0[0], (half)local_2_ub_scalar, tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, addsApicallWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Add add_op("add");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(add_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  add_op.x1 = load_op.y;
  add_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto add = graph.FindNode("add");
  add->attr.api.unit = af::ComputeUnit::kUnitVector;
  add->outputs[0].attr.dtype = ge::DT_FLOAT16;
  add->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  add->outputs[0].attr.mem.tensor_id = 3;
  add->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  add->outputs[0].attr.que.id = 3;
  add->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // add load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(add->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Add");
  EXPECT_EQ(call.Init(add), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{"Adds(local_3[0], local_0[0], (half)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(CodegenKernel, AddWithSecondInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Add add_op("add");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(add_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  add_op.x2 = load_op.y;
  add_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto add = graph.FindNode("add");
  add->attr.api.unit = af::ComputeUnit::kUnitVector;
  add->outputs[0].attr.dtype = ge::DT_FLOAT16;
  add->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  add->outputs[0].attr.mem.tensor_id = 2;
  add->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  add->outputs[0].attr.que.id = 2;
  add->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(add->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Add");
  EXPECT_EQ(call.Init(add), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"Adds(local_2[0], local_0[0], (half)1.0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, GtWithSecondInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Gt gt_op("gt");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(gt_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  gt_op.x2 = load_op.y;
  gt_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto gt = graph.FindNode("gt");
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->outputs[0].attr.dtype = ge::DT_FLOAT16;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 2;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(gt->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("gt");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  codegen::Kernel kernel("test");
  EXPECT_EQ(kernel.ParseScalarNeedGenBlkTensors(constant_node, 1), af::SUCCESS);
}

TEST(CodegenKernel, MulsApicallWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Mul mul_op("mul");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.2");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(mul_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  mul_op.x1 = load_op.y;
  mul_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto mul = graph.FindNode("mul");
  mul->attr.api.unit = af::ComputeUnit::kUnitVector;
  mul->outputs[0].attr.dtype = ge::DT_FLOAT16;
  mul->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  mul->outputs[0].attr.mem.tensor_id = 3;
  mul->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  mul->outputs[0].attr.que.id = 3;
  mul->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // mul load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // mul load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add mul tensor
  EXPECT_EQ(tpipe.AddTensor(mul->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Mul");
  EXPECT_EQ(call.Init(mul), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{"Muls(local_3[0], local_0[0], (half)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(CodegenKernel, MaxApicallWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Maximum max_op("max");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.2");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(max_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {One, One};
  *load_op2.y.strides = {One, One};

  max_op.x1 = load_op.y;
  max_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.vectorized_strides = {One, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto max = graph.FindNode("max");
  max->attr.api.unit = af::ComputeUnit::kUnitVector;
  max->outputs[0].attr.dtype = ge::DT_FLOAT16;
  max->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  max->outputs[0].attr.mem.tensor_id = 3;
  max->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  max->outputs[0].attr.que.id = 3;
  max->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // max load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // max load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add max tensor
  EXPECT_EQ(tpipe.AddTensor(max->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("AscendC::Max");
  EXPECT_EQ(call.Init(max), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result,
            std::string{"AscendC::Maxs(local_3[0], local_0[0], (half)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(CodegenKernel, MaxWithSecondInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Maximum max_op("max");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(max_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  max_op.x2 = load_op.y;
  max_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto max = graph.FindNode("max");
  max->attr.api.unit = af::ComputeUnit::kUnitVector;
  max->outputs[0].attr.dtype = ge::DT_FLOAT16;
  max->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  max->outputs[0].attr.mem.tensor_id = 2;
  max->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  max->outputs[0].attr.que.id = 2;
  max->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(max->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("AscendC::Max");
  ;
  EXPECT_EQ(call.Init(max), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"AscendC::Maxs(local_2[0], local_0[0], (half)1.0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, MinApicallWithSecondInputIsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Minimum min_op("min");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.2");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(min_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {One, One};
  *load_op2.y.strides = {One, One};

  min_op.x1 = load_op.y;
  min_op.x2 = load_op2.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.vectorized_strides = {One, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto min = graph.FindNode("min");
  min->attr.api.unit = af::ComputeUnit::kUnitVector;
  min->outputs[0].attr.dtype = ge::DT_FLOAT16;
  min->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  min->outputs[0].attr.mem.tensor_id = 3;
  min->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  min->outputs[0].attr.que.id = 3;
  min->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // min load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

  // min load2 tensor // 构造load2是ub scalar的输入
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);

  // add min tensor
  EXPECT_EQ(tpipe.AddTensor(min->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("AscendC::Min");
  EXPECT_EQ(call.Init(min), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result,
            std::string{"AscendC::Mins(local_3[0], local_0[0], (half)local_2_ub_scalar, local_0_actual_size);\n"});
}

TEST(CodegenKernel, MinWithSecondInputIsScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Minimum min_op("min");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  graph.AddNode(load_op);
  graph.AddNode(min_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  min_op.x2 = load_op.y;
  min_op.x1 = constant_op.y;

  // graph.SetInputs({x_op});

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  auto min = graph.FindNode("min");
  min->attr.api.unit = af::ComputeUnit::kUnitVector;
  min->outputs[0].attr.dtype = ge::DT_FLOAT16;
  min->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  min->outputs[0].attr.mem.tensor_id = 2;
  min->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  min->outputs[0].attr.que.id = 2;
  min->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(min->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;
  x1.id = load->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("AscendC::Min");
  ;
  EXPECT_EQ(call.Init(min), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result, std::string{"AscendC::Mins(local_2[0], local_0[0], (half)1.0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, brc_inline_apicall_test) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x1_op("x1", graph);
  Data x2_op("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  af::ascir_op::Add add_op("add");

  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(add_op);

  load_op1.x = x1_op.y;
  load_op1.attr.sched.axis = {z0.id, z1.id};
  *load_op1.y.axis = {z0.id, z1.id};
  *load_op1.y.repeats = {s0, s1};
  *load_op1.y.strides = {s1, One};

  load_op2.x = x2_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {One, s1};
  *load_op2.y.strides = {One};

  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.attr.sched.axis = {z0.id, z1.id};
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load1->outputs[0].attr.vectorized_strides = {s1, One};
  load1->outputs[0].attr.dtype = ge::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto add = graph.FindNode("add");
  add->attr.api.unit = af::ComputeUnit::kUnitVector;
  add->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  add->outputs[0].attr.vectorized_strides = {s1, One};
  add->outputs[0].attr.dtype = ge::DT_FLOAT;
  add->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  add->outputs[0].attr.mem.tensor_id = 3;
  add->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  add->outputs[0].attr.que.id = 3;
  add->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  EXPECT_EQ(tpipe.AddTensor(load1->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(add->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("Add");
  EXPECT_EQ(call.Init(add), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result, std::string{"BinaryBrcInlineApiWithTwoVectorizedAxis<float>(local_3[0], local_0[0], local_2[0], "
                                "t->s0, t->s1, 1, 0, t->s1, 4, &AscendC::Add, &AscendC::Add);\n"});
}

TEST(CodegenKernel, BinaryApiCallLogicalOrWhenBothAreTensor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::LogicalOr logical_or_op("logical_or");
  graph.AddNode(logical_or_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  logical_or_op.x1 = load_op.y;
  logical_or_op.x2 = load_op2.y;
  *logical_or_op.y.axis = {z0.id, z1.id};
  *logical_or_op.y.repeats = {s0, s1};
  *logical_or_op.y.strides = {s1, One};

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

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto logical_or = graph.FindNode("logical_or");
  logical_or->attr.api.compute_type = af::ComputeType::kComputeElewise;
  logical_or->attr.api.type = af::ApiType::kAPITypeCompute;
  logical_or->attr.api.unit = af::ComputeUnit::kUnitVector;
  logical_or->attr.sched.loop_axis = z0.id;
  logical_or->outputs[0].attr.vectorized_axis = {z1.id};
  logical_or->outputs[0].attr.vectorized_strides = {One};
  logical_or->outputs[0].attr.dtype = ge::DT_BOOL;
  logical_or->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  logical_or->outputs[0].attr.mem.tensor_id = 3;
  logical_or->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  logical_or->outputs[0].attr.que.id = 2;
  logical_or->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(logical_or->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("LogicalOrExtend");
  EXPECT_EQ(call.Init(logical_or), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"LogicalOrExtend(local_3[0], local_0[0], local_1[0], local_0_actual_size);\n"});
}

TEST(CodegenKernel, BinaryApiCallLogicalOrWithFloatUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::LogicalOr logical_or_op("logical_or");
  graph.AddNode(logical_or_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  logical_or_op.x1 = load_op.y;
  logical_or_op.x2 = load_op2.y;
  *logical_or_op.y.axis = {z0.id, z1.id};
  *logical_or_op.y.repeats = {s0, s1};
  *logical_or_op.y.strides = {s1, One};

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

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto logical_or = graph.FindNode("logical_or");
  logical_or->attr.api.compute_type = af::ComputeType::kComputeElewise;
  logical_or->attr.api.type = af::ApiType::kAPITypeCompute;
  logical_or->attr.api.unit = af::ComputeUnit::kUnitVector;
  logical_or->attr.sched.loop_axis = z0.id;
  logical_or->outputs[0].attr.vectorized_axis = {z1.id};
  logical_or->outputs[0].attr.vectorized_strides = {One};
  logical_or->outputs[0].attr.dtype = ge::DT_BOOL;
  logical_or->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  logical_or->outputs[0].attr.mem.tensor_id = 3;
  logical_or->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  logical_or->outputs[0].attr.que.id = 2;
  logical_or->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为LogicalOr的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(logical_or->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("LogicalOrExtend");
  EXPECT_EQ(call.Init(logical_or), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{"LogicalOrExtends(local_3[0], local_0[0], (float)local_1_ub_scalar, local_0_actual_size);\n"});
}

TEST(CodegenKernel, BinaryApiCallLogicalAndWhenBothAreTensor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::LogicalAnd logical_and_op("logical_and");
  graph.AddNode(logical_and_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  logical_and_op.x1 = load_op.y;
  logical_and_op.x2 = load_op2.y;
  *logical_and_op.y.axis = {z0.id, z1.id};
  *logical_and_op.y.repeats = {s0, s1};
  *logical_and_op.y.strides = {s1, One};

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

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto logical_and = graph.FindNode("logical_and");
  logical_and->attr.api.compute_type = af::ComputeType::kComputeElewise;
  logical_and->attr.api.type = af::ApiType::kAPITypeCompute;
  logical_and->attr.api.unit = af::ComputeUnit::kUnitVector;
  logical_and->attr.sched.loop_axis = z0.id;
  logical_and->outputs[0].attr.vectorized_axis = {z1.id};
  logical_and->outputs[0].attr.vectorized_strides = {One};
  logical_and->outputs[0].attr.dtype = ge::DT_BOOL;
  logical_and->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  logical_and->outputs[0].attr.mem.tensor_id = 3;
  logical_and->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  logical_and->outputs[0].attr.que.id = 2;
  logical_and->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(logical_and->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("LogicalAndExtend");
  EXPECT_EQ(call.Init(logical_and), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"LogicalAndExtend(local_3[0], local_0[0], local_1[0], local_0_actual_size);\n"});
}

TEST(CodegenKernel, BinaryApiCallLogicalAndWithFloatUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::LogicalAnd logical_and_op("logical_and");
  graph.AddNode(logical_and_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  logical_and_op.x1 = load_op.y;
  logical_and_op.x2 = load_op2.y;
  *logical_and_op.y.axis = {z0.id, z1.id};
  *logical_and_op.y.repeats = {s0, s1};
  *logical_and_op.y.strides = {s1, One};

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

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto logical_and = graph.FindNode("logical_and");
  logical_and->attr.api.compute_type = af::ComputeType::kComputeElewise;
  logical_and->attr.api.type = af::ApiType::kAPITypeCompute;
  logical_and->attr.api.unit = af::ComputeUnit::kUnitVector;
  logical_and->attr.sched.loop_axis = z0.id;
  logical_and->outputs[0].attr.vectorized_axis = {z1.id};
  logical_and->outputs[0].attr.vectorized_strides = {One};
  logical_and->outputs[0].attr.dtype = ge::DT_INT16;
  logical_and->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  logical_and->outputs[0].attr.mem.tensor_id = 3;
  logical_and->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  logical_and->outputs[0].attr.que.id = 2;
  logical_and->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为LogicalOr的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(logical_and->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::BinaryApiCall call("LogicalAndExtend");
  EXPECT_EQ(call.Init(logical_and), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{"LogicalAndExtends(local_3[0], local_0[0], (float)local_1_ub_scalar, local_0_actual_size);\n"});
}
