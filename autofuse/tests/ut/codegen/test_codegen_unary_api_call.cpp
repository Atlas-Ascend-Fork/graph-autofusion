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
#include "elewise/binary_api_call.h"
#include "utils/api_call_utils.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
codegen::Tensor MakeCvPointwiseTensor(af::AscGraph &graph, int64_t id) {
  auto tensor = graph.FindNode("x")->outputs[0];
  tensor.attr.dtype = DT_FLOAT;
  tensor.attr.mem.tensor_id = id;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.que.id = id;
  tensor.attr.mem.reuse_id = 0;
  tensor.attr.axis = {0, 1};
  tensor.attr.repeats = {af::Symbol("m"), af::Symbol("n")};
  tensor.attr.strides = {af::Symbol("n"), One};
  tensor.attr.vectorized_axis = {0, 1};
  tensor.attr.vectorized_strides = {af::Symbol("n"), One};
  std::string dtype = "float";
  return codegen::Tensor(tensor, dtype);
}
}  // namespace

TEST(CodegenKernel, CvUbPointwiseFlatCallRetainsPaddingFallback) {
  af::AscGraph graph("test_graph");
  Data data("x", graph);
  auto input = MakeCvPointwiseTensor(graph, 0);
  auto rhs = MakeCvPointwiseTensor(graph, 1);
  auto output = MakeCvPointwiseTensor(graph, 2);
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  tpipe.cube_output_tensor_id = input.id;
  std::string result;
  UnaryApiCall unary("Abs");
  ASSERT_EQ(unary.Generate(tpipe, {}, {input}, {output}, result), af::SUCCESS);
  EXPECT_NE(result.find("curAlignN == curAivN"), std::string::npos);
  EXPECT_NE(result.find(GenBlockAlignNExpr(output, "curAivN") + " == curAivN"), std::string::npos);
  EXPECT_NE(result.find("Abs(local_2[0], local_0[0], ConvertToUint32(curAivM * curAivN));"), std::string::npos);
  EXPECT_NE(result.find("} else {\nfor (uint32_t cv_row"), std::string::npos);
  EXPECT_NE(result.find("local_0[cv_row * ConvertToUint32(curAlignN)]"), std::string::npos);
  EXPECT_EQ(result.find("curAivN *="), std::string::npos);

  BinaryApiCall binary("Sub");
  ASSERT_EQ(binary.Generate(tpipe, {}, {input, rhs}, {output}, result), af::SUCCESS);
  EXPECT_NE(result.find("Sub(local_2[0], local_0[0], local_1[0], ConvertToUint32(curAivM * curAivN));"),
            std::string::npos);
  EXPECT_NE(result.find("} else {\nfor (uint32_t cv_row"), std::string::npos);

  tpipe.is_conv2d = true;
  ASSERT_EQ(unary.Generate(tpipe, {}, {input}, {output}, result), af::SUCCESS);
  EXPECT_EQ(result.find("curAivM * curAivN"), std::string::npos);
  tpipe.is_conv2d = false;
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kDefault;
  ASSERT_EQ(unary.Generate(tpipe, {}, {input}, {output}, result), af::SUCCESS);
  EXPECT_EQ(result, "Abs(local_2[0], local_0[0], local_0_actual_size);\n");
  ASSERT_EQ(binary.Generate(tpipe, {}, {input, rhs}, {output}, result), af::SUCCESS);
  EXPECT_EQ(result, "Sub(local_2[0], local_0[0], local_1[0], local_0_actual_size);\n");
}

TEST(CodegenKernel, CvUbPointwiseFlatCallRejectsUnsupportedViewsAndAliasing) {
  af::AscGraph graph("test_graph");
  Data data("x", graph);
  auto input = MakeCvPointwiseTensor(graph, 0);
  auto output = MakeCvPointwiseTensor(graph, 1);
  Tiler tiler;
  TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  tpipe.cube_output_tensor_id = input.id;
  const auto generate = [&]() { return GenCvUbFuseFlatApiCondition(tpipe, {input}, output); };
  const auto dense = input;
  ASSERT_FALSE(generate().empty());
  input.vectorized_strides = {Zero, One};
  EXPECT_TRUE(generate().empty());
  input.vectorized_strides = {One, af::Symbol("m")};
  EXPECT_TRUE(generate().empty());
  input.vectorized_strides = dense.vectorized_strides;
  input.axis_size = {af::Symbol("n"), af::Symbol("m")};
  EXPECT_TRUE(generate().empty());
  input.axis_size = dense.axis_size;
  input.axis = {1, 0};
  EXPECT_TRUE(generate().empty());
  input.axis = dense.axis;
  input.dtype = DT_FLOAT16;
  EXPECT_TRUE(generate().empty());
  input.dtype = dense.dtype;
  input.is_constant = true;
  EXPECT_TRUE(generate().empty());
  input.is_constant = false;
  input.alloc_type = af::AllocType::kAllocTypeInvalid;
  EXPECT_TRUE(generate().empty());
  input.alloc_type = dense.alloc_type;
  input.que_id = output.que_id;
  input.reuse_id = output.reuse_id;
  EXPECT_TRUE(generate().empty());
  input.reuse_id = output.reuse_id + 1;
  EXPECT_TRUE(generate().empty());
  std::string position = "VECIN";
  auto queue = tpipe.ques.emplace(output.que_id, TQue(output.que_id, af::Position::kPositionVecIn, position));
  queue.first->second.share_group[input.reuse_id] = {input.id};
  queue.first->second.share_group[output.reuse_id] = {output.id};
  EXPECT_FALSE(generate().empty());
  queue.first->second.share_group[input.reuse_id].push_back(2);
  EXPECT_TRUE(generate().empty());
  input.que_id = dense.que_id;
  input.reuse_id = dense.reuse_id;
  output.dtype = DT_FLOAT16;
  EXPECT_TRUE(generate().empty());
  output.dtype = DT_FLOAT;
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kCommon;
  EXPECT_TRUE(generate().empty());
}

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
