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
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "compare_v2_api_call.h"
#include "ascir_node_param/ascir_node_param.h"

#include "runtime_stub.h"
#include "platform_context.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

class CompareV2ApicallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Counter) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = load_op2.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2, s2, One};

  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
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
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {s2, One};
  ge->outputs[0].attr.dtype = af::DT_UINT8;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 3;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{
          "CompareExtend<float, 2, CMPMODE::ge>(local_3[0], local_0[0], local_1[0], {static_cast<uint16_t>(t->s1), "
          "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
          "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});

  const auto params = ascir_param::GetAscirNodeParams(ge);
  ASSERT_NE(params, nullptr);
  const auto *compare_params = std::get_if<ascir_param::CompareNodeParams>(&params->specific_params);
  ASSERT_NE(compare_params, nullptr);
  EXPECT_TRUE(compare_params->valid);
  EXPECT_FALSE(compare_params->is_scalar);
  ASSERT_EQ(compare_params->output_dims.size(), 2U);
  EXPECT_STREQ(compare_params->output_dims[0].Serialize().get(), "s1");
  EXPECT_STREQ(compare_params->output_dims[1].Serialize().get(), "s2");
  EXPECT_STREQ(compare_params->output_strides[0].Serialize().get(), "s2");
  EXPECT_STREQ(compare_params->input_strides[0].Serialize().get(), "s2");
}

TEST(CompareV2ApiCallTest, BoolDtypeNameIsBool) {
  std::string dtype_name;
  EXPECT_EQ(codegen::Tensor::DtypeName(ge::DT_BOOL, dtype_name), af::SUCCESS);
  EXPECT_EQ(dtype_name, "bool");
}

namespace {
void BuildCompareScalarCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                    const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                    const af::Axis &z2);
void InitCompareScalarCvStageAttrs(af::AscGraph &graph, const af::Expression &s2, const af::Axis &z0,
                                   const af::Axis &z1, const af::Axis &z2);
}  // namespace

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Scalar) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildCompareScalarCvStageGraph(graph, s0, s1, s2, z0, z1, z2);
  InitCompareScalarCvStageAttrs(graph, s2, z0, z1, z2);

  auto load1 = graph.FindNode("load1");
  auto constant_node = graph.FindNode("constant");
  auto ge = graph.FindNode("ge");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{
          "CompareScalarExtend<float, 1, CMPMODE::ge>(local_2[0], local_0[0], scalar_1, "
          "{static_cast<uint16_t>(local_0_actual_size)}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});\n"});

  const auto params = ascir_param::GetAscirNodeParams(ge);
  ASSERT_NE(params, nullptr);
  const auto *compare_params = std::get_if<ascir_param::CompareNodeParams>(&params->specific_params);
  ASSERT_NE(compare_params, nullptr);
  EXPECT_TRUE(compare_params->valid);
  EXPECT_TRUE(compare_params->is_scalar);
  ASSERT_EQ(compare_params->output_dims.size(), 1U);
  EXPECT_STREQ(compare_params->output_dims[0].Serialize().get(), "(s1 * s2)");
  EXPECT_STREQ(compare_params->output_strides[0].Serialize().get(), "1");
  EXPECT_STREQ(compare_params->input_strides[0].Serialize().get(), "1");
}

namespace {
void BuildCompareScalarCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                    const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                    const af::Axis &z2) {
  Data x_op1("x1", graph);
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  Load load_op1("load1");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(constant_op);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = constant_op.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2, s2, One};
}

void InitCompareScalarCvStageAttrs(af::AscGraph &graph, const af::Expression &s2, const af::Axis &z0,
                                   const af::Axis &z1, const af::Axis &z2) {
  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant_node->outputs[0].attr.dtype = af::DT_FLOAT;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {s2, One};
  ge->outputs[0].attr.dtype = af::DT_UINT8;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void InitCompareScalarCvStageTpipe(codegen::TPipe &tpipe, codegen::Tiler &tiler, const af::AscGraph &graph,
                                   const af::AscNodePtr &load1, const af::AscNodePtr &constant_node,
                                   const af::AscNodePtr &ge, const af::Expression &s0, const af::Expression &s1,
                                   const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                   const af::Axis &z2) {
  tpipe.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  tpipe.cube_output_tensor_id = load1->outputs[0].attr.mem.tensor_id;
  tpipe.is_inductor = false;
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
}
}  // namespace

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Scalar_CVStage) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildCompareScalarCvStageGraph(graph, s0, s1, s2, z0, z1, z2);
  InitCompareScalarCvStageAttrs(graph, s2, z0, z1, z2);

  auto load1 = graph.FindNode("load1");
  auto constant_node = graph.FindNode("constant");
  auto ge = graph.FindNode("ge");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  InitCompareScalarCvStageTpipe(tpipe, tiler, graph, load1, constant_node, ge, s0, s1, s2, z0, z1, z2);
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  EXPECT_NE(result.find("CompareScalarExtend<float, 2, CMPMODE::ge>(local_2[0], local_0[0], scalar_1,"),
            std::string::npos);
  EXPECT_NE(result.find("{static_cast<uint16_t>(curAivM), static_cast<uint16_t>(curAivN)}"), std::string::npos);
  EXPECT_NE(result.find("{static_cast<uint16_t>(((curAivN + 32 - 1) / 32 * 32)), static_cast<uint16_t>(1)}"),
            std::string::npos);
  EXPECT_NE(result.find("{static_cast<uint16_t>(curAlignN), static_cast<uint16_t>(1)}"), std::string::npos);
}

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Scalar_Normal) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1.0");
  Load load_op1("load1");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(constant_op);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = constant_op.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2 + s1 * s2 + s1 * s2, s2 + s2, One};

  auto load1 = graph.FindNode("load1");
  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {stride * z1.size, stride, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT16;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant_node->outputs[0].attr.dtype = af::DT_FLOAT;

  auto ge = graph.FindNode("ge");
  auto size1 = af::GetSizeByDataType(af::DT_UINT8);
  auto stride1 = af::sym::Align(z2.size, 32 / size1);
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {stride1 * z1.size + stride1 * z1.size, stride1, One};
  ge->outputs[0].attr.dtype = af::DT_UINT8;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1.0", constant_node->outputs[0], "const_y");
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            std::string{"for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\nCompareExtend<float,2, "
                        "CMPMODE::ge>(local_2[outer_for_0 * ((64 * Ceiling((Rational(1 , 32) * t->s2)) * t->s1))/(1)], "
                        "local_0[outer_for_0 * ((8 * Ceiling((Rational(1 , 8) * t->s2)) * t->s1))/(1)], "
                        "local_blk_tensor_of_scalar_1[0], {static_cast<uint16_t>(t->s1), "
                        "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(((32 * Ceiling((Rational(1 , 32) * "
                        "t->s2))))/(1)), static_cast<uint16_t>(1)}, {static_cast<uint16_t>(((8 * Ceiling((Rational(1 , "
                        "8) * t->s2))))/(1)), static_cast<uint16_t>(1)});\n\n}\n"});
}

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Normal) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = load_op2.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2 + s1 * s2 + s1 * s2, s2 + s2, One};

  auto load1 = graph.FindNode("load1");
  auto size = af::GetSizeByDataType(af::DT_FLOAT);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {stride * z1.size, stride, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
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
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {stride * z1.size, stride, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 0;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  auto size1 = af::GetSizeByDataType(af::DT_UINT8);
  auto stride1 = af::sym::Align(z2.size, 32 / size1);
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {stride1 * z1.size + stride1 * z1.size, stride1, One};
  ge->outputs[0].attr.dtype = af::DT_UINT8;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{"for(int outer_for_0 = 0; outer_for_0 < t->s0; outer_for_0++) {\nCompareExtend<float, 2, "
                  "CMPMODE::ge>(local_2[outer_for_0 * ((64 * Ceiling((Rational(1 , 32) * t->s2)) * t->s1))/(1)], "
                  "local_0[outer_for_0 * ((8 * Ceiling((Rational(1 , 8) * t->s2)) * t->s1))/(1)], local_0[outer_for_0 "
                  "* ((8 * Ceiling((Rational(1 , 8) * t->s2)) * t->s1))/(1)], {static_cast<uint16_t>(t->s1), "
                  "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(((32 * Ceiling((Rational(1 , 32) * "
                  "t->s2))))/(1)), static_cast<uint16_t>(1)}, {static_cast<uint16_t>(((8 * Ceiling((Rational(1 , 8) * "
                  "t->s2))))/(1)), static_cast<uint16_t>(1)});\n\n}\n"});
}

namespace {
void BuildCompareNormalCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                    const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                    const af::Axis &z2) {
  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op2.x = x_op2.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};
  *load_op2.y.strides = {s1 * s2, s2, One};
  ge_op.x1 = load_op1.y;
  ge_op.x2 = load_op2.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2, s2, One};
}

void InitCompareNormalCvStageAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Axis &z2) {
  auto input_stride = af::sym::Align(z2.size, 8);
  auto output_stride = af::sym::Align(z2.size, 32);
  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {input_stride * z1.size, input_stride, One};
  load1->outputs[0].attr.dtype = af::DT_FLOAT;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {input_stride * z1.size, input_stride, One};
  load2->outputs[0].attr.dtype = af::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 0;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {output_stride * z1.size, output_stride, One};
  ge->outputs[0].attr.dtype = af::DT_UINT8;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void InitCompareNormalCvStageTpipe(codegen::TPipe &tpipe, codegen::Tiler &tiler, const af::AscNodePtr &load1,
                                   const af::AscNodePtr &load2, const af::AscNodePtr &ge, const af::Expression &s0,
                                   const af::Expression &s1, const af::Expression &s2, const af::Axis &z0,
                                   const af::Axis &z1, const af::Axis &z2) {
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
}
}  // namespace

TEST(CompareV2ApiCallTest, CompareV2ApiCall_Normal_CVStageKeepsLogicalCount) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildCompareNormalCvStageGraph(graph, s0, s1, s2, z0, z1, z2);
  InitCompareNormalCvStageAttrs(graph, z0, z1, z2);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto ge = graph.FindNode("ge");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  InitCompareNormalCvStageTpipe(tpipe, tiler, load1, load2, ge, s0, s1, s2, z0, z1, z2);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, result), 0);
  EXPECT_NE(result.find("static_cast<uint16_t>(t->s2)"), std::string::npos);
  EXPECT_EQ(result.find("static_cast<uint16_t>(((t->s2 + 32 - 1) / 32 * 32))"), std::string::npos);
}

namespace {
void BuildCompareBoolNormalGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                 const af::Expression &s2, const af::Axis &z0, const af::Axis &z1, const af::Axis &z2) {
  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = load_op2.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2, s2, One};
}

void InitCompareBoolNormalAttrs(af::AscGraph &graph, const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                const af::Axis &z2) {
  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_BOOL;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;
  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2 + s2, One};
  load2->outputs[0].attr.dtype = af::DT_BOOL;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {s2, One};
  ge->outputs[0].attr.dtype = af::DT_BOOL;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 3;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;
}
}  // namespace

// BOOL 输入：模板参数为 bool，调用点无 ReinterpretCast（regbase 内部按 uint8 执行）
TEST(CompareV2ApiCallTest, CompareV2ApiCall_BoolNormal) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildCompareBoolNormalGraph(graph, s0, s1, s2, z0, z1, z2);
  InitCompareBoolNormalAttrs(graph, s2, z0, z1, z2);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto ge = graph.FindNode("ge");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result.find(".ReinterpretCast"), std::string::npos);
  EXPECT_EQ(
      result,
      std::string{
          "CompareExtend<bool, 2, CMPMODE::ge>(local_3[0], local_0[0], local_1[0], {static_cast<uint16_t>(t->s1), "
          "static_cast<uint16_t>(t->s2)}, {static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)}, "
          "{static_cast<uint16_t>(t->s2), static_cast<uint16_t>(1)});\n"});
}

namespace {
void BuildCompareBoolScalarGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                 const af::Expression &s2, const af::Axis &z0, const af::Axis &z1, const af::Axis &z2) {
  Data x_op1("x1", graph);
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("1");
  Load load_op1("load1");
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(load_op1);
  graph.AddNode(constant_op);
  graph.AddNode(ge_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.axis = {z0.id, z1.id, z2.id};
  *load_op1.y.repeats = {s0, s1, s2};
  *load_op1.y.strides = {s1 * s2, s2, One};

  ge_op.x1 = load_op1.y;
  ge_op.x2 = constant_op.y;
  *ge_op.y.axis = {z0.id, z1.id, z2.id};
  *ge_op.y.repeats = {s0, s1, s2};
  *ge_op.y.strides = {s1 * s2, s2, One};
}

void InitCompareBoolScalarAttrs(af::AscGraph &graph, const af::Expression &s2, const af::Axis &z0, const af::Axis &z1,
                                const af::Axis &z2) {
  auto load1 = graph.FindNode("load1");
  load1->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load1->attr.api.type = af::ApiType::kAPITypeCompute;
  load1->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load1->attr.sched.loop_axis = z0.id;
  load1->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load1->outputs[0].attr.vectorized_strides = {s2, One};
  load1->outputs[0].attr.dtype = af::DT_BOOL;
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.tensor_id = 0;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.que.id = 1;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 1;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  constant_node->outputs[0].attr.dtype = af::DT_BOOL;

  auto ge = graph.FindNode("ge");
  ge->attr.api.compute_type = af::ComputeType::kComputeElewise;
  ge->attr.api.type = af::ApiType::kAPITypeCompute;
  ge->attr.api.unit = af::ComputeUnit::kUnitVector;
  ge->attr.sched.loop_axis = z0.id;
  ge->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  ge->outputs[0].attr.vectorized_strides = {s2, One};
  ge->outputs[0].attr.dtype = af::DT_BOOL;
  ge->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ge->outputs[0].attr.mem.tensor_id = 2;
  ge->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ge->outputs[0].attr.que.id = 2;
  ge->outputs[0].attr.opt.merge_scope = af::kIdNone;
}
}  // namespace

// BOOL 输入 + const scalar：CompareScalarExtend 模板参数为 bool
TEST(CompareV2ApiCallTest, CompareV2ApiCall_BoolScalar) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  BuildCompareBoolScalarGraph(graph, s0, s1, s2, z0, z1, z2);
  InitCompareBoolScalarAttrs(graph, s2, z0, z1, z2);

  auto load1 = graph.FindNode("load1");
  auto constant_node = graph.FindNode("constant");
  auto ge = graph.FindNode("ge");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor("1", constant_node->outputs[0], "const_y");
  tpipe.AddTensor(ge->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1, x2;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  x2.id = constant_node->outputs[0].attr.mem.tensor_id;

  codegen::CompareV2ApiCall call("ge");
  EXPECT_EQ(call.Init(ge), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result.find(".ReinterpretCast"), std::string::npos);
  EXPECT_EQ(
      result,
      std::string{"CompareScalarExtend<bool, 1, CMPMODE::ge>(local_2[0], local_0[0], scalar_1, "
                  "{static_cast<uint16_t>(local_0_actual_size)}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>"
                  "(1)});\n"});
}
