/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir_ops.h"
#include "utils/api_call_factory.h"
#include "codegen_kernel.h"
#include "rand_int_api_call.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

namespace {

// 构造 RandInt 测试图：4 输入 Load（x1 seed/x2 counter/x3 low/x4 high）+ RandInt
// x1_is_scalar 为 true 时 seed 视图压缩为 ub_scalar（全 1 repeat），对齐前端 load_seed 标量形态
void BuildRandIntGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1, const af::Axis &z0,
                       const af::Axis &z1, bool x1_is_scalar) {
  Data x_op1("x1", graph);
  Data x_op2("x2", graph);
  Data x_op3("x3", graph);
  Data x_op4("x4", graph);
  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");
  Load load_op4("load4");
  af::ascir_op::RandInt rand_int_op("rand_int");
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(load_op3);
  graph.AddNode(load_op4);
  graph.AddNode(rand_int_op);

  load_op1.x = x_op1.y;
  load_op1.attr.sched.axis = {z0.id, z1.id};
  *load_op1.y.axis = {z0.id, z1.id};
  if (x1_is_scalar) {
    *load_op1.y.repeats = {One, One};
    *load_op1.y.strides = {Zero, Zero};
  } else {
    *load_op1.y.repeats = {s0, s1};
    *load_op1.y.strides = {s1, One};
  }

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1};
  *load_op2.y.strides = {s1, One};

  load_op3.x = x_op3.y;
  load_op3.attr.sched.axis = {z0.id, z1.id};
  *load_op3.y.axis = {z0.id, z1.id};
  *load_op3.y.repeats = {s0, s1};
  *load_op3.y.strides = {s1, One};

  load_op4.x = x_op4.y;
  load_op4.attr.sched.axis = {z0.id, z1.id};
  *load_op4.y.axis = {z0.id, z1.id};
  *load_op4.y.repeats = {s0, s1};
  *load_op4.y.strides = {s1, One};

  rand_int_op.x1 = load_op1.y;
  rand_int_op.x2 = load_op2.y;
  rand_int_op.x3 = load_op3.y;
  rand_int_op.x4 = load_op4.y;
  *rand_int_op.y.axis = {z0.id, z1.id};
  *rand_int_op.y.repeats = {s0, s1};
  *rand_int_op.y.strides = {s1, One};
}

void InitLoadNodeAttrs(af::AscGraph &graph, const std::string &name, const af::Axis &z0, const af::Axis &z1,
                       ge::DataType dtype, int64_t tensor_id) {
  auto load = graph.FindNode(name.c_str());
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = dtype;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = tensor_id;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void InitRandIntNodeAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, int64_t tensor_id) {
  auto rand_int = graph.FindNode("rand_int");
  rand_int->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rand_int->attr.api.type = af::ApiType::kAPITypeCompute;
  rand_int->attr.api.unit = af::ComputeUnit::kUnitVector;
  rand_int->attr.sched.loop_axis = z0.id;
  rand_int->outputs[0].attr.vectorized_axis = {z1.id};
  rand_int->outputs[0].attr.vectorized_strides = {One};
  rand_int->outputs[0].attr.dtype = ge::DT_INT64;
  rand_int->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rand_int->outputs[0].attr.mem.tensor_id = tensor_id;
  rand_int->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rand_int->outputs[0].attr.que.id = 2;
  rand_int->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

}  // namespace

// RandInt codegen 实现基本信息：API 名、调用行类、头文件、SIMT dcache 预留
TEST(RandIntRegApiCallTest, RandIntCodegenBasicInfo) {
  af::AscGraph graph("test_rand_int_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandIntGraph(graph, s0, s0, z0, z0, true);
  auto rand_int = graph.FindNode("rand_int");

  af::ascir::RandIntAscIrCodegenImplV2 codegen;

  EXPECT_EQ(codegen.GetApiName(), std::string("RandInt"));
  EXPECT_EQ(codegen.GetApiCallName(), std::string("RandIntApiCall"));

  auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1U);
  EXPECT_EQ(headers[0], std::string("randint_reg_base.h"));

  auto include_headers = codegen.IncludeApiHeaderFiles();
  ASSERT_FALSE(include_headers.empty());
  EXPECT_EQ(include_headers[0], std::string("simt_api/cpp/kernel_simt_intf.h"));

  // SIMT dcache 预留 40KB（与 Remainder/Sin 等 SIMT regbase 家族一致）
  EXPECT_EQ(codegen.GetSimtDcacheSize(*rand_int), 40 * 1024);
}

// seed 为 ub_scalar 标量场景（对齐前端 load_seed 标量形态）：
// seed 参数生成 (int64_t)xxx_ub_scalar 标量形态，size 尾参取 y 的 actual_size（非 seed 的）
TEST(RandIntRegApiCallTest, RandIntApiCall_SeedIsUbScalar) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildRandIntGraph(graph, s0, s1, z0, z1, true);

  InitLoadNodeAttrs(graph, "load1", z0, z1, ge::DT_INT64, 0);
  InitLoadNodeAttrs(graph, "load2", z0, z1, ge::DT_INT32, 1);
  InitLoadNodeAttrs(graph, "load3", z0, z1, ge::DT_INT64, 2);
  InitLoadNodeAttrs(graph, "load4", z0, z1, ge::DT_INT64, 3);
  InitRandIntNodeAttrs(graph, z0, z1, 4);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto load3 = graph.FindNode("load3");
  auto load4 = graph.FindNode("load4");
  auto rand_int = graph.FindNode("rand_int");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  // x1 seed：ub_scalar 标量（repeats 全 1），白名单放行后 need_gen_get_value 生效
  std::string dtype_name;
  codegen::Tensor::DtypeName(load1->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x1(load1->outputs[0], dtype_name, "t_x1");
  EXPECT_EQ(t_x1.Init(), 0);
  t_x1.is_ub_scalar = true;
  t_x1.need_gen_get_value_of_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t_x1), 0);

  codegen::Tensor t_x2(load2->outputs[0], dtype_name, "t_x2");
  EXPECT_EQ(t_x2.Init(), 0);
  EXPECT_EQ(tpipe.AddTensor(t_x2), 0);

  codegen::Tensor t_x3(load3->outputs[0], dtype_name, "t_x3");
  EXPECT_EQ(t_x3.Init(), 0);
  EXPECT_EQ(tpipe.AddTensor(t_x3), 0);

  codegen::Tensor t_x4(load4->outputs[0], dtype_name, "t_x4");
  EXPECT_EQ(t_x4.Init(), 0);
  EXPECT_EQ(tpipe.AddTensor(t_x4), 0);

  tpipe.AddTensor(rand_int->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x4;
  x4.id = load4->outputs[0].attr.mem.tensor_id;
  codegen::RandIntApiCall call("RandInt");
  EXPECT_EQ(call.Init(rand_int), 0);
  call.tmp_buf_id[-1] = 0;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);
  call.inputs.push_back(&x4);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS);
  std::cout << result << std::endl;

  // seed 标量形态：命中 (int64_t)xxx_ub_scalar 参数，命中设备 kSeedIsScalar=true 重载
  EXPECT_NE(result.find("RandInt("), std::string::npos);
  EXPECT_NE(result.find("_ub_scalar"), std::string::npos);
  EXPECT_NE(result.find("(int64_t)"), std::string::npos);
  // x2 位置流恒张量寻址
  EXPECT_NE(result.find("local_1[0]"), std::string::npos);
  // size 尾参取 y（本块输出元素数），不得再取 seed 的 actual_size
  EXPECT_NE(result.find("local_4_actual_size"), std::string::npos);
  EXPECT_EQ(result.find("local_0_actual_size"), std::string::npos);
}

// 全张量场景（对照）：seed 按向量偏移寻址，无标量变量
TEST(RandIntRegApiCallTest, RandIntApiCall_AllTensor) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildRandIntGraph(graph, s0, s1, z0, z1, false);

  InitLoadNodeAttrs(graph, "load1", z0, z1, ge::DT_INT64, 0);
  InitLoadNodeAttrs(graph, "load2", z0, z1, ge::DT_INT32, 1);
  InitLoadNodeAttrs(graph, "load3", z0, z1, ge::DT_INT64, 2);
  InitLoadNodeAttrs(graph, "load4", z0, z1, ge::DT_INT64, 3);
  InitRandIntNodeAttrs(graph, z0, z1, 4);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto load3 = graph.FindNode("load3");
  auto load4 = graph.FindNode("load4");
  auto rand_int = graph.FindNode("rand_int");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(load3->outputs[0]);
  tpipe.AddTensor(load4->outputs[0]);
  tpipe.AddTensor(rand_int->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x4;
  x4.id = load4->outputs[0].attr.mem.tensor_id;
  codegen::RandIntApiCall call("RandInt");
  EXPECT_EQ(call.Init(rand_int), 0);
  call.tmp_buf_id[-1] = 0;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);
  call.inputs.push_back(&x4);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS);
  std::cout << result << std::endl;

  // 全张量：seed 走向量偏移寻址，无 ub_scalar 标量变量
  EXPECT_NE(result.find("RandInt("), std::string::npos);
  EXPECT_NE(result.find("local_0[0]"), std::string::npos);
  EXPECT_EQ(result.find("_ub_scalar"), std::string::npos);
  EXPECT_NE(result.find("local_4_actual_size"), std::string::npos);
}

// low/high 为编译期常量场景（seed 张量）：常量输入走裸变量名参数（DefineConst 声明的标量变量），
// 不做向量寻址——锁定 (kSeedIsScalar=false, kLowIsScalar=true, kHighIsScalar=true) 重载选择，
// 防止标量槽参数错位在现有全张量用例下不可见
TEST(RandIntRegApiCallTest, RandIntApiCall_LowHighConstant) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildRandIntGraph(graph, s0, s1, z0, z1, false);

  InitLoadNodeAttrs(graph, "load1", z0, z1, ge::DT_INT64, 0);
  InitLoadNodeAttrs(graph, "load2", z0, z1, ge::DT_INT32, 1);
  InitLoadNodeAttrs(graph, "load3", z0, z1, ge::DT_INT64, 2);
  InitLoadNodeAttrs(graph, "load4", z0, z1, ge::DT_INT64, 3);
  InitRandIntNodeAttrs(graph, z0, z1, 4);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto load3 = graph.FindNode("load3");
  auto load4 = graph.FindNode("load4");
  auto rand_int = graph.FindNode("rand_int");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load1->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);

  // x3/x4 为编译期常量：const_value 打印为标量变量参数，不做寻址
  std::string dtype_name;
  codegen::Tensor::DtypeName(load3->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t_x3("0", load3->outputs[0], dtype_name, "t_x3");
  EXPECT_EQ(tpipe.AddTensor(t_x3), 0);
  codegen::Tensor t_x4("24", load4->outputs[0], dtype_name, "t_x4");
  EXPECT_EQ(tpipe.AddTensor(t_x4), 0);
  tpipe.AddTensor(rand_int->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load1->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x2;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x3;
  x3.id = load3->outputs[0].attr.mem.tensor_id;
  codegen::ApiTensor x4;
  x4.id = load4->outputs[0].attr.mem.tensor_id;
  codegen::RandIntApiCall call("RandInt");
  EXPECT_EQ(call.Init(rand_int), 0);
  call.tmp_buf_id[-1] = 0;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.inputs.push_back(&x3);
  call.inputs.push_back(&x4);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS);
  std::cout << result << std::endl;

  EXPECT_NE(result.find("RandInt("), std::string::npos);
  // seed/x2 恒张量寻址
  EXPECT_NE(result.find("local_0[0]"), std::string::npos);
  EXPECT_NE(result.find("local_1[0]"), std::string::npos);
  // 常量输入不得出现寻址括号（出现即说明常量分支退化成张量寻址）
  EXPECT_EQ(result.find("local_2["), std::string::npos);
  EXPECT_EQ(result.find("local_3["), std::string::npos);
  // size 尾参仍取 y 的 actual_size
  EXPECT_NE(result.find("local_4_actual_size"), std::string::npos);
}
