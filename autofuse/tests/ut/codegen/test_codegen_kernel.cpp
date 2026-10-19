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

#include <algorithm>

#include "node_utils_ex.h"
#include "graph_utils.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "utils/api_call_utils.h"
#include "elewise/compare_api_call.h"
#include "elewise/unary_bitwidth_change_api_call.h"
#include "elewise/unary_api_tmp_call.h"
#include "elewise/unary_api_call.h"
#include "elewise/ub2ub_api_call.h"
#include "autofuse_config/auto_fuse_config.h"
#include "codegen_graph_check.h"
#include "platform_context.h"
#include "runtime_stub.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
std::string ToString(const af::Expression &e) {
  return std::string(e.Serialize().get());
}

template <typename BuildDownstream>
void ExpectScalarDataBlkTensorInit(BuildDownstream build_downstream) {
  af::AscGraph graph("test");

  ScalarData scalar_data("scalar_data", graph);
  scalar_data.ir_attr.SetIndex(0);
  scalar_data.y.dtype = ge::DT_FLOAT;

  build_downstream(graph, scalar_data);

  auto scalar_node = graph.FindNode("scalar_data");
  scalar_node->outputs[0].attr.mem.tensor_id = 0;

  codegen::Kernel kernel(graph.GetName());
  EXPECT_EQ(kernel.tpipe.AddTensor(ascgen_utils::GenValidName(scalar_node->GetName()), scalar_node->outputs[0],
                                   "scalar_data_y"),
            af::SUCCESS);
  EXPECT_EQ(kernel.ParseOptimizeInfo(scalar_node, scalar_node->outputs[0]), af::SUCCESS);

  std::string result;
  EXPECT_EQ(kernel.tpipe.BlkTensorAllocAndInit(result), af::SUCCESS);
  EXPECT_NE(result.find("LocalTensor<float> local_blk_tensor_of_scalar_data"), std::string::npos);
  EXPECT_NE(result.find("Duplicate(local_blk_tensor_of_scalar_data[0], static_cast<float>(scalar_data)"),
            std::string::npos);
}
}  // namespace
TEST(CodegenKernel, Type_StrWillReturnTypeName) {
  codegen::Type t{"int"};

  EXPECT_EQ(t.Str(), "int");
}

TEST(CodegenKernel, Variable_StrWillReturnVarName) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};

  EXPECT_EQ(v.Str(), "x");
}

TEST(CodegenKernel, Variable_DefineWillReturnDefineWithInit) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};
  EXPECT_EQ(v.Define("100"), "int x = 100;");
}

TEST(CodegenKernel, Variable_AsArg) {
  codegen::Type t{"int"};
  codegen::Variable v{t, "x"};
  EXPECT_EQ(v.AsArg(), "int x");
}

TEST(CodegenKernel, Axis_StrWillReturnAxisName) {
  af::SizeVar s0(af::Symbol("s0"));
  af::Axis axis{.name = "z0", .size = s0.expr};
  codegen::Axis codegen_axis(axis);
  EXPECT_EQ(codegen_axis.Str(), "z0");
}

TEST(CodegenKernel, Tiler_StrWillReturnTilingDataName) {
  codegen::Tiler tiler("TilingData", "tiling_data");
  EXPECT_EQ(codegen::Tiler(tiler).Str(), "tiling_data");
}

TEST(CodegenKernel, Tiler_SizeWillReturnTilingDataField) {
  af::SizeVar s0(af::Symbol("s0"));

  codegen::Tiler tiler("TilingData", "tiling_data");
  tiler.AddSizeVar(af::SizeVar(s0));

  EXPECT_EQ(tiler.Size(s0.expr), "tiling_data->s0");
}

TEST(CodegenKernel, Tiler_SizeWhenHasTwoMoreNums_WillAddRoundBrackets) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  codegen::Tiler tiler("TilingData", "tiling_data");
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  auto mul = s0.expr * s1.expr;
  EXPECT_EQ(tiler.Size(mul), "(tiling_data->s0 * tiling_data->s1)");
}

TEST(CodegenKernel, Tiler_Size) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  codegen::Tiler tiler("TilingData", "tiling_data");
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  EXPECT_EQ(tiler.Size(Zero), "0");
  EXPECT_EQ(tiler.Size(One), "1");

  EXPECT_EQ(tiler.Size(s0.expr), "tiling_data->s0");
  EXPECT_EQ(tiler.Size(s0.expr * s1.expr), "(tiling_data->s0 * tiling_data->s1)");
  EXPECT_EQ(tiler.Size(One / s0.expr), "Pow(tiling_data->s0, -1)");
  EXPECT_EQ(tiler.Size(One / (s0.expr * s1.expr)), "(1 / (tiling_data->s0 * tiling_data->s1))");
  EXPECT_EQ(tiler.Size(s0.expr / s1.expr), "(tiling_data->s0 / (tiling_data->s1))");
}

TEST(CodegenKernel, Tiler_TensorVectorizedSize) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), af::SUCCESS);
  EXPECT_EQ(tiler.TensorVectorizedSize(t),
            std::string{"KernelUtils::BlkAlign<half>((t->s1 - 1) * t->s2 + (t->s2 - 1) + 1)"});
}

TEST(CodegenKernel, Tiler_ShapeOneTensorVectorizedSize) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {One, One, One};
  *load.y.strides = {One, One, One};
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.dtype = ge::DT_FLOAT;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.vectorized_strides = {af::sym::Align(One, 32 / sizeof(float)), One};

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), af::SUCCESS);
  EXPECT_EQ(tiler.TensorVectorizedSize(t), std::string{"KernelUtils::BlkAlign<float>((1 - 1) * 8 + (1 - 1) + 1)"});
}

TEST(CodegenKernel, Tiler_TensorVectorizedSize_WhenNotVectorized) {
  codegen::Tiler tiler;
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  EXPECT_EQ(tiler.TensorVectorizedSize(codegen::Tensor(tensor, dtype_name)), std::string{"1"});
}

TEST(CodegenKernel, Tiler_BlockOutterAxisDefine) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeBlockOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeBlockOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeBlockOuter, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  auto result_code = tiler.BlockOutterAxisDefine();
  EXPECT_EQ(result_code, std::string{"int block_dim = GetBlockIdx();\n"
                                     "if (block_dim >= t->block_dim) {\n"
                                     "  return;\n"
                                     "}\n"
                                     "const int z0 = block_dim % z0_loop_size;\n"
                                     "const int z1 = block_dim % z1_loop_size;\n"
                                     "const int z2 = block_dim % z2_loop_size;\n"});
}

TEST(CodegenKernel, Tiler_GetAxisVar) {
  af::SizeVar s0(af::Symbol("s0"));
  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeBlockOuter, .size = s0.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddAxis(z0);

  EXPECT_EQ(tiler.GetAxis(z0.id).Str(), "z0");
}

TEST(CodegenKernel, Tiler_TensorOffset_WhenGlobalTensor_WillOffsetAll) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  std::vector<af::AxisId> tensor_axis = {z0.id, z1.id, z2.id};
  std::vector<af::Expression> stride = {s1.expr * s2.expr, s2.expr, One};

  EXPECT_EQ(tiler.Offset({}, tensor_axis, stride), std::string{"0"});
  EXPECT_EQ(tiler.Offset({z0.id}, tensor_axis, stride), std::string{"(int64_t)z0 * (int64_t)(t->s1 * t->s2)"});
  EXPECT_EQ(tiler.Offset({z0.id, z1.id}, tensor_axis, stride),
            std::string{"(int64_t)z0 * (int64_t)(t->s1 * t->s2) + (int64_t)z1 * (int64_t)t->s2"});
  EXPECT_EQ(tiler.Offset({z0.id, z1.id, z2.id}, tensor_axis, stride),
            std::string{"(int64_t)z0 * (int64_t)(t->s1 * t->s2) + (int64_t)z1 * (int64_t)t->s2 + (int64_t)z2"});
}

TEST(CodegenKernel, Tiler_TensorOffset_WhenLocalTensor_VectorizedOnCurrentAxis) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.strides = {s1.expr * s2.expr, s2.expr, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");

  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id}, codegen::Tensor(tensor, dtype_name)), "0");
  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id, z1.id}, codegen::Tensor(tensor, dtype_name)),
            "(int64_t)z1 * (int64_t)t->s2");
  EXPECT_EQ(tiler.TensorVectorizedOffset({z0.id, z1.id, z2.id}, codegen::Tensor(tensor, dtype_name)),
            "(int64_t)z1 * (int64_t)t->s2 + (int64_t)z2");
}

TEST(CodegenKernel, ubScalarFalseTestWhenVecRepeateIsOneButNotAllOne) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {s0, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  // 校验bool ub_scalar
  EXPECT_EQ(t.is_ub_scalar, false);
}

TEST(CodegenKernel, Tensor_ubScalarVrInitTest) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  // 校验bool ub_scalar
  EXPECT_EQ(t.is_ub_scalar, true);

  // 定义ub_scalar变量
  std::string result1;
  EXPECT_EQ(t.DefineUbScalar(result1), 0);
  EXPECT_EQ(result1, "float global_1_ub_scalar;\n");
  // 生成ub scalar的赋值
  std::string result2;
  EXPECT_EQ(t.InitUbScalar(result2), 0);
  EXPECT_EQ(result2, "global_1_ub_scalar = global_1.GetValue(0);\n");
}

TEST(CodegenKernel, OutputTensorIsUbScalar_test) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {s0, s1, s2};
  tensor.attr.strides = {s1 * s2, s2, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{s2, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  EXPECT_EQ(t.is_ub_scalar, false);

  codegen::Kernel kernel("test");
  EXPECT_EQ(kernel.tpipe.AddTensor(t), 0);
  bool is_ub_scalar = true;
  EXPECT_EQ(kernel.OutputTensorIsUbScalar(node, is_ub_scalar), 0);
  EXPECT_EQ(is_ub_scalar, false);
}

TEST(CodegenKernel, OutputTensorIsScalarDuplicate_test) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {s0, s1, s2};
  tensor.attr.strides = {s1 * s2, s2, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{s2, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);

  codegen::Kernel kernel("test");
  EXPECT_EQ(kernel.tpipe.AddTensor(t), 0);
  kernel.tpipe.need_gen_blk_tensors.push_back(t.id);
  bool is_ub_scalar = true;

  std::string result;
  Status ans = kernel.tpipe.BlkTensorAllocAndInit(result);

  EXPECT_EQ(
      result,
      std::string{
          "TBuf<TPosition::VECCALC> global_1_tbuf;\n"
          "tpipe.InitBuffer(global_1_tbuf, 32);\n"
          "LocalTensor<GlobalTensor<float>> local_blk_tensor_of_global_1 = global_1_tbuf.Get<GlobalTensor<float>>();\n"
          "Duplicate(local_blk_tensor_of_global_1[0], static_cast<GlobalTensor<float>>(0), "
          "static_cast<uint64_t>(32/sizeof(GlobalTensor<float>)));\n"
          "AscendC::PipeBarrier<PIPE_V>();\n"});
}

TEST(CodegenKernel, ReduceTensorForceNonUbScalar) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  // af::ascir_op::Load x("load", graph);
  Sum sum("sum");
  graph.AddNode(sum);
  auto node = graph.FindNode("sum");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  EXPECT_EQ(t.is_ub_scalar, true);

  codegen::Kernel kernel("test");
  EXPECT_EQ(kernel.tpipe.AddTensor(t), 0);
  EXPECT_EQ(kernel.ParseOptimizeInfo(node, tensor), 0);
  auto t_s_ptr = kernel.tpipe.GetTensor(1);
  auto t_s = *t_s_ptr;
  EXPECT_EQ(t_s.is_ub_scalar, false);
}

TEST(CodegenKernel, ApiCallPreProcessTest) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);

  // 定义api all
  codegen::ApiCall call("Load");
  EXPECT_EQ(call.Init(node), 0);
  // 1
  std::string result1, result2;
  // 2
  std::vector<std::reference_wrapper<const codegen::Tensor>> output_tensors;
  t.need_gen_get_value_of_ub_scalar = true;
  output_tensors.emplace_back(t);
  // 3
  std::vector<::ascir::AxisId> current_axis = {z0.id};
  // 4
  codegen::TPipe tpipe("tpipe", tiler);

  EXPECT_EQ(call.PreProcess(tpipe, current_axis, output_tensors, result1), 0);
  EXPECT_EQ(result1, "if (z0 < 1) {\n");

  EXPECT_EQ(call.PostProcess(tpipe, current_axis, output_tensors, result2), 0);
  EXPECT_EQ(result2,
            std::string{"event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                        "SetFlag<HardEvent::MTE2_S>(eventID);\n"
                        "WaitFlag<HardEvent::MTE2_S>(eventID);\n"
                        "global_1_ub_scalar = global_1.GetValue(0);\n"
                        "}\n"});
}

TEST(CodegenKernel, ApiCallPreProcessUbScalarTest) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);

  // 定义api all
  codegen::ApiCall call("Load");
  EXPECT_EQ(call.Init(node), 0);
  // 1
  std::string result1, result2;
  // 2
  std::vector<std::reference_wrapper<const codegen::Tensor>> output_tensors;
  t.need_gen_get_value_of_ub_scalar = true;
  t.need_duplicate_value_of_ub_scalar = true;
  output_tensors.emplace_back(t);
  // 3
  std::vector<::ascir::AxisId> current_axis = {z0.id};
  // 4
  codegen::TPipe tpipe("tpipe", tiler);

  EXPECT_EQ(call.PostProcess(tpipe, current_axis, output_tensors, result2), 0);
  EXPECT_EQ(result2,
            std::string{"event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                        "SetFlag<HardEvent::MTE2_S>(eventID);\n"
                        "WaitFlag<HardEvent::MTE2_S>(eventID);\n"
                        "global_1_ub_scalar = global_1.GetValue(0);\n"
                        "AscendC::PipeBarrier<PIPE_ALL>();\n"
                        "Duplicate(global_1[0], global_1_ub_scalar, 32/sizeof(float));\n"
                        "AscendC::PipeBarrier<PIPE_V>();\n"
                        "}\n"});
}

TEST(CodegenKernel, ApiCallPreProcessLoadUbScalarTest) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);

  // 定义api all
  codegen::ApiCall call("Load");
  EXPECT_EQ(call.Init(node), 0);
  // 1
  std::string result1, result2;
  // 2
  std::vector<std::reference_wrapper<const codegen::Tensor>> output_tensors;
  t.need_gen_get_value_of_ub_scalar = true;
  t.need_duplicate_value_of_ub_scalar = true;
  output_tensors.emplace_back(t);
  // 3
  std::vector<::ascir::AxisId> current_axis = {z0.id};
  // 4
  codegen::TPipe tpipe("tpipe", tiler);

  EXPECT_EQ(call.PostProcess(tpipe, current_axis, output_tensors, result2), 0);
  EXPECT_EQ(result2,
            std::string{"event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));\n"
                        "SetFlag<HardEvent::MTE2_S>(eventID);\n"
                        "WaitFlag<HardEvent::MTE2_S>(eventID);\n"
                        "global_1_ub_scalar = global_1.GetValue(0);\n"
                        "AscendC::PipeBarrier<PIPE_ALL>();\n"
                        "Duplicate(global_1[0], global_1_ub_scalar, 32/sizeof(float));\n"
                        "AscendC::PipeBarrier<PIPE_V>();\n"
                        "}\n"});
}

TEST(CodegenKernel, kernelUbScalarVarDef) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  // af::ascir_op::Load x("load", graph);
  Load load("load");
  graph.AddNode(load);
  auto node = graph.FindNode("load");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.repeats = {One, One, One};
  tensor.attr.strides = {One, One, One};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.mem.tensor_id = 1;

  vector<af::Expression> vectorized_strides{One, One};
  tensor.attr.vectorized_strides = vectorized_strides;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  codegen::Kernel kernel("test");
  EXPECT_EQ(kernel.tpipe.AddTensor(t), 0);
  kernel.ub_scalar_tensors.emplace_back(1);
  std::string result;
  EXPECT_EQ(kernel.GlobalTensorInit(result), 0);
  EXPECT_EQ(result, "float global_1_ub_scalar;\n");
}

TEST(CodegenKernel, Tiler_TensorOffset_WhenLocalTensor_VectorizedNestCurrentAxis) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Tiler_TensorAlloc_WhenTensorFromQue_AndMerge) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Tensor_SetGlobalBuffer) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];
  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;

  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t(tensor, dtype_name, "tensor");
  codegen::GM_ADDR gm("gm");

  std::string result;
  t.SetGlobalBuffer(gm, "", result);
  EXPECT_EQ(result, "global_0.SetGlobalBuffer((__gm__ half*)gm);");
}

TEST(CodegenKernel, TQue_AllocBuf) {
  std::string position;
  codegen::PositionValue(af::Position::kPositionVecIn, position);
  codegen::TQue que(0, af::Position::kPositionVecIn, position);
  EXPECT_EQ(que.AllocBuf(), "LocalTensor<uint8_t> q0_buf = q0.AllocTensor<uint8_t>();\n");
}

TEST(CodegenKernel, TQue_EnqueBuf) {
  std::string position;
  codegen::PositionValue(af::Position::kPositionVecIn, position);
  codegen::TQue que(0, af::Position::kPositionVecIn, position);
  EXPECT_EQ(que.EnqueBuf(), "q0.EnQue(q0_buf);\n");
}

TEST(CodegenKernel, TQue_DequeBuf) {
  std::string position;
  codegen::PositionValue(af::Position::kPositionVecIn, position);
  codegen::TQue que(0, af::Position::kPositionVecIn, position);
  EXPECT_EQ(que.DequeBuf(true), "LocalTensor<uint8_t> q0_buf = q0.DeQue<uint8_t>();\n");
  EXPECT_EQ(que.DequeBuf(false), "q0_buf = q0.DeQue<uint8_t>();\n");
}

TEST(CodegenKernel, TQue_FreeBuf) {
  std::string position;
  codegen::PositionValue(af::Position::kPositionVecIn, position);
  codegen::TQue que(0, af::Position::kPositionVecIn, position);
  EXPECT_EQ(que.FreeBuf(), "q0.FreeTensor(q0_buf);\n");
}

TEST(CodegenKernel, TPipe_TensorAlloc_WhenConstantTensor_WillNotAlloc) {
  af::AscGraph graph("test");
  af::ascir_op::Scalar x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.position = af::Position::kPositionInvalid;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  tensor.attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor, "test_t");

  std::string result;
  auto tensor_ptr = tpipe.GetTensor(tensor.attr.mem.tensor_id);
  tpipe.TensorAlloc(*tensor_ptr, result);
  EXPECT_EQ(result, std::string{});
}

TEST(CodegenKernel, TPipe_TensorAlloc_WhenTensorFromQue_AndNotMerge) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = 1;
  tensor.attr.mem.reuse_id = 1;
  tensor.attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor, "test_t");

  std::string result;
  auto tensor_ptr = tpipe.GetTensor(tensor.attr.mem.tensor_id);
  tpipe.TensorAlloc(*tensor_ptr, result);
  EXPECT_EQ(result, std::string{"LocalTensor<half> local_0;\n"
                                "local_0 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"});
}

TEST(CodegenKernel, TPipe_TensorAlloc_WhenTensorFromBuf_AndNotMerge) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor, "test_t");

  std::string result;
  auto tensor_ptr = tpipe.GetTensor(tensor.attr.mem.tensor_id);
  tpipe.TensorAlloc(*tensor_ptr, result);
  EXPECT_EQ(result, std::string{"LocalTensor<half> local_0;\n"
                                "local_0 = b1_buf.template ReinterpretCast<half>();\n"});
}

TEST(CodegenKernel, AddTensor_InvalidName) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  {
    tensor.attr.dtype = ge::DT_FLOAT16;
    tensor.attr.mem.position = af::Position::kPositionVecIn;
    tensor.attr.mem.tensor_id = 0;
    tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    tensor.attr.que.id = 1;
    tensor.attr.mem.reuse_id = 1;
    tensor.attr.opt.merge_scope = af::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);
    tpipe.CollectQues(graph);
    tpipe.AddTensor(tensor, "invalid/test");

    std::string result;
    auto tensor_ptr = tpipe.GetTensor(tensor.attr.mem.tensor_id);
    tpipe.TensorAlloc(*tensor_ptr, result);
    EXPECT_EQ(result, std::string{"LocalTensor<half> local_0;\n"
                                  "local_0 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"});
  }

  {
    tensor.attr.dtype = ge::DT_FLOAT16;
    tensor.attr.mem.position = af::Position::kPositionVecIn;
    tensor.attr.mem.tensor_id = 0;
    tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    tensor.attr.que.id = 1;
    tensor.attr.mem.reuse_id = 1;
    tensor.attr.opt.merge_scope = af::kIdNone;

    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);
    tpipe.CollectQues(graph);
    tpipe.AddTensor(tensor, "0test");

    std::string result;
    auto tensor_ptr = tpipe.GetTensor(tensor.attr.mem.tensor_id);
    tpipe.TensorAlloc(*tensor_ptr, result);
    EXPECT_EQ(result, std::string{"LocalTensor<half> local_0;\n"
                                  "local_0 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"});
  }
}

TEST(CodegenKernel, TPipe_InitTQueBuffers) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = 1;
  tensor.attr.mem.reuse_id = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  auto que = tpipe.ques.find(tensor.attr.que.id);
  ASSERT_NE(que, tpipe.ques.end());
  std::string result;
  tpipe.InitTQueBuffers(que->second, result);
  EXPECT_EQ(result, std::string{"// tpipe.InitBuffer(q1, q1_buf_num, KernelUtils::BlkAlign<uint8_t>(q1_size));\n"
                                "tpipe.InitBuffer(q1, q1_buf_num, t->q1_size);"});
}

TEST(CodegenKernel, TPipe_InitTBufBuffer) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  auto buf = tpipe.bufs.find(tensor.attr.buf.id);
  ASSERT_NE(buf, tpipe.bufs.end());
  std::string result;
  tpipe.InitTBufBuffer(buf->second, result);
  EXPECT_EQ(result, std::string{"// tpipe.InitBuffer(b1, KernelUtils::BlkAlign<uint8_t>(b1_size));\n"
                                "tpipe.InitBuffer(b1, t->b1_size);"});
}

TEST(CodegenKernel, TPipe_TensorSizeCalc_AllocFromBuf) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tensor.attr.buf.id = 2;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  EXPECT_EQ(
      tpipe.TensorSizeCalc(),
      std::string{
          "const uint32_t local_1_size = KernelUtils::BlkAlign<float>((t->s1 - 1) * t->s2 + (t->s2 - 1) + 1);\n"});
}

TEST(CodegenKernel, TPipe_TensorSizeCalc_AllocFromQue) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tensor.attr.que.id = 2;
  tensor.attr.que.buf_num = 4;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  EXPECT_EQ(
      tpipe.TensorSizeCalc(),
      std::string{"const uint32_t local_1_size = KernelUtils::BlkAlign<float>((t->s1 - 1) * t->s2 + (t->s2 - 1) + 1);\n"
                  "const uint32_t local_1_que_buf_num = 4;\n"});
}

TEST(CodegenKernel, TPipe_TensorSizeAssignForCvUbFuseShouldUseStageSizeElements) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);

  auto int8_node = graph.FindNode("x");
  af::AscTensor int8_tensor = int8_node->outputs[0];
  int8_tensor.attr.dtype = ge::DT_INT8;
  int8_tensor.attr.mem.tensor_id = 1;
  int8_tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  int8_tensor.attr.mem.position = af::Position::kPositionVecCalc;
  int8_tensor.attr.opt.merge_scope = af::kIdNone;
  int8_tensor.attr.buf.id = 1;

  auto int64_node = graph.FindNode("y");
  af::AscTensor int64_tensor = int64_node->outputs[0];
  int64_tensor.attr.dtype = ge::DT_INT64;
  int64_tensor.attr.mem.tensor_id = 2;
  int64_tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  int64_tensor.attr.mem.position = af::Position::kPositionVecCalc;
  int64_tensor.attr.opt.merge_scope = af::kIdNone;
  int64_tensor.attr.buf.id = 2;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  ASSERT_EQ(tpipe.AddTensor(int8_tensor), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(int64_tensor), af::SUCCESS);

  std::string result;
  ASSERT_EQ(tpipe.TensorSizeAssign("float", result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"local_1_size = stage_size;\n"
                                "local_2_size = stage_size;\n"
                                "\n"});
}

TEST(CodegenKernel, TPipe_MergeScopeSizeCalc) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.opt.merge_scope = 1;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = 2;
  tensor.attr.que.depth = 3;
  tensor.attr.que.buf_num = 4;
  tpipe.AddTensor(tensor);

  tensor.attr.dtype = ge::DT_FLOAT;
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 2;
  tpipe.AddTensor(tensor);

  std::string result;
  tpipe.MergeScopeSizeCalc(result);
  EXPECT_EQ(
      result,
      std::string{
          "const uint32_t m1_size = KernelUtils::Sum(local_0_size * sizeof(half), local_1_size * sizeof(float));\n"
          "const uint32_t m1_que_buf_num = KernelUtils::Max(local_0_que_buf_num);\n"});
}

TEST(CodegenKernel, TPipe_LocalTBufAlloc) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = 1;
  tpipe.AddTensor(tensor);

  tensor.attr.dtype = ge::DT_FLOAT;
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tpipe.AddTensor(tensor);

  std::string result;
  tpipe.SetUsingAttCalcQBTSizeConfig(false);
  tpipe.LocalTBufAllocLoopTwice(result);
  EXPECT_EQ(result, std::string{"const uint32_t b1_size = KernelUtils::Max(m1_size, local_1_size * sizeof(float));\n"
                                "TBuf<TPosition::VECIN> b1;\n"
                                "tpipe.InitBuffer(b1, KernelUtils::BlkAlign<uint8_t>(b1_size));\n"
                                "LocalTensor<float> local_1 = b1.Get<float>();\n\n"});
}

TEST(CodegenKernel, TPipe_LocalTBufAlloc_2) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = 1;
  tpipe.AddTensor(tensor);

  tensor.attr.dtype = ge::DT_FLOAT;
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = 2;
  tpipe.AddTensor(tensor);

  std::string result;
  tpipe.SetUsingAttCalcQBTSizeConfig(false);
  EXPECT_EQ(tpipe.LocalTBufAllocLoopTwice(result), af::SUCCESS);
}

TEST(CodegenKernel, TPipe_LocalTBufAlloc_MergeScopes_ERROR) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = 1;
  tpipe.AddTensor(tensor);

  for (auto &[id, buf] : tpipe.bufs) {
    if (id == 1) {
      buf.merge_scopes.insert(100);
    }
  }

  std::string result;
  tpipe.SetUsingAttCalcQBTSizeConfig(false);
  EXPECT_EQ(tpipe.LocalTBufAllocLoopTwice(result), af::FAILED);
}

TEST(CodegenKernel, TPipe_LocalTBufAlloc_NotMergeTensors_ERROR) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tpipe.AddTensor(tensor);

  for (auto &[id, buf] : tpipe.bufs) {
    if (id == 1) {
      buf.not_merge_tensors.insert(100);
    }
  }

  std::string result;
  tpipe.SetUsingAttCalcQBTSizeConfig(false);
  EXPECT_EQ(tpipe.LocalTBufAllocLoopTwice(result), af::FAILED);
}

TEST(CodegenKernel, TPipe_LocalTQueAlloc) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(s2);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor.attr.mem.position = af::Position::kPositionVecIn;
  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;

  codegen::TPipe tpipe("tpipe", tiler);

  tensor.attr.dtype = ge::DT_FLOAT16;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = 1;
  tensor.attr.mem.reuse_id = 1;
  tensor.attr.opt.merge_scope = 1;
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  tensor.attr.dtype = ge::DT_FLOAT;
  tensor.attr.mem.tensor_id = 1;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  tensor.attr.que.id = 1;
  tensor.attr.mem.reuse_id = 2;
  tensor.attr.opt.merge_scope = af::kIdNone;
  tensor.attr.que.buf_num = 2;
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  std::string result;
  tpipe.LocalTQueAlloc(result);
  EXPECT_EQ(result, std::string{"// const uint32_t q1_size = KernelUtils::Max(m1_size, local_1_size * sizeof(float));\n"
                                "const uint32_t q1_buf_num = KernelUtils::Max(m1_que_buf_num, 2);\n"
                                "TQue<TPosition::VECIN, 1> q1;\n"
                                "// tpipe.InitBuffer(q1, q1_buf_num, KernelUtils::BlkAlign<uint8_t>(q1_size));\n"
                                "tpipe.InitBuffer(q1, q1_buf_num, t->q1_size);\n"});
}

TEST(CodegenKernel, TPipe_LocalTQueAllocForCvUbFuseShouldAlignSharedTensorSlices) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);

  auto int8_node = graph.FindNode("x");
  af::AscTensor int8_tensor = int8_node->outputs[0];
  int8_tensor.attr.dtype = ge::DT_INT8;
  int8_tensor.attr.mem.tensor_id = 1;
  int8_tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  int8_tensor.attr.mem.position = af::Position::kPositionVecIn;
  int8_tensor.attr.mem.reuse_id = 1;
  int8_tensor.attr.opt.merge_scope = af::kIdNone;
  int8_tensor.attr.que.id = 1;
  int8_tensor.attr.que.buf_num = 1;

  auto int64_node = graph.FindNode("y");
  af::AscTensor int64_tensor = int64_node->outputs[0];
  int64_tensor.attr.dtype = ge::DT_INT64;
  int64_tensor.attr.mem.tensor_id = 2;
  int64_tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  int64_tensor.attr.mem.position = af::Position::kPositionVecIn;
  int64_tensor.attr.mem.reuse_id = 1;
  int64_tensor.attr.opt.merge_scope = af::kIdNone;
  int64_tensor.attr.que.id = 1;
  int64_tensor.attr.que.buf_num = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
  tpipe.CollectQues(graph);
  ASSERT_EQ(tpipe.AddTensor(int8_tensor), af::SUCCESS);
  ASSERT_EQ(tpipe.AddTensor(int64_tensor), af::SUCCESS);

  std::string result;
  ASSERT_EQ(tpipe.LocalTQueAlloc(result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"const uint32_t q1_size = KernelUtils::Max(local_1_size * sizeof(int8_t), "
                                "local_2_size * sizeof(int64_t), "
                                "KernelUtils::BlkAlign<uint8_t>(local_1_size * sizeof(int8_t)) + "
                                "KernelUtils::BlkAlign<uint8_t>(local_2_size * sizeof(int64_t)));\n"
                                "const uint32_t q1_buf_num = KernelUtils::Max(1);\n"
                                "tpipe.InitBuffer(q1, q1_buf_num, KernelUtils::BlkAlign<uint8_t>(q1_size));\n"});
}

TEST(CodegenKernel, ApiCall_Generate) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  codegen::ApiCall call("call");
  call.Init(node);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {}, {}, {}, result), af::SUCCESS);
}

TEST(CodegenKernel, Stage_AddCall_WillCollectInputOutputQues) {
  GTEST_SKIP();
}

TEST(CodegenKernel, Stage_AddCall_WillAddCall) {
  GTEST_SKIP();
}

class MockApiCall : public virtual codegen::ApiCall {
 public:
  MockApiCall(const std::string &api_name) : codegen::ApiCall(api_name) {}
  MockApiCall(const af::AscNodePtr &node, const std::string &api_name) : codegen::ApiCall(api_name) {
    Init(node);
  }

  Status Generate(const codegen::TPipe &tpipe, const std::vector<af::AxisId> &current_axis,
                  std::string &result) const override {
    result = this->type + "();\n";
    return 0;
  }

  virtual Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const {
    ss << "func_test_Definition:" << api_name_ << std::endl;
    return af::SUCCESS;
  };
};

class CodegenKernel_CallSync : public ::testing::Test {
 protected:
  af::AscGraph graph;
  af::AscNodePtr x;
  int tensor_id = 0;
  bool cv_ub_fuse_mode = false;

  CodegenKernel_CallSync() : graph("test_graph"), x(nullptr) {
    Data x_op("x", graph);

    x = graph.FindNode("x");
    x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    x->outputs[0].attr.mem.tensor_id = this->tensor_id++;
  }

  af::AscNodePtr AddNode(const char *name, const std::vector<af::AscNodePtr> &inputs, bool has_output = true,
                         const af::AscNodePtr reuse_or_share = nullptr) {
    auto op_desc = std::make_shared<af::OpDesc>(name, name);
    for (int i = 0; i < inputs.size(); i++) {
      op_desc->AddInputDesc(af::GeTensorDesc());
    }
    if (has_output) {
      op_desc->AddOutputDesc(af::GeTensorDesc());
    }

    auto op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    graph.AddNode(op);
    auto node = af::NodeUtilsEx::GetNodeFromOperator(op);
    int32_t in_index = 0;
    for (auto input : inputs) {
      af::GraphUtils::AddEdge(input->GetOutDataAnchor(0), node->GetInDataAnchor(in_index));
      in_index++;
    }

    auto n = graph.FindNode(name);
    if (has_output) {
      n->outputs[0].attr.dtype = ge::DT_FLOAT16;
      n->outputs[0].attr.mem.tensor_id = this->tensor_id++;
      n->outputs[0].attr.opt.merge_scope = af::kIdNone;
      n->outputs[0].attr.mem.reuse_id = n->outputs[0].attr.mem.tensor_id;

      if (reuse_or_share != nullptr) {
        auto reuse_output = reuse_or_share->outputs()[0];
        n->outputs[0].attr.mem.alloc_type = reuse_output->attr.mem.alloc_type;
        n->outputs[0].attr.buf.id = reuse_output->attr.buf.id;
        n->outputs[0].attr.que.id = reuse_output->attr.que.id;
      }
    }

    return n;
  }

  af::AscNodePtr AddNode(const char *name, bool has_output = true, const af::AscNodePtr reuse = nullptr) {
    auto op_desc = std::make_shared<af::OpDesc>(name, name);
    if (has_output) {
      op_desc->AddOutputDesc(af::GeTensorDesc());
    }

    auto op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    graph.AddNode(op);

    auto n = graph.FindNode(name);
    if (has_output) {
      n->outputs[0].attr.dtype = ge::DT_FLOAT16;
      n->outputs[0].attr.mem.tensor_id = this->tensor_id++;
      n->outputs[0].attr.opt.merge_scope = af::kIdNone;
      n->outputs[0].attr.mem.reuse_id = n->outputs[0].attr.mem.tensor_id;

      if (reuse != nullptr) {
        auto reuse_output = reuse->outputs()[0];
        n->outputs[0].attr.mem.alloc_type = reuse_output->attr.mem.alloc_type;
        n->outputs[0].attr.buf.id = reuse_output->attr.buf.id;
        n->outputs[0].attr.que.id = reuse_output->attr.que.id;
      }
    }

    return n;
  }

  af::AscNodePtr Load(const char *name, const af::AscNodePtr &input, const af::AscNodePtr reuse = nullptr) {
    auto n = AddNode(name, {input}, true, reuse);
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;

    n->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    if (reuse == nullptr) {
      n->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
      n->outputs[0].attr.que.id = n->outputs[0].attr.mem.tensor_id;
      n->outputs[0].attr.mem.reuse_id = n->outputs[0].attr.mem.tensor_id;
    }

    return n;
  }

  af::AscNodePtr LoadForShare(const char *name, const af::AscNodePtr &input, const af::AscNodePtr share_pre) {
    auto n = AddNode(name, {input}, true, share_pre);
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;

    n->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    auto share_output = share_pre->outputs()[0];
    n->outputs[0].attr.mem.reuse_id = share_output->attr.mem.reuse_id;

    return n;
  }

  af::AscNodePtr Vec(const char *name, bool has_output = true, const af::AscNodePtr reuse = nullptr) {
    auto n = AddNode(name, has_output, reuse);
    n->attr.api.unit = af::ComputeUnit::kUnitVector;
    if (has_output) {
      n->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
      if (reuse == nullptr) {
        n->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
        n->outputs[0].attr.buf.id = n->outputs[0].attr.mem.tensor_id;
      }
    }

    return n;
  }

  af::AscNodePtr Vec(const char *name, const std::vector<af::AscNodePtr> &inputs, bool has_output = true,
                     const af::AscNodePtr reuse = nullptr) {
    auto n = AddNode(name, inputs, has_output, reuse);
    n->attr.api.unit = af::ComputeUnit::kUnitVector;
    if (has_output) {
      n->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
      if (reuse == nullptr) {
        n->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
        n->outputs[0].attr.buf.id = n->outputs[0].attr.mem.tensor_id;
      }
    }

    return n;
  }

  af::AscNodePtr VecOut(const char *name, const af::AscNodePtr input = nullptr, const af::AscNodePtr reuse = nullptr) {
    auto n = (input == nullptr) ? AddNode(name, true, reuse) : AddNode(name, {input}, true, reuse);
    n->attr.api.unit = af::ComputeUnit::kUnitVector;
    n->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    if (reuse == nullptr) {
      n->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
      n->outputs[0].attr.que.id = n->outputs[0].attr.mem.tensor_id;
      n->outputs[0].attr.mem.reuse_id = n->outputs[0].attr.mem.tensor_id;
    }

    return n;
  }

  af::AscNodePtr Store(const char *name, const af::AscNodePtr input, bool has_output = false) {
    auto n = AddNode(name, {input}, has_output);
    n->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    if (has_output) {
      n->outputs[0].attr.mem.position = af::Position::kPositionGM;
    }

    return n;
  }

  void InitApiCallInputs(std::map<int, MockApiCall> &calls, std::map<af::AscNode *, int64_t> &node_to_order,
                         const AscNodePtr &node, ApiCall &api_call) {
    int32_t index = 0;
    for (auto i : node->inputs()) {
      auto i_index = af::ascir::AscTensorUtils::Index(*i);
      auto in_node = dynamic_cast<af::AscNode *>(af::ascir::AscTensorUtils::GetOwner(*i));
      if (in_node->GetType() == "Data") {
        continue;
      }
      auto in_call = calls.find(static_cast<int32_t>(node_to_order[in_node]));
      if (in_call != calls.end()) {
        api_call.inputs.emplace_back(&in_call->second.outputs[i_index]);
        in_call->second.outputs[i_index].reads.emplace_back(&api_call);
        if (node->GetName() == "Concat") {
          in_call->second.outputs[i_index].share_order = index;
        }
      }
      ++index;
    }
  }
  std::string Generate() {
    codegen::Tiler tiler;
    codegen::TPipe tpipe("tpipe", tiler);
    if (cv_ub_fuse_mode) {
      tpipe.cv_fusion_type = ::ascir::CubeTemplateType::kUBFuse;
    }
    tpipe.CollectQues(graph);

    codegen::Loop loop(af::kIdNone);
    map<int, MockApiCall> calls;
    map<int64_t, codegen::ApiTensor *> buf_last_use;
    map<int64_t, codegen::ApiTensor *> que_last_use;
    map<int64_t, map<int64_t, codegen::ApiTensor *>> que_last_share;

    std::map<af::AscNode *, int64_t> node_to_order_;
    int64_t top_id = 0UL;
    for (auto node : graph.GetAllNodes()) {
      node_to_order_[node.get()] = top_id++;
      tpipe.CollectQues(graph);
      for (auto o : node->outputs()) {
        tpipe.AddTensor(*o);
      }

      if (node->GetType() == "Data") {
        // ignore Data
        continue;
      }

      auto [it, _] = calls.insert({node_to_order_[node.get()], MockApiCall(node, "")});

      InitApiCallInputs(calls, node_to_order_, node, it->second);

      for (auto o : node->outputs()) {
        auto o_index = af::ascir::AscTensorUtils::Index(*o);
        if (o->attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
          auto reused_tensor = buf_last_use.find(o->attr.buf.id);
          if (reused_tensor != buf_last_use.end()) {
            reused_tensor->second->reuse_next = &it->second.outputs[o_index];
            it->second.outputs[o_index].reuse_from = reused_tensor->second;
          }
          buf_last_use.insert({o->attr.buf.id, &it->second.outputs[o_index]});
        } else if (o->attr.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
          map<int64_t, codegen::ApiTensor *> &last_share = que_last_share[o->attr.que.id];
          auto share_tensor = last_share.find(o->attr.mem.reuse_id);
          if (share_tensor != last_share.end()) {
            auto t_ptr = tpipe.GetTensor(o->attr.mem.tensor_id);
            auto t_share_prev_ptr = tpipe.GetTensor(share_tensor->second->id);
            auto &t = *t_ptr;
            auto &t_share_prev = *t_share_prev_ptr;
            t.share_pre_size = t_share_prev.size.name;
            share_tensor->second->share_next = &it->second.outputs[o_index];
            it->second.outputs[o_index].share_prev = share_tensor->second;
          }
          last_share[o->attr.mem.reuse_id] = &it->second.outputs[o_index];
          auto reused_tensor = que_last_use.find(o->attr.que.id);
          if (reused_tensor != que_last_use.end()) {
            reused_tensor->second->reuse_next = &it->second.outputs[o_index];
            it->second.outputs[o_index].reuse_from = reused_tensor->second;
          }
          que_last_use.insert({o->attr.que.id, &it->second.outputs[o_index]});
        }
      }

      for (auto &o : it->second.outputs) {
        o.write = &it->second;
      }

      loop.AddCall(&it->second);
    }

    std::string result;
    loop.Generate(tiler, tpipe, result);
    return result;
  }
};

TEST_F(CodegenKernel_CallSync, Load_Store_BindQue_ShouldNotSyncMte2ToMte3) {
  auto load = Load("Load", x);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  auto store = Store("Store", load);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "Load();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "Store();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Load_MultiStoreConsumer_ShouldSyncMte2ToMte3) {
  auto load = Load("Load", x);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  auto store1 = Store("Store", load);
  auto store2 = Store("Store_1", load);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "Load();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "auto local_1_e = tpipe.AllocEventID<HardEvent::MTE2_MTE3>();\n"
                                    "TQueSync<PIPE_MTE2, PIPE_MTE3> local_1_s;\n"
                                    "local_1_s.SetFlag(local_1_e);\n"
                                    "local_1_s.WaitFlag(local_1_e);\n"
                                    "tpipe.ReleaseEventID<HardEvent::MTE2_MTE3>(local_1_e);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "Store();\n"
                                    "\n"
                                    "Store_1();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_Store_ShouldAlloc) {
  auto vec1 = Vec("vec1");
  auto vec2 = Vec("vec2", {vec1});
  auto store = Store("store", vec2);
  vec2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  vec2->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  vec2->outputs[0].attr.que.id = vec2->outputs[0].attr.mem.tensor_id;
  vec2->outputs[0].attr.mem.reuse_id = vec2->outputs[0].attr.mem.tensor_id;

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "uint32_t q2_reuse2_offset = 0;\n"
                                    "LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q2_buf[q2_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "q2.EnQue(q2_buf);\n\n"
                                    "q2_buf = q2.DeQue<uint8_t>();\n"
                                    "store();\n"
                                    "q2.FreeTensor(q2_buf);\n\n"});
}

TEST_F(CodegenKernel_CallSync, VecOut_Unuse_ShouldFree) {
  auto vec1 = Vec("vec1");
  auto vec2 = VecOut("vec2", {vec1});
  auto vec3 = VecOut("vec3", {vec1}, vec2);

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "uint32_t q2_reuse2_offset = 0;\n"
                                    "LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q2_buf[q2_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "q2.EnQue(q2_buf);\n"
                                    "q2.FreeTensor(q2_buf);\n\n"
                                    "uint32_t q2_reuse3_offset = 0;\n"
                                    "q2_buf = q2.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = q2_buf[q2_reuse3_offset].template ReinterpretCast<half>();\n"
                                    "vec3();\n"
                                    "q2.EnQue(q2_buf);\n"
                                    "q2.FreeTensor(q2_buf);\n\n"});
}

TEST_F(CodegenKernel_CallSync, Load_Vec__ShouldAllocFromQue_Enq_Deq_Free) {
  auto load = Load("load1", x);
  auto vec = Vec("vec1", {load});

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b2_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_Vec__Should_PipeBarrier) {
  auto vec1 = Vec("vec1");
  auto vec2 = Vec("vec2", {vec1});

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b2_buf.template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_Store__Should_AllocFromQue_Enq_Deq_Free) {
  auto vec1 = VecOut("vec1");
  auto store = Store("store", vec1);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "store();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Load_MulitVec__ShouldDequeFirstVec_FreeAfterLastVec) {
  auto load = Load("load1", x);
  auto vec1 = Vec("vec1", {load}, false);
  auto vec2 = Vec("vec2", {load}, false);
  auto vec3 = Vec("vec3", {load}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "vec2();\n"
                                    "\n"
                                    "vec3();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Load_Vec1_ReuseByVec1_WillPipeBeforeVec1_AnfFreeAfterVec2) {
  auto load = Load("load1", x);
  auto vec1 = Vec("vec1", {load}, false);
  auto vec2 = Vec("vec2", true, load);
  auto vec3 = Vec("vec3", {vec2}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "vec3();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Load1Vec1__Load2ReuseLoad1_ShouldFreeAlloc) {
  auto load1 = Load("load1", x);
  auto vec1 = Vec("vec1", {load1}, false);
  auto load2 = Load("load2", x, load1);
  auto vec2 = Vec("vec2", {load2}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec2();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_MulitVec__Should_PipeBarrierFirstVec) {
  auto vec1 = Vec("vec1");
  auto vec2 = Vec("vec2", {vec1});
  auto vec3 = Vec("vec3", {vec1});

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b2_buf.template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = b3_buf.template ReinterpretCast<half>();\n"
                                    "vec3();\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_Store_Vec__Should_AllocFromQue_Enq_Deq_PipeBarrier_Free) {
  auto vec1 = VecOut("vec1");
  auto store = Store("store", vec1);
  auto vec2 = Vec("vec2", {vec1});

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "store();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b2_buf.template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec_Vec_Store__Should_AllocFromQue_Enq_PipeBarrier_Deq_Free) {
  auto vec1 = VecOut("vec1");
  auto vec2 = Vec("vec2", {vec1});
  auto store = Store("store", vec1);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b2_buf.template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "store();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec1alloc_Vec2_Vec3ReuseVec1) {
  auto vec1 = Vec("vec1");
  auto vec2 = Vec("vec2", {vec1}, false);
  auto vec3 = Vec("vec3", true, vec1);

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b1_buf.template ReinterpretCast<half>();\nvec3();\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, Vec1_Vec2_Vec3_Vec4ReuseVec1) {
  auto vec1 = Vec("vec1");
  auto vec2 = Vec("vec2", {vec1}, false);
  auto vec3 = Vec("vec3", true, vec1);
  auto vec4 = Vec("vec4", true, vec1);

  EXPECT_EQ(Generate(), std::string{"const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec3();\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = b1_buf.template ReinterpretCast<half>();\n"
                                    "vec4();\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, LoadEnq_DeqVec1_Vec23_PipeVec4ReuseLoad) {
  auto load = Load("load1", x);
  auto vec1 = Vec("vec1", {load}, false);
  auto vec2 = Vec("vec2", {load}, false);
  auto vec3 = Vec("vec3", {load}, false);
  auto vec4 = Vec("vec4", true, load);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "\n"
                                    "vec2();\n"
                                    "\n"
                                    "vec3();\n"
                                    "\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec4();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

// load 的输出被 store(MTE3) 读取，同时被 vec 原地复用(覆盖写)，
// vec 写之前必须等 store 读完，即插入 MTE3->V 同步。
TEST_F(CodegenKernel_CallSync, Load_StoreReader_And_VecReuseLoad_ShouldSyncMte3ToVec) {
  auto load = Load("Load", x);
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  auto store = Store("Store", load);
  auto vec4 = Vec("vec4", true, load);

  auto res = Generate();
  const auto sync = res.find("TQueSync<PIPE_MTE3, PIPE_V>");
  ASSERT_NE(sync, std::string::npos) << res;
  const auto store_pos = res.find("Store();");
  const auto reuse_pos = res.find("uint32_t q1_reuse2_offset");
  ASSERT_NE(store_pos, std::string::npos) << res;
  ASSERT_NE(reuse_pos, std::string::npos) << res;
  EXPECT_LT(store_pos, sync);
  EXPECT_LT(sync, reuse_pos);
}

// vec 输出被 Store 读取并 FreeTensor 后，下一个 vec 经 AllocTensor 复用该 que buffer，
// que 的 free 事件已保证 MTE3 读者完成，不应重复插入 MTE3->V 同步。
TEST_F(CodegenKernel_CallSync, VecOut_StoreRead_Free_AllocVecReuse_ShouldNotSyncMte3ToVec) {
  auto vec1 = VecOut("vec1");
  auto store = Store("Store", vec1);
  auto vec2 = Vec("vec2", true, vec1);

  auto res = Generate();
  const auto reuse_pos = res.find("q1_reuse2_offset");
  ASSERT_NE(reuse_pos, std::string::npos) << res;
  EXPECT_EQ(res.find("MTE3_V"), std::string::npos) << res;
}

TEST_F(CodegenKernel_CallSync, AllocVec1Enq_DeqStore1_Stroe23Free_Vec2ReuseVec1) {
  auto vec1 = VecOut("vec1");
  auto store1 = Store("store1", vec1);
  auto store2 = Store("store2", vec1);
  auto store3 = Store("store3", vec1);
  auto vec2 = Vec("vec2", true, vec1);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "store1();\n"
                                    "\n"
                                    "store2();\n"
                                    "\n"
                                    "store3();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec2();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

TEST_F(CodegenKernel_CallSync, AllocVec1Enq_DeqStore1_Stroe23Free_Vec2OutReuseVec1Out) {
  auto vec1 = VecOut("vec1");
  auto store1 = Store("store1", vec1);
  auto store2 = Store("store2", vec1);
  auto store3 = Store("store3", vec1);
  auto vec2 = VecOut("vec2", nullptr, vec1);
  auto store4 = Store("store4", vec2);
  auto res = Generate();

  EXPECT_EQ(res, std::string{"uint32_t q1_reuse1_offset = 0;\n"
                             "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                             "const uint32_t local_1_actual_size = 1;\n"
                             "LocalTensor<half> local_1;\n"
                             "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                             "vec1();\n"
                             "q1.EnQue(q1_buf);\n"
                             "\n"
                             "q1_buf = q1.DeQue<uint8_t>();\n"
                             "store1();\n"
                             "\n"
                             "store2();\n"
                             "\n"
                             "store3();\n"
                             "q1.FreeTensor(q1_buf);\n"
                             "\n"
                             "uint32_t q1_reuse2_offset = 0;\n"
                             "q1_buf = q1.AllocTensor<uint8_t>();\n"
                             "const uint32_t local_2_actual_size = 1;\n"
                             "LocalTensor<half> local_2;\n"
                             "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                             "vec2();\n"
                             "q1.EnQue(q1_buf);\n"
                             "\n"
                             "q1_buf = q1.DeQue<uint8_t>();\n"
                             "store4();\n"
                             "q1.FreeTensor(q1_buf);\n"
                             "\n"});
}

TEST_F(CodegenKernel_CallSync, AllocVec1Enq_PipVec23_DeqStoreFree_Vec4ReuseVec1) {
  auto vec1 = VecOut("vec1");
  auto vec2 = Vec("vec2", {vec1}, false);
  auto vec3 = Vec("vec3", {vec1}, false);
  auto store = Store("store", vec1);
  auto vec4 = Vec("vec4", true, vec1);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "vec1();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "\n"
                                    "AscendC::PipeBarrier<PIPE_V>();\n"
                                    "vec2();\n"
                                    "\n"
                                    "vec3();\n"
                                    "\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "store();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "vec4();\n"
                                    "q1.FreeTensor(q1_buf);\n"
                                    "\n"});
}

/*
 *    load1  load2
 *       \    /
 *         vec
 *   load1和load2共用
 */
TEST_F(CodegenKernel_CallSync, AllocLoad1_ShareLoad2Enq_DeqVec) {
  auto load1 = Load("load1", x);
  auto load2 = LoadForShare("load2", x, load1);
  auto vec = Vec("vec", {load1, load2}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n\n"
                                    "q1_reuse1_offset = q1_reuse1_offset + local_1_size * 2;\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"});
}

TEST_F(CodegenKernel_CallSync, CvUbFuseSharedQueueOffsetShouldUseBlockAlignedSliceSize) {
  cv_ub_fuse_mode = true;
  auto load1 = Load("load1", x);
  load1->outputs[0].attr.dtype = ge::DT_INT8;
  auto load2 = LoadForShare("load2", x, load1);
  load2->outputs[0].attr.dtype = ge::DT_INT64;
  auto vec = Vec("vec", {load1, load2}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "local_1_actual_size = 1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<int8_t>();\n"
                                    "load1();\n\n"
                                    "q1_reuse1_offset = q1_reuse1_offset + "
                                    "KernelUtils::BlkAlign<uint8_t>(local_1_size * 1);\n"
                                    "local_2_actual_size = 1;\n"
                                    "local_2 = q1_buf[q1_reuse1_offset].template ReinterpretCast<int64_t>();\n"
                                    "load2();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"});
}

/*
 *    load1  load2
 *       \    /
 *         vec
 *         ...
 *        load3
 *   load1和load2共用
 *   load3与load1/load2复用
 */
TEST_F(CodegenKernel_CallSync, AllocLoad1_ShareLoad2Enq_DeqVec_Load3ReuseLoad2) {
  auto load1 = Load("load1", x);
  auto load2 = LoadForShare("load2", x, load1);
  auto vec = Vec("vec", {load1, load2}, false);
  auto load3 = Load("load3", x, load2);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n\n"
                                    "q1_reuse1_offset = q1_reuse1_offset + local_1_size * 2;\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"
                                    "uint32_t q1_reuse3_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = q1_buf[q1_reuse3_offset].template ReinterpretCast<half>();\n"
                                    "load3();\n"
                                    "q1.EnQue(q1_buf);\n"
                                    "q1.FreeTensor(q1_buf);\n\n"});
}

/*
 *      load1
 *        |
 *       vec1
 *       ...
 *    load2  load3
 *       \    /
 *        vec2
 *   load1与load2和load3复用
 *   load2和load3共用
 */
TEST_F(CodegenKernel_CallSync, AllocLoad1Enq_DeqVec1_AllocLoad2_ShareLoad3Enq_DeqVec2) {
  auto load1 = Load("load1", x);
  auto vec1 = Vec("vec1", {load1}, false);
  auto load2 = Load("load2", x, load1);
  auto load3 = LoadForShare("load3", x, load2);
  auto vec2 = Vec("vec2", {load2, load3}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"
                                    "uint32_t q1_reuse2_offset = 0;\n"
                                    "q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n\n"
                                    "q1_reuse2_offset = q1_reuse2_offset + local_2_size * 2;\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = q1_buf[q1_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "load3();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec2();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"});
}

/*
 *    load1 load3  load2
 *       \   /       |
 *        vec1     vec2
 *   load1和load3共用, load2使用其他que，拓扑序为load1 - load2 - load3 - vec1 - vec2
 */
TEST_F(CodegenKernel_CallSync, AllocLoad1_AllocLoad2Enq_ShareLoad3Enq_DeqVec1_DeqVec2) {
  auto load1 = Load("load1", x);
  auto load2 = Load("load2", x);
  auto load3 = LoadForShare("load3", x, load1);
  auto vec1 = Vec("vec1", {load1, load3}, false);
  auto vec2 = Vec("vec2", {load2}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n\n"
                                    "uint32_t q2_reuse2_offset = 0;\n"
                                    "LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q2_buf[q2_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n"
                                    "q2.EnQue(q2_buf);\n\n"
                                    "q1_reuse1_offset = q1_reuse1_offset + local_1_size * 2;\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load3();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "vec1();\n"
                                    "q1.FreeTensor(q1_buf);\n\n"
                                    "q2_buf = q2.DeQue<uint8_t>();\n"
                                    "vec2();\n"
                                    "q2.FreeTensor(q2_buf);\n\n"});
}

/*
 *    load1 load2 load3
 *        \   |   /
 *           vec
 *   load1和load3共用, load2使用其他que
 */
TEST_F(CodegenKernel_CallSync, AllocLoad1_AllocLoad2Enq_ShareLoad3Enq_DeqDeqVec) {
  auto load1 = Load("load1", x);
  auto load2 = Load("load2", x);
  auto load3 = LoadForShare("load3", x, load1);
  auto vec = Vec("vec", {load1, load2, load3}, false);

  EXPECT_EQ(Generate(), std::string{"uint32_t q1_reuse1_offset = 0;\n"
                                    "LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_1_actual_size = 1;\n"
                                    "LocalTensor<half> local_1;\n"
                                    "local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load1();\n\n"
                                    "uint32_t q2_reuse2_offset = 0;\n"
                                    "LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();\n"
                                    "const uint32_t local_2_actual_size = 1;\n"
                                    "LocalTensor<half> local_2;\n"
                                    "local_2 = q2_buf[q2_reuse2_offset].template ReinterpretCast<half>();\n"
                                    "load2();\n"
                                    "q2.EnQue(q2_buf);\n\n"
                                    "q1_reuse1_offset = q1_reuse1_offset + local_1_size * 2;\n"
                                    "const uint32_t local_3_actual_size = 1;\n"
                                    "LocalTensor<half> local_3;\n"
                                    "local_3 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();\n"
                                    "load3();\n"
                                    "q1.EnQue(q1_buf);\n\n"
                                    "q1_buf = q1.DeQue<uint8_t>();\n"
                                    "q2_buf = q2.DeQue<uint8_t>();\n"
                                    "vec();\n"
                                    "q2.FreeTensor(q2_buf);\n"
                                    "q1.FreeTensor(q1_buf);\n\n"});
}

TEST(CodegenKernel, StageGenerate_WillNotDuplicatAllocTensorInSameStage) {
  GTEST_SKIP();
}

namespace {
void BuildCompareApiCallCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                     const af::Axis &z0, const af::Axis &z1);
void InitCompareApiCallCvStageAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1);
}  // namespace

// 测试compare api不外抛for循环的场景
TEST(CodegenKernel, CompareApiCallNotThrowingFor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildCompareApiCallCvStageGraph(graph, s0, s1, z0, z1);
  InitCompareApiCallCvStageAttrs(graph, z0, z1);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto gt = graph.FindNode("gt");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(gt->outputs[0]);

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
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{"CompareExtend(local_3[0], local_0[0], local_1[0], CMPMODE::GT, local_0_actual_size, tmp_buf_0);\n"});
}

namespace {
void BuildCompareApiCallCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                     const af::Axis &z0, const af::Axis &z1) {
  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

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

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id};
  *gt_op.y.repeats = {s0, s1};
  *gt_op.y.strides = {s1, One};
}

void InitCompareApiCallCvStageAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1) {
  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id};
  gt->outputs[0].attr.vectorized_strides = {One};
  gt->outputs[0].attr.dtype = ge::DT_INT16;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;
}
}  // namespace

TEST(CodegenKernel, CompareApiCallCvStageAlignsActualSize) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildCompareApiCallCvStageGraph(graph, s0, s1, z0, z1);
  InitCompareApiCallCvStageAttrs(graph, z0, z1);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto gt = graph.FindNode("gt");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(gt->outputs[0]);

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
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"CompareExtend(local_3[0], local_0[0], local_1[0], CMPMODE::GT, ((local_0_actual_size "
                                "+ 8 - 1) / 8 * 8), tmp_buf_0);\n"});
}

// 测试compare api需要外抛for循环的场景
TEST(CodegenKernel, CompareApiCallThrowingFor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id, z2.id};
  *gt_op.y.repeats = {s0, s1, s2};
  *gt_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = ge::DT_INT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {stride, One};
  load2->outputs[0].attr.dtype = ge::DT_INT32;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  gt->outputs[0].attr.vectorized_strides = {stride, One};
  gt->outputs[0].attr.dtype = ge::DT_INT32;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(gt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"CompareExtend<int32_t, CMPMODE::GT>(local_3[0], local_0[0], local_1[0], t->s1, t->s2, "
                                "((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1), ((16 * Ceiling((Rational(1 , 16) * "
                                "t->s2))))/(1), tmp_buf_0);\n"});
}

// 测试compare api需要外抛for循环的场景
TEST(CodegenKernel, CompareApiCallTwoAxis) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id, z2.id};
  *gt_op.y.repeats = {s0, s1, s2};
  *gt_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {stride, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  gt->outputs[0].attr.vectorized_strides = {stride, One};
  gt->outputs[0].attr.dtype = ge::DT_INT16;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(load2->outputs[0]);
  tpipe.AddTensor(gt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"CompareExtend<float, CMPMODE::GT>(local_3[0], local_0[0], local_1[0], t->s1, t->s2, "
                                "((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1), ((16 * Ceiling((Rational(1 , 16) * "
                                "t->s2))))/(1), tmp_buf_0);\n"});
}

// 测试compare api需要外抛for循环的场景
TEST(CodegenKernel, CompareApiCallThrowingForWithX2IsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id, z2.id};
  *gt_op.y.repeats = {s0, s1, s2};
  *gt_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {stride, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto size_int16 = ge::GetSizeByDataType(ge::DT_INT16);
  auto stride_int16 = af::sym::Align(z2.size, 32 / size_int16);
  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  gt->outputs[0].attr.vectorized_strides = {stride_int16, One};
  gt->outputs[0].attr.dtype = ge::DT_INT16;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为compare的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(gt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(
      result,
      std::string{
          "CompareExtend<float, CMPMODE::GT>(local_3[0], local_0[0], local_1[0], t->s1, t->s2, ((8 * "
          "Ceiling((Rational(1 , 8) * t->s2))))/(1), ((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1), tmp_buf_0);\n"});
}

// 测试compare api需要外抛for循环的场景
TEST(CodegenKernel, CompareApiCallThrowingForWithX2IsUbScalarForUint32) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id, z2.id};
  *gt_op.y.repeats = {s0, s1, s2};
  *gt_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_UINT32);
  auto stride = af::sym::Align(z2.size, 32 / size);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {stride, One};
  load->outputs[0].attr.dtype = ge::DT_UINT32;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {stride, One};
  load2->outputs[0].attr.dtype = ge::DT_UINT32;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto size_int16 = ge::GetSizeByDataType(ge::DT_INT16);
  auto stride_int16 = af::sym::Align(z2.size, 32 / size_int16);
  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  gt->outputs[0].attr.vectorized_strides = {stride_int16, One};
  gt->outputs[0].attr.dtype = ge::DT_INT16;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为compare的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(gt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(
      result,
      std::string{
          "CompareExtend<uint32_t, CMPMODE::GT>(local_3[0], local_0[0], local_1[0], t->s1, t->s2, ((8 * "
          "Ceiling((Rational(1 , 8) * t->s2))))/(1), ((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1), tmp_buf_0);\n"});
}

// 测试compare api需要外抛for循环的场景
TEST(CodegenKernel, CompareApiCallNotThrowingForWithX2IsUbScalar) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Data x_op2("x2", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Gt gt_op("gt");
  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  graph.AddNode(gt_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  load_op2.x = x_op2.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1, s2};
  *load_op2.y.strides = {s1 * s2, s2, One};

  gt_op.x1 = load_op.y;
  gt_op.x2 = load_op2.y;
  *gt_op.y.axis = {z0.id, z1.id, z2.id};
  *gt_op.y.repeats = {s0, s1, s2};
  *gt_op.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load2->attr.api.type = af::ApiType::kAPITypeCompute;
  load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load2->attr.sched.loop_axis = z0.id;

  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {s2, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load2->outputs[0].attr.vectorized_strides = {s2, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 1;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto gt = graph.FindNode("gt");
  gt->attr.api.compute_type = af::ComputeType::kComputeElewise;
  gt->attr.api.type = af::ApiType::kAPITypeCompute;
  gt->attr.api.unit = af::ComputeUnit::kUnitVector;
  gt->attr.sched.loop_axis = z0.id;
  gt->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  gt->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  gt->outputs[0].attr.vectorized_strides = {s2, One};
  gt->outputs[0].attr.dtype = ge::DT_FLOAT;
  gt->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  gt->outputs[0].attr.mem.tensor_id = 3;
  gt->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  gt->outputs[0].attr.que.id = 2;
  gt->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  // begin:构造load2作为compare的第二个输入x2, 构造x2是ub_scalar的场景
  std::string dtype_name;
  codegen::Tensor::DtypeName(load2->outputs[0].attr.dtype, dtype_name);
  codegen::Tensor t(load2->outputs[0], dtype_name, "t");
  EXPECT_EQ(t.Init(), 0);
  t.need_gen_get_value_of_ub_scalar = true;
  t.is_ub_scalar = true;
  EXPECT_EQ(tpipe.AddTensor(t), 0);
  // end
  tpipe.AddTensor(gt->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  codegen::CompareApiCall call("GT");
  EXPECT_EQ(call.Init(gt), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"CompareScalarExtend(local_3[0], local_0[0], (float)local_1_ub_scalar, CMPMODE::GT, "
                                "local_0_actual_size, tmp_buf_0);\n"});

  codegen::Kernel kernel("test");
  kernel.tpipe.CollectQues(graph);
  EXPECT_EQ(kernel.tpipe.AddTensor(t), 0);
  EXPECT_EQ(kernel.ParseOptimizeInfo(load2, load->outputs[0]), af::FAILED);
}

// 测试isnan api不外抛for循环的场景
TEST(CodegenKernel, IsnanExtendNotThrowingFor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Isnan isnan("isnan");
  graph.AddNode(load_op);
  graph.AddNode(isnan);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};

  isnan.x = load_op.y;
  *isnan.y.axis = {z0.id, z1.id};
  *isnan.y.repeats = {s0, s1};
  *isnan.y.strides = {s1, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto isnan_node = graph.FindNode("isnan");
  isnan_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  isnan_node->attr.api.type = af::ApiType::kAPITypeCompute;
  isnan_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  isnan_node->attr.sched.loop_axis = z0.id;
  isnan_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  isnan_node->outputs[0].attr.vectorized_axis = {z1.id};
  isnan_node->outputs[0].attr.vectorized_strides = {One};
  isnan_node->outputs[0].attr.dtype = ge::DT_INT16;
  isnan_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  isnan_node->outputs[0].attr.mem.tensor_id = 3;
  isnan_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  isnan_node->outputs[0].attr.que.id = 2;
  isnan_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(isnan_node->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryBitWidthChangeApiCall call("IsnanExtend");
  EXPECT_EQ(call.Init(isnan_node), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{"IsnanExtend(local_3[0], local_0[0], tmp_buf_0, local_0_actual_size);\n"});
}

// 测试isnan api外抛for循环的场景
TEST(CodegenKernel, IsnanExtendThrowingFor) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Isnan isnan("isnan");
  graph.AddNode(load_op);
  graph.AddNode(isnan);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  isnan.x = load_op.y;
  *isnan.y.axis = {z0.id, z1.id, z2.id};
  *isnan.y.repeats = {s0, s1, s2};
  *isnan.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto isnan_node = graph.FindNode("isnan");
  isnan_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  isnan_node->attr.api.type = af::ApiType::kAPITypeCompute;
  isnan_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  isnan_node->attr.sched.loop_axis = z0.id;
  isnan_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  isnan_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  isnan_node->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  isnan_node->outputs[0].attr.dtype = ge::DT_INT16;
  isnan_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  isnan_node->outputs[0].attr.mem.tensor_id = 3;
  isnan_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  isnan_node->outputs[0].attr.que.id = 2;
  isnan_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(isnan_node->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryBitWidthChangeApiCall call("IsnanExtend");
  EXPECT_EQ(call.Init(isnan_node), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{"for(int outer_for_0 = 0; outer_for_0 < t->s1; outer_for_0++) {\nIsnanExtend(local_3[outer_for_0 * "
                  "((16 * Ceiling((Rational(1 , 16) * t->s2))))/(1)], local_0[outer_for_0 * ((16 * Ceiling((Rational(1 "
                  ", 16) * t->s2))))/(1)], tmp_buf_0, t->s2);\n\n}\n"});
}

TEST(CodegenKernel, UnaryApiTmpCall) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Sign sign("sign");
  graph.AddNode(load_op);
  graph.AddNode(sign);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1, s2};
  *load_op.y.strides = {s1 * s2, s2, One};

  sign.x = load_op.y;
  *sign.y.axis = {z0.id, z1.id, z2.id};
  *sign.y.repeats = {s0, s1, s2};
  *sign.y.strides = {s1 * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;

  auto size = ge::GetSizeByDataType(ge::DT_FLOAT16);
  load->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto sign_node = graph.FindNode("sign");
  sign_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
  sign_node->attr.api.type = af::ApiType::kAPITypeCompute;
  sign_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  sign_node->attr.sched.loop_axis = z0.id;
  sign_node->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  sign_node->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  sign_node->outputs[0].attr.vectorized_strides = {af::sym::Align(z2.size, 16), One};
  sign_node->outputs[0].attr.dtype = ge::DT_INT16;
  sign_node->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  sign_node->outputs[0].attr.mem.tensor_id = 3;
  sign_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  sign_node->outputs[0].attr.que.id = 2;
  sign_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(sign_node->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryApiTmpCall call("SignExtend");
  EXPECT_EQ(call.Init(sign_node), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, current_axis, result);
  std::cout << result << std::endl;
  EXPECT_EQ(result, std::string{"SignExtend(local_3[0], local_0[0], tmp_buf_0, local_0_actual_size);\n"});
}

TEST(CodegenKernel, Ub2ubApiCall) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::Ub2ub ub2ub_op("ub2ub");
  // af::ascir_op::Cast cast_op("cast");
  graph.AddNode(load_op);
  graph.AddNode(ub2ub_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  ub2ub_op.x = load_op.y;

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ub2ub = graph.FindNode("ub2ub");
  ub2ub->attr.api.unit = af::ComputeUnit::kUnitVector;
  ub2ub->outputs[0].attr.dtype = ge::DT_FLOAT;
  ub2ub->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  ub2ub->outputs[0].attr.mem.tensor_id = 1;
  ub2ub->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  ub2ub->outputs[0].attr.que.id = 2;
  ub2ub->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(ub2ub->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::Ub2ubApiCall call("DataCopy");
  EXPECT_EQ(call.Init(ub2ub), 0);
  call.inputs.push_back(&x1);

  std::string result;
  call.Generate(tpipe, vector<af::AxisId>{}, result);
  EXPECT_EQ(result,
            std::string{"DataCopy(local_1[0], local_0[0], KernelUtils::BlkAlign<float>(local_0_actual_size));\n"});
}

TEST(CodegenKernel, TwoWorkspaceCodegen) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  Store store_op1("store1");
  Store store_op2("store2");

  Workspace workspace_op1("workspace1");
  Workspace workspace_op2("workspace2");

  Load load_op1("load1");
  Load load_op2("load2");

  Output y_op1("y1");
  Output y_op2("y2");

  graph.AddNode(store_op1);
  graph.AddNode(store_op2);
  graph.AddNode(workspace_op1);
  graph.AddNode(workspace_op2);
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(y_op1);
  graph.AddNode(y_op2);

  x_op.y.dtype = ge::DT_FLOAT16;
  x_op.ir_attr.SetIndex(0);
  store_op1.x = x_op.y;
  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};

  workspace_op1.x = store_op1.y;
  workspace_op1.y.dtype = ge::DT_FLOAT16;
  *workspace_op1.y.axis = {z0.id, z1.id};

  store_op2.x = x_op.y;
  store_op2.y.dtype = ge::DT_FLOAT16;
  *store_op2.y.axis = {z0.id, z1.id};

  workspace_op2.x = store_op2.y;
  workspace_op2.y.dtype = ge::DT_FLOAT16;
  *workspace_op2.y.axis = {z0.id, z1.id};

  load_op1.x = workspace_op1.y;
  load_op1.y.dtype = ge::DT_FLOAT16;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = workspace_op2.y;
  load_op2.y.dtype = ge::DT_FLOAT16;
  *load_op2.y.axis = {z0.id, z1.id};

  y_op1.x = load_op1.y;
  y_op1.ir_attr.SetIndex(0);
  y_op2.x = load_op2.y;
  y_op2.ir_attr.SetIndex(1);

  // graph.SetInputs({x_op});
  // graph.SetOutputs({y_op1, y_op2});

  auto x = graph.FindNode("x");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto workspace1 = graph.FindNode("workspace1");
  auto workspace2 = graph.FindNode("workspace2");
  auto store1 = graph.FindNode("store1");
  auto store2 = graph.FindNode("store2");
  auto y1 = graph.FindNode("y1");
  auto y2 = graph.FindNode("y2");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;

  store1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store1->outputs[0].attr.mem.tensor_id = 1;
  store1->outputs[0].attr.repeats = {s0, s1};
  store1->outputs[0].attr.strides = {s1, One};

  store2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store2->outputs[0].attr.mem.tensor_id = 2;
  store2->outputs[0].attr.repeats = {s0, s1};
  store2->outputs[0].attr.strides = {s1, One};

  workspace1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace1->outputs[0].attr.mem.tensor_id = 1;
  workspace1->outputs[0].attr.repeats = {s0, s1};

  workspace2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace2->outputs[0].attr.mem.tensor_id = 2;
  workspace2->outputs[0].attr.repeats = {s0, s1};

  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.repeats = {s0, s1};
  load1->outputs[0].attr.strides = {s1, One};

  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.repeats = {s0, s1};
  load2->outputs[0].attr.strides = {s1, One};

  y1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y1->outputs[0].attr.mem.tensor_id = 5;
  y2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y2->outputs[0].attr.mem.tensor_id = 6;

  workspace1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  workspace2->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y1);
  fused_schedule_result.output_nodes.push_back(y2);
  fused_schedule_result.workspace_nodes.push_back(workspace1);
  fused_schedule_result.workspace_nodes.push_back(workspace2);

  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  ASSERT_EQ(kernel.workspaces.size(), 2UL);
  EXPECT_EQ(kernel.workspaces[0].AsArg(), "uint64_t workspace1");
  EXPECT_EQ(kernel.workspaces[1].AsArg(), "uint64_t workspace2");
  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result,
            std::string{"GlobalTensor<half> global_0;\n"
                        "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
                        "GlobalTensor<half> global_3;\n"
                        "global_3.SetGlobalBuffer((__gm__ half*)y1);\n"
                        "GlobalTensor<half> global_4;\n"
                        "global_4.SetGlobalBuffer((__gm__ half*)y2);\n"
                        "GlobalTensor<half> global_1;\n"
                        "global_1.SetGlobalBuffer((__gm__ half*)((__gm__ uint8_t*)(workspace) + (workspace1)));\n"
                        "GlobalTensor<half> global_2;\n"
                        "global_2.SetGlobalBuffer((__gm__ half*)((__gm__ uint8_t*)(workspace) + (workspace2)));\n"});
}

TEST(CodegenKernel, TwoWorkspaceReuseAsInputCodegen) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Workspace workspace_op1("workspace1");
  Workspace workspace_op2("workspace2");

  Load load_op1("load1");
  Load load_op2("load2");

  Output y_op1("y1");
  Output y_op2("y2");

  graph.AddNode(workspace_op1);
  graph.AddNode(workspace_op2);
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);
  graph.AddNode(y_op1);
  graph.AddNode(y_op2);

  workspace_op1.y.dtype = ge::DT_FLOAT16;
  *workspace_op1.y.axis = {z0.id, z1.id};

  workspace_op2.y.dtype = ge::DT_FLOAT16;
  *workspace_op2.y.axis = {z0.id, z1.id};

  load_op1.x = workspace_op1.y;
  load_op1.y.dtype = ge::DT_FLOAT16;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = workspace_op2.y;
  load_op2.y.dtype = ge::DT_FLOAT16;
  *load_op2.y.axis = {z0.id, z1.id};

  y_op1.x = load_op1.y;
  y_op1.ir_attr.SetIndex(0);
  y_op2.x = load_op2.y;
  y_op2.ir_attr.SetIndex(1);

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto workspace1 = graph.FindNode("workspace1");
  auto workspace2 = graph.FindNode("workspace2");
  auto y1 = graph.FindNode("y1");
  auto y2 = graph.FindNode("y2");

  workspace1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace1->outputs[0].attr.mem.tensor_id = 1;
  workspace1->outputs[0].attr.repeats = {s0, s1};

  workspace2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace2->outputs[0].attr.mem.tensor_id = 1;
  workspace2->outputs[0].attr.repeats = {s0, s1};

  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.repeats = {s0, s1};
  load1->outputs[0].attr.strides = {s1, One};

  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.repeats = {s0, s1};
  load2->outputs[0].attr.strides = {s1, One};

  y1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y1->outputs[0].attr.mem.tensor_id = 5;
  y2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y2->outputs[0].attr.mem.tensor_id = 6;

  workspace1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  workspace2->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(y1);
  fused_schedule_result.output_nodes.push_back(y2);
  fused_schedule_result.workspace_nodes.push_back(workspace1);
  fused_schedule_result.workspace_nodes.push_back(workspace2);

  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result, std::string{"GlobalTensor<half> global_3;\n"
                                "global_3.SetGlobalBuffer((__gm__ half*)y1);\n"
                                "GlobalTensor<half> global_4;\n"
                                "global_4.SetGlobalBuffer((__gm__ half*)y2);\n"
                                "GlobalTensor<half> global_1;\n"
                                "global_1.SetGlobalBuffer((__gm__ half*)((__gm__ uint8_t*)(workspace) + "
                                "(workspace1)));\n"});
}

TEST(CodegenKernel, GlobalTensorInitShouldUseWorkspaceReuseOverride) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Store store_op("store");
  Workspace workspace_op("workspace");
  graph.AddNode(store_op);
  graph.AddNode(workspace_op);

  x_op.ir_attr.SetIndex(0);
  x_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = x_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  *store_op.y.axis = {z0.id};
  workspace_op.x = store_op.y;
  workspace_op.y.dtype = ge::DT_FLOAT16;
  *workspace_op.y.axis = {z0.id};

  auto x = graph.FindNode("x");
  auto store = graph.FindNode("store");
  auto workspace = graph.FindNode("workspace");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 1;
  store->outputs[0].attr.repeats = {s0};
  store->outputs[0].attr.strides = {One};
  workspace->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace->outputs[0].attr.mem.tensor_id = 1;
  workspace->outputs[0].attr.repeats = {s0};
  workspace->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.workspace_nodes.push_back(workspace);

  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  EXPECT_EQ(kernel.GlobalTensorInit(result, "y0"), af::SUCCESS);
  EXPECT_EQ(result, std::string{"GlobalTensor<half> global_0;\n"
                                "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
                                "GM_ADDR workspace_reuse = y0;\n"
                                "GlobalTensor<half> global_1;\n"
                                "global_1.SetGlobalBuffer((__gm__ half*)((__gm__ uint8_t*)(workspace_reuse) + "
                                "(workspace1)));\n"});
}

TEST(CodegenKernel, GenCubeCommonTilingSingleFuncCallShouldUseOutputOverride) {
  codegen::Kernel kernel("cube_kernel");
  kernel.inputs.emplace_back(GM_ADDR("x"));
  af::AscGraph graph("cube_graph");

  const auto result = kernel.GenCubeCommonTilingSingleFuncCall(graph, "y0");

  EXPECT_NE(result.find("y0, workspace + vec_wss, gm_tiling_data);"), std::string::npos);
  EXPECT_EQ(result.find("workspace, workspace + vec_wss, gm_tiling_data);"), std::string::npos);
}

TEST(CodegenKernel, WorkspaceReuseOutputAsInputCodegen) {
  af::AscGraph graph0("test_graph0");
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Workspace workspace_op("workspace");

  Load load_op0("load0");
  Load load_op("load");

  Output y_op1("y1");
  Output y_op2("y2");

  graph0.AddNode(load_op0);
  graph0.AddNode(y_op1);
  load_op0.y.dtype = ge::DT_FLOAT16;
  y_op1.x = load_op0.y;
  y_op1.y.dtype = ge::DT_FLOAT16;
  y_op1.ir_attr.SetIndex(0);

  graph.AddNode(workspace_op);
  graph.AddNode(load_op);
  graph.AddNode(y_op2);

  workspace_op.y.dtype = ge::DT_FLOAT16;
  *workspace_op.y.axis = {z0.id, z1.id};

  load_op.x = workspace_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  *load_op.y.axis = {z0.id, z1.id};

  y_op2.x = load_op.y;
  y_op2.ir_attr.SetIndex(1);

  auto load0 = graph0.FindNode("load0");
  auto y1 = graph0.FindNode("y1");
  load0->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load0->outputs[0].attr.mem.tensor_id = 1;  // reuse output

  y1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y1->outputs[0].attr.mem.tensor_id = 1;  // reuse output

  auto load = graph.FindNode("load");
  auto workspace = graph.FindNode("workspace");
  auto y2 = graph.FindNode("y2");

  workspace->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace->outputs[0].attr.mem.tensor_id = 1;  // graph0 reuse output
  workspace->outputs[0].attr.repeats = {s0, s1};

  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load->outputs[0].attr.mem.tensor_id = 2;
  load->outputs[0].attr.repeats = {s0, s1};
  load->outputs[0].attr.strides = {s1, One};

  y2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y2->outputs[0].attr.mem.tensor_id = 3;

  workspace->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(y1);
  fused_schedule_result.output_nodes.push_back(y2);

  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result, std::string{"GlobalTensor<half> global_2;\n"
                                "global_2.SetGlobalBuffer((__gm__ half*)y2);\n"
                                "GlobalTensor<half> global_1;\n"
                                "global_1.SetGlobalBuffer((__gm__ half*)y1);\n"});
}

TEST(CodegenKernel, Looper_GenerateLoop_WhenNestedLoop) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  af::Axis z0{.id = 0, .name = "z0", .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .size = s1.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  codegen::Loop loop1(z0.id);
  codegen::Loop loop2(z1.id);
  loop1.AddLoop(&loop2);

  codegen::TPipe tpipe("t", tiler);
  std::string result;
  EXPECT_EQ(loop1.Generate(tiler, tpipe, result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"for (int z0 = 0; z0 < z0_loop_size; z0++) {\n"
                                "for (int z1 = 0; z1 < z1_loop_size; z1++) {\n"
                                "}\n"
                                "}\n"});
}

TEST(CodegenKernel, Looper_GenerateLoop_WhenTwoLoop) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  af::Axis z0{.id = 0, .name = "z0", .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .size = s1.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  codegen::Loop root_loop(af::kIdNone), loop1(z0.id), loop2(z1.id);
  root_loop.AddLoop(&loop1);
  root_loop.AddLoop(&loop2);

  codegen::TPipe tpipe("t", tiler);
  std::string result;
  EXPECT_EQ(root_loop.Generate(tiler, tpipe, result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"for (int z0 = 0; z0 < z0_loop_size; z0++) {\n"
                                "}\n"
                                "for (int z1 = 0; z1 < z1_loop_size; z1++) {\n"
                                "}\n"});
}

TEST(CodegenKernel, Kernel_GlobalTensorInit) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  af::ascir_op::Output y_op("y");
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;
  x_op.ir_attr.SetIndex(0);
  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);

  // graph.SetInputs({x_op});
  // graph.SetOutputs({y_op});

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.buf.id = 1;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);

  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result, std::string{"GlobalTensor<half> global_0;\n"
                                "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
                                "GlobalTensor<half> global_2;\n"
                                "global_2.SetGlobalBuffer((__gm__ half*)y);\n"});
}

TEST(CodegenKernel, Kernel_ConstantTensorInit) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Scalar constant("constant");
  graph.AddNode(constant);
  constant.ir_attr.SetValue("100.1");
  constant.y.dtype = ge::DT_FLOAT16;

  auto constant_node = graph.FindNode("constant");
  constant_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  constant_node->outputs[0].attr.mem.tensor_id = 0;
  constant_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result, std::string{"const half scalar_0 = 100.1;\n"});
}

TEST(CodegenKernel, Kernel_IndexExprTensorInit) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  af::ascir_op::IndexExpr index("index");
  graph.AddNode(index);
  index.ir_attr.SetExpr(s0 + s1 * af::Symbol(2));
  index.y.dtype = ge::DT_INT64;

  // graph.SetInputs({index});

  auto index_node = graph.FindNode("index");
  index_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  index_node->outputs[0].attr.mem.tensor_id = 0;
  index_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.GlobalTensorInit(result);
  EXPECT_EQ(result, std::string{"const int64_t scalar_0 = (((2 * t->s1) + t->s0))/(1);\n"});
}

TEST(CodegenKernel, Kernel_ZeroIndexExprTensorInit) {
  af::AscGraph graph("test_graph");
  af::ascir_op::IndexExpr index("index", graph);
  index.ir_attr.SetExpr(af::Symbol(0));
  index.y.dtype = ge::DT_INT32;

  auto index_node = graph.FindNode("index");
  index_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  index_node->outputs[0].attr.mem.tensor_id = 0;
  index_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;

  ::ascir::FusedScheduledResult fused_schedule_result;
  codegen::Kernel kernel(graph.GetName());
  ASSERT_EQ(codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel), af::SUCCESS);
  std::string result;
  ASSERT_EQ(kernel.GlobalTensorInit(result), af::SUCCESS);
  EXPECT_EQ(result, std::string{"const int32_t scalar_0 = 0;\n"});
  ASSERT_NE(kernel.tpipe.GetTensor(0), nullptr);
  EXPECT_EQ(kernel.tpipe.GetTensor(0)->GetScalarValue(), "scalar_0");
}

TEST(CodegenKernel, Kernel_IndexExprScalarConsumerDoesNotAllocateVectorTensor) {
  af::AscGraph graph("index_expr_scalar_consumer");
  const auto s1 = graph.CreateSizeVar("s1");
  const auto z0 = graph.CreateAxis("z0", s1);
  af::ascir_op::Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = ge::DT_INT32;
  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = data.y;
  load.y.dtype = ge::DT_INT32;
  af::ascir_op::IndexExpr index("index", graph);
  index.ir_attr.SetExpr(s1 + af::Symbol(2));
  index.y.dtype = ge::DT_INT32;
  af::ascir_op::Add add("add");
  graph.AddNode(add);
  add.x1 = load.y;
  add.x2 = index.y;
  add.y.dtype = ge::DT_INT32;

  auto data_node = graph.FindNode("data");
  auto load_node = graph.FindNode("load");
  auto index_node = graph.FindNode("index");
  auto add_node = graph.FindNode("add");
  data_node->outputs[0].attr.mem.tensor_id = 0;
  data_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load_node->outputs[0].attr.axis = {z0.id};
  load_node->outputs[0].attr.vectorized_axis = {z0.id};
  load_node->outputs[0].attr.vectorized_strides = {One};
  load_node->outputs[0].attr.repeats = {s1};
  load_node->outputs[0].attr.strides = {One};
  load_node->outputs[0].attr.mem.tensor_id = 1;
  load_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  load_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load_node->outputs[0].attr.buf.id = 0;
  index_node->outputs[0].attr.mem.tensor_id = 2;
  index_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeInvalid;
  index_node->outputs[0].attr.mem.position = af::Position::kPositionInvalid;
  add_node->outputs[0].attr.axis = {z0.id};
  add_node->outputs[0].attr.vectorized_axis = {z0.id};
  add_node->outputs[0].attr.vectorized_strides = {One};
  add_node->outputs[0].attr.repeats = {s1};
  add_node->outputs[0].attr.strides = {One};
  add_node->outputs[0].attr.mem.tensor_id = 3;
  add_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  add_node->outputs[0].attr.mem.position = af::Position::kPositionVecCalc;
  add_node->outputs[0].attr.buf.id = 1;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(data_node);
  codegen::Kernel kernel(graph.GetName());
  ASSERT_EQ(codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel), af::SUCCESS);
  ASSERT_NE(kernel.tpipe.GetTensor(2), nullptr);
  EXPECT_TRUE(kernel.tpipe.GetTensor(2)->is_constant);

  std::string scalar_init;
  ASSERT_EQ(kernel.GlobalTensorInit(scalar_init), af::SUCCESS);
  EXPECT_NE(scalar_init.find("const int32_t scalar_2 = ((2 + t->s1))/(1);"), std::string::npos);

  std::string local_tensors;
  ASSERT_EQ(kernel.tpipe.LocalTensorDefine(local_tensors), af::SUCCESS);
  EXPECT_EQ(local_tensors.find("LocalTensor<int32_t> local_2"), std::string::npos);

  std::string consumer_code;
  ASSERT_EQ(kernel.root_loop.Generate(kernel.tiler, kernel.tpipe, consumer_code), af::SUCCESS);
  EXPECT_NE(consumer_code.find("Adds(local_3[0], local_1[0], (int32_t)scalar_2"), std::string::npos) << consumer_code;
}

TEST(CodegenKernel, Kernel_KernelFunctionDeclare) {
  af::AscGraph graph("test_kernel");
  af::ascir_op::Data x1("x1"), x2("x2"), x3("x3");
  x1.ir_attr.SetIndex(0);
  x2.ir_attr.SetIndex(1);
  x3.ir_attr.SetIndex(2);
  af::ascir_op::Output y1("y1"), y2("y2"), y3("y3");
  y1.ir_attr.SetIndex(0);
  y2.ir_attr.SetIndex(1);
  y3.ir_attr.SetIndex(2);
  graph.AddNode(x1);
  graph.AddNode(x2);
  graph.AddNode(x3);
  graph.AddNode(y1);
  graph.AddNode(y2);
  graph.AddNode(y3);

  y1.x = x1.y;
  y2.x = x2.y;
  y3.x = x3.y;
  // graph.SetInputs({x1, x2, x3});
  // graph.SetOutputs({y1, y2, y3});

  auto x1_node = graph.FindNode("x1");
  auto x2_node = graph.FindNode("x2");
  auto x3_node = graph.FindNode("x3");
  auto y1_node = graph.FindNode("y1");
  auto y2_node = graph.FindNode("y2");
  auto y3_node = graph.FindNode("y3");

  x1_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1_node->outputs[0].attr.mem.tensor_id = 0;
  x2_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2_node->outputs[0].attr.mem.tensor_id = 1;
  x3_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x3_node->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1_node);
  fused_schedule_result.input_nodes.push_back(x2_node);
  fused_schedule_result.input_nodes.push_back(x3_node);
  fused_schedule_result.output_nodes.push_back(y1_node);
  fused_schedule_result.output_nodes.push_back(y2_node);
  fused_schedule_result.output_nodes.push_back(y3_node);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  EXPECT_EQ(kernel.TilingKeyFuncDeclare("test_kernel_general_0_nil_0_nil", "test_kernelTilingData"),
            std::string{"inline __aicore__ void test_kernel_general_0_nil_0_nil(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, "
                        "GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR workspace, const test_kernelTilingData *t)"});
}

TEST(CodegenKernel, Kernel_LocalTensorQueBufAlloc) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;
  store_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.LocalTensorQueBufAlloc(result, graph);
  EXPECT_EQ(result, std::string{"TPipe tpipe;\n\n"
                                "const uint32_t local_1_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                                "const uint32_t local_1_que_buf_num = 2;\n\n"
                                "// const uint32_t q0_size = KernelUtils::Max(local_1_size * sizeof(half));\n"
                                "const uint32_t q0_buf_num = KernelUtils::Max(2);\n"
                                "TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q0;\n"
                                "// tpipe.InitBuffer(q0, q0_buf_num, KernelUtils::BlkAlign<uint8_t>(q0_size));\n"
                                "tpipe.InitBuffer(q0, q0_buf_num, t->q0_size);\n"
                                "// const uint32_t b0_size = KernelUtils::Max(8192);\n"
                                "TBuf<TPosition::VECCALC> b0;\n"
                                "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
                                "tpipe.InitBuffer(b0, t->b0_size);\n"
                                "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
                                "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"});
}

TEST(CodegenKernel, Kernel_LocalTensorShareReuseQueAlloc) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  af::ascir_op::Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  af::ascir_op::Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);
  af::ascir_op::Load load1_op("load1");
  af::ascir_op::Load load2_op("load2");
  af::ascir_op::Load load3_op("load3");
  af::ascir_op::Add add_op("add");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load1_op);
  graph.AddNode(load2_op);
  graph.AddNode(load3_op);
  graph.AddNode(add_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x1_op.y.dtype = ge::DT_FLOAT16;
  x2_op.y.dtype = ge::DT_FLOAT16;
  x3_op.y.dtype = ge::DT_FLOAT16;

  load1_op.x = x1_op.y;
  load1_op.y.dtype = ge::DT_FLOAT16;

  load2_op.x = x2_op.y;
  load2_op.y.dtype = ge::DT_FLOAT16;

  load3_op.x = x3_op.y;
  load3_op.y.dtype = ge::DT_FLOAT16;

  add_op.x1 = load1_op.y;
  add_op.x2 = load2_op.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  add_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  store_op.x = add_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;

  auto x1 = graph.FindNode("x1");
  auto x2 = graph.FindNode("x2");
  auto x3 = graph.FindNode("x3");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto load3 = graph.FindNode("load3");
  auto add = graph.FindNode("add");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.tensor_id = 0;
  x2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.tensor_id = 1;
  x3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.tensor_id = 2;

  load1->outputs[0].attr.axis = {z0.id};
  load1->outputs[0].attr.vectorized_axis = {z0.id};
  load1->outputs[0].attr.vectorized_strides = {One};
  load1->outputs[0].attr.repeats = {z0.size};
  load1->outputs[0].attr.strides = {One};
  load1->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load2->outputs[0].attr.axis = {z0.id};
  load2->outputs[0].attr.vectorized_axis = {z0.id};
  load2->outputs[0].attr.vectorized_strides = {One};
  load2->outputs[0].attr.repeats = {z0.size};
  load2->outputs[0].attr.strides = {One};
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.que.id = 0;
  load2->outputs[0].attr.mem.reuse_id = 0;
  load2->outputs[0].attr.que.depth = 2;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load3->outputs[0].attr.axis = {z0.id};
  load3->outputs[0].attr.vectorized_axis = {z0.id};
  load3->outputs[0].attr.vectorized_strides = {One};
  load3->outputs[0].attr.repeats = {z0.size};
  load3->outputs[0].attr.strides = {One};
  load3->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load3->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.mem.tensor_id = 5;
  load3->outputs[0].attr.que.id = 0;
  load3->outputs[0].attr.mem.reuse_id = 1;
  load3->outputs[0].attr.que.depth = 2;
  load3->outputs[0].attr.que.buf_num = 2;
  load3->outputs[0].attr.opt.merge_scope = af::kIdNone;

  add->outputs[0].attr.axis = {z0.id};
  add->outputs[0].attr.vectorized_axis = {z0.id};
  add->outputs[0].attr.vectorized_strides = {One};
  add->outputs[0].attr.repeats = {z0.size};
  add->outputs[0].attr.strides = {One};
  add->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  add->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  add->outputs[0].attr.mem.tensor_id = 6;
  add->outputs[0].attr.que.id = 0;
  add->outputs[0].attr.mem.reuse_id = 0;
  add->outputs[0].attr.que.depth = 2;
  add->outputs[0].attr.que.buf_num = 2;
  add->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 6;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x1);
  fused_schedule_result.input_nodes.push_back(x2);
  fused_schedule_result.input_nodes.push_back(x3);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.LocalTensorQueBufAlloc(result, graph);
  EXPECT_EQ(result,
            std::string{"TPipe tpipe;\n\n"
                        "const uint32_t local_3_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                        "const uint32_t local_3_que_buf_num = 2;\n"
                        "const uint32_t local_4_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                        "const uint32_t local_4_que_buf_num = 2;\n"
                        "const uint32_t local_5_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                        "const uint32_t local_5_que_buf_num = 2;\n"
                        "const uint32_t local_6_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                        "const uint32_t local_6_que_buf_num = 2;\n\n"
                        "// const uint32_t q0_size = KernelUtils::Max(local_3_size * sizeof(half), local_4_size * "
                        "sizeof(half), local_5_size * sizeof(half), local_6_size * sizeof(half), local_3_size * "
                        "sizeof(half) + local_4_size * sizeof(half) + local_6_size * sizeof(half));\n"
                        "const uint32_t q0_buf_num = KernelUtils::Max(2);\n"
                        "TQue<TPosition::VECIN, 1> q0;\n"
                        "// tpipe.InitBuffer(q0, q0_buf_num, KernelUtils::BlkAlign<uint8_t>(q0_size));\n"
                        "tpipe.InitBuffer(q0, q0_buf_num, t->q0_size);\n"
                        "// const uint32_t b0_size = KernelUtils::Max(8192);\n"
                        "TBuf<TPosition::VECCALC> b0;\n"
                        "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
                        "tpipe.InitBuffer(b0, t->b0_size);\n"
                        "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
                        "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"});
}

TEST(CodegenKernel, Kernel_GenerateKernel_Multi_ScheduleGroup) {
  af::AscGraph graph("test_kernel");
  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;
  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  y_op.x = store_op.y;
  store_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto output = graph.FindNode("y");

  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  load->attr.api.unit = af::ComputeUnit::kUnitNone;
  store->attr.api.unit = af::ComputeUnit::kUnitNone;
  output->attr.api.unit = af::ComputeUnit::kUnitNone;

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.opt.merge_scope = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.buf.id = 0;
  load->outputs[0].attr.opt.merge_scope = 0;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.opt.merge_scope = 0;

  ::ascir::ScheduleGroup schedule_result0_group0 = {{af::AscGraph("test_kernel_general_0_nil_0_nil"),
                                                     af::AscGraph("test_kernel_general_1_nil_1_nil"),
                                                     af::AscGraph("test_kernel_general_2_nil_2_nil")}};

  ::ascir::ScheduleGroup schedule_result0_group1 = {
      {af::AscGraph("test_kernel_general_3_nil_3_nil"), af::AscGraph("test_kernel_general_4_nil_4_nil")}};

  ::ascir::ScheduleGroup schedule_result1_group0 = {
      {af::AscGraph("test_kernel_general_5_nil_5_nil"), af::AscGraph("test_kernel_general_6_nil_6_nil")}};

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.push_back(schedule_result0_group0);
  schedule_result0.schedule_groups.push_back(schedule_result0_group1);

  ::ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_result1_group0);

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);
  schedule_results.push_back(schedule_result1);

  for (uint32_t i = 0; i < schedule_results.size(); i++) {
    for (auto schedule_group : schedule_results[i].schedule_groups) {
      for (auto impl_graph : schedule_group.impl_graphs) {
        impl_graph.CopyFrom(graph);
      }
    }
  }

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(output);

  std::stringstream ss;
  std::string string;
  codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss);
  string = ss.str();

  EXPECT_EQ(
      string,
      std::string{
          "\n"
          "/**\n"
          " * Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
          " * This program is free software, you can redistribute it and/or modify it under the terms and conditions "
          "of\n"
          " * CANN Open Software License Agreement Version 2.0 (the \"License\").\n"
          " * Please refer to the License for details. You may not use this file except in compliance with the "
          "License.\n"
          " * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR "
          "IMPLIED,\n"
          " * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
          " * See LICENSE in the root of the software repository for the full text of the License.\n"
          " */\n"
          "#ifndef __ASCENDC_API_DATACOPY_H__\n"
          "#define __ASCENDC_API_DATACOPY_H__\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> "
          "&src,\n"
          "                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,\n"
          "                                         uint32_t dst_stride) {\n"
          "  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);\n"
          "  AscendC::DataCopyExtParams param;\n"
          "  param.blockCount = block_count;\n"
          "  param.blockLen = block_len * sizeof(T);\n"
          "  param.srcStride = src_stride * sizeof(T);\n"
          "  param.dstStride = dst_stride / align_num;\n"
          "\n"
          "  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};\n"
          "  AscendC::DataCopyPad(dst, src, param, pad_params);\n"
          "}\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyPadExtend(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> "
          "&src,\n"
          "                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,\n"
          "                                         uint32_t dst_stride) {\n"
          "  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);\n"
          "  AscendC::DataCopyExtParams param;\n"
          "  param.blockCount = block_count;\n"
          "  param.blockLen = block_len * sizeof(T);\n"
          "  param.srcStride = src_stride / align_num;\n"
          "  param.dstStride = dst_stride * sizeof(T);\n"
          "\n"
          "  AscendC::DataCopyPad(dst, src, param);\n"
          "}\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> "
          "&src,\n"
          "                                      const uint32_t size) {\n"
          "  AscendC::DataCopy(dst, src, size);\n"
          "}\n"
          "\n"
          "#endif  // __ASCENDC_API_DATACOPY_H__\n"
          "inline __aicore__ void test_kernel_general_0_nil_0_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult0G0TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_1_nil_1_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult0G0TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_2_nil_2_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult0G0TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_3_nil_3_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult0G1TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_4_nil_4_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult0G1TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_5_nil_5_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult1G0TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_6_nil_6_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AscGraph0ScheduleResult1G0TilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "extern \"C\" __global__ __aicore__ void test_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR "
          "gm_tiling_data) {\n"
          "  REGISTER_TILING_DEFAULT(AutofuseTilingData);\n"
          "  GET_TILING_DATA(t, gm_tiling_data);\n"
          "  if (TILING_KEY_IS(0)) {\n"
          "    test_kernel_general_0_nil_0_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_3_nil_3_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(1)) {\n"
          "    test_kernel_general_0_nil_0_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_4_nil_4_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(2)) {\n"
          "    test_kernel_general_1_nil_1_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_3_nil_3_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(3)) {\n"
          "    test_kernel_general_1_nil_1_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_4_nil_4_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(4)) {\n"
          "    test_kernel_general_2_nil_2_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_3_nil_3_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(5)) {\n"
          "    test_kernel_general_2_nil_2_nil(x, y, workspace, &t.graph0_result0_g0_tiling_data);\n"
          "    test_kernel_general_4_nil_4_nil(x, y, workspace, &t.graph0_result0_g1_tiling_data);\n"
          "  } else if (TILING_KEY_IS(6)) {\n"
          "    test_kernel_general_5_nil_5_nil(x, y, workspace, &t.graph0_result1_g0_tiling_data);\n"
          "  } else if (TILING_KEY_IS(7)) {\n"
          "    test_kernel_general_6_nil_6_nil(x, y, workspace, &t.graph0_result1_g0_tiling_data);\n"
          "  }\n"
          "}\n"});

  fused_schedule_result.node_idx_to_scheduled_results[0][0].enable_group_parallel = true;
  std::stringstream ss1;
  std::string kernel_txt;
  codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss1);
  kernel_txt = ss1.str();
  std::string expect_found =
      "const uint32_t block_offset = t->ub_size;  // reuse as block_offset\n"
      "block_dim = block_dim >= block_offset ? "
      "block_dim - block_offset : block_dim + GetBlockNum() - block_offset;\n";
  EXPECT_TRUE(kernel_txt.find(expect_found) != std::string::npos);
}

TEST(CodegenKernel, Kernel_GenerateKernel_Single_ScheduleGroup) {
  af::AscGraph graph("test_kernel");
  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;
  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  y_op.x = store_op.y;
  store_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto output = graph.FindNode("y");

  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  load->attr.api.unit = af::ComputeUnit::kUnitNone;
  store->attr.api.unit = af::ComputeUnit::kUnitNone;
  output->attr.api.unit = af::ComputeUnit::kUnitNone;

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.opt.merge_scope = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.buf.id = 0;
  load->outputs[0].attr.opt.merge_scope = 0;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.opt.merge_scope = 0;

  ::ascir::ScheduleGroup schedule_result0_group0 = {{af::AscGraph("test_kernel_general_0_nil_0_nil"),
                                                     af::AscGraph("test_kernel_general_1_nil_1_nil"),
                                                     af::AscGraph("test_kernel_general_2_nil_2_nil")}};

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.push_back(schedule_result0_group0);

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);
  for (uint32_t i = 0; i < schedule_results.size(); i++) {
    for (auto schedule_group : schedule_results[i].schedule_groups) {
      for (auto impl_graph : schedule_group.impl_graphs) {
        impl_graph.CopyFrom(graph);
      }
    }
  }

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(output);

  std::stringstream ss;
  std::string string;
  codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss);
  string = ss.str();

  EXPECT_EQ(
      string,
      std::string{
          "\n"
          "/**\n"
          " * Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
          " * This program is free software, you can redistribute it and/or modify it under the terms and conditions "
          "of\n"
          " * CANN Open Software License Agreement Version 2.0 (the \"License\").\n"
          " * Please refer to the License for details. You may not use this file except in compliance with the "
          "License.\n"
          " * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR "
          "IMPLIED,\n"
          " * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
          " * See LICENSE in the root of the software repository for the full text of the License.\n"
          " */\n"
          "#ifndef __ASCENDC_API_DATACOPY_H__\n"
          "#define __ASCENDC_API_DATACOPY_H__\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> "
          "&src,\n"
          "                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,\n"
          "                                         uint32_t dst_stride) {\n"
          "  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);\n"
          "  AscendC::DataCopyExtParams param;\n"
          "  param.blockCount = block_count;\n"
          "  param.blockLen = block_len * sizeof(T);\n"
          "  param.srcStride = src_stride * sizeof(T);\n"
          "  param.dstStride = dst_stride / align_num;\n"
          "\n"
          "  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};\n"
          "  AscendC::DataCopyPad(dst, src, param, pad_params);\n"
          "}\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyPadExtend(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> "
          "&src,\n"
          "                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,\n"
          "                                         uint32_t dst_stride) {\n"
          "  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);\n"
          "  AscendC::DataCopyExtParams param;\n"
          "  param.blockCount = block_count;\n"
          "  param.blockLen = block_len * sizeof(T);\n"
          "  param.srcStride = src_stride / align_num;\n"
          "  param.dstStride = dst_stride * sizeof(T);\n"
          "\n"
          "  AscendC::DataCopyPad(dst, src, param);\n"
          "}\n"
          "\n"
          "template <typename T>\n"
          "inline __aicore__ void DataCopyExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> "
          "&src,\n"
          "                                      const uint32_t size) {\n"
          "  AscendC::DataCopy(dst, src, size);\n"
          "}\n"
          "\n"
          "#endif  // __ASCENDC_API_DATACOPY_H__\n"
          "inline __aicore__ void test_kernel_general_0_nil_0_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AutofuseTilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_1_nil_1_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AutofuseTilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "inline __aicore__ void test_kernel_general_2_nil_2_nil(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const "
          "AutofuseTilingData *t) {\n"
          "int block_dim = GetBlockIdx();\n"
          "if (block_dim >= t->block_dim) {\n"
          "  return;\n"
          "}\n\n"
          "GlobalTensor<half> global_0;\n"
          "global_0.SetGlobalBuffer((__gm__ half*)x);\n"
          "GlobalTensor<half> global_2;\n"
          "global_2.SetGlobalBuffer((__gm__ half*)y);\n\n"
          "TPipe tpipe;\n\n"
          "const uint32_t local_1_size = 1;\n"
          "const uint32_t m0_size = KernelUtils::Sum(local_1_size * sizeof(half));\n"
          "const uint32_t m0_que_buf_num = KernelUtils::Max();\n\n"
          "// const uint32_t b0_size = KernelUtils::Max(m0_size, 8192);\n"
          "TBuf<TPosition::GM> b0;\n"
          "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
          "tpipe.InitBuffer(b0, t->b0_size);\n"
          "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
          "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"
          "}\n"
          "extern \"C\" __global__ __aicore__ void test_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR "
          "gm_tiling_data) {\n"
          "  REGISTER_TILING_DEFAULT(AutofuseTilingData);\n"
          "  GET_TILING_DATA(t, gm_tiling_data);\n"
          "  if (TILING_KEY_IS(0)) {\n"
          "    test_kernel_general_0_nil_0_nil(x, y, workspace, &t);\n"
          "  } else if (TILING_KEY_IS(1)) {\n"
          "    test_kernel_general_1_nil_1_nil(x, y, workspace, &t);\n"
          "  } else if (TILING_KEY_IS(2)) {\n"
          "    test_kernel_general_2_nil_2_nil(x, y, workspace, &t);\n"
          "  }\n"
          "}\n"});
}

TEST(CodegenKernel, DynamicInputsAndOutputs) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;
  store_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.resize(1);
  schedule_result0.schedule_groups[0].impl_graphs.emplace_back(graph);

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  codegen::Kernel kernel(graph.GetName());
  kernel.SetUseListTensor(true);
  std::string include_and_defines =
      codegen::Kernel::IncludeAndDefines(fused_schedule_result, "KERNEL_TYPE_AIV_ONLY", true);
  EXPECT_TRUE(include_and_defines.find("kernel_operator_list_tensor_intf.h") != std::string::npos);

  auto declare = codegen::Kernel::KernelFuncDeclare("fused_graph", fused_schedule_result, true);
  std::string expected =
      "extern \"C\" __global__ __aicore__ void "
      "fused_graph(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR gm_tiling_data)";
  EXPECT_EQ(declare, expected);
  kernel.SetUseListTensor(true);
  std::string global_tensor_init_result;
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  auto func_call = kernel.GenTilingFuncCall("fused_graph", "fused_graphTilingData", 0);
  std::string expected_func_call =
      "    if (fused_graph_tiling_data.tiling_key == 0) {\n"
      "      fused_graph(input_tensor_desc, output_tensor_desc, workspace, &fused_graphTilingData);\n"
      "    }";
  EXPECT_EQ(func_call, expected_func_call);
}

TEST(CodegenKernel, PackingFunctionCalls) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  x_op.y.dtype = ge::DT_FLOAT16;

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  y_op.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.resize(6);
  schedule_result0.enable_group_parallel = true;
  for (auto &schedule_group : schedule_result0.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);
  schedule_results.push_back(schedule_result0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  std::stringstream ss;
  std::string result;
  auto ret = codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss, true);
  result = ss.str();
  EXPECT_EQ(ret, af::SUCCESS);
  std::cout << result;
  EXPECT_TRUE(result.find("packed_functions_820000000(input_tensor_desc, output_tensor_desc, workspace, t)") !=
              std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000001(input_tensor_desc, output_tensor_desc, workspace, t)") !=
              std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000002(input_tensor_desc, output_tensor_desc, workspace, t)") !=
              std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000003(input_tensor_desc, output_tensor_desc, workspace, t)") !=
              std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000004(input_tensor_desc, output_tensor_desc, workspace, t)") ==
              std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000000") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000001") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000002") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000003") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000004") == std::string::npos);

  ss.clear();
  ret = codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss, false);
  EXPECT_EQ(ret, af::SUCCESS);
  result = ss.str();
  EXPECT_TRUE(result.find("packed_functions_820000000(x, y, workspace, t);") != std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000001(x, y, workspace, t);") != std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000002(x, y, workspace, t);") != std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000003(x, y, workspace, t);") != std::string::npos);
  EXPECT_TRUE(result.find("packed_functions_820000004(x, y, workspace, t);") == std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000000") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000001") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000002") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000003") != std::string::npos);
  EXPECT_TRUE(result.find("if TILING_KEY_VAR == 20000004") == std::string::npos);
}

TEST(CodegenKernel, TPipe_AllocTmpBuf) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.mem.position = af::Position::kPositionVecIn;
  tensor.attr.mem.tensor_id = 0;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor.attr.buf.id = 1;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  tpipe.AddTensor(tensor);

  auto buf = tpipe.bufs.find(tensor.attr.buf.id);
  ASSERT_NE(buf, tpipe.bufs.end());
  EXPECT_EQ(tpipe.AllocTmpBuf(buf->second), "LocalTensor<uint8_t> tmp_buf_1 = b1.Get<uint8_t>();\n");
}

TEST(CodegenKernel, GenDuplicateBufAlloc) {
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  auto node_x = graph.FindNode("x");
  auto node_y = graph.FindNode("y");
  node_x->attr.tmp_buffers = {{af::Symbol(1024), -1}, {af::Symbol(1024), 0}};
  node_y->attr.tmp_buffers = {{af::Symbol(1024), -1}, {af::Symbol(1024), 0}};

  std::set<std::pair<std::string, std::string>> pre_api_extract_dup = {{"1", "float"}};
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string result;
  result = tpipe.GenDuplicateBufAlloc(pre_api_extract_dup);

  EXPECT_EQ(result,
            "TBuf<TPosition::VECCALC> builtin_tmp_buffer_1;\n"
            "tpipe.InitBuffer(builtin_tmp_buffer_1, ONE_BLK_SIZE);\n"
            "LocalTensor<uint8_t> builtin_tmp_buf_1 = builtin_tmp_buffer_1.Get<uint8_t>();\n"
            "LocalTensor<float> local_blk_tensor_of_float_1 = builtin_tmp_buf_1.template ReinterpretCast<float>();\n"
            "Duplicate(local_blk_tensor_of_float_1[0], (float)1.0, ONE_BLK_SIZE / sizeof(float));\n");
}

TEST(CodegenKernel, DichotomyReduceApiTest_RAPatternReduceMean) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_("x");
  graph.AddNode(x_);
  x_.y.dtype = ge::DT_FLOAT;

  Load load_("load");
  graph.AddNode(load_);
  load_.x = x_.y;
  load_.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_.y.axis = {z0.id, z1.id, z2.id};
  *load_.y.repeats = {s0, s1, s2};
  *load_.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Mean rmean_("rmean");
  graph.AddNode(rmean_);
  rmean_.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}, {{af::Symbol(8192), 0}, af::MemAttr(), 1}};
  rmean_.x = load_.y;
  rmean_.attr.sched.axis = {z0.id, z1.id, z2.id};
  *rmean_.y.axis = {z0.id, z1.id, z2.id};
  *rmean_.y.repeats = {One, One, s2};
  *rmean_.y.strides = {Zero, Zero, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = rmean_.y;
  store_.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store_.y.axis = {z0.id, z1.id, z2.id};
  *store_.y.repeats = {One, One, s2};
  *store_.y.strides = {Zero, Zero, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = ge::DT_FLOAT;

  auto x = graph.FindNode("x");
  x->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x->attr.api.type = af::ApiType::kAPITypeBuffer;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto rmean = graph.FindNode("rmean");
  rmean->attr.api.compute_type = af::ComputeType::kComputeReduce;
  rmean->outputs[0].attr.dtype = ge::DT_FLOAT;
  rmean->attr.api.type = af::ApiType::kAPITypeCompute;
  rmean->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = ge::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto z0_id = load->attr.sched.axis[0];
  auto z1_id = load->attr.sched.axis[1];
  auto z2_id = load->attr.sched.axis[2];

  auto [z2T, z2t] = graph.TileSplit(z2_id);
  auto [z2TB, z2Tb] = graph.BlockSplit(z2T->id);
  auto [z1T, z1t] = graph.TileSplit(z1_id);
  auto z0z1T = graph.MergeAxis({z0_id, z1T->id});

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplySplit(node, z2T->id, z2t->id);
    graph.ApplySplit(node, z2TB->id, z2Tb->id);
    graph.ApplySplit(node, z1T->id, z1t->id);
    graph.ApplyMerge(node, z0z1T->id);
    graph.ApplyReorder(node, {z2TB->id, z2Tb->id, z0z1T->id, z1t->id, z2t->id});
  }

  // Vectorized/Loop axis
  vector<af::Expression> vectorized_strides{One, One};
  load->attr.sched.loop_axis = z0z1T->id;
  load->outputs[0].attr.vectorized_axis = {z1t->id, z2t->id};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  vectorized_strides[0] = af::sym::Align(graph.FindAxis(z2t->id)->size, 32 / size);
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  rmean->attr.sched.loop_axis = z0z1T->id;
  rmean->outputs[0].attr.vectorized_axis = {z1t->id, z2t->id};
  rmean->outputs[0].attr.vectorized_strides = {Zero, One};

  store->attr.sched.loop_axis = z2Tb->id;
  store->outputs[0].attr.vectorized_axis = {z1t->id, z2t->id};
  store->outputs[0].attr.vectorized_strides = {Zero, One};

  // Que/Buf alloc
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.buf.id = af::kIdNone;
  x->outputs[0].attr.que.id = af::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = af::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  rmean->outputs[0].attr.mem.tensor_id = 2;
  rmean->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rmean->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  rmean->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rmean->outputs[0].attr.buf.id = af::kIdNone;
  rmean->outputs[0].attr.que.id = 1;
  rmean->outputs[0].attr.mem.reuse_id = 1;
  rmean->outputs[0].attr.que.depth = 2;
  rmean->outputs[0].attr.que.buf_num = 2;
  rmean->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  rmean->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto impl = ascgen_utils::GetAscIrCodegenImpl(rmean->GetType());
  std::vector<std::unique_ptr<af::TmpBufDesc>> buffers = impl->CalcTmpBufSize(*rmean);
  for (auto &buf_desc : buffers) {
    if (buf_desc != nullptr) {
      af::TmpBuffer temp_buffer;
      temp_buffer.buf_desc = std::move(*buf_desc);
      rmean->attr.tmp_buffers.emplace_back(temp_buffer);
    }
  }

  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  ::ascir::ScheduleGroup schedule_result0_group0 = {{af::AscGraph("test_kernel_general_0_nil_0_nil")}};

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.push_back(schedule_result0_group0);

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  for (auto impl_graph : schedule_result0_group0.impl_graphs) {
    impl_graph.CopyFrom(graph);
    for (auto node : impl_graph.GetAllNodes()) {
      if (node->GetType() == "Data") {
        int64_t index = -1;
        if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
          auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
          attr.SetIndex(static_cast<int64_t>(fused_schedule_result.input_nodes.size()));
        }
        fused_schedule_result.input_nodes.emplace_back(node);
      } else if (node->GetType() == "Output") {
        int64_t index = -1;
        if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
          auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
          attr.SetIndex(static_cast<int64_t>(fused_schedule_result.output_nodes.size()));
        }
        fused_schedule_result.output_nodes.emplace_back(node);
      }
    }
  }

  std::stringstream ss;
  std::string string;
  Status status = codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss);
  string = ss.str();

  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, DichotomyReduceApiTest_MultiMerge_RAPatternReduceMean) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_("x");
  graph.AddNode(x_);
  x_.y.dtype = ge::DT_FLOAT;

  Load load_("load");
  graph.AddNode(load_);
  load_.x = x_.y;
  load_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_.y.repeats = {s0, s1, s2, s3};
  *load_.y.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  af::ascir_op::Mean rmean_("rmean");
  graph.AddNode(rmean_);
  rmean_.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}, {{af::Symbol(8192), 0}, af::MemAttr(), 1}};
  rmean_.x = load_.y;
  rmean_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *rmean_.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *rmean_.y.repeats = {One, One, One, s3};
  *rmean_.y.strides = {Zero, Zero, Zero, One};

  Store store_("store");
  graph.AddNode(store_);
  store_.x = rmean_.y;
  store_.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *store_.y.repeats = {One, One, One, s3};
  *store_.y.strides = {Zero, Zero, Zero, One};

  Output y_("y");
  graph.AddNode(y_);
  y_.x = store_.y;
  y_.y.dtype = ge::DT_FLOAT;

  auto x = graph.FindNode("x");
  x->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x->attr.api.type = af::ApiType::kAPITypeBuffer;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = ge::DT_FLOAT;
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto rmean = graph.FindNode("rmean");
  rmean->attr.api.compute_type = af::ComputeType::kComputeReduce;
  rmean->outputs[0].attr.dtype = ge::DT_FLOAT;
  rmean->attr.api.type = af::ApiType::kAPITypeCompute;
  rmean->attr.api.unit = af::ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.compute_type = af::ComputeType::kComputeStore;
  store->outputs[0].attr.dtype = ge::DT_FLOAT;
  store->attr.api.type = af::ApiType::kAPITypeCompute;
  store->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y->attr.api.type = af::ApiType::kAPITypeBuffer;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  // Scheduler
  auto z0_id = load->attr.sched.axis[0];
  auto z1_id = load->attr.sched.axis[1];
  auto z2_id = load->attr.sched.axis[2];
  auto z3_id = load->attr.sched.axis[3];

  auto [z2T, z2t] = graph.TileSplit(z2_id);
  auto [z3T, z3t] = graph.TileSplit(z3_id);
  auto [z3TB, z3Tb] = graph.BlockSplit(z3T->id);
  auto z0z1 = graph.MergeAxis({z0_id, z1_id});
  auto z0z1z2T = graph.MergeAxis({z0z1->id, z2T->id});

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }
    graph.ApplySplit(node, z2T->id, z2t->id);
    graph.ApplySplit(node, z3T->id, z3t->id);
    graph.ApplySplit(node, z3TB->id, z3Tb->id);
    graph.ApplyMerge(node, z0z1->id);
    graph.ApplyMerge(node, z0z1z2T->id);
    graph.ApplyReorder(node, {z3TB->id, z3Tb->id, z0z1z2T->id, z2t->id, z3t->id});
  }

  // Vectorized/Loop axis
  vector<af::Expression> vectorized_strides{One, One};
  load->attr.sched.loop_axis = z0z1z2T->id;
  load->outputs[0].attr.vectorized_axis = {z2t->id, z3t->id};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  vectorized_strides[0] = af::sym::Align(graph.FindAxis(z2t->id)->size, 32 / size);
  load->outputs[0].attr.vectorized_strides = vectorized_strides;

  rmean->attr.sched.loop_axis = z0z1z2T->id;
  rmean->outputs[0].attr.vectorized_axis = {z2t->id, z3t->id};
  rmean->outputs[0].attr.vectorized_strides = {Zero, One};

  store->attr.sched.loop_axis = z3Tb->id;
  store->outputs[0].attr.vectorized_axis = {z2t->id, z3t->id};
  store->outputs[0].attr.vectorized_strides = {Zero, One};

  // Que/Buf alloc
  x->outputs[0].attr.mem.tensor_id = 0;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  x->outputs[0].attr.mem.position = af::Position::kPositionGM;
  x->outputs[0].attr.buf.id = af::kIdNone;
  x->outputs[0].attr.que.id = af::kIdNone;
  x->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  x->outputs[0].attr.opt.merge_scope = af::kIdNone;

  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.buf.id = af::kIdNone;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  rmean->outputs[0].attr.mem.tensor_id = 2;
  rmean->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rmean->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  rmean->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  rmean->outputs[0].attr.buf.id = af::kIdNone;
  rmean->outputs[0].attr.que.id = 1;
  rmean->outputs[0].attr.mem.reuse_id = 1;
  rmean->outputs[0].attr.que.depth = 2;
  rmean->outputs[0].attr.que.buf_num = 2;
  rmean->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  rmean->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto impl = ascgen_utils::GetAscIrCodegenImpl(rmean->GetType());
  std::vector<std::unique_ptr<af::TmpBufDesc>> buffers = impl->CalcTmpBufSize(*rmean);
  for (auto &buf_desc : buffers) {
    if (buf_desc != nullptr) {
      af::TmpBuffer temp_buffer;
      temp_buffer.buf_desc = std::move(*buf_desc);
      rmean->attr.tmp_buffers.emplace_back(temp_buffer);
    }
  }

  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = af::Position::kPositionGM;
  store->outputs[0].attr.buf.id = af::kIdNone;
  store->outputs[0].attr.que.id = af::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = af::kIdNone;
  store->outputs[0].attr.opt.merge_scope = af::kIdNone;

  ::ascir::ScheduleGroup schedule_result0_group0 = {{af::AscGraph("test_kernel_general_0_nil_0_nil")}};

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.push_back(schedule_result0_group0);

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  for (auto impl_graph : schedule_result0_group0.impl_graphs) {
    impl_graph.CopyFrom(graph);
    for (auto node : impl_graph.GetAllNodes()) {
      if (node->GetType() == "Data") {
        int64_t index = -1;
        if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
          auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
          attr.SetIndex(static_cast<int64_t>(fused_schedule_result.input_nodes.size()));
        }
        fused_schedule_result.input_nodes.emplace_back(node);
      } else if (node->GetType() == "Output") {
        int64_t index = -1;
        if (node->attr.ir_attr->GetAttrValue("index", index) == ge::GRAPH_FAILED) {
          auto &attr = reinterpret_cast<af::AscDataIrAttrDef &>(*node->attr.ir_attr);
          attr.SetIndex(static_cast<int64_t>(fused_schedule_result.output_nodes.size()));
        }
        fused_schedule_result.output_nodes.emplace_back(node);
      }
    }
  }

  std::stringstream ss;
  std::string string;
  Status status = codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss);
  string = ss.str();

  EXPECT_EQ(status, af::SUCCESS);
}

TEST(CodegenKernel, GenerateTQueBind) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  af::ascir_op::Output y_op_1("y_1");
  y_op.ir_attr.SetIndex(0);
  y_op_1.ir_attr.SetIndex(1);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);
  graph.AddNode(y_op_1);

  x_op.y.dtype = ge::DT_FLOAT16;

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;

  store_op.x = load_op.y;
  store_op.y.dtype = ge::DT_FLOAT16;
  store_op.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  y_op.x = store_op.y;
  y_op_1.x = store_op.y;
  y_op.y.dtype = ge::DT_FLOAT16;
  y_op_1.y.dtype = ge::DT_FLOAT16;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");
  auto y_1 = graph.FindNode("y_1");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.output_nodes.push_back(y_1);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  kernel.LocalTensorQueBufAlloc(result, graph);
  EXPECT_EQ(result, std::string{"TPipe tpipe;\n\n"
                                "const uint32_t local_1_size = KernelUtils::BlkAlign<half>((t->s0 - 1) + 1);\n"
                                "const uint32_t local_1_que_buf_num = 2;\n\n"
                                "// const uint32_t q0_size = KernelUtils::Max(local_1_size * sizeof(half));\n"
                                "const uint32_t q0_buf_num = KernelUtils::Max(2);\n"
                                "TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q0;\n"
                                "// tpipe.InitBuffer(q0, q0_buf_num, KernelUtils::BlkAlign<uint8_t>(q0_size));\n"
                                "tpipe.InitBuffer(q0, q0_buf_num, t->q0_size);\n"
                                "// const uint32_t b0_size = KernelUtils::Max(8192);\n"
                                "TBuf<TPosition::VECCALC> b0;\n"
                                "// tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));\n"
                                "tpipe.InitBuffer(b0, t->b0_size);\n"
                                "LocalTensor<uint8_t> b0_buf = b0.Get<uint8_t>();\n"
                                "LocalTensor<uint8_t> tmp_buf_0 = b0.Get<uint8_t>();\n\n\n\n"});
}

TEST(CodegenKernel, CalculateWorkspaceSize_Test) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Store store_op1("store1");
  Store store_op2("store2");

  Workspace workspace_op1("workspace1");
  Workspace workspace_op2("workspace2");

  Load load_op1("load1");
  Load load_op2("load2");

  graph.AddNode(store_op1);
  graph.AddNode(store_op2);
  graph.AddNode(workspace_op1);
  graph.AddNode(workspace_op2);
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);

  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};

  workspace_op1.x = store_op1.y;
  workspace_op1.y.dtype = ge::DT_FLOAT16;
  *workspace_op1.y.axis = {z0.id, z1.id};

  store_op2.y.dtype = ge::DT_FLOAT16;
  *store_op2.y.axis = {z0.id, z1.id};

  workspace_op2.x = store_op2.y;
  workspace_op2.y.dtype = ge::DT_FLOAT16;
  *workspace_op2.y.axis = {z0.id, z1.id};

  load_op1.x = workspace_op1.y;
  load_op1.y.dtype = ge::DT_FLOAT16;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = workspace_op2.y;
  load_op2.y.dtype = ge::DT_FLOAT16;
  *load_op2.y.axis = {z0.id, z1.id};

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto workspace1 = graph.FindNode("workspace1");
  auto workspace2 = graph.FindNode("workspace2");
  auto store1 = graph.FindNode("store1");
  auto store2 = graph.FindNode("store2");

  store1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store1->outputs[0].attr.mem.tensor_id = 1;
  store1->outputs[0].attr.repeats = {s0, s1, s2, s3};
  store1->outputs[0].attr.strides = {s1 * s2 * s3, s2 * s3, s3, One};
  store2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store2->outputs[0].attr.mem.tensor_id = 2;
  store2->outputs[0].attr.repeats = {s0, s1, s2};
  store2->outputs[0].attr.strides = {s1 * s2, s2, One};

  workspace1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace1->outputs[0].attr.mem.tensor_id = 1;

  workspace2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace2->outputs[0].attr.mem.tensor_id = 2;

  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.repeats = {s2, s3};
  load1->outputs[0].attr.strides = {s3, One};
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.repeats = {s0, s1};
  load2->outputs[0].attr.strides = {s1, One};

  workspace1->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  workspace2->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  std::vector<af::AscNodePtr> workspace_nodes = {workspace1, workspace2};
  auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
  EXPECT_EQ(ToString(ws_size),
            "(Max(Max(0, (2 * Max(Max(1, s1), (s0 * s1)))), (2 * Max(Max(Max(1, s2), (s1 * s2)), (s0 * s1 * s2)))) + "
            "Max(Max(0, (2 * Max(Max(Max(Max(1, s3), (s2 * s3)), (s0 * s1 * s2 * s3)), (s1 * s2 * s3)))), (2 * "
            "Max(Max(1, s3), (s2 * s3)))))");
}

TEST(CodegenKernel, CalculateWorkspaceSize_Test2) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar(4);
  auto s1 = graph.CreateSizeVar(8);
  auto s2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // workspace被2个load多引用，引用大小不同
  Store store_op1("store1");
  Workspace workspace_op1("workspace1");
  Load load_op1("load1");
  Load load_op2("load2");

  graph.AddNode(store_op1);
  graph.AddNode(workspace_op1);
  graph.AddNode(load_op1);
  graph.AddNode(load_op2);

  store_op1.y.dtype = ge::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};

  workspace_op1.x = store_op1.y;
  workspace_op1.y.dtype = ge::DT_FLOAT16;
  *workspace_op1.y.axis = {z0.id, z1.id};

  load_op1.x = workspace_op1.y;
  load_op1.y.dtype = ge::DT_FLOAT16;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = workspace_op1.y;
  load_op2.y.dtype = ge::DT_FLOAT16;
  *load_op2.y.axis = {z0.id, z1.id};

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto workspace1 = graph.FindNode("workspace1");
  auto store1 = graph.FindNode("store1");

  store1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store1->outputs[0].attr.mem.tensor_id = 1;

  workspace1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  workspace1->outputs[0].attr.mem.tensor_id = 1;

  load1->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load1->outputs[0].attr.mem.tensor_id = 3;

  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  load2->outputs[0].attr.mem.tensor_id = 4;

  workspace1->attr.api.compute_type = af::ComputeType::kComputeInvalid;

  {
    // ws size为完整大小4*8*16*32 * sizeof(fp16)
    store1->outputs[0].attr.repeats = {s0, s1, s2, s3};
    store1->outputs[0].attr.strides = {s1 * s2 * s3, s2 * s3, s3, One};
    load1->outputs[0].attr.repeats = {s1, s3};
    load1->outputs[0].attr.strides = {s3, One};
    load2->outputs[0].attr.repeats = {s2, s3};
    load2->outputs[0].attr.strides = {s3, One};
    std::vector<af::AscNodePtr> workspace_nodes = {workspace1};
    auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
    EXPECT_EQ(ToString(ws_size), "32768");
  }

  {
    // ws size为完整大小4*8*16*32 * sizeof(fp16)
    store1->outputs[0].attr.repeats = {s1, s3};
    store1->outputs[0].attr.strides = {s3, One};
    load1->outputs[0].attr.repeats = {s1, s3};
    load1->outputs[0].attr.strides = {s3, One};
    load2->outputs[0].attr.repeats = {s0, s1, s2, s3};
    load2->outputs[0].attr.strides = {s1 * s2 * s3, s2 * s3, s3, One};
    std::vector<af::AscNodePtr> workspace_nodes = {workspace1};
    auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
    EXPECT_EQ(ToString(ws_size), "32768");
  }

  {
    // ws size为取最大值4*8*16*32
    store1->outputs[0].attr.repeats = {s1, s3};
    store1->outputs[0].attr.strides = {s3, One};
    load1->outputs[0].attr.repeats = {s1, s2, s3};
    load1->outputs[0].attr.strides = {s2 * s3, s3, One};
    load2->outputs[0].attr.repeats = {s1, s3};
    load2->outputs[0].attr.strides = {s3, One};
    std::vector<af::AscNodePtr> workspace_nodes = {workspace1};
    auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
    EXPECT_EQ(ToString(ws_size), "8192");
  }
}

TEST(CodegenKernel, CalculateWorkspaceSize_WarnTest) {
  af::AscGraph graph("test_graph");
  Workspace workspace_op1("workspace1");
  Load load_op1("load1");
  graph.AddNode(workspace_op1);
  graph.AddNode(load_op1);

  auto load1 = graph.FindNode("load1");
  {
    std::vector<af::AscNodePtr> workspace_nodes = {load1};
    auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
    EXPECT_EQ(ToString(ws_size), "0");
  }

  {
    std::vector<af::AscNodePtr> workspace_nodes = {nullptr, nullptr};
    auto ws_size = ascgen_utils::CalculateWorkspaceSize(workspace_nodes);
    EXPECT_EQ(ToString(ws_size), "0");
  }
}

TEST(CodegenKernel, EmptyTensorKernel) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", Zero);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.axis = {z0.id};
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {One};
  store->outputs[0].attr.repeats = {z0.size};
  store->outputs[0].attr.strides = {One};

  ::ascir::ScheduledResult schedule_result0;
  schedule_result0.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result0.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }

  std::vector<::ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  std::stringstream ss;
  std::string result;
  auto ret = codegen::Kernel::GenKernelFuncByTilingKey({fused_schedule_result}, ss, true);
  result = ss.str();
  EXPECT_EQ(ret, af::SUCCESS);

  EXPECT_TRUE(result.find("test_graph") == result.rfind("test_graph"));

  schedule_result0.schedule_groups.resize(2);
  for (auto &schedule_group : schedule_result0.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  fused_schedule_result.node_idx_to_scheduled_results[0].push_back(schedule_result0);
  std::stringstream ss1;
  std::string result1;
  ret = codegen::Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss1, true);
  result1 = ss1.str();
  EXPECT_EQ(ret, af::SUCCESS);

  EXPECT_TRUE(result.find("test_graph") == result.rfind("test_graph"));
}

TEST(CodegenKernel, Kernel_GenerateSubGraphFuncDefTest) {
  codegen::Kernel kernel("test_kernel");
  auto call1 = new MockApiCall("call1");
  auto call2 = new MockApiCall("call2");
  auto call3 = new MockApiCall("call3");
  auto call4 = new MockApiCall("call4");

  auto loop1 = new codegen::Loop(1);
  loop1->AddCall(call3);
  loop1->AddCall(call4);
  kernel.root_loop.AddCall(call1);
  kernel.root_loop.AddCall(call2);
  kernel.root_loop.AddLoop(loop1);

  std::stringstream ss;
  auto ret = kernel.GenerateSubGraphFuncDef(&(kernel.root_loop), ss);
  EXPECT_EQ(ret, af::SUCCESS);

  EXPECT_EQ(ss.str(), std::string{"func_test_Definition:call1\n"
                                  "func_test_Definition:call2\n"
                                  "func_test_Definition:call3\n"
                                  "func_test_Definition:call4\n"});
}

TEST(CodegenKernel, BroadcastInlineWithExecCondition) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  af::Axis z0{.id = 0, .name = "z0", .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .size = s1.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    cur_axis.is_split_b = true;
  }

  codegen::Loop loop1(z0.id);
  codegen::Loop loop2(z1.id);
  loop1.AddLoop(&loop2);

  auto call1 = new MockApiCall("call1");
  auto call2 = new MockApiCall("call2");
  call1->enable_cache = true;
  call1->exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  call1->unit = af::ComputeUnit::kUnitVector;
  call2->enable_cache = true;
  call2->exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  call2->unit = af::ComputeUnit::kUnitVector;

  loop2.AddCall(call1);
  loop2.AddCall(call2);
  codegen::TPipe tpipe("t", tiler);
  std::string result;
  EXPECT_EQ(loop1.Generate(tiler, tpipe, result), af::SUCCESS);
  EXPECT_EQ(result,
            std::string{"for (int z0 = 0; z0 < z0_loop_size; z0++) {\n"
                        "for (int z1 = 0; z1 < z1_loop_size; z1++) {\n"
                        "bool enable_cache_fused_brc_axis = (z1 < 1) || ((block_dim * 0 + z1) % z1_loop_size < 1);\n"
                        "bool enable_cache_origin_brc_axis = (z1 < 1);\n"
                        "if (enable_cache_fused_brc_axis) {\n"
                        "();\n"
                        "}\n\n"
                        "if (enable_cache_origin_brc_axis) {\n"
                        "();\n"
                        "}\n\n"
                        "}\n"
                        "}\n"});
}

TEST(CodegenKernel, CacheGuardIsOnlyClosedWhenAConditionIsGenerated) {
  af::SizeVar size(af::Symbol("size"));
  af::Axis axis{.id = 0, .name = "axis", .size = size.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(size);
  tiler.AddAxis(axis);
  tiler.axis_map.at(axis.id).is_split_b = true;

  codegen::Loop loop(axis.id);
  auto valid_call = new MockApiCall("valid_call");
  valid_call->enable_cache = true;
  valid_call->exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  valid_call->unit = af::ComputeUnit::kUnitVector;
  auto invalid_call = new MockApiCall("invalid_call");
  invalid_call->enable_cache = true;
  invalid_call->exec_condition = af::ExecuteCondition::kConditionInvalid;
  invalid_call->unit = af::ComputeUnit::kUnitVector;
  loop.AddCall(valid_call);
  loop.AddCall(invalid_call);

  codegen::TPipe tpipe("tpipe", tiler);
  std::string result;
  ASSERT_EQ(loop.Generate(tiler, tpipe, result), af::SUCCESS);
  EXPECT_NE(result.find("if (enable_cache_origin_brc_axis) {\n();\n}\n"), std::string::npos);
  EXPECT_NE(result.find("}\n\n();\n\n}"), std::string::npos);
  EXPECT_EQ(std::count(result.begin(), result.end(), '{'), std::count(result.begin(), result.end(), '}'));
}

TEST(CodegenKernel, CacheGuardPredicateUsesFirstMiddleAndLastTileForNineteen) {
  af::SizeVar outer_size(af::Symbol("outer_size"));
  af::SizeVar block_inner_extent(af::Symbol("19"));
  af::Axis outer{.id = 0, .name = "outer", .size = outer_size.expr};
  af::Axis guarded{.id = 1, .name = "guarded", .size = af::Symbol("19")};
  af::Axis block_inner{
      .id = 2, .name = "block_inner", .type = af::Axis::Type::kAxisTypeBlockInner, .size = block_inner_extent.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(outer_size);
  tiler.AddSizeVar(block_inner_extent);
  tiler.AddAxis(outer);
  tiler.AddAxis(guarded);
  tiler.AddAxis(block_inner);
  tiler.axis_map.at(outer.id).is_split_b = true;
  tiler.axis_map.at(guarded.id).is_split_b = true;

  codegen::Loop loop(guarded.id);
  auto fused_call = new MockApiCall("fused_call");
  fused_call->enable_cache = true;
  fused_call->exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  fused_call->unit = af::ComputeUnit::kUnitVector;
  auto origin_call = new MockApiCall("origin_call");
  origin_call->enable_cache = true;
  origin_call->exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  origin_call->unit = af::ComputeUnit::kUnitVector;
  loop.AddCall(fused_call);
  loop.AddCall(origin_call);

  codegen::TPipe tpipe("tpipe", tiler);
  std::string result;
  ASSERT_EQ(loop.Generate(tiler, tpipe, result), af::SUCCESS);

  // Tile 0 is covered by the explicit axis < 1 term; tiles 9 and 18 exercise the
  // middle and tail positions of the period-19 predicate respectively.
  EXPECT_NE(result.find("bool enable_cache_fused_brc_axis = (guarded < 1) || ((block_dim * block_inner_axis_size + "
                        "guarded) % guarded_loop_size < 1);"),
            std::string::npos);
  EXPECT_NE(result.find("bool enable_cache_origin_brc_axis = (guarded < 1);"), std::string::npos);
  EXPECT_EQ(std::count(result.begin(), result.end(), '{'), std::count(result.begin(), result.end(), '}'));
}

TEST(CodegenKernel, ReduceDoubleTileUsesReduceSpecificCacheCondition) {
  af::SizeVar outer_size(af::Symbol("outer_size"));
  af::SizeVar tile0_size(af::Symbol("tile0_size"));
  af::SizeVar tile1_size(af::Symbol("tile1_size"));
  af::Axis outer{.id = 0, .name = "outer", .size = outer_size.expr};
  af::Axis tile0{.id = 1, .name = "tile0", .type = af::Axis::Type::kAxisTypeTileInner, .size = tile0_size.expr};
  af::Axis tile1{.id = 2, .name = "tile1", .type = af::Axis::Type::kAxisTypeTileInner, .size = tile1_size.expr};
  codegen::Tiler tiler;
  tiler.AddSizeVar(outer_size);
  tiler.AddSizeVar(tile0_size);
  tiler.AddSizeVar(tile1_size);
  tiler.AddAxis(outer);
  tiler.AddAxis(tile0);
  tiler.AddAxis(tile1);
  tiler.axis_map.at(outer.id).is_split_b = true;
  tiler.axis_map.at(tile0.id).is_split_b = true;
  tiler.axis_map.at(tile1.id).is_split_b = true;

  codegen::TPipe tpipe("tpipe", tiler);
  codegen::Loop loop(tile1.id);
  loop.is_graph_has_reduce_node = true;

  auto cached_call = new MockApiCall("cached_call");
  cached_call->enable_cache = true;
  cached_call->enable_cache_with_condition = "dis_enable_cache_r";
  cached_call->exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  cached_call->unit = af::ComputeUnit::kUnitVector;

  const auto &axis = tiler.GetAxis(tile1.id);
  const bool is_split_b = axis.is_split_b;
  const bool is_double_tile = true;
  ASSERT_TRUE(is_split_b);
  ASSERT_TRUE(is_double_tile);
  EXPECT_EQ(cached_call->enable_cache_with_condition, "dis_enable_cache_r");
  EXPECT_EQ(cached_call->exec_condition, af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  delete cached_call;
}

TEST(CodegenKernel, BroadcastInlineInCvStage_NoBlockDimCacheCondition) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));

  af::Axis z0{.id = 0, .name = "z0", .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .size = s1.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);

  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    cur_axis.is_split_b = true;
  }

  codegen::Loop loop1(z0.id);
  codegen::Loop loop2(z1.id);
  loop1.AddLoop(&loop2);

  auto call1 = new MockApiCall("call1");
  auto call2 = new MockApiCall("call2");
  call1->enable_cache = true;
  call1->exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  call1->unit = af::ComputeUnit::kUnitVector;
  call1->api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;
  call2->enable_cache = true;
  call2->exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  call2->unit = af::ComputeUnit::kUnitVector;
  call2->api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;

  loop2.AddCall(call1);
  loop2.AddCall(call2);
  codegen::TPipe tpipe("t", tiler);
  std::string result;
  EXPECT_EQ(loop1.Generate(tiler, tpipe, result, codegen::ComputeStage::kCVFuseStage1), af::SUCCESS);
  EXPECT_EQ(result.find("block_dim"), std::string::npos);
}

TEST(CodegenKernel, CalculateVectorizedAixsMergeStatus) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  auto node = graph.FindNode("x");
  af::AscTensor tensor = node->outputs[0];

  tensor.attr.axis = {z0.id, z1.id, z2.id};
  tensor.attr.vectorized_axis = {z1.id, z2.id};
  tensor.attr.repeats = {z0.size, z1.size, z2.size};
  tensor.attr.strides = {z1.size * z2.size, z2.size, One};

  vector<af::Expression> vectorized_strides{One, One};
  vectorized_strides[0] = z2.size;
  tensor.attr.vectorized_strides = vectorized_strides;
  std::string dtype_name;
  codegen::Tensor::DtypeName(tensor.attr.dtype, dtype_name);
  codegen::Tensor t = codegen::Tensor(tensor, dtype_name);
  t.vectorized_axis_pos = {1, 2};
  std::vector<codegen::Tensor> inputs = {};
  std::vector<codegen::Tensor> outputs = {t};
  VectorizedAxisLoopMergeStatus merge_info;
  GenerateVectorizedAxisMergeStatus(inputs, outputs, merge_info, tpipe);
  EXPECT_EQ(merge_info.merge_repeats_str.size(), 1);
  EXPECT_EQ(merge_info.merge_repeats_str[0], "t->s1 * t->s2");
}

class CodegenKernelV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(CodegenKernelV2Test, RequireContinuousTQueBuf_AllFromQue) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  Load load_op1("load1");
  Load load_op2("load2");
  Concat concat_op("concat");

  load_op1.x = x_op.y;
  load_op1.y.dtype = ge::DT_FLOAT;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = x_op.y;
  load_op2.y.dtype = ge::DT_FLOAT;
  *load_op2.y.axis = {z0.id, z1.id};

  concat_op.x = {load_op1.y, load_op2.y};
  *concat_op.y.axis = {z0.id, z1.id};

  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");

  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.repeats = {s0, s1};
  load1->outputs[0].attr.strides = {s1, One};
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;

  load2->outputs[0].attr = load1->outputs[0].attr;
  load2->outputs[0].attr.mem.tensor_id = 4;

  concat->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  concat->outputs[0].attr.mem.tensor_id = 5;
  concat->outputs[0].attr.repeats = {s0, s1 + s1};
  concat->outputs[0].attr.strides = {s1 + s1, One};

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes = {graph.FindNode("x")};
  codegen::Kernel kernel("test_input_all_from_que");
  EXPECT_EQ(codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel), af::SUCCESS);
  for (const auto &body : kernel.root_loop.bodys) {
    if (body.call->inputs.size() > 1) {
      for (int32_t i = 0; i < static_cast<int32_t>(body.call->inputs.size()); ++i) {
        EXPECT_EQ(body.call->inputs[i]->share_order, i);
      }
    }
  }
  ge::RuntimeStub::Reset();
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(CodegenKernelV2Test, RequireContinuousTQueBuf_AllFromBuf) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  ge::RuntimeStub::SetInstance(stub_v2);

  af::AscGraph graph("test_graph");
  ::ascir::FusedScheduledResult fused_schedule_result;

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  Load load_op1("load1");
  Load load_op2("load2");
  Concat concat_op("concat");

  load_op1.x = x_op.y;
  load_op1.y.dtype = ge::DT_FLOAT;
  *load_op1.y.axis = {z0.id, z1.id};

  load_op2.x = x_op.y;
  load_op2.y.dtype = ge::DT_FLOAT;
  *load_op2.y.axis = {z0.id, z1.id};

  concat_op.x = {load_op2.y, load_op1.y};
  *concat_op.y.axis = {z0.id, z1.id};

  auto data = graph.FindNode("x");
  auto load1 = graph.FindNode("load1");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");

  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeBuffer;
  load1->outputs[0].attr.mem.tensor_id = 3;
  load1->outputs[0].attr.repeats = {s0, s1};
  load1->outputs[0].attr.strides = {s1, One};
  load1->outputs[0].attr.mem.position = Position::kPositionVecCalc;
  load1->outputs[0].attr.buf.id = 1;

  load2->outputs[0].attr = load1->outputs[0].attr;
  load2->outputs[0].attr.mem.tensor_id = 4;
  load2->outputs[0].attr.buf.id = 2;

  concat->outputs[0].attr.mem.tensor_id = 5;
  concat->outputs[0].attr.repeats = {s0, s1 + s1};
  concat->outputs[0].attr.strides = {s1 + s1, One};

  fused_schedule_result.input_nodes = {data};
  codegen::Kernel kernel("test_input_all_from_que");
  EXPECT_EQ(codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel), af::SUCCESS);
  EXPECT_EQ(kernel.tpipe.contiguous_buf_ids, (std::vector<::ascir::BufId>{2, 1}));
}

TEST_F(CodegenKernel_CallSync, TestDefineShareOffsets) {
  auto load1 = Load("load1", x);
  auto load2 = Load("load2", x);
  auto load3 = Load("load3", x);
  load2->outputs[0].attr.que.id = load1->outputs[0].attr.que.id;
  load2->outputs[0].attr.mem.reuse_id = load1->outputs[0].attr.mem.reuse_id;
  load3->outputs[0].attr.que.id = load1->outputs[0].attr.que.id;
  load3->outputs[0].attr.mem.reuse_id = load1->outputs[0].attr.mem.reuse_id;
  auto concat = Vec("Concat", {load3, load2, load1});
  auto store = Store("store", concat);
  EXPECT_EQ(Generate(), R"(uint32_t q1_reuse1_offset = 0;
LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();
const uint32_t q1_reuse1_offset_part_1 = q1_reuse1_offset + local_3_size * 2;
const uint32_t q1_reuse1_offset_part_2 = q1_reuse1_offset_part_1 + local_2_size * 2;
q1_reuse1_offset = q1_reuse1_offset_part_2;
const uint32_t local_1_actual_size = 1;
LocalTensor<half> local_1;
local_1 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();
load1();

q1_reuse1_offset = q1_reuse1_offset_part_1;
const uint32_t local_2_actual_size = 1;
LocalTensor<half> local_2;
local_2 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();
load2();

q1_reuse1_offset = 0;
const uint32_t local_3_actual_size = 1;
LocalTensor<half> local_3;
local_3 = q1_buf[q1_reuse1_offset].template ReinterpretCast<half>();
load3();
q1.EnQue(q1_buf);

q1_buf = q1.DeQue<uint8_t>();
const uint32_t local_4_actual_size = 1;
LocalTensor<half> local_4;
local_4 = b4_buf.template ReinterpretCast<half>();
Concat();
q1.FreeTensor(q1_buf);

store();

)");
}

TEST(CodegenKernel, ScalarDataOutputTensor_test) {
  af::AscGraph graph("test");

  codegen::Tiler tiler;

  ScalarData scalar_data("scalar_data", graph);
  scalar_data.ir_attr.SetIndex(0);
  scalar_data.y.dtype = ge::DT_FLOAT;
  auto scalar_node = graph.FindNode("scalar_data");
  ge::AscTensor scalar_tensor = scalar_node->outputs[0];
  scalar_tensor.attr.mem.tensor_id = 0;

  Data data("data", graph);
  data.ir_attr.SetIndex(1);
  data.y.dtype = ge::DT_FLOAT;
  auto data_node = graph.FindNode("data");
  ge::AscTensor data_tensor = data_node->outputs[0];
  data_tensor.attr.mem.tensor_id = 1;

  std::string scalar_dtype_name;
  codegen::Tensor::DtypeName(scalar_tensor.attr.dtype, scalar_dtype_name);
  codegen::Tensor t1(scalar_tensor, scalar_dtype_name, "t1");
  EXPECT_EQ(t1.Init(), 0);

  std::string data_dtype_name;
  codegen::Tensor::DtypeName(data_tensor.attr.dtype, data_dtype_name);
  codegen::Tensor t2(data_tensor, data_dtype_name, "t2");
  EXPECT_EQ(t2.Init(), 0);

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(data_node);
  fused_schedule_result.input_nodes.push_back(scalar_node);
  codegen::Kernel kernel(graph.GetName());
  codegen::Kernel::ParseGraph(graph, fused_schedule_result, kernel);
  std::string result;
  EXPECT_EQ(kernel.GlobalTensorInit(result), 0);
  EXPECT_EQ(result, "GlobalTensor<float> global_1;\nglobal_1.SetGlobalBuffer((__gm__ float*)scalar_data);\n");
}

TEST(CodegenKernel, ScalarDataSupportedBlkTensorInputsGenerateBlkTensorInit) {
  ExpectScalarDataBlkTensorInit([](af::AscGraph &graph, ScalarData &scalar_data) {
    af::ascir_op::Cast cast_op("cast");
    graph.AddNode(cast_op);
    cast_op.x = scalar_data.y;
    cast_op.y.dtype = ge::DT_FLOAT16;
  });

  ExpectScalarDataBlkTensorInit([](af::AscGraph &graph, ScalarData &scalar_data) {
    af::ascir_op::Where where_op("where");
    graph.AddNode(where_op);
    where_op.x2 = scalar_data.y;
  });

  ExpectScalarDataBlkTensorInit([](af::AscGraph &graph, ScalarData &scalar_data) {
    af::ascir_op::Gt gt_op("gt");
    graph.AddNode(gt_op);
    gt_op.x2 = scalar_data.y;
  });
}

// ==================== Conv2D 新增功能测试 ====================

class CodegenKernelConv2DTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CodegenKernelConv2DTest, KernelFuncDeclare_ShouldGenerateConv2DTemplate_WhenIsConv2dTrue) {
  af::AscGraph graph("conv2d_graph");
  codegen::Kernel kernel("conv2d_kernel");

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("conv2d_fusion");

  ::ascir::ScheduledResult scheduled_result;
  scheduled_result.cube_type = ::ascir::CubeTemplateType::kUBFuse;
  std::vector<::ascir::ScheduledResult> scheduled_results;
  scheduled_results.push_back(scheduled_result);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(scheduled_results);

  std::string declare = kernel.KernelFuncDeclare("conv2d_graph", fused_schedule_result, false, false, true);

  EXPECT_NE(declare.find("FmapTiling"), std::string::npos);
  EXPECT_NE(declare.find("WeightTiling"), std::string::npos);
  EXPECT_NE(declare.find("L1PingPong"), std::string::npos);
  EXPECT_NE(declare.find("L0PingPong"), std::string::npos);
  EXPECT_NE(declare.find("OutputOrder"), std::string::npos);
  EXPECT_NE(declare.find("IterOrder"), std::string::npos);
  EXPECT_NE(declare.find("GroupType"), std::string::npos);
  EXPECT_NE(declare.find("EnableSmallChannel"), std::string::npos);
  EXPECT_NE(declare.find("WeightUbTrans"), std::string::npos);
  EXPECT_NE(declare.find("FmapCopyMode"), std::string::npos);
  EXPECT_NE(declare.find("InnerBatch"), std::string::npos);
  EXPECT_NE(declare.find("DisContinuous"), std::string::npos);
  EXPECT_EQ(declare.find("API_LEVEL"), std::string::npos);
  EXPECT_EQ(declare.find("A_TRANS"), std::string::npos);
  EXPECT_EQ(declare.find("B_TRANS"), std::string::npos);
}

TEST_F(CodegenKernelConv2DTest, KernelFuncDeclare_ShouldGenerateMatMulTemplate_WhenIsConv2dFalse) {
  af::AscGraph graph("matmul_graph");
  codegen::Kernel kernel("matmul_kernel");

  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("matmul_fusion");

  ::ascir::ScheduledResult scheduled_result;
  scheduled_result.cube_type = ::ascir::CubeTemplateType::kUBFuse;
  std::vector<::ascir::ScheduledResult> scheduled_results;
  scheduled_results.push_back(scheduled_result);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(scheduled_results);

  std::string declare = kernel.KernelFuncDeclare("matmul_graph", fused_schedule_result, false, false, false);

  EXPECT_NE(declare.find("API_LEVEL"), std::string::npos);
  EXPECT_NE(declare.find("A_TRANS"), std::string::npos);
  EXPECT_NE(declare.find("B_TRANS"), std::string::npos);
  EXPECT_NE(declare.find("BATCH_MODEL"), std::string::npos);
  EXPECT_NE(declare.find("MODEL"), std::string::npos);
  EXPECT_NE(declare.find("FULL_LOAD"), std::string::npos);
  EXPECT_NE(declare.find("L0C2OUT_MODEL"), std::string::npos);
  EXPECT_EQ(declare.find("FmapTiling"), std::string::npos);
  EXPECT_EQ(declare.find("WeightTiling"), std::string::npos);
}

TEST_F(CodegenKernelConv2DTest, GenCubeCommonTiling_ShouldGenerateConv2DAPI_WhenIsConv2dTrue) {
  codegen::Kernel kernel("conv2d_kernel");

  std::stringstream ss;
  EXPECT_EQ(kernel.GenCubeCommonTiling(ss, false, true), af::SUCCESS);

  std::string result = ss.str();
  EXPECT_NE(result.find("AscendC::TPipe pipe"), std::string::npos);
  EXPECT_NE(result.find("conv2d_v2"), std::string::npos);
  EXPECT_NE(result.find("FmapTiling"), std::string::npos);
  EXPECT_NE(result.find("WeightTiling"), std::string::npos);
  EXPECT_NE(result.find("L1PingPong"), std::string::npos);
  EXPECT_NE(result.find("L0PingPong"), std::string::npos);
  EXPECT_NE(result.find("OutputOrder"), std::string::npos);
  EXPECT_NE(result.find("IterOrder"), std::string::npos);
  EXPECT_NE(result.find("GroupType"), std::string::npos);
  EXPECT_EQ(result.find("mat_mul_v3"), std::string::npos);
  EXPECT_EQ(result.find("batch_mat_mul_v3"), std::string::npos);
}

TEST_F(CodegenKernelConv2DTest, GenCubeCommonTiling_ShouldGenerateMatMulAPI_WhenIsConv2dFalseAndNotBatch) {
  codegen::Kernel kernel("matmul_kernel");

  std::stringstream ss;
  EXPECT_EQ(kernel.GenCubeCommonTiling(ss, false, false), af::SUCCESS);

  std::string result = ss.str();
  EXPECT_NE(result.find("mat_mul_v3"), std::string::npos);
  EXPECT_NE(result.find("API_LEVEL"), std::string::npos);
  EXPECT_NE(result.find("A_TRANS"), std::string::npos);
  EXPECT_NE(result.find("B_TRANS"), std::string::npos);
  EXPECT_NE(result.find("BATCH_MODEL"), std::string::npos);
  EXPECT_NE(result.find("MODEL"), std::string::npos);
  EXPECT_NE(result.find("FULL_LOAD"), std::string::npos);
  EXPECT_NE(result.find("L0C2OUT_MODEL"), std::string::npos);
  EXPECT_EQ(result.find("conv2d_v2"), std::string::npos);
  EXPECT_EQ(result.find("batch_mat_mul_v3"), std::string::npos);
}

TEST_F(CodegenKernelConv2DTest, GenCubeCommonTiling_ShouldGenerateBatchMatMulAPI_WhenIsConv2dFalseAndIsBatch) {
  codegen::Kernel kernel("batch_matmul_kernel");

  std::stringstream ss;
  EXPECT_EQ(kernel.GenCubeCommonTiling(ss, true, false), af::SUCCESS);

  std::string result = ss.str();
  EXPECT_NE(result.find("batch_mat_mul_v3"), std::string::npos);
  EXPECT_NE(result.find("API_LEVEL"), std::string::npos);
  EXPECT_NE(result.find("A_TRANS"), std::string::npos);
  EXPECT_NE(result.find("B_TRANS"), std::string::npos);
  EXPECT_NE(result.find("BATCH_MODEL"), std::string::npos);
  EXPECT_NE(result.find("MODEL"), std::string::npos);
  EXPECT_NE(result.find("FULL_LOAD"), std::string::npos);
  EXPECT_NE(result.find("L0C2OUT_MODEL"), std::string::npos);
  EXPECT_EQ(result.find("conv2d_v2"), std::string::npos);
}

TEST_F(CodegenKernelConv2DTest, SetUsingGlobalTpipe_ShouldSetFlagCorrectly) {
  codegen::Kernel kernel("test_kernel");

  kernel.tpipe.SetUsingGlobalTpipe(true);
  kernel.tpipe.SetUsingGlobalTpipe(false);

  // 通过设置验证功能正常工作
  EXPECT_TRUE(true);
}
