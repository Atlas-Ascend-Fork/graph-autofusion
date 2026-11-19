/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "graph_construct_utils.h"

using namespace att;
using namespace af::sym;
class UTestAscendcApiPerf : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UTestAscendcApiPerf, case0) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::L0C;
  att::Expr dim0 = CreateExpr(20);
  att::Expr dim1 = CreateExpr(10);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::GM;
  dim0 = CreateExpr(20);
  dim1 = CreateExpr(10);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  std::string op_type = "CopyL0CToL2";
  NodeInfo node;
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIC_FIXPIPE];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(51200 , 387)");
}

TEST_F(UTestAscendcApiPerf, case1) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::L0C;
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::GM;
  dim0 = CreateExpr(20);
  dim1 = CreateExpr(10);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  std::string op_type = "T_LoadTscm";
  NodeInfo node;
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE2];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(56960 , 9)");
}

TEST_F(UTestAscendcApiPerf, case2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::L0C;
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::GM;
  dim0 = CreateExpr(20);
  dim1 = CreateExpr(100);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_LoadA";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE1];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(8656 , 99)");
}

TEST_F(UTestAscendcApiPerf, case3) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input1;
  input1.data_type = "float16";
  input1.data_type_size = 2U;
  input1.loc = att::HardwareDef::L0C;
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input1.dims.push_back(dim0);
  input1.dims.push_back(dim1);
  input_shapes.emplace_back(input1);

  att::TensorShapeInfo input2;
  input2.data_type = "float16";
  input2.data_type_size = 2U;
  input2.loc = att::HardwareDef::L0C;
  dim0 = CreateExpr(100);
  dim1 = CreateExpr(30);
  input2.dims.push_back(dim0);
  input2.dims.push_back(dim1);
  input_shapes.emplace_back(input2);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::GM;
  dim0 = CreateExpr(220);
  dim1 = CreateExpr(10);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_Mmad";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_CUBE];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(232875 , 4096)");
}

TEST_F(UTestAscendcApiPerf, case4) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::L0C;
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::GM;
  dim0 = CreateExpr(200);
  dim1 = CreateExpr(10);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "VectorCompute";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(4132 , 33)");
}

TEST_F(UTestAscendcApiPerf, case5) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.data_type_size = 2U;
  att::Expr dim0 = CreateExpr(10);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.data_type_size = 2U;
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  std::string op_type = "CopyL2ToL1";
  NodeInfo node;
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE2];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(51200 , 1089)");
}

TEST_F(UTestAscendcApiPerf, Abs) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  dim0 = CreateExpr(30);
  dim1 = CreateExpr(100);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Abs";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "43.1153006255627");
}

TEST_F(UTestAscendcApiPerf, Adds) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  dim0 = CreateExpr(30);
  dim1 = CreateExpr(100);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Adds";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "43.3937997296453");
}

TEST_F(UTestAscendcApiPerf, And) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(30);
  att::Expr dim1 = CreateExpr(100);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  dim0 = CreateExpr(30);
  dim1 = CreateExpr(100);
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "And";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49.2393007427454");
}

TEST_F(UTestAscendcApiPerf, Broadcast) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  att::Expr dim0 = af::Symbol(30, "a");
  att::Expr dim1 = af::Symbol(100, "b");
  output.dims.push_back(dim0);
  output.dims.push_back(dim1);
  output_shapes.emplace_back(output);

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.data_type = "float16";
  input.dims.push_back(CreateExpr(1));
  input.dims.push_back(dim1);
  input_shapes.emplace_back(input);

  std::vector<att::TensorShapeInfo> input_shapes2;
  att::TensorShapeInfo input2;
  input2.data_type_size = 2U;
  input2.data_type = "float16";
  input2.dims.push_back(dim0);
  input2.dims.push_back(CreateExpr(1));
  input_shapes2.emplace_back(input2);

  std::vector<att::TensorShapeInfo> input_shapes3;
  att::TensorShapeInfo input3;
  input3.data_type_size = 2U;
  input3.data_type = "float16";
  input3.dims.push_back(CreateExpr(1));
  input3.dims.push_back(CreateExpr(1));
  input_shapes3.emplace_back(input3);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "31.4819874202504");

  EXPECT_EQ(perf(input_shapes2, output_shapes, node, perf_res), af::SUCCESS);
  res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36.576176448704");

  EXPECT_EQ(perf(input_shapes3, output_shapes, node, perf_res), af::SUCCESS);
  res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "40.3992993682623");
}

TEST_F(UTestAscendcApiPerf, BroadcastFourDim) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  att::Expr dim0 = af::Symbol(32, "a");
  att::Expr dim1 = af::Symbol(64, "b");
  att::Expr dim2 = af::Symbol(128, "c");
  att::Expr dim3 = af::Symbol(256, "d");
  output.dims = {dim0, dim1, dim2, dim3};
  output_shapes.emplace_back(output);

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.data_type = "float16";
  input.dims = {CreateExpr(1), dim1, CreateExpr(1), dim3};
  input_shapes.emplace_back(input);

  std::vector<att::TensorShapeInfo> input_shapes2;
  att::TensorShapeInfo input2;
  input2.data_type_size = 2U;
  input2.data_type = "float16";
  input2.dims = {dim0, CreateExpr(1), dim2, CreateExpr(1)};
  input_shapes2.emplace_back(input2);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "41511.3727959352");

  EXPECT_EQ(perf(input_shapes2, output_shapes, node, perf_res), af::SUCCESS);
  res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "894553.699330963");
}

/**
 * CastApi测试用例：
 * 1. float16转float32的数据类型转换
 * 2. float32转float16的数据类型转换
 * 3. float16转uint8的数据类型转换
 * 4. 输入形状为空的边界情况
 * 5. 输出形状为空的边界情况
 * 6. 不支持的数类型转换情况
 * 7. 输出形状维度为空的边界情况
 */

// 测试float16到float32的数据类型转换
TEST_F(UTestAscendcApiPerf, TestCastFloat16ToFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状（float16类型）
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(256), CreateExpr(1024)};
  input.repeats = {CreateExpr(256), CreateExpr(1024)};
  input.strides = {CreateExpr(1024), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状（float32类型）
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(256), CreateExpr(1024)};
  output.repeats = {CreateExpr(256), CreateExpr(1024)};
  output.strides = {CreateExpr(1024), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0147f;
  const float h = 20.1204f;
  const int dim_product = 256 * 1024;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试float32到float16的数据类型转换
TEST_F(UTestAscendcApiPerf, TestCastFloat32ToFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(512), CreateExpr(512)};
  input.repeats = {CreateExpr(512), CreateExpr(512)};
  input.strides = {CreateExpr(512), CreateExpr(1)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(512), CreateExpr(512)};
  output.repeats = {CreateExpr(512), CreateExpr(512)};
  output.strides = {CreateExpr(512), CreateExpr(1)};
  output_shapes.push_back(output);

  const float k = 0.0087f;
  const float h = 20.4393f;
  const int dim_product = 512 * 512;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试float16到uint8的数据类型转换
TEST_F(UTestAscendcApiPerf, TestCastFloat16ToUint8) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input.repeats = {CreateExpr(100), CreateExpr(200)};
  input.strides = {CreateExpr(200), CreateExpr(1)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "uint8";
  output.dims = {CreateExpr(100), CreateExpr(200)};
  output.repeats = {CreateExpr(100), CreateExpr(200)};
  output.strides = {CreateExpr(200), CreateExpr(1)};
  output_shapes.push_back(output);

  const float k = 0.0083f;
  const float h = 19.9408f;
  const int dim_product = 100 * 200;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestCastEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试输出形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestCastEmptyOutputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(5)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数类型转换情况
TEST_F(UTestAscendcApiPerf, TestCastUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type = "float32";
  output_shapes[0].data_type_size = 4;
  input_shapes[0].dims = {CreateExpr(100)};
  input_shapes[0].repeats = {CreateExpr(100)};
  input_shapes[0].strides = {CreateExpr(1)};
  output_shapes[0].dims = {CreateExpr(100)};
  output_shapes[0].repeats = {CreateExpr(100)};
  output_shapes[0].strides = {CreateExpr(1)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试输出形状维度为空的边界情况
TEST_F(UTestAscendcApiPerf, TestCastEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  output_shapes[0].data_type = "float32";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Cast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CompareScalarEQ API
TEST_F(UTestAscendcApiPerf, TestCompareScalarEQ) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(256), CreateExpr(512)};
  input.repeats = {CreateExpr(256), CreateExpr(512)};
  input.strides = {CreateExpr(512), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(256), CreateExpr(512)};
  output.repeats = {CreateExpr(256), CreateExpr(512)};
  output.strides = {CreateExpr(512), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0084f;
  const float h = 21.9204f;
  const int dim_product = 256 * 512;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGE API
TEST_F(UTestAscendcApiPerf, TestCompareScalarGE) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input.repeats = {CreateExpr(128), CreateExpr(256)};
  input.strides = {CreateExpr(256), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output.repeats = {CreateExpr(128), CreateExpr(256)};
  output.strides = {CreateExpr(256), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0086f;
  const float h = 21.9025f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGT API
TEST_F(UTestAscendcApiPerf, TestCompareScalarGT) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0084f;
  const float h = 21.9200f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarLE API
TEST_F(UTestAscendcApiPerf, TestCompareScalarLE) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input.repeats = {CreateExpr(32), CreateExpr(64)};
  input.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output.repeats = {CreateExpr(32), CreateExpr(64)};
  output.strides = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0084f;
  const float h = 21.9210f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarLT) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input.repeats = {CreateExpr(128), CreateExpr(256)};
  input.strides = {CreateExpr(256), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output.repeats = {CreateExpr(128), CreateExpr(256)};
  output.strides = {CreateExpr(256), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "297.189291000366");
}

// 测试 CompareScalarNE API
TEST_F(UTestAscendcApiPerf, TestCompareScalarNE) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input.repeats = {CreateExpr(128), CreateExpr(256)};
  input.strides = {CreateExpr(256), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output.repeats = {CreateExpr(128), CreateExpr(256)};
  output.strides = {CreateExpr(256), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarNE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "297.162590026855");
}

TEST_F(UTestAscendcApiPerf, TestPowerTensorTensor) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input.repeats = {CreateExpr(128), CreateExpr(256)};
  input.strides = {CreateExpr(256), CreateExpr(1)};
  input_shapes.push_back(input);
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output.repeats = {CreateExpr(128), CreateExpr(256)};
  output.strides = {CreateExpr(256), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Power";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "24297.4187011719");
}

TEST_F(UTestAscendcApiPerf, TestPowerTensorScalar) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input.repeats = {CreateExpr(128), CreateExpr(256)};
  input.strides = {CreateExpr(256), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output.repeats = {CreateExpr(128), CreateExpr(256)};
  output.strides = {CreateExpr(256), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Power";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "23804.9104003906");
}

// 测试 Ascir Compare API
TEST_F(UTestAscendcApiPerf, TestCompareAscir) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(256), CreateExpr(512)};
  input.repeats = {CreateExpr(256), CreateExpr(512)};
  input.strides = {CreateExpr(512), CreateExpr(1)};
  input_shapes.push_back(input);
  input_shapes.push_back(input);
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(256), CreateExpr(512)};
  output.repeats = {CreateExpr(256), CreateExpr(512)};
  output.strides = {CreateExpr(512), CreateExpr(1)};
  output_shapes.push_back(output);
  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = kGe;
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kEq;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kNe;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kGt;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kLe;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kLt;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
}

TEST_F(UTestAscendcApiPerf, TestCompareInt64Ascir) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type = "int64";
  input.dims = {CreateExpr(256), CreateExpr(512)};
  input.repeats = {CreateExpr(256), CreateExpr(512)};
  input.strides = {CreateExpr(512), CreateExpr(1)};
  input_shapes.push_back(input);
  input_shapes.push_back(input);
  att::TensorShapeInfo output;
  output.data_type = "int64";
  output.dims = {CreateExpr(256), CreateExpr(512)};
  output.repeats = {CreateExpr(256), CreateExpr(512)};
  output.strides = {CreateExpr(512), CreateExpr(1)};
  output_shapes.push_back(output);
  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = kGe;
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "1336224.05372559");
  op_type = kEq;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kNe;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
  op_type = kGt;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "1336224.05372559");
  op_type = kLe;
  perf = GetPerfFunc(op_type);
  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "1336224.05372559");
}

// 测试边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestCompareScalarEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试边界情况 - 空输出
TEST_F(UTestAscendcApiPerf, TestCompareScalarEmptyOutput) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].data_type_size = 8;
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].data_type_size = 8;
  output_shapes[0].dims = {CreateExpr(32)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Div API
TEST_F(UTestAscendcApiPerf, TestDiv) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0460f;
  const float h = 29.1233f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(UTestAscendcApiPerf, TestUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Exp API
TEST_F(UTestAscendcApiPerf, TestExp) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0311f;
  const float h = 28.0144f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Exp";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 LogicalAnd API
TEST_F(UTestAscendcApiPerf, TestLogicalAnd) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(16), CreateExpr(32)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(16), CreateExpr(32)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalAnd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "58.3298370838165");
}

// 测试 LogicalOr API
TEST_F(UTestAscendcApiPerf, TestLogicalOr) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(8), CreateExpr(16)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(8), CreateExpr(16)};
  output_shapes.push_back(output);

  // 计算预期结果
  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalOr";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "88.1791579127312");
}

// 测试 MoveGmToUb API
TEST_F(UTestAscendcApiPerf, TestMoveGmToUbSmallblk) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(16), CreateExpr(32)};
  input.repeats = {CreateExpr(16), CreateExpr(32)};
  input.strides = {CreateExpr(32), CreateExpr(1)};
  input.gm_strides = {CreateExpr(32), CreateExpr(1)};
  input.loc = att::HardwareDef::GM;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(16), CreateExpr(32)};
  output.repeats = {CreateExpr(16), CreateExpr(32)};
  output.strides = {CreateExpr(32), CreateExpr(1)};
  output.gm_strides = {CreateExpr(32), CreateExpr(1)};
  output.loc = att::HardwareDef::UB;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 验证结果
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(7.90520000457764), Div(CreateExpr(7.30999994277954), blockdim));
  auto cycles = Add(Div(Mul(CreateExpr(16 * 32 * 2), CreateExpr(1)), T), CreateExpr(27.0100002288818));
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试 MoveGmToUb API
TEST_F(UTestAscendcApiPerf, TestMoveGmToUb) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(50), CreateExpr(1000)};
  input.repeats = {CreateExpr(50), CreateExpr(1000)};
  input.strides = {CreateExpr(1000), CreateExpr(1)};
  input.gm_strides = {CreateExpr(1000), CreateExpr(1)};
  input.loc = att::HardwareDef::GM;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(50), CreateExpr(1000)};
  output.repeats = {CreateExpr(50), CreateExpr(1000)};
  output.strides = {CreateExpr(1000), CreateExpr(1)};
  output.gm_strides = {CreateExpr(1000), CreateExpr(1)};
  output.loc = att::HardwareDef::UB;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 验证结果
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(9.90740013122559), Div(CreateExpr(15.8959999084473), blockdim));
  auto cycles = Add(Div(Mul(CreateExpr(1000 * 50 * 2), CreateExpr(1)), T), CreateExpr(27.0100002288818));
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试 MoveGmToUb 非连续搬运API
TEST_F(UTestAscendcApiPerf, TestMoveGmToUbStride) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(10), CreateExpr(20)};
  input.repeats = {CreateExpr(100), CreateExpr(400)};
  input.strides = {CreateExpr(400), CreateExpr(1)};
  input.gm_strides = {CreateExpr(800), CreateExpr(1)};
  input.loc = att::HardwareDef::GM;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(10), CreateExpr(20)};
  output.repeats = {CreateExpr(100), CreateExpr(200)};
  output.strides = {CreateExpr(400), CreateExpr(1)};
  output.gm_strides = {CreateExpr(800), CreateExpr(1)};
  output.loc = att::HardwareDef::UB;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 验证结果
  auto pad = Mul(Mod(CreateExpr(400), att::CreateExpr(256)), att::CreateExpr(0.07));
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(9.90740013122559), Div(CreateExpr(15.8959999084473), blockdim));
  auto cycles = Add(Add(Div(Mul(CreateExpr(1000 * 40), CreateExpr(1)), T), CreateExpr(632.930002851486)), pad);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试 MoveGmToUb 非连续搬运API，gm和ub非连续
TEST_F(UTestAscendcApiPerf, TestMoveGmToUbStride2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(10), CreateExpr(20)};
  input.repeats = {CreateExpr(200), CreateExpr(1)};
  input.origin_repeats = {CreateExpr(200), CreateExpr(1)};
  input.gm_strides = {CreateExpr(5), CreateExpr(1)};
  input.strides = {CreateExpr(5), CreateExpr(1)};
  input.loc = att::HardwareDef::GM;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(10), CreateExpr(20)};
  output.repeats = {CreateExpr(200), CreateExpr(1)};
  output.origin_repeats = {CreateExpr(200), CreateExpr(1)};
  output.gm_strides = {CreateExpr(5), CreateExpr(1)};
  output.strides = {CreateExpr(5), CreateExpr(1)};
  output.loc = att::HardwareDef::UB;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  EXPECT_EQ(Str(res), "((102400 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)");
}

// 测试 MoveUbToGm 非连续搬运API
TEST_F(UTestAscendcApiPerf, TestMoveUbToGmStride) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(10), CreateExpr(20)};
  input.repeats = {CreateExpr(100), CreateExpr(400)};
  input.strides = {CreateExpr(400), CreateExpr(1)};
  input.gm_strides = {CreateExpr(400), CreateExpr(1)};
  input.loc = att::HardwareDef::UB;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(10), CreateExpr(20)};
  output.repeats = {CreateExpr(100), CreateExpr(400)};
  output.strides = {CreateExpr(400), CreateExpr(1)};
  output.gm_strides = {CreateExpr(400), CreateExpr(1)};
  output.loc = att::HardwareDef::GM;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 验证结果
  auto pad = Mul(Mod(CreateExpr(400), att::CreateExpr(256)), att::CreateExpr(0.02));
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(9.96000003814697), Div(CreateExpr(3.78999996185303), blockdim));
  auto cycles = Add(Add(Div(Mul(CreateExpr(1000 * 40 * 2), CreateExpr(1)), T), CreateExpr(9.21000015258792)), pad);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestMoveEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试边界情况 - 空输出
TEST_F(UTestAscendcApiPerf, TestMoveEmptyOutput) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(UTestAscendcApiPerf, TestMoveUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].repeats = {CreateExpr(32)};
  input_shapes[0].strides = {CreateExpr(1)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].repeats = {CreateExpr(32)};
  output_shapes[0].strides = {CreateExpr(1)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Maximum API
TEST_F(UTestAscendcApiPerf, TestMaximum) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0111f;
  const float h = 20.1200f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maximum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Maxs API
TEST_F(UTestAscendcApiPerf, TestMaxs) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0071f;
  const float h = 20.0912f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maxs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Max API
TEST_F(UTestAscendcApiPerf, TestMax) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 21.0027f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Max";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Minimum API
TEST_F(UTestAscendcApiPerf, TestMinimum) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(256), CreateExpr(512)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(256), CreateExpr(512)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0111f;
  const float h = 20.1104f;
  const int dim_product = 256 * 512;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Minimum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mins API
TEST_F(UTestAscendcApiPerf, TestMins) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0071f;
  const float h = 20.0896f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Mins";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Min API
TEST_F(UTestAscendcApiPerf, TestMin) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 21.0056f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Min";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mul API
TEST_F(UTestAscendcApiPerf, TestMul) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0110f;
  const float h = 23.1243f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Mul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Muls API
TEST_F(UTestAscendcApiPerf, TestMuls) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0071f;
  const float h = 23.1006f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Muls";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Neg API
TEST_F(UTestAscendcApiPerf, TestNeg) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0071f;
  const float h = 23.1006f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Neg";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Reciprocal API
TEST_F(UTestAscendcApiPerf, TestReciprocal) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0078f;
  const float h = 21.0076f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Reciprocal";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Relu API
TEST_F(UTestAscendcApiPerf, TestRelu) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0077f;
  const float h = 20.0173f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Relu";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Rsqrt API
TEST_F(UTestAscendcApiPerf, TestRsqrt) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0071f;
  const float h = 21.0970f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Rsqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Select API
TEST_F(UTestAscendcApiPerf, TestSelectCase1) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input.repeats = {CreateExpr(32), CreateExpr(64)};
  input.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float32";
  input2.dims = {CreateExpr(32), CreateExpr(64)};
  input2.repeats = {CreateExpr(32), CreateExpr(64)};
  input2.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float32";
  input3.dims = {CreateExpr(32), CreateExpr(64)};
  input3.repeats = {CreateExpr(32), CreateExpr(64)};
  input3.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output.repeats = {CreateExpr(32), CreateExpr(64)};
  output.strides = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 2048, -9260586.41082808, 689197.283277584)");
}

// 测试 Select API
TEST_F(UTestAscendcApiPerf, TestSelectCase2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input.repeats = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input.strides = {CreateExpr(64 * 32), CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float16";
  input2.dims = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input2.repeats = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input2.strides = {CreateExpr(64 * 32), CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float16";
  input3.dims = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input3.repeats = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  input3.strides = {CreateExpr(64 * 32), CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  output.repeats = {CreateExpr(32), CreateExpr(32), CreateExpr(64)};
  output.strides = {CreateExpr(64 * 32), CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 65536, 16926703.4515522, 22081696.3273777)");
}

// 测试 SetVectorMask API
TEST_F(UTestAscendcApiPerf, TestSetVectorMask) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "SetVectorMask";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API
TEST_F(UTestAscendcApiPerf, TestSigmoid) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1011f;
  const float h = 116.0436f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sigmoid";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sign API
TEST_F(UTestAscendcApiPerf, TestSign) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0855f;
  const float h = 119.0821f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sign";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sqrt API
TEST_F(UTestAscendcApiPerf, TestSqrt) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0312f;
  const float h = 29.0056f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sub API
TEST_F(UTestAscendcApiPerf, TestSub) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0107f;
  const float h = 22.1226f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sub";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sum API
TEST_F(UTestAscendcApiPerf, TestSum) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 35.0021f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Tanh API
TEST_F(UTestAscendcApiPerf, TestTanh) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1976f;
  const float h = 181.6919f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Tanh";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMax API
TEST_F(UTestAscendcApiPerf, TestWholeReduceMax) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 21.0027f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMin API
TEST_F(UTestAscendcApiPerf, TestWholeReduceMin) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 21.0056f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceSum API
TEST_F(UTestAscendcApiPerf, TestWholeReduceSum) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0547f;
  const float h = 35.0021f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 ZerosLike API
TEST_F(UTestAscendcApiPerf, TestZerosLike) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float16";
  input2.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float16";
  input3.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0078f;
  const float h = 16.9993f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ZerosLike";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试Where API的WhereBase分支
TEST_F(UTestAscendcApiPerf, TestWhereBase) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input.repeats = {CreateExpr(32), CreateExpr(64)};
  input.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float16";
  input2.dims = {CreateExpr(32), CreateExpr(64)};
  input2.repeats = {CreateExpr(32), CreateExpr(64)};
  input2.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float16";
  input3.dims = {CreateExpr(32), CreateExpr(64)};
  input3.repeats = {CreateExpr(32), CreateExpr(64)};
  input3.strides = {CreateExpr(64), CreateExpr(1)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output.repeats = {CreateExpr(32), CreateExpr(64)};
  output.strides = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Where";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 2048, -9273412.97818479, 690200.90328796)");
}

// 测试Where API的WhereExtend分支
TEST_F(UTestAscendcApiPerf, TestWhereExtend) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input.repeats = {CreateExpr(32), CreateExpr(64)};
  input.strides = {CreateExpr(64), CreateExpr(0)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float16";
  input2.dims = {CreateExpr(32), CreateExpr(64)};
  input2.repeats = {CreateExpr(32), CreateExpr(64)};
  input2.strides = {CreateExpr(64), CreateExpr(0)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float16";
  input3.dims = {CreateExpr(32), CreateExpr(64)};
  input3.repeats = {CreateExpr(32), CreateExpr(64)};
  input3.strides = {CreateExpr(64), CreateExpr(0)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(32), CreateExpr(64)};
  output.repeats = {CreateExpr(32), CreateExpr(64)};
  output.strides = {CreateExpr(64), CreateExpr(0)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Where";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "3707.78800158992");
}

// 测试 Constant API
TEST_F(UTestAscendcApiPerf, TestConstant) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128), CreateExpr(256)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Constant";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 FlashSoftmax API
TEST_F(UTestAscendcApiPerf, TestFlashSoftmax) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float t = 0.89f;
  const float a = 6.14f;
  const float b = -5.60f;
  const float c = 0.06f;
  const float h = 44.39f;
  const int dim_product = 64 * 128;

  auto cycles = Div(CreateExpr(dim_product), CreateExpr(t));
  auto weight = Add(Div(CreateExpr(a), Add(CreateExpr(128), CreateExpr(b))), CreateExpr(c));
  cycles = Mul(cycles, weight);
  auto expected_value = Add(cycles, CreateExpr(h));

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "FlashSoftmax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 DropOut API
TEST_F(UTestAscendcApiPerf, TestDropOut) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float t = 50.05f;
  const float a = 0.0f;
  const float b = 0.0f;
  const float c = 0.99f;
  const float h = 74.52f;
  const int dim_product = 64 * 128;

  auto cycles = Div(CreateExpr(dim_product), CreateExpr(t));
  auto weight = Add(Div(CreateExpr(a), Add(CreateExpr(128), CreateExpr(b))), CreateExpr(c));
  cycles = Mul(cycles, weight);
  auto expected_value = Add(cycles, CreateExpr(h));

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Dropout";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 MatMul API
TEST_F(UTestAscendcApiPerf, TestMatMul) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状A
  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input_a);

  // 构造输入形状B
  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input_b);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(256)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "MatMul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Erf API
TEST_F(UTestAscendcApiPerf, TestErf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.6996f;
  const float h = 478.4175f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Erf";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BroadcastMiddle 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastMiddle) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(16, "a");
  att::Expr dim1 = af::Symbol(48, "b");
  att::Expr dim2 = af::Symbol(64, "c");

  // 构造输入形状 - 3维,中间维度为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, CreateExpr(1), dim2};
  input_shapes.push_back(input);

  // 构造输出形状 - 3维,中间维度被广播
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1, dim2};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "2113.36481142044");
}

// 测试 BroadcastMerge 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastMerge) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(16, "a");
  att::Expr dim1 = af::Symbol(2, "b");
  att::Expr dim2 = af::Symbol(64, "c");
  att::Expr dim3 = af::Symbol(4, "d");
  att::Expr dim4 = af::Symbol(32, "e");

  // 构造输入形状 - 3维,中间维度为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, dim1, CreateExpr(1), dim3, dim4};
  input_shapes.push_back(input);

  // 构造输出形状 - 3维,中间维度被广播
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1, dim2, dim3, dim4};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "7183.13282108307");
}

// 测试 BroadcastMerge2场景
TEST_F(UTestAscendcApiPerf, TestBroadcastMerge2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状 - 3维,中间维度为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(1000), CreateExpr("z0t_size"), CreateExpr(1000)};
  input_shapes.push_back(input);

  // 构造输出形状 - 3维,中间维度被广播
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1000), CreateExpr(2), CreateExpr("z0t_size"), CreateExpr(1000)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMerge3场景
TEST_F(UTestAscendcApiPerf, TestBroadcastMerge3) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状 - 3维,中间维度为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr("z0t_size"), CreateExpr(1000), CreateExpr(1000)};
  input_shapes.push_back(input);

  // 构造输出形状 - 3维,中间维度被广播
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr("z0t_size"), CreateExpr(2), CreateExpr(1000), CreateExpr(1000)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMiddle 边界情况 - 输入维度不足
TEST_F(UTestAscendcApiPerf, TestBroadcastMiddleInvalidDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = CreateExpr(32);
  att::Expr dim1 = CreateExpr(64);
  att::Expr dim2 = CreateExpr(128);

  // 构造输入形状 - 只有2维
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, dim2};
  input_shapes.push_back(input);

  // 构造输出形状 - 3维
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1, dim2};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_GE(result, af::SUCCESS);
}

// 测试 BroadcastMiddle 边界情况 - 不支持的数据类型
TEST_F(UTestAscendcApiPerf, TestBroadcastMiddleUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状 - int32类型
  att::TensorShapeInfo input;
  input.data_type = "int32";
  input.dims = {CreateExpr(32), CreateExpr(1), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "int32";
  output.dims = {CreateExpr(32), CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);
  input_shapes[0].data_type_size = 4;
  output_shapes[0].data_type_size = 4;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMiddle float32类型
TEST_F(UTestAscendcApiPerf, TestBroadcastMiddleFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(32, "a");
  att::Expr dim1 = af::Symbol(64, "b");
  att::Expr dim2 = af::Symbol(128, "c");

  // 构造输入形状 - float32类型
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {dim0, CreateExpr(1), dim2};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {dim0, dim1, dim2};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "7183.13282108307");
}

// 测试 Erf API - float32类型
TEST_F(UTestAscendcApiPerf, TestErfFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.6038f;
  const float h = 458.2933f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Erf";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BroadcastOuter 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastOuter) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");

  // 构造输入形状 - 第一维为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(1), dim1};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "67.1615731632008");
}

TEST_F(UTestAscendcApiPerf, TestBroadcastOuterFP32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");

  // 构造输入形状 - 第一维为1
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(1), dim1};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.021f;
  const float h = 43.966f;
  auto expected_value = output.dims[0] * (input.dims[1] * CreateExpr(k) + CreateExpr(h));

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "67.5690893734203");
}
// 测试 BroadcastInner 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastInner) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");

  // 构造输入形状 - 第二维为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.0291764239019");
}

// 测试 BroadcastInner 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastInnerWithStride) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");

  // 构造输入形状 - 第二维为4
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, CreateExpr(4)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.1671440434225");
}

// 测试 BroadcastInner2 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastInner2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");
  att::Expr dim2 = af::Symbol(256, "c");

  // 构造输入形状 - 第二维为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, dim1, CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1, dim2};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastInnerMerge 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastInnerMerge) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(4, "a");
  att::Expr dim1 = af::Symbol(16, "b");
  att::Expr dim2 = af::Symbol(128, "c");

  // 构造输入形状 - 第二维为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim0, dim1, CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1, dim2};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.0291764239019");
}

TEST_F(UTestAscendcApiPerf, TestBroadcastInnerFP32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(64, "a");
  att::Expr dim1 = af::Symbol(128, "b");

  // 构造输入形状 - 第二维为1
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {dim0, CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.3254064690833");
}

// 测试 BroadcastBoth 场景
TEST_F(UTestAscendcApiPerf, TestBroadcastBoth) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状 - 两个维度都为1
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(1), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "80.8968982696533");
}

// 测试 BroadcastBoth float32类型
TEST_F(UTestAscendcApiPerf, TestBroadcastBothFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状 - 两个维度都为1
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(1), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "144.791696548462");
}

// 测试 Erf 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestErfEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Erf";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Erf 不支持的数据类型
TEST_F(UTestAscendcApiPerf, TestErfUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int32";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int32";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 4;
  output_shapes[0].data_type_size = 4;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Erf";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LoadApi - bfloat16类型
TEST_F(UTestAscendcApiPerf, TestLoadApiBFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "bfloat16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input.gm_strides = {CreateExpr(128), CreateExpr(1)};
  input.loc = att::HardwareDef::GM;
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "bfloat16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output.gm_strides = {CreateExpr(128), CreateExpr(1)};
  output.loc = att::HardwareDef::UB;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(7.90520000457764), Div(CreateExpr(7.30999994277954), blockdim));
  auto cycles = Add(Div(Mul(CreateExpr(64 * 128 * 2), CreateExpr(1)), T), CreateExpr(27.0100002288818));
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试 StoreApi - uint8类型
TEST_F(UTestAscendcApiPerf, TestStoreApiUint8) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input.gm_strides = {CreateExpr(128), CreateExpr(1)};
  input.loc = att::HardwareDef::UB;
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "uint8";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output.gm_strides = {CreateExpr(128), CreateExpr(1)};
  output.loc = att::HardwareDef::GM;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(9.96000003814697), Div(CreateExpr(3.78999996185303), blockdim));
  auto cycles = Add(Div(Mul(CreateExpr(64 * 128 * 1), CreateExpr(1)), T), CreateExpr(12.0900001525879));
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试空维度情况
TEST_F(UTestAscendcApiPerf, TestEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Add";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不存在的操作类型
TEST_F(UTestAscendcApiPerf, TestInvalidOpType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(32)};

  std::string op_type = "InvalidOp";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf, nullptr);
}
// 测试 Abs API - float32类型
TEST_F(UTestAscendcApiPerf, TestAbsFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0147f;
  const float h = 20.0592f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Abs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Store API - float32类型
TEST_F(UTestAscendcApiPerf, TestStoreFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input.gm_strides = {CreateExpr(128), CreateExpr(1)};
  input.loc = att::HardwareDef::UB;
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output.gm_strides = {CreateExpr(128), CreateExpr(1)};
  output.loc = att::HardwareDef::GM;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  auto blockdim = CreateExpr("block_dim");
  auto T = Add(CreateExpr(9.96000003814697), Div(CreateExpr(3.78999996185303), blockdim));
  auto cycles = Add(Div(Mul(CreateExpr(64 * 128 * 4), CreateExpr(1)), T), CreateExpr(12.0900001525879));
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res), Str(cycles));
}

// 测试 Adds API - float32类型
TEST_F(UTestAscendcApiPerf, TestAddsFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0141f;
  const float h = 22.0936f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Adds";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Add API - float32类型
TEST_F(UTestAscendcApiPerf, TestAddFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0206f;
  const float h = 23.2225f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Add";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 And API - float32类型
TEST_F(UTestAscendcApiPerf, TestAndFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0112f;
  const float h = 17.5611f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "And";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Broadcast API - 1dim的float32类型
TEST_F(UTestAscendcApiPerf, TestBroadCastOneDimFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(1)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0156f;
  const float h = 16.9965f;
  const int dim_product = 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Duplicate API - float32类型
TEST_F(UTestAscendcApiPerf, TestDuplicateFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0156f;
  const float h = 16.9965f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Duplicate";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Exp API - float32类型
TEST_F(UTestAscendcApiPerf, TestExpFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0307f;
  const float h = 28.0376f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Exp";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Maxs API - float32类型
TEST_F(UTestAscendcApiPerf, TestMaxsFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0141f;
  const float h = 20.0887f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maxs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Max API - float32类型
TEST_F(UTestAscendcApiPerf, TestMaxFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1094f;
  const float h = 20.0051f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Max";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mins API - float32类型
TEST_F(UTestAscendcApiPerf, TestMinsFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0142f;
  const float h = 20.0876f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Mins";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Min API - float32类型
TEST_F(UTestAscendcApiPerf, TestMinFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1094f;
  const float h = 20.0068f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Min";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 LogicalAnd API - float32类型
TEST_F(UTestAscendcApiPerf, TestLogicalAndFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalAnd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "552.477502822876");
}

// 测试 LogicalOr API - float32类型
TEST_F(UTestAscendcApiPerf, TestLogicalOrFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalOr";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "544.283803939819");
}

// 测试 Maximum API - float32类型
TEST_F(UTestAscendcApiPerf, TestMaximumFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0215f;
  const float h = 20.1333f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maximum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Minimum API - float32类型
TEST_F(UTestAscendcApiPerf, TestMinimumFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0215f;
  const float h = 20.1271f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Minimum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Muls API - float32类型
TEST_F(UTestAscendcApiPerf, TestMulsFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0142f;
  const float h = 23.0966f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Muls";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mul API - float32类型
TEST_F(UTestAscendcApiPerf, TestMulFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0206f;
  const float h = 23.2291f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Mul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Neg API - float32类型
TEST_F(UTestAscendcApiPerf, TestNegFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0142f;
  const float h = 23.0966f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Neg";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Reciprocal API - float32类型
TEST_F(UTestAscendcApiPerf, TestReciprocalFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0146f;
  const float h = 21.0639f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Reciprocal";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Relu API - float32类型
TEST_F(UTestAscendcApiPerf, TestReluFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0154f;
  const float h = 20.0189f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Relu";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Rsqrt API - float32类型
TEST_F(UTestAscendcApiPerf, TestRsqrtFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0143f;
  const float h = 21.0979f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Rsqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 SetVectorMask API - float32类型
TEST_F(UTestAscendcApiPerf, TestSetVectorMaskFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "SetVectorMask";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API - float32类型
TEST_F(UTestAscendcApiPerf, TestSigmoidFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1256f;
  const float h = 115.9747f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sigmoid";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sign API - float32类型
TEST_F(UTestAscendcApiPerf, TestSignFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1701f;
  const float h = 119.0656f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sign";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sqrt API - float32类型
TEST_F(UTestAscendcApiPerf, TestSqrtFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0313f;
  const float h = 28.9961f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sub API - float32类型
TEST_F(UTestAscendcApiPerf, TestSubFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0213f;
  const float h = 22.1254f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sub";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Tanh API - float32类型
TEST_F(UTestAscendcApiPerf, TestTanhFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1570f;
  const float h = 153.9298f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Tanh";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Select API - float32类型
TEST_F(UTestAscendcApiPerf, TestSelectFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "uint8";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input_shapes.push_back(input);
  att::TensorShapeInfo input2;
  input2.data_type = "float32";
  input2.dims = {CreateExpr(64), CreateExpr(128)};
  input2.repeats = {CreateExpr(64), CreateExpr(128)};
  input2.strides = {CreateExpr(128), CreateExpr(1)};
  input_shapes.push_back(input2);
  att::TensorShapeInfo input3;
  input3.data_type = "float32";
  input3.dims = {CreateExpr(64), CreateExpr(128)};
  input3.repeats = {CreateExpr(64), CreateExpr(128)};
  input3.strides = {CreateExpr(128), CreateExpr(1)};
  input_shapes.push_back(input3);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 8192, -7731.84273529891, 4450.97347860783)");
}

// 测试 WholeReduceMax API - float32类型
TEST_F(UTestAscendcApiPerf, TestWholeReduceMaxFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1094f;
  const float h = 20.0051f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMin API - float32类型
TEST_F(UTestAscendcApiPerf, TestWholeReduceMinFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1094f;
  const float h = 20.0068f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceSum API - float32类型
TEST_F(UTestAscendcApiPerf, TestWholeReduceSumFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.1094f;
  const float h = 32.0029f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 ZerosLike API - float32类型
TEST_F(UTestAscendcApiPerf, TestZerosLikeFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0156f;
  const float h = 16.9965f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ZerosLike";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarEQ API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarEQFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0160f;
  const float h = 21.9749f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarEQ API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarEQUnsupported) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "int8";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.data_type_size = 1;
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "int8";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.data_type_size = 1;
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGE API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarGEFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0161f;
  const float h = 21.9643f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGT API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarGTFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0160f;
  const float h = 21.9712f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarLE API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarLEFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0161f;
  const float h = 21.9722f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarNE API - float32类型
TEST_F(UTestAscendcApiPerf, TestCompareScalarNEFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0161f;
  const float h = 22.9690f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarNE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 MatMul API 边界情况 - 输入数量不足
TEST_F(UTestAscendcApiPerf, TestMatMulInsufficientInputs) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 只添加一个输入
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(256)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "MatMul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 MatMul API 边界情况 - 维度不匹配
TEST_F(UTestAscendcApiPerf, TestMatMulMismatchedDimensions) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input_a);

  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(256), CreateExpr(512)};  // 维度不匹配
  input_shapes.push_back(input_b);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(512)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "MatMul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 MatMul API 边界情况 - 维度小于2
TEST_F(UTestAscendcApiPerf, TestMatMulInvalidDimCount) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(64)};  // 只有一个维度
  input_shapes.push_back(input_a);

  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input_b);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "MatMul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Copy API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestCopyEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CopyL0CToL2";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Copy API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestCopyEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CopyL0CToL2";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyGMtoL1 API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestCopyGMtoL1EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_LoadTscm";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyGMtoL1 API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestCopyGMtoL1EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_LoadTscm";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyFromL1 API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestCopyFromL1EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_LoadA";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyFromL1 API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestCopyFromL1EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_LoadA";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CubeCompute API 边界情况 - 输入数量不足
TEST_F(UTestAscendcApiPerf, TestCubeComputeInsufficientInputs) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_Mmad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CubeCompute API 边界情况 - 维度小于2
TEST_F(UTestAscendcApiPerf, TestCubeComputeInvalidDimCount) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(64)};  // 只有一个维度
  input_shapes.push_back(input_a);

  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input_b);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_Mmad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 VectorCompute API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestVectorComputeEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "VectorCompute";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 VectorCompute API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestVectorComputeEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "VectorCompute";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 DropoutCompute API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestDropoutComputeEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Dropout";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 DropoutCompute API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestDropoutComputeEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Dropout";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 SoftmaxFlashV2 API 边界情况 - 空输入
TEST_F(UTestAscendcApiPerf, TestSoftmaxFlashV2EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "FlashSoftmax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 SoftmaxFlashV2 API 边界情况 - 空维度
TEST_F(UTestAscendcApiPerf, TestSoftmaxFlashV2EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "FlashSoftmax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 MatMul API - 高维矩阵乘法
TEST_F(UTestAscendcApiPerf, TestMatMulHighDimensions) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造4维输入形状A
  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input_a);

  // 构造4维输入形状B
  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input_b);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "MatMul";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Copy API - 不同硬件位置组合
TEST_F(UTestAscendcApiPerf, TestCopyDifferentLocations) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input.repeats = {CreateExpr(64), CreateExpr(128)};
  input.strides = {CreateExpr(128), CreateExpr(1)};
  input.gm_strides = {CreateExpr(128), CreateExpr(1)};
  input_shapes.push_back(input);

  // 构造输出形状
  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output.repeats = {CreateExpr(64), CreateExpr(128)};
  output.strides = {CreateExpr(128), CreateExpr(1)};
  output.gm_strides = {CreateExpr(128), CreateExpr(1)};
  output_shapes.push_back(output);

  // 测试所有可能的硬件位置组合
  std::vector<std::pair<att::HardwareDef, att::HardwareDef>> locations = {
      {att::HardwareDef::GM, att::HardwareDef::L1},  {att::HardwareDef::L1, att::HardwareDef::L0A},
      {att::HardwareDef::L1, att::HardwareDef::L0B}, {att::HardwareDef::L0C, att::HardwareDef::L2},
      {att::HardwareDef::L0C, att::HardwareDef::GM}, {att::HardwareDef::GM, att::HardwareDef::UB},
      {att::HardwareDef::UB, att::HardwareDef::GM}};

  std::vector<std::string> op_types = {"T_LoadTscm",     "T_LoadA", "T_LoadB", "CopyL0CToL2",
                                       "T_FixPipeTrans", "Load",    "Store"};

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (size_t i = 0; i < locations.size(); i++) {
    input.loc = locations[i].first;
    output.loc = locations[i].second;

    auto perf = GetPerfFunc(op_types[i]);
    node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_types[i], 1, 1);
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 VectorCompute API - 不同数据类型组合
TEST_F(UTestAscendcApiPerf, TestVectorComputeDifferentTypes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::string> data_types = {"float16", "float32", "int8", "int32"};

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (const auto &input_type : data_types) {
    for (const auto &output_type : data_types) {
      input_shapes.clear();
      output_shapes.clear();

      // 构造输入形状
      att::TensorShapeInfo input;
      input.data_type = input_type;
      input.dims = {CreateExpr(64), CreateExpr(128)};
      input_shapes.push_back(input);

      // 构造输出形状
      att::TensorShapeInfo output;
      output.data_type = output_type;
      output.dims = {CreateExpr(64), CreateExpr(128)};
      output_shapes.push_back(output);

      std::string op_type = "VectorCompute";
      auto perf = GetPerfFunc(op_type);
      node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
      auto result = perf(input_shapes, output_shapes, node, perf_res);

      if (input_type == "float16" || input_type == "float32") {
        EXPECT_EQ(result, af::SUCCESS);
      } else {
        EXPECT_EQ(result, af::SUCCESS);
      }
    }
  }
}

// 测试 CubeCompute API - 特殊维度组合
TEST_F(UTestAscendcApiPerf, TestCubeComputeSpecialDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 测试1维输入
  att::TensorShapeInfo input_a;
  input_a.data_type = "float16";
  input_a.dims = {CreateExpr(64)};
  input_shapes.push_back(input_a);

  att::TensorShapeInfo input_b;
  input_b.data_type = "float16";
  input_b.dims = {CreateExpr(64)};
  input_shapes.push_back(input_b);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "T_Mmad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 测试3维输入
  input_shapes.clear();
  output_shapes.clear();

  input_a.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32)};
  input_b.dims = {CreateExpr(8), CreateExpr(32), CreateExpr(64)};
  output.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(64)};

  input_shapes.push_back(input_a);
  input_shapes.push_back(input_b);
  output_shapes.push_back(output);

  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Broadcast API - 特殊维度组合
TEST_F(UTestAscendcApiPerf, TestBroadcastSpecialDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = af::Symbol(32, "a");
  att::Expr dim1 = af::Symbol(64, "b");

  // 测试1维到2维广播
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {dim1};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {dim0, dim1};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);

  // 测试3维到4维广播
  input_shapes.clear();
  output_shapes.clear();

  input.dims = {CreateExpr(1), CreateExpr(16), CreateExpr(32)};
  output.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)};

  input_shapes.push_back(input);
  output_shapes.push_back(output);

  result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Load/Store API - 不同数据类型大小
TEST_F(UTestAscendcApiPerf, TestLoadStoreDataTypeSizes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::pair<std::string, int>> type_sizes = {
      {"float16", 2}, {"float32", 4}, {"int8", 1}, {"int32", 4}, {"bfloat16", 2}};

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (const auto &type_size : type_sizes) {
    input_shapes.clear();
    output_shapes.clear();

    // 构造输入形状
    att::TensorShapeInfo input;
    input.data_type = type_size.first;
    input.data_type_size = type_size.second;
    input.dims = {CreateExpr(64), CreateExpr(128)};
    input.repeats = {CreateExpr(64), CreateExpr(128)};
    input.strides = {CreateExpr(128), CreateExpr(1)};
    input.gm_strides = {CreateExpr(128), CreateExpr(1)};
    input.loc = att::HardwareDef::GM;
    input_shapes.push_back(input);

    // 构造输出形状
    att::TensorShapeInfo output;
    output.data_type = type_size.first;
    output.data_type_size = type_size.second;
    output.dims = {CreateExpr(64), CreateExpr(128)};
    output.repeats = {CreateExpr(64), CreateExpr(128)};
    output.strides = {CreateExpr(128), CreateExpr(1)};
    output.gm_strides = {CreateExpr(128), CreateExpr(1)};
    output.loc = att::HardwareDef::UB;
    output_shapes.push_back(output);

    // 测试Load
    std::string op_type = "Load";
    auto perf = GetPerfFunc(op_type);
    node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    EXPECT_EQ(result, af::SUCCESS);

    // 测试Store
    input.loc = att::HardwareDef::UB;
    output.loc = att::HardwareDef::GM;

    op_type = "Store";
    perf = GetPerfFunc(op_type);
    node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    result = perf(input_shapes, output_shapes, node, perf_res);

    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 MatMul API - 不同数据类型组合
TEST_F(UTestAscendcApiPerf, TestMatMulDifferentTypes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::string> data_types = {"float16", "float32", "int8", "int32"};

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (const auto &type_a : data_types) {
    for (const auto &type_b : data_types) {
      input_shapes.clear();
      output_shapes.clear();

      // 构造输入形状A
      att::TensorShapeInfo input_a;
      input_a.data_type = type_a;
      input_a.dims = {CreateExpr(64), CreateExpr(128)};
      input_shapes.push_back(input_a);

      // 构造输入形状B
      att::TensorShapeInfo input_b;
      input_b.data_type = type_b;
      input_b.dims = {CreateExpr(128), CreateExpr(256)};
      input_shapes.push_back(input_b);

      // 构造输出形状
      att::TensorShapeInfo output;
      output.data_type = type_a;
      output.dims = {CreateExpr(64), CreateExpr(256)};
      output_shapes.push_back(output);

      std::string op_type = "MatMul";
      auto perf = GetPerfFunc(op_type);
      node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
      auto result = perf(input_shapes, output_shapes, node, perf_res);

      if (type_a == "float16" && type_b == "float16") {
        EXPECT_EQ(result, af::SUCCESS);
      } else {
        EXPECT_EQ(result, af::SUCCESS);
      }
    }
  }
}

// 测试 FlashSoftmax API - 不同维度组合
TEST_F(UTestAscendcApiPerf, TestFlashSoftmaxDifferentDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::vector<Expr>> dim_combinations = {
      {CreateExpr(64)},                                                // 1维
      {CreateExpr(32), CreateExpr(64)},                                // 2维
      {CreateExpr(16), CreateExpr(32), CreateExpr(64)},                // 3维
      {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)}  // 4维
  };

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (const auto &dims : dim_combinations) {
    input_shapes.clear();
    output_shapes.clear();

    // 构造输入形状
    att::TensorShapeInfo input;
    input.data_type = "float16";
    input.dims = dims;
    input_shapes.push_back(input);

    // 构造输出形状
    att::TensorShapeInfo output;
    output.data_type = "float16";
    output.dims = dims;
    output_shapes.push_back(output);

    std::string op_type = "FlashSoftmax";
    auto perf = GetPerfFunc(op_type);
    node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 DropOut API - 不同维度组合
TEST_F(UTestAscendcApiPerf, TestDropOutDifferentDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::vector<Expr>> dim_combinations = {
      {CreateExpr(64)},                                                // 1维
      {CreateExpr(32), CreateExpr(64)},                                // 2维
      {CreateExpr(16), CreateExpr(32), CreateExpr(64)},                // 3维
      {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)}  // 4维
  };

  PerfOutputInfo perf_res;
  NodeInfo node;
  for (const auto &dims : dim_combinations) {
    input_shapes.clear();
    output_shapes.clear();

    // 构造输入形状
    att::TensorShapeInfo input;
    input.data_type = "float16";
    input.dims = dims;
    input_shapes.push_back(input);

    // 构造输出形状
    att::TensorShapeInfo output;
    output.data_type = "float16";
    output.dims = dims;
    output_shapes.push_back(output);

    std::string op_type = "Dropout";
    auto perf = GetPerfFunc(op_type);
    node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 GetPerfFunc - 无效操作类型
TEST_F(UTestAscendcApiPerf, TestGetPerfFuncInvalidOp) {
  std::string op_type = "InvalidOp";
  auto perf = GetPerfFunc(op_type);
  EXPECT_EQ(perf, nullptr);
}

// 测试 LogicalOr API 边界条件
TEST_F(UTestAscendcApiPerf, TestLogicalOrEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalOr";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLogicalOrEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalOr";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLogicalOrUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalOr";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LogicalAnd API 边界条件
TEST_F(UTestAscendcApiPerf, TestLogicalAndEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalAnd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLogicalAndEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalAnd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLogicalAndUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalAnd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceSum API 边界条件
TEST_F(UTestAscendcApiPerf, TestWholeReduceSumEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceSumEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceSumUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceMin API 边界条件
TEST_F(UTestAscendcApiPerf, TestWholeReduceMinEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceMinEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceMinUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceMax API 边界条件
TEST_F(UTestAscendcApiPerf, TestWholeReduceMaxEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceMaxEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestWholeReduceMaxUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "WholeReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Select API 边界条件
TEST_F(UTestAscendcApiPerf, TestSelectEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSelectEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSelectUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Select";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Tanh API 边界条件
TEST_F(UTestAscendcApiPerf, TestTanhEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Tanh";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestTanhEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Tanh";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestTanhUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Tanh";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sub API 边界条件
TEST_F(UTestAscendcApiPerf, TestSubEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sub";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSubEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sub";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSubUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sub";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sqrt API 边界条件
TEST_F(UTestAscendcApiPerf, TestSqrtEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSqrtEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSqrtUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sign API 边界条件
TEST_F(UTestAscendcApiPerf, TestSignEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sign";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSignEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sign";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSignUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sign";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API 边界条件
TEST_F(UTestAscendcApiPerf, TestSigmoidEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sigmoid";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSigmoidEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sigmoid";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSigmoidUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Sigmoid";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 SetVectorMask API 边界条件
TEST_F(UTestAscendcApiPerf, TestSetVectorMaskEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "SetVectorMask";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSetVectorMaskEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "SetVectorMask";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestSetVectorMaskUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "SetVectorMask";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Rsqrt API 边界条件
TEST_F(UTestAscendcApiPerf, TestRsqrtEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Rsqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestRsqrtEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Rsqrt";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 BroadcastOuter API 边界条件
TEST_F(UTestAscendcApiPerf, TestBroadcastOuterEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastOuterEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastOuterUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(1), CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LoadApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestLoadApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLoadApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLoadApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].repeats = {CreateExpr((32))};
  input_shapes[0].strides = {CreateExpr((1))};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].repeats = {CreateExpr((32))};
  output_shapes[0].strides = {CreateExpr((1))};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestLoadApiForTypev1) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");

  input_shapes[0].data_type = "float32";
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  input_shapes[0].gm_strides = {z0t_size * CreateExpr(64), CreateExpr(64), CreateExpr(1)};
  input_shapes[0].strides = {CreateExpr(64), CreateExpr(4096), CreateExpr(1)};
  output_shapes[0].data_type = "float32";
  output_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  output_shapes[0].gm_strides = {z0t_size * CreateExpr(64), CreateExpr(64), CreateExpr(1)};
  output_shapes[0].strides = {CreateExpr(64), CreateExpr(4096), CreateExpr(1)};
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Load";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  ExprExprMap replace_vars;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  for (const auto &pair : ret) {
    replace_vars[pair.first] = pair.second;
  }
  EXPECT_FALSE(replace_vars.empty());
  auto iter = replace_vars.find(res);
  EXPECT_TRUE(iter != replace_vars.end());
  EXPECT_EQ(Str(iter->second),
            "TernaryOp(z1t_size < z0t_size, (TernaryOp((256 * z0t_size) < 25000, TernaryOp(IsEqual(False, 0), "
            "((0.0700000002980232 * Mod(Abs((64 - (64 * z0t_size))), 256) * z0t_size) + (256 * z0t_size / "
            "(((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818), ((512 * z0t_size / "
            "(((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)), ((0.0700000002980232 * "
            "Mod(Abs((64 - (64 * z0t_size))), 256) * z0t_size) + (256 * z0t_size / (((15.8959999084473 / (block_dim)) "
            "+ 9.90740013122559))) + 27.0100002288818)) * z1t_size), (TernaryOp((256 * z1t_size) < 25000, "
            "TernaryOp(IsEqual(False, 0), ((256 * z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) "
            "+ 27.0100002288818), ((512 * z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + "
            "27.0100002288818)), ((256 * z1t_size / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + "
            "27.0100002288818)) * z0t_size))");
}

TEST_F(UTestAscendcApiPerf, TestLoadApiForTypev2) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z4t_size = CreateExpr("z4t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  // 连续 {true, false, true, true}
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7), af::sym::kSymbolOne};
  input_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34),
                                z4t_size * CreateExpr(7), CreateExpr(7), af::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7), af::sym::kSymbolOne};
  output_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34),
                                 z4t_size * CreateExpr(7), CreateExpr(7), af::sym::kSymbolOne};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  std::string op_type = "Load";
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  auto node_ptr = asc_graph.FindNode("load");
  EXPECT_FALSE(node_ptr == nullptr);
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 2, 3};
  node_ptr->outputs[0].attr.axis = {0, 1, 2, 3};
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(Str(res.Replace(ret)),
            "(7 * TernaryOp((1904 * z4t_size) < 25000, TernaryOp(IsEqual(False, 0), ((1904 * z4t_size / "
            "(((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818), ((17408 / "
            "(((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)), ((1904 * z4t_size / "
            "(((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818)) * z0z1t_size)");
}

TEST_F(UTestAscendcApiPerf, TestStoreApiForType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                                af::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  output_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                                 af::sym::kSymbolOne};

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_FALSE(ternary_ops.empty());
  auto iter = ternary_ops.find(res);
  auto replace_vars = ConcursiveReplaceVars(ternary_ops);
  EXPECT_TRUE(iter != ternary_ops.end());
  auto ret = iter->second;
  ret.Replace(replace_vars);
  EXPECT_EQ(ret.GetTernaryOpStr(),
            "TernaryOp(7 < z0z1t_size, (7 * TernaryOp(IsEqual(Mod((2 * z6t_size), 4), 0), ((0.0199999995529652 * "
            "Mod(Abs((33 * z6t_size)), 256) * z0z1t_size) + (272 * z0z1t_size * z6t_size / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879), TernaryOp(((34 * z6t_size) + -512) < 0, "
            "TernaryOp(((34 * z6t_size) + -4) < 0, (((((-2.20000004768372 - (0.101000003516674 * block_dim)) * 272 * "
            "z6t_size) + (8.89000034332275 * block_dim) + 96.2399978637695) * z0z1t_size) + 12.0900001525879), ((136.0 "
            "* z0z1t_size * z6t_size) + 1.29999995231628)), (((256 - ((512 / (((3.78999996185303 / (block_dim)) + "
            "9.96000003814697))) + 12.0900001525879)) * 1.0) + (272 * z0z1t_size * z6t_size / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879)))), (TernaryOp(IsEqual(Mod((2 * z6t_size), 4), 0), "
            "((1904 * z6t_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879), "
            "TernaryOp(((34 * z6t_size) + -512) < 0, TernaryOp(((34 * z6t_size) + -4) < 0, (((((-2.20000004768372 - "
            "(0.101000003516674 * block_dim)) * 272 * z6t_size) + (8.89000034332275 * block_dim) + 96.2399978637695) * "
            "7) + 12.0900001525879), ((952.0 * z6t_size) + 1.29999995231628)), (((256 - ((512 / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879)) * 1.0) + (1904 * z6t_size / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879))) * z0z1t_size))");
}

// 测试 StoreApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestStoreApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestStoreApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestStoreApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].repeats = {CreateExpr(32)};
  input_shapes[0].strides = {CreateExpr(1)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].repeats = {CreateExpr(32)};
  output_shapes[0].strides = {CreateExpr(1)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestStoreApiMisalignedSizeV1) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].repeats = {CreateExpr(6)};
  input_shapes[0].strides = {CreateExpr(0)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].repeats = {CreateExpr(6)};
  output_shapes[0].strides = {CreateExpr(0)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestStoreApiMisalignedSizeV2) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].repeats = {CreateExpr(70)};
  input_shapes[0].strides = {CreateExpr(0)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].repeats = {CreateExpr(70)};
  output_shapes[0].strides = {CreateExpr(0)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Store";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}
// 测试 AbsApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestAbsApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Abs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAbsApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Abs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAbsApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Abs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AddsApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestAddsApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Adds";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAddsApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Adds";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAddsApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Adds";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AddApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestAddApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Add";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAddApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Add";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAddApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Add";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AndApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestAndApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "And";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAndApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "And";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestAndApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "And";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastInner API 边界条件
TEST_F(UTestAscendcApiPerf, TestBroadcastInnerEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastInnerEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastInnerUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32), CreateExpr(1)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastBoth API 边界条件
TEST_F(UTestAscendcApiPerf, TestBroadcastBothEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastBothEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastBothUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(1), CreateExpr(1)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastApi API 边界条件
TEST_F(UTestAscendcApiPerf, TestBroadcastApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestBroadcastApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Broadcast";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CompareScalarEQ API 边界条件
TEST_F(UTestAscendcApiPerf, TestCompareScalarEQEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarEQEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarEQUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarEQ";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGE API 边界条件
TEST_F(UTestAscendcApiPerf, TestCompareScalarGEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarGEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarGEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGT API 边界条件
TEST_F(UTestAscendcApiPerf, TestCompareScalarGTEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarGTEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarGTUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarGT";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarLE API 边界条件
TEST_F(UTestAscendcApiPerf, TestCompareScalarLEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarLEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarLEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarLE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarNE API 边界条件
TEST_F(UTestAscendcApiPerf, TestCompareScalarNEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarNE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarNEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarNE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestCompareScalarNEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "CompareScalarNE";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Div API 边界条件
TEST_F(UTestAscendcApiPerf, TestDivEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestDivEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestDivUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Div";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Duplicate API 边界条件
TEST_F(UTestAscendcApiPerf, TestDuplicateEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Duplicate";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestDuplicateEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Duplicate";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestDuplicateUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Duplicate";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestErfEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Erf";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}
// 测试 Exp API 边界条件
TEST_F(UTestAscendcApiPerf, TestExpEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Exp";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestExpEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Exp";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestExpUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Exp";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Maxs API 边界条件
TEST_F(UTestAscendcApiPerf, TestMaxsEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maxs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestMaxsEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maxs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestMaxsUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Maxs";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Max API 边界条件
TEST_F(UTestAscendcApiPerf, TestMaxEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Max";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestMaxEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Max";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscendcApiPerf, TestMaxUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Max";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Mins API 边界条件
TEST_F(UTestAscendcApiPerf, TestMinsEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Mins";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试default api
TEST_F(UTestAscendcApiPerf, TestDefaultApi) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "uint8";
  output.dims = {CreateExpr(100), CreateExpr(200)};
  output_shapes.push_back(output);
  auto expected_value = CreateExpr(1);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "UnitMTE1";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE1];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试ReduceAny api
TEST_F(UTestAscendcApiPerf, TestReduceAnyPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceAny";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.478002905846");
}

// 测试ReduceMax api
TEST_F(UTestAscendcApiPerf, TestReduceMaxPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.478002905846");
}

// 测试ReduceAll api
TEST_F(UTestAscendcApiPerf, TestReduceAllPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceAll";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.362982153893");
}

// 测试ReduceMin api
TEST_F(UTestAscendcApiPerf, TestReduceMinPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.362982153893");
}

// 测试ReduceSum api
TEST_F(UTestAscendcApiPerf, TestReduceSumPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceSum";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "504.153908845037");
}

// 测试ReduceProd api
TEST_F(UTestAscendcApiPerf, TestReduceProdPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceProd";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "536.264096941799");
}

// 测试ReduceMean api
TEST_F(UTestAscendcApiPerf, TestReduceMeanPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "ReduceMean";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "527.964508138597");
}

// 测试BlockReduceMax api
TEST_F(UTestAscendcApiPerf, TestBlockReduceMax) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "BlockReduceMax";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "1112.00416696072");
}

// 测试BlockReduceMin api
TEST_F(UTestAscendcApiPerf, TestBlockReduceMin) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "BlockReduceMin";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "1112.00916612148");
}

// 测试LogicalNot api
TEST_F(UTestAscendcApiPerf, TestLogicalNot) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(200), CreateExpr(100)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(1), CreateExpr(100)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "LogicalNot";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "105.123966054297");
}

// 测试 Or API - uint16类型
TEST_F(UTestAscendcApiPerf, TestOrUint16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "uint16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "uint16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0132f;
  const float h = 12.4018f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Or";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestOrEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Or";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyUbtoUb API - float16类型
TEST_F(UTestAscendcApiPerf, TestCopyUbtoUbFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0076f;
  const float h = 11.6372f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = kUb2ub;
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CopyUbtoUb API - float32类型
TEST_F(UTestAscendcApiPerf, TestCopyUbtoUbFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0152f;
  const float h = 11.6372f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = kUb2ub;
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestCopyUbtoUbEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = kUb2ub;
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 BrcbAPI - float16类型
TEST_F(UTestAscendcApiPerf, TestBrcbFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0074f;
  const float h = 13.0572f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Brcb";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BrcbAPI - float32类型
TEST_F(UTestAscendcApiPerf, TestBrcb32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(128)};
  output_shapes.push_back(output);

  // 计算预期结果
  const float k = 0.0146f;
  const float h = 13.0732f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Brcb";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestBrcbEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Brcb";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 GatherAPI - float16类型
TEST_F(UTestAscendcApiPerf, TestGatherFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input1;
  input1.data_type = "float16";
  input1.dims = {CreateExpr(64), CreateExpr(128)};
  att::TensorShapeInfo input2;
  input2.data_type = "int64";
  input2.dims = {CreateExpr(128)};
  input_shapes.push_back(input1);
  input_shapes.push_back(input2);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(128)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Gather";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res1 = perf_res.pipe_res[PipeType::AIV_VEC];
  Expr res2 = perf_res.pipe_res[PipeType::AIV_MTE2];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res1) << std::endl;
  std::cout << Str(res2) << std::endl;
  EXPECT_EQ(Str(res1), "354.892294883728");
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res2), "((32768 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 54.0200004577637)");
}

// 测试 GatherAPI - float32类型
TEST_F(UTestAscendcApiPerf, TestGather32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input1;
  input1.data_type = "float32";
  input1.dims = {CreateExpr(64), CreateExpr(128)};
  att::TensorShapeInfo input2;
  input2.data_type = "int64";
  input2.dims = {CreateExpr(32)};
  input_shapes.push_back(input1);
  input_shapes.push_back(input2);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(32)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Gather";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res1 = perf_res.pipe_res[PipeType::AIV_VEC];
  Expr res2 = perf_res.pipe_res[PipeType::AIV_MTE2];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res1) << std::endl;
  std::cout << Str(res2) << std::endl;
  EXPECT_EQ(Str(res1), "333.16509604454");
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res2), "((65536 / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 54.0200004577637)");
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestGatherEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "Gather";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 RemovePadAPI - float16类型
TEST_F(UTestAscendcApiPerf, TestRemovePadFloat16) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(32)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(64), CreateExpr(31)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "RemovePad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "44.9745995998383");
}

// 测试 RemovePadAPI - float32类型
TEST_F(UTestAscendcApiPerf, TestRemovePad32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float32";
  input.dims = {CreateExpr(64), CreateExpr(32), CreateExpr(128)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float32";
  output.dims = {CreateExpr(64), CreateExpr(32), CreateExpr(120)};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "RemovePad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "7706.30890846252");
}

// 测试输入形状为空的边界情况
TEST_F(UTestAscendcApiPerf, TestRemovePadEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  std::string op_type = "RemovePad";
  auto perf = GetPerfFunc(op_type);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}
