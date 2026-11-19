/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "util/att_utils.h"

using namespace att;
using namespace ge::sym;
class TestAscendcApiPerf : public ::testing::Test {
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

namespace {
void CheckComparePerfContains(const std::string &op_type, const std::vector<att::TensorShapeInfo> &input_shapes,
                              const std::vector<att::TensorShapeInfo> &output_shapes, NodeInfo &node) {
  PerfOutputInfo perf_res;
  auto perf = GetPerfFunc(op_type);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_TRUE(std::string(perf_res.pipe_res[PipeType::AIV_VEC].Str().get()).find("compare_node") != std::string::npos);
}

void CheckComparePerfValue(const std::string &op_type, const std::vector<att::TensorShapeInfo> &input_shapes,
                           const std::vector<att::TensorShapeInfo> &output_shapes, NodeInfo &node,
                           const std::string &expected) {
  PerfOutputInfo perf_res;
  auto perf = GetPerfFunc(op_type);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), expected);
}

ge::AscNodePtr ConstructLoadOp() {
  GraphBuilder graph_builder("test");
  auto data = graph_builder.AddNode("data", "Data", 1, 1);
  auto load = graph_builder.AddNode("load", "Load", 1, 1);
  GE_ASSERT_SUCCESS(graph_builder.AddDataEdge(data, 0, load, 0));
  ge::AscGraph asc_graph("test");
  GE_ASSERT_SUCCESS(ge::AscGraphUtils::ConvertComputeGraphToAscGraph(graph_builder.GetGraph(), asc_graph));
  auto load_asc_node = asc_graph.FindNode("load");
  GE_ASSERT_NOTNULL(load_asc_node);
  load_asc_node->inputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  return load_asc_node;
}

ge::AscNodePtr ConstructStoreOp() {
  GraphBuilder graph_builder("test");
  auto store = graph_builder.AddNode("store", "Store", 1, 1);
  auto output = graph_builder.AddNode("output", "Output", 1, 0);
  graph_builder.AddDataEdge(store, 0, output, 0);
  ge::AscGraph asc_graph("test");
  GE_ASSERT_SUCCESS(ge::AscGraphUtils::ConvertComputeGraphToAscGraph(graph_builder.GetGraph(), asc_graph));
  auto store_asc_node = asc_graph.FindNode("store");
  GE_ASSERT_NOTNULL(store_asc_node);
  store_asc_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  return store_asc_node;
}
}  // namespace

TEST_F(TestAscendcApiPerf, case0) {
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
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIC_FIXPIPE];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(51200 , 387)");
}

TEST_F(TestAscendcApiPerf, case1) {
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
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE2];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(56960 , 9)");
}

TEST_F(TestAscendcApiPerf, case2) {
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
  std::string op_type = "T_LoadA";
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE1];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(8656 , 99)");
}

TEST_F(TestAscendcApiPerf, case3) {
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
  std::string op_type = "T_Mmad";
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_CUBE];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(232875 , 4096)");
}

TEST_F(TestAscendcApiPerf, case4) {
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
  std::string op_type = "VectorCompute";
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(4132 , 33)");
}

TEST_F(TestAscendcApiPerf, case5) {
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
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE2];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "Rational(51200 , 1089)");
}

TEST_F(TestAscendcApiPerf, Abs) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Abs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "43.1153006255627");
}

TEST_F(TestAscendcApiPerf, Adds) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Adds";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "43.3937997296453");
}

TEST_F(TestAscendcApiPerf, And) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "And";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49.2393007427454");
}

TEST_F(TestAscendcApiPerf, Broadcast) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  att::Expr dim0 = ge::Symbol(30, "a"), dim1 = ge::Symbol(100, "b");
  output.dims = {dim0, dim1};
  output_shapes.emplace_back(output);

  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.data_type = "float16";
  input.dims = {CreateExpr(1), dim1};
  input_shapes.emplace_back(input);

  att::TensorShapeInfo input2;
  input2.data_type_size = 2U;
  input2.data_type = "float16";
  input2.dims = {dim0, CreateExpr(1)};
  std::vector<att::TensorShapeInfo> input_shapes2 = {input2};

  att::TensorShapeInfo input3;
  input3.data_type_size = 2U;
  input3.data_type = "float16";
  input3.dims = {CreateExpr(1), CreateExpr(1)};
  std::vector<att::TensorShapeInfo> input_shapes3 = {input3};

  PerfOutputInfo perf_res;
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Broadcast", 1, 1);
  auto perf = GetPerfFunc("Broadcast");
  EXPECT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "31.4819874202504");

  EXPECT_EQ(perf(input_shapes2, output_shapes, node, perf_res), af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "36.576176448704");

  EXPECT_EQ(perf(input_shapes3, output_shapes, node, perf_res), af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_VEC]), "40.3992993682623");
}

TEST_F(TestAscendcApiPerf, BroadcastFourDim) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  att::Expr dim0 = ge::Symbol(32, "a");
  att::Expr dim1 = ge::Symbol(64, "b");
  att::Expr dim2 = ge::Symbol(128, "c");
  att::Expr dim3 = ge::Symbol(256, "d");
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
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
TEST_F(TestAscendcApiPerf, TestCastFloat16ToFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试float32到float16的数据类型转换
TEST_F(TestAscendcApiPerf, TestCastFloat32ToFloat16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试float16到uint8的数据类型转换
TEST_F(TestAscendcApiPerf, TestCastFloat16ToUint8) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestCastEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试输出形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestCastEmptyOutputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(5)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数类型转换情况
TEST_F(TestAscendcApiPerf, TestCastUnsupportedDataType) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试输出形状维度为空的边界情况
TEST_F(TestAscendcApiPerf, TestCastEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  output_shapes[0].data_type = "float32";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Cast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CompareScalarEQ API
TEST_F(TestAscendcApiPerf, TestCompareScalarEQ) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGE API
TEST_F(TestAscendcApiPerf, TestCompareScalarGE) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGT API
TEST_F(TestAscendcApiPerf, TestCompareScalarGT) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarLE API
TEST_F(TestAscendcApiPerf, TestCompareScalarLE) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

TEST_F(TestAscendcApiPerf, TestCompareScalarLT) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "297.189291000366");
}

// 测试 CompareScalarNE API
TEST_F(TestAscendcApiPerf, TestCompareScalarNE) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarNE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "297.162590026855");
}

TEST_F(TestAscendcApiPerf, TestPowerTensorTensor) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Power";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "24297.4187011719");
}

TEST_F(TestAscendcApiPerf, TestPowerTensorScalar) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Power";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "23804.9104003906");
}

TEST_F(TestAscendcApiPerf, TestCompareAscir) {
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

  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(kGe, 1, 1);
  CheckComparePerfContains(kGe, input_shapes, output_shapes, node);
  CheckComparePerfContains(kEq, input_shapes, output_shapes, node);
  CheckComparePerfContains(kNe, input_shapes, output_shapes, node);
  CheckComparePerfContains(kGt, input_shapes, output_shapes, node);
  CheckComparePerfContains(kLe, input_shapes, output_shapes, node);
  CheckComparePerfContains(kLt, input_shapes, output_shapes, node);
}

TEST_F(TestAscendcApiPerf, TestCompareInt64Ascir) {
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

  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp(kGe, 1, 1);
  CheckComparePerfValue(kGe, input_shapes, output_shapes, node, "1336224.05372559");
  CheckComparePerfContains(kEq, input_shapes, output_shapes, node);
  CheckComparePerfContains(kNe, input_shapes, output_shapes, node);
  CheckComparePerfValue(kGt, input_shapes, output_shapes, node, "1336224.05372559");
  CheckComparePerfValue(kLe, input_shapes, output_shapes, node, "1336224.05372559");
}

// 测试边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestCompareScalarEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试边界情况 - 空输出
TEST_F(TestAscendcApiPerf, TestCompareScalarEmptyOutput) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(TestAscendcApiPerf, TestCompareScalarUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].data_type_size = 8;
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].data_type_size = 8;
  output_shapes[0].dims = {CreateExpr(32)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Div API
TEST_F(TestAscendcApiPerf, TestDiv) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(TestAscendcApiPerf, TestUnsupportedDataType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Exp API
TEST_F(TestAscendcApiPerf, TestExp) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Exp";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 LogicalAnd API
TEST_F(TestAscendcApiPerf, TestLogicalAnd) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalAnd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "58.3298370838165");
}

// 测试 LogicalOr API
TEST_F(TestAscendcApiPerf, TestLogicalOr) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalOr";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "88.1791579127312");
}

// 测试 MoveGmToUb API
TEST_F(TestAscendcApiPerf, TestMoveGmToUbSmallblk) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Load";
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
TEST_F(TestAscendcApiPerf, TestMoveGmToUb) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Load";
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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

// 测试边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestMoveEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试边界情况 - 空输出
TEST_F(TestAscendcApiPerf, TestMoveEmptyOutput) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes;

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不支持的数据类型
TEST_F(TestAscendcApiPerf, TestMoveUnsupportedDataType) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Maximum API
TEST_F(TestAscendcApiPerf, TestMaximum) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maximum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Maxs API
TEST_F(TestAscendcApiPerf, TestMaxs) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maxs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Max API
TEST_F(TestAscendcApiPerf, TestMax) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Max";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Minimum API
TEST_F(TestAscendcApiPerf, TestMinimum) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Minimum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mins API
TEST_F(TestAscendcApiPerf, TestMins) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Mins";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Min API
TEST_F(TestAscendcApiPerf, TestMin) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Min";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mul API
TEST_F(TestAscendcApiPerf, TestMul) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Mul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Muls API
TEST_F(TestAscendcApiPerf, TestMuls) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Muls";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Neg API
TEST_F(TestAscendcApiPerf, TestNeg) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Neg";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Reciprocal API
TEST_F(TestAscendcApiPerf, TestReciprocal) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Reciprocal";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Relu API
TEST_F(TestAscendcApiPerf, TestRelu) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Relu";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Rsqrt API
TEST_F(TestAscendcApiPerf, TestRsqrt) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Rsqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Select API
TEST_F(TestAscendcApiPerf, TestSelectCase1) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 2048, -9260586.41082808, 689197.283277584)");
}

// 测试 Select API
TEST_F(TestAscendcApiPerf, TestSelectCase2) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 65536, 16926703.4515522, 22081696.3273777)");
}

// 测试 SetVectorMask API
TEST_F(TestAscendcApiPerf, TestSetVectorMask) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "SetVectorMask";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API
TEST_F(TestAscendcApiPerf, TestSigmoid) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sigmoid";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sign API
TEST_F(TestAscendcApiPerf, TestSign) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sign";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sqrt API
TEST_F(TestAscendcApiPerf, TestSqrt) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sub API
TEST_F(TestAscendcApiPerf, TestSub) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sub";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sum API
TEST_F(TestAscendcApiPerf, TestSum) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Tanh API
TEST_F(TestAscendcApiPerf, TestTanh) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Tanh";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMax API
TEST_F(TestAscendcApiPerf, TestWholeReduceMax) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMin API
TEST_F(TestAscendcApiPerf, TestWholeReduceMin) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceSum API
TEST_F(TestAscendcApiPerf, TestWholeReduceSum) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 ZerosLike API
TEST_F(TestAscendcApiPerf, TestZerosLike) {
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
  const float k = 0.0078f;
  const float h = 16.9993f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ZerosLike";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试Where API的WhereBase分支
TEST_F(TestAscendcApiPerf, TestWhereBase) {
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
  ge::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp("Where", 3, 1);
  auto perf = GetPerfFunc(node_ptr->GetType());
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 2048, -9273412.97818479, 690200.90328796)");
}

// 测试Where API的WhereExtend分支
TEST_F(TestAscendcApiPerf, TestWhereExtend) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Where";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "3707.78800158992");
}

// 测试 Constant API
TEST_F(TestAscendcApiPerf, TestConstant) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Constant";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 FlashSoftmax API
TEST_F(TestAscendcApiPerf, TestFlashSoftmax) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "FlashSoftmax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 DropOut API
TEST_F(TestAscendcApiPerf, TestDropOut) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Dropout";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 MatMul API
TEST_F(TestAscendcApiPerf, TestMatMul) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "MatMul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Erf API
TEST_F(TestAscendcApiPerf, TestErf) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Erf";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BroadcastMiddle 场景
TEST_F(TestAscendcApiPerf, TestBroadcastMiddle) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(16, "a");
  att::Expr dim1 = ge::Symbol(48, "b");
  att::Expr dim2 = ge::Symbol(64, "c");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "2113.36481142044");
}

// 测试 BroadcastMerge 场景
TEST_F(TestAscendcApiPerf, TestBroadcastMerge) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(16, "a");
  att::Expr dim1 = ge::Symbol(2, "b");
  att::Expr dim2 = ge::Symbol(64, "c");
  att::Expr dim3 = ge::Symbol(4, "d");
  att::Expr dim4 = ge::Symbol(32, "e");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "7183.13282108307");
}

// 测试 BroadcastMerge2场景
TEST_F(TestAscendcApiPerf, TestBroadcastMerge2) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMerge3场景
TEST_F(TestAscendcApiPerf, TestBroadcastMerge3) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMiddle 边界情况 - 输入维度不足
TEST_F(TestAscendcApiPerf, TestBroadcastMiddleInvalidDims) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_GE(result, af::SUCCESS);
}

// 测试 BroadcastMiddle 边界情况 - 不支持的数据类型
TEST_F(TestAscendcApiPerf, TestBroadcastMiddleUnsupportedType) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastMiddle float32类型
TEST_F(TestAscendcApiPerf, TestBroadcastMiddleFloat32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(32, "a");
  att::Expr dim1 = ge::Symbol(64, "b");
  att::Expr dim2 = ge::Symbol(128, "c");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "7183.13282108307");
}

// 测试 Erf API - float32类型
TEST_F(TestAscendcApiPerf, TestErfFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Erf";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BroadcastOuter 场景
TEST_F(TestAscendcApiPerf, TestBroadcastOuter) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "67.1615731632008");
}

TEST_F(TestAscendcApiPerf, TestBroadcastOuterFP32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "67.5690893734203");
}
// 测试 BroadcastInner 场景
TEST_F(TestAscendcApiPerf, TestBroadcastInner) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.0291764239019");
}

// 测试 BroadcastInner 场景
TEST_F(TestAscendcApiPerf, TestBroadcastInnerWithStride) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.1671440434225");
}

// 测试 BroadcastInner2 场景
TEST_F(TestAscendcApiPerf, TestBroadcastInner2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");
  att::Expr dim2 = ge::Symbol(256, "c");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastInnerMerge 场景
TEST_F(TestAscendcApiPerf, TestBroadcastInnerMerge) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(4, "a");
  att::Expr dim1 = ge::Symbol(16, "b");
  att::Expr dim2 = ge::Symbol(128, "c");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.0291764239019");
}

TEST_F(TestAscendcApiPerf, TestBroadcastInnerFP32) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(64, "a");
  att::Expr dim1 = ge::Symbol(128, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "78.3254064690833");
}

// 测试 BroadcastBoth 场景
TEST_F(TestAscendcApiPerf, TestBroadcastBoth) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "80.8968982696533");
}

// 测试 BroadcastBoth float32类型
TEST_F(TestAscendcApiPerf, TestBroadcastBothFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "144.791696548462");
}

// 测试 Erf 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestErfEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Erf";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Erf 不支持的数据类型
TEST_F(TestAscendcApiPerf, TestErfUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int32";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int32";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 4;
  output_shapes[0].data_type_size = 4;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Erf";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LoadApi - bfloat16类型
TEST_F(TestAscendcApiPerf, TestLoadApiBFloat16) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
TEST_F(TestAscendcApiPerf, TestStoreApiUint8) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
TEST_F(TestAscendcApiPerf, TestEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Add";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试不存在的操作类型
TEST_F(TestAscendcApiPerf, TestInvalidOpType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(32)};

  std::string op_type = "InvalidOp";
  auto node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf, nullptr);
}
// 测试 Abs API - float32类型
TEST_F(TestAscendcApiPerf, TestAbsFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Abs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Store API - float32类型
TEST_F(TestAscendcApiPerf, TestStoreFloat32) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
TEST_F(TestAscendcApiPerf, TestAddsFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Adds";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Add API - float32类型
TEST_F(TestAscendcApiPerf, TestAddFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Add";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 And API - float32类型
TEST_F(TestAscendcApiPerf, TestAndFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "And";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Broadcast API - 1dim的float32类型
TEST_F(TestAscendcApiPerf, TestBroadCastOneDimFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Duplicate API - float32类型
TEST_F(TestAscendcApiPerf, TestDuplicateFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Duplicate";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Exp API - float32类型
TEST_F(TestAscendcApiPerf, TestExpFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Exp";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Maxs API - float32类型
TEST_F(TestAscendcApiPerf, TestMaxsFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maxs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Max API - float32类型
TEST_F(TestAscendcApiPerf, TestMaxFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Max";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mins API - float32类型
TEST_F(TestAscendcApiPerf, TestMinsFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Mins";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Min API - float32类型
TEST_F(TestAscendcApiPerf, TestMinFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Min";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 LogicalAnd API - float32类型
TEST_F(TestAscendcApiPerf, TestLogicalAndFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalAnd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "552.477502822876");
}

// 测试 LogicalOr API - float32类型
TEST_F(TestAscendcApiPerf, TestLogicalOrFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalOr";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "544.283803939819");
}

// 测试 Maximum API - float32类型
TEST_F(TestAscendcApiPerf, TestMaximumFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maximum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Minimum API - float32类型
TEST_F(TestAscendcApiPerf, TestMinimumFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Minimum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Muls API - float32类型
TEST_F(TestAscendcApiPerf, TestMulsFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Muls";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Mul API - float32类型
TEST_F(TestAscendcApiPerf, TestMulFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Mul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Neg API - float32类型
TEST_F(TestAscendcApiPerf, TestNegFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Neg";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Reciprocal API - float32类型
TEST_F(TestAscendcApiPerf, TestReciprocalFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Reciprocal";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Relu API - float32类型
TEST_F(TestAscendcApiPerf, TestReluFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Relu";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Rsqrt API - float32类型
TEST_F(TestAscendcApiPerf, TestRsqrtFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Rsqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 SetVectorMask API - float32类型
TEST_F(TestAscendcApiPerf, TestSetVectorMaskFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "SetVectorMask";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API - float32类型
TEST_F(TestAscendcApiPerf, TestSigmoidFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sigmoid";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sign API - float32类型
TEST_F(TestAscendcApiPerf, TestSignFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sign";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sqrt API - float32类型
TEST_F(TestAscendcApiPerf, TestSqrtFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Sub API - float32类型
TEST_F(TestAscendcApiPerf, TestSubFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sub";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Tanh API - float32类型
TEST_F(TestAscendcApiPerf, TestTanhFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Tanh";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 Select API - float32类型
TEST_F(TestAscendcApiPerf, TestSelectFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "where_base_node");
  EXPECT_EQ(Str(res.Replace(ConcursiveReplaceVars(perf_res.ternary_ops))),
            "TernaryOp(16320 <= 8192, -7731.84273529891, 4450.97347860783)");
}

// 测试 WholeReduceMax API - float32类型
TEST_F(TestAscendcApiPerf, TestWholeReduceMaxFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceMin API - float32类型
TEST_F(TestAscendcApiPerf, TestWholeReduceMinFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 WholeReduceSum API - float32类型
TEST_F(TestAscendcApiPerf, TestWholeReduceSumFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 ZerosLike API - float32类型
TEST_F(TestAscendcApiPerf, TestZerosLikeFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "ZerosLike";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarEQ API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarEQFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarEQ API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarEQUnsupported) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGE API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarGEFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarGT API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarGTFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarLE API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarLEFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CompareScalarNE API - float32类型
TEST_F(TestAscendcApiPerf, TestCompareScalarNEFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarNE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 MatMul API 边界情况 - 输入数量不足
TEST_F(TestAscendcApiPerf, TestMatMulInsufficientInputs) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "MatMul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 MatMul API 边界情况 - 维度不匹配
TEST_F(TestAscendcApiPerf, TestMatMulMismatchedDimensions) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "MatMul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 MatMul API 边界情况 - 维度小于2
TEST_F(TestAscendcApiPerf, TestMatMulInvalidDimCount) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "MatMul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Copy API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestCopyEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CopyL0CToL2";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Copy API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestCopyEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CopyL0CToL2";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyGMtoL1 API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestCopyGMtoL1EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_LoadTscm";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyGMtoL1 API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestCopyGMtoL1EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_LoadTscm";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyFromL1 API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestCopyFromL1EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_LoadA";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyFromL1 API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestCopyFromL1EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_LoadA";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CubeCompute API 边界情况 - 输入数量不足
TEST_F(TestAscendcApiPerf, TestCubeComputeInsufficientInputs) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_Mmad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CubeCompute API 边界情况 - 维度小于2
TEST_F(TestAscendcApiPerf, TestCubeComputeInvalidDimCount) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_Mmad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 VectorCompute API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestVectorComputeEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "VectorCompute";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 VectorCompute API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestVectorComputeEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "VectorCompute";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 DropoutCompute API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestDropoutComputeEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Dropout";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 DropoutCompute API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestDropoutComputeEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Dropout";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 SoftmaxFlashV2 API 边界情况 - 空输入
TEST_F(TestAscendcApiPerf, TestSoftmaxFlashV2EmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "FlashSoftmax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 SoftmaxFlashV2 API 边界情况 - 空维度
TEST_F(TestAscendcApiPerf, TestSoftmaxFlashV2EmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "FlashSoftmax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 MatMul API - 高维矩阵乘法
TEST_F(TestAscendcApiPerf, TestMatMulHighDimensions) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "MatMul";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Copy API - 不同硬件位置组合
TEST_F(TestAscendcApiPerf, TestCopyDifferentLocations) {
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
  ge::AscNodePtr node_ptr;
  for (size_t i = 0; i < locations.size(); i++) {
    input.loc = locations[i].first;
    output.loc = locations[i].second;
    if (op_types[i] == "Load") {
      node_ptr = ConstructLoadOp();
    } else if (op_types[i] == "Store") {
      node_ptr = ConstructStoreOp();
    } else {
      node_ptr = GraphConstructUtils::ConstructSingleOp(op_types[i], 1, 1);
    }
    auto perf = GetPerfFunc(op_types[i]);
    NodeInfo node;
    node.node_ptr = node_ptr;
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    ;
    EXPECT_EQ(result, af::SUCCESS) << "Failed for location i=" << i;
  }
}

// 测试 VectorCompute API - 不同数据类型组合
TEST_F(TestAscendcApiPerf, TestVectorComputeDifferentTypes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::string> data_types = {"float16", "float32", "int8", "int32"};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
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
      node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
      auto perf = GetPerfFunc(op_type);
      NodeInfo node;
      node.node_ptr = node_ptr;
      auto result = perf(input_shapes, output_shapes, node, perf_res);
      ;

      if (input_type == "float16" || input_type == "float32") {
        EXPECT_EQ(result, af::SUCCESS);
      } else {
        EXPECT_EQ(result, af::SUCCESS);
      }
    }
  }
}

// 测试 CubeCompute API - 特殊维度组合
TEST_F(TestAscendcApiPerf, TestCubeComputeSpecialDims) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "T_Mmad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Broadcast API - 特殊维度组合
TEST_F(TestAscendcApiPerf, TestBroadcastSpecialDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::Expr dim0 = ge::Symbol(32, "a");
  att::Expr dim1 = ge::Symbol(64, "b");

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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);

  // 测试3维到4维广播
  input_shapes.clear();
  output_shapes.clear();

  input.dims = {CreateExpr(1), CreateExpr(16), CreateExpr(32)};
  output.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)};

  input_shapes.push_back(input);
  output_shapes.push_back(output);

  result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Load/Store API - 不同数据类型大小
TEST_F(TestAscendcApiPerf, TestLoadStoreDataTypeSizes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::pair<std::string, int>> type_sizes = {
      {"float16", 2}, {"float32", 4}, {"int8", 1}, {"int32", 4}, {"bfloat16", 2}};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
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
    node_ptr = ConstructLoadOp();
    auto perf = GetPerfFunc(op_type);
    NodeInfo node;
    node.node_ptr = node_ptr;
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    ;

    EXPECT_EQ(result, af::SUCCESS);

    // 测试Store
    input.loc = att::HardwareDef::UB;
    output.loc = att::HardwareDef::GM;

    op_type = "Store";
    node_ptr = ConstructStoreOp();
    perf = GetPerfFunc(op_type);
    result = perf(input_shapes, output_shapes, node, perf_res);
    ;

    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 MatMul API - 不同数据类型组合
TEST_F(TestAscendcApiPerf, TestMatMulDifferentTypes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::string> data_types = {"float16", "float32", "int8", "int32"};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
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
      node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
      auto perf = GetPerfFunc(op_type);
      NodeInfo node;
      node.node_ptr = node_ptr;
      auto result = perf(input_shapes, output_shapes, node, perf_res);
      ;

      if (type_a == "float16" && type_b == "float16") {
        EXPECT_EQ(result, af::SUCCESS);
      } else {
        EXPECT_EQ(result, af::SUCCESS);
      }
    }
  }
}

// 测试 FlashSoftmax API - 不同维度组合
TEST_F(TestAscendcApiPerf, TestFlashSoftmaxDifferentDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::vector<Expr>> dim_combinations = {
      {CreateExpr(64)},                                                // 1维
      {CreateExpr(32), CreateExpr(64)},                                // 2维
      {CreateExpr(16), CreateExpr(32), CreateExpr(64)},                // 3维
      {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)}  // 4维
  };

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
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
    node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    auto perf = GetPerfFunc(op_type);
    NodeInfo node;
    node.node_ptr = node_ptr;
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    ;
    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 DropOut API - 不同维度组合
TEST_F(TestAscendcApiPerf, TestDropOutDifferentDims) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  std::vector<std::vector<Expr>> dim_combinations = {
      {CreateExpr(64)},                                                // 1维
      {CreateExpr(32), CreateExpr(64)},                                // 2维
      {CreateExpr(16), CreateExpr(32), CreateExpr(64)},                // 3维
      {CreateExpr(8), CreateExpr(16), CreateExpr(32), CreateExpr(64)}  // 4维
  };

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
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
    node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
    auto perf = GetPerfFunc(op_type);
    NodeInfo node;
    node.node_ptr = node_ptr;
    auto result = perf(input_shapes, output_shapes, node, perf_res);
    ;
    EXPECT_EQ(result, af::SUCCESS);
  }
}

// 测试 GetPerfFunc - 无效操作类型
TEST_F(TestAscendcApiPerf, TestGetPerfFuncInvalidOp) {
  std::string op_type = "InvalidOp";
  auto node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  EXPECT_EQ(perf, nullptr);
}

// 测试 LogicalOr API 边界条件
TEST_F(TestAscendcApiPerf, TestLogicalOrEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalOr";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLogicalOrEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalOr";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLogicalOrUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalOr";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LogicalAnd API 边界条件
TEST_F(TestAscendcApiPerf, TestLogicalAndEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalAnd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLogicalAndEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalAnd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLogicalAndUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalAnd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceSum API 边界条件
TEST_F(TestAscendcApiPerf, TestWholeReduceSumEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceSumEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceSumUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceMin API 边界条件
TEST_F(TestAscendcApiPerf, TestWholeReduceMinEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceMinEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceMinUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 WholeReduceMax API 边界条件
TEST_F(TestAscendcApiPerf, TestWholeReduceMaxEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceMaxEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestWholeReduceMaxUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "WholeReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Select API 边界条件
TEST_F(TestAscendcApiPerf, TestSelectEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSelectEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSelectUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Select";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Tanh API 边界条件
TEST_F(TestAscendcApiPerf, TestTanhEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Tanh";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestTanhEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Tanh";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestTanhUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Tanh";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sub API 边界条件
TEST_F(TestAscendcApiPerf, TestSubEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sub";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSubEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sub";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSubUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sub";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sqrt API 边界条件
TEST_F(TestAscendcApiPerf, TestSqrtEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSqrtEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSqrtUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sign API 边界条件
TEST_F(TestAscendcApiPerf, TestSignEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sign";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSignEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sign";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSignUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sign";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Sigmoid API 边界条件
TEST_F(TestAscendcApiPerf, TestSigmoidEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sigmoid";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSigmoidEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sigmoid";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSigmoidUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Sigmoid";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 SetVectorMask API 边界条件
TEST_F(TestAscendcApiPerf, TestSetVectorMaskEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "SetVectorMask";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSetVectorMaskEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "SetVectorMask";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestSetVectorMaskUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "SetVectorMask";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Rsqrt API 边界条件
TEST_F(TestAscendcApiPerf, TestRsqrtEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Rsqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestRsqrtEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Rsqrt";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 BroadcastOuter API 边界条件
TEST_F(TestAscendcApiPerf, TestBroadcastOuterEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastOuterEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastOuterUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(1), CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 LoadApi API 边界条件
TEST_F(TestAscendcApiPerf, TestLoadApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLoadApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLoadApiUnsupportedType) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestLoadApiForTypev1) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");
  // z0,z1,z2轴均不连续，测试外抛场景
  input_shapes[0].data_type = "float32";
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  input_shapes[0].dims = input_shapes[0].repeats;
  input_shapes[0].gm_strides = {z1t_size * CreateExpr(65 * 2), CreateExpr(65), CreateExpr(1)};
  input_shapes[0].strides = input_shapes[0].gm_strides;
  output_shapes[0].data_type = "float32";
  output_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  output_shapes[0].dims = output_shapes[0].repeats;
  output_shapes[0].gm_strides = {z1t_size * CreateExpr(65 * 2), CreateExpr(65), CreateExpr(1)};
  output_shapes[0].strides = output_shapes[0].gm_strides;
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  ge::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  auto node_ptr = asc_graph.FindNode("load");
  EXPECT_FALSE(node_ptr == nullptr);
  EXPECT_TRUE(att::AttUtils::IsLoadNode(node_ptr.get()));
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
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
  const std::string is_small_block = "(4 * z0t_size) < 25000";
  const std::string perf1 =
      "((4 * z0t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)";
  const std::string perf2 = "((512 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)";
  const std::string perf3 =
      "((4 * z0t_size / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818)";
  const std::string always_true = "TernaryOp(IsEqual(False, 0), " + perf1 + ", " + perf2 + ")";
  const std::string expect_perf = "TernaryOp(" + is_small_block + ", " + always_true + ", " + perf3 + ")";
  EXPECT_EQ(Str(iter->second), expect_perf);
}

TEST_F(TestAscendcApiPerf, TestLoadApiForTypev2) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z4t_size = CreateExpr("z4t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  // 连续 {true, false, true, true}
  input_shapes[0].strides = {CreateExpr(7 * 40 * 34), CreateExpr(40 * 34), z4t_size, CreateExpr(7),
                             af::sym::kSymbolOne};
  input_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34),
                                z4t_size * CreateExpr(7), CreateExpr(7), af::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].strides = {CreateExpr(7 * 40 * 34), CreateExpr(40 * 34), z4t_size, CreateExpr(7),
                              af::sym::kSymbolOne};
  output_shapes[0].gm_strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34),
                                 z4t_size * CreateExpr(7), CreateExpr(7), af::sym::kSymbolOne};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructLoadOp();
  auto perf = GetPerfFunc("Load");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 外抛for循环
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(
      Str(res.Replace(ret)),
      "TernaryOp(34 < (7 * z0z1t_size), (34 * TernaryOp((392 * z0z1t_size * z4t_size) < 25000, "
      "TernaryOp(IsEqual(False, 0), ((0.490000002086163 * Mod(Abs((7 - (7 * z4t_size))), 256) * z0z1t_size) + (392 * "
      "z0z1t_size * z4t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818), "
      "((3584 * z0z1t_size / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 27.0100002288818)), "
      "((0.490000002086163 * Mod(Abs((7 - (7 * z4t_size))), 256) * z0z1t_size) + (392 * z0z1t_size * z4t_size / "
      "(((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 27.0100002288818))), (7 * TernaryOp((1904 * "
      "z4t_size) < 25000, TernaryOp(IsEqual(False, 0), ((1904 * z4t_size / (((7.30999994277954 / (block_dim)) + "
      "7.90520000457764))) + 27.0100002288818), ((17408 / (((7.30999994277954 / (block_dim)) + "
      "7.90520000457764))) + 27.0100002288818)), ((1904 * z4t_size / (((15.8959999084473 / (block_dim)) + "
      "9.90740013122559))) + 27.0100002288818)) * z0z1t_size))");
}

TEST_F(TestAscendcApiPerf, TestStoreApiForType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  // 连续 {true, false, true, true}
  input_shapes[0].strides = {CreateExpr(7 * 40) * z6t_size, CreateExpr(40) * z6t_size, z6t_size, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                                af::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  // 连续 {true, false, true, true}
  output_shapes[0].strides = {CreateExpr(7 * 40) * z6t_size, CreateExpr(40) * z6t_size, z6t_size, af::sym::kSymbolOne};
  output_shapes[0].gm_strides = {CreateExpr(7 * 34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                                 af::sym::kSymbolOne};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_FALSE(ternary_ops.empty());
  auto iter = ternary_ops.find(res);
  EXPECT_TRUE(iter != ternary_ops.end());
  EXPECT_EQ(iter->second.GetTernaryOpStr(),
            "TernaryOp(IsEqual(Mod((2 * z6t_size), 4), 0), ((1904 * z0z1t_size * z6t_size / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879), TernaryOp(((34 * z6t_size) + -512) < 0, "
            "TernaryOp(((34 * z6t_size) + -4) < 0, (((((-2.20000004768372 - (0.101000003516674 * block_dim)) * 272 * "
            "z6t_size) + (8.89000034332275 * block_dim) + 96.2399978637695) * 7 * z0z1t_size) + 12.0900001525879), "
            "((952.0 * z0z1t_size * z6t_size) + 1.29999995231628)), (((256 - ((512 / (((3.78999996185303 / "
            "(block_dim)) + 9.96000003814697))) + 12.0900001525879)) * 1.0) + (1904 * z0z1t_size * z6t_size / "
            "(((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879)))");
}

// 测试 StoreApi API 边界条件
TEST_F(TestAscendcApiPerf, TestStoreApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestStoreApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestStoreApiUnsupportedType) {
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
  ge::AscNodePtr node_ptr;
  node_ptr = ConstructStoreOp();
  auto perf = GetPerfFunc("Store");
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AbsApi API 边界条件
TEST_F(TestAscendcApiPerf, TestAbsApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Abs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAbsApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Abs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAbsApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Abs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AddsApi API 边界条件
TEST_F(TestAscendcApiPerf, TestAddsApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Adds";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAddsApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Adds";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAddsApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Adds";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AddApi API 边界条件
TEST_F(TestAscendcApiPerf, TestAddApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Add";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAddApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Add";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAddApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Add";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 AndApi API 边界条件
TEST_F(TestAscendcApiPerf, TestAndApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "And";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAndApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "And";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestAndApiUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "And";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastInner API 边界条件
TEST_F(TestAscendcApiPerf, TestBroadcastInnerEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastInnerEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastInnerUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32), CreateExpr(1)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastBoth API 边界条件
TEST_F(TestAscendcApiPerf, TestBroadcastBothEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastBothEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastBothUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(1), CreateExpr(1)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 BroadcastApi API 边界条件
TEST_F(TestAscendcApiPerf, TestBroadcastApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestBroadcastApiEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Broadcast";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CompareScalarEQ API 边界条件
TEST_F(TestAscendcApiPerf, TestCompareScalarEQEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarEQEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarEQUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarEQ";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGE API 边界条件
TEST_F(TestAscendcApiPerf, TestCompareScalarGEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarGEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarGEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarGT API 边界条件
TEST_F(TestAscendcApiPerf, TestCompareScalarGTEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarGTEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarGTUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarGT";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarLE API 边界条件
TEST_F(TestAscendcApiPerf, TestCompareScalarLEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarLEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarLEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarLE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 CompareScalarNE API 边界条件
TEST_F(TestAscendcApiPerf, TestCompareScalarNEEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarNE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarNEEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarNE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestCompareScalarNEUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "CompareScalarNE";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Div API 边界条件
TEST_F(TestAscendcApiPerf, TestDivEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestDivEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestDivUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Div";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Duplicate API 边界条件
TEST_F(TestAscendcApiPerf, TestDuplicateEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Duplicate";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestDuplicateEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Duplicate";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestDuplicateUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Duplicate";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestErfEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Erf";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}
// 测试 Exp API 边界条件
TEST_F(TestAscendcApiPerf, TestExpEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Exp";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestExpEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Exp";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestExpUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Exp";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Maxs API 边界条件
TEST_F(TestAscendcApiPerf, TestMaxsEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maxs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestMaxsEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maxs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestMaxsUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Maxs";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Max API 边界条件
TEST_F(TestAscendcApiPerf, TestMaxEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Max";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestMaxEmptyDims) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  // dims保持为空
  output_shapes[0].data_type = "float16";

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Max";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(TestAscendcApiPerf, TestMaxUnsupportedType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "int8";
  input_shapes[0].dims = {CreateExpr(32)};
  output_shapes[0].data_type = "int8";
  output_shapes[0].dims = {CreateExpr(32)};
  input_shapes[0].data_type_size = 1;
  output_shapes[0].data_type_size = 1;

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Max";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 Mins API 边界条件
TEST_F(TestAscendcApiPerf, TestMinsEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Mins";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试default api
TEST_F(TestAscendcApiPerf, TestDefaultApi) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "UnitMTE1";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AICORE_MTE1];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试ReduceAny api
TEST_F(TestAscendcApiPerf, TestReduceAnyPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output.strides = {CreateExpr(1), ge::sym::kSymbolZero};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceAny";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.478002905846");
}

// 测试ReduceMax api
TEST_F(TestAscendcApiPerf, TestReduceMaxPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output.strides = {CreateExpr(1), ge::sym::kSymbolZero};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.478002905846");
}

// 测试ReduceAll api
TEST_F(TestAscendcApiPerf, TestReduceAllPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output.strides = {CreateExpr(1), ge::sym::kSymbolZero};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceAll";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.362982153893");
}

// 测试ReduceMin api
TEST_F(TestAscendcApiPerf, TestReduceMinPerf) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(100), CreateExpr(200)};
  input_shapes.push_back(input);

  att::TensorShapeInfo output;
  output.data_type = "float16";
  output.dims = {CreateExpr(100), CreateExpr(1)};
  output.strides = {CreateExpr(1), ge::sym::kSymbolZero};
  output_shapes.push_back(output);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "612.362982153893");
}

// 测试ReduceSum api
TEST_F(TestAscendcApiPerf, TestReduceSumPerf) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceSum";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "504.153908845037");
}

// 测试ReduceProd api
TEST_F(TestAscendcApiPerf, TestReduceProdPerf) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceProd";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "536.264096941799");
}

// 测试ReduceMean api
TEST_F(TestAscendcApiPerf, TestReduceMeanPerf) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "ReduceMean";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "527.964508138597");
}

// 测试BlockReduceMax api
TEST_F(TestAscendcApiPerf, TestBlockReduceMax) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "BlockReduceMax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "1112.00416696072");
}

// 测试BlockReduceMin api
TEST_F(TestAscendcApiPerf, TestBlockReduceMin) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "BlockReduceMin";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "1112.00916612148");
}

// 测试LogicalNot api
TEST_F(TestAscendcApiPerf, TestLogicalNot) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "LogicalNot";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), "105.123966054297");
}

// 测试 Or API - uint16类型
TEST_F(TestAscendcApiPerf, TestOrUint16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Or";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestOrEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Or";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 CopyUbtoUb API - float16类型
TEST_F(TestAscendcApiPerf, TestCopyUbtoUbFloat16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = kUb2ub;
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 CopyUbtoUb API - float32类型
TEST_F(TestAscendcApiPerf, TestCopyUbtoUbFloat32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = kUb2ub;
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestCopyUbtoUbEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = kUb2ub;
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 BrcbAPI - float16类型
TEST_F(TestAscendcApiPerf, TestBrcbFloat16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Brcb";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试 BrcbAPI - float32类型
TEST_F(TestAscendcApiPerf, TestBrcb32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Brcb";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res), Str(expected_value));
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestBrcbEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Brcb";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 GatherAPI - float16类型
TEST_F(TestAscendcApiPerf, TestGatherFloat16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Gather";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res1 = perf_res.pipe_res[PipeType::AIV_VEC];
  Expr res2 = perf_res.pipe_res[PipeType::AIV_MTE2];
  EXPECT_EQ(result, af::SUCCESS);
  EXPECT_EQ(Str(res1), "354.892294883728");
  auto ternary_ops = perf_res.ternary_ops;
  EXPECT_TRUE(ternary_ops.empty());
  EXPECT_EQ(Str(res2), "((32768 / (((7.30999994277954 / (block_dim)) + 7.90520000457764))) + 54.0200004577637)");
}

// 测试 GatherAPI - float32类型
TEST_F(TestAscendcApiPerf, TestGather32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "Gather";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res1 = perf_res.pipe_res[PipeType::AIV_VEC];
  Expr res2 = perf_res.pipe_res[PipeType::AIV_MTE2];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res1) << std::endl;
  std::cout << Str(res2) << std::endl;
  EXPECT_EQ(Str(res1), "333.16509604454");
  EXPECT_EQ(Str(res2), "((65536 / (((15.8959999084473 / (block_dim)) + 9.90740013122559))) + 54.0200004577637)");
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestGatherEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Gather";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 RemovePadAPI - float16类型
TEST_F(TestAscendcApiPerf, TestRemovePadFloat16) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "RemovePad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "44.9745995998383");
}

// 测试 RemovePadAPI - float32类型
TEST_F(TestAscendcApiPerf, TestRemovePad32) {
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
  ge::AscNodePtr node_ptr;
  std::string op_type = "RemovePad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "7706.30890846252");
}

// 测试输入形状为空的边界情况
TEST_F(TestAscendcApiPerf, TestRemovePadEmptyInputShapes) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float32";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "RemovePad";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 Argmax API
TEST_F(TestAscendcApiPerf, TestArgmax) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(32), CreateExpr(64)};
  input_shapes.push_back(input);

  // 构造输出形状 - Argmax输出int64索引
  att::TensorShapeInfo output;
  output.data_type = "int64";
  output.dims = {CreateExpr(32), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果 - Argmax复用ReduceMaxPerf
  const float k = 0.0547f;
  const float h = 21.0027f;
  const int dim_product = 32 * 64;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "Argmax";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 ArgMaxMultiRPhase1 API
TEST_F(TestAscendcApiPerf, TestArgMaxMultiRPhase1) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(64), CreateExpr(128)};
  input_shapes.push_back(input);

  // 构造输出形状 - Phase1输出int64索引
  att::TensorShapeInfo output;
  output.data_type = "int64";
  output.dims = {CreateExpr(64), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果 - 复用ArgmaxPerf
  const float k = 0.0547f;
  const float h = 21.0027f;
  const int dim_product = 64 * 128;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ArgMaxMultiRPhase1";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
}

// 测试 ArgMaxMultiRPhase2 API
TEST_F(TestAscendcApiPerf, TestArgMaxMultiRPhase2) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;

  // 构造输入形状
  att::TensorShapeInfo input;
  input.data_type = "float16";
  input.dims = {CreateExpr(128), CreateExpr(256)};
  input_shapes.push_back(input);

  // 构造输出形状 - Phase2输出int64索引
  att::TensorShapeInfo output;
  output.data_type = "int64";
  output.dims = {CreateExpr(128), CreateExpr(1)};
  output_shapes.push_back(output);

  // 计算预期结果 - 复用ArgmaxPerf
  const float k = 0.0547f;
  const float h = 21.0027f;
  const int dim_product = 128 * 256;
  auto expected_value = CreateExpr(dim_product) * CreateExpr(k) + CreateExpr(h);

  PerfOutputInfo perf_res;
  ge::AscNodePtr node_ptr;
  std::string op_type = "ArgMaxMultiRPhase2";
  node_ptr = GraphConstructUtils::ConstructSingleOp(op_type, 1, 1);
  auto perf = GetPerfFunc(op_type);
  NodeInfo node;
  node.node_ptr = node_ptr;
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  ;
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(result, af::SUCCESS);
}
