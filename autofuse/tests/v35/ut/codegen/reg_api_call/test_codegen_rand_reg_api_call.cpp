/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"
#include "codegen_kernel.h"
#include "codegen_graph_check.h"
#include "common_utils.h"
#include "codegen_api_param/codegen_api_param.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "utils/api_call_factory.h"
#include "unary_output_api_call.h"
#include "rand_seed_offset_api_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace codegen {
namespace {

class RandGraphCheckTest : public testing::Test {
 protected:
  RandGraphCheckTest() {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  ~RandGraphCheckTest() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

// 创建 Rand 算子的测试图
void BuildRandGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0, ge::DataType dtype) {
  Rand rand_op("Rand");
  graph.AddNode(rand_op);

  *rand_op.y.axis = {z0.id};
  *rand_op.y.repeats = {s0};
  *rand_op.y.strides = {One};
}

void InitRandAttrs(af::AscGraph &graph, ge::DataType dtype, const af::Axis &z0) {
  auto rand = graph.FindNode("Rand");
  rand->attr.api.compute_type = af::ComputeType::kComputeElewise;
  rand->attr.api.type = af::ApiType::kAPITypeCompute;
  rand->attr.api.unit = af::ComputeUnit::kUnitVector;
  rand->attr.sched.loop_axis = z0.id;

  auto &rand_attr = rand->outputs[0].attr;
  rand_attr.vectorized_axis = {z0.id};
  rand_attr.vectorized_strides = {One};
  rand_attr.dtype = dtype;
  rand_attr.mem.position = af::Position::kPositionVecOut;
  rand_attr.mem.tensor_id = 0;
  rand_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  rand_attr.que.id = 1;
  rand_attr.opt.merge_scope = af::kIdNone;
}

// 创建 Randn 算子的测试图
void BuildRandnGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0, ge::DataType dtype) {
  Randn randn_op("Randn");
  graph.AddNode(randn_op);

  *randn_op.y.axis = {z0.id};
  *randn_op.y.repeats = {s0};
  *randn_op.y.strides = {One};
}

void InitRandnAttrs(af::AscGraph &graph, ge::DataType dtype, const af::Axis &z0) {
  auto randn = graph.FindNode("Randn");
  randn->attr.api.compute_type = af::ComputeType::kComputeElewise;
  randn->attr.api.type = af::ApiType::kAPITypeCompute;
  randn->attr.api.unit = af::ComputeUnit::kUnitVector;
  randn->attr.sched.loop_axis = z0.id;

  auto &randn_attr = randn->outputs[0].attr;
  randn_attr.vectorized_axis = {z0.id};
  randn_attr.vectorized_strides = {One};
  randn_attr.dtype = dtype;
  randn_attr.mem.position = af::Position::kPositionVecOut;
  randn_attr.mem.tensor_id = 0;
  randn_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  randn_attr.que.id = 1;
  randn_attr.opt.merge_scope = af::kIdNone;
}

void BuildSeededRandGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0,
                          const std::vector<ge::DataType> &input_dtypes, ge::DataType output_dtype,
                          const std::string &node_name) {
  af::ascir_op::Data seed("seed", graph);
  af::ascir_op::Data offset("offset", graph);
  graph.AddNode(seed);
  graph.AddNode(offset);
  *seed.y.axis = {z0.id};
  *seed.y.repeats = {s0};
  *seed.y.strides = {One};
  seed.y.dtype = input_dtypes[0];
  *offset.y.axis = {z0.id};
  *offset.y.repeats = {s0};
  *offset.y.strides = {One};
  offset.y.dtype = input_dtypes[1];

  af::ascir_op::Rand rand_op(node_name.c_str(), graph);
  rand_op.seed = seed.y;
  rand_op.offset = offset.y;
  graph.AddNode(rand_op);
  InitRandAttrs(graph, output_dtype, z0);
}

void BuildSeededRandnGraph(af::AscGraph &graph, const af::Expression &s0, const af::Axis &z0,
                           const std::vector<ge::DataType> &input_dtypes, ge::DataType output_dtype,
                           const std::string &node_name) {
  af::ascir_op::Data seed("seed", graph);
  af::ascir_op::Data offset("offset", graph);
  graph.AddNode(seed);
  graph.AddNode(offset);
  *seed.y.axis = {z0.id};
  *seed.y.repeats = {s0};
  *seed.y.strides = {One};
  seed.y.dtype = input_dtypes[0];
  *offset.y.axis = {z0.id};
  *offset.y.repeats = {s0};
  *offset.y.strides = {One};
  offset.y.dtype = input_dtypes[1];

  af::ascir_op::Randn randn_op(node_name.c_str(), graph);
  randn_op.seed = seed.y;
  randn_op.offset = offset.y;
  graph.AddNode(randn_op);
  InitRandnAttrs(graph, output_dtype, z0);
}

}  // namespace

// 测试 Rand Codegen 基本信息
TEST(CodegenKernel, TestRandCodegenBasicInfo) {
  af::AscGraph graph("test_rand_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");

  // 创建 Codegen 实例并测试基本信息
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;

  std::string api_name = codegen.GetApiName();
  EXPECT_EQ(api_name, "RandSeedOffset");

  // 测试 GetApiCallName
  std::string api_call_name = codegen.GetApiCallName();
  EXPECT_EQ(api_call_name, "RandSeedOffsetApiCall");

  // 测试 LoadApiHeaderFiles
  auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1);
  EXPECT_EQ(headers[0], "random_reg_base.h");

  // 测试 IncludeApiHeaderFiles（legacy 路径依赖 Philox 头文件）
  auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 2);
  EXPECT_EQ(include_headers[0], "simt_api/cpp/kernel_simt_intf.h");
  EXPECT_EQ(include_headers[1], "adv_api/math/philox.h");
}

// 测试 Randn Codegen 基本信息
TEST(CodegenKernel, TestRandnCodegenBasicInfo) {
  af::AscGraph graph("test_randn_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");

  // 创建 Codegen 实例并测试基本信息
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;

  std::string api_name = codegen.GetApiName();
  EXPECT_EQ(api_name, "RandSeedOffset");

  // 测试 GetApiCallName
  std::string api_call_name = codegen.GetApiCallName();
  EXPECT_EQ(api_call_name, "RandSeedOffsetApiCall");

  // 测试 LoadApiHeaderFiles
  auto headers = codegen.LoadApiHeaderFiles(false);
  EXPECT_EQ(headers.size(), 1);
  EXPECT_EQ(headers[0], "random_reg_base.h");

  // 测试 IncludeApiHeaderFiles（legacy 路径依赖 Philox 头文件）
  auto include_headers = codegen.IncludeApiHeaderFiles();
  EXPECT_EQ(include_headers.size(), 2);
  EXPECT_EQ(include_headers[0], "simt_api/cpp/kernel_simt_intf.h");
  EXPECT_EQ(include_headers[1], "adv_api/math/philox.h");
}

// 测试 Rand 不同 dtype 支持
TEST(CodegenKernel, TestRandDifferentDtypes) {
  std::vector<ge::DataType> supported_dtypes = {DT_UINT32, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16};

  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_rand_dtype_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildRandGraph(graph, s0, z0, dtype);
    InitRandAttrs(graph, dtype, z0);

    auto rand = graph.FindNode("Rand");
    EXPECT_EQ(rand->outputs[0].attr.dtype, dtype);
    EXPECT_TRUE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*rand));
  }
}

// 测试 Randn 不同 dtype 支持
TEST(CodegenKernel, TestRandnDifferentDtypes) {
  std::vector<ge::DataType> supported_dtypes = {DT_UINT32, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16};

  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_randn_dtype_graph");

    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildRandnGraph(graph, s0, z0, dtype);
    InitRandnAttrs(graph, dtype, z0);

    auto randn = graph.FindNode("Randn");
    EXPECT_EQ(randn->outputs[0].attr.dtype, dtype);
    EXPECT_TRUE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*randn));
  }
}

TEST(CodegenKernel, TestRandSeedOffsetRejectsUnsupportedDtypesAndInputs) {
  const std::vector<ge::DataType> supported_dtypes = {DT_UINT32, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16};
  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_rand_seeded_valid_dtypes_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandGraph(graph, s0, z0, {DT_INT64, DT_INT32}, dtype, "Rand");
    auto rand = graph.FindNode("Rand");
    EXPECT_TRUE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*rand));
  }

  {
    af::AscGraph graph("test_rand_seeded_fp64_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandGraph(graph, s0, z0, {DT_INT64, DT_INT32}, DT_DOUBLE, "Rand");
    auto rand = graph.FindNode("Rand");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*rand));
  }

  {
    af::AscGraph graph("test_rand_seeded_invalid_seed_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandGraph(graph, s0, z0, {DT_INT32, DT_INT32}, DT_FLOAT, "Rand");
    auto rand = graph.FindNode("Rand");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*rand));
  }

  {
    af::AscGraph graph("test_rand_seeded_invalid_offset_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandGraph(graph, s0, z0, {DT_INT64, DT_INT64}, DT_FLOAT, "Rand");
    auto rand = graph.FindNode("Rand");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*rand));
  }
}

TEST(CodegenKernel, TestRandnSeedOffsetRejectsUnsupportedDtypesAndInputs) {
  const std::vector<ge::DataType> supported_dtypes = {DT_UINT32, DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16};
  for (auto dtype : supported_dtypes) {
    af::AscGraph graph("test_randn_seeded_valid_dtypes_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandnGraph(graph, s0, z0, {DT_INT64, DT_INT32}, dtype, "Randn");
    auto randn = graph.FindNode("Randn");
    EXPECT_TRUE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*randn));
  }

  {
    af::AscGraph graph("test_randn_seeded_fp64_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandnGraph(graph, s0, z0, {DT_INT64, DT_INT32}, DT_DOUBLE, "Randn");
    auto randn = graph.FindNode("Randn");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*randn));
  }

  {
    af::AscGraph graph("test_randn_seeded_invalid_seed_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandnGraph(graph, s0, z0, {DT_FLOAT, DT_INT32}, DT_FLOAT, "Randn");
    auto randn = graph.FindNode("Randn");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*randn));
  }

  {
    af::AscGraph graph("test_randn_seeded_invalid_offset_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", s0);
    BuildSeededRandnGraph(graph, s0, z0, {DT_INT64, DT_FLOAT}, DT_FLOAT, "Randn");
    auto randn = graph.FindNode("Randn");
    EXPECT_FALSE(af::ascir::RandSeedOffsetAscIrCodegenImplV2().IsNodeValid(*randn));
  }
}

// 测试 Rand 生成的 API 调用字符串（简化版）
TEST(CodegenKernel, TestRandApiCallStringGeneration) {
  af::AscGraph graph("test_rand_api_call_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(rand->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  // 使用 UnaryOutputApiCall 来生成调用字符串
  codegen::UnaryOutputApiCall call("Rand");
  EXPECT_EQ(call.Init(rand), 0);

  std::string result;
  call.Generate(tpipe, {}, result);

  // 验证生成的字符串包含关键信息
  EXPECT_NE(result.find("Rand"), std::string::npos);
  EXPECT_NE(result.find("local_0"), std::string::npos);
  EXPECT_NE(result.find("local_0_actual_size"), std::string::npos);
}

TEST(CodegenKernel, TestRandSeedOffsetApiCallKeepsLegacyCall) {
  af::AscGraph graph("test_rand_legacy_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(rand->outputs[0]);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.axis_map.at(z0.id).type = af::Axis::Type::kAxisTypeTileInner;

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(rand), 0);
  std::string result;
  call.tmp_buf_id[-1] = 0;
  std::string output_dtype_name;
  Tensor::DtypeName(rand->outputs[0].attr.dtype, output_dtype_name);
  Tensor output_tensor(rand->outputs[0], output_dtype_name);
  EXPECT_EQ(call.Generate(tpipe, {}, {}, {output_tensor}, result), 0);
  const auto api_param = codegen::CodegenApiParam::GetNodeApiParam(rand);
  EXPECT_EQ(api_param->api_name, "RandSeedOffset");
  EXPECT_EQ(api_param->semantic_feature, std::string());
  EXPECT_TRUE(api_param->input_params.empty());
  ASSERT_EQ(api_param->output_params.size(), 1U);
  EXPECT_EQ(api_param->output_params[0].name, output_tensor.name);
  EXPECT_NE(result.find("Rand("), std::string::npos);
  EXPECT_EQ(result.find("RandSeedOffset("), std::string::npos);
}

TEST(CodegenKernel, TestRandSeedOffsetApiCallRejectsInvalidInputsAndOutputs) {
  af::AscGraph graph("test_rand_invalid_api_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");
  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(rand), 0);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string dtype_name;
  ASSERT_EQ(Tensor::DtypeName(rand->outputs[0].attr.dtype, dtype_name), 0);
  Tensor output_tensor(rand->outputs[0], dtype_name);
  Tensor duplicate_output(rand->outputs[0], dtype_name);
  std::string result;
  EXPECT_NE(call.Generate(tpipe, {}, {}, {}, result), 0);
  EXPECT_NE(call.Generate(tpipe, {}, {}, {output_tensor, duplicate_output}, result), 0);
}

TEST(CodegenKernel, TestRandSeedOffsetApiCallGeneratesSeededCall) {
  af::AscGraph graph("test_rand_seed_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  auto rand = graph.FindNode("Rand");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  std::string dtype_name;
  auto seed_output = af::ascir::cg::IndexExpr("seed", graph, af::Symbol(12345L));
  auto offset_output = af::ascir::cg::IndexExpr("offset", graph, af::Symbol(0L));
  seed_output.mem->tensor_id = 1;
  offset_output.mem->tensor_id = 2;
  af::AscTensor seed_tensor_attr = graph.FindNode("seed")->outputs[0];
  af::AscTensor offset_tensor_attr = graph.FindNode("offset")->outputs[0];
  af::AscTensor output_tensor_attr = rand->outputs[0];
  EXPECT_EQ(Tensor::DtypeName(seed_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor seed_tensor(seed_tensor_attr, dtype_name, af::Symbol(12345L));
  EXPECT_EQ(Tensor::DtypeName(offset_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor offset_tensor(offset_tensor_attr, dtype_name);
  EXPECT_EQ(Tensor::DtypeName(output_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor output_tensor(output_tensor_attr, dtype_name);
  tpipe.AddTensor(seed_tensor);
  tpipe.AddTensor(offset_tensor);
  tpipe.AddTensor(output_tensor);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.axis_map.at(z0.id).type = af::Axis::Type::kAxisTypeTileInner;

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(rand), 0);
  call.tmp_buf_id[-1] = 0;
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, {seed_tensor, offset_tensor}, {output_tensor}, result), 0);
  EXPECT_EQ(codegen::CodegenApiParam::GetNodeApiParam(rand)->semantic_feature, ascgen_utils::template_semantic::kSimt);
  EXPECT_NE(result.find("RandSeedOffset("), std::string::npos);
  EXPECT_NE(result.find(offset_tensor.Str() + "["), std::string::npos);
  EXPECT_NE(result.find("tmp_buf_0"), std::string::npos);
}

TEST(CodegenKernel, TestRandSeedOffsetApiCallUsesTensorOffset) {
  af::AscGraph graph("test_rand_seed_tensor_offset_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data offset_op("offset", graph);
  offset_op.y.dtype = DT_INT32;
  *offset_op.y.axis = {z0.id};
  *offset_op.y.repeats = {s0};
  *offset_op.y.strides = {One};
  auto &offset_attr = graph.FindNode("offset")->outputs[0].attr;
  offset_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  offset_attr.mem.position = af::Position::kPositionVecIn;
  offset_attr.mem.tensor_id = 2;
  offset_attr.que.id = 2;
  offset_attr.vectorized_axis = {z0.id};
  offset_attr.vectorized_strides = {One};
  auto seed_output = af::ascir::cg::IndexExpr("seed", graph, DT_INT64, {z0.id}, {s0}, {One}, af::Symbol(12345L));
  seed_output.mem->tensor_id = 1;
  (void)af::ascir::cg::Rand("Rand", graph, DT_FLOAT, {z0.id}, {s0}, {One}, seed_output, offset_op.y);

  auto rand = graph.FindNode("Rand");
  InitRandAttrs(graph, DT_FLOAT, z0);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  std::string dtype_name;
  af::AscTensor seed_tensor_attr = graph.FindNode("seed")->outputs[0];
  af::AscTensor offset_tensor_attr = graph.FindNode("offset")->outputs[0];
  EXPECT_EQ(Tensor::DtypeName(seed_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor seed_tensor(seed_tensor_attr, dtype_name, af::Symbol(12345L));
  EXPECT_EQ(Tensor::DtypeName(offset_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor offset_tensor(offset_tensor_attr, dtype_name);
  EXPECT_EQ(Tensor::DtypeName(rand->outputs[0].attr.dtype, dtype_name), 0);
  Tensor output_tensor(rand->outputs[0], dtype_name);
  tpipe.AddTensor(seed_tensor);
  tpipe.AddTensor(offset_tensor);
  tpipe.AddTensor(output_tensor);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(rand), 0);
  call.tmp_buf_id[-1] = 0;
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, {seed_tensor, offset_tensor}, {output_tensor}, result), 0);
  EXPECT_NE(result.find("RandSeedOffset("), std::string::npos);
  EXPECT_NE(result.find(offset_tensor.Str() + "["), std::string::npos);
  EXPECT_NE(result.find("tmp_buf_0"), std::string::npos);
}

TEST(CodegenKernel, TestRandnSeedOffsetApiCallUsesRandnApi) {
  af::AscGraph graph("test_randn_seed_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string dtype_name;
  af::ascir_op::Data seed("seed", graph);
  af::ascir_op::Data offset("offset", graph);
  graph.AddNode(seed);
  graph.AddNode(offset);
  *seed.y.axis = {z0.id};
  *seed.y.repeats = {s0};
  *seed.y.strides = {One};
  seed.y.dtype = DT_INT64;
  *offset.y.axis = {z0.id};
  *offset.y.repeats = {s0};
  *offset.y.strides = {One};
  offset.y.dtype = DT_INT32;

  af::AscTensor seed_tensor_attr = graph.FindNode("seed")->outputs[0];
  af::AscTensor offset_tensor_attr = graph.FindNode("offset")->outputs[0];
  af::AscTensor output_tensor_attr = randn->outputs[0];
  EXPECT_EQ(Tensor::DtypeName(seed_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor seed_tensor(seed_tensor_attr, dtype_name);
  seed_tensor.is_constant = true;
  EXPECT_EQ(Tensor::DtypeName(offset_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor offset_tensor(offset_tensor_attr, dtype_name);
  EXPECT_EQ(Tensor::DtypeName(output_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor output_tensor(output_tensor_attr, dtype_name);
  tpipe.AddTensor(seed_tensor);
  tpipe.AddTensor(offset_tensor);
  tpipe.AddTensor(output_tensor);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(randn), 0);
  call.tmp_buf_id[-1] = 0;
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, {seed_tensor, offset_tensor}, {output_tensor}, result), 0);
  EXPECT_NE(result.find("RandnSeedOffset("), std::string::npos);
}

TEST(CodegenKernel, TestRandnFloatSeedOffsetApiCallUsesRandnApi) {
  af::AscGraph graph("test_randn_float_seed_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_FLOAT);
  InitRandnAttrs(graph, DT_FLOAT, z0);

  auto randn = graph.FindNode("Randn");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string dtype_name;
  af::ascir_op::Data seed("seed", graph);
  af::ascir_op::Data offset("offset", graph);
  graph.AddNode(seed);
  graph.AddNode(offset);
  *seed.y.axis = {z0.id};
  *seed.y.repeats = {s0};
  *seed.y.strides = {One};
  seed.y.dtype = DT_INT64;
  *offset.y.axis = {z0.id};
  *offset.y.repeats = {s0};
  *offset.y.strides = {One};
  offset.y.dtype = DT_INT32;

  af::AscTensor seed_tensor_attr = graph.FindNode("seed")->outputs[0];
  af::AscTensor offset_tensor_attr = graph.FindNode("offset")->outputs[0];
  af::AscTensor output_tensor_attr = randn->outputs[0];
  EXPECT_EQ(Tensor::DtypeName(seed_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor seed_tensor(seed_tensor_attr, dtype_name);
  seed_tensor.is_constant = true;
  EXPECT_EQ(Tensor::DtypeName(offset_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor offset_tensor(offset_tensor_attr, dtype_name);
  EXPECT_EQ(Tensor::DtypeName(output_tensor_attr.attr.dtype, dtype_name), 0);
  Tensor output_tensor(output_tensor_attr, dtype_name);
  tpipe.AddTensor(seed_tensor);
  tpipe.AddTensor(offset_tensor);
  tpipe.AddTensor(output_tensor);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(randn), 0);
  call.tmp_buf_id[-1] = 0;
  std::string result;
  EXPECT_EQ(call.Generate(tpipe, {z0.id}, {seed_tensor, offset_tensor}, {output_tensor}, result), 0);
  EXPECT_NE(result.find("RandnSeedOffset("), std::string::npos);
}

TEST(CodegenKernel, TestRandSeedOffsetRejectsOffsetOnly) {
  af::AscGraph graph("test_rand_offset_only_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  auto seed_output = af::ascir::cg::IndexExpr("seed", graph, af::Symbol(12345L));
  seed_output.mem->tensor_id = 2;
  Rand rand_op("Rand");
  rand_op.seed = seed_output;
  graph.AddNode(rand_op);

  auto rand = graph.FindNode("Rand");
  InitRandAttrs(graph, DT_FLOAT, z0);
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;
  EXPECT_FALSE(codegen.IsNodeValid(*rand));
}

TEST(CodegenKernel, TestRandSeedOffsetRejectsSeedOnly) {
  af::AscGraph graph("test_rand_seed_only_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto offset_output = af::ascir::cg::IndexExpr("offset", graph, af::Symbol(0L));
  offset_output.mem->tensor_id = 2;
  Rand rand_op("Rand");
  rand_op.offset = offset_output;
  graph.AddNode(rand_op);

  auto rand = graph.FindNode("Rand");
  InitRandAttrs(graph, DT_FLOAT, z0);
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;
  EXPECT_FALSE(codegen.IsNodeValid(*rand));
}

TEST(CodegenKernel, TestRandLegacySignatureRejectsUnsupportedDtype) {
  af::AscGraph graph("test_rand_legacy_dtype_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_DOUBLE);
  InitRandAttrs(graph, DT_DOUBLE, z0);

  auto rand = graph.FindNode("Rand");
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;
  EXPECT_FALSE(codegen.IsNodeValid(*rand));
}

TEST_F(RandGraphCheckTest, TestRandLegacyDtypeCheckKeepsLegacyContract) {
  af::AscGraph graph("test_rand_legacy_dtype_check_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandGraph(graph, s0, z0, DT_FLOAT);
  InitRandAttrs(graph, DT_FLOAT, z0);

  EXPECT_EQ(IsDataTypeSupported(graph), 0);
}

TEST(CodegenKernel, TestRandLegacyFunctionAddsNodeToGraph) {
  af::AscGraph graph("test_rand_legacy_function_graph");
  const auto rand_output = af::ascir::cg::Rand("Rand", graph);
  const auto rand = graph.FindNode("Rand");
  ASSERT_NE(rand, nullptr);
  EXPECT_EQ(rand->GetType(), Rand::Type);
  EXPECT_EQ(rand->GetInDataNodes().size(), 0UL);
  EXPECT_EQ(rand->GetAllOutDataAnchorsSize(), 1UL);
  EXPECT_EQ(rand_output.GetOwnerOp().GetName(), "Rand");
}

TEST(CodegenKernel, TestRandnLegacyFunctionAddsNodeToGraph) {
  af::AscGraph graph("test_randn_legacy_function_graph");
  const auto randn_output = af::ascir::cg::Randn("Randn", graph);
  const auto randn = graph.FindNode("Randn");
  ASSERT_NE(randn, nullptr);
  EXPECT_EQ(randn->GetType(), Randn::Type);
  EXPECT_EQ(randn->GetInDataNodes().size(), 0UL);
  EXPECT_EQ(randn->GetAllOutDataAnchorsSize(), 1UL);
  EXPECT_EQ(randn_output.GetOwnerOp().GetName(), "Randn");
}

TEST(CodegenKernel, TestRandnLegacySignatureRejectsUnsupportedDtype) {
  af::AscGraph graph("test_randn_legacy_dtype_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_DOUBLE);
  InitRandnAttrs(graph, DT_DOUBLE, z0);

  auto randn = graph.FindNode("Randn");
  af::ascir::RandSeedOffsetAscIrCodegenImplV2 codegen;
  EXPECT_FALSE(codegen.IsNodeValid(*randn));
}

TEST_F(RandGraphCheckTest, TestRandnLegacyDtypeCheckKeepsLegacyContract) {
  af::AscGraph graph("test_randn_legacy_dtype_check_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  EXPECT_EQ(IsDataTypeSupported(graph), 0);
}

TEST(CodegenKernel, TestRandnSeedOffsetApiCallKeepsLegacyCall) {
  af::AscGraph graph("test_randn_legacy_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(randn->outputs[0]);
  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.axis_map.at(z0.id).type = af::Axis::Type::kAxisTypeTileInner;

  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(randn), 0);
  std::string result;
  std::string output_dtype_name;
  Tensor::DtypeName(randn->outputs[0].attr.dtype, output_dtype_name);
  Tensor output_tensor(randn->outputs[0], output_dtype_name);
  EXPECT_EQ(call.Generate(tpipe, {}, {}, {output_tensor}, result), 0);
  EXPECT_EQ(codegen::CodegenApiParam::GetNodeApiParam(randn)->semantic_feature, std::string());
  EXPECT_NE(result.find("Randn("), std::string::npos);
  EXPECT_EQ(result.find("Rand("), std::string::npos);
  EXPECT_EQ(result.find("RandnSeedOffset("), std::string::npos);
}

TEST(CodegenKernel, TestRandnSeedOffsetApiCallRejectsInvalidInputsAndOutputs) {
  af::AscGraph graph("test_randn_invalid_api_call_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");
  codegen::RandSeedOffsetApiCall call("RandSeedOffset");
  EXPECT_EQ(call.Init(randn), 0);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  std::string dtype_name;
  ASSERT_EQ(Tensor::DtypeName(randn->outputs[0].attr.dtype, dtype_name), 0);
  Tensor output_tensor(randn->outputs[0], dtype_name);
  Tensor duplicate_output(randn->outputs[0], dtype_name);
  std::string result;
  EXPECT_NE(call.Generate(tpipe, {}, {}, {}, result), 0);
  EXPECT_NE(call.Generate(tpipe, {}, {}, {output_tensor, duplicate_output}, result), 0);
}

// 测试 Randn 生成的 API 调用字符串
TEST(CodegenKernel, TestRandnApiCallStringGeneration) {
  af::AscGraph graph("test_randn_api_call_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  BuildRandnGraph(graph, s0, z0, DT_UINT32);
  InitRandnAttrs(graph, DT_UINT32, z0);

  auto randn = graph.FindNode("Randn");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(randn->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddSizeVar(af::SizeVar(s0));

  // 使用 UnaryOutputApiCall 来生成调用字符串
  codegen::UnaryOutputApiCall call("Rand");
  EXPECT_EQ(call.Init(randn), 0);

  std::string result;
  call.Generate(tpipe, {}, result);

  // 验证生成的字符串包含关键信息
  EXPECT_NE(result.find("Rand"), std::string::npos);  // Randn 也调用 Rand
  EXPECT_NE(result.find("local_0"), std::string::npos);
  EXPECT_NE(result.find("local_0_actual_size"), std::string::npos);
}

}  // namespace codegen
