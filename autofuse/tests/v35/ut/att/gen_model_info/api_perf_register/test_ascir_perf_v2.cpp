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
#include <memory>
#include <numeric>
#include "gtest/gtest.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "ascir_ops.h"
#include "att/api_perf_register/perf_param_v2.h"
#include "ascir/generator/v2_ascir_att_impl.h"
#include "api_perf_register/api_perf_factory.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "../../../../../ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"
#include "common/platform_context.h"
#include "api_perf_register/utils/api_perf_utils.h"
#include "graph_construct_utils.h"
#include "parser/specific_params_builder.h"
#include "tests/ut/common/ascir_reduce_test_helpers.h"
#include "tests/depends/slog/src/slog_stub.h"
#include "../../../../att/gen_model_info/api_perf_register/nddma_test_utils.h"
#include "soc_perf_test_config.h"

using namespace att;
using namespace af::sym;
using namespace af::ascir;

namespace att {
af::Status LoadStoreStrideV2Func(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                                 const Expr &stride, Expr &res);
}

class UTestAscirPerfV2 : public ::testing::Test {
 public:
  static ge::RuntimeStubV2 stub_v_2;
  static void TearDownTestCase() {
    ge::RuntimeStub::UnInstall(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    ge::RuntimeStub::Install(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(soc_perf_test::kV35NpuArch, soc_perf_test::kSocEmpty), af::SUCCESS);
  }
  void TearDown() override {}
};
ge::RuntimeStubV2 UTestAscirPerfV2::stub_v_2;

TEST_F(UTestAscirPerfV2, IsFiniteUsesFixedMaxLatency) {
  NodeDetail node_info;
  node_info.input_dtype = {kFloat32};
  node_info.output_dtype = {kBool};
  node_info.input_dims = {CreateExpr(64)};
  node_info.output_dims = {CreateExpr(64)};
  PerfOutputInfo perf;

  ASSERT_EQ(ascendcperf_v2::IsFinitePerf(node_info, perf), af::SUCCESS);
  EXPECT_EQ(Str(perf.pipe_res[PipeType::AIV_VEC]), "44");
}

void SetSingleReduceSpecificParams(NodeInfo &node) {
  auto &params = node.reduce_specific_params;
  params.canonical_params.valid = true;
  params.canonical_params.pattern = codegen::ReducePattern::kAR;
  params.canonical_params.merge_mode = codegen::ReduceMergeMode::kNone;
  params.canonical_params.merge_size = CreateExpr(8);
  params.canonical_params.merge_times = CreateExpr(1);
  params.canonical_params.reuse = {true, false};
  params.exprs.merge_size = {true, {{CreateExpr(8), ascir_param::ParamExprRole::kSemantic}}};
  params.exprs.merge_times = {true, {{CreateExpr(1), ascir_param::ParamExprRole::kSemantic}}};
}

att::TensorShapeInfo MakeCastShape(const std::string &data_type, const Expr &dim) {
  att::TensorShapeInfo shape;
  shape.loc = att::HardwareDef::UB;
  shape.data_type = data_type;
  shape.dims = {dim};
  shape.repeats = shape.dims;
  shape.strides = {CreateExpr(1)};
  shape.gm_strides = shape.strides;
  return shape;
}

std::string GetUnaryPerfExpr(const std::string &api_name, const std::string &input_dtype,
                             const std::string &output_dtype, const Expr &dim) {
  // IsInf has no V1 registration, its factory name is the V2 entry "IsInfV2".
  const std::string create_name = api_name == kIsInf ? kIsInf + "V2" : api_name;
  auto api = ApiPerfFactory::Instance().Create(create_name);
  EXPECT_NE(api, nullptr);
  NodeDetail node_info;
  node_info.input_dtype = {input_dtype};
  node_info.output_dtype = {output_dtype};
  node_info.input_dims = {dim};
  node_info.output_dims = {dim};
  node_info.unary_bitwidth_change_node_params.valid = true;
  node_info.unary_bitwidth_change_node_params.cal_count = dim;
  node_info.unary_bitwidth_change_node_params.outer_repeats = {CreateExpr(1)};
  PerfOutputInfo perf_res;
  if (api_name == kIsnan) {
    EXPECT_EQ(ascendcperf_v2::IsNanPerf(node_info, perf_res), af::SUCCESS);
  } else if (api_name == kIsInf) {
    EXPECT_EQ(ascendcperf_v2::IsInfPerf(node_info, perf_res), af::SUCCESS);
  } else {
    EXPECT_EQ(ascendcperf_v2::IsFinitePerf(node_info, perf_res), af::SUCCESS);
  }
  return Str(perf_res.pipe_res[PipeType::AIV_VEC]);
}

struct TransposePerfResult {
  std::string pipe_expr;
  std::string ternary_expr;
};

TransposePerfResult GetTransposePerfExpr(uint32_t total_dim, uint32_t inner_dim, const std::vector<Expr> &output_dims,
                                         const std::vector<Expr> &outer_loop_axes,
                                         const std::vector<Expr> &input_strides, bool valid = true) {
  auto api = ApiPerfFactory::Instance().Create("TransposeV2");
  EXPECT_NE(api, nullptr);
  NodeDetail node_info;
  node_info.input_dtype = {kFloat16};
  node_info.input_dims = output_dims;
  node_info.transpose_node_params.valid = valid;
  node_info.transpose_node_params.total_dim = total_dim;
  node_info.transpose_node_params.inner_dim = inner_dim;
  node_info.transpose_node_params.output_dims = output_dims;
  node_info.transpose_node_params.outer_loop_axes = outer_loop_axes;
  node_info.transpose_node_params.input_strides = input_strides;
  PerfOutputInfo perf_res;
  EXPECT_EQ(ascendcperf_v2::TransposePerf(node_info, perf_res), af::SUCCESS);

  std::string ternary_expr;
  if (!perf_res.ternary_ops.empty()) {
    const auto &ternary = perf_res.ternary_ops.begin()->second;
    ternary_expr = ternary.GetTernaryOpStr();
  }
  return {Str(perf_res.pipe_res[PipeType::AIV_VEC]), ternary_expr};
}

TEST_F(UTestAscirPerfV2, IsNanPerfUsesFloat32BoolFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsnan, kFloat32, kBool, CreateExpr(64)), "52");
}

TEST_F(UTestAscirPerfV2, IsNanPerfUsesFloat16BoolFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsnan, kFloat16, kBool, CreateExpr(64)), "50");
}

TEST_F(UTestAscirPerfV2, IsNanPerfUsesNonBoolOutputFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsnan, kFloat32, kFloat32, CreateExpr(64)), "48");
}

TEST_F(UTestAscirPerfV2, IsFinitePerfUsesFloat16BoolFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsFinite, kFloat16, kBool, CreateExpr(64)), "42");
}

TEST_F(UTestAscirPerfV2, IsFinitePerfUsesNonBoolOutputFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsFinite, kFloat32, kFloat32, CreateExpr(64)), "40");
}

TEST_F(UTestAscirPerfV2, IsInfPerfUsesFloat32BoolFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsInf, kFloat32, kBool, CreateExpr(64)), "55");
}

TEST_F(UTestAscirPerfV2, IsInfPerfUsesFloat16BoolFormula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsInf, kFloat16, kBool, CreateExpr(64)), "53");
}

TEST_F(UTestAscirPerfV2, IsInfPerfUsesFloat32Uint8Formula) {
  EXPECT_EQ(GetUnaryPerfExpr(kIsInf, kFloat32, kUInt8, CreateExpr(64)), "55");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim2Inner1Formula) {
  const auto result =
      GetTransposePerfExpr(2, 1, {CreateExpr(2), CreateExpr(64)}, {CreateExpr(3)}, {CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "(((4 * transpose_stride_count) + 94) * 3)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim3Inner1Formula) {
  const auto result = GetTransposePerfExpr(3, 1, {CreateExpr(2), CreateExpr(3), CreateExpr(64)}, {CreateExpr(4)},
                                           {CreateExpr(192), CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "(((12 * transpose_stride_count) + 98) * 4)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim3Inner2Formula) {
  const auto result = GetTransposePerfExpr(3, 2, {CreateExpr(2), CreateExpr(3), CreateExpr(64)}, {CreateExpr(4)},
                                           {CreateExpr(192), CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "(((8 * transpose_stride_count) + 225) * 4)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim4Inner1Formula) {
  const auto result =
      GetTransposePerfExpr(4, 1, {CreateExpr(2), CreateExpr(2), CreateExpr(2), CreateExpr(64)}, {CreateExpr(3)},
                           {CreateExpr(256), CreateExpr(128), CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "(((16 * transpose_stride_count) + 100) * 3)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim4Inner2Formula) {
  const auto result =
      GetTransposePerfExpr(4, 2, {CreateExpr(2), CreateExpr(2), CreateExpr(2), CreateExpr(64)}, {CreateExpr(3)},
                           {CreateExpr(256), CreateExpr(128), CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "(((8 * transpose_stride_count) + 208) * 3)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesDim4Inner3Formula) {
  const auto result =
      GetTransposePerfExpr(4, 3, {CreateExpr(2), CreateExpr(2), CreateExpr(2), CreateExpr(64)}, {CreateExpr(3)},
                           {CreateExpr(256), CreateExpr(128), CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(result.pipe_expr, "((((1 + transpose_stride_count) * 4) + 442) * 3)");
}

TEST_F(UTestAscirPerfV2, TransposePerfUsesGenericFormulaAndStrideTernary) {
  const auto result = GetTransposePerfExpr(
      5, 4, {CreateExpr(1), CreateExpr(1), CreateExpr(1), CreateExpr(1), CreateExpr(64)}, {},
      {CreateExpr(256), CreateExpr(128), CreateExpr(64), CreateExpr(32), CreateExpr("transpose_stride")});
  EXPECT_EQ(result.pipe_expr, "(22 + transpose_stride_count)");
  EXPECT_EQ(result.ternary_expr, "TernaryOp(IsEqual(Mod(transpose_stride, 64), 0), 64, Mod(transpose_stride, 64))");
}

TEST_F(UTestAscirPerfV2, TransposePerfFallsBackToUnitVectorWhenParamsInvalid) {
  const auto result = GetTransposePerfExpr(1, 1, {CreateExpr(64)}, {}, {CreateExpr(1)}, false);
  EXPECT_EQ(result.pipe_expr, "20");
  EXPECT_TRUE(result.ternary_expr.empty());
}

std::string GetCastPerfExpr(const std::string &api_name, const std::string &input_dtype,
                            const std::string &output_dtype, const ascir_param::CastNodeParams &cast_params = {}) {
  auto cast_v2 = ApiPerfFactory::Instance().Create(api_name);
  EXPECT_NE(cast_v2, nullptr);
  auto cast_v2_perf = cast_v2->GetPerfFunc();
  const Expr dim = CreateExpr(129);
  std::vector<att::TensorShapeInfo> input_shapes = {MakeCastShape(input_dtype, dim)};
  std::vector<att::TensorShapeInfo> output_shapes = {MakeCastShape(output_dtype, dim)};
  NodeInfo node;
  node.cast_node_params = cast_params;
  PerfOutputInfo perf_res;
  EXPECT_EQ(cast_v2_perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  return Str(perf_res.pipe_res[PipeType::AIV_VEC]);
}

ascir_param::CastNodeParams MakeCastParams(const std::vector<Expr> &output_dims,
                                           const std::vector<Expr> &output_strides,
                                           const std::vector<Expr> &input_strides) {
  ascir_param::CastNodeParams params;
  params.valid = true;
  params.output_dims = output_dims;
  params.output_strides = output_strides;
  params.input_strides = input_strides;
  return params;
}

ascir_param::CompareNodeParams MakeCompareParams(bool is_scalar, const std::vector<Expr> &output_dims,
                                                 const std::vector<Expr> &output_strides,
                                                 const std::vector<Expr> &input_strides) {
  ascir_param::CompareNodeParams params;
  params.valid = true;
  params.is_scalar = is_scalar;
  params.output_dims = output_dims;
  params.output_strides = output_strides;
  params.input_strides = input_strides;
  return params;
}

std::string GetComparePerfExpr(const std::string &api_name, const std::string &input_dtype,
                               const ascir_param::CompareNodeParams &compare_params) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create(api_name);
  EXPECT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  const Expr dim = CreateExpr(129);
  std::vector<att::TensorShapeInfo> input_shapes = {MakeCastShape(input_dtype, dim), MakeCastShape(input_dtype, dim)};
  std::vector<att::TensorShapeInfo> output_shapes = {MakeCastShape(kUInt8, dim)};
  NodeInfo node;
  node.compare_node_params = compare_params;
  PerfOutputInfo perf_res;
  EXPECT_EQ(cmp_v2_perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  return Str(perf_res.pipe_res[PipeType::AIV_VEC]);
}

std::string GetWhereImplPerfExpr(const std::string &data_dtype) {
  NodeDetail node_detail;
  node_detail.name = "WhereNode";
  node_detail.optype = kWhere;
  node_detail.input_dtype = {kUInt8, data_dtype, data_dtype};
  node_detail.output_dtype = {data_dtype};
  node_detail.input_dims = {CreateExpr(32)};
  node_detail.output_dims = node_detail.input_dims;
  node_detail.where_node_params.valid = true;
  node_detail.where_node_params.output_dims = {CreateExpr(32)};

  PerfOutputInfo perf_res;
  EXPECT_EQ(att::ascendcperf_v2::WherePerf(node_detail, perf_res), af::SUCCESS);
  return Str(perf_res.pipe_res[PipeType::AIV_VEC]);
}

std::string GetWhereComputePerfExpr(const std::string &data_dtype, const std::vector<Expr> &output_dims) {
  NodeDetail node_detail;
  node_detail.name = "WhereNode";
  node_detail.optype = kWhere;
  node_detail.input_dtype = {kUInt8, data_dtype, data_dtype};
  node_detail.output_dtype = {data_dtype};
  node_detail.input_dims = output_dims;
  node_detail.output_dims = node_detail.input_dims;
  node_detail.where_node_params.valid = true;
  node_detail.where_node_params.output_dims = output_dims;

  PerfOutputInfo perf_res;
  EXPECT_EQ(att::ascendcperf_v2::WherePerf(node_detail, perf_res), af::SUCCESS);
  return Str(perf_res.pipe_res[PipeType::AIV_VEC]);
}

std::string GetWhereComputePerfExpr(const std::string &data_dtype) {
  return GetWhereComputePerfExpr(data_dtype, {CreateExpr(1), CreateExpr(32)});
}

NodeDetail MakeCastNodeDetail(const std::string &input_dtype, const std::string &output_dtype) {
  NodeDetail node_detail;
  node_detail.name = "CastNode";
  node_detail.optype = kCast;
  node_detail.input_dtype = {input_dtype};
  node_detail.output_dtype = {output_dtype};
  node_detail.input_dims = {CreateExpr("cast_outer"), CreateExpr("cast_inner")};
  node_detail.output_dims = node_detail.input_dims;
  node_detail.cast_node_params.valid = true;
  node_detail.cast_node_params.output_dims = node_detail.output_dims;
  node_detail.cast_node_params.output_strides = {CreateExpr("cast_output_stride"), CreateExpr(1)};
  node_detail.cast_node_params.input_strides = {CreateExpr("cast_input_stride"), CreateExpr(1)};
  return node_detail;
}

TEST_F(UTestAscirPerfV2, CastDynamicCallCountForNonSameBit2D) {
  NodeDetail node_detail = MakeCastNodeDetail(kFloat16, kFloat32);
  PerfOutputInfo perf;

  EXPECT_EQ(att::ascendcperf_v2::CastPerf(node_detail, perf), af::SUCCESS);

  EXPECT_EQ(perf.ternary_ops.size(), 2U);
  EXPECT_NE(Str(perf.pipe_res[PipeType::AIV_VEC]).find("cast_call_count"), std::string::npos);
}

TEST_F(UTestAscirPerfV2, CastSameBitCallCountUsesOuterDim) {
  NodeDetail node_detail = MakeCastNodeDetail(kInt32, kUInt32);
  PerfOutputInfo perf;

  EXPECT_EQ(att::ascendcperf_v2::CastPerf(node_detail, perf), af::SUCCESS);

  EXPECT_TRUE(perf.ternary_ops.empty());
  EXPECT_EQ(Str(perf.pipe_res[PipeType::AIV_VEC]).find("cast_call_count"), std::string::npos);
  EXPECT_NE(Str(perf.pipe_res[PipeType::AIV_VEC]).find("cast_outer"), std::string::npos);
}
TEST_F(UTestAscirPerfV2, FillSpecificParamsStoresSharedReduceParamsOnAscNode) {
  using ascir_reduce_test_helpers::BuildReduceNodeInfo;
  using ascir_reduce_test_helpers::ReduceTestEnv;
  ReduceTestEnv env("reduce");
  env.SetIoAttrs({env.s1, CreateExpr(1)}, {env.s0, CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)});
  auto node_info = BuildReduceNodeInfo(env, "reduce");
  const auto params = ascir_param::GetAscirNodeParams(env.node);
  ASSERT_NE(params, nullptr);
  const auto *reduce_params = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
  ASSERT_NE(reduce_params, nullptr);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.pattern, reduce_params->canonical_params.pattern);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.merge_mode, reduce_params->canonical_params.merge_mode);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.merge_size, reduce_params->canonical_params.merge_size);
}

TEST_F(UTestAscirPerfV2, FillSpecificParamsStoresCompareParamsOnAscNode) {
  af::AscGraph graph("compare_param_graph");
  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Ge ge_op("ge");
  graph.AddNode(ge_op);
  ge_op.x1 = x1.y;
  ge_op.x2 = x2.y;

  auto ge_node = graph.FindNode("ge");
  ASSERT_NE(ge_node, nullptr);
  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "Ge";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  auto compare_params =
      MakeCompareParams(true, {CreateExpr("outer"), CreateExpr("inner")}, {CreateExpr("out_stride"), CreateExpr(1)},
                        {CreateExpr("in_stride"), CreateExpr(1)});
  compare_params.outer_call_count = CreateExpr("outer_call_count");
  params->specific_params = compare_params;
  ASSERT_TRUE(ge_node->GetOpDesc()->SetExtAttr("AscirNodeParams", params));

  NodeInfo node_info;
  node_info.name = "ge";
  node_info.node_type = kGe;
  EXPECT_EQ(FillSpecificParams(ge_node, node_info), af::SUCCESS);
  EXPECT_TRUE(node_info.compare_node_params.valid);
  EXPECT_TRUE(node_info.compare_node_params.is_scalar);
  ASSERT_EQ(node_info.compare_node_params.output_dims.size(), 2U);
  EXPECT_EQ(Str(node_info.compare_node_params.outer_call_count), "outer_call_count");
  EXPECT_EQ(Str(node_info.compare_node_params.output_dims[0]), "outer");
  EXPECT_EQ(Str(node_info.compare_node_params.input_strides[0]), "in_stride");
}

TEST_F(UTestAscirPerfV2, FillSpecificParamsStoresCastParamsOnCeil2IntNode) {
  af::AscGraph graph("ceil2int_param_graph");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Ceil2Int ceil2int("ceil2int");
  graph.AddNode(ceil2int);
  ceil2int.x = x.y;

  auto ceil2int_node = graph.FindNode("ceil2int");
  ASSERT_NE(ceil2int_node, nullptr);
  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "Ceil2Int";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params =
      MakeCastParams({CreateExpr("outer"), CreateExpr("inner")}, {CreateExpr("out_stride"), CreateExpr(1)},
                     {CreateExpr("in_stride"), CreateExpr(1)});
  ASSERT_TRUE(ceil2int_node->GetOpDesc()->SetExtAttr("AscirNodeParams", params));

  NodeInfo node_info;
  node_info.name = "ceil2int";
  node_info.node_type = kCeil2Int;
  EXPECT_EQ(FillSpecificParams(ceil2int_node, node_info), af::SUCCESS);
  EXPECT_TRUE(node_info.cast_node_params.valid);
  ASSERT_EQ(node_info.cast_node_params.output_dims.size(), 2U);
  EXPECT_EQ(Str(node_info.cast_node_params.output_dims[0]), "outer");
  EXPECT_EQ(Str(node_info.cast_node_params.output_strides[0]), "out_stride");
  EXPECT_EQ(Str(node_info.cast_node_params.input_strides[0]), "in_stride");
}

TEST_F(UTestAscirPerfV2, FillSpecificParamsStoresUnaryBitWidthParamsOnIsInfNode) {
  af::AscGraph graph("isinf_param_graph");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::IsInf is_inf("is_inf");
  graph.AddNode(is_inf);
  is_inf.x = x.y;

  auto is_inf_node = graph.FindNode("is_inf");
  ASSERT_NE(is_inf_node, nullptr);
  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "IsInf";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  ascir_param::UnaryBitWidthChangeNodeParams unary_params;
  unary_params.valid = true;
  unary_params.cal_count = CreateExpr("cal_count");
  unary_params.outer_repeats = {CreateExpr("outer_repeat")};
  unary_params.input_strides = {CreateExpr("in_stride")};
  unary_params.output_strides = {CreateExpr("out_stride")};
  params->specific_params = unary_params;
  ASSERT_TRUE(is_inf_node->GetOpDesc()->SetExtAttr("AscirNodeParams", params));

  NodeInfo node_info;
  node_info.name = "is_inf";
  node_info.node_type = kIsInf;
  EXPECT_EQ(FillSpecificParams(is_inf_node, node_info), af::SUCCESS);
  EXPECT_TRUE(node_info.unary_bitwidth_change_node_params.valid);
  EXPECT_EQ(Str(node_info.unary_bitwidth_change_node_params.cal_count), "cal_count");
  ASSERT_EQ(node_info.unary_bitwidth_change_node_params.outer_repeats.size(), 1U);
  EXPECT_EQ(Str(node_info.unary_bitwidth_change_node_params.outer_repeats[0]), "outer_repeat");
  EXPECT_EQ(Str(node_info.unary_bitwidth_change_node_params.input_strides[0]), "in_stride");
  EXPECT_EQ(Str(node_info.unary_bitwidth_change_node_params.output_strides[0]), "out_stride");
}

TEST_F(UTestAscirPerfV2, CompareVfHeadCostByInputType) {
  const auto params = MakeCompareParams(false, {CreateExpr(1)}, {CreateExpr(1)}, {CreateExpr(1)});

  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kUInt8, params), "45");
  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kFloat16, params), "60");
  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kFloat32, params), "66");
  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kInt64, params), "68");
}

TEST_F(UTestAscirPerfV2, CompareDoesNotMultiplyOuterRepeat) {
  const auto params = MakeCompareParams(false, {CreateExpr(4), CreateExpr(256)}, {CreateExpr(256), CreateExpr(1)},
                                        {CreateExpr(256), CreateExpr(1)});

  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kFloat16, params), "88");
}

TEST_F(UTestAscirPerfV2, CompareMultipliesOuterCallCount) {
  auto params = MakeCompareParams(false, {CreateExpr(4), CreateExpr(256)}, {CreateExpr(256), CreateExpr(1)},
                                  {CreateExpr(256), CreateExpr(1)});
  params.outer_call_count = CreateExpr(3);

  EXPECT_EQ(GetComparePerfExpr(kGe + "V2", kFloat16, params), "264");
}

// 测试 LoadApi API 边界条件
TEST_F(UTestAscirPerfV2, TestLoadApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

// 测试 StoreApi API 边界条件
TEST_F(UTestAscirPerfV2, TestStoreApiEmptyInput) {
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes(1);

  output_shapes[0].data_type = "float16";
  output_shapes[0].dims = {CreateExpr(10)};

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_NE(result, af::SUCCESS);
}

TEST_F(UTestAscirPerfV2, TestLoadApiForTypev1) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");

  input_shapes[0].data_type = "float32";
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(64)};
  input_shapes[0].gm_strides = {z0t_size * CreateExpr(512), CreateExpr(512), CreateExpr(1)};
  input_shapes[0].strides = input_shapes[0].gm_strides;
  output_shapes[0].data_type = "float32";
  output_shapes[0].repeats = input_shapes[0].repeats;
  output_shapes[0].gm_strides = {z0t_size * CreateExpr(512), CreateExpr(512), CreateExpr(1)};
  output_shapes[0].strides = output_shapes[0].gm_strides;
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  const auto ternary_ops = perf_res.ternary_ops;
  const auto ret = ConcursiveReplaceVars(ternary_ops);
  // LoadStride calculation: k=0.005, block_count=z0t_size*z1t_size, stride_used=min(448*4, 4096)=1792
  // Result: 0.005 * z0t_size * z1t_size * 1792 = 8.96 * z0t_size * z1t_size
  const std::string expect_stride = "(8.95999979972839 * z0t_size * z1t_size)";
  const std::string load_perf = "(256 * z0t_size * z1t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236)))";
  // Note: The order of terms in the output is (load_perf + stride + 160.0)
  EXPECT_EQ(Str(res.Replace(ret)), "(" + load_perf + " + " + expect_stride + " + 160.0)");
}

TEST_F(UTestAscirPerfV2, TestLoadApiForTypev2) {
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
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(Str(res.Replace(ret)),
            "((256 * z0t_size * z1t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236))) + 160.0)");
}

TEST_F(UTestAscirPerfV2, TestLoadApiForTypev3) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z4t_size = CreateExpr("z4t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  input_shapes[0].repeats = input_shapes[0].dims;
  // 连续 {true, false, true, true}
  input_shapes[0].strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34), z4t_size * CreateExpr(7),
                             CreateExpr(7), af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z4t_size, CreateExpr(7)};
  output_shapes[0].repeats = output_shapes[0].dims;
  output_shapes[0].strides = {z4t_size * CreateExpr(7 * 7 * 34), z4t_size * CreateExpr(7 * 34),
                              z4t_size * CreateExpr(7), CreateExpr(7), af::sym::kSymbolOne};
  output_shapes[0].gm_strides = output_shapes[0].strides;
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  std::vector<CacheLineConfig> config;
  perf_res.cache_line_config = &config;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(Str(res.Replace(ret)),
            "TernaryOp(((1666 * z0z1t_size * z4t_size) + -256) < 0, ((13328 * z0z1t_size * z4t_size / "
            "(((6.40880012512207 / (block_dim)) + 13.1354999542236))) + 160.0), ((13328 * z0z1t_size * z4t_size / "
            "(((6.61549997329712 / (block_dim)) + 11.8291997909546))) + 160.0))");
  EXPECT_EQ(config.size(), 1);
}

TEST_F(UTestAscirPerfV2, TestStoreApiForType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = input_shapes[0].dims;
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = output_shapes[0].dims;
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  output_shapes[0].gm_strides = output_shapes[0].strides;

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store_v2, nullptr);
  auto perf = store_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(Str(res.Replace(ret)),
            "((1904 * z0z1t_size * z6t_size / (((10.2650003433228 / (block_dim)) + 11.7740001678467))) + 160.0)");
}

TEST_F(UTestAscirPerfV2, TestStoreApiCacheLineExprUsesSingleTransferLen) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr s18 = CreateExpr("s18");
  Expr s1 = CreateExpr("s1");
  Expr s20 = CreateExpr("s20");
  Expr z0Tb_size = CreateExpr("z0Tb_size");
  Expr z0t_size = CreateExpr("z0t_size");
  Expr inner_bytes = s1 * s20 * CreateExpr(4);

  input_shapes[0].data_type = "float32";
  input_shapes[0].dims = {s18 / (z0Tb_size * z0t_size), z0Tb_size, z0t_size, s1 * s20};
  input_shapes[0].repeats = input_shapes[0].dims;
  input_shapes[0].strides = {s1 * s20 * z0Tb_size * z0t_size, s1 * s20 * z0t_size, s1 * s20, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0] = input_shapes[0];
  input_shapes[0].data_type_size = 4;
  output_shapes[0].data_type_size = 4;

  PerfOutputInfo perf_res;
  std::vector<CacheLineConfig> config;
  perf_res.cache_line_config = &config;
  auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store_v2, nullptr);
  auto perf = store_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);

  EXPECT_EQ(result, af::SUCCESS);
  ASSERT_EQ(config.size(), 1U);
  EXPECT_TRUE(config[0].cache_line_expr == inner_bytes);
  EXPECT_EQ(config[0].direction, CacheLineDirection::kUbToGm);
}

TEST_F(UTestAscirPerfV2, TestStoreApiForSmallStride) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = input_shapes[0].dims;
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = output_shapes[0].dims;
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size + CreateExpr(128), af::sym::kSymbolOne};
  output_shapes[0].gm_strides = output_shapes[0].strides;

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store_v2, nullptr);
  auto perf = store_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // MTE3 padding: 由于gm_stride>0且z6t_size可能<16(int64的128B是16个元素),会生成三元表达式
  // StoreStride uses the DMA block count (34) after the two LoopMode dimensions are removed.
  const std::string kStride = "1500.41599273682";
  // Note: 由于padding, z6t_size会被替换为TernaryOp表达式,且TernaryOp在z0z1t_size之前
  EXPECT_EQ(Str(res.Replace(ret)),
            "((1904 * TernaryOp(IsEqual(ExpectLt((8 * z6t_size), 128), 0), z6t_size, 16) * z0z1t_size / "
            "(((10.2650003433228 / (block_dim)) + 11.7740001678467))) + " +
                kStride + ")");
}

TEST_F(UTestAscirPerfV2, TestStoreApiForBiggerStride) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = input_shapes[0].dims;
  input_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size, af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = output_shapes[0].dims;
  output_shapes[0].strides = {CreateExpr(7), CreateExpr(34), z6t_size + CreateExpr(40960), af::sym::kSymbolOne};
  output_shapes[0].gm_strides = output_shapes[0].strides;

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store_v2, nullptr);
  auto perf = store_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE3];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // MTE3 padding: 由于gm_stride>0且z6t_size可能<16(int64的128B是16个元素),会生成三元表达式
  // StoreStride uses the DMA block count (34) after the two LoopMode dimensions are removed.
  const std::string kStride = "5521.66397094727";
  EXPECT_EQ(Str(res.Replace(ret)),
            "((1904 * TernaryOp(IsEqual(ExpectLt((8 * z6t_size), 128), 0), z6t_size, 16) * z0z1t_size / "
            "(((10.2650003433228 / (block_dim)) + 11.7740001678467))) + " +
                kStride + ")");
}

TEST_F(UTestAscirPerfV2, TestStoreStrideLoopModeUsesDmaBlockCount) {
  const std::map<std::string, float> params = {{"k", 0.0385F},
                                               {"u", 4096.0F},
                                               {"data_type_size", 4.0F},
                                               {"block_count_idx", 1.0F},
                                               {"loop_mode_dim_count", 1.0F}};
  Expr result;
  ASSERT_EQ(LoadStoreStrideV2Func(params, {CreateExpr(20), CreateExpr(57), CreateExpr(20)}, CreateExpr(380), result),
            af::SUCCESS);
  Expr expected = CreateExpr(0.0385F) * CreateExpr(57) * CreateExpr(380) * CreateExpr(4);
  result.Simplify();
  expected.Simplify();
  EXPECT_EQ(Str(result), Str(expected));
}

TEST_F(UTestAscirPerfV2, TestStoreApiLoopModeEndToEnd) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z2t_size = CreateExpr("z2t_size");
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr block_len = CreateExpr(20);
  input_shapes[0].data_type = "float32";
  input_shapes[0].dims = {z2t_size, z0z1t_size, block_len};
  input_shapes[0].repeats = input_shapes[0].dims;
  input_shapes[0].strides = {z0z1t_size * CreateExpr(400), CreateExpr(400), CreateExpr(1)};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  input_shapes[0].data_type_size = 4;
  output_shapes[0] = input_shapes[0];
  output_shapes[0].strides = {z0z1t_size * CreateExpr(400), CreateExpr(400), CreateExpr(1)};
  output_shapes[0].gm_strides = output_shapes[0].strides;

  PerfOutputInfo perf_res;
  auto store_v2 = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store_v2, nullptr);
  auto perf = store_v2->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  ASSERT_EQ(perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_MTE3].Replace(ret)),
            "((58.5199996829033 * z0z1t_size) + (80 * z0z1t_size * z2t_size / "
            "(((10.2650003433228 / (block_dim)) + 11.7740001678467))) + 160.0)");
}

TEST_F(UTestAscirPerfV2, TestNddmaApiForType) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].strides = {CreateExpr(7) * CreateExpr(34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                             af::sym::kSymbolOne};
  input_shapes[0].gm_strides = input_shapes[0].strides;
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].strides = {CreateExpr(7) * CreateExpr(34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                              af::sym::kSymbolOne};
  output_shapes[0].gm_strides = output_shapes[0].strides;

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  auto perf = nddma->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  EXPECT_EQ(
      Str(res.Replace(ret)),
      "((1904 * z0z1t_size * z6t_size / (((6.3899998664856 / (block_dim)) + 7.6100001335144))) + 418.978912353516)");
}

TEST_F(UTestAscirPerfV2, TestNddma1DModelSupportsFourDtypeSizes) {
  struct DtypeCase {
    const char *dtype;
    int32_t dtype_size;
  };
  const std::vector<DtypeCase> cases = {{"int8", 1}, {"float16", 2}, {"float32", 4}, {"int64", 8}};
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  const auto perf = nddma->GetPerfFunc();
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  node.node_ptr->outputs[0].attr.vectorized_axis = {0};

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.dtype);
    const auto shape =
        Make1DNddmaShape(test_case.dtype, test_case.dtype_size, CreateExpr(256), CreateExpr(4), CreateExpr(2));
    const std::vector<TensorShapeInfo> shapes = {shape};
    PerfOutputInfo perf_res;
    ASSERT_EQ(perf(shapes, shapes, node, perf_res), af::SUCCESS);
    EXPECT_NE(Str(perf_res.pipe_res[PipeType::AIV_MTE2]).find("nddma_1d_multicore"), std::string::npos);
    EXPECT_EQ(perf_res.ternary_ops.size(), 1U);
  }
}

TEST_F(UTestAscirPerfV2, TestNddmaEffectiveRank2To5UsesNewModel) {
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  for (size_t rank = 2U; rank <= 5U; ++rank) {
    SCOPED_TRACE(rank);
    TensorShapeInfo shape;
    shape.data_type = "int64";
    shape.data_type_size = 8;
    shape.dims.assign(rank - 2U, CreateExpr(1));
    shape.dims.insert(shape.dims.end(), {CreateExpr(4), CreateExpr(64)});
    shape.repeats = shape.dims;
    shape.gm_strides.assign(rank - 2U, CreateExpr(256));
    shape.gm_strides.insert(shape.gm_strides.end(), {CreateExpr(64), CreateExpr(1)});
    shape.strides = shape.gm_strides;
    node.node_ptr->outputs[0].attr.vectorized_axis.resize(rank);
    std::iota(node.node_ptr->outputs[0].attr.vectorized_axis.begin(),
              node.node_ptr->outputs[0].attr.vectorized_axis.end(), 0);
    const std::vector<TensorShapeInfo> shapes = {shape};
    PerfOutputInfo perf_res;

    ASSERT_EQ(nddma->GetPerfFunc()(shapes, shapes, node, perf_res), af::SUCCESS);
    const auto cycles = Str(perf_res.pipe_res[PipeType::AIV_MTE2]);
    EXPECT_NE(cycles.find("nddma_1d_multicore"), std::string::npos);
  }
}

TEST_F(UTestAscirPerfV2, TestNddmaFallsBackOnDtypeSizeMismatch) {
  const auto shape = Make1DNddmaShape("float16", 4, CreateExpr(256), CreateExpr(1), CreateExpr(1));
  const std::vector<TensorShapeInfo> shapes = {shape};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  PerfOutputInfo perf_res;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  ASSERT_EQ(nddma->GetPerfFunc()(shapes, shapes, node, perf_res), af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_MTE2]).find("nddma_1d_multicore"), std::string::npos);
}

TEST_F(UTestAscirPerfV2, TestNddma1DModelFallsBackForCvUbFusion) {
  const auto shape = Make1DNddmaShape("int8", 1, CreateExpr(256), CreateExpr(1), CreateExpr(1));
  const std::vector<TensorShapeInfo> shapes = {shape};
  PerfOutputInfo perf_res;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  node.node_ptr->outputs[0].attr.vectorized_axis = {0};
  node.is_cv_ub_fusion = true;
  auto *slog = ge::SlogStub::GetInstance();
  const int32_t old_level = slog->GetLevel();
  slog->SetLevel(DLOG_WARN);

  testing::internal::CaptureStdout();
  const auto status = nddma->GetPerfFunc()(shapes, shapes, node, perf_res);
  const std::string output = testing::internal::GetCapturedStdout();
  slog->SetLevel(old_level);
  ASSERT_EQ(status, af::SUCCESS);
  EXPECT_EQ(Str(perf_res.pipe_res[PipeType::AIV_MTE2]).find("nddma_1d_multicore"), std::string::npos);
  EXPECT_NE(output.find("fallback_reason=codegen_mismatch"), std::string::npos);
}

TEST_F(UTestAscirPerfV2, TestNddmaApiSmallBlockLen) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0z1t_size = CreateExpr("z0z1t_size");
  Expr z6t_size = CreateExpr("z6t_size");

  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  input_shapes[0].strides = {CreateExpr(7) * CreateExpr(34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                             af::sym::kSymbolOne};
  input_shapes[0].gm_strides = {CreateExpr(34 * 32 * 7), CreateExpr(34 * 32), CreateExpr(32), af::sym::kSymbolOne};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].repeats = {z0z1t_size, CreateExpr(7), CreateExpr(34), z6t_size};
  output_shapes[0].strides = {CreateExpr(7) * CreateExpr(34) * z6t_size, CreateExpr(34) * z6t_size, z6t_size,
                              af::sym::kSymbolOne};
  output_shapes[0].gm_strides = {CreateExpr(34 * 32 * 7), CreateExpr(34 * 32), CreateExpr(32), af::sym::kSymbolOne};

  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  auto perf = nddma->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  const std::string kIsSmallBlockLen =
      "IsEqual(LogicAnd(ExpectLt(0, Abs((32 - z6t_size))), ExpectLt(z6t_size, 16)), 0)";
  const std::string kLastAxisLen = "TernaryOp(" + kIsSmallBlockLen + ", z6t_size, 16)";
  // NddmaStride with penalty: penalty + stride calculation
  // penalty = block_count_idx * stride_used * penalty_coeff = 2 * Abs((32-z6t_size))*8) * 4 = 64.0 * Abs((32-z6t_size))
  // stride = k * block_count * stride_used
  //        = 0.005 * (238*z0z1t_size) * Min(4096, Abs((32-z6t_size))*8)
  const std::string kPenalty = "(64.0 * Abs((32 - z6t_size)))";
  const std::string kStride = "(1.18999997340143 * Min(4096.0, (8 * Abs((32 - z6t_size)))) * z0z1t_size)";
  // Note: SymEngine may reorder additive terms; the stride term comes before nddma_perf and penalty.
  EXPECT_EQ(Str(res.Replace(ret)), "(" + kStride + " + (1904 * " + kLastAxisLen +
                                       " * z0z1t_size / (((6.3899998664856 / (block_dim)) + 7.6100001335144))) + " +
                                       kPenalty + " + 418.978912353516)");
}

TEST_F(UTestAscirPerfV2, TestNddmaApiGmStrideTranspose) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0Tb_size = CreateExpr("z0Tb_size");
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");
  Expr z2z3_size = CreateExpr(2211);

  // 输入为Data
  input_shapes[0].data_type = "int64";
  input_shapes[0].dims = {};
  input_shapes[0].repeats = {};
  input_shapes[0].strides = {};
  input_shapes[0].gm_strides = {};
  output_shapes[0].data_type = "int64";
  output_shapes[0].dims = {z1t_size, z0t_size, CreateExpr(2211)};
  output_shapes[0].repeats = {z1t_size, z0t_size, CreateExpr(2211)};
  output_shapes[0].strides = {CreateExpr(2240) * z0t_size, CreateExpr(2240), af::sym::kSymbolOne};
  // GM->UB侧存在转置，需要考虑特殊处理
  output_shapes[0].gm_strides = {CreateExpr(2211), CreateExpr(1134243), af::sym::kSymbolOne};
  input_shapes[0].data_type_size = 8;
  output_shapes[0].data_type_size = 8;
  PerfOutputInfo perf_res;
  NodeInfo node;
  auto nddma = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(nddma, nullptr);
  auto perf = nddma->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Nddma", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  // 存在外抛
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // NddmaStride with penalty: stride calculation
  // stride = k * block_count * stride_used = 0.005 * (z0t_size * z1t_size) * 4096 = 20.4799995422363 * z0t_size *
  // z1t_size penalty = block_count_idx * stride_used * penalty_coeff = 1 * 4096 * 4 = 16384.0
  const std::string kStride = "(20.4799995422363 * z0t_size * z1t_size)";
  // Note: The order of terms in the output is (nddma_perf + stride + penalty + constant)
  EXPECT_EQ(Str(res.Replace(ret)),
            "((17688 * z0t_size * z1t_size / (((6.3899998664856 / (block_dim)) + "
            "7.6100001335144))) + " +
                kStride + " + 16802.9789123535)");
}

// 测试 CalculateStride 函数 - Nddma 节点验证 block_count_idx
TEST_F(UTestAscirPerfV2, TestCalculateStrideNddmaBlockCountIdx) {
  // 构造 TensorShapeInfo
  att::TensorShapeInfo shape_info;

  Expr z2t = CreateExpr("z2t");
  Expr z1t = CreateExpr("z1t");

  // 按照用户指定的条件：
  // gm_stride = [1, 90, 6930]
  // repeats = [z2t, z1t, 90]
  // need_swap = 0
  shape_info.dims = {z2t, z1t, CreateExpr(90)};
  shape_info.repeats = {z2t, z1t, CreateExpr(90)};
  shape_info.gm_strides = {CreateExpr(1), CreateExpr(90), CreateExpr(6930)};
  shape_info.strides = {CreateExpr(1), CreateExpr(90), CreateExpr(6930)};
  shape_info.origin_repeats = {z2t, z1t, CreateExpr(90)};

  // 创建 NodeDetail
  att::NodeDetail node_info;
  node_info.name = "Nddma";
  node_info.optype = "NddmaV2";
  node_info.input_dtype = {"float16"};
  node_info.output_dtype = {"float16"};
  node_info.input_dims = {z2t, z1t, CreateExpr(90)};
  node_info.output_dims = {z2t, z1t, CreateExpr(90)};
  node_info.repeats = {z2t, z1t, CreateExpr(90)};

  // 设置 supported_max_dma_len
  const int32_t supported_max_dma_len = 2;
  const bool need_swap = false;

  // 调用 CalculateStride 函数
  auto result = att::CalculateStride(shape_info, false, node_info, supported_max_dma_len, need_swap);

  // 验证 block_count_idx = 1（从第1维 z1t 那维开始出现stride）
  EXPECT_EQ(result.block_count_idx, 1);

  // 验证 stride 表达式
  // 根据 CalculateStride 逻辑：
  // filtered_strides = [1, 90, 6930]
  // filtered_repeats = [z2t, z1t, 90]
  // filtered_dim_size = 3
  // need_swap = 0, 所以 actually_swap = false
  // block_count_idx = 3 - 2 = 1
  // stride = filtered_strides[1] - filtered_repeats[2] = 90 - 90 = 0
  EXPECT_EQ(Str(result.stride), "6930");

  GELOGD("TestCalculateStrideNddmaBlockCountIdx: block_count_idx=%d, stride=%s", result.block_count_idx,
         Str(result.stride).c_str());
}

// 测试 CalculateStride 函数 - 动态shape场景下尾轴stride为符号表达式
TEST_F(UTestAscirPerfV2, TestCalculateStrideDynamicShapeLastStride) {
  // 构造 TensorShapeInfo
  att::TensorShapeInfo shape_info;

  Expr z2t = CreateExpr("z2t");
  Expr z1t = CreateExpr("z1t");
  Expr dynamic_stride = CreateExpr("dynamic_stride");  // 动态shape的stride

  // 设置动态shape场景：last_stride是符号表达式而非常量
  shape_info.dims = {z2t, z1t, CreateExpr(90)};
  shape_info.repeats = {z2t, z1t, CreateExpr(90)};
  shape_info.gm_strides = {CreateExpr(1), CreateExpr(90), dynamic_stride};  // 尾轴为动态stride
  shape_info.strides = {CreateExpr(1), CreateExpr(90), dynamic_stride};
  shape_info.origin_repeats = {z2t, z1t, CreateExpr(90)};

  // 创建 NodeDetail
  att::NodeDetail node_info;
  node_info.name = "Nddma";
  node_info.optype = "NddmaV2";
  node_info.input_dtype = {"float16"};
  node_info.output_dtype = {"float16"};
  node_info.input_dims = {z2t, z1t, CreateExpr(90)};
  node_info.output_dims = {z2t, z1t, CreateExpr(90)};
  node_info.repeats = {z2t, z1t, CreateExpr(90)};

  // 设置 supported_max_dma_len
  const int32_t supported_max_dma_len = 2;
  const bool need_swap = false;

  // 调用 CalculateStride 函数
  auto result = att::CalculateStride(shape_info, false, node_info, supported_max_dma_len, need_swap);

  // 验证 block_count_idx
  EXPECT_EQ(result.block_count_idx, 1);

  // 验证动态shape时返回的stride是新的符号变量（gm_stride_select）
  EXPECT_EQ(Str(result.stride), "gm_stride_select");

  // 验证 ternary_ops 被正确填充
  EXPECT_EQ(result.ternary_ops.size(), 1u);
  EXPECT_NE(result.ternary_ops.find(result.stride), result.ternary_ops.end());

  // 验证 TernaryOp 的条件是 K_GT (dynamic_stride > 1)
  const auto &ternary_op = result.ternary_ops.at(result.stride);
  EXPECT_EQ(ternary_op.GetVariable(), result.stride);

  GELOGD("TestCalculateStrideDynamicShapeLastStride: block_count_idx=%d, stride=%s, ternary_ops size=%zu",
         result.block_count_idx, Str(result.stride).c_str(), result.ternary_ops.size());
}

// 测试 SetStride 函数 - 动态shape场景下合并ternary_ops到node_info
TEST_F(UTestAscirPerfV2, TestSetStrideDynamicShapeMergesTernaryOps) {
  // 构造 TensorShapeInfo
  att::TensorShapeInfo shape_info;

  Expr z2t = CreateExpr("z2t");
  Expr z1t = CreateExpr("z1t");
  Expr dynamic_stride = CreateExpr("dynamic_stride");

  // 设置动态shape场景
  shape_info.dims = {z2t, z1t, CreateExpr(90)};
  shape_info.repeats = {z2t, z1t, CreateExpr(90)};
  shape_info.gm_strides = {CreateExpr(1), CreateExpr(90), dynamic_stride};
  shape_info.strides = {CreateExpr(1), CreateExpr(90), dynamic_stride};
  shape_info.origin_repeats = {z2t, z1t, CreateExpr(90)};

  // 创建 NodeDetail
  att::NodeDetail node_info;
  node_info.name = "Nddma";
  node_info.optype = "NddmaV2";
  node_info.input_dtype = {"float16"};
  node_info.output_dtype = {"float16"};
  node_info.input_dims = {z2t, z1t, CreateExpr(90)};
  node_info.output_dims = {z2t, z1t, CreateExpr(90)};
  node_info.repeats = {z2t, z1t, CreateExpr(90)};

  const int32_t supported_max_dma_len = 2;

  // 调用 SetStride 函数
  auto status = att::SetStride(shape_info, node_info, supported_max_dma_len);
  EXPECT_EQ(status, af::SUCCESS);

  // 验证 ternary_ops 被合并到 node_info
  EXPECT_EQ(node_info.ternary_ops.size(), 2u);  // gm_stride 和 ub_stride 各一个

  GELOGD("TestSetStrideDynamicShapeMergesTernaryOps: ternary_ops size=%zu", node_info.ternary_ops.size());
}

TEST_F(UTestAscirPerfV2, TestMicroApiPerfTableSize) {
  AbsAscIrAttImplV2 default_ir_att_v2;
  EXPECT_NE(default_ir_att_v2.GetAscendCApiPerfTable(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetAddApiPerf) {
  AbsAscIrAttImplV2 default_ir_att_v2;
  EXPECT_NE(default_ir_att_v2.GetApiPerf(), nullptr);
  AddAscIrAttImplV2 add_ir_att_v2;
  EXPECT_TRUE(strcmp(af::PtrToPtr<void, af::char_t>(add_ir_att_v2.GetAscendCApiPerfTable()), "AddV2") == 0);
  EXPECT_NE(add_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetGatherApiPerf) {
  GatherAscIrAttImplV2 gather_ir_att_v2;
  EXPECT_NE(gather_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetBroadcastApiPerf) {
  BroadcastAscIrAttImplV2 broadcast_ir_att_v2;
  EXPECT_NE(broadcast_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetCastApiPerf) {
  CastAscIrAttImplV2 cast_ir_att_v2;
  EXPECT_NE(cast_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetDivApiPerf) {
  DivAscIrAttImplV2 div_ir_att_v2;
  EXPECT_NE(div_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetErfApiPerf) {
  ErfAscIrAttImplV2 erf_ir_att_v2;
  EXPECT_NE(erf_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetExpApiPerf) {
  ExpAscIrAttImplV2 exp_ir_att_v2;
  EXPECT_NE(exp_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetAbsApiPerf) {
  AbsAscIrAttImplV2 abs_ir_att_v2;
  EXPECT_NE(abs_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetLogicalAndApiPerf) {
  LogicalAndAscIrAttImplV2 logical_and_ir_att_v2;
  EXPECT_NE(logical_and_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetLogicalOrApiPerf) {
  LogicalOrAscIrAttImplV2 logical_or_ir_att_v2;
  EXPECT_NE(logical_or_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetLogicalNotApiPerf) {
  LogicalNotAscIrAttImplV2 logical_not_ir_att_v2;
  EXPECT_NE(logical_not_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetMaximumApiPerf) {
  MaximumAscIrAttImplV2 maximum_ir_att_v2;
  EXPECT_NE(maximum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetMinimumApiPerf) {
  MinimumAscIrAttImplV2 minimum_ir_att_v2;
  EXPECT_NE(minimum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetMinApiPerf) {
  MinAscIrAttImplV2 min_ir_att_v2;
  EXPECT_NE(min_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetMulApiPerf) {
  MulAscIrAttImplV2 mul_ir_att_v2;
  EXPECT_NE(mul_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetNegApiPerf) {
  NegAscIrAttImplV2 neg_ir_att_v2;
  EXPECT_NE(neg_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReciprocalApiPerf) {
  ReciprocalAscIrAttImplV2 reciprocal_ir_att_v2;
  EXPECT_NE(reciprocal_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReluApiPerf) {
  ReluAscIrAttImplV2 relu_ir_att_v2;
  EXPECT_NE(relu_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceAllApiPerf) {
  ReduceAllAscIrAttImplV2 reduce_all_ir_att_v2;
  EXPECT_NE(reduce_all_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceAnyApiPerf) {
  ReduceAnyAscIrAttImplV2 reduce_any_ir_att_v2;
  EXPECT_NE(reduce_any_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceMaxApiPerf) {
  ReduceMaxAscIrAttImplV2 reduce_max_ir_att_v2;
  EXPECT_NE(reduce_max_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceMeanApiPerf) {
  ReduceMeanAscIrAttImplV2 reduce_mean_ir_att_v2;
  EXPECT_NE(reduce_mean_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceMinApiPerf) {
  ReduceMinAscIrAttImplV2 reduce_min_ir_att_v2;
  EXPECT_NE(reduce_min_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceSumApiPerf) {
  ReduceSumAscIrAttImplV2 reduce_sum_ir_att_v2;
  EXPECT_NE(reduce_sum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetReduceProdApiPerf) {
  ReduceProdAscIrAttImplV2 reduce_prod_ir_att_v2;
  EXPECT_NE(reduce_prod_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetRemovePadApiPerf) {
  RemovePadAscIrAttImplV2 remove_pad_ir_att_v2;
  EXPECT_NE(remove_pad_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetRsqrtApiPerf) {
  RsqrtAscIrAttImplV2 rsqrt_ir_att_v2;
  EXPECT_NE(rsqrt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetGeApiPerf) {
  GeAscIrAttImplV2 ge_ir_att_v2;
  EXPECT_NE(ge_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetEqApiPerf) {
  EqAscIrAttImplV2 eq_ir_att_v2;
  EXPECT_NE(eq_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetNeApiPerf) {
  NeAscIrAttImplV2 ne_ir_att_v2;
  EXPECT_NE(ne_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetGtApiPerf) {
  GtAscIrAttImplV2 gt_ir_att_v2;
  EXPECT_NE(gt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetLeApiPerf) {
  LeAscIrAttImplV2 le_ir_att_v2;
  EXPECT_NE(le_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetLtApiPerf) {
  LtAscIrAttImplV2 lt_ir_att_v2;
  EXPECT_NE(lt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetSignApiPerf) {
  SignAscIrAttImplV2 sign_ir_att_v2;
  EXPECT_NE(sign_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetSqrtApiPerf) {
  SqrtAscIrAttImplV2 sqrt_ir_att_v2;
  EXPECT_NE(sqrt_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetSubApiPerf) {
  SubAscIrAttImplV2 sub_ir_att_v2;
  EXPECT_NE(sub_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetSumApiPerf) {
  SumAscIrAttImplV2 sum_ir_att_v2;
  EXPECT_NE(sum_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetTanhApiPerf) {
  TanhAscIrAttImplV2 tanh_ir_att_v2;
  EXPECT_NE(tanh_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetWhereApiPerf) {
  WhereAscIrAttImplV2 where_ir_att_v2;
  EXPECT_NE(where_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetUb2ubApiPerf) {
  Ub2ubAscIrAttImplV2 ub2ub_ir_att_v2;
  EXPECT_NE(ub2ub_ir_att_v2.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV2, TestGetMicroApiPerfTableInvalid) {
  PerfParamTableV2 perf_param_table_v2;
  EXPECT_EQ(perf_param_table_v2.GetVfInstructPerfTable("invalid").size(), 0);
}

TEST_F(UTestAscirPerfV2, TestLoadStoreStrideModelParamsCoverSupportedTypes) {
  const auto table = Json::parse(kParamV2Info);
  const std::vector<std::string> dtypes = {"int8",  "uint8",  "int16",   "uint16",   "int32",   "uint32",
                                           "int64", "uint64", "float16", "bfloat16", "float32", "bool"};
  for (const auto &op : {"LoadStride", "StoreStride"}) {
    const auto &params = table.at(op).at("model_params");
    EXPECT_EQ(params.size(), dtypes.size());
    for (const auto &dtype : dtypes) {
      SCOPED_TRACE(std::string(op) + ":" + dtype);
      const auto key = dtype + "to" + dtype;
      ASSERT_TRUE(params.contains(key));
      EXPECT_FLOAT_EQ(params.at(key).at("k").get<float>(), std::string(op) == "LoadStride" ? 0.005f : 0.0385f);
      EXPECT_FLOAT_EQ(params.at(key).at("u").get<float>(), 4096.0f);
      EXPECT_FLOAT_EQ(params.at(key).at("penalty_coeff").get<float>(), 0.0f);
    }
    EXPECT_FALSE(params.contains("int4toint4"));
  }
}

TEST_F(UTestAscirPerfV2, TestLoadStoreStrideUsesConfiguredUpperBound) {
  const Expr dynamic_stride = CreateExpr("stride");
  for (const float upper_bound : {2048.0f, 8192.0f}) {
    const std::map<std::string, float> params = {{"k", 1.0f}, {"u", upper_bound}, {"data_type_size", 2.0f}};
    for (const Expr &stride : {CreateExpr(100), CreateExpr(5000), dynamic_stride}) {
      SCOPED_TRACE(std::to_string(upper_bound) + ":" + Str(stride));
      Expr result;
      ASSERT_EQ(LoadStoreStrideV2Func(params, {CreateExpr(1)}, stride, result), af::SUCCESS);
      Expr expected = CreateExpr(1.0f) * af::sym::Min(stride * CreateExpr(2), CreateExpr(upper_bound));
      result.Simplify();
      expected.Simplify();
      EXPECT_EQ(Str(result), Str(expected));
    }
  }
}

TEST_F(UTestAscirPerfV2, TestApiNameNotRegistered) {
  const auto api_perf = GetApiPerf("invalid");
  EXPECT_EQ(api_perf, nullptr);
}

TEST_F(UTestAscirPerfV2, TestCompareGeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareV2StoreUsesDtypeMappingPerf) {
  auto compare_params = MakeCompareParams(false, {CreateExpr(129)}, {CreateExpr(1)}, {CreateExpr(1)});
  EXPECT_EQ(GetComparePerfExpr("GeV2", kInt32, compare_params), "76");
}

TEST_F(UTestAscirPerfV2, TestCompareV2UsesRegbaseMicroApiCost) {
  auto compare_params = MakeCompareParams(false, {CreateExpr(129)}, {CreateExpr(1)}, {CreateExpr(1)});
  EXPECT_EQ(GetComparePerfExpr("GeV2", kFloat16, compare_params), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareV2ScalarAddsScalarDuplicateCost) {
  auto compare_params = MakeCompareParams(true, {CreateExpr(129)}, {CreateExpr(1)}, {CreateExpr(1)});
  EXPECT_EQ(GetComparePerfExpr("GeV2", kFloat16, compare_params), "65");
}

TEST_F(UTestAscirPerfV2, TestCompareEqV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("EqV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareNeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("NeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareGtV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareLeV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareLtV2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "64");
}

TEST_F(UTestAscirPerfV2, TestCompareEqInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("EqV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestCompareNeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("NeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestCompareGtInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestCompareGeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("GeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestCompareLtInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LtV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestCompareLeInt64V2) {
  auto cmp_v2 = ApiPerfFactory::Instance().Create("LeV2");
  ASSERT_NE(cmp_v2, nullptr);
  auto cmp_v2_perf = cmp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "int64";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "uint8";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cmp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "78");
}

TEST_F(UTestAscirPerfV2, TestGetOpHeadCostValid) {
  PerfParamTableV2 perf_param_table_v2;
  auto head_cost = perf_param_table_v2.GetOpHeadCost();
  EXPECT_TRUE(head_cost.IsConstExpr());
  uint64_t head_cost_val = 0L;
  EXPECT_TRUE(head_cost.GetConstValue(head_cost_val));
  EXPECT_EQ(head_cost_val, 0);
}

TEST_F(UTestAscirPerfV2, TestAbsV2) {
  auto abs_v2 = ApiPerfFactory::Instance().Create("AbsV2");
  ASSERT_NE(abs_v2, nullptr);
  auto abs_v2_perf = abs_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  abs_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(UTestAscirPerfV2, TestExpV2) {
  auto exp_v2 = ApiPerfFactory::Instance().Create("ExpV2");
  ASSERT_NE(exp_v2, nullptr);
  auto exp_v2_perf = exp_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  exp_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "45");
}

TEST_F(UTestAscirPerfV2, TestLnV2) {
  auto ln_v2 = ApiPerfFactory::Instance().Create("LnV2");
  ASSERT_NE(ln_v2, nullptr);
  auto ln_v2_perf = ln_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  ln_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "47");
}

TEST_F(UTestAscirPerfV2, TestSqrtV2) {
  auto sqrt_v2 = ApiPerfFactory::Instance().Create("SqrtV2");
  ASSERT_NE(sqrt_v2, nullptr);
  auto sqrt_v2_perf = sqrt_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  sqrt_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(UTestAscirPerfV2, TestDivV2) {
  auto div_v2 = ApiPerfFactory::Instance().Create("DivV2");
  ASSERT_NE(div_v2, nullptr);
  auto div_v2_perf = div_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  div_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(UTestAscirPerfV2, TestTrueDivV2) {
  auto div_v2 = ApiPerfFactory::Instance().Create("TrueDivV2");
  ASSERT_NE(div_v2, nullptr);
  auto div_v2_perf = div_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  div_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "46");
}

TEST_F(UTestAscirPerfV2, TestRsqrtV2) {
  auto rsqrt_v2 = ApiPerfFactory::Instance().Create("RsqrtV2");
  ASSERT_NE(rsqrt_v2, nullptr);
  auto rsqrt_v2_perf = rsqrt_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  rsqrt_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "61");
}

TEST_F(UTestAscirPerfV2, TestReciprocalV2) {
  auto reciprocal_v2 = ApiPerfFactory::Instance().Create("ReciprocalV2");
  ASSERT_NE(reciprocal_v2, nullptr);
  auto reciprocal_v2_perf = reciprocal_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  reciprocal_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49");
}

TEST_F(UTestAscirPerfV2, TestReluV2) {
  auto relu_v2 = ApiPerfFactory::Instance().Create("ReluV2");
  ASSERT_NE(relu_v2, nullptr);
  auto relu_v2_perf = relu_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  relu_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(UTestAscirPerfV2, TestMaxV2) {
  auto max_v2 = ApiPerfFactory::Instance().Create("MaxV2");
  ASSERT_NE(max_v2, nullptr);
  auto max_v2_perf = max_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(8);
  att::Expr dim1 = CreateExpr(64);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input.repeats = input.dims;
  input.strides = {dim1, CreateExpr(1)};
  input.gm_strides = input.strides;
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output.dims.push_back(CreateExpr(1));
  output.repeats = output.dims;
  output.strides = {CreateExpr(1), CreateExpr(0)};
  output.gm_strides = output.strides;
  output_shapes.emplace_back(output);

  NodeInfo node;
  SetSingleReduceSpecificParams(node);
  PerfOutputInfo perf_res;
  max_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_FALSE(Str(res).empty());
}

TEST_F(UTestAscirPerfV2, TestAnyV2) {
  auto any_v2 = ApiPerfFactory::Instance().Create("AnyV2");
  ASSERT_NE(any_v2, nullptr);
  auto any_v2_perf = any_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 4U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(8);
  att::Expr dim1 = CreateExpr(64);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input.repeats = input.dims;
  input.strides = {dim1, CreateExpr(1)};
  input.gm_strides = input.strides;
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 4U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output.dims.push_back(CreateExpr(1));
  output.repeats = output.dims;
  output.strides = {CreateExpr(1), CreateExpr(0)};
  output.gm_strides = output.strides;
  output_shapes.emplace_back(output);

  NodeInfo node;
  SetSingleReduceSpecificParams(node);
  PerfOutputInfo perf_res;
  any_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_FALSE(Str(res).empty());
}

TEST_F(UTestAscirPerfV2, TestMaximumV2) {
  auto max_v2 = ApiPerfFactory::Instance().Create("MaximumV2");
  ASSERT_NE(max_v2, nullptr);
  auto max_v2_perf = max_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  max_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(UTestAscirPerfV2, TestMinV2) {
  auto min_v2 = ApiPerfFactory::Instance().Create("MinV2");
  ASSERT_NE(min_v2, nullptr);
  auto min_v2_perf = min_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float16";
  att::Expr dim0 = CreateExpr(8);
  att::Expr dim1 = CreateExpr(64);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input.repeats = input.dims;
  input.strides = {dim1, CreateExpr(1)};
  input.gm_strides = input.strides;
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output.dims.push_back(CreateExpr(1));
  output.repeats = output.dims;
  output.strides = {CreateExpr(1), CreateExpr(0)};
  output.gm_strides = output.strides;
  output_shapes.emplace_back(output);

  NodeInfo node;
  SetSingleReduceSpecificParams(node);
  PerfOutputInfo perf_res;
  min_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_FALSE(Str(res).empty());
}

TEST_F(UTestAscirPerfV2, TestAllV2) {
  auto all_v2 = ApiPerfFactory::Instance().Create("AllV2");
  ASSERT_NE(all_v2, nullptr);
  auto all_v2_perf = all_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 4U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(8);
  att::Expr dim1 = CreateExpr(64);
  input.dims.push_back(dim0);
  input.dims.push_back(dim1);
  input.repeats = input.dims;
  input.strides = {dim1, CreateExpr(1)};
  input.gm_strides = input.strides;
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 4U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output.dims.push_back(CreateExpr(1));
  output.repeats = output.dims;
  output.strides = {CreateExpr(1), CreateExpr(0)};
  output.gm_strides = output.strides;
  output_shapes.emplace_back(output);

  NodeInfo node;
  SetSingleReduceSpecificParams(node);
  PerfOutputInfo perf_res;
  all_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_FALSE(Str(res).empty());
}

TEST_F(UTestAscirPerfV2, TestMinimumV2) {
  auto min_v2 = ApiPerfFactory::Instance().Create("MinimumV2");
  ASSERT_NE(min_v2, nullptr);
  auto min_v2_perf = min_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  min_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "26");
}

TEST_F(UTestAscirPerfV2, TestNegV2) {
  auto neg_v2 = ApiPerfFactory::Instance().Create("NegV2");
  ASSERT_NE(neg_v2, nullptr);
  auto neg_v2_perf = neg_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  neg_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(UTestAscirPerfV2, TestMeanV2) {
  auto mean_v2 = ApiPerfFactory::Instance().Create("MeanV2");
  ASSERT_NE(mean_v2, nullptr);
  auto mean_v2_perf = mean_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  mean_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(UTestAscirPerfV2, TestAddV2) {
  auto add_v2 = ApiPerfFactory::Instance().Create("AddV2");
  ASSERT_NE(add_v2, nullptr);
  auto add_v2_perf = add_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  add_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "27");
}

TEST_F(UTestAscirPerfV2, TestSubV2) {
  auto sub_v2 = ApiPerfFactory::Instance().Create("SubV2");
  ASSERT_NE(sub_v2, nullptr);
  auto sub_v2_perf = sub_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  sub_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "27");
}

TEST_F(UTestAscirPerfV2, TestMulV2) {
  auto mul_v2 = ApiPerfFactory::Instance().Create("MulV2");
  ASSERT_NE(mul_v2, nullptr);
  auto mul_v2_perf = mul_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  mul_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(UTestAscirPerfV2, TestProdV2) {
  auto prod_v2 = ApiPerfFactory::Instance().Create("ProdV2");
  ASSERT_NE(prod_v2, nullptr);
  auto prod_v2_perf = prod_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  prod_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "28");
}

TEST_F(UTestAscirPerfV2, TestLeakyReluV2) {
  auto leaky_relu_v2 = ApiPerfFactory::Instance().Create("LeakyReluV2");
  ASSERT_NE(leaky_relu_v2, nullptr);
  auto leaky_relu_v2_perf = leaky_relu_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float32";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  leaky_relu_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "25");
}

TEST_F(UTestAscirPerfV2, TestCastV2) {
  auto cast_v2 = ApiPerfFactory::Instance().Create("CastV2");
  ASSERT_NE(cast_v2, nullptr);
  auto cast_v2_perf = cast_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  cast_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  EXPECT_EQ(Str(res), "30");
}

TEST_F(UTestAscirPerfV2, TestCastV2SameBitIntegerUsesDataCopyCostOnly) {
  auto cast_params =
      MakeCastParams({CreateExpr(2), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)}, {CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(GetCastPerfExpr("CastV2", "int8", "uint8", cast_params), "52");
}

TEST_F(UTestAscirPerfV2, TestCastV2UInt8ToInt64UsesInterleaveCost) {
  auto cast_params = MakeCastParams({CreateExpr(64)}, {CreateExpr(1)}, {CreateExpr(1)});
  EXPECT_EQ(GetCastPerfExpr("CastV2", "uint8", "int64", cast_params), "42");
}

TEST_F(UTestAscirPerfV2, TestCastV2Int64ToUInt8UsesPackCost) {
  auto cast_params = MakeCastParams({CreateExpr(64)}, {CreateExpr(1)}, {CreateExpr(1)});
  EXPECT_EQ(GetCastPerfExpr("CastV2", "int64", "uint8", cast_params), "38");
}

TEST_F(UTestAscirPerfV2, TestCastV2DynamicStrideCreatesRepeatTernary) {
  auto cast_v2 = ApiPerfFactory::Instance().Create("CastV2");
  ASSERT_NE(cast_v2, nullptr);
  auto cast_v2_perf = cast_v2->GetPerfFunc();
  const Expr dim = CreateExpr(129);
  std::vector<att::TensorShapeInfo> input_shapes = {MakeCastShape("float32", dim)};
  std::vector<att::TensorShapeInfo> output_shapes = {MakeCastShape("float16", dim)};
  NodeInfo node;
  node.cast_node_params =
      MakeCastParams({CreateExpr("outer"), CreateExpr("inner")}, {CreateExpr("out_stride"), CreateExpr(1)},
                     {CreateExpr("in_stride"), CreateExpr(1)});
  PerfOutputInfo perf_res;
  EXPECT_EQ(cast_v2_perf(input_shapes, output_shapes, node, perf_res), af::SUCCESS);
  EXPECT_EQ(perf_res.ternary_ops.size(), 2U);
  EXPECT_NE(Str(perf_res.pipe_res[PipeType::AIV_VEC]).find("cast_repeat_time"), std::string::npos);
  EXPECT_NE(Str(perf_res.pipe_res[PipeType::AIV_VEC]).find("cast_call_count"), std::string::npos);
}

TEST_F(UTestAscirPerfV2, TestCastV2DynamicCallCountMultipliesFinalCost) {
  auto cast_params =
      MakeCastParams({CreateExpr(2), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)}, {CreateExpr(64), CreateExpr(1)});
  EXPECT_EQ(GetCastPerfExpr("CastV2", "float32", "float16", cast_params),
            "(((2 * cast_repeat_time) + 24) * cast_call_count)");
}

TEST_F(UTestAscirPerfV2, TestCeil2IntV2ReusesCastV2Formula) {
  auto cast_params = MakeCastParams({CreateExpr(129)}, {CreateExpr(1)}, {CreateExpr(1)});
  for (const auto &input_dtype : {"float32", "float16", "bfloat16"}) {
    SCOPED_TRACE(input_dtype);
    const std::string ceil2int_expr = GetCastPerfExpr("Ceil2IntV2", input_dtype, "int32", cast_params);
    EXPECT_EQ(ceil2int_expr, GetCastPerfExpr("CastV2", input_dtype, "int32", cast_params));
    EXPECT_NE(ceil2int_expr, "0");
  }
  EXPECT_EQ(GetCastPerfExpr("Ceil2IntV2", "float32", "int32", cast_params), "34");
}

TEST_F(UTestAscirPerfV2, TestSumV2) {
  auto sum_v2 = ApiPerfFactory::Instance().Create("SumV2");
  ASSERT_NE(sum_v2, nullptr);
  auto sum_v2_perf = sum_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  sum_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "(39 + reduce_sum_node)");
}

TEST_F(UTestAscirPerfV2, TestRemovePadV2) {
  auto removepad_v2 = ApiPerfFactory::Instance().Create("RemovePadV2");
  ASSERT_NE(removepad_v2, nullptr);
  auto removepad_v2_perf = removepad_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  removepad_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "50");
}

TEST_F(UTestAscirPerfV2, TestWhereV2) {
  auto where_v2 = ApiPerfFactory::Instance().Create("WhereV2");
  ASSERT_NE(where_v2, nullptr);
  auto where_v2_perf = where_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  where_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(UTestAscirPerfV2, TestWhereImplSkipMaskPerfForB8B16B32) {
  EXPECT_EQ(GetWhereImplPerfExpr(kUInt8), "48");
  EXPECT_EQ(GetWhereImplPerfExpr(kFloat16), "56");
  EXPECT_EQ(GetWhereImplPerfExpr(kFloat32), "61");
}

TEST_F(UTestAscirPerfV2, TestWhereImplKeepsMaskPerfForB64) {
  EXPECT_EQ(GetWhereImplPerfExpr(kInt64), "72");
}

TEST_F(UTestAscirPerfV2, TestWhereComputeSkipMaskPerfForB16B32) {
  EXPECT_EQ(GetWhereComputePerfExpr(kFloat16), "52");
  EXPECT_EQ(GetWhereComputePerfExpr(kFloat32), "56");
}

TEST_F(UTestAscirPerfV2, TestWhereComputeKeepsMaskPerfForB8B64) {
  EXPECT_EQ(GetWhereComputePerfExpr(kUInt8), "50");
  EXPECT_EQ(GetWhereComputePerfExpr(kInt64), "72");
}

TEST_F(UTestAscirPerfV2, TestWhereComputeB16B32MultipliesDim0AfterInnerCost) {
  const auto output_dims = std::vector<Expr>{CreateExpr("where_dim0"), CreateExpr("where_dim1")};

  EXPECT_EQ(GetWhereComputePerfExpr(kFloat16, output_dims),
            "((((3 * Ceiling((Rational(1 , 128) * where_dim1))) + 4) * where_dim0) + 45)");
  EXPECT_EQ(GetWhereComputePerfExpr(kFloat32, output_dims),
            "((((4 * Ceiling((Rational(1 , 64) * where_dim1))) + 3) * where_dim0) + 49)");
}

TEST_F(UTestAscirPerfV2, TestSelectV2) {
  auto select_v2 = ApiPerfFactory::Instance().Create("SelectV2");
  ASSERT_NE(select_v2, nullptr);
  auto select_v2_perf = select_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  select_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "36");
}

TEST_F(UTestAscirPerfV2, TestPowV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("PowV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "307");
}

TEST_F(UTestAscirPerfV2, TestErfV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("ErfV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "154");
}

TEST_F(UTestAscirPerfV2, TestTanhV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("TanhV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "85");
}

TEST_F(UTestAscirPerfV2, TestSigmoidV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("SigmoidV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "76");
}

TEST_F(UTestAscirPerfV2, TestGeluV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("GeluV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "109");
}

TEST_F(UTestAscirPerfV2, TestSignV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("SignV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "49");
}

TEST_F(UTestAscirPerfV2, TestLogicalNotV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalNotV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "42");
}

TEST_F(UTestAscirPerfV2, TestLogicalOrV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalOrV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "51");
}

TEST_F(UTestAscirPerfV2, TestLogicalAndV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("LogicalAndV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "51");
}

TEST_F(UTestAscirPerfV2, TestClipByValueV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("ClipByValueV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "35");
}

TEST_F(UTestAscirPerfV2, TestBitwiseAndV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("BitwiseAndV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "35");
}

TEST_F(UTestAscirPerfV2, TestFloorDivV2) {
  auto pow_v2 = ApiPerfFactory::Instance().Create("FloorDivV2");
  ASSERT_NE(pow_v2, nullptr);
  auto pow_v2_perf = pow_v2->GetPerfFunc();
  std::vector<att::TensorShapeInfo> input_shapes;
  std::vector<att::TensorShapeInfo> output_shapes;
  att::TensorShapeInfo input;
  input.data_type_size = 2U;
  input.loc = att::HardwareDef::UB;
  input.data_type = "float32";
  att::Expr dim0 = CreateExpr(129);
  input.dims.push_back(dim0);
  input_shapes.emplace_back(input);
  input_shapes.emplace_back(input);

  att::TensorShapeInfo output;
  output.data_type_size = 2U;
  output.loc = att::HardwareDef::UB;
  output.data_type = "float16";
  output.dims.push_back(dim0);
  output_shapes.emplace_back(output);
  NodeInfo node;
  PerfOutputInfo perf_res;
  pow_v2_perf(input_shapes, output_shapes, node, perf_res);
  Expr res = perf_res.pipe_res[PipeType::AIV_VEC];
  std::cout << Str(res) << std::endl;
  EXPECT_EQ(Str(res), "80");
}

// ============================================================================
// ReorderGmStrideByTranspose 函数 UT 测试 - 100% 覆盖所有分支
// ============================================================================

// 测试用例 1: 空 outputs 的情况
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeEmptyOutputs) {
  // 创建一个没有输入的节点来测试空 outputs 分支
  af::AscNodePtr node_ptr = GraphConstructUtils::ConstructSingleOp("TestOp", 0, 1);
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30), CreateExpr(1)};

  // 应该成功返回，不崩溃（因为 node->outputs.Size() == 0）
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 2: 空 gm_strides 的情况
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeEmptyGmStrides) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides.clear();  // 空 gm_strides

  // 应该成功返回，不崩溃
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 3: 空 vectorized_axis 的情况
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeEmptyVectorizedAxis) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30), CreateExpr(1)};

  // 清空 vectorized_axis 来测试空 vectorized_axis 分支
  node_ptr->outputs[0].attr.vectorized_axis.clear();

  // 应该成功返回，不崩溃
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 4: 空 axis 的情况
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeEmptyAxis) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30), CreateExpr(1)};

  // 清空 axis 来测试空 axis 分支
  node_ptr->outputs[0].attr.axis.clear();

  // 应该成功返回，不崩溃
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 5: 无转置的情况 (vectorized_axis == axis)
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeNoTranspose) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30), CreateExpr(1)};

  // 设置相同的 vectorized_axis 和 axis（无转置）
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 2};
  node_ptr->outputs[0].attr.axis = {0, 1, 2};

  std::vector<Expr> original_strides = tensor.gm_strides;

  // 应该成功返回，且 gm_strides 不应该改变
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));

  // 验证 gm_strides 没有被修改
  EXPECT_EQ(tensor.gm_strides.size(), original_strides.size());
  for (size_t i = 0; i < tensor.gm_strides.size(); ++i) {
    EXPECT_EQ(Str(tensor.gm_strides[i]), Str(original_strides[i]));
  }
}

// 测试用例 8: 边界情况 - vectorized_axis 和 axis 大小不匹配
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeSizeMismatch) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30), CreateExpr(1)};

  // vectorized_axis 有 3 个元素，axis 有 4 个元素
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 2};
  node_ptr->outputs[0].attr.axis = {0, 1, 2, 3};

  // 应该成功返回，不崩溃
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 9: 边界情况 - gm_strides 维度少于 axis
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeGmStridesFewer) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  tensor.dims = {CreateExpr(10), CreateExpr(20), CreateExpr(30), CreateExpr(40)};
  tensor.gm_strides = {CreateExpr(600), CreateExpr(30)};  // 只有 2 个元素

  // 设置 4 个元素的 axis
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 2, 3};
  node_ptr->outputs[0].attr.axis = {0, 1, 2, 3};

  // 应该成功返回，不崩溃
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));
}

// 测试用例 10: 复杂转置 - 多个轴同时转置
TEST_F(UTestAscirPerfV2, TestReorderGmStrideByTransposeComplex) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  Expr d0 = CreateExpr("d0");
  Expr d1 = CreateExpr("d1");
  Expr d2 = CreateExpr("d2");
  Expr d3 = CreateExpr("d3");
  Expr d4 = CreateExpr("d4");

  tensor.dims = {d0, d1, d2, d3, d4};

  // 原始顺序 [0,1,2,3,4] 转置为 [4,3,2,1,0]
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 2, 3, 4};
  node_ptr->outputs[0].attr.axis = {4, 3, 2, 1, 0};

  // 转置后的 gm_strides (按 axis [4,3,2,1,0] 顺序)
  tensor.gm_strides = {CreateExpr("stride0"), CreateExpr("stride1"), CreateExpr("stride2"), CreateExpr("stride3"),
                       CreateExpr("stride4")};
  node_ptr->outputs[0].attr.strides = tensor.gm_strides;

  // 执行重排
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));

  // 当前实现只重排 repeats，不重排 gm_strides
  // gm_strides 应该保持不变
  EXPECT_EQ(Str(tensor.gm_strides[0]), "stride0");
  EXPECT_EQ(Str(tensor.gm_strides[1]), "stride1");
  EXPECT_EQ(Str(tensor.gm_strides[2]), "stride2");
  EXPECT_EQ(Str(tensor.gm_strides[3]), "stride3");
  EXPECT_EQ(Str(tensor.gm_strides[4]), "stride4");
}

// 测试用例 11: 复杂转置 - axis与gm_strides个数不一致，多个轴同时转置
TEST_F(UTestAscirPerfV2, TestReorderGmStrideMismatchUbStrideByTransposeComplex) {
  af::AscGraph asc_graph("test");
  EXPECT_EQ(af::SUCCESS, GraphConstructUtils::CreateSimpleLoadStoreOp(asc_graph));
  af::AscNodePtr node_ptr = asc_graph.FindNode("load");
  ASSERT_NE(node_ptr, nullptr);

  att::TensorShapeInfo tensor;
  Expr d0 = CreateExpr("d0");
  Expr d1 = CreateExpr("d1");
  Expr d2 = CreateExpr("d2");
  Expr d3 = CreateExpr("d3");
  Expr d4 = CreateExpr("d4");

  tensor.dims = {d0, d1, d3};
  tensor.repeats = {d0, d1, d3};

  // 原始顺序 [0,1,2,3,4] 转置为 [4,3,2,1,0]
  node_ptr->outputs[0].attr.vectorized_axis = {0, 1, 3};
  node_ptr->outputs[0].attr.axis = {4, 3, 2, 1, 0};

  // 转置前的 gm_strides 已经按照gm的顺序排列
  tensor.gm_strides = {CreateExpr("stride0"), CreateExpr("stride1"), CreateExpr("stride3")};
  node_ptr->outputs[0].attr.repeats = {d4, d3, d2, d1, d0};
  node_ptr->outputs[0].attr.strides = {CreateExpr("stride4"), CreateExpr("stride3"), CreateExpr("stride2"),
                                       CreateExpr("stride1"), CreateExpr("stride0")};
  // 执行重排
  EXPECT_EQ(af::SUCCESS, ReorderGmStrideByTranspose(node_ptr, tensor));

  // 验证重排后的 gm_strides 符合原始 vectorized_axis 顺序
  EXPECT_EQ(Str(tensor.repeats[0]), "d3");
  EXPECT_EQ(Str(tensor.repeats[1]), "d1");
  EXPECT_EQ(Str(tensor.repeats[2]), "d0");
}

// GetDmaParams swap=false：标准窗口，取最后 kMaxDmaLen 维
TEST_F(UTestAscirPerfV2, TestGetDmaParamsNoSwap) {
  Expr z1z2z3t = CreateExpr("z1z2z3t");
  Expr z0t = CreateExpr("z0t");
  vector<Expr> dims = {z1z2z3t, z0t, CreateExpr(34)};
  Expr outer_repeat;
  vector<Expr> used_dims;
  const int32_t kMaxDmaLen = 2;

  EXPECT_EQ(af::SUCCESS, att::GetDmaParams(dims, outer_repeat, used_dims, kMaxDmaLen, false));

  ASSERT_EQ(used_dims.size(), 2U);
  EXPECT_EQ(Str(used_dims[0]), "z0t");
  EXPECT_EQ(Str(used_dims[1]), "34");
  EXPECT_EQ(Str(outer_repeat), "z1z2z3t");
}

// GetDmaParams swap=true：将倒数第3维换入，倒数第2维外抛
// dims=[z1z2z3t, z0t, 34], kMaxDmaLen=2
// 期望: used_dims=[z1z2z3t, 34], outer_repeat=z0t
TEST_F(UTestAscirPerfV2, TestGetDmaParamsSwap) {
  Expr z1z2z3t = CreateExpr("z1z2z3t");
  Expr z0t = CreateExpr("z0t");
  vector<Expr> dims = {z1z2z3t, z0t, CreateExpr(34)};
  Expr outer_repeat;
  vector<Expr> used_dims;
  const int32_t kMaxDmaLen = 2;

  EXPECT_EQ(af::SUCCESS, att::GetDmaParams(dims, outer_repeat, used_dims, kMaxDmaLen, true));

  ASSERT_EQ(used_dims.size(), 2U);
  EXPECT_EQ(Str(used_dims[0]), "z1z2z3t");
  EXPECT_EQ(Str(used_dims[1]), "34");
  EXPECT_EQ(Str(outer_repeat), "z0t");
}

// GetDmaParams swap=true，4维场景：kMaxDmaLen=4
// dims=[A, B, C, D, E]，swap 将倒数第5维换入，倒数第4维外抛
// 期望: used_dims=[A, C, D, E], outer_repeat=B
TEST_F(UTestAscirPerfV2, TestGetDmaParamsSwapFourDim) {
  Expr a = CreateExpr("A");
  Expr b = CreateExpr("B");
  Expr c = CreateExpr("C");
  Expr d = CreateExpr("D");
  Expr e = CreateExpr("E");
  vector<Expr> dims = {a, b, c, d, e};
  Expr outer_repeat;
  vector<Expr> used_dims;
  const int32_t kMaxDmaLen = 4;

  EXPECT_EQ(af::SUCCESS, att::GetDmaParams(dims, outer_repeat, used_dims, kMaxDmaLen, true));

  ASSERT_EQ(used_dims.size(), 4U);
  EXPECT_EQ(Str(used_dims[0]), "A");
  EXPECT_EQ(Str(used_dims[1]), "C");
  EXPECT_EQ(Str(used_dims[2]), "D");
  EXPECT_EQ(Str(used_dims[3]), "E");
  EXPECT_EQ(Str(outer_repeat), "B");
}

// LoadV2: GM非连续 + block_len < cache_line(128B) 时，ExpandLoadBlockLen将block_len pad到cache_line
TEST_F(UTestAscirPerfV2, TestLoadApiExpandBlockLenSmallBlock) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");

  input_shapes[0].data_type = "float16";
  // block_len = 8 elements * 2B = 16B < 128B cache line, 且 gm_stride > 0
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(8)};
  input_shapes[0].gm_strides = {z0t_size * CreateExpr(64), CreateExpr(64), CreateExpr(1)};
  input_shapes[0].strides = input_shapes[0].gm_strides;
  output_shapes[0].data_type = "float16";
  output_shapes[0].repeats = input_shapes[0].repeats;
  output_shapes[0].gm_strides = input_shapes[0].gm_strides;
  output_shapes[0].strides = output_shapes[0].gm_strides;
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // ExpandLoadBlockLen: block_len(8)*2=16B < 128B, stride>0 → pad to 64(=128/2)
  // data_size = z0t * z1t * 64 * 2 = 128 * z0t * z1t, SmallBlk case
  // stride_cost = 0.005 * z0t * z1t * min((64-8)*2, 4096) = 0.56 * z0t * z1t
  const std::string expect_stride = "(0.559999987483025 * z0t_size * z1t_size)";
  const std::string load_perf = "(128 * z0t_size * z1t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236)))";
  EXPECT_EQ(Str(res.Replace(ret)), "(" + expect_stride + " + " + load_perf + " + 160.0)");
}

// LoadV2: codegen 对非连续尾轴补 repeat=1，ATT 必须保留合轴后的 block_count。
TEST_F(UTestAscirPerfV2, TestLoadApiCodegenTailAxisCase0) {
  TensorShapeInfo shape;
  const Expr z0z1t_size = CreateExpr("z0z1t_size");
  shape.data_type = "float16";
  shape.dims = {CreateExpr(128) * z0z1t_size};
  shape.repeats = {z0z1t_size, CreateExpr(128)};
  shape.strides = {CreateExpr(128), CreateExpr(1)};
  shape.gm_strides = {CreateExpr(384), CreateExpr(3)};

  NodeDetail node;
  node.optype = kLoad;
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  ASSERT_EQ(node.vectorized_shape.size(), 2U);
  EXPECT_EQ(Str(node.vectorized_shape[0]), "(128 * z0z1t_size)");
  EXPECT_EQ(Str(node.vectorized_shape[1]), "1");
}

// LoadV2: 单根合轴后的 case1 也需要补 repeat=1，不能只依赖 repeats.size()>1。
TEST_F(UTestAscirPerfV2, TestLoadApiCodegenTailAxisCase1) {
  TensorShapeInfo shape;
  const Expr z2t_size = CreateExpr("z2t_size");
  shape.data_type = "float16";
  shape.dims = {z2t_size};
  shape.repeats = {z2t_size};
  shape.strides = {CreateExpr(1)};
  shape.gm_strides = {CreateExpr(3)};

  NodeDetail node;
  node.optype = kLoad;
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  ASSERT_EQ(node.vectorized_shape.size(), 2U);
  EXPECT_EQ(Str(node.vectorized_shape[0]), "z2t_size");
  EXPECT_EQ(Str(node.vectorized_shape[1]), "1");
}

TEST_F(UTestAscirPerfV2, TestLoadApiCodegenTailAxisStrideMatchesDataCopy) {
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.dims = {CreateExpr(128)};
  shape.repeats = {CreateExpr(128)};
  shape.strides = {CreateExpr(1)};
  shape.gm_strides = {CreateExpr(3)};

  NodeDetail node;
  node.optype = kLoad;
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  ASSERT_EQ(SetStride(shape, node, 4), af::SUCCESS);
  // DataCopyPadExtend emits src_stride = GM stride - block_len = 3 - 1.
  EXPECT_EQ(Str(node.gm_stride), "2");
}

TEST_F(UTestAscirPerfV2, TestLoadApiCodegenDynamicTailAxisStrideMatchesDataCopy) {
  TensorShapeInfo shape;
  const Expr dynamic_stride = CreateExpr("dynamic_stride");
  shape.data_type = "float16";
  shape.dims = {CreateExpr(8), CreateExpr(16), CreateExpr(32)};
  shape.repeats = shape.dims;
  shape.origin_repeats = shape.repeats;
  shape.strides = {CreateExpr(512), CreateExpr(32), CreateExpr(1)};
  shape.gm_strides = {CreateExpr(1536), CreateExpr(96), dynamic_stride};

  NodeDetail node;
  node.optype = kLoad;
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  ASSERT_FALSE(node.vectorized_shape.empty());

  const auto result = CalculateStride(shape, false, node, 4, false);
  ASSERT_EQ(result.ternary_ops.size(), 1U);
  const auto if_case = result.ternary_ops.at(result.stride).DeepCopyIfCase();
  ASSERT_NE(if_case, nullptr);
  ASSERT_NE(if_case->GetChoiceA(), nullptr);
  EXPECT_EQ(Str(if_case->GetCondLeft()), "dynamic_stride");
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(if_case->GetChoiceA()->GetExpr(), dynamic_stride - CreateExpr(1)),
            af::TriBool::kTrue);
}

TEST_F(UTestAscirPerfV2, TestLoadApiCodegenTailAxisSwapUsesVectorizedShape) {
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.data_type_size = 2;
  shape.dims = {CreateExpr(8), CreateExpr(16)};
  shape.repeats = shape.dims;
  shape.origin_repeats = shape.repeats;
  shape.strides = {CreateExpr(16), CreateExpr(1)};
  shape.gm_strides = {CreateExpr(48), CreateExpr(3)};

  NodeDetail node;
  node.name = "LoadWithTailAxis";
  node.optype = kLoad;
  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  ASSERT_EQ(node.input_dims.size(), 2U);
  ASSERT_EQ(node.vectorized_shape.size(), 3U);

  PerfOutputInfo perf;
  ASSERT_EQ(GetDmaPerf(shape, node, perf, 2, true), af::SUCCESS);
  EXPECT_NE(perf.pipe_res.find(PipeType::AIV_MTE2), perf.pipe_res.end());
  EXPECT_TRUE(perf.ternary_ops.empty());
}

TEST_F(UTestAscirPerfV2, TestLoadApiCodegenTailAxisContinuous) {
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.dims = {CreateExpr(128)};
  shape.repeats = {CreateExpr(128)};
  shape.strides = {CreateExpr(1)};
  shape.gm_strides = {CreateExpr(1)};

  NodeDetail node;
  node.optype = kLoad;
  ASSERT_EQ(SetDims(shape, node), af::SUCCESS);
  EXPECT_TRUE(node.vectorized_shape.empty());
}

// LoadV2: GM非连续 + block_len >= cache_line(128B) 时，不触发ExpandLoadBlockLen
TEST_F(UTestAscirPerfV2, TestLoadApiExpandBlockLenLargeBlock) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);
  Expr z0t_size = CreateExpr("z0t_size");
  Expr z1t_size = CreateExpr("z1t_size");

  input_shapes[0].data_type = "float16";
  // block_len = 128 elements * 2B = 256B > 128B cache line, gm_stride > 0 但不触发
  input_shapes[0].repeats = {z0t_size, z1t_size, CreateExpr(128)};
  input_shapes[0].gm_strides = {z0t_size * CreateExpr(256), CreateExpr(256), CreateExpr(1)};
  input_shapes[0].strides = input_shapes[0].gm_strides;
  output_shapes[0].data_type = "float16";
  output_shapes[0].repeats = input_shapes[0].repeats;
  output_shapes[0].gm_strides = input_shapes[0].gm_strides;
  output_shapes[0].strides = output_shapes[0].gm_strides;
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // block_len=128*2=256B > 128B, 不触发pad; SmallBlk case
  // data_size = 256 * z0t * z1t
  // stride = 0.005 * z0t * z1t * min((256-128)*2, 4096) = 1.28 * z0t * z1t
  const std::string expect_stride = "(1.27999997138977 * z0t_size * z1t_size)";
  const std::string load_perf = "(256 * z0t_size * z1t_size / (((6.40880012512207 / (block_dim)) + 13.1354999542236)))";
  EXPECT_EQ(Str(res.Replace(ret)), "(" + expect_stride + " + " + load_perf + " + 160.0)");
}

// LoadV2: 连续场景(gm_stride=0)，不触发ExpandLoadBlockLen
TEST_F(UTestAscirPerfV2, TestLoadApiExpandBlockLenContinuous) {
  std::vector<att::TensorShapeInfo> input_shapes(1);
  std::vector<att::TensorShapeInfo> output_shapes(1);

  input_shapes[0].data_type = "float16";
  input_shapes[0].repeats = {CreateExpr(10), CreateExpr(8)};
  input_shapes[0].gm_strides = {CreateExpr(8), CreateExpr(1)};
  input_shapes[0].strides = input_shapes[0].gm_strides;
  output_shapes[0].data_type = "float16";
  output_shapes[0].repeats = input_shapes[0].repeats;
  output_shapes[0].gm_strides = input_shapes[0].gm_strides;
  output_shapes[0].strides = output_shapes[0].gm_strides;
  input_shapes[0].data_type_size = 2;
  output_shapes[0].data_type_size = 2;

  PerfOutputInfo perf_res;
  NodeInfo node;
  auto load_v2 = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load_v2, nullptr);
  auto perf = load_v2->GetPerfFunc();
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto result = perf(input_shapes, output_shapes, node, perf_res);
  EXPECT_EQ(result, af::SUCCESS);
  Expr res = perf_res.pipe_res[PipeType::AIV_MTE2];
  auto ternary_ops = perf_res.ternary_ops;
  auto ret = ConcursiveReplaceVars(ternary_ops);
  // 连续: gm_stride=0, 不触发pad, SmallBlk case
  // data_size = 10 * 8 * 2 = 160
  const std::string load_perf = "(160 / (((6.40880012512207 / (block_dim)) + 13.1354999542236)))";
  EXPECT_EQ(Str(res.Replace(ret)), "(" + load_perf + " + 160.0)");
}

// GetDmaParams：dims 长度 <= kMaxDmaLen 时，swap=true 不生效，直接返回全部维度
TEST_F(UTestAscirPerfV2, TestGetDmaParamsSwapDimLessEqual) {
  Expr z0t = CreateExpr("z0t");
  vector<Expr> dims = {z0t, CreateExpr(34)};
  Expr outer_repeat;
  vector<Expr> used_dims;
  const int32_t kMaxDmaLen = 4;

  EXPECT_EQ(af::SUCCESS, att::GetDmaParams(dims, outer_repeat, used_dims, kMaxDmaLen, true));

  ASSERT_EQ(used_dims.size(), 2U);
  EXPECT_EQ(Str(used_dims[0]), "z0t");
  EXPECT_EQ(Str(used_dims[1]), "34");
  EXPECT_EQ(Str(outer_repeat), "1");
}
