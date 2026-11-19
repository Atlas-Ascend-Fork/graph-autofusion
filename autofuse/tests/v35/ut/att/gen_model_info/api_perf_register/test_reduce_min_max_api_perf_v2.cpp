/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/utils/api_perf_utils.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "ascir_ops.h"
#include "parser/specific_params_builder.h"
#include "common/platform_context.h"
#include "codegen_api_param/codegen_api_param.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph_construct_utils.h"
#include "../../../../../ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"
#include "tests/ut/common/ascir_reduce_test_helpers.h"
#include "soc_perf_test_config.h"
#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"

using namespace att;
using namespace af::sym;

namespace {
class UTestReduceMinMaxApiPerfV2 : public ::testing::Test {
 public:
  static ge::RuntimeStubV2Common stub_v_2;

  static void SetUpTestCase() {
    ge::RuntimeStub::Install(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
  }

  static void TearDownTestCase() {
    ge::RuntimeStub::UnInstall(&stub_v_2);
    ge::PlatformContext::GetInstance().Reset();
  }

  void SetUp() override {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
    ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(soc_perf_test::kV35NpuArch, soc_perf_test::kSocEmpty), af::SUCCESS);
  }
};

ge::RuntimeStubV2Common UTestReduceMinMaxApiPerfV2::stub_v_2;

NodeDetail MakeNodeDetail(const std::string &dtype, const std::vector<Expr> &dims) {
  NodeDetail node_detail;
  node_detail.name = "ReduceMinMaxNode";
  node_detail.optype = "Reduce";
  node_detail.input_dtype = {dtype};
  node_detail.output_dtype = {dtype};
  node_detail.input_dims = dims;
  node_detail.output_dims = {dims.front()};
  return node_detail;
}

TensorShapeInfo MakeTensorShape(const std::string &dtype, uint32_t type_size, const std::vector<Expr> &dims,
                                const std::vector<Expr> &strides) {
  TensorShapeInfo shape;
  shape.data_type = dtype;
  shape.data_type_size = type_size;
  shape.dims = dims;
  shape.repeats = dims;
  shape.strides = strides;
  shape.gm_strides = strides;
  return shape;
}

TensorShapeInfo MakeTensorShape(const std::string &dtype, uint32_t type_size, const std::vector<Expr> &dims,
                                const std::vector<Expr> &repeats, const std::vector<Expr> &strides) {
  TensorShapeInfo shape = MakeTensorShape(dtype, type_size, dims, strides);
  shape.repeats = repeats;
  shape.origin_repeats = repeats;
  return shape;
}

std::string PipeString(const PerfOutputInfo &perf, PipeType pipe_type) {
  const auto iter = perf.pipe_res.find(pipe_type);
  if (iter == perf.pipe_res.end()) {
    return "";
  }
  auto pipe_expr = iter->second.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  pipe_expr.Simplify();
  return Str(pipe_expr);
}

std::string TernaryChoiceString(const PerfOutputInfo &perf, bool choice_a) {
  if (perf.ternary_ops.empty()) {
    return "";
  }
  auto if_case = perf.ternary_ops.begin()->second.DeepCopyIfCase();
  auto choice = choice_a ? if_case->GetChoiceA() : if_case->GetChoiceB();
  auto choice_expr = choice->GetExpr().Replace(ConcursiveReplaceVars(perf.ternary_ops));
  choice_expr.Simplify();
  return Str(choice_expr);
}

Expr ResolvedPipeExpr(const PerfOutputInfo &perf, PipeType pipe_type) {
  const auto iter = perf.pipe_res.find(pipe_type);
  if (iter == perf.pipe_res.end()) {
    return CreateExpr(0);
  }
  auto pipe_expr = iter->second.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  pipe_expr.Simplify();
  return pipe_expr;
}

const PerfBreakdownItem *FindBreakdownItem(const PerfOutputInfo &perf, const std::string &name) {
  for (const auto &group : perf.perf_breakdowns) {
    for (const auto &item : group.items) {
      if (item.name == name) {
        return &item;
      }
    }
  }
  return nullptr;
}

ascendcapi_v2::ReduceApiPerfContext MakeRaContext(const std::string &dtype, const std::vector<Expr> &dims,
                                                  bool is_reuse_source = true) {
  ascendcapi_v2::ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(dtype, dims);
  context.pattern = ascendcapi_v2::ReducePattern::kRA;
  context.is_reuse_source = is_reuse_source;
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kNone;
  return context;
}

ascendcapi_v2::ReduceApiPerfContext MakeArContext(const std::string &dtype, const std::vector<Expr> &dims,
                                                  bool is_reuse_source = true) {
  ascendcapi_v2::ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(dtype, dims);
  context.pattern = ascendcapi_v2::ReducePattern::kAR;
  context.is_reuse_source = is_reuse_source;
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kNone;
  return context;
}

void SetReduceSpecificParams(NodeInfo &node, codegen::ReducePattern pattern, codegen::ReduceMergeMode merge_mode,
                             const Expr &merge_size, const Expr &merge_times, bool is_reuse_source) {
  auto &params = node.reduce_specific_params;
  params = ascir_param::ReduceNodeParams{};
  params.canonical_params.valid = true;
  params.canonical_params.pattern = pattern;
  params.canonical_params.merge_mode = merge_mode;
  params.canonical_params.merge_size = merge_size;
  params.canonical_params.merge_times = merge_times;
  params.canonical_params.reuse = {true, is_reuse_source};
  params.exprs.merge_size = {true, {{merge_size, ascir_param::ParamExprRole::kSemantic}}};
  params.exprs.merge_times = {true, {{merge_times, ascir_param::ParamExprRole::kSemantic}}};
}

}  // namespace

// --- Parser integration tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, FillSpecificParamsRegistersSharedReduceNodeParams) {
  using ascir_reduce_test_helpers::BuildReduceNodeInfo;
  using ascir_reduce_test_helpers::ReduceTestEnv;
  ReduceTestEnv env("max");
  env.SetIoAttrs({env.s1, CreateExpr(1)}, {env.s0, CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)});
  auto node_info = BuildReduceNodeInfo(env, "max");
  const auto params = ascir_param::GetAscirNodeParams(env.node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kBuilt);
  const auto *reduce = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.pattern, reduce->canonical_params.pattern);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.merge_mode, reduce->canonical_params.merge_mode);
  EXPECT_EQ(node_info.reduce_specific_params.canonical_params.merge_size, reduce->canonical_params.merge_size);
}

TEST_F(UTestReduceMinMaxApiPerfV2, FillSpecificParamsRegistersSkippedForUnsupportedApi) {
  af::AscGraph graph("skip_param_graph");
  af::ascir_op::Add add("add");
  graph.AddNode(add);
  auto node = graph.FindNode("add");
  ASSERT_NE(node, nullptr);

  NodeInfo node_info;
  node_info.name = "add";
  node_info.node_type = "Add";
  node_info.node_ptr = node;
  EXPECT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  EXPECT_EQ(FillSpecificParams(node, node_info), af::SUCCESS);
  const auto params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kSkipped);
  EXPECT_FALSE(node_info.reduce_specific_params.canonical_params.valid);
}

TEST_F(UTestReduceMinMaxApiPerfV2, FillSpecificParamsKeepsInvalidWhenParamsMissing) {
  using ascir_reduce_test_helpers::ReduceTestEnv;
  ReduceTestEnv env("max");
  NodeInfo node_info;
  node_info.name = "max";
  node_info.node_type = "Max";
  node_info.node_ptr = env.node;

  EXPECT_EQ(FillSpecificParams(env.node, node_info), af::SUCCESS);
  EXPECT_EQ(ascir_param::GetAscirNodeParams(env.node), nullptr);
  EXPECT_FALSE(node_info.reduce_specific_params.canonical_params.valid);
}

// --- RegBase VF lookup tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, RegBaseReduceMinMaxUseVfTable) {
  PerfOutputInfo min_perf;
  PerfOutputInfo max_perf;
  NodeDetail node_detail = MakeNodeDetail(kFloat16, {CreateExpr(128)});

  EXPECT_EQ(ascendcperf_v2::ReduceMinPerf(node_detail, min_perf), af::SUCCESS);
  EXPECT_EQ(ascendcperf_v2::ReduceMaxPerf(node_detail, max_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(max_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RegBaseReduceMinMaxCoverIntegerDtypes) {
  const std::vector<std::string> dtypes = {kInt64, kUInt64, kUInt16, kUInt32};
  for (const auto &dtype : dtypes) {
    PerfOutputInfo min_perf;
    PerfOutputInfo max_perf;
    NodeDetail node_detail = MakeNodeDetail(dtype, {CreateExpr(128)});
    EXPECT_EQ(ascendcperf_v2::ReduceMinPerf(node_detail, min_perf), af::SUCCESS) << dtype;
    EXPECT_EQ(ascendcperf_v2::ReduceMaxPerf(node_detail, max_perf), af::SUCCESS) << dtype;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceMergedShapeUsesZeroStrideWhenLegacyPathHasNoNonZeroStride) {
  const auto plan = codegen::BuildReduceMergedAxisPlan({false, true, false}, {true, false, false});
  EXPECT_TRUE(plan.valid);
  EXPECT_TRUE(plan.use_last_non_zero_stride);
  EXPECT_TRUE(plan.use_zero_stride);

  const auto shape = codegen::BuildReduceMergedShape({CreateExpr(2), CreateExpr(3), CreateExpr(4)},
                                                     {CreateExpr(1), CreateExpr(0), CreateExpr(1)},
                                                     {CreateExpr(0), CreateExpr(1), CreateExpr(1)}, 2U);
  EXPECT_TRUE(shape.valid);
  EXPECT_EQ(Str(shape.first), "2");
  EXPECT_EQ(Str(shape.last), "0");
}

// --- AscendC API layer tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMinMaxArRaBranches) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext ar_context;
  ar_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  ar_context.pattern = ReducePattern::kAR;
  ar_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext ra_context = ar_context;
  ra_context.pattern = ReducePattern::kRA;

  PerfOutputInfo ar_min_perf;
  PerfOutputInfo ar_max_perf;
  PerfOutputInfo ra_min_perf;
  PerfOutputInfo ra_max_perf;

  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ar_context, ar_min_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ar_context, ar_max_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ra_context, ra_min_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ra_context, ra_max_perf), af::SUCCESS);

  EXPECT_FALSE(PipeString(ar_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ar_max_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ra_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ra_max_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(ar_min_perf.ternary_ops.empty());
  EXPECT_FALSE(ar_max_perf.ternary_ops.empty());
  EXPECT_FALSE(ra_min_perf.ternary_ops.empty());
  EXPECT_FALSE(ra_max_perf.ternary_ops.empty());
  EXPECT_NE(PipeString(ar_max_perf, PipeType::AIV_VEC), PipeString(ra_max_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceSumProdMeanUseReduceBranches) {
  auto ar_context = MakeArContext(kFloat32, {CreateExpr(8), CreateExpr(64)}, false);
  auto ra_context = MakeRaContext(kFloat32, {CreateExpr(8), CreateExpr(64)}, true);

  PerfOutputInfo ar_sum_perf;
  PerfOutputInfo ra_sum_perf;
  PerfOutputInfo prod_perf;
  PerfOutputInfo mean_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(ar_context, ar_sum_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(ra_context, ra_sum_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceProdPerf(ra_context, prod_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMeanPerf(ra_context, mean_perf), af::SUCCESS);

  EXPECT_FALSE(PipeString(ar_sum_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(ra_sum_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(prod_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(mean_perf, PipeType::AIV_VEC).empty());
  EXPECT_NE(PipeString(ar_sum_perf, PipeType::AIV_VEC), PipeString(ra_sum_perf, PipeType::AIV_VEC));
  EXPECT_NE(PipeString(ra_sum_perf, PipeType::AIV_VEC), PipeString(prod_perf, PipeType::AIV_VEC));

  Expr mean_delta = ResolvedPipeExpr(mean_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(ra_sum_perf, PipeType::AIV_VEC);
  mean_delta.Simplify();
  PerfOutputInfo muls_perf;
  EXPECT_EQ(ascendcperf_v2::MeanPerf(MakeNodeDetail(kFloat32, {CreateExpr(64)}), muls_perf), af::SUCCESS);
  EXPECT_EQ(Str(mean_delta), Str(ResolvedPipeExpr(muls_perf, PipeType::AIV_VEC)));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceSumMeanProdRejectUnsupportedDtypes) {
  auto float16_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(64)}, true);
  auto int8_context = MakeRaContext(kInt8, {CreateExpr(8), CreateExpr(64)}, true);
  auto bfloat16_context = MakeRaContext(kBfloat16, {CreateExpr(8), CreateExpr(64)}, true);
  auto uint64_context = MakeRaContext(kUInt64, {CreateExpr(8), CreateExpr(64)}, true);

  PerfOutputInfo sum_perf;
  PerfOutputInfo mean_perf;
  PerfOutputInfo prod_perf;
  PerfOutputInfo bfloat16_mean_perf;
  PerfOutputInfo uint64_prod_perf;
  EXPECT_NE(ascendcapi_v2::ReduceSumPerf(float16_context, sum_perf), af::SUCCESS);
  EXPECT_NE(ascendcapi_v2::ReduceMeanPerf(uint64_context, mean_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceProdPerf(int8_context, prod_perf), af::SUCCESS);
  EXPECT_NE(ascendcapi_v2::ReduceMeanPerf(bfloat16_context, bfloat16_mean_perf), af::SUCCESS);
  EXPECT_NE(ascendcapi_v2::ReduceProdPerf(uint64_context, uint64_prod_perf), af::SUCCESS);
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceProdSupportsIntegerDtypes) {
  auto float_context = MakeArContext(kFloat32, {CreateExpr(8), CreateExpr(64)});
  PerfOutputInfo float_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceProdPerf(float_context, float_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(float_perf, PipeType::AIV_VEC).empty());

  const std::vector<std::string> dtypes = {kInt8, kInt16, kInt32, kInt64};

  for (const auto &dtype : dtypes) {
    auto ra_context = MakeRaContext(dtype, {CreateExpr(8), CreateExpr(64)});
    auto ar_context = MakeArContext(dtype, {CreateExpr(8), CreateExpr(64)});
    PerfOutputInfo perf;
    EXPECT_EQ(ascendcapi_v2::ReduceProdPerf(ra_context, perf), af::SUCCESS) << dtype;
    EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty()) << dtype;
    EXPECT_EQ(ascendcapi_v2::ReduceProdPerf(ar_context, perf), af::SUCCESS) << dtype;
    EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty()) << dtype;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceSumSupportsAscendC3510Dtypes) {
  const std::vector<std::string> dtypes = {kInt32, kUInt32, kFloat32, kInt64, kUInt64};

  for (const auto &dtype : dtypes) {
    auto ar_context = MakeArContext(dtype, {CreateExpr(8), CreateExpr(64)}, true);
    auto ra_context = MakeRaContext(dtype, {CreateExpr(8), CreateExpr(64)}, true);
    PerfOutputInfo ar_perf;
    PerfOutputInfo ra_perf;
    EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(ar_context, ar_perf), af::SUCCESS) << dtype;
    EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(ra_context, ra_perf), af::SUCCESS) << dtype;
    EXPECT_FALSE(PipeString(ar_perf, PipeType::AIV_VEC).empty()) << dtype;
    EXPECT_FALSE(PipeString(ra_perf, PipeType::AIV_VEC).empty()) << dtype;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceSumRaUnalignedTailRDoesNotRepeatVfHead) {
  auto context = MakeRaContext(kFloat32, {CreateExpr(1917), CreateExpr(10)}, true);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::ReduceSumPerf(context, perf), af::SUCCESS);

  const std::string vec_perf = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(vec_perf, "TernaryOp(IsEqual(8, 0), 1310, 11892)");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceSumRaAlignedTailRDoesNotRepeatVfHead) {
  auto context = MakeRaContext(kFloat32, {CreateExpr(1917), CreateExpr(80)}, true);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::ReduceSumPerf(context, perf), af::SUCCESS);

  const std::string vec_perf = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(vec_perf, "TernaryOp(IsEqual(0, 0), 13808, 15732)");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceSumRaSymbolicAlignedTailDoesNotRepeatVfHead) {
  const Expr a2t_size = CreateExpr("a2t_size");
  auto context =
      MakeRaContext(kFloat32, {CreateExpr(1917), CreateExpr(8) * af::sym::Ceiling(a2t_size / CreateExpr(8))}, true);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::ReduceSumPerf(context, perf), af::SUCCESS);

  const std::string aligned_formula = TernaryChoiceString(perf, true);
  EXPECT_NE(aligned_formula.find("+ 4) * 893"), std::string::npos) << aligned_formula;
  EXPECT_EQ(aligned_formula.find("+ 64) * 893"), std::string::npos) << aligned_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceRaB64TailRDoesNotRepeatVfHead) {
  auto context = MakeRaContext(kInt64, {CreateExpr(1224), CreateExpr(12)}, true);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::ReduceSumPerf(context, perf), af::SUCCESS);

  const std::string vec_perf = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(vec_perf, "TernaryOp(IsEqual(0, 0), 57326, 57526)");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceArTailRDoesNotRepeatVfHead) {
  auto context = MakeArContext(kFloat32, {CreateExpr(16), CreateExpr(1917)}, true);
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::ReduceSumPerf(context, perf), af::SUCCESS);

  const std::string vec_perf = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(vec_perf, "TernaryOp(IsEqual(38320, 0), 944, 1168)");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceSumB64UsesSpecializedAscendC3510Cost) {
  auto int64_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(64)}, true);
  auto uint64_context = MakeArContext(kUInt64, {CreateExpr(8), CreateExpr(64)}, true);

  PerfOutputInfo int64_perf;
  PerfOutputInfo uint64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(int64_context, int64_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceSumPerf(uint64_context, uint64_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(int64_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(uint64_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReducePerfProvidesReadableBreakdownItems) {
  auto context = MakeRaContext(kInt64, {CreateExpr(16), CreateExpr(4)}, true);
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  context.merge_size = CreateExpr(4);
  context.merge_times = CreateExpr(2);

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);

  const auto *body_item = FindBreakdownItem(perf, "reduce_body_perf");
  const auto *merge_item = FindBreakdownItem(perf, "reduce_merge_perf");
  const auto *total_item = FindBreakdownItem(perf, "reduce_total_perf");
  ASSERT_NE(body_item, nullptr);
  ASSERT_NE(merge_item, nullptr);
  ASSERT_NE(total_item, nullptr);
  EXPECT_NE(body_item->desc.find("single Reduce API body"), std::string::npos);
  EXPECT_NE(merge_item->desc.find("multi-reduce merge"), std::string::npos);
  EXPECT_NE(total_item->desc.find("Reduce API total"), std::string::npos);

  Expr total_expr = total_item->expr.Replace(ConcursiveReplaceVars(perf.ternary_ops));
  total_expr.Simplify();
  EXPECT_EQ(Str(total_expr), PipeString(perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMaxB64AndMergeBranches) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext b64_context;
  b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(65)});
  b64_context.pattern = ReducePattern::kRA;
  b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(b64_context, b64_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(b64_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(b64_perf.ternary_ops.empty());

  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(65)});
  context.pattern = ReducePattern::kRA;
  context.merge_size = CreateExpr(128);

  PerfOutputInfo copy_perf;
  context.merge_mode = ReduceMergeMode::kCopy;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, copy_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(copy_perf, PipeType::AIV_VEC).empty());

  PerfOutputInfo merge_perf;
  context.merge_mode = ReduceMergeMode::kMergeByElementwise;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, merge_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(merge_perf, PipeType::AIV_VEC).empty());
  EXPECT_NE(PipeString(copy_perf, PipeType::AIV_VEC), PipeString(merge_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceReuseSourceChangesFormula) {
  auto reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, true);
  auto non_reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);
  EXPECT_NE(PipeString(reuse_perf, PipeType::AIV_VEC), PipeString(non_reuse_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceArAlignedNonReuseGreaterThanReuse) {
  auto reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, true);
  auto non_reuse_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(256)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);

  Expr reuse_cost = ResolvedPipeExpr(reuse_perf, PipeType::AIV_VEC);
  Expr non_reuse_cost = ResolvedPipeExpr(non_reuse_perf, PipeType::AIV_VEC);
  EXPECT_FALSE(PipeString(non_reuse_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(non_reuse_perf.ternary_ops.empty());
  EXPECT_NE(Str(reuse_cost), Str(non_reuse_cost));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceNonReuseCopyFallsBackForB64Dtype) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext reuse_context;
  reuse_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  reuse_context.pattern = ReducePattern::kRA;
  reuse_context.is_reuse_source = true;
  reuse_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext non_reuse_context = reuse_context;
  non_reuse_context.is_reuse_source = false;

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);
  EXPECT_NE(PipeString(reuse_perf, PipeType::AIV_VEC), PipeString(non_reuse_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscendCApiReduceMaxB64AlignedRaBranch) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64 + last=64: align_key = 64*8 % 32 = 0 → aligned branch, b64=true
  ReduceApiPerfContext aligned_b64_context;
  aligned_b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  aligned_b64_context.pattern = ReducePattern::kRA;
  aligned_b64_context.is_reuse_source = true;
  aligned_b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo aligned_b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_b64_context, aligned_b64_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(aligned_b64_perf, PipeType::AIV_VEC).empty());
  // aligned branch: ternary_ops should NOT contain align case for b64 (last*8 % 32 == 0)
  EXPECT_FALSE(aligned_b64_perf.ternary_ops.empty());

  // Same dims but unaligned (last=65) should produce a different formula
  ReduceApiPerfContext unaligned_b64_context;
  unaligned_b64_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(65)});
  unaligned_b64_context.pattern = ReducePattern::kRA;
  unaligned_b64_context.is_reuse_source = true;
  unaligned_b64_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo unaligned_b64_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_b64_context, unaligned_b64_perf), af::SUCCESS);
  EXPECT_NE(PipeString(aligned_b64_perf, PipeType::AIV_VEC), PipeString(unaligned_b64_perf, PipeType::AIV_VEC));
}

// --- RA Tree Reduction modeling tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedTreeReduction) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64, first=1224, last=9 → large first triggers multi-round tree reduction
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(1224), CreateExpr(9)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());

  // Larger first should produce different (higher) formula than small first
  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(9)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), af::SUCCESS);
  EXPECT_NE(PipeString(perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalAlignedTreeReduction) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=8, last=64 → aligned Concat subpath (dimA <= vlSize / 2), mainR=8, tailR=0
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalUnalignedTreeReduction) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=8, last=65 → unaligned (65*2 % 32 ≠ 0), tree reduction applies
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(65)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalLargeFirstMultiRound) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // float16, first=256, last=128 → mainR=256, tailR=0, folds=8, avgFolds=4, mainTimes=2, foldZero=0
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(256), CreateExpr(128)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());

  // Multi-round should produce more cost than single-round
  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(16), CreateExpr(128)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), af::SUCCESS);
  // Multi-round should produce a different formula than single-round
  EXPECT_NE(PipeString(perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64AlignedTreeReduction) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // int64, first=8, last=64 → aligned (64*8%32==0), mainR=8, tailR=0, folds=3, avgFolds=3, mainTimes=1, foldZero=1
  ReduceApiPerfContext context;
  context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(64)});
  context.pattern = ReducePattern::kRA;
  context.is_reuse_source = true;
  context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(perf.ternary_ops.empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SymbolicDimRIncludesTreeReduceCost) {
  const Expr z0t_size = CreateExpr("z0t_size");
  auto context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, CreateExpr(12)});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);

  const std::string perf_expr = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_NE(perf_expr.find("z0t_size"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalSymbolicDimRIncludesTreeReduceCost) {
  const Expr z0t_size = CreateExpr("z0t_size");
  auto concat_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(64)});
  auto over_vl_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(144)});
  auto unaligned_context = MakeRaContext(kFloat16, {CreateExpr(136) * z0t_size, CreateExpr(65)});

  PerfOutputInfo concat_perf;
  PerfOutputInfo over_vl_perf;
  PerfOutputInfo unaligned_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(concat_context, concat_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(over_vl_context, over_vl_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_context, unaligned_perf), af::SUCCESS);

  EXPECT_NE(TernaryChoiceString(concat_perf, true).find("z0t_size"), std::string::npos);
  EXPECT_NE(TernaryChoiceString(over_vl_perf, true).find("z0t_size"), std::string::npos);
  EXPECT_NE(TernaryChoiceString(unaligned_perf, false).find("z0t_size"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, SymbolicDimRUsesFiniteTreeCases) {
  const Expr z0t_size = CreateExpr("z0t_size");
  auto ra_context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, CreateExpr(12)});
  auto ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size});

  PerfOutputInfo ra_perf;
  PerfOutputInfo ar_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(ra_context, ra_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(ar_context, ar_perf), af::SUCCESS);

  const std::string ra_formula = PipeString(ra_perf, PipeType::AIV_VEC);
  const std::string ar_formula = PipeString(ar_perf, PipeType::AIV_VEC);
  EXPECT_NE(ra_formula.find("< 32768"), std::string::npos) << ra_formula;
  EXPECT_NE(ar_formula.find("z0t_size <= 128"), std::string::npos) << ar_formula;
  EXPECT_NE(ar_formula.find("< 32768"), std::string::npos) << ar_formula;
  EXPECT_EQ(ra_formula.find("Ceiling(((136 * z0t_size) * Rational(1 , 8)))"), std::string::npos) << ra_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, SymbolicDimRNonReuseExactPowerUsesSplitBranch) {
  const Expr z0t_size = CreateExpr("z0t_size");
  auto reuse_ra_context = MakeRaContext(kFloat16, {z0t_size, CreateExpr(144)}, true);
  auto non_reuse_ra_context = MakeRaContext(kFloat16, {z0t_size, CreateExpr(144)}, false);
  auto reuse_ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size}, true);
  auto non_reuse_ar_context = MakeArContext(kFloat16, {CreateExpr(16), z0t_size}, false);

  PerfOutputInfo reuse_ra_perf;
  PerfOutputInfo non_reuse_ra_perf;
  PerfOutputInfo reuse_ar_perf;
  PerfOutputInfo non_reuse_ar_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_ra_context, reuse_ra_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_ra_context, non_reuse_ra_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_ar_context, reuse_ar_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_ar_context, non_reuse_ar_perf), af::SUCCESS);

  const std::string reuse_ra_formula = PipeString(reuse_ra_perf, PipeType::AIV_VEC);
  const std::string non_reuse_ra_formula = PipeString(non_reuse_ra_perf, PipeType::AIV_VEC);
  const std::string reuse_ar_formula = PipeString(reuse_ar_perf, PipeType::AIV_VEC);
  const std::string non_reuse_ar_formula = PipeString(non_reuse_ar_perf, PipeType::AIV_VEC);
  EXPECT_EQ(reuse_ra_formula.find("IsEqual(z0t_size, 16)"), std::string::npos) << reuse_ra_formula;
  EXPECT_NE(non_reuse_ra_formula.find("IsEqual(z0t_size, 16)"), std::string::npos) << non_reuse_ra_formula;
  EXPECT_EQ(reuse_ar_formula.find("IsEqual(z0t_size, 256)"), std::string::npos) << reuse_ar_formula;
  EXPECT_NE(non_reuse_ar_formula.find("IsEqual(z0t_size, 256)"), std::string::npos) << non_reuse_ar_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, Bfloat16NormalReduceUsesHalfVectorRepeat) {
  const Expr z1t_size = CreateExpr("z1t_size");
  auto context = MakeRaContext(kBfloat16, {CreateExpr(136), z1t_size});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);

  const std::string aligned_formula = TernaryChoiceString(perf, true);
  EXPECT_NE(aligned_formula.find("Rational(1 , 128)"), std::string::npos) << aligned_formula;
  EXPECT_EQ(aligned_formula.find("Rational(1 , 64)"), std::string::npos) << aligned_formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SmallLastUsesOneVectorBlock) {
  auto aligned_small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(12)});
  auto aligned_one_block_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(64)});
  auto aligned_two_blocks_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(128)});
  auto unaligned_small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)});
  auto unaligned_one_block_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(63)});
  auto unaligned_two_blocks_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(65)});

  PerfOutputInfo aligned_small_perf;
  PerfOutputInfo aligned_one_block_perf;
  PerfOutputInfo aligned_two_blocks_perf;
  PerfOutputInfo unaligned_small_perf;
  PerfOutputInfo unaligned_one_block_perf;
  PerfOutputInfo unaligned_two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_small_context, aligned_small_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_one_block_context, aligned_one_block_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_two_blocks_context, aligned_two_blocks_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_small_context, unaligned_small_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_one_block_context, unaligned_one_block_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(unaligned_two_blocks_context, unaligned_two_blocks_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(aligned_small_perf, true), TernaryChoiceString(aligned_one_block_perf, true));
  EXPECT_NE(TernaryChoiceString(aligned_small_perf, true), TernaryChoiceString(aligned_two_blocks_perf, true));
  EXPECT_EQ(TernaryChoiceString(unaligned_small_perf, false), TernaryChoiceString(unaligned_one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(unaligned_small_perf, false), TernaryChoiceString(unaligned_two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64ExpandsInt64BinaryFuncCost) {
  auto aligned_context = MakeRaContext(kInt64, {CreateExpr(2176), CreateExpr(64)});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(aligned_context, perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(perf, true), "92295");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArB64UnalignedSmallLastUsesOneVectorBlock) {
  auto small_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(9)});
  auto one_block_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(63)});
  auto two_blocks_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)});

  PerfOutputInfo small_perf;
  PerfOutputInfo one_block_perf;
  PerfOutputInfo two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(one_block_context, one_block_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(two_blocks_context, two_blocks_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(small_perf, false), TernaryChoiceString(one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(small_perf, false), TernaryChoiceString(two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArNormalUnalignedSmallLastUsesOneVectorBlock) {
  auto small_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(65)});
  auto one_block_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(127)});
  auto two_blocks_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(129)});

  PerfOutputInfo small_perf;
  PerfOutputInfo one_block_perf;
  PerfOutputInfo two_blocks_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(small_context, small_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(one_block_context, one_block_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(two_blocks_context, two_blocks_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(small_perf, false), TernaryChoiceString(one_block_perf, false));
  EXPECT_NE(TernaryChoiceString(small_perf, false), TernaryChoiceString(two_blocks_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArUnalignedNonReuseDoesNotAddCopyCost) {
  auto reuse_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)}, true);
  auto non_reuse_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArUnalignedDoesNotUseGlobalPenalty) {
  auto b64_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(65)});
  auto normal_context = MakeArContext(kFloat16, {CreateExpr(8), CreateExpr(129)});

  PerfOutputInfo b64_perf;
  PerfOutputInfo normal_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(b64_context, b64_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(normal_context, normal_perf), af::SUCCESS);

  const std::string b64_unaligned = TernaryChoiceString(b64_perf, false);
  const std::string normal_unaligned = TernaryChoiceString(normal_perf, false);
  EXPECT_EQ(b64_unaligned.find("2.5"), std::string::npos) << b64_unaligned;
  EXPECT_EQ(normal_unaligned.find("2.5"), std::string::npos) << normal_unaligned;
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64SymbolicAlignedLastUsesVectorBlockCeil) {
  const Expr z0t_size = CreateExpr("z0t_size");
  const Expr z5t_size = CreateExpr("z5t_size");
  auto aligned_last = CreateExpr(4) * af::sym::Ceiling(z5t_size / CreateExpr(4));
  auto context = MakeRaContext(kInt64, {CreateExpr(136) * z0t_size, aligned_last});

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(context, perf), af::SUCCESS);
  const std::string aligned_formula = TernaryChoiceString(perf, true);
  EXPECT_NE(aligned_formula.find("z5t_size"), std::string::npos);
  EXPECT_NE(aligned_formula.find("Rational(1 , 16)"), std::string::npos);
  EXPECT_EQ(aligned_formula.find("Rational(1 , 8)"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedNonReuseMatchesReuseSource) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  ReduceApiPerfContext reuse_context;
  reuse_context.node_detail = MakeNodeDetail(kInt64, {CreateExpr(8), CreateExpr(9)});
  reuse_context.pattern = ReducePattern::kRA;
  reuse_context.is_reuse_source = true;
  reuse_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext non_reuse_context = reuse_context;
  non_reuse_context.is_reuse_source = false;

  PerfOutputInfo reuse_perf, non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(reuse_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(non_reuse_perf, PipeType::AIV_VEC).empty());

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaTreeReductionCostIncreasesWithFirst) {
  using ascendcapi_v2::ReduceApiPerfContext;
  using ascendcapi_v2::ReduceMergeMode;
  using ascendcapi_v2::ReducePattern;

  // first=16 > first=8, same last=64, should produce higher cost
  ReduceApiPerfContext big_context;
  big_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(16), CreateExpr(64)});
  big_context.pattern = ReducePattern::kRA;
  big_context.is_reuse_source = true;
  big_context.merge_mode = ReduceMergeMode::kNone;

  ReduceApiPerfContext small_context;
  small_context.node_detail = MakeNodeDetail(kFloat16, {CreateExpr(8), CreateExpr(64)});
  small_context.pattern = ReducePattern::kRA;
  small_context.is_reuse_source = true;
  small_context.merge_mode = ReduceMergeMode::kNone;

  PerfOutputInfo big_perf, small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(big_context, big_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), af::SUCCESS);

  // Different first dimensions should produce different formulas
  EXPECT_NE(PipeString(big_perf, PipeType::AIV_VEC), PipeString(small_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaPipeBarrierVPerfIsZeroInFirstPhase) {
  auto base_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(64)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kMergeByElementwise;
  merge_context.merge_size = CreateExpr(128);

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  PerfOutputInfo elementwise_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(base_context, base_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(merge_context, merge_perf), af::SUCCESS);
  EXPECT_EQ(ascendcperf_v2::MaxPerf(MakeNodeDetail(kFloat16, {CreateExpr(128)}), elementwise_perf), af::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  EXPECT_EQ(Str(merge_delta), Str(ResolvedPipeExpr(elementwise_perf, PipeType::AIV_VEC)));
}

TEST_F(UTestReduceMinMaxApiPerfV2, MultiReduceCopyMergeUsesAlignedTempSizeAndElementwiseTail) {
  auto base_context = MakeRaContext(kInt64, {CreateExpr(1224), CreateExpr(12)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  merge_context.merge_size = CreateExpr(9);
  merge_context.merge_times = CreateExpr(9);

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(base_context, base_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(merge_context, merge_perf), af::SUCCESS);

  Expr ub_copy_res = CreateExpr(0);
  EXPECT_EQ(GetPerf({kUb2ub, kInt64, kInt64, {CreateExpr(32)}, CreateExpr(0)}, ub_copy_res), af::SUCCESS);

  PerfOutputInfo elementwise_perf;
  EXPECT_EQ(ascendcperf_v2::MaxPerf(MakeNodeDetail(kInt64, {CreateExpr(32)}), elementwise_perf), af::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  Expr expected = (ub_copy_res + CreateExpr(8) * ResolvedPipeExpr(elementwise_perf, PipeType::AIV_VEC)) / CreateExpr(9);
  expected.Simplify();
  EXPECT_EQ(Str(merge_delta), Str(expected));
}

TEST_F(UTestReduceMinMaxApiPerfV2, MultiReduceCopyMergeSymbolicTailUsesVectorBlock) {
  const Expr z5t_size = CreateExpr("z5t_size");
  auto base_context = MakeRaContext(kInt64, {CreateExpr(2176), CreateExpr(4) * af::sym::Ceiling(z5t_size / kSymFour)});
  auto merge_context = base_context;
  merge_context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  merge_context.merge_size = z5t_size;
  merge_context.merge_times = CreateExpr("z0z1z5Tb_size");

  PerfOutputInfo base_perf;
  PerfOutputInfo merge_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(base_context, base_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(merge_context, merge_perf), af::SUCCESS);

  Expr merge_delta = ResolvedPipeExpr(merge_perf, PipeType::AIV_VEC) - ResolvedPipeExpr(base_perf, PipeType::AIV_VEC);
  merge_delta.Simplify();
  const std::string merge_formula = Str(merge_delta);
  EXPECT_NE(merge_formula.find("Rational(1 , 32)"), std::string::npos);
  EXPECT_EQ(merge_formula.find("Rational(1 , 4)"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64UnalignedNonReuseDoesNotAddCopyCost) {
  auto reuse_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)}, true);
  auto non_reuse_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalUnalignedNonReuseDoesNotAddCopyCost) {
  auto reuse_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(65)}, true);
  auto non_reuse_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(65)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMinPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);

  EXPECT_EQ(TernaryChoiceString(reuse_perf, false), TernaryChoiceString(non_reuse_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaNormalAlignedSubpathsProduceDistinctFormulas) {
  auto concat_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(32)});
  auto less_than_vl_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(96)});
  auto over_vl_context = MakeRaContext(kFloat16, {CreateExpr(8), CreateExpr(144)});

  PerfOutputInfo concat_perf;
  PerfOutputInfo less_than_vl_perf;
  PerfOutputInfo over_vl_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(concat_context, concat_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(less_than_vl_context, less_than_vl_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(over_vl_context, over_vl_perf), af::SUCCESS);

  EXPECT_NE(TernaryChoiceString(concat_perf, true), TernaryChoiceString(less_than_vl_perf, true));
  EXPECT_NE(TernaryChoiceString(less_than_vl_perf, true), TernaryChoiceString(over_vl_perf, true));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaB64LargeUnalignedDoesNotCollapseToSmallFirstFormula) {
  auto large_context = MakeRaContext(kInt64, {CreateExpr(1224), CreateExpr(9)});
  auto small_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(9)});

  PerfOutputInfo large_perf;
  PerfOutputInfo small_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(large_context, large_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(small_context, small_perf), af::SUCCESS);

  EXPECT_NE(TernaryChoiceString(large_perf, false), TernaryChoiceString(small_perf, false));
}

TEST_F(UTestReduceMinMaxApiPerfV2, RaAlignedNonReuseChangesTreeStructure) {
  auto reuse_context = MakeRaContext(kFloat16, {CreateExpr(16), CreateExpr(96)}, true);
  auto non_reuse_context = MakeRaContext(kFloat16, {CreateExpr(16), CreateExpr(96)}, false);

  PerfOutputInfo reuse_perf;
  PerfOutputInfo non_reuse_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(reuse_context, reuse_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(non_reuse_context, non_reuse_perf), af::SUCCESS);

  EXPECT_NE(TernaryChoiceString(reuse_perf, true), TernaryChoiceString(non_reuse_perf, true));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRaReduceUsesCodegenMergedShapeForMultiAAxes) {
  const Expr z1t_size = CreateExpr("z1t_size");
  const Expr z2t_size = CreateExpr("z2t_size");
  const std::vector<Expr> repeats = {z2t_size, CreateExpr(17), z1t_size, CreateExpr(16), CreateExpr(9)};
  const std::vector<Expr> input_strides = {CreateExpr(3264) * z1t_size, CreateExpr(192) * z1t_size, CreateExpr(192),
                                           CreateExpr(12), CreateExpr(1)};
  const std::vector<Expr> output_strides = {CreateExpr(0), CreateExpr(0), CreateExpr(192), CreateExpr(12),
                                            CreateExpr(1)};
  const std::vector<TensorShapeInfo> inputs = {
      MakeTensorShape(kInt64, 8U, {CreateExpr(272) * z1t_size * z2t_size, CreateExpr(12)}, repeats, input_strides)};
  const std::vector<TensorShapeInfo> outputs = {
      MakeTensorShape(kInt64, 8U, {CreateExpr(192) * z1t_size}, repeats, output_strides)};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kRA, codegen::ReduceMergeMode::kCopy,
                          CreateExpr(192) * z1t_size, CreateExpr("z0z2Tt_size"), true);
  // 该用例绕过 parser，直接注入 codegen 侧合并结果，覆盖 ATT 读取 merged_dims 的路径。
  reduce_node.reduce_specific_params.canonical_params.merged_dims = {true, CreateExpr(17) * z2t_size,
                                                                     CreateExpr(192) * z1t_size};
  PerfOutputInfo perf;
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(max_v2->GetPerfFunc()(inputs, outputs, reduce_node, perf), af::SUCCESS);

  const std::string formula = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_EQ(formula.find("272 * z1t_size * z2t_size"), std::string::npos) << formula;
  EXPECT_NE(formula.find("17 * z2t_size"), std::string::npos) << formula;
  EXPECT_EQ(formula.find("Rational(17 , 8) * z2t_size"), std::string::npos) << formula;
  EXPECT_NE(formula.find("Ceiling((3 * z1t_size))"), std::string::npos) << formula;
  EXPECT_EQ(formula.find("Ceiling((48 * z1t_size))"), std::string::npos) << formula;
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceRebuildsMergedDimsFromCurrentShape) {
  const std::vector<TensorShapeInfo> tail_inputs = {MakeTensorShape(
      kFloat16, 2U, {CreateExpr(4), CreateExpr(16)}, {CreateExpr(4), CreateExpr(16)}, {CreateExpr(16), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> tail_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(16)}, {CreateExpr(4), CreateExpr(16)}, {CreateExpr(0), CreateExpr(1)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kRA, codegen::ReduceMergeMode::kNone, CreateExpr(16),
                          CreateExpr(1), true);
  reduce_node.reduce_specific_params.canonical_params.merged_dims = {true, CreateExpr(64), CreateExpr(16)};

  PerfOutputInfo perf;
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(max_v2->GetPerfFunc()(tail_inputs, tail_outputs, reduce_node, perf), af::SUCCESS);

  auto expected_context = MakeRaContext(kFloat16, {CreateExpr(4), CreateExpr(16)}, true);
  PerfOutputInfo expected_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceMaxPerf(expected_context, expected_perf), af::SUCCESS);
  EXPECT_EQ(PipeString(perf, PipeType::AIV_VEC), PipeString(expected_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceUsesSemanticMergeParamsForPerf) {
  const Expr codegen_merge_times = CreateExpr("z0z1Tt_size");
  const Expr semantic_times_expr = CreateExpr("s0") * CreateExpr("s1");
  const std::vector<TensorShapeInfo> inputs = {MakeTensorShape(
      kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(8), CreateExpr(16)}, {CreateExpr(16), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr("out_keep")}, {CreateExpr(16), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kCopy, CreateExpr(128),
                          codegen_merge_times, true);
  reduce_node.reduce_specific_params.exprs.merge_size = {
      true, {{CreateExpr("out_keep"), ascir_param::ParamExprRole::kSemantic}}};
  reduce_node.reduce_specific_params.exprs.merge_times = {
      true, {{semantic_times_expr, ascir_param::ParamExprRole::kSemantic}}};

  PerfOutputInfo perf;
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(max_v2->GetPerfFunc()(inputs, outputs, reduce_node, perf), af::SUCCESS);

  const std::string formula = PipeString(perf, PipeType::AIV_VEC);
  EXPECT_NE(formula.find("out_keep"), std::string::npos) << formula;
  EXPECT_NE(formula.find("s0"), std::string::npos) << formula;
  EXPECT_NE(formula.find("s1"), std::string::npos) << formula;
  EXPECT_EQ(formula.find("z0z1Tt_size"), std::string::npos) << formula;
}

// --- ASCIR registration tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersReduceMinMaxOpsAndAliases) {
  const std::vector<TensorShapeInfo> reduce_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> reduce_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);
  PerfOutputInfo reduce_min_perf;
  PerfOutputInfo reduce_max_perf;
  auto reduce_min_v2 = ApiPerfFactory::Instance().Create(kReduceMin + "V2");
  auto reduce_max_v2 = ApiPerfFactory::Instance().Create(kReduceMax + "V2");
  ASSERT_NE(reduce_min_v2, nullptr);
  ASSERT_NE(reduce_max_v2, nullptr);
  EXPECT_EQ(reduce_min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_min_perf), af::SUCCESS);
  EXPECT_EQ(reduce_max_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_max_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(reduce_min_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_max_perf, PipeType::AIV_VEC).empty());

  PerfOutputInfo min_reduce_alias_perf;
  PerfOutputInfo max_reduce_alias_perf;
  auto min_v2 = ApiPerfFactory::Instance().Create(kMin + "V2");
  auto max_v2 = ApiPerfFactory::Instance().Create(kMax + "V2");
  ASSERT_NE(min_v2, nullptr);
  ASSERT_NE(max_v2, nullptr);
  EXPECT_EQ(min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, min_reduce_alias_perf), af::SUCCESS);
  EXPECT_EQ(max_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, max_reduce_alias_perf), af::SUCCESS);
  EXPECT_EQ(PipeString(reduce_min_perf, PipeType::AIV_VEC), PipeString(min_reduce_alias_perf, PipeType::AIV_VEC));
  EXPECT_EQ(PipeString(reduce_max_perf, PipeType::AIV_VEC), PipeString(max_reduce_alias_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersReduceAnyAllOps) {
  const std::vector<TensorShapeInfo> logical_reduce_inputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> logical_reduce_outputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);
  PerfOutputInfo reduce_all_perf;
  PerfOutputInfo reduce_any_perf;
  auto all_v2 = ApiPerfFactory::Instance().Create(kReduceAll + "V2");
  auto any_v2 = ApiPerfFactory::Instance().Create(kReduceAny + "V2");
  ASSERT_NE(all_v2, nullptr);
  ASSERT_NE(any_v2, nullptr);
  EXPECT_EQ(all_v2->GetPerfFunc()(logical_reduce_inputs, logical_reduce_outputs, reduce_node, reduce_all_perf),
            af::SUCCESS);
  EXPECT_EQ(any_v2->GetPerfFunc()(logical_reduce_inputs, logical_reduce_outputs, reduce_node, reduce_any_perf),
            af::SUCCESS);
  EXPECT_FALSE(PipeString(reduce_all_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_any_perf, PipeType::AIV_VEC).empty());
  EXPECT_EQ(PipeString(reduce_all_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);
  EXPECT_EQ(PipeString(reduce_any_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersReduceSumMeanProdOpsAndAliases) {
  const std::vector<TensorShapeInfo> reduce_inputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> reduce_outputs = {
      MakeTensorShape(kFloat32, 4U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);

  auto reduce_sum_v2 = ApiPerfFactory::Instance().Create(kReduceSum + "V2");
  auto reduce_mean_v2 = ApiPerfFactory::Instance().Create(kReduceMean + "V2");
  auto reduce_prod_v2 = ApiPerfFactory::Instance().Create(kReduceProd + "V2");
  auto sum_v2 = ApiPerfFactory::Instance().Create(kSum + "V2");
  auto mean_v2 = ApiPerfFactory::Instance().Create(kMean + "V2");
  auto prod_v2 = ApiPerfFactory::Instance().Create(kProd + "V2");
  ASSERT_NE(reduce_sum_v2, nullptr);
  ASSERT_NE(reduce_mean_v2, nullptr);
  ASSERT_NE(reduce_prod_v2, nullptr);
  ASSERT_NE(sum_v2, nullptr);
  ASSERT_NE(mean_v2, nullptr);
  ASSERT_NE(prod_v2, nullptr);

  PerfOutputInfo reduce_sum_perf;
  PerfOutputInfo reduce_mean_perf;
  PerfOutputInfo reduce_prod_perf;
  PerfOutputInfo sum_alias_perf;
  PerfOutputInfo mean_alias_perf;
  PerfOutputInfo prod_alias_perf;
  EXPECT_EQ(reduce_sum_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_sum_perf), af::SUCCESS);
  EXPECT_EQ(reduce_mean_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_mean_perf), af::SUCCESS);
  EXPECT_EQ(reduce_prod_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, reduce_prod_perf), af::SUCCESS);
  EXPECT_EQ(sum_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, sum_alias_perf), af::SUCCESS);
  EXPECT_EQ(mean_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, mean_alias_perf), af::SUCCESS);
  EXPECT_EQ(prod_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, reduce_node, prod_alias_perf), af::SUCCESS);

  EXPECT_FALSE(PipeString(reduce_sum_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_mean_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(reduce_prod_perf, PipeType::AIV_VEC).empty());
  EXPECT_EQ(PipeString(reduce_sum_perf, PipeType::AIV_VEC), PipeString(sum_alias_perf, PipeType::AIV_VEC));
  EXPECT_EQ(PipeString(reduce_mean_perf, PipeType::AIV_VEC), PipeString(mean_alias_perf, PipeType::AIV_VEC));
  EXPECT_EQ(PipeString(reduce_prod_perf, PipeType::AIV_VEC), PipeString(prod_alias_perf, PipeType::AIV_VEC));
  EXPECT_NE(PipeString(reduce_sum_perf, PipeType::AIV_VEC), PipeString(reduce_prod_perf, PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceSumMeanProdSkipUnsupportedDtypes) {
  const std::vector<TensorShapeInfo> int64_inputs = {
      MakeTensorShape(kInt64, 8U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> uint64_mean_inputs = {
      MakeTensorShape(kUInt64, 8U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> uint64_inputs = {
      MakeTensorShape(kUInt64, 8U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> int64_outputs = {
      MakeTensorShape(kInt64, 8U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};
  const std::vector<TensorShapeInfo> uint64_mean_outputs = {
      MakeTensorShape(kUInt64, 8U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};
  const std::vector<TensorShapeInfo> uint64_outputs = {
      MakeTensorShape(kUInt64, 8U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo reduce_node;
  SetReduceSpecificParams(reduce_node, codegen::ReducePattern::kAR, codegen::ReduceMergeMode::kNone, CreateExpr(8),
                          CreateExpr(1), false);

  auto reduce_sum_v2 = ApiPerfFactory::Instance().Create(kReduceSum + "V2");
  auto reduce_mean_v2 = ApiPerfFactory::Instance().Create(kReduceMean + "V2");
  auto reduce_prod_v2 = ApiPerfFactory::Instance().Create(kReduceProd + "V2");
  ASSERT_NE(reduce_sum_v2, nullptr);
  ASSERT_NE(reduce_mean_v2, nullptr);
  ASSERT_NE(reduce_prod_v2, nullptr);

  PerfOutputInfo sum_perf;
  PerfOutputInfo mean_perf;
  PerfOutputInfo prod_perf;
  EXPECT_EQ(reduce_sum_v2->GetPerfFunc()(int64_inputs, int64_outputs, reduce_node, sum_perf), af::SUCCESS);
  EXPECT_EQ(reduce_mean_v2->GetPerfFunc()(uint64_mean_inputs, uint64_mean_outputs, reduce_node, mean_perf),
            af::SUCCESS);
  EXPECT_EQ(reduce_prod_v2->GetPerfFunc()(uint64_inputs, uint64_outputs, reduce_node, prod_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(sum_perf, PipeType::AIV_VEC).empty());
  EXPECT_TRUE(PipeString(mean_perf, PipeType::AIV_VEC).empty());
  EXPECT_TRUE(PipeString(prod_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirRegistersElementwiseMinMaxOps) {
  const std::vector<TensorShapeInfo> elementwise_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)}),
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)})};
  const std::vector<TensorShapeInfo> elementwise_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(128)}, {CreateExpr(1)})};

  NodeInfo node;
  PerfOutputInfo minimum_perf;
  PerfOutputInfo maximum_perf;
  auto minimum_v2 = ApiPerfFactory::Instance().Create(kMinimum + "V2");
  auto maximum_v2 = ApiPerfFactory::Instance().Create(kMaximum + "V2");
  ASSERT_NE(minimum_v2, nullptr);
  ASSERT_NE(maximum_v2, nullptr);
  EXPECT_EQ(minimum_v2->GetPerfFunc()(elementwise_inputs, elementwise_outputs, node, minimum_perf), af::SUCCESS);
  EXPECT_EQ(maximum_v2->GetPerfFunc()(elementwise_inputs, elementwise_outputs, node, maximum_perf), af::SUCCESS);
  EXPECT_FALSE(PipeString(minimum_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(maximum_perf, PipeType::AIV_VEC).empty());
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceAnyAllUseLogicalReduceEntrypoints) {
  auto uint8_context = MakeRaContext(kUInt8, {CreateExpr(8), CreateExpr(64)});
  auto bool_context = MakeRaContext(kBool, {CreateExpr(8), CreateExpr(64)});
  const std::vector<std::string> direct_integer_dtypes = {kInt8, kInt16, kInt32, kInt64};
  auto all_context = MakeArContext(kFloat32, {CreateExpr(8), CreateExpr(64)});

  PerfOutputInfo uint8_perf;
  PerfOutputInfo all_perf;
  PerfOutputInfo bool_perf;
  EXPECT_EQ(ascendcapi_v2::ReduceAnyPerf(uint8_context, uint8_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceAnyPerf(bool_context, bool_perf), af::SUCCESS);
  EXPECT_EQ(ascendcapi_v2::ReduceAllPerf(all_context, all_perf), af::SUCCESS);

  EXPECT_FALSE(PipeString(uint8_perf, PipeType::AIV_VEC).empty());
  EXPECT_FALSE(PipeString(all_perf, PipeType::AIV_VEC).empty());
  EXPECT_EQ(PipeString(bool_perf, PipeType::AIV_VEC), PipeString(uint8_perf, PipeType::AIV_VEC));
  for (const auto &dtype : direct_integer_dtypes) {
    auto context = MakeRaContext(dtype, {CreateExpr(8), CreateExpr(64)});
    PerfOutputInfo perf;
    EXPECT_EQ(ascendcapi_v2::ReduceAnyPerf(context, perf), af::SUCCESS);
    EXPECT_EQ(ascendcapi_v2::ReduceAllPerf(context, perf), af::SUCCESS);
  }

  EXPECT_EQ(PipeString(uint8_perf, PipeType::AIV_VEC).find("reduce_ra_normal_align_case"), std::string::npos);
  EXPECT_EQ(PipeString(all_perf, PipeType::AIV_VEC).find("reduce_ar_normal_align_case"), std::string::npos);
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceAnyAllInt64UseB64Cost) {
  auto ar_context = MakeArContext(kInt64, {CreateExpr(8), CreateExpr(64)});
  auto ra_context = MakeRaContext(kInt64, {CreateExpr(8), CreateExpr(64)});
  const std::vector<std::pair<ascendcapi_v2::ReduceApiPerfContext, std::string>> cases = {
      {ar_context, "reduce_ar_b64_align_perf"}, {ra_context, "reduce_ra_b64_align_perf"}};
  for (const auto &dtype_case : cases) {
    const auto &context = dtype_case.first;
    PerfOutputInfo any_perf;
    PerfOutputInfo all_perf;
    ASSERT_EQ(ascendcapi_v2::ReduceAnyPerf(context, any_perf), af::SUCCESS);
    ASSERT_EQ(ascendcapi_v2::ReduceAllPerf(context, all_perf), af::SUCCESS);
    ASSERT_NE(FindBreakdownItem(any_perf, "reduce_body_perf"), nullptr);
    ASSERT_NE(FindBreakdownItem(all_perf, "reduce_body_perf"), nullptr);
    ASSERT_NE(FindBreakdownItem(any_perf, dtype_case.second), nullptr);
    ASSERT_NE(FindBreakdownItem(all_perf, dtype_case.second), nullptr);
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceAnyAllLogicalMergeCostIsIncluded) {
  auto context = MakeRaContext(kBool, {CreateExpr(16), CreateExpr(4)});
  context.merge_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
  context.merge_size = CreateExpr(4);
  context.merge_times = CreateExpr(2);

  PerfOutputInfo perf;
  EXPECT_EQ(ascendcapi_v2::ReduceAnyPerf(context, perf), af::SUCCESS);
  ASSERT_NE(FindBreakdownItem(perf, "reduce_body_perf"), nullptr);
  ASSERT_NE(FindBreakdownItem(perf, "reduce_merge_perf"), nullptr);
  ASSERT_NE(FindBreakdownItem(perf, "reduce_total_perf"), nullptr);
  EXPECT_NE(PipeString(perf, PipeType::AIV_VEC), "0");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ReduceAnyAllRejectEmptyDtypeBeforeLogicalNormalize) {
  auto empty_input_context = MakeRaContext(kBool, {CreateExpr(8), CreateExpr(64)});
  empty_input_context.node_detail.input_dtype.clear();
  auto empty_output_context = MakeRaContext(kBool, {CreateExpr(8), CreateExpr(64)});
  empty_output_context.node_detail.output_dtype.clear();

  PerfOutputInfo any_perf;
  PerfOutputInfo all_perf;
  EXPECT_NE(ascendcapi_v2::ReduceAnyPerf(empty_input_context, any_perf), af::SUCCESS);
  EXPECT_NE(ascendcapi_v2::ReduceAllPerf(empty_output_context, all_perf), af::SUCCESS);
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirMaxMinAnyAllFallbackToElementwiseWhenReduceInfoUnknown) {
  const std::vector<TensorShapeInfo> inputs = {MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)}),
                                               MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)})};
  const std::vector<TensorShapeInfo> outputs = {MakeTensorShape(kFloat32, 2U, {CreateExpr(129)}, {CreateExpr(1)})};
  const std::vector<std::string> tags = {kMax, kMin, kAny, kAll};

  NodeInfo node;
  for (const auto &tag : tags) {
    PerfOutputInfo perf;
    auto api_perf = ApiPerfFactory::Instance().Create(tag + "V2");
    ASSERT_NE(api_perf, nullptr) << tag;
    EXPECT_EQ(api_perf->GetPerfFunc()(inputs, outputs, node, perf), af::SUCCESS) << tag;
    EXPECT_EQ(PipeString(perf, PipeType::AIV_VEC), "26") << tag;
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, AscirReduceFailsWhenCodegenModeUnknown) {
  const std::vector<TensorShapeInfo> reduce_inputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(64)}, {CreateExpr(64), CreateExpr(1)})};
  const std::vector<TensorShapeInfo> reduce_outputs = {
      MakeTensorShape(kFloat16, 2U, {CreateExpr(8), CreateExpr(1)}, {CreateExpr(1), CreateExpr(0)})};

  NodeInfo node;
  PerfOutputInfo reduce_min_perf;
  auto min_v2 = ApiPerfFactory::Instance().Create(kReduceMin + "V2");
  ASSERT_NE(min_v2, nullptr);
  EXPECT_NE(min_v2->GetPerfFunc()(reduce_inputs, reduce_outputs, node, reduce_min_perf), af::SUCCESS);
}

// --- dtype registration tests ---

TEST_F(UTestReduceMinMaxApiPerfV2, MinAscirUsesMinAttImpl) {
  auto att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl("3510", "Min");
  ASSERT_NE(att_impl, nullptr);
  EXPECT_STREQ(reinterpret_cast<const char *>(att_impl->GetApiPerf()), "MinV2");
}

namespace {
PerfOutputInfo ArgReduceTestCost(bool ar, int64_t a, int64_t r, bool phase2 = false, bool with_value = false,
                                 const std::string &dtype = kFloat32, int64_t valid = 0, bool accumulate = false,
                                 int64_t valid_a = 0, bool is_min = false) {
  NodeInfo node;
  auto &params = node.argmax_node_params;
  params.valid = true;
  params.is_ar = ar;
  params.is_phase2 = phase2;
  params.with_value = with_value;
  params.has_accumulate = accumulate;
  params.src_shape =
      ar ? std::vector<Expr>{CreateExpr(a), CreateExpr(r)} : std::vector<Expr>{CreateExpr(r), CreateExpr(a)};
  if (phase2) {
    params.src_shape.emplace_back(CreateExpr(a));
  }
  params.valid_extent = CreateExpr(valid == 0 ? (phase2 ? a : r) : valid);
  params.valid_a = CreateExpr(valid_a);
  const auto input = MakeTensorShape(dtype, dtype == kInt64 ? 8U : 4U, params.src_shape, {});
  PerfOutputInfo perf;
  const auto evaluate = is_min ? ascir_reduce_v2::ArgminApi : ascir_reduce_v2::ArgmaxApi;
  EXPECT_EQ(evaluate({input}, {}, node, perf), af::SUCCESS);
  return perf;
}

void ExpectArgReduceCycles(const PerfOutputInfo &perf, int64_t instruction_cycles, int64_t max_latency = 8) {
  // Assert independently counted instructions under the CURRENT table, not a measured hardware latency.
  auto expected = VfPerfUtils::GetVFHeadCost() + CreateExpr(max_latency + instruction_cycles);
  expected.Simplify();
  EXPECT_EQ(PipeString(perf, PipeType::AIV_VEC), Str(expected));
}
}  // namespace

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxArSinglePassAllDtypesAndAlignment) {
  // f32 VF setup: five Duplicate + Arange (2 cycles) + UpdateMask = 8.
  // Full row: 2 Duplicate + 5 Select + 3 reductions + 5 comparisons + 2 And (4) + Or (2)
  // + Add (2) + index Store = 24. Each later block: GE/NE/Or/Duplicate/two Select = 7.
  ExpectArgReduceCycles(ArgReduceTestCost(true, 256, 2048), 8 + 256 * (24 + 31 * 7));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 32), 8 + 2 * (24 + 2));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 64), 8 + 2 * 24);
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 32, false, true), 8 + 2 * (24 + 2 + 4));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 64, false, true), 8 + 2 * (24 + 4));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 96), 8 + 2 * (24 + 7 + 4));
  // i32: setup 2 Duplicate + Arange + UpdateMask = 5; row 12, later block 4.
  // The final candidate mask And remains necessary for integer input.
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 32, false, false, kInt32), 5 + 2 * (12 + 2));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 64, false, false, kInt32), 5 + 2 * 12);
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, kInt32), 5 + 2 * (12 + 31 * 4));
  // Unaligned AR still scans once. LoadUnAlignPre is a zero-cost placeholder under the current table.
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 65), 8 + 2 * (24 + 7 + 4));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 65, false, false, kInt32), 5 + 2 * (12 + 4 + 4));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 65, false, false, kFloat32, 32), 8 + 2 * (24 + 2));
  // b64 VL=32: only ReduceMax/EQ/ReduceMin/DeInterleave/Store contribute per row (5), GE per update (1).
  // Other b64 table entries remain missing; do not substitute b32 instruction costs.
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 64, false, false, kInt64), 2 * (5 + 1));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 64, false, true, kInt64), 2 * (5 + 1 + 1));
  ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 65, false, true, kInt64), 2 * (5 + 2 + 1));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxArSinglePassValidRAndWithValue) {
  // Physical R=2048 stays aligned; validR determines first-partial, full updates and final partial load.
  for (const auto &dtype : {kFloat32, kInt32}) {
    const int64_t init = dtype == kFloat32 ? 8 : 5;
    const int64_t row = dtype == kFloat32 ? 24 : 12;
    const int64_t update = dtype == kFloat32 ? 7 : 4;
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, dtype, 1), init + 2 * (row + 2));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, dtype, 32), init + 2 * (row + 2));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, dtype, 64), init + 2 * row);
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, dtype, 65), init + 2 * (row + update + 4));
    // WithValue: Adds (2), UpdateMask (1), one value Store (1); source Gather/barrier cost zero.
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, true, dtype, 32), init + 2 * (row + 2 + 4));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, true, dtype, 65), init + 2 * (row + update + 4 + 4));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, true, dtype), init + 2 * (row + 31 * update + 4));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, 2048, false, false, dtype, 4096), init + 2 * (row + 31 * update));
  }
  // b64 validR tail and clamping use the same single-pass kernel, aligned or unaligned storage.
  for (const int64_t storage_r : {64, 65}) {
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, storage_r, false, true, kInt64, 1), 2 * (5 + 1));
    ExpectArgReduceCycles(ArgReduceTestCost(true, 2, storage_r, false, true, kInt64, 33), 2 * (5 + 1 + 1));
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxArSymbolicAlignmentAndFirstPartial) {
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    NodeInfo node;
    auto &params = node.argmax_node_params;
    params.valid = true;
    params.is_ar = true;
    params.src_shape = {CreateExpr(2), CreateExpr("r")};
    params.valid_extent = CreateExpr(0);
    const auto input = MakeTensorShape(dtype, dtype == kInt64 ? 8U : 4U, params.src_shape, {});
    PerfOutputInfo perf;
    ASSERT_EQ(ascir_reduce_v2::ArgmaxApi({input}, {}, node, perf), af::SUCCESS);
    bool found_alignment = false;
    bool found_first_partial = false;
    for (const auto &entry : perf.ternary_ops) {
      const auto branch = entry.second.DeepCopyIfCase();
      ASSERT_NE(branch, nullptr);
      if (Str(entry.first).find("arg_reduce_ar_aligned") != std::string::npos) {
        found_alignment = true;
        EXPECT_EQ(branch->GetCondType(), CondType::K_EQ);
        EXPECT_EQ(branch->GetCondLeft(), af::sym::Mod(CreateExpr("r"), CreateExpr(dtype == kInt64 ? 4 : 8)));
        EXPECT_EQ(branch->GetCondRight(), CreateExpr(0));
      }
      if (Str(entry.first).find("arg_reduce_row_first_partial") != std::string::npos) {
        found_first_partial = true;
        EXPECT_EQ(branch->GetCondType(), CondType::K_EQ);
        EXPECT_EQ(branch->GetCondLeft(), af::sym::Floor(CreateExpr("r") / CreateExpr(dtype == kInt64 ? 32 : 64)));
        EXPECT_EQ(branch->GetCondRight(), CreateExpr(0));
        EXPECT_EQ(branch->GetChoiceA()->GetExpr(), CreateExpr(2));
        EXPECT_EQ(branch->GetChoiceB()->GetExpr(), CreateExpr(0));
      }
    }
    EXPECT_TRUE(found_alignment);
    EXPECT_TRUE(found_first_partial);
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxRaFullAndPartialColumnsCountAllR) {
  // Five-dimensional target: outer A=4*8 calls, core descriptor R=2048, A=32*16.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 512, 2048), 8 * (4 + 7 * 2047));
  // A=65: two column blocks, 254 updates, 128 partial-row masks/aranges; no final Adds(indexBase).
  ExpectArgReduceCycles(ArgReduceTestCost(false, 65, 128), 2 * 3 + 254 * 7 + 128 * 3);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 64, 1), 4);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 64, 3, false, false, kInt32), 4 + 4 * 2);
  auto short_r = ArgReduceTestCost(false, 512, 1024);
  EXPECT_NE(PipeString(short_r, PipeType::AIV_VEC), PipeString(ArgReduceTestCost(false, 512, 2048), PipeType::AIV_VEC));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxPhase2CountsIndexLoadsAndMergeMasked) {
  // Two A blocks (one partial), N=4: UpdateMask 2+4+4, Arange 4+4, 6 merges * 19, 2 stores.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 65, 4, true), 10 + 16 + 6 * 19 + 2, 4);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 65, 4, true, true), 10 + 16 + 6 * 19 + 4, 4);
  // Mixed-width Phase2, A=32: all int32 input rows are gathered, Interleave each row, DeInterleave once.
  // The existing table lacks b64 Select; leave that limitation visible, without borrowing b32 costs.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 32, 4, true, false, kInt64), 4 + 8 + 9 + 4 + 1 + 1 + 1);
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxRaPartialColumnGateBoundaries) {
  // Partial columns: one mask, initial index Duplicate and index Store = 3 cycles/block;
  // each partial source row adds UpdateMask + Arange = 3, each f32/i32 update costs 7/4.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 1, 2), 3 + 2 * 3 + 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 31, 62), 3 + 62 * 3 + 61 * 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 32, 64), 3 + 64 * 3 + 63 * 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 63, 64), 3 + 64 * 3 + 63 * 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 65, 65), 2 * 3 + 65 * 3 + 128 * 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 31, 62, false, true, kInt32), 3 + 62 * 3 + 61 * 4 + 1);
  // Crossing validR<=2*A (while R<=VL) or validR<=VL chooses row-gather.
  // Row-gather: setup 8, row body 24; each block's Arange/Muls/Adds adds 6 cycles.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 1, 3), 8 + 24 + 6);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 31, 63), 8 + 31 * (24 + 6));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 32, 65), 8 + 32 * (24 + 12 + 7 + 2));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 63, 65), 8 + 63 * (24 + 12 + 7 + 2));
  // The gate uses validR, not storage R: both predicates must hold.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 32, 129, false, false, kFloat32, 64), 3 + 64 * 3 + 63 * 7);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 32, 129, false, false, kFloat32, 65), 8 + 32 * (24 + 12 + 7 + 2));
  // b64 VL=32: partial-column mask/arange/duplicate/select have no b64 entries;
  // only GE updates, int32 index Store and DeInterleave contribute.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 16, 32, false, true, kInt64), 2 + 31 + 1);
  ExpectArgReduceCycles(ArgReduceTestCost(false, 16, 33, false, true, kInt64), 16 * (5 + 1 + 1));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 33, 65, false, true, kInt64), 2 * (2 + 64 + 1));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxRaRowGatherSinglePassAndOriginalValue) {
  // A=13, R=129: three blocks (18 offset cycles), two updates (14), tail mask/select (2).
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129), 8 + 13 * (24 + 18 + 14 + 2));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129, false, false, kInt32), 5 + 13 * (12 + 18 + 8 + 2));
  // Original-value recovery adds two Adds (4), Muls (2), mask (1), and one Store (1) per row.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129, false, true), 8 + 13 * (24 + 18 + 14 + 2 + 8));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129, false, true, kInt32), 5 + 13 * (12 + 18 + 8 + 2 + 8));
  // validR=64 still exceeds 2*A: one gathered block, no subsequent tail-mask/select.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129, false, true, kFloat32, 64), 8 + 13 * (24 + 6 + 8));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 13, 129, false, true, kInt64), 13 * (5 + 4 + 1));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxRaColumnGroupsCountUpdatesAndIndexBaseOnce) {
  // Per vector: Duplicate index + Adds indexBase + Store. Updates cost 7/4 for float32/int32.
  for (const int64_t a : {64, 128, 192, 256, 512}) {
    const int64_t columns = a / 64;
    for (const int64_t r : {1, 3, 4, 65}) {
      ExpectArgReduceCycles(ArgReduceTestCost(false, a, r), columns * (4 + (r - 1) * 7));
      ExpectArgReduceCycles(ArgReduceTestCost(false, a, r, false, false, kInt32), columns * (4 + (r - 1) * 4));
    }
  }
  ExpectArgReduceCycles(ArgReduceTestCost(false, 128, 65, false, true), 2 * (5 + 64 * 7));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 128, 65, false, false, kFloat32, 3), 2 * (4 + 2 * 7));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 128, 65, false, false, kFloat32, 4), 2 * (4 + 3 * 7));
  // b64 full columns use the same 4/1-group dispatch; missing b64 entries remain zero.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 128, 5, false, true, kInt64), 4 * (2 + 4 + 1));
  ExpectArgReduceCycles(ArgReduceTestCost(false, 96, 5, false, true, kInt64), 3 * (2 + 4 + 1));
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxWithValueRecoversRowSourceAndStoresColumnWinner) {
  auto base = ArgReduceTestCost(true, 4, 128);
  auto with_value = ArgReduceTestCost(true, 4, 128, false, true);
  auto delta = ResolvedPipeExpr(with_value, PipeType::AIV_VEC) - ResolvedPipeExpr(base, PipeType::AIV_VEC);
  delta.Simplify();
  EXPECT_EQ(Str(delta), "16");
  base = ArgReduceTestCost(false, 128, 4);
  with_value = ArgReduceTestCost(false, 128, 4, false, true);
  delta = ResolvedPipeExpr(with_value, PipeType::AIV_VEC) - ResolvedPipeExpr(base, PipeType::AIV_VEC);
  delta.Simplify();
  EXPECT_EQ(Str(delta), "2");
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxMissingParamsFailsInsteadOfReusingReduceMax) {
  NodeInfo node;
  PerfOutputInfo perf;
  EXPECT_NE(ascir_reduce_v2::ArgmaxApi({}, {}, node, perf), af::SUCCESS);
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxAccumulateCountsHistoryInsideTheSameVf) {
  // Current table: a float MergeMasked is 19 cycles; history kLoad is zero.
  // Preserve one VF head and the existing two stores, including R<4.
  for (int64_t r : {1, 3, 4, 55}) {
    auto base = ArgReduceTestCost(false, 512, r, false, true);
    auto fused = ArgReduceTestCost(false, 512, r, false, true, kFloat32, 0, true);
    auto delta = ResolvedPipeExpr(fused, PipeType::AIV_VEC) - ResolvedPipeExpr(base, PipeType::AIV_VEC);
    delta.Simplify();
    EXPECT_EQ(delta, CreateExpr(8 * 19));
  }
  // i32 MergeMasked costs 9 table cycles, with no float NaN instructions.
  auto base = ArgReduceTestCost(false, 128, 7, false, true, kInt32);
  auto fused = ArgReduceTestCost(false, 128, 7, false, true, kInt32, 0, true);
  auto delta = ResolvedPipeExpr(fused, PipeType::AIV_VEC) - ResolvedPipeExpr(base, PipeType::AIV_VEC);
  delta.Simplify();
  EXPECT_EQ(delta, CreateExpr(2 * 9));
  // Wrapper fallback calls the same reduction; standalone Copy/MergeRound remain outside this model.
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    for (const bool ar : {true, false}) {
      for (const int64_t a : {13, 65}) {
        EXPECT_EQ(PipeString(ArgReduceTestCost(ar, a, 129, false, true, dtype, 0, true), PipeType::AIV_VEC),
                  PipeString(ArgReduceTestCost(ar, a, 129, false, true, dtype), PipeType::AIV_VEC));
      }
    }
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxValidAChargesLogicalRowsAndForcesGather) {
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    for (const bool with_value : {false, true}) {
      for (const bool accumulate : {false, true}) {
        for (const int64_t pitch : {4, 8, 16, 64, 128}) {
          for (const int64_t r : {7, 65, 129}) {
            const auto logical = ArgReduceTestCost(false, pitch, r, false, with_value, dtype, 0, accumulate, 1);
            const auto row = ArgReduceTestCost(false, 1, r, false, with_value, dtype, 0, accumulate);
            EXPECT_EQ(ResolvedPipeExpr(logical, PipeType::AIV_VEC), ResolvedPipeExpr(row, PipeType::AIV_VEC));
          }
        }
      }
    }
  }
  // R=1 would otherwise select the column path. validA<physicalA must force row-gather even at full VL.
  ExpectArgReduceCycles(ArgReduceTestCost(false, 64, 1, false, false, kFloat32, 0, false, 1), 8 + 24 + 6);
  // Original-value recovery: two Adds (4), Muls (2), UpdateMask (1), Store (1).
  ExpectArgReduceCycles(ArgReduceTestCost(false, 16, 1, false, true, kFloat32, 0, true, 1), 8 + 24 + 6 + 8);
  for (const int64_t valid_a : {0, 16, 17}) {
    EXPECT_EQ(PipeString(ArgReduceTestCost(false, 16, 7, false, true, kFloat32, 0, false, valid_a), PipeType::AIV_VEC),
              PipeString(ArgReduceTestCost(false, 16, 7, false, true), PipeType::AIV_VEC));
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxSymbolicValidAIsPreservedInDispatch) {
  NodeInfo node;
  auto &params = node.argmax_node_params;
  params.valid = true;
  params.src_shape = {CreateExpr(7), CreateExpr("physical_a")};
  params.valid_extent = CreateExpr(7);
  params.valid_a = CreateExpr("logical_a");
  PerfOutputInfo perf;
  const auto input = MakeTensorShape(kFloat32, 4U, params.src_shape, {});
  ASSERT_EQ(ascir_reduce_v2::ArgmaxApi({input}, {}, node, perf), af::SUCCESS);
  bool found_valid_a = false;
  for (const auto &entry : perf.ternary_ops) {
    if (Str(entry.first).find("arg_reduce_ra_valid_a") != std::string::npos) {
      const auto branch = entry.second.DeepCopyIfCase();
      ASSERT_NE(branch, nullptr);
      EXPECT_EQ(branch->GetCondRight(), CreateExpr("physical_a"));
      found_valid_a = true;
    }
  }
  EXPECT_TRUE(found_valid_a);
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgmaxSymbolicRaRetainsRuntimeDispatch) {
  NodeInfo node;
  auto &params = node.argmax_node_params;
  params.valid = true;
  params.src_shape = {CreateExpr("r"), CreateExpr("a")};
  params.valid_extent = CreateExpr(0);  // API default: use the symbolic storage R.
  const auto input = MakeTensorShape(kFloat32, 4U, params.src_shape, {});
  PerfOutputInfo perf;
  ASSERT_EQ(ascir_reduce_v2::ArgmaxApi({input}, {}, node, perf), af::SUCCESS);
  bool found_dispatch = false;
  bool found_partial = false;
  bool found_short = false;
  bool found_wide = false;
  for (const auto &entry : perf.ternary_ops) {
    const auto branch = entry.second.DeepCopyIfCase();
    ASSERT_NE(branch, nullptr);
    if (Str(entry.first).find("arg_reduce_ra_full_columns") != std::string::npos) {
      found_dispatch = true;
      EXPECT_EQ(branch->GetCondType(), CondType::K_EQ);
      EXPECT_EQ(branch->GetCondLeft(), af::sym::Mod(CreateExpr("a"), CreateExpr(64)));
      EXPECT_EQ(branch->GetCondRight(), CreateExpr(0));
      EXPECT_NE(Str(branch->GetChoiceA()->GetExpr()), Str(branch->GetChoiceB()->GetExpr()));
    }
    if (Str(entry.first).find("arg_reduce_ra_partial_columns") != std::string::npos) {
      found_partial = true;
      EXPECT_EQ(branch->GetCondType(), CondType::K_LE);
      EXPECT_EQ(branch->GetCondLeft(), CreateExpr(64));
      EXPECT_EQ(branch->GetCondRight(), CreateExpr("a"));
    }
    if (Str(entry.first).find("arg_reduce_ra_short_r") != std::string::npos) {
      found_short = true;
      EXPECT_EQ(branch->GetCondType(), CondType::K_LE);
      EXPECT_EQ(branch->GetCondLeft(), CreateExpr("r"));
      EXPECT_EQ(branch->GetCondRight(), CreateExpr(64));
    }
    if (Str(entry.first).find("arg_reduce_ra_short_wide") != std::string::npos) {
      found_wide = true;
      EXPECT_EQ(branch->GetCondType(), CondType::K_LE);
      EXPECT_EQ(branch->GetCondLeft(), CreateExpr("r"));
      EXPECT_EQ(branch->GetCondRight(), CreateExpr(2) * CreateExpr("a"));
    }
  }
  EXPECT_TRUE(found_dispatch);
  EXPECT_TRUE(found_partial);
  EXPECT_TRUE(found_short);
  EXPECT_TRUE(found_wide);
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgminRegistersItsOwnCallbackForEveryPhase) {
  const auto maximum = ApiPerfFactory::Instance().Create(kArgmax + "V2");
  ASSERT_NE(maximum, nullptr);
  for (const auto &type :
       {kArgmin, kArgminWithValue, kArgMinMultiRPhase1, kArgMinMultiRPhase2, kArgMinMultiRPhase2WithValue}) {
    SCOPED_TRACE(type);
    const auto minimum = ApiPerfFactory::Instance().Create(type + "V2");
    ASSERT_NE(minimum, nullptr);
    EXPECT_EQ(minimum->GetPerfFunc(), &ascir_reduce_v2::ArgminApi);
    EXPECT_NE(minimum->GetPerfFunc(), &ascir_reduce_v2::ArgmaxApi);
    EXPECT_EQ(minimum->GetTilingScheduleConfigTable(), maximum->GetTilingScheduleConfigTable());
    const auto impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(soc_perf_test::kV35NpuArch, type.c_str());
    ASSERT_NE(impl, nullptr);
    EXPECT_STREQ(reinterpret_cast<const char *>(impl->GetApiPerf()), (type + "V2").c_str());
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgminKeepsSharedInstructionCountsAcrossLayoutsTailsAndDtypes) {
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    for (int mode : {0, 1, 2}) {
      const bool ar = mode == 0, phase2 = mode == 2;
      for (const bool with_value : {false, true}) {
        for (int64_t a : {1, 31, 32, 33, 63, 64, 65, 128, 256}) {
          for (int64_t r : {1, 3, 32, 33, 64, 65, 129}) {
            const auto maximum = ArgReduceTestCost(ar, a, r, phase2, with_value, dtype);
            const auto minimum = ArgReduceTestCost(ar, a, r, phase2, with_value, dtype, 0, false, 0, true);
            EXPECT_EQ(ResolvedPipeExpr(minimum, PipeType::AIV_VEC), ResolvedPipeExpr(maximum, PipeType::AIV_VEC))
                << dtype << " mode=" << mode << " A=" << a << " R=" << r;
          }
        }
      }
    }
  }
}

TEST_F(UTestReduceMinMaxApiPerfV2, ArgminSymbolicRowsRetainAlignmentAndFirstPartialChoices) {
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    NodeInfo node;
    auto &params = node.argmax_node_params;
    params.valid = true;
    params.is_ar = true;
    params.with_value = true;
    params.src_shape = {CreateExpr(2), CreateExpr("r")};
    params.valid_extent = CreateExpr(0);
    const auto input = MakeTensorShape(dtype, dtype == kInt64 ? 8U : 4U, params.src_shape, {});
    PerfOutputInfo minimum;
    ASSERT_EQ(ascir_reduce_v2::ArgminApi({input}, {}, node, minimum), af::SUCCESS);
    bool first_partial = false;
    for (const auto &entry : minimum.ternary_ops) {
      if (Str(entry.first).find("arg_reduce_row_first_partial") != std::string::npos) {
        const auto choice = entry.second.DeepCopyIfCase();
        ASSERT_NE(choice, nullptr);
        EXPECT_EQ(choice->GetCondLeft(), Floor(CreateExpr("r") / CreateExpr(dtype == kInt64 ? 32 : 64)));
        EXPECT_EQ(choice->GetCondType(), CondType::K_EQ);
        EXPECT_EQ(choice->GetChoiceA()->GetExpr(), CreateExpr(2));
        EXPECT_EQ(choice->GetChoiceB()->GetExpr(), CreateExpr(0));
        first_partial = true;
      }
    }
    EXPECT_TRUE(first_partial);
  }
}
