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
#include "graph/ascendc_ir/ascir_registry.h"
#include "base/att_const_values.h"
#include "api_perf_register/v1/perf_param_v1.h"
#include "ascir/generator/v1_ascir_att_impl.h"
#include "api_perf_register/api_perf_factory.h"

using namespace att;
using namespace af::sym;
using namespace af::ascir;
class UTestAscirPerfV1 : public ::testing::Test {
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

TEST_F(UTestAscirPerfV1, TestPerfParamTableV1_GetAscendCApiPerfTable) {
  PerfParamTableV1 default_ir_att_v1;
  EXPECT_NE(default_ir_att_v1.GetAscendCApiPerfTable(), nullptr);
  auto pipe_head_perf_func = default_ir_att_v1.GetPipeHeadPerfFunc(att::PipeType::AIV_MTE3);
  ASSERT_NE(pipe_head_perf_func, nullptr);
  std::map<att::Expr, att::TernaryOp, att::ExprCmp> cond_map;
  auto expr1 = pipe_head_perf_func({}, cond_map);
  EXPECT_NE(std::string(expr1.Str().get()), "");
  pipe_head_perf_func = default_ir_att_v1.GetPipeHeadPerfFunc(att::PipeType::AIV_MTE2);
  ASSERT_NE(pipe_head_perf_func, nullptr);
  expr1 = pipe_head_perf_func({}, cond_map);
  EXPECT_NE(std::string(expr1.Str().get()), "");
  pipe_head_perf_func = default_ir_att_v1.GetPipeHeadPerfFunc(att::PipeType::AIV_VEC);
  ASSERT_NE(pipe_head_perf_func, nullptr);
  expr1 = pipe_head_perf_func({}, cond_map);
  EXPECT_NE(std::string(expr1.Str().get()), "");
}

TEST_F(UTestAscirPerfV1, TestAscendCApiPerfTableSize) {
  AbsAscIrAttImpl default_ir_att_v1;
  EXPECT_NE(default_ir_att_v1.GetAscendCApiPerfTable(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetAddApiPerf) {
  AbsAscIrAttImpl default_ir_att_v1;
  EXPECT_NE(default_ir_att_v1.GetApiPerf(), nullptr);
  AddAscIrAttImpl add_ir_att_v1;
  EXPECT_NE(add_ir_att_v1.GetAscendCApiPerfTable(), nullptr);
  EXPECT_NE(add_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetGatherApiPerf) {
  GatherAscIrAttImpl gather_ir_att_v1;
  EXPECT_NE(gather_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetBroadcastApiPerf) {
  BroadcastAscIrAttImpl broadcast_ir_att_v1;
  EXPECT_NE(broadcast_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetCastApiPerf) {
  CastAscIrAttImpl cast_ir_att_v1;
  EXPECT_NE(cast_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetDivApiPerf) {
  DivAscIrAttImpl div_ir_att_v1;
  EXPECT_NE(div_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetErfApiPerf) {
  ErfAscIrAttImpl erf_ir_att_v1;
  EXPECT_NE(erf_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetExpApiPerf) {
  ExpAscIrAttImpl exp_ir_att_v1;
  EXPECT_NE(exp_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetAbsApiPerf) {
  AbsAscIrAttImpl abs_ir_att_v1;
  EXPECT_NE(abs_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetLogicalAndApiPerf) {
  LogicalAndAscIrAttImpl logical_and_ir_att_v1;
  EXPECT_NE(logical_and_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetLogicalOrApiPerf) {
  LogicalOrAscIrAttImpl logical_or_ir_att_v1;
  EXPECT_NE(logical_or_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetLogicalNotApiPerf) {
  LogicalNotAscIrAttImpl logical_not_ir_att_v1;
  EXPECT_NE(logical_not_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetMaximumApiPerf) {
  MaximumAscIrAttImpl maximum_ir_att_v1;
  EXPECT_NE(maximum_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetMinimumApiPerf) {
  MinimumAscIrAttImpl minimum_ir_att_v1;
  EXPECT_NE(minimum_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetMinApiPerf) {
  MinAscIrAttImpl min_ir_att_v1;
  EXPECT_NE(min_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetMulApiPerf) {
  MulAscIrAttImpl mul_ir_att_v1;
  EXPECT_NE(mul_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetNegApiPerf) {
  NegAscIrAttImpl neg_ir_att_v1;
  EXPECT_NE(neg_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReciprocalApiPerf) {
  ReciprocalAscIrAttImpl reciprocal_ir_att_v1;
  EXPECT_NE(reciprocal_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReluApiPerf) {
  ReluAscIrAttImpl relu_ir_att_v1;
  EXPECT_NE(relu_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceAllApiPerf) {
  ReduceAllAscIrAttImpl reduce_all_ir_att_v1;
  EXPECT_NE(reduce_all_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceAnyApiPerf) {
  ReduceAnyAscIrAttImpl reduce_any_ir_att_v1;
  EXPECT_NE(reduce_any_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceMaxApiPerf) {
  ReduceMaxAscIrAttImpl reduce_max_ir_att_v1;
  EXPECT_NE(reduce_max_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceMeanApiPerf) {
  ReduceMeanAscIrAttImpl reduce_mean_ir_att_v1;
  EXPECT_NE(reduce_mean_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceMinApiPerf) {
  ReduceMinAscIrAttImpl reduce_min_ir_att_v1;
  EXPECT_NE(reduce_min_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceSumApiPerf) {
  ReduceSumAscIrAttImpl reduce_sum_ir_att_v1;
  EXPECT_NE(reduce_sum_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetReduceProdApiPerf) {
  ReduceProdAscIrAttImpl reduce_prod_ir_att_v1;
  EXPECT_NE(reduce_prod_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetRemovePadApiPerf) {
  RemovePadAscIrAttImpl remove_pad_ir_att_v1;
  EXPECT_NE(remove_pad_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetRsqrtApiPerf) {
  RsqrtAscIrAttImpl rsqrt_ir_att_v1;
  EXPECT_NE(rsqrt_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetSelectApiPerf) {
  SelectAscIrAttImpl select_ir_att_v1;
  EXPECT_NE(select_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetGeApiPerf) {
  GeAscIrAttImpl ge_ir_att_v1;
  EXPECT_NE(ge_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetEqApiPerf) {
  EqAscIrAttImpl eq_ir_att_v1;
  EXPECT_NE(eq_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetNeApiPerf) {
  NeAscIrAttImpl ne_ir_att_v1;
  EXPECT_NE(ne_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetGtApiPerf) {
  GtAscIrAttImpl gt_ir_att_v1;
  EXPECT_NE(gt_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetLeApiPerf) {
  LeAscIrAttImpl le_ir_att_v1;
  EXPECT_NE(le_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetLtApiPerf) {
  LtAscIrAttImpl lt_ir_att_v1;
  EXPECT_NE(lt_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetSignApiPerf) {
  SignAscIrAttImpl sign_ir_att_v1;
  EXPECT_NE(sign_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetSqrtApiPerf) {
  SqrtAscIrAttImpl sqrt_ir_att_v1;
  EXPECT_NE(sqrt_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetSubApiPerf) {
  SubAscIrAttImpl sub_ir_att_v1;
  EXPECT_NE(sub_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetSumApiPerf) {
  SumAscIrAttImpl sum_ir_att_v1;
  EXPECT_NE(sum_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetTanhApiPerf) {
  TanhAscIrAttImpl tanh_ir_att_v1;
  EXPECT_NE(tanh_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetWhereApiPerf) {
  WhereAscIrAttImpl where_ir_att_v1;
  EXPECT_NE(where_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestGetUb2ubApiPerf) {
  Ub2ubAscIrAttImpl ub2ub_ir_att_v1;
  EXPECT_NE(ub2ub_ir_att_v1.GetApiPerf(), nullptr);
}

TEST_F(UTestAscirPerfV1, TestDefaultGetPerf) {
  std::vector<TensorShapeInfo> input_shapes;
  std::vector<TensorShapeInfo> output_shapes;
  NodeInfo node;
  PerfOutputInfo res;
  EXPECT_EQ(DefaultGetPerf(input_shapes, output_shapes, node, res), af::SUCCESS);
}
