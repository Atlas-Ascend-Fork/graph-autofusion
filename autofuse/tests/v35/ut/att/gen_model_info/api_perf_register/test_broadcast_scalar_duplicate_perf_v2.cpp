/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ascir_node_param/ascir_node_param.h"
#include "common/checker.h"
#include "base/att_const_values.h"
#include "api_perf_register/api_perf_factory.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_perf_utils_v2.h"
#include "../../../../../ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"
#include "tests/depends/slog/src/slog_stub.h"

namespace att {
namespace {
using ascendcapi_v2::VfCostAccumulator;

TEST(BroadcastScalarDuplicatePerfV2, ScalarDuplicateCountsVectorRegisters) {
  const std::vector<std::tuple<std::string, int64_t, double>> cases = {
      {kFloat16, 8, 29}, {kFloat16, 129, 30}, {kFloat32, 65, 30}, {kFloat32, 45056, 732},
      {kUInt8, 256, 29}, {kUInt8, 257, 30},   {kInt64, 33, 30},   {kInt64, 64, 30},
  };
  for (const auto &[dtype, duplicate_count, expected_value] : cases) {
    TensorShapeInfo input;
    input.data_type = dtype;
    TensorShapeInfo output;
    output.data_type = dtype;
    NodeInfo node;
    node.broadcast_node_params.valid = true;
    node.broadcast_node_params.is_scalar = true;
    node.broadcast_node_params.duplicate_count = CreateExpr(duplicate_count);

    const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
    ASSERT_NE(api, nullptr);
    PerfOutputInfo perf;
    ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
    const auto iter = perf.pipe_res.find(PipeType::AIV_VEC);
    ASSERT_NE(iter, perf.pipe_res.end());
    double value = 0.0;
    const std::vector<std::pair<af::Expression, af::Expression>> no_vars;
    ASSERT_EQ(iter->second.GetResult(no_vars, value), af::GRAPH_SUCCESS);
    EXPECT_EQ(value, expected_value) << "dtype=" << dtype << " duplicate_count=" << duplicate_count;
  }
}

TEST(BroadcastScalarDuplicatePerfV2, ScalarDuplicateSymbolicCountDividesByVectorLength) {
  TensorShapeInfo input;
  input.data_type = kFloat32;
  TensorShapeInfo output;
  output.data_type = kFloat32;
  NodeInfo node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.is_scalar = true;
  node.broadcast_node_params.duplicate_count = CreateExpr("z1z2t_size");

  const auto api = ApiPerfFactory::Instance().Create(kBroadcast + "V2");
  ASSERT_NE(api, nullptr);
  PerfOutputInfo perf;
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  const auto iter = perf.pipe_res.find(PipeType::AIV_VEC);
  ASSERT_NE(iter, perf.pipe_res.end());

  Expr vl;
  Expr half_vl;
  Expr block_elements;
  ASSERT_EQ(ascendcapi_v2::GetBroadcastVectorElements(kFloat32, vl, half_vl, block_elements), af::SUCCESS);
  Expr repeat_time;
  ASSERT_EQ(ascendcapi_v2::CeilDiv(CreateExpr("z1z2t_size"), vl, repeat_time), af::SUCCESS);
  VfCostAccumulator acc;
  ASSERT_EQ(ascendcapi_v2::AddVfInstructPerf(kDuplicate, kFloat32, repeat_time, 1U, acc), af::SUCCESS);
  Expr expected = ascendcapi_v2::GetVfCost(acc);
  EXPECT_EQ(iter->second, expected);

  // A fixed-rank BroadcastExtend still has scalar shape metadata. Its single
  // source element is loaded by the vector pipeline, independently of graph rank.
  node.broadcast_node_params.const_rank = 1;
  ASSERT_EQ(api->GetPerfFunc()({input}, {output}, node, perf), af::SUCCESS);
  VfCostAccumulator vector_acc;
  ASSERT_EQ(ascendcapi_v2::AddVfInstructPerf(kLoad, kFloat32, CreateExpr(1), 1U, vector_acc), af::SUCCESS);
  ASSERT_EQ(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, kFloat32, repeat_time, 1U, vector_acc), af::SUCCESS);
  ASSERT_EQ(ascendcapi_v2::AddVfInstructPerf(kStore, kFloat32, repeat_time, 1U, vector_acc), af::SUCCESS);
  EXPECT_EQ(perf.pipe_res.at(PipeType::AIV_VEC), ascendcapi_v2::GetVfCost(vector_acc));
}
}  // namespace
}  // namespace att
