/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "att/base/att_const_values.h"
#include "parser/tuning_space.h"
#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"

using namespace att;

namespace {
NodeInfo MakeArgmaxPerfNode(bool is_ar, bool is_phase2, bool with_value, bool accumulate,
                            const std::vector<Expr> &src_shape, const Expr &valid_extent, const Expr &valid_a) {
  NodeInfo node;
  node.name = "argmax_perf_node";
  auto &params = node.argmax_node_params;
  params.valid = true;
  params.is_ar = is_ar;
  params.is_phase2 = is_phase2;
  params.with_value = with_value;
  params.has_accumulate = accumulate;
  params.src_shape = src_shape;
  params.valid_extent = valid_extent;
  params.valid_a = valid_a;
  return node;
}

TensorShapeInfo MakeArgmaxPerfShape(const std::string &dtype) {
  TensorShapeInfo shape;
  shape.data_type = dtype;
  shape.data_type_size = 4;
  shape.dims = {CreateExpr(8), CreateExpr(64)};
  shape.repeats = shape.dims;
  shape.strides = {CreateExpr(64), CreateExpr(1)};
  shape.gm_strides = shape.strides;
  return shape;
}

void RunArgmaxPerfCombo(const std::string &dtype, bool is_ar, bool is_phase2, bool with_value, bool accumulate,
                        const std::vector<Expr> &src_shape, const Expr &valid_extent, const Expr &valid_a) {
  const auto shape = MakeArgmaxPerfShape(dtype);
  const std::vector<TensorShapeInfo> inputs{shape};
  const std::vector<TensorShapeInfo> outputs{shape};
  const auto node = MakeArgmaxPerfNode(is_ar, is_phase2, with_value, accumulate, src_shape, valid_extent, valid_a);
  PerfOutputInfo perf;
  EXPECT_EQ(ascir_reduce_v2::ArgmaxApi(inputs, outputs, node, perf), af::SUCCESS);
  EXPECT_FALSE(perf.pipe_res.empty());
}
}  // namespace

TEST(ArgmaxPerfApi, CoversPatterns) {
  for (const auto &dtype : {kFloat32, kInt32, kInt64}) {
    for (const bool with_value : {false, true}) {
      for (const auto &valid_a : {CreateExpr(0), CreateExpr(4)}) {
        RunArgmaxPerfCombo(dtype, true, false, with_value, false, {CreateExpr(8), CreateExpr(64)}, CreateExpr(64),
                           valid_a);
        RunArgmaxPerfCombo(dtype, false, false, with_value, false, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8),
                           valid_a);
        RunArgmaxPerfCombo(dtype, false, false, with_value, true, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8),
                           valid_a);
        RunArgmaxPerfCombo(dtype, false, true, with_value, false, {CreateExpr(2), CreateExpr(8), CreateExpr(8)},
                           CreateExpr(8), valid_a);
      }
    }
  }
  const Expr symbolic_a = CreateExpr("argmax_perf_a");
  const Expr symbolic_r = CreateExpr("argmax_perf_r");
  const Expr symbolic_valid_a = CreateExpr("argmax_perf_valid_a");
  RunArgmaxPerfCombo(kFloat32, true, false, true, false, {symbolic_a, symbolic_r}, symbolic_r, symbolic_valid_a);
  RunArgmaxPerfCombo(kInt32, false, false, true, true, {symbolic_a, symbolic_r}, symbolic_a, symbolic_valid_a);
  RunArgmaxPerfCombo(kInt64, false, true, true, false, {CreateExpr(2), symbolic_a, symbolic_r}, symbolic_a,
                     symbolic_valid_a);
}

TEST(ArgmaxPerfApi, Entrypoints) {
  const auto shape = MakeArgmaxPerfShape(kFloat32);
  const std::vector<TensorShapeInfo> inputs{shape};
  const std::vector<TensorShapeInfo> outputs{shape};
  PerfOutputInfo perf;
  auto with_value =
      MakeArgmaxPerfNode(false, false, true, false, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8), CreateExpr(0));
  EXPECT_EQ(ascir_reduce_v2::ArgmaxWithValueApi(inputs, outputs, with_value, perf), af::SUCCESS);
  auto phase1 =
      MakeArgmaxPerfNode(false, false, true, true, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8), CreateExpr(0));
  EXPECT_EQ(ascir_reduce_v2::ArgMaxMultiRPhase1Api(inputs, outputs, phase1, perf), af::SUCCESS);
  auto phase2 = MakeArgmaxPerfNode(false, true, false, false, {CreateExpr(2), CreateExpr(8), CreateExpr(8)},
                                   CreateExpr(8), CreateExpr(0));
  EXPECT_EQ(ascir_reduce_v2::ArgMaxMultiRPhase2Api(inputs, outputs, phase2, perf), af::SUCCESS);
  auto phase2_value = MakeArgmaxPerfNode(false, true, true, false, {CreateExpr(2), CreateExpr(8), CreateExpr(8)},
                                         CreateExpr(8), CreateExpr(4));
  EXPECT_EQ(ascir_reduce_v2::ArgMaxMultiRPhase2WithValueApi(inputs, outputs, phase2_value, perf), af::SUCCESS);
}

TEST(ArgmaxPerfApi, RejectsInvalidInputs) {
  const auto shape = MakeArgmaxPerfShape(kFloat32);
  const std::vector<TensorShapeInfo> inputs{shape};
  const std::vector<TensorShapeInfo> outputs{shape};
  PerfOutputInfo perf;
  auto node =
      MakeArgmaxPerfNode(false, false, false, false, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8), CreateExpr(0));
  EXPECT_NE(ascir_reduce_v2::ArgmaxApi({}, outputs, node, perf), af::SUCCESS);
  node.argmax_node_params.valid = false;
  EXPECT_NE(ascir_reduce_v2::ArgmaxApi(inputs, outputs, node, perf), af::SUCCESS);
  node = MakeArgmaxPerfNode(false, false, false, false, {CreateExpr(8)}, CreateExpr(8), CreateExpr(0));
  EXPECT_NE(ascir_reduce_v2::ArgmaxApi(inputs, outputs, node, perf), af::SUCCESS);
  node = MakeArgmaxPerfNode(false, false, false, false, {CreateExpr(8), CreateExpr(16)}, CreateExpr(8), CreateExpr(0));
  auto unsupported = shape;
  unsupported.data_type = kFloat16;
  EXPECT_NE(ascir_reduce_v2::ArgmaxApi({unsupported}, {unsupported}, node, perf), af::SUCCESS);
}
