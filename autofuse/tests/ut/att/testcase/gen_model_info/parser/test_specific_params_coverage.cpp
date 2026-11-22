/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "ascir_ops.h"
#include "base/att_const_values.h"
#include "parser/specific_params_builder.h"
#include "ascir_node_param/ascir_param_builder.h"

namespace att {
namespace {
class SpecificParamsCoverageTest : public testing::Test {
 protected:
  void SetUp() override {
    af::ascir_op::Add op("node");
    graph_.AddNode(op);
    node_ = graph_.FindNode("node");
    ASSERT_NE(node_, nullptr);
    params_ = ascir_param::GetOrCreateAscirNodeParams(node_);
    ASSERT_NE(params_, nullptr);
    params_->status = ascir_param::ParamBuildStatus::kBuilt;
    info_.name = "node";
  }

  template <typename T>
  af::Status Fill(const std::string &type, const T &payload) {
    info_.node_type = type;
    params_->specific_params = payload;
    return FillSpecificParams(node_, info_);
  }

  af::AscGraph graph_{"specific_params"};
  af::AscNodePtr node_;
  ascir_param::AscirNodeParamsPtr params_;
  NodeInfo info_;
};

TEST_F(SpecificParamsCoverageTest, CopiesCastLayoutWithoutAliasingPayload) {
  ascir_param::CastNodeParams payload;
  payload.valid = true;
  payload.output_dims = {af::Symbol(32)};
  payload.output_strides = {af::Symbol(1)};
  payload.input_strides = {af::Symbol(2)};
  for (const auto &type : {kCast, kCeil2Int}) {
    SCOPED_TRACE(type);
    info_.cast_node_params = {};
    ASSERT_EQ(Fill(type, payload), af::SUCCESS);
    EXPECT_TRUE(info_.cast_node_params.valid);
    EXPECT_EQ(info_.cast_node_params.output_dims, payload.output_dims);
    EXPECT_EQ(info_.cast_node_params.output_strides, payload.output_strides);
    EXPECT_EQ(info_.cast_node_params.input_strides, payload.input_strides);
    params_->specific_params = std::monostate{};
    EXPECT_EQ(info_.cast_node_params.output_dims, payload.output_dims);
  }
}

TEST_F(SpecificParamsCoverageTest, CopiesComparisonPayloadForAllComparisonKinds) {
  ascir_param::CompareNodeParams payload;
  payload.valid = true;
  payload.is_scalar = true;
  payload.outer_call_count = af::Symbol(3);
  payload.output_dims = {af::Symbol(8)};
  payload.output_strides = {af::Symbol(1)};
  payload.input_strides = {af::Symbol(0)};
  for (const auto &type : {kGe, kEq, kNe, kGt, kLe, kLt}) {
    SCOPED_TRACE(type);
    info_.compare_node_params = {};
    ASSERT_EQ(Fill(type, payload), af::SUCCESS);
    EXPECT_TRUE(info_.compare_node_params.valid);
    EXPECT_TRUE(info_.compare_node_params.is_scalar);
    EXPECT_EQ(info_.compare_node_params.outer_call_count, payload.outer_call_count);
    EXPECT_EQ(info_.compare_node_params.output_dims, payload.output_dims);
    EXPECT_EQ(info_.compare_node_params.input_strides, payload.input_strides);
    EXPECT_EQ(info_.compare_node_params.output_strides, payload.output_strides);
  }
}

TEST_F(SpecificParamsCoverageTest, CopiesWhereAndSelectBroadcastAndMaskMetadata) {
  ascir_param::WhereNodeParams payload;
  payload.valid = true;
  payload.is_bcast_src0 = true;
  payload.is_bcast_src1 = true;
  payload.outer_call_count = af::Symbol(5);
  payload.output_dims = {af::Symbol(16)};
  payload.output_strides = {af::Symbol(1)};
  payload.mask_strides = {af::Symbol(0)};
  payload.input_strides = {af::Symbol(2)};
  for (const auto &type : {kWhere, kSelect}) {
    SCOPED_TRACE(type);
    info_.where_node_params = {};
    ASSERT_EQ(Fill(type, payload), af::SUCCESS);
    EXPECT_TRUE(info_.where_node_params.valid);
    EXPECT_TRUE(info_.where_node_params.is_bcast_src0);
    EXPECT_TRUE(info_.where_node_params.is_bcast_src1);
    EXPECT_EQ(info_.where_node_params.outer_call_count, payload.outer_call_count);
    EXPECT_EQ(info_.where_node_params.output_dims, payload.output_dims);
    EXPECT_EQ(info_.where_node_params.output_strides, payload.output_strides);
    EXPECT_EQ(info_.where_node_params.mask_strides, payload.mask_strides);
    EXPECT_EQ(info_.where_node_params.input_strides, payload.input_strides);
  }
}

TEST_F(SpecificParamsCoverageTest, CopiesUnaryBitWidthMetadataForAllPredicates) {
  ascir_param::UnaryBitWidthChangeNodeParams payload;
  payload.valid = true;
  payload.cal_count = af::Symbol(64);
  payload.outer_repeats = {af::Symbol(3)};
  payload.output_strides = {af::Symbol(1)};
  payload.input_strides = {af::Symbol(4)};
  for (const auto &type : {kIsnan, kIsFinite, kIsInf}) {
    SCOPED_TRACE(type);
    info_.unary_bitwidth_change_node_params = {};
    ASSERT_EQ(Fill(type, payload), af::SUCCESS);
    EXPECT_TRUE(info_.unary_bitwidth_change_node_params.valid);
    EXPECT_EQ(info_.unary_bitwidth_change_node_params.cal_count, payload.cal_count);
    EXPECT_EQ(info_.unary_bitwidth_change_node_params.outer_repeats, payload.outer_repeats);
    EXPECT_EQ(info_.unary_bitwidth_change_node_params.output_strides, payload.output_strides);
    EXPECT_EQ(info_.unary_bitwidth_change_node_params.input_strides, payload.input_strides);
  }
}

TEST_F(SpecificParamsCoverageTest, CopiesTransposeAndVectorFunctionMetadata) {
  ascir_param::TransposeNodeParams transpose;
  transpose.valid = true;
  transpose.inner_dim = 2;
  transpose.total_dim = 3;
  transpose.outer_loop_axes = {af::Symbol(5)};
  transpose.output_dims = {af::Symbol(8), af::Symbol(4)};
  transpose.output_strides = {af::Symbol(4), af::Symbol(1)};
  transpose.input_strides = {af::Symbol(1), af::Symbol(8)};
  ASSERT_EQ(Fill(kTranspose, transpose), af::SUCCESS);
  EXPECT_TRUE(info_.transpose_node_params.valid);
  EXPECT_EQ(info_.transpose_node_params.inner_dim, 2U);
  EXPECT_EQ(info_.transpose_node_params.total_dim, 3U);
  EXPECT_EQ(info_.transpose_node_params.outer_loop_axes, transpose.outer_loop_axes);
  EXPECT_EQ(info_.transpose_node_params.output_dims, transpose.output_dims);
  EXPECT_EQ(info_.transpose_node_params.input_strides, transpose.input_strides);
  EXPECT_EQ(info_.transpose_node_params.output_strides, transpose.output_strides);
  ascir_param::VectorFuncNodeParams vector_func;
  vector_func.is_double_loop = true;
  vector_func.all_strides = {af::Symbol(4), af::Symbol(1)};
  vector_func.output_dims = {af::Symbol(8), af::Symbol(4)};
  ASSERT_EQ(Fill(kVectorFunc, vector_func), af::SUCCESS);
  EXPECT_TRUE(info_.vector_func_params.is_double_loop);
  EXPECT_EQ(info_.vector_func_params.all_strides, vector_func.all_strides);
  EXPECT_EQ(info_.vector_func_params.output_dims, vector_func.output_dims);
}

TEST_F(SpecificParamsCoverageTest, RejectsMissingPayloadAndInvalidBuildStateWithoutOverwritingOutput) {
  info_.cast_node_params.valid = true;
  info_.cast_node_params.output_dims = {af::Symbol(19)};
  for (const auto &type : {kCast, kCeil2Int, kGe, kWhere, kIsnan, kIsInf, kTranspose, kVectorFunc, kSum}) {
    SCOPED_TRACE(type);
    EXPECT_NE(Fill(type, std::monostate{}), af::SUCCESS);
  }
  EXPECT_TRUE(info_.cast_node_params.valid);
  EXPECT_EQ(info_.cast_node_params.output_dims, (std::vector<ge::Expression>{af::Symbol(19)}));
  params_->status = ascir_param::ParamBuildStatus::kInvalid;
  EXPECT_NE(Fill(kCast, ascir_param::CastNodeParams{}), af::SUCCESS);
  params_->status = ascir_param::ParamBuildStatus::kSkipped;
  EXPECT_EQ(Fill(kCast, std::monostate{}), af::SUCCESS);
  EXPECT_EQ(info_.cast_node_params.output_dims, (std::vector<ge::Expression>{af::Symbol(19)}));
  params_->status = ascir_param::ParamBuildStatus::kBuilt;
  EXPECT_EQ(Fill("Add", std::monostate{}), af::SUCCESS);
}
TEST_F(SpecificParamsCoverageTest, CacheExecutionSurvivesSkippedSpecificPayloadAndIsCopiedByValue) {
  params_->status = ascir_param::ParamBuildStatus::kSkipped;
  params_->cache_executions.emplace_back();
  params_->cache_executions.front().mode = ascir_param::CacheExecutionMode::kNoGuard;
  params_->cache_executions.front().call_loop_axis_id = 9;
  ASSERT_EQ(FillSpecificParams(node_, info_), af::SUCCESS);
  params_->cache_executions.front().mode = ascir_param::CacheExecutionMode::kUnknown;
  EXPECT_EQ(info_.cache_executions.front().mode, ascir_param::CacheExecutionMode::kNoGuard);
  EXPECT_EQ(info_.cache_executions.front().call_loop_axis_id, 9);
}

TEST_F(SpecificParamsCoverageTest, CacheExecutionRetainsDistinctCalls) {
  ascir_param::CacheExecutionParams execution;
  execution.mode = ascir_param::CacheExecutionMode::kNoGuard;
  execution.call_loop_axis_id = 9;
  ascir_param::RecordCacheExecution(*params_, execution);
  execution.mode = ascir_param::CacheExecutionMode::kOriginBroadcast;
  execution.call_loop_axis_id = 10;
  ascir_param::RecordCacheExecution(*params_, execution);
  ASSERT_EQ(params_->cache_executions.size(), 2U);
  EXPECT_EQ(params_->cache_executions[0].mode, ascir_param::CacheExecutionMode::kNoGuard);
  EXPECT_EQ(params_->cache_executions[0].call_loop_axis_id, 9);
  EXPECT_EQ(params_->cache_executions[1].mode, ascir_param::CacheExecutionMode::kOriginBroadcast);
  EXPECT_EQ(params_->cache_executions[1].call_loop_axis_id, 10);
}

TEST_F(SpecificParamsCoverageTest, EnrichmentClearsPriorGenerationCacheExecution) {
  params_->cache_executions.emplace_back();
  params_->cache_executions.front().mode = ascir_param::CacheExecutionMode::kOriginBroadcast;
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph_), af::SUCCESS);
  EXPECT_TRUE(ascir_param::GetAscirNodeParams(node_)->cache_executions.empty());
}
TEST_F(SpecificParamsCoverageTest, EnrichmentDoesNotClearAnotherCandidatesRecords) {
  node_->GetOpDesc()->SetType("Broadcast");
  ascir_param::BroadcastNodeParams payload;
  payload.valid = true;
  params_->specific_params = payload;
  params_->cache_executions.emplace_back();
  params_->cache_executions.front().mode = ascir_param::CacheExecutionMode::kNoGuard;
  af::AscGraph clone("cloned_candidate");
  ASSERT_TRUE(clone.CopyFrom(graph_));
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(clone), af::SUCCESS);
  const auto cloned_params = ascir_param::GetAscirNodeParams(clone.FindNode("node"));
  ASSERT_NE(cloned_params, nullptr);
  EXPECT_NE(cloned_params, params_);
  EXPECT_TRUE(cloned_params->cache_executions.empty());
  ASSERT_EQ(params_->cache_executions.size(), 1U);
  EXPECT_EQ(params_->cache_executions.front().mode, ascir_param::CacheExecutionMode::kNoGuard);
}

}  // namespace
}  // namespace att
