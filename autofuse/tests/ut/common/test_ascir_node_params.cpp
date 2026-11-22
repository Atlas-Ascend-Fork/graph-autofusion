/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir_node_param/ascir_param_builder.h"

#include <array>

#include <gtest/gtest.h>

#include "ascir_ops.h"
#include "tests/ut/common/ascir_reduce_test_helpers.h"

namespace {
ge::Expression Expr(int64_t value) {
  return ge::Symbol(value);
}

ge::Expression Align16(const ge::Expression &value) {
  return af::sym::Ceiling(value / Expr(16)) * Expr(16);
}

const ascir_param::ReduceNodeParams *GetReduceNodeParams(const af::AscNodePtr &node) {
  const auto params = ascir_param::GetAscirNodeParams(node);
  if (params == nullptr) {
    return nullptr;
  }
  return ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
}

const codegen::ReduceSpecificParams *GetReduceCodegenParams(const af::AscNodePtr &node) {
  const auto *reduce = GetReduceNodeParams(node);
  return reduce == nullptr ? nullptr : &reduce->canonical_params;
}

void ExpectEnrichSuccess(const af::AscGraph &graph) {
  EXPECT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
}

ascir_reduce_test_helpers::ReduceTestEnv MakeArReduceEnv(const char *api_name) {
  ascir_reduce_test_helpers::ReduceTestEnv env(api_name);
  env.SetIoAttrs({env.s1, Expr(1)}, {env.s0, Expr(1)}, {Expr(1), af::sym::kSymbolZero});
  return env;
}

TEST(AscirNodeParamsTest, EnrichGraphRegistersSkippedForUnsupportedApi) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Add add_op("add");
  graph.AddNode(add_op);
  auto node = graph.FindNode("add");
  ASSERT_NE(node, nullptr);

  ExpectEnrichSuccess(graph);
  auto params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->api_name, "Add");
  EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kSkipped);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(params->specific_params));
}

TEST(AscirNodeParamsTest, BroadcastCacheRecordsUseBoundDefinitionWithoutInferringAxes) {
  using Mode = ascir_param::CacheExecutionMode;
  af::AscGraph graph("bound_cache_guard");
  af::ascir_op::Add add("add");
  graph.AddNode(add);
  auto node = graph.FindNode("add");
  ASSERT_NE(node, nullptr);
  ascir::Axis loop;
  loop.id = 0;
  loop.name = "loop";
  loop.size = Expr(20);
  ascir::Axis decoy;
  decoy.id = 1;
  decoy.type = ascir::Axis::Type::kAxisTypeTileOuter;
  decoy.size = Expr(1116);
  ascir::Axis emitted_period;
  emitted_period.size = Expr(7);
  const ascir_param::CacheExecutionContext::AxisMap axes{{loop.id, &loop}, {decoy.id, &decoy}};
  ascir_param::CacheExecutionContext context{axes, 0U, {}, {}};
  for (const auto mode : {Mode::kOriginBroadcast, Mode::kFusedBroadcast}) {
    SCOPED_TRACE(static_cast<int>(mode));
    ascir_param::BindCacheCondition(context, mode, loop, emitted_period, Expr(23));
    ASSERT_EQ(ascir_param::CollectCacheExecution(context, node, {loop.id}, loop.id, mode, {}), af::SUCCESS);
    const auto params = ascir_param::GetAscirNodeParams(node);
    ASSERT_NE(params, nullptr);
    const auto &record = params->cache_executions.back();
    EXPECT_EQ(record.mode, mode);
    EXPECT_EQ(record.guard.guard_axis, af::Symbol("loop"));
    EXPECT_EQ(record.guard.loop_extent, Expr(20));
    if (mode == Mode::kFusedBroadcast) {
      EXPECT_EQ(record.guard.period, Expr(7));
      EXPECT_EQ(record.guard.block_inner_extent, Expr(23));
    }
    context.broadcast_bindings.erase(mode);
    ASSERT_EQ(ascir_param::CollectCacheExecution(context, node, {loop.id}, loop.id, mode, {}), af::SUCCESS);
    EXPECT_EQ(params->cache_executions.back().mode, Mode::kUnknown);
  }
}

TEST(AscirNodeParamsTest, VectorFuncParamsCanBeStoredInAscirNodeParams) {
  ascir_param::AscirNodeParams params;
  ascir_param::VectorFuncNodeParams vf_params;
  vf_params.is_double_loop = true;
  vf_params.all_strides = {Expr(4), Expr(8)};
  vf_params.output_dims = {Expr(2), Expr(16)};

  params.specific_params = vf_params;

  const auto *stored = ascir_param::GetSpecificParams<ascir_param::VectorFuncNodeParams>(params);
  ASSERT_NE(stored, nullptr);
  EXPECT_TRUE(stored->is_double_loop);
  ASSERT_EQ(stored->all_strides.size(), 2U);
  EXPECT_EQ(stored->all_strides[0], Expr(4));
  ASSERT_EQ(stored->output_dims.size(), 2U);
  EXPECT_EQ(stored->output_dims[1], Expr(16));
}

TEST(AscirNodeParamsTest, EnrichGraphRegistersVectorFuncParams) {
  af::AscGraph graph("test_graph");
  af::ascir_op::VectorFunc vf_op("vf");
  graph.AddNode(vf_op);
  auto node = graph.FindNode("vf");
  ASSERT_NE(node, nullptr);

  ExpectEnrichSuccess(graph);
  auto params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->api_name, "VectorFunc");
  EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kBuilt);
  const auto *stored = ascir_param::GetSpecificParams<ascir_param::VectorFuncNodeParams>(*params);
  ASSERT_NE(stored, nullptr);
  EXPECT_FALSE(stored->is_double_loop);
  EXPECT_TRUE(stored->all_strides.empty());
  EXPECT_TRUE(stored->output_dims.empty());
}

TEST(AscirNodeParamsTest, BroadcastParamsCanBeStoredInAscirNodeParams) {
  ascir_param::AscirNodeParams params;
  params.specific_params = ascir_param::BroadcastNodeParams{};

  const auto *stored = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(params);
  ASSERT_NE(stored, nullptr);
  EXPECT_FALSE(stored->valid);
}

TEST(AscirNodeParamsTest, EnrichGraphRegistersBroadcastParams) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  auto node = graph.FindNode("broadcast");
  ASSERT_NE(node, nullptr);

  ExpectEnrichSuccess(graph);
  auto params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->api_name, "Broadcast");
  EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kBuilt);
  const auto *stored = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(*params);
  ASSERT_NE(stored, nullptr);
  EXPECT_FALSE(stored->valid);
}

TEST(AscirNodeParamsTest, EnrichGraphPreservesBroadcastParamsAcrossRepeatedCalls) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  auto node = graph.FindNode("broadcast");
  ASSERT_NE(node, nullptr);

  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "Broadcast";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  ascir_param::BroadcastNodeParams broadcast;
  broadcast.valid = true;
  broadcast.is_scalar = true;
  broadcast.duplicate_count = Expr(8);
  broadcast.src_shape = {{Expr(1), ascir_param::ParamExprRole::kSemantic},
                         {Expr(8), ascir_param::ParamExprRole::kActualSize}};
  broadcast.dst_shape = {{Expr(4), ascir_param::ParamExprRole::kActualSize},
                         {Expr(8), ascir_param::ParamExprRole::kActualSize}};
  params->specific_params = broadcast;
  ASSERT_TRUE(node->GetOpDesc()->SetExtAttr("AscirNodeParams", params));

  ExpectEnrichSuccess(graph);
  ExpectEnrichSuccess(graph);
  params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  const auto *stored = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(*params);
  ASSERT_NE(stored, nullptr);
  EXPECT_TRUE(stored->valid);
  EXPECT_TRUE(stored->is_scalar);
  EXPECT_EQ(stored->duplicate_count, Expr(8));
  EXPECT_EQ(stored->src_shape[0].expr, Expr(1));
  EXPECT_EQ(stored->dst_shape[1].role, ascir_param::ParamExprRole::kActualSize);
}

TEST(AscirNodeParamsTest, EnrichGraphPreservesPrebuiltBroadcastParams) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Broadcast broadcast_op("broadcast");
  graph.AddNode(broadcast_op);
  auto node = graph.FindNode("broadcast");
  ASSERT_NE(node, nullptr);

  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "Broadcast";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  ascir_param::BroadcastNodeParams broadcast;
  broadcast.valid = true;
  broadcast.duplicate_count = Expr(16);
  broadcast.src_shape = {{Expr(1), ascir_param::ParamExprRole::kSemantic},
                         {Expr(16), ascir_param::ParamExprRole::kSize}};
  broadcast.dst_shape = {{Expr(2), ascir_param::ParamExprRole::kSize}, {Expr(16), ascir_param::ParamExprRole::kSize}};
  params->specific_params = broadcast;
  ASSERT_TRUE(node->GetOpDesc()->SetExtAttr("AscirNodeParams", params));

  ExpectEnrichSuccess(graph);
  params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  const auto *stored = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(*params);
  ASSERT_NE(stored, nullptr);
  EXPECT_TRUE(stored->valid);
  EXPECT_EQ(stored->duplicate_count, Expr(16));
  EXPECT_EQ(stored->src_shape[1].expr, Expr(16));
  EXPECT_EQ(stored->dst_shape[0].role, ascir_param::ParamExprRole::kSize);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsForArSingleReduce) {
  auto env = MakeArReduceEnv("max");

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_TRUE(reduce->valid);
  EXPECT_EQ(reduce->pattern, codegen::ReducePattern::kAR);
  EXPECT_EQ(reduce->merge_mode, codegen::ReduceMergeMode::kCopy);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsForRaSingleReduce) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  env.SetIoAttrs({env.s1, Expr(1)}, {Expr(1), env.s1}, {af::sym::kSymbolZero, Expr(1)});
  env.node->attr.sched.loop_axis = env.z0.id;

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(reduce->pattern, codegen::ReducePattern::kRA);
  EXPECT_EQ(reduce->merge_mode, codegen::ReduceMergeMode::kCopy);
  EXPECT_TRUE(reduce->merged_dims.valid);
  EXPECT_EQ(reduce->merged_dims.first, env.s0);
  EXPECT_EQ(reduce->merged_dims.last, Align16(env.s1));
}

TEST(AscirNodeParamsTest, EnrichReduceParamsForAllAxisReduce) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  env.SetIoAttrs({env.s1, Expr(1)}, {Expr(1), Expr(1)}, {af::sym::kSymbolZero, af::sym::kSymbolZero});

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(reduce->pattern, codegen::ReducePattern::kAR);
  EXPECT_TRUE(reduce->merged_dims.valid);
  EXPECT_EQ(reduce->merged_dims.first, Expr(1));
  EXPECT_EQ(reduce->merged_dims.last, Align16(env.s1) * env.s0);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsCountsRecursiveLoopAxis) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  auto &merged_axis =
      env.graph.CreateAxis("z0z1", af::Axis::kAxisTypeMerged, env.s0 * env.s1, {env.z0.id, env.z1.id}, af::kIdNone);
  env.SetIoAttrs({env.s1, Expr(1)}, {Expr(1), Expr(1)}, {af::sym::kSymbolZero, af::sym::kSymbolZero});
  env.node->attr.sched.axis = {merged_axis.id};
  env.node->attr.sched.loop_axis = merged_axis.id;

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(reduce->merge_mode, codegen::ReduceMergeMode::kCopy);
  EXPECT_EQ(reduce->merge_times, ge::Symbol((env.s0 * env.s1).Str().get()));
}

TEST(AscirNodeParamsTest, EnrichReduceParamsKeepsSemanticMergeSizeAndTimes) {
  const auto semantic_size = ge::Symbol("s0") * ge::Symbol("s1");
  ascir_reduce_test_helpers::ReduceTestEnv env("sum");
  env.node->GetOpDesc()->SetType("Sum");
  env.s1 = semantic_size;
  env.z1.size = semantic_size;
  env.graph.FindAxis(env.z1.id)->size = semantic_size;
  env.node->inputs[0].attr.repeats = {env.s0, semantic_size};
  env.node->outputs[0].attr.repeats = {env.s0, Expr(1)};
  env.SetIoAttrs({semantic_size, Expr(1)}, {env.s0, Expr(1)}, {Expr(1), af::sym::kSymbolZero});

  ExpectEnrichSuccess(env.graph);
  const auto *reduce_node = GetReduceNodeParams(env.node);
  ASSERT_NE(reduce_node, nullptr);
  EXPECT_EQ(reduce_node->canonical_params.merge_size, env.s0);
  EXPECT_EQ(reduce_node->canonical_params.merge_times, ge::Symbol(semantic_size.Str().get()));
  EXPECT_TRUE(reduce_node->exprs.merge_size.valid);
  EXPECT_TRUE(reduce_node->exprs.merge_times.valid);
  EXPECT_EQ(ascir_param::ResolveForAtt(reduce_node->exprs.merge_size), env.s0);
  EXPECT_EQ(ascir_param::ResolveForAtt(reduce_node->exprs.merge_times), semantic_size);
}

TEST(AscirNodeParamsTest, ValidateReduceNodeParamsRejectsMismatchedSemanticExpr) {
  ascir_param::ReduceNodeParams params;
  params.canonical_params.valid = true;
  params.canonical_params.merge_size = Expr(8);
  params.canonical_params.merge_times = Expr(2);
  params.exprs.merge_size = {true, {{Expr(4), ascir_param::ParamExprRole::kSemantic}}};
  params.exprs.merge_times = {true, {{Expr(2), ascir_param::ParamExprRole::kSemantic}}};

  EXPECT_NE(ascir_param::ValidateReduceNodeParams(params), af::SUCCESS);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsUsesFullAxisForMultiReduceCheck) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  auto &merged_axis =
      env.graph.CreateAxis("z0z1", af::Axis::kAxisTypeMerged, env.s0 * env.s1, {env.z0.id, env.z1.id}, af::kIdNone);
  env.node->attr.sched.axis = {merged_axis.id};
  env.node->attr.sched.loop_axis = merged_axis.id;
  env.node->inputs[0].attr.strides = {env.s1, Expr(1)};
  env.node->inputs[0].attr.vectorized_axis = {env.z1.id};
  env.node->inputs[0].attr.vectorized_strides = {Expr(1)};
  env.node->outputs[0].attr.repeats = {env.s0, Expr(1)};
  env.node->outputs[0].attr.strides = {Expr(1), af::sym::kSymbolZero};
  env.node->outputs[0].attr.vectorized_axis = {env.z1.id};
  env.node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolZero};

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(reduce->merge_mode, codegen::ReduceMergeMode::kNone);
  EXPECT_EQ(reduce->merge_times, Expr(1));
}

TEST(AscirNodeParamsTest, EnrichReduceParamsKeepsBlockOuterLoopAxis) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  auto split = env.graph.BlockSplit(env.z0.id);
  auto z0_block_outer = split.first;
  auto z0_block_inner = split.second;
  ASSERT_NE(z0_block_outer, nullptr);
  ASSERT_NE(z0_block_inner, nullptr);
  env.SetIoAttrs({env.s1, Expr(1)}, {env.s0, Expr(1)}, {Expr(1), af::sym::kSymbolZero});
  env.graph.ApplySplit(env.node, z0_block_outer->id, z0_block_inner->id);
  env.node->attr.sched.loop_axis = z0_block_outer->id;
  env.node->inputs[0].attr.axis = {z0_block_outer->id, z0_block_inner->id, env.z1.id};
  env.node->inputs[0].attr.repeats = {env.s0 / z0_block_inner->size, z0_block_inner->size, env.s1};
  env.node->inputs[0].attr.strides = {env.s1 * z0_block_inner->size, env.s1, Expr(1)};
  env.node->inputs[0].attr.vectorized_axis = {z0_block_inner->id, env.z1.id};
  env.node->inputs[0].attr.vectorized_strides = {env.s1, Expr(1)};
  env.node->outputs[0].attr.vectorized_axis = {z0_block_inner->id, env.z1.id};
  env.node->outputs[0].attr.vectorized_strides = {Expr(1), af::sym::kSymbolZero};

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_EQ(reduce->merge_mode, codegen::ReduceMergeMode::kNone);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsFillsReuseSourceFromNode) {
  auto env = MakeArReduceEnv("max");

  ExpectEnrichSuccess(env.graph);
  const auto *reduce = GetReduceCodegenParams(env.node);
  ASSERT_NE(reduce, nullptr);
  EXPECT_TRUE(reduce->reuse.valid);
  EXPECT_TRUE(reduce->reuse.is_reuse_source);
}

TEST(AscirNodeParamsTest, EnrichReduceParamsForDirectV35ReduceApis) {
  constexpr std::array<const char *, 3U> kApis = {"sum", "mean", "prod"};
  constexpr std::array<const char *, 3U> kTypes = {"Sum", "Mean", "Prod"};
  for (size_t i = 0U; i < kApis.size(); ++i) {
    auto env = MakeArReduceEnv(kApis[i]);
    env.node->GetOpDesc()->SetType(kTypes[i]);

    ExpectEnrichSuccess(env.graph);
    const auto params = ascir_param::GetAscirNodeParams(env.node);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->status, ascir_param::ParamBuildStatus::kBuilt);
    const auto *reduce = GetReduceCodegenParams(env.node);
    ASSERT_NE(reduce, nullptr);
    EXPECT_TRUE(reduce->valid);
    EXPECT_EQ(reduce->reduce_type, kTypes[i]);
  }
}

TEST(AscirNodeParamsTest, EnrichReduceParamsRejectsMissingVectorizedStrides) {
  ascir_reduce_test_helpers::ReduceTestEnv env("max");
  env.node->inputs[0].attr.strides = {env.s1, Expr(1)};
  env.node->inputs[0].attr.vectorized_axis = {env.z0.id, env.z1.id};
  env.node->inputs[0].attr.vectorized_strides.clear();
  env.node->outputs[0].attr.repeats = {env.s0, Expr(1)};
  env.node->outputs[0].attr.strides = {Expr(1), af::sym::kSymbolZero};
  env.node->outputs[0].attr.vectorized_axis = {env.z0.id, env.z1.id};
  env.node->outputs[0].attr.vectorized_strides.clear();

  EXPECT_NE(ascir_param::EnrichAscirGraphNodeParams(env.graph), af::SUCCESS);
}
}  // namespace
