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
#include "common/ub_expr/asc_graph_ub_expr_builder.h"
#include "common/ub_expr/ub_expr_utils.h"
#include "graph/utils/graph_utils.h"
#include "schedule_result.h"

namespace {
struct UbQueueAttrConfig {
  ge::DataType dtype = ge::DT_FLOAT16;
  std::vector<af::Expression> repeats;
  std::vector<int64_t> vectorized_axis;
  std::vector<af::Expression> vectorized_strides;
  af::Position position = af::Position::kPositionVecIn;
  int64_t que_id = 0;
  int64_t buf_num = 1;
};

af::AscNodePtr BuildLoadNode(af::AscGraph &graph, const std::string &name) {
  const auto data_name = name + "_data";
  af::ascir_op::Data data_op(data_name.c_str(), graph);
  af::ascir_op::Load load_op(name.c_str());
  graph.AddNode(load_op);
  load_op.x = data_op.y;
  return graph.FindNode(name.c_str());
}

void SetUbQueueOutputAttr(af::AscTensorAttr &attr, const UbQueueAttrConfig &config) {
  attr.dtype = config.dtype;
  attr.repeats = config.repeats;
  attr.vectorized_axis = config.vectorized_axis;
  attr.vectorized_strides = config.vectorized_strides;
  attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  attr.mem.position = config.position;
  attr.que.id = config.que_id;
  attr.que.buf_num = config.buf_num;
}

af::Expression ReplaceContainers(const ascir::UbExprContext &context) {
  std::vector<std::pair<af::Expression, af::Expression>> replacements;
  for (const auto &container_expr : context.container_expr) {
    replacements.emplace_back(container_expr.first, container_expr.second);
  }
  return context.ub_expr.Replace(replacements).Simplify();
}
}  // namespace

TEST(UbExprUtilsTest, BuildUbExprReturnsFalseForInvalidExpr) {
  ascir::UbExprContext context;

  const auto result = ascir::UbExprUtils::BuildUbExpr(context);

  EXPECT_FALSE(result.has_ub_expr);
}

TEST(UbExprUtilsTest, BuildUbExprReturnsOriginExpr) {
  ascir::UbExprContext context;
  context.ub_expr = af::sym::Add(af::Symbol("s0"), af::Symbol(32));

  const auto result = ascir::UbExprUtils::BuildUbExpr(context);

  EXPECT_TRUE(result.has_ub_expr);
  EXPECT_EQ(result.ub_expr, context.ub_expr);
  EXPECT_EQ(result.origin_expr, context.ub_expr.Str().get());
}

TEST(AscGraphUbExprBuilderTest, BuildReturnsFalseForGraphWithoutUbAllocation) {
  af::AscGraph graph("no_ub_alloc");
  ascir::UbExprContext context;

  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  EXPECT_FALSE(ascir::UbExprUtils::BuildUbExpr(context).has_ub_expr);
}

TEST(AscGraphUbExprBuilderTest, BuildAggregatesQueueBufferAndTmpBuffer) {
  af::AscGraph graph("ub_alloc");
  auto &axis = graph.CreateAxis("s0", af::Symbol("s0"));
  auto &tile_axis = graph.CreateAxis("t0", af::Axis::kAxisTypeTileInner, af::Symbol("t0"), {axis.id}, af::kIdNone);
  auto node = BuildLoadNode(graph, "load");
  ASSERT_NE(node, nullptr);

  auto &output_attr = node->outputs[0].attr;
  output_attr.dtype = ge::DT_FLOAT16;
  output_attr.repeats = {tile_axis.size};
  output_attr.vectorized_axis = {tile_axis.id};
  output_attr.vectorized_strides = {af::Symbol(1)};
  output_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  output_attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  output_attr.mem.position = af::Position::kPositionVecIn;
  output_attr.mem.reuse_id = 0;
  output_attr.que.id = 0;
  output_attr.que.buf_num = 2;

  af::TmpBuffer tmp_buffer;
  tmp_buffer.id = 1;
  tmp_buffer.buf_desc.size = af::Symbol(64);
  tmp_buffer.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tmp_buffer.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->attr.tmp_buffers.emplace_back(tmp_buffer);

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);
  const auto result = ascir::UbExprUtils::BuildUbExpr(context);

  EXPECT_TRUE(result.has_ub_expr);
  EXPECT_NE(result.origin_expr.find("q0_size"), std::string::npos);
  EXPECT_NE(result.origin_expr.find("b1_size"), std::string::npos);
  EXPECT_FALSE(context.container_expr.empty());
  EXPECT_FALSE(context.ub_related_vars.empty());
  EXPECT_EQ(context.graph_name, "ub_alloc");
}

TEST(AscGraphUbExprBuilderTest, BuildStripsV1RepeatCapForRawTmpSize) {
  af::AscGraph graph("raw_tmp");
  auto node = BuildLoadNode(graph, "load");
  ASSERT_NE(node, nullptr);

  const auto raw_tmp_size = af::Symbol("raw_tmp_size");
  af::TmpBuffer tmp_buffer;
  tmp_buffer.id = 1;
  tmp_buffer.buf_desc.size = af::sym::Min(raw_tmp_size, af::Symbol(255 * 256 + 32));
  tmp_buffer.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tmp_buffer.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->attr.tmp_buffers.emplace_back(tmp_buffer);

  ascir::UbExprContext capped_context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, capped_context), af::SUCCESS);
  auto capped_expr = ReplaceContainers(capped_context).Replace({{raw_tmp_size, af::Symbol(80000)}}).Simplify();
  int64_t capped_usage = 0;
  EXPECT_TRUE(capped_expr.GetConstValue(capped_usage)) << capped_expr.Str().get();
  EXPECT_EQ(capped_usage, 65312);

  ascir::UbExprContext raw_context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, raw_context, true), af::SUCCESS);
  auto raw_expr = ReplaceContainers(raw_context).Replace({{raw_tmp_size, af::Symbol(80000)}}).Simplify();
  int64_t raw_usage = 0;
  EXPECT_TRUE(raw_expr.GetConstValue(raw_usage)) << raw_expr.Str().get();
  EXPECT_EQ(raw_usage, 80000);
}

TEST(AscGraphUbExprBuilderTest, BuildAlignsTensorBytesAfterDtypeSize) {
  af::AscGraph graph("fp16_align");
  auto &axis = graph.CreateAxis("s0", af::Symbol(33));
  auto node = BuildLoadNode(graph, "load");
  ASSERT_NE(node, nullptr);

  auto &output_attr = node->outputs[0].attr;
  SetUbQueueOutputAttr(output_attr, {ge::DT_FLOAT16, {axis.size}, {axis.id}, {af::Symbol(1)}});

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);
  ASSERT_EQ(context.container_expr.size(), 1U);
  auto min_expr = ReplaceContainers(context);

  int64_t min_ub_usage = 0;
  EXPECT_TRUE(min_expr.GetConstValue(min_ub_usage)) << min_expr.Str().get();
  EXPECT_EQ(min_ub_usage, 96);
}

TEST(AscGraphUbExprBuilderTest, BuildUsesRepeatStrideForVectorizedTensorSize) {
  af::AscGraph graph("repeat_stride_size");
  auto &outer_axis = graph.CreateAxis("z0t", af::Symbol(100));
  auto &inner_axis = graph.CreateAxis("z1", af::Symbol(8));
  auto node = BuildLoadNode(graph, "load");
  ASSERT_NE(node, nullptr);

  auto &output_attr = node->outputs[0].attr;
  SetUbQueueOutputAttr(output_attr, {ge::DT_INT32,
                                     {outer_axis.size, af::Symbol(16)},
                                     {outer_axis.id, inner_axis.id},
                                     {af::Symbol(16), af::Symbol(1)},
                                     af::Position::kPositionVecOut,
                                     1});
  output_attr.axis = {outer_axis.id, inner_axis.id};

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  const auto q1_iter = context.container_expr.find(af::Symbol("q1_size"));
  ASSERT_NE(q1_iter, context.container_expr.end());
  int64_t q1_size = 0;
  EXPECT_TRUE(q1_iter->second.GetConstValue(q1_size)) << q1_iter->second.Str().get();
  EXPECT_EQ(q1_size, 6400);
}

TEST(AscGraphUbExprBuilderTest, BuildKeepsQueueContainerExprAsSingleBufferSlot) {
  af::AscGraph graph("queue_buf_num");
  auto &axis = graph.CreateAxis("s0", af::Symbol(16));
  auto node = BuildLoadNode(graph, "load");
  ASSERT_NE(node, nullptr);

  auto &output_attr = node->outputs[0].attr;
  SetUbQueueOutputAttr(output_attr,
                       {ge::DT_FLOAT16, {axis.size}, {axis.id}, {af::Symbol(1)}, af::Position::kPositionVecIn, 0, 2});

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  const auto q0_iter = context.container_expr.find(af::Symbol("q0_size"));
  ASSERT_NE(q0_iter, context.container_expr.end());
  int64_t q0_size = 0;
  EXPECT_TRUE(q0_iter->second.GetConstValue(q0_size)) << q0_iter->second.Str().get();
  EXPECT_EQ(q0_size, 32);

  int64_t ub_usage = 0;
  auto min_expr = ReplaceContainers(context);
  EXPECT_TRUE(min_expr.GetConstValue(ub_usage)) << min_expr.Str().get();
  EXPECT_EQ(ub_usage, 64);
}

TEST(AscGraphUbExprBuilderTest, BuildKeepsUserDefinedContainerNames) {
  af::AscGraph graph("named_container");
  auto &axis = graph.CreateAxis("s0", af::Symbol(16));
  auto queue_node = BuildLoadNode(graph, "queue_load");
  auto buffer_node = BuildLoadNode(graph, "buffer_load");
  ASSERT_NE(queue_node, nullptr);
  ASSERT_NE(buffer_node, nullptr);

  auto &queue_attr = queue_node->outputs[0].attr;
  queue_attr.dtype = ge::DT_FLOAT16;
  queue_attr.repeats = {axis.size};
  queue_attr.vectorized_axis = {axis.id};
  queue_attr.vectorized_strides = {af::Symbol(1)};
  queue_attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  queue_attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  queue_attr.que.id = 0;
  queue_attr.que.name = "custom_queue_size";
  queue_attr.que.buf_num = 2;

  auto &buffer_attr = buffer_node->outputs[0].attr;
  buffer_attr.dtype = ge::DT_FLOAT16;
  buffer_attr.repeats = {axis.size};
  buffer_attr.vectorized_axis = {axis.id};
  buffer_attr.vectorized_strides = {af::Symbol(1)};
  buffer_attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  buffer_attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  buffer_attr.buf.id = 1;
  buffer_attr.buf.name = "custom_buffer_size";

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  EXPECT_NE(context.container_expr.find(af::Symbol("custom_queue_size")), context.container_expr.end());
  EXPECT_NE(context.container_expr.find(af::Symbol("custom_buffer_size")), context.container_expr.end());
  EXPECT_EQ(context.container_expr.find(af::Symbol("q0_size")), context.container_expr.end());
  EXPECT_EQ(context.container_expr.find(af::Symbol("b1_size")), context.container_expr.end());
}

TEST(AscGraphUbExprBuilderTest, BuildSumsCoexistBufferTensorsWithSameReuseId) {
  af::AscGraph graph("buffer_reuse_group");
  auto &axis = graph.CreateAxis("s0", af::Symbol(16));
  auto node0 = BuildLoadNode(graph, "load0");
  auto node1 = BuildLoadNode(graph, "load1");
  ASSERT_NE(node0, nullptr);
  ASSERT_NE(node1, nullptr);

  auto set_buffer_attr = [&axis](af::AscTensorAttr &attr) {
    attr.dtype = ge::DT_FLOAT16;
    attr.repeats = {axis.size};
    attr.vectorized_axis = {axis.id};
    attr.vectorized_strides = {af::Symbol(1)};
    attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    attr.mem.hardware = af::MemHardware::kMemHardwareUB;
    attr.mem.position = af::Position::kPositionVecIn;
    attr.mem.reuse_id = 3;
    attr.buf.id = 2;
  };
  set_buffer_attr(node0->outputs[0].attr);
  set_buffer_attr(node1->outputs[0].attr);

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  const auto b2_iter = context.container_expr.find(af::Symbol("b2_size"));
  ASSERT_NE(b2_iter, context.container_expr.end());
  int64_t b2_size = 0;
  EXPECT_TRUE(b2_iter->second.GetConstValue(b2_size)) << b2_iter->second.Str().get();
  EXPECT_EQ(b2_size, 64);
}

TEST(AscGraphUbExprBuilderTest, BuildIncludesBuiltinTmpBuffer) {
  af::AscGraph graph("builtin_tmp_buffer");
  auto &axis = graph.CreateAxis("s0", af::Symbol(1));
  auto load_node = BuildLoadNode(graph, "load");
  ASSERT_NE(load_node, nullptr);
  auto &load_attr = load_node->outputs[0].attr;
  load_attr.dtype = ge::DT_BOOL;
  load_attr.repeats = {axis.size};
  load_attr.vectorized_axis = {axis.id};
  load_attr.vectorized_strides = {af::Symbol(1)};

  af::ascir_op::LogicalNot logical_not_op("logical_not");
  graph.AddNode(logical_not_op);
  auto logical_not_node = graph.FindNode("logical_not");
  ASSERT_NE(logical_not_node, nullptr);
  ASSERT_EQ(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), logical_not_node->GetInDataAnchor(0)),
            ge::GRAPH_SUCCESS);
  auto &logical_not_attr = logical_not_node->outputs[0].attr;
  logical_not_attr.dtype = ge::DT_BOOL;
  logical_not_attr.repeats = {axis.size};
  logical_not_attr.vectorized_axis = {axis.id};
  logical_not_attr.vectorized_strides = {af::Symbol(1)};

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  int64_t min_ub_usage = 0;
  EXPECT_TRUE(context.ub_expr.Simplify().GetConstValue(min_ub_usage)) << context.ub_expr.Str().get();
  EXPECT_EQ(min_ub_usage, 32);
}

TEST(AscGraphUbExprBuilderTest, BuildSumsTmpBuffersWithSameIdInOneNode) {
  af::AscGraph graph("same_id_tmp_buffer");
  af::Operator op("Compute", "Compute");
  auto node = graph.AddNode(op);
  ASSERT_NE(node, nullptr);
  node->attr.tmp_buffers.emplace_back(af::TmpBuffer{{af::Symbol(4096), -1}, af::MemAttr(), 0});
  node->attr.tmp_buffers.emplace_back(af::TmpBuffer{{af::Symbol(8192), -1}, af::MemAttr(), 0});

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  int64_t min_ub_usage = 0;
  auto min_expr = ReplaceContainers(context);
  EXPECT_TRUE(min_expr.GetConstValue(min_ub_usage)) << min_expr.Str().get();
  EXPECT_EQ(min_ub_usage, 12288);
}

TEST(AscGraphUbExprBuilderTest, BuildIgnoresInvalidTmpBufferSize) {
  af::AscGraph graph("invalid_tmp_buffer_size");
  af::Operator op("Compute", "Compute");
  auto node = graph.AddNode(op);
  ASSERT_NE(node, nullptr);
  af::TmpBuffer tmp_buffer;
  tmp_buffer.id = 99;
  node->attr.tmp_buffers.emplace_back(tmp_buffer);

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  EXPECT_EQ(context.container_expr.find(af::Symbol("b99_size")), context.container_expr.end());
  EXPECT_EQ(ascir::UbExprUtils::BuildUbExpr(context).origin_expr.find("Max(,"), std::string::npos);
}

TEST(AscGraphUbExprBuilderTest, BuildIncludesReservedUbForGather) {
  af::AscGraph graph("reserved_ub");
  af::Operator op("Gather", "Gather");
  auto node = graph.AddNode(op);
  ASSERT_NE(node, nullptr);
  // dcache 标记的生产路径（CompleteApiInfo 询问 impl）由 optimize UT 覆盖；
  // test_common 未链接 ascir_builtin_ops 注册库，本用例直接打标记验证 builder 的预留计算
  ASSERT_EQ(::ascir::SetDcacheSize(node, 40 * 1024), af::SUCCESS);

  ascir::UbExprContext context;
  EXPECT_EQ(ascir::AscGraphUbExprBuilder().Build(graph, context), af::SUCCESS);

  int64_t min_ub_usage = 0;
  EXPECT_TRUE(context.ub_expr.Simplify().GetConstValue(min_ub_usage)) << context.ub_expr.Str().get();
  EXPECT_EQ(min_ub_usage, 49152);
}
