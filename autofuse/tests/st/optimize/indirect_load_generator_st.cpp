/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "optimize.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

namespace {
struct NodeView {
  std::vector<af::AxisId> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
};

af::AscNodePtr AddDataNode(af::AscGraph &graph, const char *name, af::DataType dtype, int64_t index,
                           const NodeView &view) {
  af::ascir_op::Data data(name);
  graph.AddNode(data);
  data.ir_attr.SetIndex(index);
  data.y.dtype = dtype;
  data.attr.sched.axis = view.axis;
  *data.y.axis = view.axis;
  *data.y.repeats = view.repeats;
  *data.y.strides = view.strides;
  return graph.FindNode(name);
}

af::AscNodePtr AddLoadNode(af::AscGraph &graph, const char *name, const af::AscNodePtr &input, const NodeView &view,
                           af::DataType dtype) {
  af::ascir_op::Load load(name);
  graph.AddNode(load);
  load.attr.sched.axis = view.axis;
  load.y.dtype = dtype;
  *load.y.axis = view.axis;
  *load.y.repeats = view.repeats;
  *load.y.strides = view.strides;
  const auto node = graph.FindNode(name);
  EXPECT_EQ(af::GraphUtils::AddEdge(input->GetOutDataAnchor(0), node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
  return node;
}

template <typename Op>
af::AscNodePtr AddUnaryNode(af::AscGraph &graph, const char *name, const af::AscNodePtr &input, const NodeView &view) {
  Op op(name);
  graph.AddNode(op);
  op.attr.sched.axis = view.axis;
  op.y.dtype = af::DT_FLOAT;
  *op.y.axis = view.axis;
  *op.y.repeats = view.repeats;
  *op.y.strides = view.strides;
  const auto node = graph.FindNode(name);
  EXPECT_EQ(af::GraphUtils::AddEdge(input->GetOutDataAnchor(0), node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
  return node;
}

template <typename Op>
af::AscNodePtr AddBinaryNode(af::AscGraph &graph, const char *name, const af::AscNodePtr &lhs,
                             const af::AscNodePtr &rhs, const NodeView &view) {
  Op op(name);
  graph.AddNode(op);
  op.attr.sched.axis = view.axis;
  op.y.dtype = af::DT_FLOAT;
  *op.y.axis = view.axis;
  *op.y.repeats = view.repeats;
  *op.y.strides = view.strides;
  const auto node = graph.FindNode(name);
  EXPECT_EQ(af::GraphUtils::AddEdge(lhs->GetOutDataAnchor(0), node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
  EXPECT_EQ(af::GraphUtils::AddEdge(rhs->GetOutDataAnchor(0), node->GetInDataAnchor(1)), af::GRAPH_SUCCESS);
  return node;
}

af::AscNodePtr AddIndirectLoadNode(af::AscGraph &graph, const char *name, const af::AscNodePtr &input,
                                   const af::AscNodePtr &index, int64_t axis, const NodeView &view) {
  af::ascir_op::IndirectLoad il(name);
  graph.AddNode(il);
  il.ir_attr.SetAxis(axis);
  il.attr.sched.axis = view.axis;
  il.y.dtype = af::DT_FLOAT;
  *il.y.axis = view.axis;
  *il.y.repeats = view.repeats;
  *il.y.strides = view.strides;
  const auto node = graph.FindNode(name);
  EXPECT_EQ(af::GraphUtils::AddEdge(input->GetOutDataAnchor(0), node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
  EXPECT_EQ(af::GraphUtils::AddEdge(index->GetOutDataAnchor(0), node->GetInDataAnchor(1)), af::GRAPH_SUCCESS);
  return node;
}

void AddStoreOutput(af::AscGraph &graph, const af::AscNodePtr &input, const NodeView &view) {
  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.attr.sched.axis = view.axis;
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = view.axis;
  *store.y.repeats = view.repeats;
  *store.y.strides = view.strides;
  const auto store_node = graph.FindNode("store");
  EXPECT_EQ(af::GraphUtils::AddEdge(input->GetOutDataAnchor(0), store_node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
  af::ascir_op::Output out("out");
  graph.AddNode(out);
  out.ir_attr.SetIndex(0);
  out.y.dtype = af::DT_FLOAT;
  const auto out_node = graph.FindNode("out");
  EXPECT_EQ(af::GraphUtils::AddEdge(store_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0)), af::GRAPH_SUCCESS);
}

enum class GroupVariant { kValid, kAxisMismatch, kIndexMismatch };

af::AscGraph BuildMultiIlGroupGraph(GroupVariant variant = GroupVariant::kValid) {
  af::AscGraph graph("multi_il_group");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  const NodeView w_view{{z0.id, z3.id, z1.id}, {s0, s3, s1}, {s3 * s1, s1, af::ops::One}};
  const NodeView io_view{{z2.id, z3.id, z1.id}, {s2, s3, s1}, {s3 * s1, s1, af::ops::One}};
  const NodeView idx2_view =
      variant == GroupVariant::kIndexMismatch
          ? NodeView{{z2.id, z3.id, z1.id}, {s2, af::ops::One, s1}, {s1, af::ops::One, af::ops::One}}
          : io_view;
  auto w = AddDataNode(graph, "w", af::DT_FLOAT, 0, w_view);
  auto idx1 = AddDataNode(graph, "idx1", af::DT_INT32, 1, io_view);
  auto idx2 = AddDataNode(graph, "idx2", af::DT_INT32, 2, idx2_view);
  auto w_ld = AddLoadNode(graph, "w_ld", w, w_view, af::DT_FLOAT);
  auto idx1_ld = AddLoadNode(graph, "idx1_ld", idx1, io_view, af::DT_INT32);
  auto idx2_ld = AddLoadNode(graph, "idx2_ld", idx2, idx2_view, af::DT_INT32);
  auto il1 = AddIndirectLoadNode(graph, "il1", w_ld, idx1_ld, 0, io_view);
  auto il2 = AddIndirectLoadNode(graph, "il2", w_ld, idx2_ld, variant == GroupVariant::kAxisMismatch ? 1 : 0, io_view);
  auto abs1 = AddUnaryNode<af::ascir_op::Abs>(graph, "abs1", il1, io_view);
  auto abs2 = AddUnaryNode<af::ascir_op::Abs>(graph, "abs2", il2, io_view);
  auto add = AddBinaryNode<af::ascir_op::Add>(graph, "add", abs1, abs2, io_view);
  AddStoreOutput(graph, add, io_view);
  return graph;
}

af::AscGraph BuildIlNormChainGraph() {
  af::AscGraph graph("il_norm_chain");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  const NodeView w_view{{z0.id, z3.id, z1.id}, {s0, s3, s1}, {s3 * s1, s1, af::ops::One}};
  const NodeView io_view{{z2.id, z3.id, z1.id}, {s2, s3, s1}, {s3 * s1, s1, af::ops::One}};
  const NodeView red_view{{z2.id, z3.id, z1.id}, {af::ops::One, s3, s1}, {af::ops::Zero, s1, af::ops::One}};
  auto w = AddDataNode(graph, "w", af::DT_FLOAT, 0, w_view);
  auto idx = AddDataNode(graph, "idx", af::DT_INT32, 1, io_view);
  auto w_ld = AddLoadNode(graph, "w_ld", w, w_view, af::DT_FLOAT);
  auto idx_ld = AddLoadNode(graph, "idx_ld", idx, io_view, af::DT_INT32);
  auto il = AddIndirectLoadNode(graph, "il", w_ld, idx_ld, 0, io_view);
  auto sum = AddUnaryNode<af::ascir_op::Sum>(graph, "sum", il, red_view);
  auto brc = AddUnaryNode<af::ascir_op::Broadcast>(graph, "brc", sum, io_view);
  auto sub = AddBinaryNode<af::ascir_op::Sub>(graph, "sub", il, brc, io_view);
  AddStoreOutput(graph, sub, io_view);
  return graph;
}

// SK R-before-gather：reduce 轴(0) 早于 gather 轴(1)，SIMD/SIMT 被跳过，只剩 SK。
af::AscGraph BuildSkReduceBeforeGatherGraph() {
  af::AscGraph graph("sk_reduce_before_gather");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto x0 = graph.CreateAxis("x0", s0);
  auto x1 = graph.CreateAxis("x1", s1);
  auto x2 = graph.CreateAxis("x2", s2);
  auto x3 = graph.CreateAxis("x3", s3);
  auto y0 = graph.CreateAxis("y0", s0);
  auto y1 = graph.CreateAxis("y1", s1);
  auto y2 = graph.CreateAxis("y2", s2);
  auto y3 = graph.CreateAxis("y3", s3);
  const NodeView in_view{{x0.id, x1.id, x2.id, x3.id}, {s0, s1, s2, s3}, {s1 * s2 * s3, s2 * s3, s3, af::ops::One}};
  const NodeView out_view{{y0.id, y1.id, y2.id, y3.id}, {s0, s1, s2, s3}, {s1 * s2 * s3, s2 * s3, s3, af::ops::One}};
  const NodeView red_view{{y0.id, y1.id, y2.id, y3.id},
                          {af::ops::One, af::ops::One, af::ops::One, af::ops::One},
                          {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero}};
  auto x = AddDataNode(graph, "x", af::DT_FLOAT, 0, in_view);
  auto idx = AddDataNode(graph, "idx", af::DT_INT32, 1, out_view);
  auto x_ld = AddLoadNode(graph, "x_ld", x, in_view, af::DT_FLOAT);
  auto idx_ld = AddLoadNode(graph, "idx_ld", idx, out_view, af::DT_INT32);
  auto il = AddIndirectLoadNode(graph, "il", x_ld, idx_ld, 1, out_view);
  auto sum = AddUnaryNode<af::ascir_op::Sum>(graph, "sum", il, red_view);
  AddStoreOutput(graph, sum, red_view);
  return graph;
}
}  // namespace

// 多 gather 组候选路径覆盖：2 路共享 weight、独立 index 的 IndirectLoad 经 Add 汇聚。
TEST(IndirectLoadGeneratorSt, MultiIndirectLoadGroupOptimize) {
  auto graph = BuildMultiIlGroupGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  optimize::OptimizerOptions options{};
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(tasks.empty());
}

// IL-Norm 链覆盖：IndirectLoad → Sum → Broadcast → Sub → Store。
TEST(IndirectLoadGeneratorSt, IndirectLoadNormChainOptimize) {
  auto graph = BuildIlNormChainGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  optimize::OptimizerOptions options{};
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(tasks.empty());
}

// SK R-before-gather 路径：reduce 轴早于 gather 轴，候选只剩 SK。
TEST(IndirectLoadGeneratorSt, SkReduceBeforeGatherOptimize) {
  auto graph = BuildSkReduceBeforeGatherGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  optimize::OptimizerOptions options{};
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(tasks.empty());
}

// 组非法：两路 gather 轴不一致，必须拒绝组候选。
TEST(IndirectLoadGeneratorSt, RejectsGroupAxisMismatch) {
  auto graph = BuildMultiIlGroupGraph(GroupVariant::kAxisMismatch);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  optimize::OptimizerOptions options{};
  EXPECT_NE(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
}

// 组非法：两路 index shape 不一致，必须拒绝组候选。
TEST(IndirectLoadGeneratorSt, RejectsGroupIndexShapeMismatch) {
  auto graph = BuildMultiIlGroupGraph(GroupVariant::kIndexMismatch);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  optimize::OptimizerOptions options{};
  EXPECT_NE(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
}
