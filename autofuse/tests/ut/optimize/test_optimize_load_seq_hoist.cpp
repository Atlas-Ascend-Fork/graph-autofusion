/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// “外提 Load 前置”不变式与 LoadOpSeqAdjust 的交互：
// HoistInvariantLoads 把 brc 缓存 Load 外提到 BlockOuter 后由 SortHoistedLoadsFirst 保证前置；
// 但 LoadOpSeqAdjust 会改写 node id 并 ReorderByNodeId，若其后不恢复该不变式，被前移的普通 Load
// 会插到外提 Load 之前，codegen 构建循环树时在同一层重复下降并拆出同轴循环，导致 kernel 编译失败。

#include <algorithm>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#define private public
#include "optimize.h"
#undef private
#include "schedule_utils.h"

namespace optimize {
namespace {

// 切分后的调度轴：schedule_axis 为完整切分链，loop_axis 为循环内轴（不含 BlockOuter）
struct SplitAxes {
  std::vector<int64_t> schedule_axis;
  std::vector<int64_t> loop_axis;
  int64_t tile_axis_id = af::kIdNone;
};

// data -> load 片段：一维向量化使 GM offset 与循环轴无关，满足地址循环恒定，具备外提资格
// 注：ascir_op 生成的 op 类删除了移动构造，不能按值返回，经引用输出参数传出
void AddLoadChain(af::AscGraph &graph, const SplitAxes &axes, const std::string &load_name, int64_t data_index,
                  const af::Expression &repeats, af::ascir_op::Load &load) {
  af::ascir_op::Data data(("data_" + load_name).c_str(), graph);
  data.ir_attr.SetIndex(data_index);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  load.x = data.y;
  load.y.dtype = af::DT_FLOAT16;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.api.type = af::ApiType::kAPITypeCompute;
  load.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load.attr.sched.axis = axes.schedule_axis;
  load.attr.sched.loop_axis = axes.tile_axis_id;
  *load.y.axis = axes.loop_axis;
  *load.y.repeats = {repeats};
  *load.y.strides = {af::ops::One};
  *load.y.vectorized_axis = {axes.tile_axis_id};
  *load.y.vectorized_strides = {af::ops::One};
}

template <typename OpType>
void SetComputeAttr(OpType &op, const SplitAxes &axes, const af::Expression &s1, const af::ComputeType compute_type,
                    const af::ComputeUnit unit) {
  op.attr.api.compute_type = compute_type;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
  op.attr.api.unit = unit;
  op.attr.sched.axis = axes.schedule_axis;
  op.attr.sched.loop_axis = axes.tile_axis_id;
  op.y.dtype = af::DT_FLOAT16;
  *op.y.axis = axes.loop_axis;
  *op.y.repeats = {s1};
  *op.y.strides = {af::ops::One};
}

// 已调度形态的 brc 缓存 Load 图（与 ..._inline_S*G*C* 候选同构）：
//   data_load_hoist -> load_hoist \
//                                   add -> mul -> store -> output
//   data_load_main  -> load_main -> broadcast /
//   data_load_tail  -> load_tail -------------/
// BlockOuter 轴只出现在 sched.axis 中，loop_axis 取循环内 tile 轴（外提前形态）。
af::AscGraph MakeHoistedLoadOrderGraph(const std::string &name) {
  af::AscGraph graph(name.c_str());
  const auto s0 = graph.CreateSizeVar(8);
  const auto s1 = graph.CreateSizeVar(1024);
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z0_block = graph.BlockSplit(z0.id).first;
  const auto z1_block = graph.BlockSplit(z1.id).first;
  const auto z1_tile = graph.TileSplit(z1.id).second;
  const SplitAxes axes{{z0_block->id, z1_block->id, z1_tile->id}, {z1_block->id, z1_tile->id}, z1_tile->id};

  af::ascir_op::Load load_hoist("load_hoist");
  AddLoadChain(graph, axes, "load_hoist", 0, af::ops::One, load_hoist);
  af::ascir_op::Load load_main("load_main");
  AddLoadChain(graph, axes, "load_main", 1, s1, load_main);
  af::ascir_op::Load load_tail("load_tail");
  AddLoadChain(graph, axes, "load_tail", 2, s1, load_tail);

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = load_main.y;
  SetComputeAttr(broadcast, axes, s1, af::ComputeType::kComputeBroadcast, af::ComputeUnit::kUnitVector);

  af::ascir_op::Add add("add");
  add.x1 = load_hoist.y;
  add.x2 = broadcast.y;
  SetComputeAttr(add, axes, s1, af::ComputeType::kComputeElewise, af::ComputeUnit::kUnitVector);

  af::ascir_op::Mul mul("mul");
  mul.x1 = load_tail.y;
  mul.x2 = add.y;
  SetComputeAttr(mul, axes, s1, af::ComputeType::kComputeElewise, af::ComputeUnit::kUnitVector);

  af::ascir_op::Store store("store");
  store.x = mul.y;
  SetComputeAttr(store, axes, s1, af::ComputeType::kComputeStore, af::ComputeUnit::kUnitMTE3);

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = af::DT_FLOAT16;
  output.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  output.ir_attr.SetIndex(0);
  return graph;
}

// 判定节点是否为外提 Load：loop_axis 指向 BlockOuter 轴
// 注：FindAxis 无 const 重载，与产品代码 schedule_utils.h 保持一致按非 const 引用取图
bool IsHoistedLoadNode(af::AscGraph &graph, const af::AscNodePtr &node) {
  if (!af::ops::IsOps<af::ascir_op::Load>(node) || node->attr.sched.loop_axis == af::kIdNone) {
    return false;
  }
  const auto axis = graph.FindAxis(node->attr.sched.loop_axis);
  return axis != nullptr && axis->type == af::Axis::kAxisTypeBlockOuter;
}

std::vector<std::string> GetNodeNames(const af::AscGraph &graph) {
  std::vector<std::string> names;
  for (const auto &node : graph.GetAllNodes()) {
    names.emplace_back(node->GetName());
  }
  return names;
}

size_t GetNodeIndex(const std::vector<std::string> &names, const std::string &name) {
  const auto it = std::find(names.begin(), names.end(), name);
  return it == names.end() ? static_cast<size_t>(-1L) : static_cast<size_t>(std::distance(names.begin(), it));
}

// 前置不变式：外提 Load（loop_axis 为 BlockOuter 轴的 Load）位于第一个非前置闭包节点之前。
// 前置闭包 = 外提 Load 及其 Data 前驱（SortHoistedLoadsFirst 会一并前置以保证拓扑序合法）。
// Data 是无调度语义的图输入叶子，一律视为可前置；计算/访存节点出现后不允许再出现外提 Load。
void ExpectHoistedLoadsFront(af::AscGraph &graph) {
  bool non_front_seen = false;
  size_t hoisted_cnt = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (IsHoistedLoadNode(graph, node)) {
      EXPECT_FALSE(non_front_seen) << "hoisted Load[" << node->GetName() << "] is not at graph front";
      hoisted_cnt++;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;  // 图输入叶子，前置闭包成员（外提 Load 的 Data 前驱必须前置）
    }
    non_front_seen = true;
  }
  EXPECT_GT(hoisted_cnt, 0UL);
}

::ascir::FusedScheduledResult MakeFusedScheduledResult(af::AscGraph &graph) {
  ::ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.resize(1UL);
  auto &scheduled_result = fused_scheduled_result.node_idx_to_scheduled_results[0].emplace_back();
  scheduled_result.schedule_groups.emplace_back().impl_graphs.emplace_back(graph);
  return fused_scheduled_result;
}

}  // namespace

// 外提标记后 LoadOpSeqAdjust 的重排不得破坏“外提 Load 前置”，否则 codegen 会拆出重复循环。
TEST(LoadSeqHoistTest, LoadOpSeqAdjustKeepsHoistedLoadsFirst) {
  auto graph = MakeHoistedLoadOrderGraph("load_seq_hoist_order");
  const auto hoisted_node = graph.FindNode("load_hoist");
  ASSERT_NE(hoisted_node, nullptr);

  // 与 BufQueAllocator::HoistInvariantLoads 外提后的标记一致：loop_axis 指向 BlockOuter 轴
  af::AxisId block_outer_id = af::kIdNone;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->type == af::Axis::kAxisTypeBlockOuter) {
      block_outer_id = axis->id;
      break;
    }
  }
  ASSERT_NE(block_outer_id, af::kIdNone);
  hoisted_node->attr.sched.loop_axis = block_outer_id;
  hoisted_node->attr.sched.exec_condition = af::ExecuteCondition::kNoCache;
  ASSERT_TRUE(ScheduleUtils::SortHoistedLoadsFirst(graph));

  const auto names_before = GetNodeNames(graph);
  ASSERT_LT(GetNodeIndex(names_before, "load_hoist"), GetNodeIndex(names_before, "load_main"));
  ExpectHoistedLoadsFront(graph);

  auto fused_scheduled_result = MakeFusedScheduledResult(graph);
  Optimizer::ExecSeqAdvancedOfLoad(fused_scheduled_result);

  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  const auto names_after = GetNodeNames(impl_graph);
  // 只重排不增删节点
  ASSERT_EQ(names_after.size(), names_before.size());
  EXPECT_TRUE(std::is_permutation(names_after.begin(), names_after.end(), names_before.begin()));
  // 重排后外提 Load 仍前置，且先于循环内普通 Load
  ExpectHoistedLoadsFront(impl_graph);
  EXPECT_LT(GetNodeIndex(names_after, "load_hoist"), GetNodeIndex(names_after, "load_main"));
  EXPECT_LT(GetNodeIndex(names_after, "load_hoist"), GetNodeIndex(names_after, "load_tail"));
}

// 无外提 Load 的图：SortHoistedLoadsFirst 为 no-op，LoadOpSeqAdjust 不改变节点集合，也不引入外提标记。
TEST(LoadSeqHoistTest, LoadOpSeqAdjustWithoutHoistedLoadKeepsRewiringBenefit) {
  auto graph = MakeHoistedLoadOrderGraph("load_seq_no_hoist");
  ASSERT_FALSE(ScheduleUtils::SortHoistedLoadsFirst(graph));
  const auto names_before = GetNodeNames(graph);

  auto fused_scheduled_result = MakeFusedScheduledResult(graph);
  Optimizer::ExecSeqAdvancedOfLoad(fused_scheduled_result);

  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  const auto names_after = GetNodeNames(impl_graph);
  ASSERT_EQ(names_after.size(), names_before.size());
  EXPECT_TRUE(std::is_permutation(names_after.begin(), names_after.end(), names_before.begin()));
  for (const auto &node : impl_graph.GetAllNodes()) {
    EXPECT_FALSE(IsHoistedLoadNode(impl_graph, node));
  }
}

// 多循环轴图：两条链的 load 分属不同 loop_axis 且汇合于同一计算节点（满足原前移搜索条件）。
// id 前移 + ReorderByNodeId 的重分配隐含单轴节点连续排列假设，多轴交错下会破坏拓扑序，
// LoadOpSeqAdjust 应整体跳过：不执行任何 id 改写/重排，节点序保持建图原序。
af::AscGraph MakeMultiLoopAxesGraph(const std::string &name) {
  af::AscGraph graph(name.c_str());
  const auto s0 = graph.CreateSizeVar(8);
  const auto s1 = graph.CreateSizeVar(1024);
  const auto s2 = graph.CreateSizeVar(512);
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z0_block = graph.BlockSplit(z0.id).first;
  const auto z1_tile = graph.TileSplit(z1.id).second;
  const auto z2_tile = graph.TileSplit(z2.id).second;
  const SplitAxes axes_a{{z0_block->id, z1_tile->id}, {z1_tile->id}, z1_tile->id};
  const SplitAxes axes_b{{z0_block->id, z2_tile->id}, {z2_tile->id}, z2_tile->id};

  af::ascir_op::Load load_a("load_a");
  AddLoadChain(graph, axes_a, "load_a", 0, s1, load_a);
  af::ascir_op::Load load_b("load_b");
  AddLoadChain(graph, axes_b, "load_b", 1, s2, load_b);

  af::ascir_op::Add add("add");
  add.x1 = load_a.y;
  add.x2 = load_b.y;
  SetComputeAttr(add, axes_a, s1, af::ComputeType::kComputeElewise, af::ComputeUnit::kUnitVector);

  af::ascir_op::Store store("store");
  store.x = add.y;
  SetComputeAttr(store, axes_a, s1, af::ComputeType::kComputeStore, af::ComputeUnit::kUnitMTE3);

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = af::DT_FLOAT16;
  output.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output.attr.api.type = af::ApiType::kAPITypeBuffer;
  output.ir_attr.SetIndex(0);

  // 内存分配后的形态：两 load 的 que.id 不同（否则被共复用判定拦截，无法体现前移路径）
  graph.FindNode("load_a")->outputs[0].attr.que.id = 1;
  graph.FindNode("load_b")->outputs[0].attr.que.id = 2;
  return graph;
}

TEST(LoadSeqHoistTest, LoadOpSeqAdjustSkipsMultiLoopAxesGraph) {
  auto graph = MakeMultiLoopAxesGraph("load_seq_multi_axes");
  ASSERT_GT(ScheduleUtils::GetUniqueLoopAxisCnt(graph), 1U);
  const auto names_before = GetNodeNames(graph);

  auto fused_scheduled_result = MakeFusedScheduledResult(graph);
  Optimizer::ExecSeqAdvancedOfLoad(fused_scheduled_result);

  auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  // 多循环轴整体跳过：节点序与建图原序完全一致
  EXPECT_EQ(GetNodeNames(impl_graph), names_before);
}

}  // namespace optimize
