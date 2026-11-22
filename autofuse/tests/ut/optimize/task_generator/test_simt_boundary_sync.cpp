/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <vector>

#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "task_generator/simt_boundary_sync.h"
#include "symbolizer/symbolic_utils.h"

namespace {
struct SyncViewGraphHandle {
  af::AscNodePtr indirect_load;
  af::AscNodePtr simt_node;
  af::AxisId outer_axis = af::kIdNone;
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  af::AxisId inner_axis = af::kIdNone;
  std::vector<af::AxisId> output_axes;
  std::vector<af::Expression> output_repeats;
  std::vector<af::AxisId> merged_axes;
};

template <typename Op>
void SetSyncNodeView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                     const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

// 在 graph 上构建 SIMT 边界同步的最小图：IndirectLoad(带 TemplateAxes 注解) -> Abs(SIMT 角色节点)。
// Abs 输出视图停留在模板 merge 后的轴空间 [outer, y2]，vectorized_axis 保留一个不在视图中的
// 原始轴（y1），模拟调度 split 前的 pass 期残留状态。
void BuildSimtBoundarySyncGraph(af::AscGraph &graph, SyncViewGraphHandle &handle, bool with_indirect_load = true,
                                bool annotate_simt_role = true) {
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = graph.CreateSizeVar(3);
  const af::Expression s2 = graph.CreateSizeVar(5);
  const af::Expression in0 = graph.CreateSizeVar(2);
  const af::Expression in1 = graph.CreateSizeVar(4);
  const af::Expression in2 = graph.CreateSizeVar(5);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const auto y2 = graph.CreateAxis("y2", s2);
  const auto x0 = graph.CreateAxis("x0", in0);
  const auto x1 = graph.CreateAxis("x1", in1);
  const auto x2 = graph.CreateAxis("x2", in2);
  handle.inner_axis = y2.id;
  handle.output_axes = {y0.id, y1.id, y2.id};
  handle.output_repeats = {s0, s1, s2};
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id};
  const std::vector<af::Expression> input_repeats = {in0, in1, in2};
  const std::vector<af::Expression> input_strides = {in1 * in2, in2, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s1 * s2, s2, af::sym::kSymbolOne};

  af::ascir_op::Data input_data("input_data", graph);
  input_data.ir_attr.SetIndex(0);
  SetSyncNodeView(input_data, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = input_data.y;
  SetSyncNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index_data("index_data", graph);
  index_data.ir_attr.SetIndex(1);
  SetSyncNodeView(index_data, af::DT_INT32, handle.output_axes, handle.output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index_data.y;
  SetSyncNodeView(index_load, af::DT_INT32, handle.output_axes, handle.output_repeats, output_strides);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  if (with_indirect_load) {
    indirect_load.x1 = input_load.y;
    indirect_load.x2 = index_load.y;
    indirect_load.ir_attr.SetAxis(1);
    SetSyncNodeView(indirect_load, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, output_strides);
  }

  // 模板轴：outer=merge(y0,y1)，inner=y2；tile split (T, t) 建立在 outer 上，
  // 与调度期 prebuilt tiling 的轴形态一致。
  const auto outer = graph.MergeAxis({y0.id, y1.id}, "indirect_load_outer");
  handle.outer_axis = outer->id;
  handle.tile_outer_axis = graph
                               .CreateAxis("indirect_load_outerT", ascir::Axis::Type::kAxisTypeTileOuter, outer->size,
                                           {outer->id}, af::kIdNone)
                               .id;
  handle.tile_inner_axis = graph
                               .CreateAxis("indirect_load_outert", ascir::Axis::Type::kAxisTypeTileInner,
                                           af::sym::kSymbolOne, {outer->id}, handle.tile_outer_axis)
                               .id;
  auto *tile_outer = graph.FindAxis(handle.tile_outer_axis);
  if (tile_outer != nullptr) {
    tile_outer->split_pair_other_id = handle.tile_inner_axis;
  }

  ascgen_utils::indirect_load::TemplateAxes template_axes;
  template_axes.outer_axis = handle.outer_axis;
  template_axes.inner_axis = handle.inner_axis;
  template_axes.vectorized_axes = {handle.inner_axis};

  // SIMT 角色节点：输出视图为模板 merge 后的 [outer, y2]。
  const std::vector<af::AxisId> merged_axes = {handle.outer_axis, handle.inner_axis};
  handle.merged_axes = merged_axes;
  const std::vector<af::Expression> merged_repeats = {s0 * s1, s2};
  const std::vector<af::Expression> merged_strides = {s2, af::sym::kSymbolOne};
  if (with_indirect_load) {
    af::ascir_op::Abs simt_abs("simt_abs");
    simt_abs.x = indirect_load.y;
    simt_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
    simt_abs.attr.api.type = af::ApiType::kAPITypeCompute;
    SetSyncNodeView(simt_abs, af::DT_FLOAT16, merged_axes, merged_repeats, merged_strides);
    af::ascir_op::Store store("store");
    store.x = simt_abs.y;
    SetSyncNodeView(store, af::DT_FLOAT16, merged_axes, merged_repeats, merged_strides);
    af::ascir_op::Output output("output");
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetSyncNodeView(output, af::DT_FLOAT16, merged_axes, merged_repeats, merged_strides);
  } else {
    // 无 IndirectLoad 的图：Abs 直接消费 input_load，同步必须安全跳过。
    af::ascir_op::Abs simt_abs("simt_abs");
    simt_abs.x = input_load.y;
    simt_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
    simt_abs.attr.api.type = af::ApiType::kAPITypeCompute;
    SetSyncNodeView(simt_abs, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, output_strides);
    af::ascir_op::Store store("store");
    store.x = simt_abs.y;
    SetSyncNodeView(store, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, output_strides);
    af::ascir_op::Output output("output");
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    SetSyncNodeView(output, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, output_strides);
  }

  handle.indirect_load = graph.FindNode("indirect_load");
  handle.simt_node = graph.FindNode("simt_abs");
  if (with_indirect_load) {
    ASSERT_NE(handle.indirect_load, nullptr);
    ASSERT_EQ(ascgen_utils::indirect_load::SetTemplateAxes(handle.indirect_load, template_axes), af::SUCCESS);
  }
  if (annotate_simt_role) {
    ASSERT_NE(handle.simt_node, nullptr);
    ASSERT_EQ(ascgen_utils::indirect_load::SetTemplateRole(
                  handle.simt_node, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform),
              af::SUCCESS);
  }
}

std::pair<af::AxisPtr, af::AxisPtr> MakeTiledPair(const af::AscGraph &graph, const SyncViewGraphHandle &handle) {
  af::AxisPtr tile_outer;
  af::AxisPtr tile_inner;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis == nullptr) {
      continue;
    }
    if (axis->id == handle.tile_outer_axis) {
      tile_outer = axis;
    } else if (axis->id == handle.tile_inner_axis) {
      tile_inner = axis;
    }
  }
  return {tile_outer, tile_inner};
}
}  // namespace

TEST(SimtBoundarySyncTest, SyncsSimtRoleViewSplitAndVectorizedAxes) {
  af::AscGraph graph("simt_boundary_sync_ut_graph");
  SyncViewGraphHandle handle;
  BuildSimtBoundarySyncGraph(graph, handle);
  ASSERT_NE(handle.simt_node, nullptr);
  // vectorized 残留 y1（不在 merge 后的视图中）。
  handle.simt_node->outputs()[0]->attr.vectorized_axis = {handle.output_axes[1]};

  const auto tiled_pair = MakeTiledPair(graph, handle);
  ASSERT_NE(tiled_pair.first, nullptr);
  ASSERT_NE(tiled_pair.second, nullptr);
  ASSERT_EQ(optimize::task_generator::SyncSimtBoundaryViews(graph, {tiled_pair}), af::SUCCESS);

  // 1) tensor view split 同步：outer 轴在视图中被替换为 (TileOuter, TileInner)。
  const auto &synced_axis = handle.simt_node->outputs()[0]->attr.axis;
  ASSERT_EQ(synced_axis.size(), 3UL);
  EXPECT_EQ(synced_axis[0], handle.tile_outer_axis);
  EXPECT_EQ(synced_axis[1], handle.tile_inner_axis);
  EXPECT_EQ(synced_axis[2], handle.inner_axis);

  // 2) vectorized_axis 重映射：不在视图中的 y1 经 outer（merge 关系）映射到 TileInner。
  const auto &synced_vectorized = handle.simt_node->outputs()[0]->attr.vectorized_axis;
  ASSERT_EQ(synced_vectorized.size(), 1UL);
  EXPECT_EQ(synced_vectorized[0], handle.tile_inner_axis);
}

TEST(SimtBoundarySyncTest, KeepsVectorizedAxisThatRemainsInView) {
  af::AscGraph graph("simt_boundary_sync_ut_graph");
  SyncViewGraphHandle handle;
  BuildSimtBoundarySyncGraph(graph, handle);
  ASSERT_NE(handle.simt_node, nullptr);
  // vectorized 保持在视图中的尾轴（inner），不应被改写。
  handle.simt_node->outputs()[0]->attr.vectorized_axis = {handle.inner_axis};

  const auto tiled_pair = MakeTiledPair(graph, handle);
  ASSERT_NE(tiled_pair.first, nullptr);
  ASSERT_NE(tiled_pair.second, nullptr);
  ASSERT_EQ(optimize::task_generator::SyncSimtBoundaryViews(graph, {tiled_pair}), af::SUCCESS);
  const auto &synced_vectorized = handle.simt_node->outputs()[0]->attr.vectorized_axis;
  ASSERT_EQ(synced_vectorized.size(), 1UL);
  EXPECT_EQ(synced_vectorized[0], handle.inner_axis);
}

TEST(SimtBoundarySyncTest, SkipsNodesWithoutSimtRole) {
  af::AscGraph graph("simt_boundary_sync_ut_graph");
  SyncViewGraphHandle handle;
  BuildSimtBoundarySyncGraph(graph, handle, true, false);
  ASSERT_NE(handle.simt_node, nullptr);
  handle.simt_node->outputs()[0]->attr.vectorized_axis = {handle.output_axes[1]};

  const auto tiled_pair = MakeTiledPair(graph, handle);
  ASSERT_NE(tiled_pair.first, nullptr);
  ASSERT_NE(tiled_pair.second, nullptr);
  ASSERT_EQ(optimize::task_generator::SyncSimtBoundaryViews(graph, {tiled_pair}), af::SUCCESS);

  // 无 SIMT 角色的节点不参与同步：视图与 vectorized 保持原状。
  const auto &output = handle.simt_node->outputs()[0]->attr;
  EXPECT_EQ(output.axis, handle.merged_axes);
  ASSERT_EQ(output.vectorized_axis.size(), 1UL);
  EXPECT_EQ(output.vectorized_axis[0], handle.output_axes[1]);
}

TEST(SimtBoundarySyncTest, EmptyTiledAxesIsNoOp) {
  af::AscGraph graph("simt_boundary_sync_ut_graph");
  SyncViewGraphHandle handle;
  BuildSimtBoundarySyncGraph(graph, handle);
  ASSERT_NE(handle.simt_node, nullptr);
  handle.simt_node->outputs()[0]->attr.vectorized_axis = {handle.output_axes[1]};

  ASSERT_EQ(optimize::task_generator::SyncSimtBoundaryViews(graph, {}), af::SUCCESS);
  const auto &output = handle.simt_node->outputs()[0]->attr;
  EXPECT_EQ(output.axis, handle.merged_axes);
  EXPECT_EQ(output.vectorized_axis, std::vector<af::AxisId>{handle.output_axes[1]});
}

TEST(SimtBoundarySyncTest, WorksOnGraphWithoutIndirectLoad) {
  af::AscGraph graph("simt_boundary_sync_ut_graph");
  SyncViewGraphHandle handle;
  BuildSimtBoundarySyncGraph(graph, handle, false);
  ASSERT_NE(handle.simt_node, nullptr);
  handle.simt_node->outputs()[0]->attr.vectorized_axis = {handle.output_axes[1]};
  const auto tiled_pair = MakeTiledPair(graph, handle);

  // 无 IndirectLoad 注解时函数必须安全返回，不做任何改写。
  ASSERT_EQ(optimize::task_generator::SyncSimtBoundaryViews(graph, {tiled_pair}), af::SUCCESS);
  const auto &output = handle.simt_node->outputs()[0]->attr;
  EXPECT_EQ(output.axis, handle.output_axes);
  EXPECT_EQ(output.vectorized_axis, std::vector<af::AxisId>{handle.output_axes[1]});
}
