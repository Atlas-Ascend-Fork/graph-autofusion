/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

bool IsInputDataSource(const af::AscNodePtr &node) {
  return ScheduleUtils::IsDataInput(node) || af::ops::IsOps<af::ascir_op::Scalar>(node);
}

bool IsSimtTemplateRole(const ascgen_utils::indirect_load::TemplateRole role) {
  return role == ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtOp;
}

af::Status ClearSimtTemplateRoles(af::AscGraph &graph) {
  size_t cleared_count = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsSimtTemplateRole(ascgen_utils::indirect_load::GetTemplateRole(node))) {
      continue;
    }
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kNone));
    ++cleared_count;
  }
  // 普通调度节点必须保留自己的 API call；清理候选图拷贝中可能残留的 SIMT 角色，
  // 防止节点虽未被本轮重新标注，仍因 skips_api_emit 被父图静默跳过。
  GELOGD("[IndirectLoad] Cleared %zu stale SIMT roles before candidate annotation.", cleared_count);
  return af::SUCCESS;
}

af::Status ClearNormalScheduleSimtRoles(af::AscGraph &graph, const NodeSet &normal_schedule_nodes,
                                        const NodeSet &exempt_nodes) {
  size_t cleared_count = 0UL;
  for (const auto &node : graph.GetAllNodes()) {
    if (normal_schedule_nodes.count(node.get()) == 0UL || exempt_nodes.count(node.get()) != 0UL ||
        !IsSimtTemplateRole(ascgen_utils::indirect_load::GetTemplateRole(node))) {
      continue;
    }
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kNone));
    ++cleared_count;
  }
  // 普通调度区域的角色标注必须在所有 SIMT 分支标注完成后兜底清理，
  // 防止后续分支遍历再次给跨 VectorFunc 输入节点写入 skips_api_emit。
  // 豁免集即 lowering 侧 BuildSimtLoweringMetadata 会校验的 index_region：
  // 与 IndirectLoad 输出直连又汇入 post Reduce 输入的共享节点同时落在
  // normal_schedule 依赖闭包内，但 lowering 侧仍按 SIMT 区域消费并要求角色，
  // 不能被清理，否则 no scalar evaluator role。
  GELOGD("[IndirectLoad] Cleared %zu SIMT roles from normal schedule nodes, %zu exempt lowering nodes kept.",
         cleared_count, exempt_nodes.size());
  return af::SUCCESS;
}

// A retained Transpose separates two physical coordinate systems. Do not overwrite its source GM view.
bool IsInputRegionBoundary(const af::AscNodePtr &node) {
  if (IsInputDataSource(node)) {
    return true;
  }
  if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
    return false;
  }
  const auto consumers = node->GetOutDataNodes();
  return std::any_of(consumers.begin(), consumers.end(), [](const af::NodePtr &consumer) {
    return af::ops::IsOps<af::ascir_op::Transpose>(std::dynamic_pointer_cast<af::AscNode>(consumer));
  });
}

// 检查已选中的 Broadcast 路径是否可以安全折叠；不支持时只淘汰当前 candidate。
bool IsSupportedBroadcastPath(const NodePath &path, size_t broadcast_index, ascir::TemplateId template_id) {
  if (broadcast_index >= path.size()) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path index is out of range.",
           static_cast<int32_t>(template_id));
    return false;
  }
  const af::AscNodePtr &broadcast = path[broadcast_index];
  if (!IsSingleConsumerWithoutControlEdge(broadcast)) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path node[%s] is not safely foldable.",
           static_cast<int32_t>(template_id), broadcast->GetNamePtr());
    return false;
  }
  // CollectInputPaths 只有 SK 会继续回溯 Broadcast 前的单输入链。
  for (size_t i = 0UL; i < broadcast_index; ++i) {
    const af::AscNodePtr &element = path[i];
    if (element->outputs().size() != 1UL || !IsSingleConsumerWithoutControlEdge(element) ||
        !ScheduleUtils::IsElewise(element)) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast pre element node[%s] is not safely foldable.",
             static_cast<int32_t>(template_id), element->GetNamePtr());
      return false;
    }
  }
  for (size_t i = broadcast_index + 1UL; i < path.size(); ++i) {
    const af::AscNodePtr &element = path[i];
    if (!IsSingleConsumerWithoutControlEdge(element)) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast post element node[%s] is not safely foldable.",
             static_cast<int32_t>(template_id), element->GetNamePtr());
      return false;
    }
  }
  return true;
}

af::Status GetBroadcastPhysicalAttr(const af::AscNodePtr &broadcast, ascir::TemplateId template_id,
                                    af::AscTensorAttr &physical_attr) {
  const auto input_anchor = broadcast->GetInDataAnchor(0UL);
  GE_ASSERT_NOTNULL(input_anchor, "IndirectLoad template[%d] Broadcast node[%s] input anchor is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "IndirectLoad template[%d] Broadcast node[%s] source anchor is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(producer, "IndirectLoad template[%d] Broadcast node[%s] source is invalid.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const size_t output_idx = static_cast<size_t>(peer_out_anchor->GetIdx());
  const auto producer_outputs = producer->outputs();
  GE_ASSERT_TRUE(output_idx < producer_outputs.size(),
                 "IndirectLoad template[%d] Broadcast node[%s] source output index is out of range.",
                 static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  GE_ASSERT_NOTNULL(producer_outputs[output_idx], "IndirectLoad template[%d] Broadcast node[%s] source output is null.",
                    static_cast<int32_t>(template_id), broadcast->GetNamePtr());
  const auto &producer_attr = producer_outputs[output_idx]->attr;
  physical_attr.axis = producer_attr.axis;
  physical_attr.repeats = producer_attr.repeats;
  physical_attr.strides = producer_attr.strides;
  return af::SUCCESS;
}

af::Status InlineBroadcastPath(NodePath &path, int64_t path_broadcast_index, ascir::TemplateId template_id) {
  GE_ASSERT_TRUE(path_broadcast_index >= 0L, "IndirectLoad Broadcast index is invalid.");
  const size_t broadcast_index = static_cast<size_t>(path_broadcast_index);
  GE_ASSERT_TRUE(broadcast_index < path.size(), "IndirectLoad Broadcast index %zu is out of range [0, %zu).",
                 broadcast_index, path.size());
  const af::AscNodePtr broadcast = path[broadcast_index];
  const auto owner_graph = broadcast->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::IsolateNodeOneIO(broadcast));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner_graph, broadcast));
  GELOGD("[IndirectLoad] Inline Broadcast node[%s] for template[%d] stride-aware physical source window.",
         broadcast->GetNamePtr(), static_cast<int32_t>(template_id));
  path.erase(path.begin() + static_cast<int64_t>(broadcast_index));
  return af::SUCCESS;
}

af::Status FoldBroadcastPath(InputViewPlan &plan, ascir::TemplateId template_id, bool &is_candidate_legal) {
  if (plan.path_broadcast_index == kInvalidBroadcastIndex) {
    return af::SUCCESS;
  }
  const size_t broadcast_index = static_cast<size_t>(plan.path_broadcast_index);
  if (!IsSupportedBroadcastPath(plan.path, broadcast_index, template_id)) {
    is_candidate_legal = false;
    return af::SUCCESS;
  }
  return InlineBroadcastPath(plan.path, plan.path_broadcast_index, template_id);
}

af::Status BuildBroadcastLogicalView(const af::AscTensorAttr &logical_attr, const af::AscTensorAttr &physical_attr,
                                     ascir::TemplateId template_id,
                                     ascgen_utils::indirect_load::LogicalTensorView &view) {
  view = {logical_attr.axis, logical_attr.repeats, logical_attr.strides};
  GE_ASSERT_TRUE(
      physical_attr.repeats.size() == view.sizes.size() && physical_attr.strides.size() == view.strides.size(),
      "IndirectLoad template[%d] Broadcast source layout rank mismatch.", static_cast<int32_t>(template_id));
  view.strides = physical_attr.strides;
  for (size_t dim = 0UL; dim < view.sizes.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(physical_attr.repeats[dim], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(view.sizes[dim], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      view.strides[dim] = af::sym::kSymbolZero;
    }
  }
  return af::SUCCESS;
}

af::Status ApplyPhysicalView(const NodePath &path,
                             const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout) {
  GE_ASSERT_TRUE(
      layout.axis_ids.size() == layout.physical_repeats.size() && layout.axis_ids.size() == layout.strides.size(),
      "IndirectLoad physical execution view rank mismatch.");
  for (const af::AscNodePtr &node : path) {
    if (IsInputRegionBoundary(node)) {
      continue;
    }
    // Gather-Norm 路径可能包含保留原始 GM View 的 Load。改写的前提是调度轴与执行
    // 布局同构（下方断言）；仅 tensor 视图 rank 与布局不一致时必须强写 layout 视图
    // 加以统一，否则节点视图停留在中间态，跨边界 Broadcast/Reduce 的 in/out 视图
    // 分叉并触发 NodeCacheMarker/IsTailBroadcastNode 断言（历史回归：Preserve 条件
    // 误用 tensor rank 导致 embedding 等场景全量失败）。因此 Preserve 仅保留
    // sched rank 不匹配的防御场景（完全移除会使 layernorm 等复合场景在
    // BuildReduceContext 阶段失败）。
    if (node->attr.sched.axis.size() != layout.axis_ids.size()) {
      GELOGD("[IndirectLoad] Preserve node[%s] view: sched rank does not match physical layout.", node->GetNamePtr());
      continue;
    }
    GE_ASSERT_EQ(node->attr.sched.axis.size(), layout.axis_ids.size());
    node->attr.sched.axis = layout.axis_ids;
    for (const auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      output->attr.axis = layout.axis_ids;
      output->attr.repeats = layout.physical_repeats;
      output->attr.strides = layout.strides;
    }
  }
  return af::SUCCESS;
}

bool NeedsAlignedUbWindow(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_index) {
  if (layout.kind != ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
    return false;
  }
  af::Expression physical_span = af::sym::kSymbolOne;
  for (size_t index = layout.sizes.size(); index > axis_index; --index) {
    const size_t dim = index - 1UL;
    if (af::SymbolicUtils::StaticCheckEq(layout.strides[dim], physical_span) != af::TriBool::kTrue) {
      return true;
    }
    physical_span = physical_span + (layout.sizes[dim] - af::sym::kSymbolOne) * layout.strides[dim];
  }
  return false;
}

// gather 轴内侧（含轴维）存在尺寸大于 1 的零步长维：该维为视图内广播（如 gather 轴
// 广播源），staging 需逐行展开广播而非批量 DMA 块，多行窗口无法批量搬运；行维
// （axis_index 之前）的零步长不受限——staging 单份、模板按零步长重读（如
// row-invariant 索引 {0,1}）。
bool HasBroadcastInsideWindow(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_index) {
  for (size_t dim = axis_index; dim < layout.strides.size(); ++dim) {
    const bool zero_stride =
        af::SymbolicUtils::StaticCheckEq(layout.strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue;
    const bool degenerate =
        af::SymbolicUtils::StaticCheckEq(layout.sizes[dim], af::sym::kSymbolOne) == af::TriBool::kTrue;
    if (zero_stride && !degenerate) {
      return true;
    }
  }
  return false;
}

af::Status AnnotateStridedUbPath(const NodePath &path) {
  for (const af::AscNodePtr &node : path) {
    if (IsInputDataSource(node) || ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    GE_ASSERT_TRUE(role == ascgen_utils::indirect_load::TemplateRole::kNone ||
                       role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath,
                   "IndirectLoad strided path node[%s] already has template role[%ld].", node->GetNamePtr(),
                   static_cast<int64_t>(role));
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }
  return af::SUCCESS;
}

af::Status ApplyIndirectLoadPathLayout(const NodePath &path,
                                       const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
                                       bool needs_alignment) {
  if (layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact) {
    return ApplyPhysicalView(path, layout);
  }
  // Use the rewritten path as the source of truth; Broadcast may have been folded before this point.
  const auto broadcast =
      std::find_if(path.begin(), path.end(), [](const af::AscNodePtr &node) { return IsBroadcastNode(node); });
  const bool has_dynamic_shape = std::any_of(layout.sizes.begin(), layout.sizes.end(),
                                             [](const af::Expression &size) { return !size.IsConstExpr(); });
  if (layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided && has_dynamic_shape &&
      broadcast != path.end()) {
    // Keep the upstream producer view intact; only the rewritten producer-to-Broadcast segment is normalized.
    const NodePath broadcast_path(path.begin(), broadcast + 1);
    GE_ASSERT_SUCCESS(ApplyPhysicalView(broadcast_path, layout));
  }
  if (needs_alignment) {
    return AnnotateStridedUbPath(path);
  }
  return af::SUCCESS;
}

af::Status MergeAxesForTemplate(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const std::string &name,
                                af::AxisId &merged_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad merge axis source is empty, name:%s.", name.c_str());
  merged_axis = axes.size() == 1UL ? axes.front() : graph.MergeAxis(axes, name)->id;
  return af::SUCCESS;
}

af::Status CreateFixedTileSplit(af::AscGraph &graph, af::AxisId axis_id, af::AxisId &outer_id, af::AxisId &inner_id,
                                int64_t group_size) {
  const auto *axis = graph.FindAxis(axis_id);
  GE_ASSERT_NOTNULL(axis, "IndirectLoad fixed tile axis %ld is not found.", axis_id);
  const af::Expression inner_size = af::Symbol(group_size);
  const af::Expression outer_size = group_size == 1L ? axis->size : af::sym::Ceiling(axis->size / inner_size);
  outer_id =
      graph.CreateAxis(axis->name + "T", ascir::Axis::Type::kAxisTypeTileOuter, outer_size, {axis_id}, af::kIdNone).id;
  inner_id =
      graph.CreateAxis(axis->name + "t", ascir::Axis::Type::kAxisTypeTileInner, inner_size, {axis_id}, outer_id).id;
  auto *outer_axis = graph.FindAxis(outer_id);
  GE_ASSERT_NOTNULL(outer_axis, "IndirectLoad fixed tile outer axis %ld is not found.", outer_id);
  outer_axis->split_pair_other_id = inner_id;
  return af::SUCCESS;
}

af::Status BuildAxisViewByBoundary(af::AscGraph &graph, const std::vector<af::AxisId> &axes, size_t boundary,
                                   af::AxisId &outer_axis, af::AxisId &inner_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad output axis is empty.");
  GE_ASSERT_TRUE(boundary <= axes.size(), "IndirectLoad axis boundary %zu is out of range [0, %zu].", boundary,
                 axes.size());
  const size_t split = boundary;
  const std::vector<af::AxisId> outer_axes(axes.begin(), axes.begin() + static_cast<int64_t>(split));
  const std::vector<af::AxisId> inner_axes(axes.begin() + static_cast<int64_t>(split), axes.end());
  if (outer_axes.empty()) {
    outer_axis = graph.CreateAxis(kSingleOuterAxisName, af::sym::kSymbolOne).id;
  } else {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, outer_axes, kOuterAxisName, outer_axis));
  }
  if (!inner_axes.empty()) {
    GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, inner_axes, kInnerAxisName, inner_axis));
  }
  return af::SUCCESS;
}

bool CanEmitSimtScalar(const af::AscNodePtr &node) {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  const auto *v2_impl = impl == nullptr ? nullptr : dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  return v2_impl != nullptr && v2_impl->IsSimtScalarSupported(*node);
}
}  // namespace indirect_load_detail
}  // namespace optimize
