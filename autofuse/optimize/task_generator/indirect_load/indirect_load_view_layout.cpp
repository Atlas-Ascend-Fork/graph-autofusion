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
#include "utils/axis_utils.h"

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
    GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast path index %zu is out of range [0, %zu).",
           static_cast<int32_t>(template_id), broadcast_index, path.size());
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
      GELOGD("[IndirectLoad] Preserve node[%s] view: sched rank (%zu) does not match physical layout (%zu).",
             node->GetNamePtr(), node->attr.sched.axis.size(), layout.axis_ids.size());
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

// [边界化列分块] 行全驻留 × 全列使下游 VF 链的 tile 块远超平台 UB（如 [4,47360] 表：
// 1 行 = 189440B，VECOUT 4 块 = 757760B）。把 IndirectLoad 输出最内层连续轴（列）拆成
// 固定 TileOut/TileInner，并同步输出链全部节点视图。
// 列块必须整除列宽：GM 布局按整行连续存储（行宽 47360），merged 轴的元素线性偏移需等于
// a0×行宽 + a1T×列块；若列块不整除列宽（如 8192），行间距与块步长不匹配会导致偏移错位/越界。
constexpr int64_t kBoundaryFusionColumnTileElems = 8192L;
constexpr int64_t kColumnTileAlign = 8L;  // 32B 对齐（fp32）

// 列块必须严格小于列宽且整除：零值/越界校验拆成独立的前置返回，便于静态检查
// （G.EXP.22-CPP）识别除数非零。
bool IsExactDivisor(const int64_t value, const int64_t divisor) {
  if (divisor == 0L) {
    return false;
  }
  if (divisor < 0L) {
    return false;
  }
  if (divisor >= value) {
    return false;
  }
  return value % divisor == 0L;
}

int64_t PickColumnTile(const int64_t col_size) {
  // 优先：≤8192 的 8 倍数（32B 对齐）且整除列宽
  for (int64_t candidate = std::min(kBoundaryFusionColumnTileElems, col_size); candidate >= kColumnTileAlign;
       candidate -= kColumnTileAlign) {
    if (IsExactDivisor(col_size, candidate)) {
      return candidate;
    }
  }
  // fallback：行宽无 8 倍数因子（如 237790）时，选 ≤8192 的最大整除因子。线性偏移
  // merged×tile 仍精确等于 a0×行宽 + a1T×tile（整除保证），仅失去 32B 对齐，
  // 由 DataCopy 的非对齐/tail 路径处理。
  for (int64_t candidate = std::min(kBoundaryFusionColumnTileElems, col_size); candidate >= 8L; --candidate) {
    if (IsExactDivisor(col_size, candidate)) {
      return candidate;
    }
  }
  return col_size;
}

// [列分块] 收集需要同步列轴的节点：输出链 backward 闭包 + index 链。返回 false 表示输出链
// 无法识别（调用方保持原图）。
bool CollectColumnTileChain(const af::AscNodePtr &indirect_load, const af::AscNodePtr &output_store,
                            const std::vector<af::AscNodePtr> &index_region, NodeSet &chain) {
  if (!CollectSimtBackwardRegion({output_store}, indirect_load, chain)) {
    return false;
  }
  chain.emplace(indirect_load.get());
  // index 链（data/load/broadcast 等）的输出作为 IL 输入也会被拆列轴，必须把其节点一并纳入，
  // 否则其 sched.axis 残留原始列轴而 output 已拆成 outer/inner，导致
  // IdentifyZeroStrideAxisIndices 的 sched ⊆ output 断言失败。
  for (const auto &region_node : index_region) {
    if (region_node != nullptr) {
      chain.emplace(region_node.get());
    }
  }
  return true;
}

// [列分块] 把单个节点的输入 producer 加入闭包（已存在则跳过）。
void AddMissingInputOwners(const af::AscNode *node, NodeSet &chain, std::vector<const af::AscNode *> &extra_nodes,
                           bool &changed) {
  if (node == nullptr) {
    return;
  }
  auto *mutable_node = const_cast<af::AscNode *>(node);
  for (const auto &input : mutable_node->inputs()) {
    if (input == nullptr || input->anchor.GetOwnerNode() == nullptr) {
      continue;
    }
    auto *owner = dynamic_cast<af::AscNode *>(input->anchor.GetOwnerNode().get());
    if (owner != nullptr && chain.count(owner) == 0UL) {
      extra_nodes.push_back(owner);
      changed = true;
    }
  }
}

// [列分块] 输入链闭包：IL 的输入 producer（表 load1、index load/broadcast 及其上游）的输出
// tensor 作为下游输入会被拆列轴，其自身 sched.axis 必须同步，否则
// IdentifyZeroStrideAxisIndices 的 sched⊆output 断言失败。
void ExpandColumnTileInputClosure(NodeSet &chain) {
  for (bool changed = true; changed;) {
    changed = false;
    std::vector<const af::AscNode *> extra_nodes;
    for (const auto *chain_node : chain) {
      AddMissingInputOwners(chain_node, chain, extra_nodes, changed);
    }
    chain.insert(extra_nodes.begin(), extra_nodes.end());
  }
}

// [列分块] 把 tensor 视图里的列轴拆成 outer/inner；不含列轴（如标量）时跳过。
void SplitTensorColumnAxis(af::AscTensor *tensor, ascir::AxisId col_axis_id, ascir::AxisId col_outer,
                           ascir::AxisId col_inner, const af::Expression &inner_size) {
  if (tensor == nullptr) {
    return;
  }
  auto &attr = tensor->attr;
  if (std::find(attr.axis.begin(), attr.axis.end(), col_axis_id) == attr.axis.end()) {
    return;
  }
  const auto view =
      af::AxisUtils::SplitView({attr.axis, attr.repeats, attr.strides}, inner_size, col_outer, col_inner, col_axis_id);
  attr.axis = view.axis_ids;
  attr.repeats = view.repeats;
  attr.strides = view.strides;
}

bool NodeOutputsCarryColumnAxis(af::AscNode &node, ascir::AxisId col_outer, ascir::AxisId col_inner) {
  for (const auto &out : node.outputs()) {
    if (out == nullptr) {
      continue;
    }
    const auto &axis = out->attr.axis;
    if (std::find(axis.begin(), axis.end(), col_outer) == axis.end() ||
        std::find(axis.begin(), axis.end(), col_inner) == axis.end()) {
      return false;
    }
  }
  return true;
}

// [列分块] 调度轴同步拆分：IdentifyZeroStrideAxisIndices 要求 sched.axis ⊆ 每个 output 的
// attr.axis。仅当所有输出都确实带列轴（已拆成 outer/inner）时同步，任一输出不含列轴
// （如标量/单元素）时保持 sched.axis 原样，避免该输出上找不到拆分后的轴而断言。
void SyncNodeScheduleColumnAxis(af::AscNode &node, ascir::AxisId col_axis_id, ascir::AxisId col_outer,
                                ascir::AxisId col_inner) {
  if (!NodeOutputsCarryColumnAxis(node, col_outer, col_inner)) {
    return;
  }
  auto &sched_axis = node.attr.sched.axis;
  const auto sched_it = std::find(sched_axis.begin(), sched_axis.end(), col_axis_id);
  if (sched_it == sched_axis.end()) {
    return;
  }
  const auto pos = std::distance(sched_axis.begin(), sched_it);
  sched_axis.erase(sched_it);
  sched_axis.insert(sched_axis.begin() + pos, col_inner);
  sched_axis.insert(sched_axis.begin() + pos, col_outer);
}

// [列分块] tmp 缓冲需容纳一个列块：算子默认 tmp（如 8KB）在图改写期固定，列分块后单块变大
// （列块元素 × dtype），固定小尺寸会越界导致结果错误。
void EnlargeNodeTmpBuffers(af::AscNode &node, int64_t min_tmp_bytes) {
  for (auto &tmp_buffer : node.attr.tmp_buffers) {
    int64_t tmp_size = 0;
    if (tmp_buffer.buf_desc.size.IsValid() && tmp_buffer.buf_desc.size.GetConstValue(tmp_size) && tmp_size > 0 &&
        tmp_size < min_tmp_bytes) {
      tmp_buffer.buf_desc.size = af::Symbol(min_tmp_bytes);
    }
  }
}

void ApplyColumnTileToNode(af::AscNode &node, ascir::AxisId col_axis_id, ascir::AxisId col_outer,
                           ascir::AxisId col_inner, const af::Expression &inner_size, int64_t min_tmp_bytes) {
  for (const auto &out : node.outputs()) {
    SplitTensorColumnAxis(out, col_axis_id, col_outer, col_inner, inner_size);
  }
  for (const auto &in : node.inputs()) {
    SplitTensorColumnAxis(in, col_axis_id, col_outer, col_inner, inner_size);
  }
  SyncNodeScheduleColumnAxis(node, col_axis_id, col_outer, col_inner);
  EnlargeNodeTmpBuffers(node, min_tmp_bytes);
}

// [列分块] 确保 IL 输入走 GM 直读（direct-GM），不按全量驻留 UB。0002 由
// AnnotateSimtTemplateRoles 完成了该标注；部分图（如含复合 index_select 链）未走到该标注时
// input 会全量计入 UB（如 0030 的 228MB），列分块无法收敛。此处按 0002 的 role 约定补齐
// （仅当 producer 尚未标注时，避免覆盖既有语义）。
void MarkColumnTileInputRoles(const af::AscNodePtr &indirect_load) {
  for (size_t input_idx = 0UL; input_idx < static_cast<size_t>(kIndirectLoadInputCount); ++input_idx) {
    const auto producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx);
    if (producer == nullptr) {
      continue;
    }
    if (ascgen_utils::indirect_load::GetTemplateRole(producer) != ascgen_utils::indirect_load::TemplateRole::kNone) {
      continue;
    }
    (void)ascgen_utils::indirect_load::SetTemplateRole(
        producer, input_idx == ascgen_utils::indirect_load::kInputTensorIndex
                      ? ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary
                      : ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
  }
}

af::Status SplitBoundaryFusionColumnTile(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                         const af::AscNodePtr &output_store,
                                         const std::vector<af::AscNodePtr> &index_region) {
  const auto &out_attr = indirect_load->outputs()[0]->attr;
  if (out_attr.axis.size() < 2UL) {
    return af::SUCCESS;
  }
  const ascir::AxisId col_axis_id = out_attr.axis.back();
  const auto *col_axis = graph.FindAxis(col_axis_id);
  GE_ASSERT_NOTNULL(col_axis);
  int64_t col_size = 0;
  if (!col_axis->size.GetConstValue(col_size)) {
    return af::SUCCESS;
  }
  const int64_t col_tile = PickColumnTile(col_size);
  if (col_tile >= col_size) {
    return af::SUCCESS;  // 找不到整除列块（或列本身很小），保持整列
  }
  ascir::AxisId col_outer = af::kIdNone;
  ascir::AxisId col_inner = af::kIdNone;
  GE_ASSERT_SUCCESS(CreateFixedTileSplit(graph, col_axis_id, col_outer, col_inner, col_tile));
  const auto *inner_axis = graph.FindAxis(col_inner);
  GE_ASSERT_NOTNULL(inner_axis);
  NodeSet chain;
  if (!CollectColumnTileChain(indirect_load, output_store, index_region, chain)) {
    return af::SUCCESS;
  }
  ExpandColumnTileInputClosure(chain);
  const int64_t min_tmp_bytes = col_tile * 4L;  // 按 4B 元素保守估算
  for (const auto *chain_node : chain) {
    if (chain_node != nullptr) {
      ApplyColumnTileToNode(*const_cast<af::AscNode *>(chain_node), col_axis_id, col_outer, col_inner, inner_axis->size,
                            min_tmp_bytes);
    }
  }
  MarkColumnTileInputRoles(indirect_load);
  const auto il_op_desc = indirect_load->GetOpDesc();
  if (il_op_desc != nullptr) {
    // 标记单次 SIMT 调用的 tile 内跨度，供 lowering metadata 只在 boundary fusion 列分块场景
    // 把 inner span 从整列宽改为列块大小。
    (void)af::AttrUtils::SetInt(il_op_desc, "af.internal.indirect_load.column_tile_inner", col_tile);
  }
  GELOGI("[IndirectLoad] boundary fusion column tile %ld applied for node[%s].", col_tile, indirect_load->GetNamePtr());
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

// [边界化判定] 扫描 SIMT 标量求值器消费集（输出链 backward 闭包，含汇入链中的
// side-input）内节点的可发射性。豁免集与发射期 EmitSimtEvaluatorNodes 的专用路径
// 分流保持一致（单一事实源）：Load（GM 直读）/Scalar/ScalarData/Store/Transpose 的
// 专用发射路径无条件成立，直接豁免；Arange 的专用路径 EmitSimtArangeExpr 有前置
// 条件（ArangeAscIrCodegenImplV2::IsSimtScalarSupported：零输入、int32/int64、
// 单一 varying 轴），可发射性必须真实查询，不可发射时调度期即降级/拒绝，避免
// 调度通过、codegen 硬挂。其余节点与 EmitSimtScalarExpr 共用同一 V2 支持查询
// （CanEmitSimtScalar）。返回首个不可发射节点（nullptr = 全部可发射）。
af::AscNodePtr FindUnemittableSimtEvaluatorNode(const af::AscGraph &graph, const NodeSet &evaluator_nodes) {
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (evaluator_nodes.count(node.get()) == 0UL) {
      continue;
    }
    // IndirectLoad 自身即模板的 gather 发射点，不进入标量求值器：post-Reduce 形态的求值器
    // 消费链从 local producer 回溯时会把它一并收进来，必须豁免，否则所有 post-Reduce 候选
    // 都会被误判为不可发射而整体降级。Reduce 节点由普通向量调度执行（不进入求值器），
    // 同样豁免——输出链检测必须与 ValidateSimtTemplateRegion 的 index 链口径保持一致。
    if (af::ops::IsOps<af::ascir_op::IndirectLoad>(node) || af::ops::IsOps<af::ascir_op::Load>(node) ||
        af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node) ||
        af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Transpose>(node) ||
        node->attr.api.compute_type == af::ComputeType::kComputeReduce) {
      continue;
    }
    if (!CanEmitSimtScalar(node)) {
      return node;
    }
  }
  return nullptr;
}
}  // namespace indirect_load_detail
}  // namespace optimize
