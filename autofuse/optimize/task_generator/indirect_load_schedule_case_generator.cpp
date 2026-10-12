/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load_schedule_case_generator.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "indirect_load_utils.h"
#include "norm_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_utils.h"
#include "schedule_utils.h"
#include "schedule_result.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace {
constexpr int64_t kIndirectLoadSimtDcacheSize = 32 * 1024;
constexpr int64_t kEmbeddingSimdPayloadBytesThreshold = 2048;
constexpr int64_t kEmbeddingSimdLookupCountThreshold = 32;
constexpr int32_t kEmbeddingFastPathScore = 2;
constexpr int32_t kEmbeddingAlternateFastPathScore = 1;
constexpr size_t kIndirectLoadInputCount = 2UL;
constexpr size_t kIndirectLoadOutputCount = 1UL;
constexpr size_t kIndirectLoadInvalidAxisIndex = std::numeric_limits<size_t>::max();
constexpr int64_t kInvalidBroadcastIndex = -1L;
constexpr char kInputInnerAxisName[] = "indirect_load_input_inner";
constexpr char kIndexInnerAxisName[] = "indirect_load_index_inner";
constexpr char kSingleOuterAxisName[] = "indirect_load_single_outer";
constexpr char kOuterAxisName[] = "indirect_load_outer";
constexpr char kInnerAxisName[] = "indirect_load_inner";
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

struct TemplateCase {
  ascir::TemplateId template_id;
  ascgen_utils::indirect_load::Implementation implementation;
};

struct RewrittenGraphAnalysis {
  NodePath input_region;  // SIMD: input 侧链；SIMT 不使用
  NodePath index_region;  // SIMD: index 侧链；SIMT: 融合集（index 链 + 输出链）
  NodePath input_path;
  NodePath index_path;
  af::AscNodePtr input_root;
  af::AscNodePtr input_boundary;
  af::AscNodePtr index_root;
  af::AscNodePtr output_store;
  af::AscNodePtr post_reduce;
  // 多 Reduce 复合区域：全部已配对同轴 Broadcast 的统计 Reduce（≥2 时非空）。
  std::vector<af::AscNodePtr> composite_reduces;
  bool align_input_path = false;
  bool align_index_path = false;
  bool simd_index_uses_output_inner_axis = false;
};

struct InputViewPlan {
  NodePath path;
  af::AscNodePtr load_transpose;
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  int64_t path_broadcast_index = kInvalidBroadcastIndex;
  bool simd_index_uses_output_inner_axis = false;
};

struct PhysicalViewPreparation {
  InputViewPlan input;
  InputViewPlan index;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
};

enum class ReduceAxisKind : uint8_t { kRetained, kReduced, kIgnored };

struct PostReduceLayout {
  std::vector<af::AxisId> axes;
  std::vector<ReduceAxisKind> kinds;
  size_t first_reduce;
};

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

bool HasControlEdge(const af::AscNodePtr &node) {
  return node->GetInControlNodesSize() != 0UL || node->GetOutControlNodesSize() != 0UL;
}

bool IsSingleConsumerWithoutControlEdge(const af::AscNodePtr &node) {
  return node != nullptr && node->GetOutDataNodesSize() == 1UL && !HasControlEdge(node);
}

bool IsBroadcastNode(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Broadcast>(node);
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

template <typename VisitFn>
void TraverseInputProducers(const NodePath &roots, NodeSet &visited, const VisitFn &visit) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second || !visit(node)) {
      continue;
    }
    for (size_t input_index = 0UL; input_index < node->inputs.Size(); ++input_index) {
      pending.emplace_back(ascgen_utils::indirect_load::GetInputProducer(node, input_index));
    }
  }
}

template <typename VisitFn>
af::Status TraverseOutputConsumers(const NodePath &roots, NodeSet &visited, const VisitFn &visit) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    GE_ASSERT_NOTNULL(node, "IndirectLoad output successor is invalid.");
    if (!visited.emplace(node.get()).second) {
      continue;
    }
    bool stop = false;
    GE_ASSERT_SUCCESS(visit(node, stop));
    if (stop) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  return af::SUCCESS;
}

af::AscNodePtr GetLoadTransposeSource(const af::AscNodePtr &node) {
  if (node == nullptr || !af::ops::IsOps<af::ascir_op::Transpose>(node)) {
    return nullptr;
  }
  const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
  return producer != nullptr && af::ops::IsOps<af::ascir_op::Load>(producer) ? producer : nullptr;
}

af::AscNodePtr GetFoldableLoadTransposeSource(const af::AscNodePtr &node) {
  const auto load = GetLoadTransposeSource(node);
  return IsSingleConsumerWithoutControlEdge(node) && IsSingleConsumerWithoutControlEdge(load) ? load : nullptr;
}

bool IsSupportedIndirectLoadTopologyNode(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (ScheduleUtils::IsIOBuffer(node) || ScheduleUtils::IsBuffer(node) ||
      af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
    return true;
  }
  if (GetLoadTransposeSource(node) != nullptr) {
    return true;
  }
  return ScheduleUtils::IsLoad(node) || ScheduleUtils::IsStore(node) || ScheduleUtils::IsElewise(node) ||
         ScheduleUtils::IsBroadcast(node) || ScheduleUtils::IsReduce(node);
}

af::Status CollectAndValidateIndirectLoadNodes(const ascir::HintGraph &graph, af::AscNodePtr &indirect_load) {
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }
  af::AscNodePtr unsupported_node;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (!IsSupportedIndirectLoadTopologyNode(node)) {
      unsupported_node = node;
      break;
    }
  }
  if (unsupported_node != nullptr) {
    GELOGE(af::FAILED,
           "[IndirectLoad] Graph[%s] node[%s] contains unsupported topology operator[%s, compute_type=%u] "
           "around IndirectLoad[%s].",
           graph.GetName().c_str(), unsupported_node->GetNamePtr(), unsupported_node->GetTypePtr(),
           static_cast<uint32_t>(unsupported_node->attr.api.compute_type), indirect_load->GetNamePtr());
    return af::FAILED;
  }
  return af::SUCCESS;
}

void CollectInputRegionMembers(const af::AscNodePtr &indirect_load, size_t input_index, NodeSet &region) {
  const af::AscNodePtr root = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
  if (root == nullptr) {
    return;
  }
  TraverseInputProducers({root}, region,
                         [](const af::AscNodePtr &node) { return !af::ops::IsOps<af::ascir_op::Load>(node); });
}

bool CollectSimtBackwardRegion(const NodePath &roots, const af::AscNodePtr &indirect_load, NodeSet &region) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr) {
      return false;
    }
    if (node == indirect_load || IsInputDataSource(node)) {
      continue;
    }
    if (!region.emplace(node.get()).second || af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    if (node->inputs.Size() == 0UL && !af::ops::IsOps<af::ascir_op::Arange>(node)) {
      return false;
    }
    if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
      continue;
    }
    for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(node, i);
      if (producer == nullptr) {
        return false;
      }
      pending.emplace_back(producer);
    }
  }
  return true;
}

void CollectSimtOutputConsumerClosure(const af::AscNodePtr &indirect_load, NodeSet &closure) {
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if (consumer != nullptr) {
      roots.emplace_back(consumer);
    }
  }
  const auto visit = [](const af::AscNodePtr &node, bool &stop) -> af::Status {
    (void)node;
    (void)stop;
    return af::SUCCESS;
  };
  (void)TraverseOutputConsumers(roots, closure, visit);
}

// [index 链外部 store] index 生产链尾部张量的区域外直连 store 会被 lowering 收为
// 额外输出链（链值 = Index() 求值器结果），此处将其纳入 SIMT 融合区域（获得
// kSimtDirectGmBoundary 角色，跳过普通调度发射）；index 链其余无法供值的外部消费者
// 静默丢弃会导致目标 buffer 永不写入，必须整候选淘汰。合法消费者集 = IndirectLoad
// 自身 ∪ 融合区域（index 链与输出链的 backward 闭包——含经多输入算子合流进输出链
// 的 side-input，如 embedding+reduce 图中消费 index load 的 Ge/Select）∪ IL 输出的
// 前向闭包（post-Reduce 普通调度链）；守卫只扫 index 链子集，不扫全融合区域。
bool FoldSimtIndexChainExternalStores(const af::AscNodePtr &indirect_load, const af::AscNodePtr &index_root,
                                      NodeSet &region) {
  NodeSet index_chain;
  if (!CollectSimtBackwardRegion({index_root}, indirect_load, index_chain)) {
    return false;
  }
  NodeSet output_consumers;
  CollectSimtOutputConsumerClosure(indirect_load, output_consumers);
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    return false;
  }
  const auto inputs = indirect_load->inputs();
  if (inputs.size() <= ascgen_utils::indirect_load::kIndexTensorIndex) {
    return false;
  }
  const ascir::TensorId index_tensor_id = inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.mem.tensor_id;
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    if (node == nullptr || index_chain.count(node.get()) == 0UL) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (consumer == nullptr || consumer == indirect_load || region.count(consumer.get()) != 0UL ||
          output_consumers.count(consumer.get()) != 0UL) {
        continue;
      }
      if (node == index_root && af::ops::IsOps<af::ascir_op::Store>(consumer) && !consumer->inputs().empty() &&
          consumer->inputs()[0]->attr.mem.tensor_id == index_tensor_id) {
        region.emplace(consumer.get());
        continue;
      }
      GELOGI("[IndirectLoad] Reject SIMT candidate: index chain node[%s] has unsupported external consumer[%s, %s].",
             node->GetNamePtr(), consumer->GetNamePtr(), consumer->GetTypePtr());
      return false;
    }
  }
  return true;
}

void CollectSimtFusedRegionMembers(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                   NodeSet &region) {
  region.clear();
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  const af::AscNodePtr output_terminal = analysis.post_reduce == nullptr
                                             ? analysis.output_store
                                             : ascgen_utils::indirect_load::GetInputProducer(analysis.post_reduce, 0UL);
  if (index_root == nullptr || output_terminal == nullptr) {
    return;
  }
  if (!CollectSimtBackwardRegion({index_root, output_terminal}, indirect_load, region)) {
    region.clear();
    return;
  }
  if (!FoldSimtIndexChainExternalStores(indirect_load, index_root, region)) {
    region.clear();
  }
}

// Normalize the IndirectLoad axis to a non-negative output axis index. Return the sentinel for invalid metadata.
size_t GetIndirectLoadAxisIndex(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto outputs = node->outputs();
  if (outputs.empty() || outputs[0] == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  if (node->attr.ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto *ir_attr = node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  if (ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  int64_t axis = 0L;
  if (ir_attr->GetAxis(axis) != af::SUCCESS) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const size_t rank = outputs[0]->attr.axis.size();
  if (rank > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const int64_t rank_value = static_cast<int64_t>(rank);
  if (axis < -rank_value || axis >= rank_value) {
    return kIndirectLoadInvalidAxisIndex;
  }
  return static_cast<size_t>(axis < 0L ? axis + rank_value : axis);
}

af::Status BuildPostReduceLayout(const af::AscNodePtr &reduce, PostReduceLayout &layout, bool &is_legal) {
  is_legal = false;
  GE_ASSERT_TRUE(reduce->inputs().size() == 1UL && reduce->outputs().size() == 1UL,
                 "IndirectLoad post Reduce must be unary, node[%s] input num:%zu, output num:%zu.",
                 reduce->GetNamePtr(), reduce->inputs().size(), reduce->outputs().size());
  const auto input = reduce->inputs()[0];
  const auto output = reduce->outputs()[0];
  GE_ASSERT_NOTNULL(input);
  GE_ASSERT_NOTNULL(output);
  const auto &input_strides = input->attr.strides;
  const auto &output_strides = output->attr.strides;
  layout.axes = output->attr.axis;
  GE_ASSERT_TRUE(input->attr.axis.size() == layout.axes.size() && input->attr.repeats.size() == layout.axes.size() &&
                     output->attr.repeats.size() == layout.axes.size() &&
                     input_strides.size() == output_strides.size() && output_strides.size() == layout.axes.size(),
                 "IndirectLoad post Reduce metadata rank mismatch.");
  const auto reduce_axes = ScheduleUtils::CalcReduceAxes(input_strides, output_strides, layout.axes);
  layout.kinds.clear();
  layout.kinds.reserve(layout.axes.size());
  layout.first_reduce = layout.axes.size();
  for (size_t i = 0UL; i < layout.axes.size(); ++i) {
    const auto input_zero = af::SymbolicUtils::StaticCheckEq(input_strides[i], af::ops::Zero);
    const auto output_zero = af::SymbolicUtils::StaticCheckEq(output_strides[i], af::ops::Zero);
    // 仅拒绝可静态证明矛盾的布局（输入广播 stride=0 但输出非零）。符号 stride 的
    // 零性为 kUnknown 不能作为非法依据：动态 shape 下保留轴 stride 恒为符号表达式，
    // 归约/保留划分由 CalcReduceAxes 依据 dst stride 是否为常量 0 推导。
    if (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kFalse) {
      return af::SUCCESS;
    }
    // 归约轴输出 repeat 恒为 1 且 stride 恒为 0。若某轴输出 repeat=1 但其
    // 输出 stride 的零性无法静态判定，则无法确认它是否为归约轴（可能被改写为
    // 非法布局），保守拒绝；真正保留轴的 repeat 不会退化为常量 1。
    if (af::SymbolicUtils::StaticCheckEq(output->attr.repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        output_zero == af::TriBool::kUnknown) {
      return af::SUCCESS;
    }
    if (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kTrue) {
      layout.kinds.emplace_back(ReduceAxisKind::kIgnored);
    } else if (std::find(reduce_axes.begin(), reduce_axes.end(), layout.axes[i]) != reduce_axes.end()) {
      layout.kinds.emplace_back(ReduceAxisKind::kReduced);
      if (layout.first_reduce == layout.axes.size()) {
        layout.first_reduce = i;
      }
    } else {
      layout.kinds.emplace_back(ReduceAxisKind::kRetained);
    }
  }
  is_legal = true;
  return af::SUCCESS;
}

bool HasSupportedReduceSuffix(const PostReduceLayout &layout, size_t begin) {
  bool has_axis = false;
  bool has_reduce_axis = false;
  bool previous_is_reduce = false;
  size_t transitions = 0UL;
  for (size_t i = begin; i < layout.kinds.size(); ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kIgnored) {
      continue;
    }
    const bool is_reduce = layout.kinds[i] == ReduceAxisKind::kReduced;
    if (has_axis && is_reduce != previous_is_reduce) {
      ++transitions;
    }
    has_axis = true;
    has_reduce_axis = has_reduce_axis || is_reduce;
    previous_is_reduce = is_reduce;
  }
  return has_reduce_axis && transitions <= 1UL;
}

af::Status ValidateSimtPostReduceLayout(const af::AscNodePtr &reduce, size_t &boundary, bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  if (layout.first_reduce == layout.axes.size()) {
    is_legal = false;
    return af::SUCCESS;
  }
  boundary = layout.first_reduce;
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

af::Status IsSimtArPostReduce(const af::AscNodePtr &reduce, bool &is_ar) {
  is_ar = false;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  bool is_legal = false;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal || layout.first_reduce == layout.axes.size()) {
    return af::SUCCESS;
  }
  is_ar = std::none_of(layout.kinds.begin() + static_cast<int64_t>(layout.first_reduce), layout.kinds.end(),
                       [](ReduceAxisKind kind) { return kind == ReduceAxisKind::kRetained; });
  return af::SUCCESS;
}

af::Status ValidateSimdPostReduceLayout(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce,
                                        bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const size_t boundary = axis_index;
  GE_ASSERT_TRUE(boundary <= layout.axes.size(),
                 "IndirectLoad axis %zu is out of range [0, %zu] for post Reduce output.", boundary,
                 layout.axes.size());

  for (size_t i = 0UL; i < boundary; ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kReduced) {
      is_legal = false;
      return af::SUCCESS;
    }
  }
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

bool HasReduceBeforeGatherAxis(const af::AscNodePtr &indirect_load) {
  const auto reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
  if (reduce == nullptr) {
    return false;
  }
  PostReduceLayout layout;
  bool is_legal = false;
  if (BuildPostReduceLayout(reduce, layout, is_legal) != af::SUCCESS || !is_legal) {
    return false;
  }
  const size_t gather_axis = GetIndirectLoadAxisIndex(indirect_load);
  if (gather_axis == kIndirectLoadInvalidAxisIndex || gather_axis >= layout.kinds.size()) {
    return false;
  }
  for (size_t index = 0UL; index < layout.kinds.size(); ++index) {
    if (layout.kinds[index] == ReduceAxisKind::kReduced) {
      return index < gather_axis;
    }
  }
  return false;
}

// 归约轴推导：复用 Reduce 布局分析的真实 zero 转变规则，不能以任意
// stride 差作为归约依据；保留轴在重排 View 下也可能改变 stride。
std::vector<af::AxisId> CalcReduceAxesFromNode(const af::AscNodePtr &reduce) {
  std::vector<af::AxisId> reduced_axes;
  const auto inputs = reduce->inputs();
  const auto outputs = reduce->outputs();
  if (inputs.empty() || outputs.empty() || inputs.size() != 1UL || outputs.size() != 1UL) {
    return reduced_axes;
  }
  const auto &input_attr = inputs[0]->attr;
  const auto &output_attr = outputs[0]->attr;
  if (input_attr.axis.size() != output_attr.axis.size() || input_attr.strides.size() != output_attr.strides.size() ||
      input_attr.axis.size() != input_attr.strides.size()) {
    return reduced_axes;
  }
  const auto axes = ScheduleUtils::CalcReduceAxes(input_attr.strides, output_attr.strides, output_attr.axis);
  for (const auto axis : axes) {
    reduced_axes.emplace_back(axis);
  }
  return reduced_axes;
}

// 广播恢复轴推导：Broadcast 输入 stride 为 0 的轴即广播恢复轴。
std::vector<af::AxisId> CalcBroadcastAxesFromNode(const af::AscNodePtr &broadcast) {
  std::vector<af::AxisId> broadcast_axes;
  const auto inputs = broadcast->inputs();
  const auto outputs = broadcast->outputs();
  if (inputs.empty() || outputs.empty() || inputs.size() != 1UL || outputs.size() != 1UL) {
    return broadcast_axes;
  }
  const auto &input_attr = inputs[0]->attr;
  const auto &output_attr = outputs[0]->attr;
  if (input_attr.axis.size() != output_attr.axis.size() || input_attr.strides.size() != output_attr.strides.size() ||
      input_attr.axis.size() != input_attr.strides.size()) {
    return broadcast_axes;
  }
  for (size_t i = 0UL; i < input_attr.axis.size(); ++i) {
    const bool zero_stride =
        af::SymbolicUtils::StaticCheckEq(input_attr.strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue;
    if (zero_stride) {
      broadcast_axes.emplace_back(output_attr.axis[i]);
    }
  }
  return broadcast_axes;
}

// Reduce → Broadcast 同轴判定：广播恢复轴集合与归约轴集合完全一致（按轴 ID 集合比较）。
bool IsReduceBroadcastCoaxial(const std::vector<af::AxisId> &reduced_axes,
                              const std::vector<af::AxisId> &broadcast_axes) {
  if (reduced_axes.size() != broadcast_axes.size() || reduced_axes.empty()) {
    return false;
  }
  std::set<af::AxisId> reduced_set(reduced_axes.begin(), reduced_axes.end());
  std::set<af::AxisId> broadcast_set(broadcast_axes.begin(), broadcast_axes.end());
  return reduced_set == broadcast_set;
}

// 从 Reduce 出发沿唯一输出链查找对应 Broadcast：Reduce 输出直接接
// Broadcast，或经 Cast 后接 Broadcast。返回空表示无对应 Broadcast。
af::AscNodePtr FindPairedBroadcast(const af::AscNodePtr &reduce) {
  if (reduce == nullptr) {
    GELOGI("[IndirectLoad] Reduce/Broadcast pairing skipped: Reduce is null.");
    return nullptr;
  }
  NodePath pending{reduce};
  NodeSet visited;
  af::AscNodePtr matched;
  for (size_t cursor = 0UL; cursor < pending.size() && cursor < 64UL; ++cursor) {
    const auto current = pending[cursor];
    if (current == nullptr || !visited.emplace(current.get()).second || current->GetOutDataNodesSize() == 0UL) {
      continue;
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      const auto successor = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (successor == nullptr || HasControlEdge(successor) || ScheduleUtils::IsReduce(successor)) {
        continue;
      }
      if (IsBroadcastNode(successor)) {
        if (matched != nullptr && matched != successor) {
          GELOGD("[IndirectLoad] Reduce[%s] pairing is ambiguous.", reduce->GetNamePtr());
          return nullptr;
        }
        matched = successor;
        continue;
      }
      // 图优化可能将 Broadcast 融入二元 Elementwise 的输入关系，允许继续
      // 遍历其消费者，但不跨越其他 Reduce 或控制边。
      if (ScheduleUtils::IsElewise(successor)) {
        pending.emplace_back(successor);
      }
    }
  }
  if (matched != nullptr) {
    GELOGD("[IndirectLoad] Reduce[%s] paired with Broadcast[%s].", reduce->GetNamePtr(), matched->GetNamePtr());
  }
  return matched;
}

// 检查 Reduce 的广播配对：每个统计 Reduce 必须有唯一 Broadcast 恢复归约轴且二者同轴。
bool HasPairedBroadcastForReduce(const af::AscNodePtr &reduce) {
  const auto broadcast = FindPairedBroadcast(reduce);
  if (broadcast == nullptr) {
    return false;
  }
  return IsReduceBroadcastCoaxial(CalcReduceAxesFromNode(reduce), CalcBroadcastAxesFromNode(broadcast));
}

bool IsFinalReduce(const af::AscNodePtr &reduce, const af::AscNodePtr &output_store) {
  // 只有 Reduce 直接收敛到最终 Store，或经单个 Cast 收敛到 Store，才允许省略 Broadcast。
  if (reduce == nullptr || output_store == nullptr || reduce->GetOutDataNodesSize() != 1UL) {
    return false;
  }
  const auto successor = std::dynamic_pointer_cast<af::AscNode>(*reduce->GetOutDataNodes().begin());
  return successor == output_store ||
         (af::ops::IsOps<af::ascir_op::Cast>(successor) && successor->GetOutDataNodesSize() == 1UL &&
          *successor->GetOutDataNodes().begin() == output_store);
}

bool HasSameLogicalAxisView(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return lhs.axis == rhs.axis && PassUtils::IsExprVectorEqual(lhs.repeats, rhs.repeats);
}

bool ValidateCompositeRegionOutputView(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis) {
  const auto input_outputs = indirect_load->outputs();
  if (input_outputs.empty() || input_outputs[0] == nullptr) {
    return false;
  }
  const auto &entry_attr = input_outputs[0]->attr;
  const af::AscNodePtr exit_node = analysis.output_store == nullptr ? nullptr : analysis.output_store;
  if (exit_node == nullptr || exit_node->inputs.Size() == 0UL) {
    return false;
  }
  if (analysis.composite_reduces.empty()) {
    return HasSameLogicalAxisView(entry_attr, exit_node->inputs[0].attr);
  }
  return HasSameLogicalAxisView(entry_attr, exit_node->inputs[0].attr);
}

// Softmax 专用路径的 IO 形态校验：单输入单输出且输入输出 View 与
// IndirectLoad 输出一致，保证 Gather 输出可直接作为 Softmax 的 A/R View。
// 校验通过时通过出参返回输入输出属性，供后续元数据构造使用。
bool TryValidateSoftmaxDedicatedIo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &softmax,
                                   af::AscTensorAttr &input_attr, af::AscTensorAttr &output_attr) {
  const auto softmax_inputs = softmax->inputs();
  const auto softmax_outputs = softmax->outputs();
  const auto indirect_outputs = indirect_load->outputs();
  if (softmax_inputs.empty() || softmax_outputs.empty() || indirect_outputs.empty() || softmax_inputs.size() != 1UL ||
      softmax_outputs.size() != 1UL) {
    GELOGI("[IndirectLoad] Softmax dedicated path rejected: invalid IO, node[%s].", softmax->GetNamePtr());
    return false;
  }
  input_attr = softmax_inputs[0]->attr;
  output_attr = softmax_outputs[0]->attr;
  const auto &indirect_output_attr = indirect_outputs[0]->attr;
  const bool view_consistent = input_attr.axis == indirect_output_attr.axis &&
                               output_attr.axis == indirect_output_attr.axis &&
                               PassUtils::IsExprVectorEqual(input_attr.repeats, indirect_output_attr.repeats) &&
                               PassUtils::IsExprVectorEqual(output_attr.repeats, indirect_output_attr.repeats);
  if (!view_consistent) {
    GELOGI("[IndirectLoad] Softmax dedicated path rejected: view mismatch, node[%s].", softmax->GetNamePtr());
    return false;
  }
  return true;
}

// G ≥ R 布局判定（SIMD 直接融合必要条件）：Softmax 归约轴是输入尾轴，
// 尾轴位置必须在 Gather G 轴位置或内侧；否则由 SIMT/SK 衔接式候选接管。
bool IsSoftmaxReduceInsideOrAtGatherAxis(const af::AscNodePtr &indirect_load, const af::AscTensorAttr &input_attr,
                                         ascir::TemplateId template_id) {
  // 非尾轴 gather（axis < rank-1）时 SIMD 输出 tile 的物理布局带行对齐空洞（行
  // stride 按 2x inner 对齐），SoftmaxARFullLoadExtend 的 {a, r} 隐含 src/dst 稠密
  // 布局假设——读 src 错位、dst 按 a*align32(r) 组织会超出 buffer 分配（生产
  // gather+layernorm 用例的 AIC 341 UB 越界即此组合）。因此仅 SIMD 模板要求 gather
  // 轴为输出尾轴；SIMT 逐元素发射、输出稠密无空洞，非尾轴 gather 同样安全（生产
  // softmax_abs_gather 图 axis=0 的 SIMT 候选不应被拒）。
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  if (axis_index == kIndirectLoadInvalidAxisIndex || input_attr.axis.empty()) {
    return false;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return axis_index == input_attr.axis.size() - 1UL;
  }
  return axis_index <= input_attr.axis.size() - 1UL;
}

// 保存 Softmax 专用路径 NormInfo：归约轴为输入尾轴（尾轴约束已在分流时验证）。
af::Status SaveSoftmaxDedicatedNormInfo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &softmax,
                                        const af::AscTensorAttr &input_attr) {
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  info.entry_node_name = softmax->GetName();
  info.exit_node_name = softmax->GetName();
  info.region_node_names = {softmax->GetName()};
  info.entry_axes = input_attr.axis;
  info.softmax_reduce_axis = input_attr.axis.empty() ? af::kIdNone : input_attr.axis.back();
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  GELOGI("[IndirectLoad] NormInfo saved: Softmax dedicated path, node[%s].", softmax->GetNamePtr());
  return af::SUCCESS;
}

// 保存多 stage 复合区域 NormInfo：每个 Reduce 与配对 Broadcast、归约轴
// 共同写入；出口必须存在且 View 与入口一致，否则淘汰当前候选。
af::Status SaveCompositeNormInfo(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (analysis.output_store == nullptr || !ValidateCompositeRegionOutputView(indirect_load, analysis)) {
    GELOGI("[IndirectLoad] Composite Norm rejected: entry and final output views differ, node[%s].",
           indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  const auto indirect_outputs = indirect_load->outputs();
  if (!indirect_outputs.empty() && indirect_outputs[0] != nullptr) {
    info.entry_axes = indirect_outputs[0]->attr.axis;
  }
  for (const auto &reduce : analysis.composite_reduces) {
    const auto broadcast = FindPairedBroadcast(reduce);
    ascgen_utils::norm::NormStage stage;
    stage.reduce_node_name = reduce->GetName();
    stage.broadcast_node_name = broadcast == nullptr ? "" : broadcast->GetName();
    stage.reduced_axes = CalcReduceAxesFromNode(reduce);
    info.stages.emplace_back(std::move(stage));
    info.region_node_names.emplace_back(reduce->GetName());
    if (broadcast != nullptr) {
      info.region_node_names.emplace_back(broadcast->GetName());
    }
  }
  info.entry_node_name = analysis.post_reduce->GetName();
  info.exit_node_name = analysis.output_store->GetName();
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  GELOGI("[IndirectLoad] NormInfo saved: composite region with %zu stages.", info.stages.size());
  is_candidate_legal = true;
  return af::SUCCESS;
}

// 保存单 Reduce NormInfo：归约轴由布局分析推导，作为区域元数据的最小记录。
af::Status SaveSingleReduceNormInfo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce) {
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  info.entry_node_name = reduce->GetName();
  info.exit_node_name = reduce->GetName();
  info.region_node_names = {reduce->GetName()};
  PostReduceLayout layout;
  bool layout_legal = false;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, layout_legal));
  if (layout_legal) {
    info.entry_axes = layout.axes;
    for (size_t i = 0UL; i < layout.kinds.size(); ++i) {
      if (layout.kinds[i] == ReduceAxisKind::kReduced) {
        info.stages.push_back({reduce->GetName(), "", {layout.axes[i]}});
      }
    }
  }
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  return af::SUCCESS;
}

// 识别 Gather 后置链中的 Norm 区域并保存元数据：
// - 专用 Softmax：尾轴约束 + IO View + G≥R 校验；
// - 多 Reduce 复合区域：配对同轴校验（收集阶段）+ 出口 View 校验；
// - 单 Reduce：布局推导记录。
af::Status CollectGatherNormInfo(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal, ascir::TemplateId template_id) {
  is_candidate_legal = true;
  if (indirect_load == nullptr || analysis.post_reduce == nullptr) {
    return af::SUCCESS;
  }

  if (af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce)) {
    af::AscTensorAttr softmax_input_attr;
    af::AscTensorAttr softmax_output_attr;
    if (!TryValidateSoftmaxDedicatedIo(indirect_load, analysis.post_reduce, softmax_input_attr, softmax_output_attr) ||
        !IsSoftmaxReduceInsideOrAtGatherAxis(indirect_load, softmax_input_attr, template_id)) {
      is_candidate_legal = false;
      return af::SUCCESS;
    }
    return SaveSoftmaxDedicatedNormInfo(indirect_load, analysis.post_reduce, softmax_input_attr);
  }

  if (!analysis.composite_reduces.empty()) {
    return SaveCompositeNormInfo(indirect_load, analysis, is_candidate_legal);
  }
  return SaveSingleReduceNormInfo(indirect_load, analysis.post_reduce);
}

bool HasBroadcastMultiInputNode(const af::AscNodePtr &node, NodeSet &visited) {
  if (node == nullptr || !visited.emplace(node.get()).second) {
    return false;
  }
  if (node->inputs.Size() > 1UL) {
    for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
      const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, input_idx);
      if (producer != nullptr && af::ops::IsOps<af::ascir_op::Broadcast>(producer)) {
        return true;
      }
    }
  }
  for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(node, input_idx), visited)) {
      return true;
    }
  }
  return false;
}

bool IsSkTemplateCandidateLegal(const af::AscNodePtr &indirect_load) {
  // SK 仅支持单输出消费者：多消费者分区（为多 Reduce 复合区域引入）
  // 已随 LayerNorm SK 支持一同回退，待后续补齐后再放开。
  if (indirect_load == nullptr || indirect_load->GetOutDataNodesSize() != 1UL) {
    GELOGI("[IndirectLoad] Reject SK: IndirectLoad output consumer count is not 1.");
    return false;
  }
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
      GELOGI("[IndirectLoad] Reject SK: input[%zu] has no connected producer.", input_idx);
      return false;
    }
  }
  NodeSet visited;
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx), visited)) {
      GELOGI("[IndirectLoad] Reject SK: input[%zu] contains multi-input Broadcast dependency.", input_idx);
      return false;
    }
  }
  const auto output_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load);
  if (HasBroadcastMultiInputNode(output_consumer, visited)) {
    return false;
  }
  return true;
}

af::Status ValidateIndirectLoadNode(const af::AscNodePtr &indirect_load) {
  const auto outputs = indirect_load->outputs();
  GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad output number:%zu, node[%s].",
                 outputs.size(), indirect_load->GetNamePtr());
  const auto output = outputs[0];
  GE_ASSERT_NOTNULL(output, "IndirectLoad output tensor is null.");
  const size_t output_rank = output->attr.axis.size();
  // 注意：AscNodeInputs/AscNodeOutputs 的 operator() 每次调用都会重建内部快照，先前调用
  // 返回的 tensor 指针会悬垂。因此这里只调用一次 outputs()，并提前拷贝后续要用的数据。
  const auto output_repeats = output->attr.repeats;
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const auto inputs = indirect_load->inputs();
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad input number:%zu, node[%s].",
                 inputs.size(), indirect_load->GetNamePtr());
  const auto input = inputs[ascgen_utils::indirect_load::kInputTensorIndex];
  const auto index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex];
  GE_ASSERT_NOTNULL(input, "IndirectLoad input tensor is null.");
  GE_ASSERT_NOTNULL(index, "IndirectLoad index tensor is null.");
  GE_ASSERT_TRUE(input->attr.repeats.size() == output_rank && index->attr.repeats.size() == output_rank &&
                     output_repeats.size() == output_rank,
                 "IndirectLoad logical shape rank is invalid.");
  for (size_t i = 0UL; i < output_rank; ++i) {
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(index->attr.repeats[i], output_repeats[i]) == af::TriBool::kTrue,
                   "IndirectLoad index and output logical shape must match.");
    if (i != axis_index) {
      GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckLt(input->attr.repeats[i], output_repeats[i]) != af::TriBool::kTrue,
                     "IndirectLoad input dimension %zu must not be smaller than index/output outside axis %zu.", i,
                     axis_index);
    }
  }
  return af::SUCCESS;
}

af::Status MergeAxesForTemplate(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const std::string &name,
                                af::AxisId &merged_axis) {
  GE_ASSERT_TRUE(!axes.empty(), "IndirectLoad merge axis source is empty, name:%s.", name.c_str());
  merged_axis = axes.size() == 1UL ? axes.front() : graph.MergeAxis(axes, name)->id;
  return af::SUCCESS;
}

af::Status CreateFixedTileSplit(af::AscGraph &graph, af::AxisId axis_id, af::AxisId &outer_id, af::AxisId &inner_id) {
  const auto *axis = graph.FindAxis(axis_id);
  GE_ASSERT_NOTNULL(axis, "IndirectLoad fixed tile axis %ld is not found.", axis_id);
  const af::Expression outer_size = axis->size;
  const af::Expression inner_size = af::sym::kSymbolOne;
  outer_id =
      graph.CreateAxis(axis->name + "T", ascir::Axis::Type::kAxisTypeTileOuter, outer_size, {axis_id}, af::kIdNone).id;
  inner_id =
      graph.CreateAxis(axis->name + "t", ascir::Axis::Type::kAxisTypeTileInner, inner_size, {axis_id}, outer_id).id;
  auto *outer_axis = graph.FindAxis(outer_id);
  GE_ASSERT_NOTNULL(outer_axis, "IndirectLoad fixed tile outer axis %ld is not found.", outer_id);
  outer_axis->split_pair_other_id = inner_id;
  return af::SUCCESS;
}

af::Status CopyBoundaryTensorAttr(const af::AscNodePtr &src_node, size_t src_output_idx,
                                  const af::AscNodePtr &dst_node) {
  GE_ASSERT_TRUE(src_output_idx < src_node->outputs().size(),
                 "IndirectLoad SK boundary output index %zu is out of range for node[%s].", src_output_idx,
                 src_node->GetNamePtr());
  GE_ASSERT_TRUE(!dst_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 dst_node->GetNamePtr());
  auto dst_op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(dst_op_desc);
  auto dst_node_attr = dst_op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_ASSERT_NOTNULL(dst_node_attr);
  if (src_node_attr != nullptr) {
    dst_node_attr->sched = src_node_attr->sched;
    if (src_node_attr->ir_attr != nullptr) {
      dst_node_attr->ir_attr = src_node_attr->ir_attr->Clone();
    }
  }
  auto output_desc = dst_op_desc->MutableOutputDesc(0UL);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  *output_attr = src_node->outputs()[src_output_idx]->attr;
  return af::SUCCESS;
}

af::Status CopyWorkspaceTensorAttr(const af::AscNodePtr &boundary_node, const af::AscNodePtr &workspace_node) {
  GE_ASSERT_TRUE(!boundary_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 boundary_node->GetNamePtr());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty(), "IndirectLoad SK workspace node[%s] has no output.",
                 workspace_node->GetNamePtr());
  workspace_node->outputs()[0]->attr = boundary_node->outputs()[0]->attr;
  return af::SUCCESS;
}

af::Status InsertWorkspaceBoundary(af::AscGraph &graph, const af::AscNodePtr &src_node, size_t src_output_idx,
                                   const af::AscNodePtr &dst_node, size_t dst_input_idx,
                                   const std::string &boundary_name, bool align_store, bool align_load) {
  const auto src_anchor = src_node->GetOutDataAnchor(src_output_idx);
  const auto dst_anchor = dst_node->GetInDataAnchor(dst_input_idx);
  GE_ASSERT_NOTNULL(src_anchor, "IndirectLoad SK source anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_NOTNULL(dst_anchor, "IndirectLoad SK destination anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_TRUE(dst_anchor->GetPeerOutAnchor() == src_anchor,
                 "IndirectLoad SK boundary[%s] does not match edge %s:%zu -> %s:%zu.", boundary_name.c_str(),
                 src_node->GetNamePtr(), src_output_idx, dst_node->GetNamePtr(), dst_input_idx);

  const std::string workspace_name = boundary_name + "_workspace";
  af::ascir_op::Workspace workspace_pre(workspace_name.c_str());
  af::ascir_op::Workspace workspace_post(workspace_name.c_str());
  af::ascir_op::Load load((boundary_name + "_load").c_str());
  af::ascir_op::Store store((boundary_name + "_store").c_str());
  auto workspace_pre_node = graph.AddNode(workspace_pre);
  auto workspace_post_node = graph.AddNode(workspace_post);
  auto load_node = graph.AddNode(load);
  auto store_node = graph.AddNode(store);
  GE_ASSERT_NOTNULL(workspace_pre_node);
  GE_ASSERT_NOTNULL(workspace_post_node);
  GE_ASSERT_NOTNULL(load_node);
  GE_ASSERT_NOTNULL(store_node);

  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, load_node));
  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, store_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(load_node, workspace_post_node));
  if (align_store) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        store_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }
  if (align_load) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        load_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(src_anchor, dst_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(src_anchor, store_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(store_node->GetOutDataAnchor(0UL), workspace_pre_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(workspace_post_node->GetOutDataAnchor(0UL), load_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0UL), dst_anchor));
  return af::SUCCESS;
}

af::Status PartitionSkGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool align_input_path,
                            bool align_index_path, bool skip_input_boundaries = false) {
  // [SK段SIMT化] 窗口超限形态：input/index 不插 workspace 边界（IL 段以 SIMT 语义
  // 从 GM 直读原输入），仅保留 output 边界连接消费子图（Reduce/Norm 段独立调度，
  // 承载 R 轴分 tile 归约）。
  if (!skip_input_boundaries) {
    for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
      const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
      GE_ASSERT_NOTNULL(input_anchor);
      const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
      GE_ASSERT_NOTNULL(producer);
      const std::string role = input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? "input" : "index";
      const bool align_path =
          input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? align_input_path : align_index_path;
      GE_ASSERT_SUCCESS(
          InsertWorkspaceBoundary(graph, producer, static_cast<size_t>(peer_out_anchor->GetIdx()), indirect_load,
                                  input_idx, indirect_load->GetName() + "_sk_" + role, align_path,
                                  input_idx == ascgen_utils::indirect_load::kIndexTensorIndex && align_path));
    }

  }  // skip_input_boundaries
  const auto output_anchor = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(output_anchor);
  const auto peer_input_anchors = output_anchor->GetPeerInDataAnchors();
  // [SK段SIMT化] IL 输出可能多消费者（LayerNorm 的 x 与 x-mean 菱形：IL 直连
  // Reduce 与 Elementwise 分支，SIMT 流程的前置改写会折叠过渡 Cast）。逐消费者
  // 边独立边界化：IL 输出写 N 份 workspace，各消费者从对应 workspace 读取——
  // 语义等价（数据相同），不要求单消费者。
  const auto &il_output_anchors = peer_input_anchors;
  GE_ASSERT_TRUE(!il_output_anchors.empty(), "IndirectLoad SK requires at least one output consumer, node[%s].",
                 indirect_load->GetNamePtr());
  // [SK段SIMT化] IL 输出可能多消费者（LayerNorm 的 x 与 x-mean 菱形）。单份
  // workspace：首个消费者走完整边界（IL→store→workspace，workspace→load→consumer），
  // 其余消费者的输入边改接到该 load 输出——共享同一份数据，不产生多副本 load。
  // [SK段SIMT化] 多消费者时单份产出、多份读取：SIMT body 的 OutputTargets 仅支持
  // 单个 GM 输出（kGmOutputCount=1），IL 段只建一个 store/pre（写一份 workspace）；
  // 每个消费段各建独立的 post/load（同名 workspace 读同一 GM buffer，post 之间无
  // 边），连通性分量互不连通。单消费者时退化为标准 InsertWorkspaceBoundary。
  const std::string boundary_name = indirect_load->GetName() + "_sk_output";
  const auto first_peer = *il_output_anchors.begin();
  GE_ASSERT_NOTNULL(first_peer);
  auto first_consumer = std::dynamic_pointer_cast<af::AscNode>(first_peer->GetOwnerNode());
  GE_ASSERT_NOTNULL(first_consumer);
  GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, indirect_load, 0UL, first_consumer,
                                            static_cast<size_t>(first_peer->GetIdx()), boundary_name, false, false));
  if (il_output_anchors.size() > 1UL) {
    const std::string workspace_name = boundary_name + "_workspace";
    std::vector<af::InDataAnchorPtr> extra_peers;
    for (const auto &peer_input_anchor : il_output_anchors) {
      if (peer_input_anchor != first_peer && peer_input_anchor != nullptr) {
        extra_peers.emplace_back(peer_input_anchor);
      }
    }
    for (const auto &peer_input_anchor : extra_peers) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(output_anchor, peer_input_anchor));
    }
    size_t load_seq = 1UL;
    for (const auto &peer_input_anchor : extra_peers) {
      af::ascir_op::Workspace workspace_post(workspace_name.c_str());
      af::ascir_op::Load load((boundary_name + "_load" + std::to_string(load_seq)).c_str());
      const auto post_node = graph.AddNode(workspace_post);
      const auto load_node = graph.AddNode(load);
      GE_ASSERT_NOTNULL(post_node);
      GE_ASSERT_NOTNULL(load_node);
      // load 拷贝产出侧视图。post 与 InsertWorkspaceBoundary 保持一致：不拷贝
      // （Workspace 节点无视图/tensor_id 语义，拷贝源 tensor_id 会污染 workspace
      // 大小收集体系，导致各段偏移分配退化为 0）。
      GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(indirect_load, 0UL, load_node));
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(post_node->GetOutDataAnchor(0UL), load_node->GetInDataAnchor(0UL)));
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0UL), peer_input_anchor));
      ++load_seq;
    }
  }
  return af::SUCCESS;
}

af::Status BuildSkPartitionOrder(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load,
                                 std::vector<af::AscNodePtr> &node_order) {
  std::set<af::NodePtr> ordered_nodes;
  for (const char *role : {"input", "index", "output"}) {
    const std::string workspace_name = indirect_load->GetName() + "_sk_" + role + "_workspace";
    af::AscNodePtr workspace_pre;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetName() == workspace_name && node->GetOutDataNodes().empty()) {
        workspace_pre = node;
        break;
      }
    }
    if (workspace_pre == nullptr) {
      // [SK分段SIMT] 窗口超限形态仅保留 output 边界（input/index 由 IL 段 GM 直读），
      // 缺失的边界 workspace 跳过而非报错。
      GELOGD("[IndirectLoad] SK partition order skips missing workspace[%s].", workspace_name.c_str());
      continue;
    }
    node_order.emplace_back(workspace_pre);
    ordered_nodes.emplace(workspace_pre);
  }
  std::vector<af::AscNodePtr> remaining_outputs;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty() && ordered_nodes.find(node) == ordered_nodes.end()) {
      remaining_outputs.emplace_back(node);
    }
  }
  std::sort(remaining_outputs.begin(), remaining_outputs.end(),
            [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
              return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
            });
  node_order.insert(node_order.end(), remaining_outputs.begin(), remaining_outputs.end());
  return af::SUCCESS;
}

af::Status BuildSimdInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                              const char *name, ascir::AxisId &input_inner_axis) {
  GE_ASSERT_TRUE(!input_producer->outputs().empty(), "IndirectLoad SIMD input tensor producer has no output.");
  const auto input_axes = input_producer->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(axis_index < input_axes.size(), "IndirectLoad SIMD input axis index %zu is out of range [0, %zu).",
                 axis_index, input_axes.size());
  std::vector<ascir::AxisId> input_inner_axes(input_axes.begin() + static_cast<int64_t>(axis_index), input_axes.end());
  GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, input_inner_axes, name, input_inner_axis));
  return af::SUCCESS;
}

af::Status BuildSkInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t axis_index,
                                 ascir::AxisId &input_inner_axis) {
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  return BuildSimdInnerAxis(graph, input_boundary, axis_index, kInputInnerAxisName, input_inner_axis);
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

af::Status NormalizeAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                    ascir::AxisId input_inner_axis, ascir::AxisId index_inner_axis,
                                    bool simd_index_uses_output_inner_axis = false, bool solve_tile_size = false) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad output axis is empty.");
  ascir::AxisId outer_axis = af::kIdNone;
  ascir::AxisId inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildAxisViewByBoundary(graph, output_axes, boundary, outer_axis, inner_axis));
  if (simd_index_uses_output_inner_axis) {
    index_inner_axis = inner_axis;
  }
  const af::Axis *outer = graph.FindAxis(outer_axis);
  GE_ASSERT_NOTNULL(outer, "IndirectLoad outer axis %ld is not found.", outer_axis);
  const bool synthetic_outer =
      outer->from.empty() && std::find(indirect_load->attr.sched.axis.begin(), indirect_load->attr.sched.axis.end(),
                                       outer_axis) == indirect_load->attr.sched.axis.end();
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  if (solve_tile_size) {
    // tile 行数不在图改写期固定：TemplateAxes 不注解 tile 轴，调度期 TileTiling 发现
    // prebuilt pair 为空后走通用 TileSplit，按 UB 容量求解 tile 内行数。
    GE_ASSERT_TRUE(inner_axis != af::kIdNone, "IndirectLoad solvable tile split requires a non-empty inner view.");
  } else {
    GE_ASSERT_SUCCESS(CreateFixedTileSplit(graph, outer_axis, tile_outer_axis, tile_inner_axis));
  }
  std::vector<af::AxisId> vectorized_axes;
  if (inner_axis != af::kIdNone) {
    const auto *inner = graph.FindAxis(inner_axis);
    GE_ASSERT_NOTNULL(inner, "IndirectLoad inner axis %ld is not found.", inner_axis);
    vectorized_axes =
        inner->type == ascir::Axis::Type::kAxisTypeMerged ? inner->from : std::vector<af::AxisId>{inner_axis};
  } else {
    // A SIMT candidate without post Reduce uses the complete output view as
    // the outer view.  The IndirectLoad/its fused direct-GM path intentionally
    // keeps an empty vectorized view, while ordinary fan-out branches still
    // need the tile-inner axis for alignment and vector-function partitioning.
    GE_ASSERT_TRUE(tile_inner_axis != af::kIdNone, "IndirectLoad tile inner axis is missing.");
    vectorized_axes.emplace_back(tile_inner_axis);
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  axes.outer_axis = outer_axis;
  axes.inner_axis = inner_axis;
  axes.input_inner_axis = input_inner_axis;
  axes.index_inner_axis = index_inner_axis;
  axes.tile_outer_axis = tile_outer_axis;
  axes.tile_inner_axis = tile_inner_axis;
  axes.vectorized_axes = std::move(vectorized_axes);
  axes.synthetic_outer = synthetic_outer;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(indirect_load, axes));
  return af::SUCCESS;
}

af::Status SeedPostReduceInputVectorizedView(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce) {
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes));
  GE_ASSERT_TRUE(!axes.vectorized_axes.empty(), "IndirectLoad post Reduce vectorized axes are empty.");
  GE_ASSERT_TRUE(reduce->inputs.Size() == 1UL, "IndirectLoad post Reduce must have one input.");
  auto &input = reduce->inputs[0].attr;
  GE_ASSERT_TRUE(input.axis.size() == input.strides.size(), "IndirectLoad post Reduce input view is invalid.");
  input.vectorized_axis = axes.vectorized_axes;
  input.vectorized_strides.clear();
  input.vectorized_strides.reserve(input.vectorized_axis.size());
  for (const auto axis : input.vectorized_axis) {
    const auto axis_it = std::find(input.axis.begin(), input.axis.end(), axis);
    GE_ASSERT_TRUE(axis_it != input.axis.end(), "IndirectLoad post Reduce input axis[%ld] is missing.", axis);
    input.vectorized_strides.emplace_back(
        input.strides[static_cast<size_t>(std::distance(input.axis.begin(), axis_it))]);
  }
  return af::SUCCESS;
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                        bool solve_tile_size);
af::Status RestoreSkTemplateAxes(std::vector<ascir::ImplGraph> &grouped_graphs) {
  for (auto &graph : grouped_graphs) {
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    if (indirect_load == nullptr) {
      continue;
    }
    const auto restored_template = ascir::GetTemplateIdOrDefault(*indirect_load);
    const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
    GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                   indirect_load->GetNamePtr());
    if (restored_template == ascir::TemplateId::kIndirectLoadSimt && ascir::IsSkSegmentedSimt(*indirect_load)) {
      // [SK分段SIMT] 分区后子图轴表仅含节点引用的原始轴，候选图期归一新建的模板轴
      // 不在其中，调度期 BuildIndirectLoadAxisGroup 的 FindAxis 会失败。与原 SK 相同，
      // 在子图内重新执行 SIMT 归一。post 链已被 output workspace 边界隔断为独立子图，
      // 本子图内 IL 输出直连边界 Store，无 post-Reduce——与主线无 post-Reduce SIMT
      // 同语义：boundary=输出 rank，outer=完整输出视图（SIMT 逐元素枚举全部输出），
      // 而非 axis 前缀（那会使 outer 退化为合成单轴、发射数=1）。
      const auto simt_output_axes = indirect_load->outputs()[0]->attr.axis;
      GE_ASSERT_SUCCESS(
          NormalizeSimtAxesForTemplate(graph, indirect_load, simt_output_axes.size(), /*solve_tile_size=*/false));
      continue;
    }
    GE_ASSERT_TRUE(restored_template == ascir::TemplateId::kIndirectLoadSK,
                   "IndirectLoad partitioned graph[%s] has unexpected template.", graph.GetName().c_str());
    ascir::AxisId input_inner_axis = af::kIdNone;
    GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
    GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, af::kIdNone));
  }
  return af::SUCCESS;
}

af::Status ReplaceAxisPrefix(std::vector<af::AxisId> &target_axes, const std::vector<af::AxisId> &output_axes,
                             size_t axis_index) {
  GE_ASSERT_TRUE(target_axes.size() >= axis_index, "IndirectLoad SIMD input region has invalid axis rank.");
  std::copy_n(output_axes.begin(), axis_index, target_axes.begin());
  return af::SUCCESS;
}

af::Status ReplaceRegionAxisPrefix(const NodePath &region, const std::vector<af::AxisId> &output_axes,
                                   size_t axis_index, bool is_input_region) {
  for (const af::AscNodePtr &node : region) {
    if (!is_input_region && IsInputDataSource(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const bool should_replace = is_input_region ? ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)
                                                : role == ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre;
    if (!should_replace) {
      continue;
    }
    GE_ASSERT_SUCCESS(ReplaceAxisPrefix(node->attr.sched.axis, output_axes, axis_index));
    for (const auto &output : node->outputs()) {
      GE_ASSERT_SUCCESS(ReplaceAxisPrefix(output->attr.axis, output_axes, axis_index));
    }
  }
  return af::SUCCESS;
}

af::Status NormalizeSimdAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                        const RewrittenGraphAnalysis &analysis) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMD output axis is empty.");
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_NOTNULL(analysis.input_root, "IndirectLoad SIMD input producer is missing.");
  GE_ASSERT_SUCCESS(BuildSimdInnerAxis(graph, analysis.input_root, axis_index, kInputInnerAxisName, input_inner_axis));
  ascir::AxisId index_inner_axis = af::kIdNone;
  const auto index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMD index producer is missing.");
  if (analysis.simd_index_uses_output_inner_axis) {
    GE_ASSERT_SUCCESS(BuildSimdInnerAxis(graph, indirect_load, axis_index, kIndexInnerAxisName, index_inner_axis));
  }
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.input_region, output_axes, axis_index, true));
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.index_region, output_axes, axis_index, false));
  return NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, index_inner_axis,
                                  analysis.simd_index_uses_output_inner_axis);
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                        bool solve_tile_size) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMT output axis is empty.");
  GE_ASSERT_TRUE(boundary <= output_axes.size(), "IndirectLoad SIMT boundary %zu is out of range [0, %zu].", boundary,
                 output_axes.size());
  // post-Reduce SIMT（boundary 被 Reduce 布局覆写为 first_reduce，小于输出 rank）不预建
  // 固定 tile 轴：tile 行数交由调度期 TileTiling 走通用 TileSplit 按 UB 容量求解，使单个
  // tile 覆盖多行，消除逐行 SIMT/向量交替（VF_CALL 启动与 PipeBarrier 次数随行数下降）。
  // 无 post-Reduce 的 SIMT boundary 保持输出 rank，维持固定单行 tile 语义不变。
  // 对 AR 后置 Reduce，solve_tile_size 允许调度期按 UB 容量求解多行 tile。
  // SIMT index evaluator 使用完整 output_index 重建每个 Index Load 的逻辑偏移，
  // 因而 direct index 与 Cast/Add/Where 等复合 index 共用该路径。
  GE_ASSERT_SUCCESS(
      NormalizeAxesForTemplate(graph, indirect_load, boundary, af::kIdNone, af::kIdNone, false, solve_tile_size));
  return af::SUCCESS;
}

// SIMT direct-GM nodes deliberately preserve their vectorized view through the
// scheduler.  User graphs, however, do not necessarily initialize that view;
// fill it from the node's physical strides before VF partitioning consumes it.
af::Status CompletePreservedVectorizedViews(const af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || node == indirect_load || af::ops::IsOps<af::ascir_op::Output>(node) ||
        !ascgen_utils::indirect_load::GetTemplateBehavior(node).preserves_vectorized_axis) {
      continue;
    }
    // [行级广播 GM Load 标记] 此处视图尚未被调度期 SIMT 边界的 split 改写——按
    // 原始结构判定（尾轴零贡献 stride==0 且 size==1、其余轴稠密递推），语义为
    // 『读 [行,列] 值沿尾轴广播』。记录到节点 attr，供 codegen 坐标重建兜底在
    // 改写后视图 rank 失配时按原始语义寻址（改写后无法再判定）。
    if (af::ops::IsOps<af::ascir_op::Load>(node) && !node->outputs().empty() && node->outputs()[0] != nullptr) {
      const auto &mark_attr = node->outputs()[0]->attr;
      const auto &mark_sizes = mark_attr.repeats;
      const auto &mark_strides = mark_attr.strides;
      // 行级广播：尾轴 stride==0（值沿尾轴广播；size 即广播因子，无约束——原始
      // 形态 size==1，折叠广播后可为任意值）。其余轴构成 row-major 稠密物理布局：
      // 递推【从后往前】（stride[last_nonzero]==1，stride[i]==stride[i+1]*size[i+1]，
      // 中间零贡献轴不占物理空间直接跳过）——如 [8,2048,*]/[2048,1,0]：
      // a1.stride==1，a0.stride==1*2048，判定成立。
      // （历史缺陷：递推曾从前往后，expected 初值 1 使 a0.stride=2048≠1 误判
      // 非稠密——标记从未写入，兜底恒不命中，生产 load3 反复报 rank mismatch。）
      const bool tail_zero =
          !mark_sizes.empty() && mark_sizes.size() == mark_strides.size() &&
          af::SymbolicUtils::StaticCheckEq(mark_strides.back(), af::sym::kSymbolZero) == af::TriBool::kTrue;
      bool dense_rows = true;
      if (tail_zero) {
        af::Expression expected = af::sym::kSymbolOne;
        for (size_t index = mark_sizes.size(); index > 0UL; --index) {
          const size_t dim = index - 1UL;
          if (af::SymbolicUtils::StaticCheckEq(mark_strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
            continue;
          }
          if (af::SymbolicUtils::StaticCheckEq(mark_strides[dim], expected) != af::TriBool::kTrue) {
            dense_rows = false;
            break;
          }
          expected = af::sym::Mul(expected, mark_sizes[dim]);
        }
      } else {
        dense_rows = false;
      }
      GE_ASSERT_SUCCESS(ascir::SetRowBroadcastLoad(node, tail_zero && dense_rows));
    }
    for (const auto &output : node->outputs()) {
      if (output == nullptr || !output->attr.vectorized_axis.empty()) {
        continue;
      }
      const auto &axes = output->attr.axis;
      const auto &strides = output->attr.strides;
      GE_ASSERT_TRUE(!axes.empty() && axes.size() == strides.size(),
                     "IndirectLoad preserved node[%s] has invalid tensor view.", node->GetNamePtr());
      size_t vectorized_index = axes.size() - 1UL;
      for (size_t index = axes.size(); index > 0UL; --index) {
        const auto stride = strides[index - 1UL];
        if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolOne) == af::TriBool::kTrue) {
          vectorized_index = index - 1UL;
          break;
        }
        if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) != af::TriBool::kTrue) {
          vectorized_index = index - 1UL;
        }
      }
      output->attr.vectorized_axis = {axes[vectorized_index]};
      output->attr.vectorized_strides = {strides[vectorized_index]};
      GELOGD("[IndirectLoad] Seed preserved vectorized view for node[%s], axis[%ld].", node->GetNamePtr(),
             axes[vectorized_index]);
    }
  }
  return af::SUCCESS;
}

bool CanEmitSimtScalar(const af::AscNodePtr &node) {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  const auto *v2_impl = impl == nullptr ? nullptr : dynamic_cast<af::ascir::AscIrCodegenV2 *>(impl.get());
  return v2_impl != nullptr && v2_impl->IsSimtScalarSupported(*node);
}

// The IndirectLoad logical view is expressed in the Transpose output order,
// while the direct-GM boundary still reads the original Load buffer. Map the
// source Load strides into that output order for the strided address policy.
af::Status BuildLoadTransposeSourceView(const af::AscNodePtr &transpose, const af::AscTensorAttr &logical_attr,
                                        ascgen_utils::indirect_load::LogicalTensorView &view) {
  const auto load = ascgen_utils::indirect_load::GetInputProducer(transpose, 0UL);
  const auto &source = load->outputs[0].attr;
  GE_ASSERT_TRUE(source.axis.size() == source.strides.size() &&
                     logical_attr.axis.size() == logical_attr.repeats.size() &&
                     logical_attr.axis.size() == logical_attr.strides.size(),
                 "IndirectLoad Load->Transpose tensor metadata rank mismatch.");
  view = {logical_attr.axis, logical_attr.repeats, logical_attr.strides};
  for (size_t output_dim = 0UL; output_dim < logical_attr.axis.size(); ++output_dim) {
    const auto source_axis = std::find(source.axis.begin(), source.axis.end(), logical_attr.axis[output_dim]);
    GE_ASSERT_TRUE(source_axis != source.axis.end(), "IndirectLoad Transpose output axis is absent from Load source.");
    const size_t source_dim = static_cast<size_t>(std::distance(source.axis.begin(), source_axis));
    view.strides[output_dim] = source.strides[source_dim];
  }
  return af::SUCCESS;
}

// 单 Reduce 输出终止校验（候选级）：Store 或 Cast→Store 均合法，
// 其他形态由调用方淘汰当前候选而非断言失败。
af::Status ValidateReduceOutputCandidate(const af::AscNodePtr &reduce, bool &is_valid) {
  is_valid = false;
  const auto reduce_outputs = reduce->GetOutDataNodes();
  if (reduce_outputs.size() != 1UL) {
    GELOGI("[IndirectLoad] Reduce[%s] output count %zu is not 1.", reduce->GetNamePtr(), reduce_outputs.size());
    return af::SUCCESS;
  }
  auto successor = std::dynamic_pointer_cast<af::AscNode>(*reduce_outputs.begin());
  if (successor == nullptr) {
    return af::SUCCESS;
  }
  if (ScheduleUtils::IsStore(successor)) {
    is_valid = true;
    return af::SUCCESS;
  }
  if (af::ops::IsOps<af::ascir_op::Cast>(successor)) {
    const auto cast_outputs = successor->GetOutDataNodes();
    if (cast_outputs.size() != 1UL) {
      GELOGI("[IndirectLoad] Reduce[%s] Cast output count %zu is not 1.", reduce->GetNamePtr(), cast_outputs.size());
      return af::SUCCESS;
    }
    const auto cast_successor = std::dynamic_pointer_cast<af::AscNode>(*cast_outputs.begin());
    if (cast_successor != nullptr && ScheduleUtils::IsStore(cast_successor)) {
      is_valid = true;
    }
  }
  return af::SUCCESS;
}

// Traverse all output branches once, collecting Store/Reduce boundaries and validating each Reduce successor.
af::Status CollectOutputBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis,
                                   bool &is_candidate_legal) {
  NodeSet visited{indirect_load.get()};
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(out_asc_node, "IndirectLoad output successor is invalid.");
    roots.emplace_back(out_asc_node);
  }
  std::vector<af::AscNodePtr> reduce_nodes;
  const auto visit = [&analysis, &reduce_nodes](const af::AscNodePtr &current, bool &stop) -> af::Status {
    if (ScheduleUtils::IsStore(current)) {
      if (analysis.output_store == nullptr) {
        analysis.output_store = current;
      }
      stop = true;
      return af::SUCCESS;
    }
    if (ScheduleUtils::IsReduce(current)) {
      // 收集全部 Reduce，并继续遍历后续链路，以识别复合区域和最终 Store。
      reduce_nodes.emplace_back(current);
      if (analysis.post_reduce == nullptr) {
        analysis.post_reduce = current;
      }
      return af::SUCCESS;
    }
    return af::SUCCESS;
  };
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(roots, visited, visit));

  if (reduce_nodes.size() <= 1UL) {
    // 单 Reduce：沿用原终止校验语义（Store 或 Cast→Store），不符合则候选级淘汰。
    if (analysis.post_reduce != nullptr) {
      bool output_valid = false;
      GE_ASSERT_SUCCESS(ValidateReduceOutputCandidate(analysis.post_reduce, output_valid));
      if (!output_valid) {
        is_candidate_legal = false;
      }
    }
    return af::SUCCESS;
  }

  // 中间 Reduce 必须与 Broadcast 同轴；最终直接收敛到 Store 的 Reduce 可无 Broadcast。
  for (size_t index = 0UL; index < reduce_nodes.size(); ++index) {
    const auto &reduce = reduce_nodes[index];
    // 末尾 Reduce 的无 Broadcast 例外仅适用于整个后置区域的最终计算节点。
    const bool final_reduce = index + 1UL == reduce_nodes.size() && IsFinalReduce(reduce, analysis.output_store);
    if (!HasPairedBroadcastForReduce(reduce) && !final_reduce) {
      is_candidate_legal = false;
      GELOGI("[IndirectLoad] Composite region rejected: Reduce[%s] has no coaxial Broadcast pair.",
             reduce->GetNamePtr());
      return af::SUCCESS;
    }
  }
  analysis.composite_reduces = std::move(reduce_nodes);
  GELOGI("[IndirectLoad] Composite region detected with %zu paired Reduce/Broadcast stages.",
         analysis.composite_reduces.size());
  return af::SUCCESS;
}

af::Status CollectRewrittenBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis,
                                      bool &is_candidate_legal) {
  analysis.input_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  analysis.input_boundary = GetLoadTransposeSource(analysis.input_root);
  if (analysis.input_boundary == nullptr) {
    analysis.input_boundary = analysis.input_root;
  }
  analysis.index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_SUCCESS(CollectOutputBoundaries(indirect_load, analysis, is_candidate_legal));
  return af::SUCCESS;
}

bool CanMoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &consumer) {
  if (node == nullptr) {
    return false;
  }
  // Data and Load terminate the movable input-pre chain without being moved.
  if (af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::Load>(node)) {
    return false;
  }
  // Branching, control dependencies, and non-unary nodes cannot be safely reordered across IndirectLoad.
  if (node->inputs.Size() != 1UL || HasControlEdge(node) ||
      ascgen_utils::indirect_load::GetOnlyOutputConsumer(node) != consumer) {
    return false;
  }
  // Ordinary unary elementwise nodes can be moved; other compute types remain on the input side.
  return ScheduleUtils::IsElewise(node);
}

bool IsSimdInputShapeEnlarged(const af::AscNodePtr &indirect_load) {
  af::Expression input_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat :
       indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats) {
    input_numel = input_numel * repeat;
  }
  af::Expression output_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat : indirect_load->outputs()[0]->attr.repeats) {
    output_numel = output_numel * repeat;
  }
  return af::SymbolicUtils::StaticCheckGt(output_numel, input_numel) == af::TriBool::kTrue;
}

af::Status MoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &indirect_load) {
  const auto node_in = node->GetInDataAnchor(0UL);
  const auto indirect_in = indirect_load->GetInDataAnchor(ascgen_utils::indirect_load::kInputTensorIndex);
  GE_ASSERT_NOTNULL(node_in);
  GE_ASSERT_NOTNULL(indirect_in);
  const auto producer_out = node_in->GetPeerOutAnchor();
  const auto node_out = indirect_in->GetPeerOutAnchor();
  const auto indirect_out = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(producer_out);
  GE_ASSERT_NOTNULL(node_out);
  GE_ASSERT_NOTNULL(indirect_out);
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_out->GetOwnerNode());
  GE_ASSERT_NOTNULL(producer);
  GE_ASSERT_TRUE(node_out->GetOwnerNode() == node, "IndirectLoad input-pre edge does not match node[%s].",
                 node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(node_out, indirect_in, producer_out));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(producer_out, node_in, indirect_out));
  const auto peer_inputs = indirect_out->GetPeerInDataAnchors();
  for (const auto &peer_in : peer_inputs) {
    if (peer_in != node_in) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(indirect_out, peer_in, node_out));
    }
  }

  const auto indirect_output = indirect_load->outputs().front();
  const size_t producer_output_index = static_cast<size_t>(producer_out->GetIdx());
  const size_t node_output_index = static_cast<size_t>(node_out->GetIdx());
  GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                 "IndirectLoad input-pre producer output index is invalid.");
  GE_ASSERT_TRUE(node_output_index < node->outputs().size(), "IndirectLoad input-pre output index is invalid.");
  const auto node_output = node->outputs()[node_output_index];
  node->attr.sched.axis = indirect_load->attr.sched.axis;
  node_output->attr.axis = indirect_output->attr.axis;
  node_output->attr.repeats = indirect_output->attr.repeats;
  node_output->attr.strides = indirect_output->attr.strides;
  indirect_output->attr.dtype = producer->outputs()[producer_output_index]->attr.dtype;
  GELOGD("[IndirectLoad] Move input pre node[%s] after IndirectLoad[%s].", node->GetNamePtr(),
         indirect_load->GetNamePtr());
  return af::SUCCESS;
}

af::Status RewriteInputPreNodes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return af::SUCCESS;
  }
  if (HasControlEdge(indirect_load)) {
    GELOGI("[IndirectLoad] Skip moving input-pre nodes for node[%s].", indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd && IsSimdInputShapeEnlarged(indirect_load)) {
    const auto &input_shape = indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats;
    const auto &output_shape = indirect_load->outputs()[0]->attr.repeats;
    GELOGI("[IndirectLoad] Skip moving SIMD input-pre nodes for node[%s]: input shape[%s], output shape[%s].",
           indirect_load->GetNamePtr(), af::ToString(input_shape).c_str(), af::ToString(output_shape).c_str());
    return af::SUCCESS;
  }
  // Keep a direct Load->Transpose boundary on the SIMT input side.  Its
  // permutation is represented by the SIMT address policy; moving it across
  // IndirectLoad would discard the source physical view.
  const auto input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  if (template_id == ascir::TemplateId::kIndirectLoadSimt && GetLoadTransposeSource(input_producer) != nullptr) {
    return af::SUCCESS;
  }
  NodePath movable_nodes;
  af::AscNodePtr consumer = indirect_load;
  for (af::AscNodePtr node =
           ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
       CanMoveInputPreNode(node, consumer); node = ascgen_utils::indirect_load::GetInputProducer(node, 0UL)) {
    movable_nodes.emplace_back(node);
    consumer = node;
  }
  if (movable_nodes.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : movable_nodes) {
    GE_ASSERT_SUCCESS(MoveInputPreNode(node, indirect_load));
  }
  return ScheduleUtils::TopologicalSorting(graph);
}

af::Status RewriteSimtInputBroadcast(const af::AscNodePtr &indirect_load, InputViewPlan &input_plan,
                                     bool &is_candidate_legal) {
  const af::AscNodePtr input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  if (!IsBroadcastNode(input_producer)) {
    return af::SUCCESS;
  }

  const auto broadcast_iter = std::find(input_plan.path.begin(), input_plan.path.end(), input_producer);
  GE_ASSERT_TRUE(broadcast_iter != input_plan.path.end(),
                 "SIMT direct input Broadcast is missing from collected path.");
  input_plan.path_broadcast_index = static_cast<int64_t>(std::distance(input_plan.path.begin(), broadcast_iter));
  GE_ASSERT_SUCCESS(FoldBroadcastPath(input_plan, ascir::TemplateId::kIndirectLoadSimt, is_candidate_legal));
  return af::SUCCESS;
}

// 不区分一元/二元路径，沿所有上游数据边收集 Load；Load 节点是输入边界，停止继续回溯。
void CollectReachableLoads(const af::AscNodePtr &root, NodeSet &visited, NodePath &loads) {
  if (root == nullptr) {
    return;
  }
  TraverseInputProducers({root}, visited, [&loads](const af::AscNodePtr &node) {
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      loads.emplace_back(node);
      return false;
    }
    return true;
  });
}

af::Status CompleteInputDataTensorAttrs(const RewrittenGraphAnalysis &analysis) {
  // 仅对可达 Load 补齐对应 Data；没有 Load 的路径无需处理。
  NodeSet visited;
  for (const af::AscNodePtr &root : {analysis.input_root, analysis.index_root}) {
    NodePath loads;
    CollectReachableLoads(root, visited, loads);
    for (const af::AscNodePtr &load : loads) {
      const af::AscNodePtr data = ascgen_utils::indirect_load::GetInputProducer(load, 0UL);
      GE_ASSERT_NOTNULL(data, "IndirectLoad input Load[%s] has no producer.", load->GetNamePtr());
      GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(data), "IndirectLoad input Load[%s] producer is not Data.",
                     load->GetNamePtr());
      const auto load_outputs = load->outputs();
      const auto data_outputs = data->outputs();
      GE_ASSERT_TRUE(!load_outputs.empty() && !data_outputs.empty(),
                     "IndirectLoad input Load[%s] or Data[%s] has no output.", load->GetNamePtr(), data->GetNamePtr());
      const auto load_out = load_outputs.front();
      const auto data_out = data_outputs.front();
      GE_ASSERT_NOTNULL(load_out, "IndirectLoad input Load[%s] output is null.", load->GetNamePtr());
      GE_ASSERT_NOTNULL(data_out, "IndirectLoad input Data[%s] output is null.", data->GetNamePtr());
      GE_ASSERT_TRUE(!load_out->attr.axis.empty() && load_out->attr.axis.size() == load_out->attr.repeats.size() &&
                         load_out->attr.axis.size() == load_out->attr.strides.size(),
                     "IndirectLoad input Load[%s] tensor attributes are incomplete.", load->GetNamePtr());
      GELOGD("[IndirectLoad] Complete tensor attrs from Load[%s] to Data[%s].", load->GetNamePtr(), data->GetNamePtr());
      data_out->attr.axis = load_out->attr.axis;
      data_out->attr.repeats = load_out->attr.repeats;
      data_out->attr.strides = load_out->attr.strides;
      data_out->attr.dtype = load_out->attr.dtype;
    }
  }
  return af::SUCCESS;
}

// 收集 IndirectLoad 两个输入的路径，并按模板规则选出待处理的 Broadcast：SIMD 只看直接生产者，SIMT/SK
// 沿单输入链回溯并查找最近 Broadcast。
void CollectInputPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                       PhysicalViewPreparation &preparation) {
  const bool collect_full_path = template_id != ascir::TemplateId::kIndirectLoadSimd;
  for (size_t input_index :
       {ascgen_utils::indirect_load::kInputTensorIndex, ascgen_utils::indirect_load::kIndexTensorIndex}) {
    InputViewPlan &plan =
        input_index == ascgen_utils::indirect_load::kInputTensorIndex ? preparation.input : preparation.index;
    NodePath &path = plan.path;
    plan.path_broadcast_index = kInvalidBroadcastIndex;
    for (af::AscNodePtr current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
         current != nullptr; current = collect_full_path && current->inputs.Size() == 1UL
                                           ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                           : nullptr) {
      if (plan.path_broadcast_index == kInvalidBroadcastIndex && IsBroadcastNode(current)) {
        plan.path_broadcast_index = static_cast<int64_t>(path.size());
      }
      path.emplace_back(current);
    }
  }
}

af::Status ResolveInputPhysicalView(const af::AscTensorAttr &logical_attr, ascir::TemplateId template_id,
                                    InputViewPlan &plan, af::AscTensorAttr &physical_attr, bool &is_supported) {
  physical_attr = logical_attr;
  plan.load_transpose = nullptr;
  const bool has_broadcast = plan.path_broadcast_index != kInvalidBroadcastIndex;
  af::AscNodePtr source = plan.path.empty() ? nullptr : plan.path.front();
  if (has_broadcast) {
    const auto broadcast = plan.path[static_cast<size_t>(plan.path_broadcast_index)];
    source = ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL);
    if (source == nullptr) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast node[%s] source is invalid.",
             static_cast<int32_t>(template_id), broadcast->GetNamePtr());
      is_supported = false;
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(GetBroadcastPhysicalAttr(broadcast, template_id, physical_attr));
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK || GetLoadTransposeSource(source) == nullptr) {
    return af::SUCCESS;
  }
  // SIMD may use the source Load only when the intervening Transpose can be folded safely.
  // SIMT already reads GM directly, so it only needs the axis-mapped view, not a graph rewrite.
  if (has_broadcast && template_id == ascir::TemplateId::kIndirectLoadSimd &&
      GetFoldableLoadTransposeSource(source) == nullptr) {
    return af::SUCCESS;
  }
  plan.load_transpose = source;
  ascgen_utils::indirect_load::LogicalTensorView source_view;
  GE_ASSERT_SUCCESS(BuildLoadTransposeSourceView(source, physical_attr, source_view));
  physical_attr.strides = std::move(source_view.strides);
  return af::SUCCESS;
}

af::Status AnalyzeInputPath(const af::AscNodePtr &indirect_load, size_t input_idx, ascir::TemplateId template_id,
                            InputViewPlan &plan, bool &is_path_supported) {
  is_path_supported = true;
  const auto inputs = indirect_load->inputs();
  const auto &logical_attr = inputs[input_idx]->attr;
  af::AscTensorAttr physical_attr;
  GE_ASSERT_SUCCESS(ResolveInputPhysicalView(logical_attr, template_id, plan, physical_attr, is_path_supported));
  if (!is_path_supported) {
    return af::SUCCESS;
  }
  const bool has_broadcast = plan.path_broadcast_index != kInvalidBroadcastIndex;
  ascgen_utils::indirect_load::LogicalTensorView view{logical_attr.axis, logical_attr.repeats, physical_attr.strides};
  if (has_broadcast) {
    GE_ASSERT_SUCCESS(BuildBroadcastLogicalView(logical_attr, physical_attr, template_id, view));
  }
  plan.simd_index_uses_output_inner_axis = template_id == ascir::TemplateId::kIndirectLoadSimd &&
                                           input_idx == ascgen_utils::indirect_load::kIndexTensorIndex &&
                                           (has_broadcast || plan.load_transpose != nullptr);
  if (plan.load_transpose != nullptr) {
    // Permutations are valid non-overlapping strided views even when strides are not monotonically decreasing.
    plan.layout = {{view.axis_ids, view.sizes, view.strides},
                   ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided,
                   physical_attr.repeats};
  } else {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(
        view, plan.layout,
        (template_id == ascir::TemplateId::kIndirectLoadSimt || template_id == ascir::TemplateId::kIndirectLoadSK) &&
            has_broadcast));
    if (template_id == ascir::TemplateId::kIndirectLoadSK && has_broadcast &&
        plan.layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      plan.layout.physical_repeats = physical_attr.repeats;
    }
  }
  if (plan.layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: input path layout%s is unsupported.",
           static_cast<int32_t>(template_id), has_broadcast ? " with Broadcast source view" : "");
    is_path_supported = false;
  }
  return af::SUCCESS;
}

af::Status RewriteSkInputPaths(PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                               bool &is_candidate_legal) {
  for (InputViewPlan *const plan : {&preparation.input, &preparation.index}) {
    GE_ASSERT_SUCCESS(FoldBroadcastPath(*plan, ascir::TemplateId::kIndirectLoadSK, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  analysis.input_path = std::move(preparation.input.path);
  analysis.index_path = std::move(preparation.index.path);
  return af::SUCCESS;
}

af::Status RewriteSimdInputLayouts(PhysicalViewPreparation &preparation) {
  for (InputViewPlan *plan : {&preparation.input, &preparation.index}) {
    if (plan->path_broadcast_index == kInvalidBroadcastIndex || plan->load_transpose == nullptr) {
      continue;
    }
    const auto broadcast = plan->path[static_cast<size_t>(plan->path_broadcast_index)];
    const auto &transpose = plan->load_transpose;
    const auto load = ascgen_utils::indirect_load::GetInputProducer(transpose, 0UL);
    GE_ASSERT_NOTNULL(load);
    const auto owner = transpose->GetOwnerComputeGraph();
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::IsolateNodeOneIO(transpose));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner, transpose));
    // Once the Transpose is removed, the Load and Broadcast share the resolved physical view.
    GE_ASSERT_SUCCESS(ApplyPhysicalView(NodePath{load, broadcast}, plan->layout));
  }
  return af::SUCCESS;
}

af::Status RewriteBroadcastPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                 PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return RewriteSkInputPaths(preparation, analysis, is_candidate_legal);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return RewriteSimtInputBroadcast(indirect_load, preparation.input, is_candidate_legal);
  }
  return RewriteSimdInputLayouts(preparation);
}

af::Status PreparePhysicalViews(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                PhysicalViewPreparation &preparation, bool &is_candidate_legal) {
  // 输入/输出个数已在 Generate 入口统一校验，axis 由 AnalyzeRewrittenGraph 统一获取，此处不再重复获取。
  is_candidate_legal = true;
  CollectInputPaths(indirect_load, template_id, preparation);
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    InputViewPlan &plan =
        input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? preparation.input : preparation.index;
    bool is_path_supported = true;
    GE_ASSERT_SUCCESS(AnalyzeInputPath(indirect_load, input_idx, template_id, plan, is_path_supported));
    if (!is_path_supported) {
      is_candidate_legal = false;
      const char *const path_name = input_idx == ascgen_utils::indirect_load::kIndexTensorIndex ? "index" : "input";
      GELOGI("[IndirectLoad] Reject candidate[%d]: %s path is unsupported.", static_cast<int32_t>(template_id),
             path_name);
      return af::SUCCESS;
    }
  }
  preparation.logical_view.input = preparation.input.layout;
  preparation.logical_view.index = preparation.index.layout;
  const auto outputs = indirect_load->outputs();
  const auto &output_attr = outputs.front()->attr;
  preparation.logical_view.output = {output_attr.axis, output_attr.repeats, output_attr.strides};
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateIndirectLoadOutputLayout(preparation.logical_view.output));
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::AnalyzeIndirectLoadAccess(indirect_load, preparation.logical_view, access_info));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetIndirectLoadAccessInfo(indirect_load, access_info));
  // 本函数只做只读分析；Broadcast 删除与物理视图写回在 AnalyzeRewrittenGraph 中统一执行。
  return ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, preparation.logical_view);
}

af::Status CollectRewrittenRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, RewrittenGraphAnalysis &analysis) {
  NodeSet input_region_set;
  NodeSet index_region_set;
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex, input_region_set);
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex, index_region_set);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 的 index 链与输出链共享统一轴、下游统一处理，融合集整体归入 index_region；region 保持为空。
    CollectSimtFusedRegionMembers(indirect_load, analysis, index_region_set);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  // 一次 GetAllNodes 收集 region 与 index_region，保证顺序确定性。
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (index_region_set.count(node.get()) != 0UL) {
      analysis.index_region.emplace_back(node);
    }
    if (input_region_set.count(node.get()) != 0UL) {
      analysis.input_region.emplace_back(node);
    }
  }
  return af::SUCCESS;
}

af::Status ApplyTemplatePathLayouts(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                    const PhysicalViewPreparation &preparation,
                                    const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 融合集（index_region）从 index/output root 回溯，不包含 IndirectLoad 的 input
    // root，因此需要单独写回直接输入生产者。
    const InputViewPlan *const m2_plans[] = {&preparation.input, &preparation.index};
    for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx);
      if (producer != nullptr) {
        GE_ASSERT_SUCCESS(ApplyIndirectLoadPathLayout(NodePath{producer}, m2_plans[input_idx]->layout, false));
      }
    }
    return af::SUCCESS;
  }

  const bool is_simd = template_id == ascir::TemplateId::kIndirectLoadSimd;
  const NodePath &input_path = is_simd ? analysis.input_region : analysis.input_path;
  const NodePath &index_path = is_simd ? analysis.index_region : analysis.index_path;
  const auto &input_layout = preparation.input.layout;
  const auto &index_layout = preparation.index.layout;
  // SK 将输入链切分为独立的 GM->workspace 子图。对 Broadcast 源的 strided
  // view，workspace 边界会从该链末端的 Load 拷贝 tensor attr；若不在切分前
  // 写回物理 view，子图会把 Broadcast 后的逻辑 extent 当成源 GM 连续布局，
  // 导致错误的 GM 偏移。SIMD/SIMT 仍沿用各自的 strided 处理方式。
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    if (input_layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      GE_ASSERT_SUCCESS(ApplyPhysicalView(input_path, input_layout));
    }
    if (index_layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      GE_ASSERT_SUCCESS(ApplyPhysicalView(index_path, index_layout));
    }
  }
  // [post-Reduce 稠密视图] IL 输出链到达 Reduce（含经 elementwise 中间链，如
  // gather+ele+sum / gather+norm）时不做 kStridedUbPath 对齐标注：该标注会经
  // SetVectorizedStridesForTensor(kAligned) 把视图尾轴按对齐块描述（如 4→8），
  // 而 SIMD 实际写出为稠密布局（only_gather 同形态实证），视图与实现不符导致
  // 下游 Abs/Reduce/DataCopy 按 stride8 错位访问（AIC 341）。无 post-Reduce 的
  // 纯 gather 输出直连 Store 场景维持原标注语义不变。
  const bool has_post_reduce = analysis.post_reduce != nullptr;
  GE_ASSERT_SUCCESS(
      ApplyIndirectLoadPathLayout(input_path, input_layout, analysis.align_input_path && !has_post_reduce));
  GE_ASSERT_SUCCESS(
      ApplyIndirectLoadPathLayout(index_path, index_layout, analysis.align_index_path && !has_post_reduce));
  return af::SUCCESS;
}

af::Status AnalyzeRewrittenGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, bool &is_candidate_legal,
                                 RewrittenGraphAnalysis &analysis) {
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  // 分析阶段：路径、布局分类、对齐标志（只读）
  PhysicalViewPreparation preparation;
  GE_ASSERT_SUCCESS(PreparePhysicalViews(indirect_load, template_id, preparation, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  analysis.align_index_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.index, axis_index);
  analysis.align_input_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.input, axis_index);
  analysis.simd_index_uses_output_inner_axis = preparation.index.simd_index_uses_output_inner_axis;

  // SK 跳过 input-pre 搬移；SIMD/SIMT 将 input-pre 单目元素链搬到 IndirectLoad 之后。
  GE_ASSERT_SUCCESS(RewriteInputPreNodes(graph, indirect_load, template_id));
  GE_ASSERT_SUCCESS(RewriteBroadcastPaths(indirect_load, template_id, preparation, analysis, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  // 收集阶段：一次遍历收集全部输出边界；SK 也必须知道 Norm 后置区域，
  // 但 SK 的输入/索引分区仍由其专用路径处理。
  GE_ASSERT_SUCCESS(CollectRewrittenBoundaries(indirect_load, analysis, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  if (template_id != ascir::TemplateId::kIndirectLoadSK) {
    GE_ASSERT_SUCCESS(CollectRewrittenRegion(graph, indirect_load, template_id, analysis));
  }

  // 路径布局处理：紧凑零 stride 写回物理视图，需对齐的 strided 路径标注 UB role。
  GE_ASSERT_SUCCESS(ApplyTemplatePathLayouts(indirect_load, template_id, preparation, analysis));
  return af::SUCCESS;
}

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis) {
  for (const af::AscNodePtr &node : analysis.input_region) {
    if (IsInputDataSource(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const auto simd_role = role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath
                               ? ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath
                               : ascgen_utils::indirect_load::TemplateRole::kSimdInputPre;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, simd_role));
  }
  if (analysis.simd_index_uses_output_inner_axis) {
    for (const af::AscNodePtr &node : analysis.index_region) {
      if (IsInputDataSource(node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(
          ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre));
    }
  }
  return af::SUCCESS;
}

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!af::ops::IsOps<af::ascir_op::Load>(analysis.input_boundary) || analysis.index_region.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : analysis.index_region) {
    if (GetLoadTransposeSource(node) != nullptr) {
      continue;
    }
    // Compile-time Scalar values are emitted as local constants by the SIMT evaluator.
    // ScalarData is runtime input and still requires an explicit context/GM binding.
    if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node) || HasControlEdge(node)) {
      return af::SUCCESS;
    }
    const bool is_gm_boundary = af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node);
    if (!is_gm_boundary && !CanEmitSimtScalar(node)) {
      GELOGD("[IndirectLoad] SIMT scalar codegen does not support node[%s, %s].", node->GetNamePtr(),
             node->GetTypePtr());
      return af::SUCCESS;
    }
  }
  is_candidate_legal = true;
  return af::SUCCESS;
}

af::Status CollectSimtNormalScheduleNodes(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                          NodeSet &normal_schedule_nodes) {
  normal_schedule_nodes.clear();
  if (analysis.post_reduce == nullptr) {
    return af::SUCCESS;
  }

  NodePath normal_schedule_path;
  const auto collect = [&normal_schedule_nodes, &normal_schedule_path](const af::AscNodePtr &node,
                                                                       bool &stop) -> af::Status {
    normal_schedule_nodes.emplace(node.get());
    normal_schedule_path.emplace_back(node);
    if (ScheduleUtils::IsStore(node)) {
      stop = true;
    }
    return af::SUCCESS;
  };
  GE_ASSERT_SUCCESS(TraverseOutputConsumers({analysis.post_reduce}, normal_schedule_nodes, collect));

  NodeSet normal_schedule_dependencies;
  const auto collect_dependency = [indirect_load](const af::AscNodePtr &node) {
    if (node == indirect_load) {
      return false;
    }
    return true;
  };
  TraverseInputProducers(normal_schedule_path, normal_schedule_dependencies, collect_dependency);
  normal_schedule_dependencies.erase(indirect_load.get());
  normal_schedule_nodes.insert(normal_schedule_dependencies.begin(), normal_schedule_dependencies.end());
  return af::SUCCESS;
}

af::Status AnnotateSimtTemplateRoles(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis) {
  GE_ASSERT_NOTNULL(analysis.input_boundary, "IndirectLoad SIMT input boundary is missing.");
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      analysis.input_boundary, ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary));
  if (analysis.input_boundary != analysis.input_root) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        analysis.input_root, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
  }
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateRole(indirect_load, ascgen_utils::indirect_load::TemplateRole::kSimtOp));
  size_t annotated_count = 0UL;
  for (const af::AscNodePtr &node : analysis.index_region) {
    // 注意：index_region 即 lowering 侧 BuildSimtLoweringMetadata 会校验的集合
    // （index_root 与 post Reduce 输入链 backward）。其中同时落在 normal schedule
    // 依赖闭包内的节点（如与 IndirectLoad 输出直连、又汇入 post Reduce 输入的
    // Broadcast），lowering 侧仍按 SIMT 区域消费并要求 scalar evaluator 角色；
    // 不能被 normal_schedule 排除，否则出现 no scalar evaluator role 断言。
    // normal schedule 排除仅作用于 fanout 标注（AnnotateSimtFanoutBranches）。
    // Every node in the fused index/output backward region is emitted by the
    // SIMT scalar evaluator, including nodes that fan out to multiple
    // consumers inside that same region.  A fan-out role is reserved for
    // branches outside the selected scalar path and is assigned by
    // AnnotateSimtFanoutBranches below; marking an internal fan-out here would
    // make ValidateSimtRegionNode reject it as not inline-transformable.
    const auto role = (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node))
                          ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                          : ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, role));
    GELOGD("[IndirectLoad] SIMT region node[%s] annotated role[%ld].", node->GetNamePtr(), static_cast<int64_t>(role));
    ++annotated_count;
  }
  GELOGD("[IndirectLoad] SIMT index_region annotated %zu nodes (region size %zu), post reduce[%s].", annotated_count,
         analysis.index_region.size(), analysis.post_reduce == nullptr ? "none" : analysis.post_reduce->GetNamePtr());
  return af::SUCCESS;
}

af::Status AnnotateSimtMainOutputPath(const af::AscNodePtr &indirect_load, const af::AscNodePtr &selected_root,
                                      const NodeSet &selected, const NodeSet &normal_schedule_nodes) {
  // Nodes on the selected output path are emitted by the SIMT scalar evaluator.
  // With a user fan-out this path can contain ordinary elementwise transforms
  // that are not part of the index region handled by AnnotateSimtTemplateRoles.
  NodePath pending = {selected_root};
  NodeSet visited;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &node = pending[cursor];
    if (node == nullptr || node == indirect_load || !visited.emplace(node.get()).second) {
      continue;
    }
    if (normal_schedule_nodes.count(node.get()) == 0UL && !IsInputDataSource(node) && !ScheduleUtils::IsReduce(node) &&
        !ScheduleUtils::IsStore(node)) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
    }
    for (const auto &producer_node : node->GetInDataNodes()) {
      const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_node);
      if (producer != nullptr && selected.count(producer.get()) != 0UL) {
        pending.emplace_back(producer);
      }
    }
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtSideInputClosure(const af::AscNodePtr &branch_node, const af::AscNodePtr &indirect_load,
                                        const NodeSet &selected, const NodeSet &normal_schedule_nodes,
                                        NodeSet &visited) {
  NodePath pending;
  for (const auto &producer_node : branch_node->GetInDataNodes()) {
    const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_node);
    if (producer != nullptr) {
      pending.emplace_back(producer);
    }
  }
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &producer = pending[cursor];
    if (producer == nullptr || producer == indirect_load || selected.count(producer.get()) != 0UL ||
        normal_schedule_nodes.count(producer.get()) != 0UL || ScheduleUtils::IsReduce(producer) ||
        !visited.emplace(producer.get()).second) {
      continue;
    }
    if (IsInputDataSource(producer) && !af::ops::IsOps<af::ascir_op::Load>(producer)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(producer);
    const auto side_role = af::ops::IsOps<af::ascir_op::Load>(producer)
                               ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                               : ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch;
    if (role == ascgen_utils::indirect_load::TemplateRole::kNone ||
        role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(producer, side_role));
    }
    for (const auto &upstream_node : producer->GetInDataNodes()) {
      const auto upstream = std::dynamic_pointer_cast<af::AscNode>(upstream_node);
      if (upstream != nullptr) {
        pending.emplace_back(upstream);
      }
    }
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtFanoutBranches(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                      NodeSet &normal_schedule_nodes) {
  const af::AscNodePtr selected_root = analysis.post_reduce == nullptr
                                           ? analysis.output_store
                                           : ascgen_utils::indirect_load::GetInputProducer(analysis.post_reduce, 0UL);
  if (selected_root == nullptr) {
    return af::SUCCESS;
  }
  NodeSet selected;
  if (!CollectSimtBackwardRegion({selected_root}, indirect_load, selected)) {
    GELOGW("[IndirectLoad] Cannot identify SIMT main output chain from root[%s]; keep existing roles.",
           selected_root->GetNamePtr());
    return af::SUCCESS;
  }

  // 复合 Norm 场景：首个 Reduce 之后的后置链路（含汇入该链路的分支）由普通调度
  // 执行，与 lowering 侧 keep_chain 的取舍保持一致；该区域不能标注 SIMT 角色，
  // 否则会跳过主 tiling 而保持原始 rank，与参与 tiling 的 Reduce 输出 view 不一致。
  // 普通调度区域及其外部输入必须统一排除 SIMT 角色；否则父图不发射 Load，
  // VectorFunc 仍把该 Tensor 作为跨子图输入，最终出现 no API call found。
  GE_ASSERT_SUCCESS(CollectSimtNormalScheduleNodes(indirect_load, analysis, normal_schedule_nodes));
  GE_ASSERT_SUCCESS(AnnotateSimtMainOutputPath(indirect_load, selected_root, selected, normal_schedule_nodes));
  if (analysis.post_reduce != nullptr) {
    GELOGD("[IndirectLoad] SIMT post Reduce[%s] keeps %zu nodes on the normal schedule chain.",
           analysis.post_reduce->GetNamePtr(), normal_schedule_nodes.size());
  }
  GELOGD("[IndirectLoad] Normal schedule dependencies keep %zu nodes out of SIMT fanout annotation.",
         normal_schedule_nodes.size());

  NodeSet visited{indirect_load.get()};
  NodeSet side_visited;
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
    roots.emplace_back(consumer);
  }
  const auto visit = [&](const af::AscNodePtr &node, bool &stop) -> af::Status {
    const bool is_fanout_branch = selected.count(node.get()) == 0UL && !IsInputDataSource(node) &&
                                  !ScheduleUtils::IsReduce(node) && normal_schedule_nodes.count(node.get()) == 0UL;
    if (is_fanout_branch) {
      GELOGD("[IndirectLoad] SIMT fanout branch node[%s] annotated role[%ld].", node->GetNamePtr(),
             static_cast<int64_t>(ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch));
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch));
      GE_ASSERT_SUCCESS(
          AnnotateSimtSideInputClosure(node, indirect_load, selected, normal_schedule_nodes, side_visited));
    }
    if (ScheduleUtils::IsStore(node) || ScheduleUtils::IsReduce(node)) {
      stop = true;
      return af::SUCCESS;
    }
    return af::SUCCESS;
  };
  return TraverseOutputConsumers(roots, visited, visit);
}

// SIMD 多阶段布局校验：区域内每个统计 Reduce 独立通过
// ValidateSimdPostReduceLayout（G≥R + 后缀连续），首个失败即淘汰候选。
af::Status ValidateSimdCompositeLayouts(const af::AscNodePtr &indirect_load,
                                        const std::vector<af::AscNodePtr> &composite_reduces,
                                        bool &is_candidate_legal) {
  is_candidate_legal = true;
  for (const auto &reduce : composite_reduces) {
    bool reduce_legal = true;
    GE_ASSERT_SUCCESS(ValidateSimdPostReduceLayout(indirect_load, reduce, reduce_legal));
    if (!reduce_legal) {
      is_candidate_legal = false;
      GELOGI("[IndirectLoad] Reject SIMD composite candidate: Reduce[%s] layout is invalid.", reduce->GetNamePtr());
      return af::SUCCESS;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal) {
  // Softmax 专用节点保持完整输入输出形状，普通 Reduce 的 stride 推导（BuildPostReduceLayout）
  // 无法识别其内部归约轴，必须先分流再做布局校验：
  // - 尾轴约束满足 → 走专用 API，仅校验 View 与 IndirectLoad 输出一致；
  // - 尾轴约束不满足 → Pattern 已在主图替换、无法回退原始结构，淘汰候选。
  const bool is_softmax_post =
      analysis.post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce);
  if (is_softmax_post) {
    GE_ASSERT_SUCCESS(CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, template_id));
    if (is_candidate_legal) {
      // Softmax 专用路径：R 轴是 Softmax 输入的尾轴，必须完整保留在 tile 内。
      // 将 boundary 覆写为尾轴前一位（与 composite 路径的 first_reduce 语义对齐）：
      // outer=[保留轴]（可切、tile 行数可求解），inner=[尾轴]（SoftmaxAR 的 R 参数，
      // 由 vectorized 视图最后一个轴推导）。否则 boundary 保持输出 rank，outer 会
      // merge 全部轴，vectorized 退化为 size=1 的 tile 轴，SoftmaxAR 的 R 恒为 1。
      const auto softmax_inputs = analysis.post_reduce->inputs();
      GE_ASSERT_TRUE(!softmax_inputs.empty() && softmax_inputs[0] != nullptr,
                     "IndirectLoad Softmax node[%s] has no input.", analysis.post_reduce->GetNamePtr());
      const size_t softmax_input_rank = softmax_inputs[0]->attr.axis.size();
      GE_ASSERT_TRUE(softmax_input_rank > 0UL, "IndirectLoad Softmax input rank is invalid, node[%s].",
                     analysis.post_reduce->GetNamePtr());
      boundary = softmax_input_rank - 1UL;
    }
    return af::SUCCESS;
  }
  // SIMD 多阶段：区域内每个 Reduce 独立布局校验后，整链保留在候选图中，
  // 由连通性分区保持同组并复用既有 Reduce/Broadcast/Elementwise Codegen。
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    if (!analysis.composite_reduces.empty()) {
      GE_ASSERT_SUCCESS(ValidateSimdCompositeLayouts(indirect_load, analysis.composite_reduces, is_candidate_legal));
    } else {
      GE_ASSERT_SUCCESS(ValidateSimdPostReduceLayout(indirect_load, analysis.post_reduce, is_candidate_legal));
    }
  } else {
    // FindPostReduceChain 保留首个 Reduce 作为 SIMT local target 锚点，并继续
    // 遍历其后的 Broadcast/Elementwise/Reduce；后续统计阶段由普通调度链执行，
    // 因而串行和共享输入两类复合区域均可复用现有 Codegen。
    GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(analysis.post_reduce, boundary, is_candidate_legal));
    if (is_candidate_legal) {
      GE_ASSERT_SUCCESS(ValidateSimtTemplateRegion(analysis, is_candidate_legal));
    }
  }
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  // 通用路径：保存 NormInfo 供后续多阶段扩展使用。
  GE_ASSERT_SUCCESS(CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, template_id));
  return af::SUCCESS;
}

af::Status AnnotateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return AnnotateSimdTemplateRoles(analysis);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return AnnotateSimtTemplateRoles(indirect_load, analysis);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

// 为区域内全部统计 Reduce 种输入向量化视图：SIMD 多阶段链中每个 Reduce
// 都按模板向量化轴消费数据，缺失种子会导致 Codegen 无法组织 A/R 参数。
af::Status SeedCompositeReduceInputViews(const af::AscNodePtr &indirect_load,
                                         const std::vector<af::AscNodePtr> &composite_reduces) {
  for (const auto &reduce : composite_reduces) {
    GE_ASSERT_SUCCESS(SeedPostReduceInputVectorizedView(indirect_load, reduce));
  }
  return af::SUCCESS;
}

af::Status NormalizeTemplateAxes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, const RewrittenGraphAnalysis &analysis,
                                 size_t boundary) {
  const bool is_softmax_post =
      analysis.post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce);
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    GE_ASSERT_SUCCESS(NormalizeSimdAxesForTemplate(graph, indirect_load, analysis));
    if (is_softmax_post) return af::SUCCESS;
    // 多阶段：全部 Reduce 各自种向量化视图；单 Reduce/无 Reduce 保持原行为。
    if (!analysis.composite_reduces.empty()) {
      return SeedCompositeReduceInputViews(indirect_load, analysis.composite_reduces);
    }
  } else {
    // 仅 AR 后置 Reduce 开启可求解 tile：TileInner 代表多个 retained
    // feature 行，reduced suffix 保持完整向量化，避免将 RA 的 retained
    // 后缀误并入同一 Reduce API 窗口。
    bool solve_tile = false;
    GE_ASSERT_SUCCESS(IsSimtArPostReduce(analysis.post_reduce, solve_tile));
    GE_ASSERT_SUCCESS(NormalizeSimtAxesForTemplate(graph, indirect_load, boundary, solve_tile));
  }
  if (is_softmax_post) return af::SUCCESS;
  return SeedPostReduceInputVectorizedView(indirect_load, analysis.post_reduce);
}

af::Status FinalizeTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(::ascir::SetDcacheSize(indirect_load, kIndirectLoadSimtDcacheSize));
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

af::Status ApplySkGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &is_candidate_legal) {
  is_candidate_legal = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      GELOGI("[IndirectLoad] Skip SK candidate: Transpose is only supported by SIMD/SIMT.");
      return af::SUCCESS;
    }
  }
  RewrittenGraphAnalysis analysis;
  GE_ASSERT_SUCCESS(
      AnalyzeRewrittenGraph(graph, indirect_load, ascir::TemplateId::kIndirectLoadSK, is_candidate_legal, analysis));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject SK: rewritten graph analysis is illegal.");
    return af::SUCCESS;
  }
  // SK 分区后消费子图走通用调度，多 Reduce 复合区域（如 LayerNorm 的
  // mean/sum 链）在通用路径会生成 R 轴切分模板，产生跨循环变量引用等
  // 未支持形态；暂时限制 SK 仅支持单 Reduce 场景，复合区域由
  // SIMD/SIMT/VectorFunc 路径处理，待 SK 消费子图调度完善后再放开。
  if (!analysis.composite_reduces.empty()) {
    is_candidate_legal = false;  // AnalyzeRewrittenGraph 已置 true，拒绝时必须显式复位
    GELOGI("[IndirectLoad] Reject SK: composite Norm region with %zu Reduce stages is not supported yet.",
           analysis.composite_reduces.size());
    return af::SUCCESS;
  }
  // SK 虽然采用 workspace 分区，但 Norm 区域的 Reduce/Broadcast 对应关系和
  // 轴保持证明仍必须与 SIMD/SIMT 共用，不能因模板不同而跳过语义校验。
  GE_ASSERT_SUCCESS(
      CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, ascir::TemplateId::kIndirectLoadSK));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject SK: composite Norm validation is illegal.");
    return af::SUCCESS;
  }
  is_candidate_legal = false;
  if (!IsSkTemplateCandidateLegal(indirect_load)) {
    GELOGI("[IndirectLoad] Reject SK candidate for node[%s]: candidate legality check failed.",
           indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  is_candidate_legal = true;
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSK));
  GE_ASSERT_SUCCESS(PartitionSkGraph(graph, indirect_load, analysis.align_input_path, analysis.align_index_path));
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis_index, input_inner_axis, af::kIdNone));
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      input_boundary, ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary));
  return af::SUCCESS;
}

// [SK段SIMT化] 平台 UB 与 tensor 逻辑总量（动态 shape 返回 false）。
constexpr int64_t kIndirectLoadUbReservedBytes = 32768;  // SIMT 预留 tmp/sync/dcache 下界
// SIMT post-Reduce 的 R 域 UB 下界系数：输出 tile 双缓冲（2 份）+ Reduce 输入完整
// R 域（1 份），共 3 份 R 域驻留 UB。
constexpr int64_t kSimtRDomainUbFactor = 3;

bool TryGetIndirectLoadPlatformUbSize(int64_t &ub_size) {
  ge::PlatformInfo platform_info;
  if (!ge::PlatformContext::GetInstance().TryGetInitializedPlatformInfo(platform_info)) {
    return false;
  }
  ub_size = platform_info.ub_size;
  return ub_size > 0;
}

bool TryGetTensorLogicalBytes(const af::AscTensor &tensor, int64_t &bytes) {
  int64_t count = 1;
  for (const auto &repeat : tensor.attr.repeats) {
    int64_t value = 0;
    if (!repeat.GetConstValue(value) || value <= 0) {
      return false;  // 动态 shape 保守放行
    }
    count *= value;
  }
  bytes = count * static_cast<int64_t>(af::GetSizeByDataType(tensor.attr.dtype));
  return true;
}

// [SK段SIMT化] SK 行窗口对当前图形态是否退化为全量不可行：
// IndirectLoadSk 设备 API 的窗口 = axis 起完整 payload 后缀 × 窗口行数，index 需
// 全量驻留该段 UB。index 源头不收敛（逐元素独立引用，沿 gather 轴枚举任意行号）
// 时窗口行数必须覆盖 axis 维全部 → 窗口 = input 全量 + index 全量，超出单核 UB
// 即该形态的 SK 不可行（调度期必然被 StaticUbTemplateFilter 淘汰）。
// 源头收敛（引用按行/列成组）的形态窗口只需被引用分片，不受此限。
bool IsSkWindowUbInfeasible(const af::AscNodePtr &indirect_load) {
  // 现场判定收敛性：AccessInfo attr 写在先跑的 SIMD/SIMT 候选图副本上，SK 候选
  // 副本拷贝自原图时为空，跨副本读取会把收敛形态误判为非收敛而误触发重定向
  // （原版 user_layernorm：收敛 index + 大表被误重定向，产生 dtype 错位的 kernel）。
  if (ascgen_utils::indirect_load::IsIndexSourceConverged(indirect_load)) {
    return false;  // 引用成组：窗口按分片装载，不按全量判定
  }
  int64_t ub_size = 0;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;  // 平台信息不可得时保守走原 SK 路径
  }
  const auto &inputs = indirect_load->inputs();
  int64_t input_bytes = 0;
  int64_t index_bytes = 0;
  if (!TryGetTensorLogicalBytes(*inputs[ascgen_utils::indirect_load::kInputTensorIndex], input_bytes) ||
      !TryGetTensorLogicalBytes(*inputs[ascgen_utils::indirect_load::kIndexTensorIndex], index_bytes)) {
    return false;  // 动态 shape 保守走原 SK 路径
  }
  if (input_bytes + index_bytes > ub_size) {
    GELOGI("[IndirectLoad] SK window (generic access) bytes %ld (input %ld + index %ld) exceeds platform UB %ld.",
           input_bytes + index_bytes, input_bytes, index_bytes, ub_size);
    return true;
  }
  return false;
}

// [SK段SIMT化] SIMT 候选自身的 UB 可行性（硬下界）：输出 tile 双缓冲（2R）+
// Reduce 输入完整 R 域（R）+ 预留（32KB），R=IL 输出尾轴尺寸×dtype。SIMT 可行时
// SK 候选无需重定向——重定向是为 SK 行窗口不可行且无其他活路时提供的分段替代；
// SIMT 可用的图（如 softmax 专用形态带 Scalar 标量链/多输入，R 可全载）走原
// SIMT 内联路径，分段形态对其不支持（生产 9.28：SIMT 可行却重定向，边界化误伤
// 标量输入链）。
bool IsSimtUbInfeasible(const af::AscNodePtr &indirect_load) {
  int64_t ub_size = 0;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return false;
  }
  // outputs() 每次调用重建内部快照，先前返回的引用会悬垂——一次性拷贝所需数据。
  const auto outputs = indirect_load->outputs();
  if (outputs.empty() || outputs[0] == nullptr || outputs[0]->attr.repeats.empty()) {
    return false;
  }
  const auto output_repeats = outputs[0]->attr.repeats;
  const auto output_dtype = outputs[0]->attr.dtype;
  int64_t tail = 0;
  if (!output_repeats.back().GetConstValue(tail) || tail <= 0) {
    return false;  // 动态 shape 保守不重定向
  }
  const int64_t r_bytes = tail * static_cast<int64_t>(af::GetSizeByDataType(output_dtype));
  return kSimtRDomainUbFactor * r_bytes + kIndirectLoadUbReservedBytes > ub_size;
}

// [SK段SIMT化] 重定向的形态约束：IL 输出的下游闭包（DAG，LayerNorm 的
// x-mean 为菱形双消费）全部由 Cast/Reduce/Broadcast/Elementwise/Scalar/Store/
// Output 构成、且至少一个 Reduce 时允许重定向——即 Norm 类复合区域结构本身。
// 这是 R 非全载需要分段归约的目标形态：IL 段 SIMT gather 写 output workspace，
// 消费子图（单/多 Reduce 的 Norm DAG）独立调度，R 轴分 tile 由其自身 tiling 承载。
// 闭包含 IL/Transpose 等特殊算子或控制边时维持原 SK 行为不重定向。
bool IsNormRegionDag(const af::AscNodePtr &indirect_load) {
  std::vector<af::AscNodePtr> pending;
  std::unordered_set<const af::AscNode *> visited;
  int32_t reduce_count = 0;
  bool seen_store = false;
  // 起点为 IL 本身（不在白名单），仅检查其下游闭包。
  for (const auto &consumer : indirect_load->GetOutDataNodes()) {
    pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
  }
  for (size_t cursor = 0UL; cursor < pending.size() && cursor < 256UL; ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    const bool is_allowed = af::ops::IsOps<af::ascir_op::Cast>(node) || ScheduleUtils::IsReduce(node) ||
                            af::ops::IsOps<af::ascir_op::Broadcast>(node) ||
                            af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Output>(node) ||
                            af::ops::IsOps<af::ascir_op::Scalar>(node) ||
                            af::ops::IsOps<af::ascir_op::ScalarData>(node) || ScheduleUtils::IsElewise(node);
    if (!is_allowed) {
      return false;
    }
    if (ScheduleUtils::IsReduce(node)) {
      ++reduce_count;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      seen_store = true;
    }
    for (const auto &consumer : node->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
    }
  }
  // [SK段SIMT化] 多 Reduce 复合区域（LayerNorm mean/var）的消费子图在通用调度
  // 生成 R 轴切分模板后，VF 节点跨循环引用统计链 tensor（codegen
  // ConnectApiCallInputs 找不到生产 ApiCall，实测 id[%d] 缺失），为通用 codegen
  // 的结构性缺口——维持单 Reduce 限制；多 Reduce 分段需消费子图内部再分段
  // （workspace 链），作为后续专项。
  return reduce_count >= 1 && seen_store;
}

// [SK段SIMT化·复合Norm] Norm 链分段化。通用调度对含 Reduce 的连通图按 R 轴切
// loop，Reduce/多消费者节点输出的下游若与其同分量，跨 loop 引用（codegen
// ConnectApiCallInputs 在单 loop 的 tensor_calls 中找不到生产者）。且连通性分区
// 为双向遍历（上下游都走），LayerNorm 的 x-mean 菱形（diff 被方差链与输出链双
// 消费）使单一 Reduce 边界无法切断分量。规则：对每个 Reduce 节点、每个多消费
// 者计算节点的输出，逐消费者独立插 workspace 边界（每边一份，彻底切断连通分
// 量）——每个子图数据流自足（输入均为 Load），R 切分不再产生跨循环依赖，与独
// 立 LayerNorm 的统计段/消费段分离形态对齐。单消费者非 Reduce 节点不切（同段
// 内常规流水）。
af::Status PartitionNormReduceBoundaries(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  // [边界化范围] 仅 IL 输出下游闭包（Norm DAG）：输入前置链（Scalar 标量广播等
  // side-input 生产链）不在范围内——对它边界化会使 SIMT 流程的输入回溯
  // （CompleteInputDataTensorAttrs：Load 的 producer 必须是 Data）遇到边界
  // load（producer 为 workspace）而失败（生产 9.28：scalar1 多消费者被误切）。
  std::unordered_set<const af::AscNode *> downstream;
  {
    std::vector<af::AscNodePtr> pending;
    for (const auto &consumer : indirect_load->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
    }
    for (size_t cursor = 0UL; cursor < pending.size() && cursor < 256UL; ++cursor) {
      const af::AscNodePtr node = pending[cursor];
      if (node == nullptr || !downstream.emplace(node.get()).second) {
        continue;
      }
      for (const auto &consumer : node->GetOutDataNodes()) {
        pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
      }
    }
  }
  std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> boundary_targets;
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || node == indirect_load || downstream.find(node.get()) == downstream.end()) {
      continue;  // 仅 IL 下游闭包内的节点；IL 输出由 PartitionSkGraph 的 output 边界处理
    }
    const bool is_boundary_node =
        af::ops::IsOps<af::ascir_op::Workspace>(node) || af::ops::IsOps<af::ascir_op::Store>(node) ||
        af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Data>(node) ||
        af::ops::IsOps<af::ascir_op::Output>(node);
    if (is_boundary_node) {
      continue;
    }
    const auto out_anchor = node->GetOutDataAnchor(0UL);
    if (out_anchor == nullptr) {
      continue;
    }
    const auto &peers = out_anchor->GetPeerInDataAnchors();
    if (peers.empty()) {
      continue;
    }
    bool has_non_store_consumer = false;
    for (const auto &peer : peers) {
      const auto consumer = peer != nullptr && peer->GetOwnerNode() != nullptr
                                ? std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode())
                                : nullptr;
      if (consumer != nullptr && !af::ops::IsOps<af::ascir_op::Store>(consumer)) {
        has_non_store_consumer = true;
        break;
      }
    }
    if (!has_non_store_consumer) {
      continue;  // 输出直连 Store（段尾）
    }
    if (ScheduleUtils::IsReduce(node) || peers.size() > 1UL) {
      for (const auto &peer : peers) {
        const auto consumer = peer != nullptr && peer->GetOwnerNode() != nullptr
                                  ? std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode())
                                  : nullptr;
        if (consumer != nullptr) {
          boundary_targets.emplace_back(node, consumer);
        }
      }
    }
  }
  size_t boundary_seq = 0UL;
  for (const auto &target : boundary_targets) {
    const auto &node = target.first;
    const auto &consumer = target.second;
    const auto out_anchor = node->GetOutDataAnchor(0UL);
    // 按消费者定位其当前连到本生产者输出的输入边（边被逐条改接，重新查找）。
    af::InDataAnchorPtr peer_anchor = nullptr;
    const auto consumer_inputs = consumer->GetInDataNodes();
    for (size_t in_idx = 0UL; in_idx < consumer_inputs.size(); ++in_idx) {
      const auto in_anchor = consumer->GetInDataAnchor(in_idx);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() == out_anchor) {
        peer_anchor = in_anchor;
        break;
      }
    }
    if (peer_anchor == nullptr) {
      continue;  // 该边已被此前的边界化改接
    }
    const std::string boundary_name = node->GetName() + "_sk_n" + std::to_string(boundary_seq);
    GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, node, 0UL, consumer, static_cast<size_t>(peer_anchor->GetIdx()),
                                              boundary_name, false, false));
    ++boundary_seq;
    GELOGI("[IndirectLoad] SK norm boundary[%s] inserted (node[%s] consumer[%s]).", boundary_name.c_str(),
           node->GetNamePtr(), consumer->GetNamePtr());
  }
  return af::SUCCESS;
}

af::Status ApplyGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                          bool &is_candidate_legal, bool has_live_candidate = true) {
  GELOGD("[IndirectLoad] Apply graph pass for node[%s], template_id[%d].", indirect_load->GetNamePtr(),
         static_cast<int32_t>(template_id));
  is_candidate_legal = true;
  const auto value_dtype = indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype;
  const bool is_64bit_value = value_dtype == af::DT_INT64 || value_dtype == af::DT_UINT64;
  const bool is_1byte_value = value_dtype == af::DT_INT8 || value_dtype == af::DT_UINT8 || value_dtype == af::DT_BOOL;
  // 64 位值（int64/uint64）由 SIMD 8 字节 ValuePolicy（RegTraitNumTwo）与 SIMT 支持；
  // 1 字节值（int8/uint8/bool）当前仅 SIMT 标量路径支持（SIMD b8 gather 语义待定）。
  // SK 的 AscendC::Gather 对 8 字节与 1 字节均不支持，需跳过。
  if ((is_64bit_value || is_1byte_value) && template_id == ascir::TemplateId::kIndirectLoadSK) {
    is_candidate_legal = false;
    GELOGI("[IndirectLoad] value dtype[%d] is unsupported by SK, skip template[%d] for node[%s].",
           static_cast<int32_t>(value_dtype), static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (is_1byte_value && template_id != ascir::TemplateId::kIndirectLoadSimt) {
    is_candidate_legal = false;
    GELOGI("[IndirectLoad] 1-byte value dtype[%d] only supports SIMT, skip template[%d] for node[%s].",
           static_cast<int32_t>(value_dtype), static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    // [SK段SIMT化] index 源头不收敛且全量窗口超出单核 UB 的形态（如
    // gather(axis=0) 逐元素 index + 尾轴大 R）：IndirectLoadSk 行窗口模型退化为
    // 全量不可行。保留 SK 的 output workspace 边界（消费子图/Reduce 段独立调度，
    // 承载 R 轴分 tile 归约），IL 段整体重定向为 SIMT 模板走全套流程（input/
    // index GM 直读，输出 tile 写 output workspace）——SK 分段设计对 R 非全载
    // 场景的本意。复合 Norm 区域（多 Reduce）维持既有拒绝语义。
    // 门禁放宽：SIMT 候选可能因布局校验被拒（如 mean(dim=-2) 中间轴归约，
    // IsSimtUbInfeasible 只看尾轴 UB 下界无法覆盖）——前序 SIMD/SIMT 候选全灭时
    // SK 是最后活路，此时即使 SIMT 的 UB 下界可行也必须重定向。
    if (IsSkWindowUbInfeasible(indirect_load) && IsNormRegionDag(indirect_load) &&
        (IsSimtUbInfeasible(indirect_load) || !has_live_candidate)) {
      GELOGI("[IndirectLoad] SK candidate redirects IL to SIMT (window exceeds UB), node[%s].",
             indirect_load->GetNamePtr());
      // 先 Norm 边界化再 output 边界：闭包按 IL 下游数据流计算，output 边界插入后
      // IL 直接消费者变为边界 store，Norm 链被隔断在闭包之外（生产 9.28 复验：
      // 边界 0 个、分区 0 子图、has_none_graph）。Norm 边界先行，IL 直连 Norm 链首，
      // 闭包完整；随后 output 边界插在 IL 与 Norm 链首之间。
      GE_ASSERT_SUCCESS(PartitionNormReduceBoundaries(graph, indirect_load));
      GE_ASSERT_SUCCESS(PartitionSkGraph(graph, indirect_load, false, false, /*skip_input_boundaries=*/true));
      template_id = ascir::TemplateId::kIndirectLoadSimt;
      GE_ASSERT_SUCCESS(ascir::SetSkSegmentedSimt(indirect_load, true));
    } else {
      return ApplySkGraphPass(graph, indirect_load, is_candidate_legal);
    }
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(ClearSimtTemplateRoles(graph));
  }
  RewrittenGraphAnalysis analysis;
  GE_ASSERT_SUCCESS(AnalyzeRewrittenGraph(graph, indirect_load, template_id, is_candidate_legal, analysis));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analysis));

  // The SIMT template overwrites this boundary from its post-reduce layout; without a post Reduce the
  // full output rank is the boundary. The SIMD template does not use this value.
  size_t boundary = indirect_load->outputs()[0]->attr.axis.size();
  GE_ASSERT_SUCCESS(ValidateTemplate(indirect_load, template_id, analysis, boundary, is_candidate_legal));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject template[%d] for node[%s]: post-reduce or region validation failed.",
           static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateTemplate(indirect_load, template_id, analysis));
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    NodeSet normal_schedule_nodes;
    GE_ASSERT_SUCCESS(AnnotateSimtFanoutBranches(indirect_load, analysis, normal_schedule_nodes));
    // 豁免集即 lowering 侧 BuildSimtLoweringMetadata 会校验的 index_region，
    // 其中与 normal_schedule 闭包重叠的共享节点不能被兜底清理掉角色。
    NodeSet lowering_exempt;
    for (const auto &region_node : analysis.index_region) {
      lowering_exempt.emplace(region_node.get());
    }
    GE_ASSERT_SUCCESS(ClearNormalScheduleSimtRoles(graph, normal_schedule_nodes, lowering_exempt));
  }
  GE_ASSERT_SUCCESS(NormalizeTemplateAxes(graph, indirect_load, template_id, analysis, boundary));
  GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph, indirect_load));
  GE_ASSERT_SUCCESS(FinalizeTemplate(indirect_load, template_id));
  bool metadata_supported = false;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
  if (!metadata_supported) {
    is_candidate_legal = false;
  }
  return af::SUCCESS;
}

bool IsEmbeddingFastPathCapable(const TemplateCase &template_case,
                                const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info) {
  if (template_case.implementation != ascgen_utils::indirect_load::Implementation::kDefault) {
    return false;
  }
  if (template_case.template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return access_info.can_use_simt_structured;
  }
  if (template_case.template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return access_info.can_use_simd_embedding;
  }
  return false;
}

std::string GenerateScoreFunc(const TemplateCase &template_case,
                              const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info,
                              const af::AscNodePtr &candidate_node) {
  int32_t score = 0;
  if (IsEmbeddingFastPathCapable(template_case, access_info)) {
    int64_t input_slice_bytes = 0L;
    int64_t lookup_count = 0L;
    const bool static_cost_inputs = access_info.input_slice_bytes.GetConstValue(input_slice_bytes) &&
                                    access_info.index_varying_extent.GetConstValue(lookup_count);
    if (!static_cost_inputs) {
      // Dynamic sizes still get a positive fast-path score.  With both paths
      // available the existing candidate order is retained until a runtime
      // shape-aware model is added.
      score = kEmbeddingAlternateFastPathScore;
    } else {
      const bool simd_preferred = input_slice_bytes >= kEmbeddingSimdPayloadBytesThreshold &&
                                  lookup_count >= kEmbeddingSimdLookupCountThreshold;
      if (template_case.template_id == ascir::TemplateId::kIndirectLoadSimd) {
        score = simd_preferred ? kEmbeddingFastPathScore : kEmbeddingAlternateFastPathScore;
      } else {
        score = simd_preferred ? kEmbeddingAlternateFastPathScore : kEmbeddingFastPathScore;
      }
    }
  }
  // Norm 区域候选：Softmax 专用路径复用既有专用 API 与 tmp，给予小幅偏好，
  // 使同分场景下优先选择可直接融合的形态；通用复合区域不加分，
  // 待完整多阶段执行协议落地后再纳入评分模型。
  if (candidate_node != nullptr) {
    ascgen_utils::norm::NormInfo norm_info;
    GE_ASSERT_SUCCESS(ascgen_utils::norm::TryGetNormInfo(candidate_node, norm_info));
    if (norm_info.kind == ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated) {
      score += 1;
    }
  }
  return "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
         "  (void)tiling_data;\n"
         "  return " +
         std::to_string(score) +
         ";\n"
         "}\n";
}

af::Status PartitionTaskGraph(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load, bool is_sk_template,
                              std::vector<ascir::ImplGraph> &grouped_graphs) {
  if (!is_sk_template) {
    return ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs);
  }
  std::vector<af::AscNodePtr> node_order;
  GE_ASSERT_SUCCESS(BuildSkPartitionOrder(graph, indirect_load, node_order));
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs, node_order));
  return RestoreSkTemplateAxes(grouped_graphs);
}

af::Status RefreshTaskAxisSizes(std::vector<ascir::ImplGraph> &grouped_graphs, bool need_update_axis,
                                bool is_sk_template) {
  if ((!need_update_axis && !is_sk_template) || grouped_graphs.size() <= 1UL) {
    return af::SUCCESS;
  }
  for (auto &subgraph : grouped_graphs) {
    if (ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Concat>(subgraph) != nullptr) {
      continue;
    }
    af::AscNodePtr subgraph_indirect_load;
    if (is_sk_template) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(subgraph, subgraph_indirect_load));
    }
    if (subgraph_indirect_load == nullptr) {
      GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::RefreshAxisSize(subgraph));
    }
  }
  return af::SUCCESS;
}

af::Status ReduceTaskGraphCount(std::vector<ascir::ImplGraph> &grouped_graphs, const OptimizerOptions &options) {
  if (options.graph_type != GraphType::kFusedAscBackend) {
    return af::SUCCESS;
  }
  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return ScheduleGroupGraphPartitioner::ReduceGraphCount(grouped_graphs, backend_spec->max_group_num_per_compile_unit);
}

af::Status FinalizeGroupedGraphLoweringMetadata(std::vector<ascir::ImplGraph> &grouped_graphs) {
  for (auto &graph : grouped_graphs) {
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    if (indirect_load == nullptr) {
      continue;
    }
    const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
    // FinalizeLoweringMetadata 仅支持 SIMD/SIMT：SK 的分区图不含该 lowering
    // 元数据，不进入本循环的元数据检查。
    if (template_id != ascir::TemplateId::kIndirectLoadSimd && template_id != ascir::TemplateId::kIndirectLoadSimt) {
      continue;
    }
    // Norm 区域完整性：区域节点必须与 IndirectLoad 同组，否则统计分支与
    // 原始输入回流被拆到不同 Kernel。SK 除外：SK 的 workspace 分区本身就是
    // 把 Gather 与后续计算合法拆到不同 grouped graph，区域节点位于其他分区
    // 属预期形态，仅记录日志不阻断。
    ascgen_utils::norm::NormInfo norm_info;
    GE_ASSERT_SUCCESS(ascgen_utils::norm::TryGetNormInfo(indirect_load, norm_info));
    if (norm_info.kind != ascgen_utils::norm::NormInfo::Kind::kNone && !norm_info.region_node_names.empty()) {
      const bool is_sk_split = template_id == ascir::TemplateId::kIndirectLoadSK;
      for (const auto &node_name : norm_info.region_node_names) {
        if (graph.FindNode(node_name.c_str()) == nullptr) {
          GE_ASSERT_TRUE(!is_sk_split, "Norm region node[%s] is split out of grouped graph[%s], IndirectLoad[%s].",
                         node_name.c_str(), graph.GetName().c_str(), indirect_load->GetNamePtr());
          GELOGI("[IndirectLoad] Norm region node[%s] resides in another SK workspace partition of graph[%s].",
                 node_name.c_str(), graph.GetName().c_str());
        }
      }
    }
    bool metadata_supported = false;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
    GE_ASSERT_TRUE(metadata_supported, "IndirectLoad lowering metadata is unsupported in grouped graph[%s], node[%s].",
                   graph.GetName().c_str(), indirect_load->GetNamePtr());
  }
  return af::SUCCESS;
}
}  // namespace

Status IndirectLoadScheduleCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                                   std::vector<std::string> &score_functions) {
  af::AscNodePtr indirect_load;
  GE_ASSERT_SUCCESS(CollectAndValidateIndirectLoadNodes(graph, indirect_load));
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(ValidateIndirectLoadNode(indirect_load));
  GELOGI("[IndirectLoad] Generate schedule candidates for graph[%s], node[%s].", graph.GetName().c_str(),
         indirect_load->GetNamePtr());
  const std::string indirect_load_name = indirect_load->GetName();
  const TemplateCase cases[] = {
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kGatherApi},
      {ascir::TemplateId::kIndirectLoadSimt, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSK, ascgen_utils::indirect_load::Implementation::kDefault}};
  for (const TemplateCase &template_case : cases) {
    const ascir::TemplateId template_id = template_case.template_id;
    if (HasReduceBeforeGatherAxis(indirect_load) && template_id != ascir::TemplateId::kIndirectLoadSK) {
      GELOGI("[IndirectLoad] Skip non-SK candidate[%d] for R-before-gather-axis post-Reduce layout, node[%s].",
             static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
      continue;
    }
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    const af::AscNodePtr candidate_indirect_load = candidate_graph.FindNode(indirect_load_name.c_str());
    GE_ASSERT_NOTNULL(candidate_indirect_load, "Failed to find copied IndirectLoad node[%s].",
                      indirect_load_name.c_str());
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetImplementation(candidate_indirect_load, template_case.implementation));
    // 候选图副本内规范化直接后置稳定 Softmax Pattern：主图因 Reduce Generator
    // 遇 IndirectLoad 提前返回而不会执行 Softmax 替换，必须在副本内完成，
    // 使 CollectOutputBoundaries 看到的是专用 Softmax 节点而非 Max/Sum 双 Reduce。
    // 仅替换原始输入直接来自本 IndirectLoad 输出的 Pattern；Pattern 未命中或
    // 中间有 Elementwise 时保持原图形态，由通用复合区域路径处理。
    bool softmax_normalized = false;
    GE_ASSERT_SUCCESS(
        softmax_pattern::NormalizeDirectPostSoftmax(candidate_graph, candidate_indirect_load, softmax_normalized));
    if (softmax_normalized) {
      // 替换后候选图中节点名不变（Softmax 以原 TrueDiv 名+"_softmax"重建），
      // 后续 ApplyGraphPass 按模板重新执行改图与校验。
      GELOGI("[IndirectLoad] Candidate[%d] normalized direct post Softmax for node[%s].",
             static_cast<int32_t>(template_id), candidate_indirect_load->GetNamePtr());
    }
    bool is_candidate_legal = false;
    GE_ASSERT_SUCCESS(
        ApplyGraphPass(candidate_graph, candidate_indirect_load, template_id, is_candidate_legal, !graphs.empty()));
    if (!is_candidate_legal) {
      GELOGW("[IndirectLoad] Skip illegal template candidate[%d] for node[%s].", static_cast<int32_t>(template_id),
             candidate_indirect_load->GetNamePtr());
      continue;
    }
    ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(candidate_indirect_load, access_info));
    graphs.emplace_back(std::move(candidate_graph));
    score_functions.emplace_back(GenerateScoreFunc(template_case, access_info, candidate_indirect_load));
    GELOGI("[IndirectLoad] Add schedule candidate[%d, %d] for node[%s].", static_cast<int32_t>(template_id),
           static_cast<int32_t>(template_case.implementation), candidate_indirect_load->GetNamePtr());
  }
  return af::SUCCESS;
}

Status IndirectLoadScheduleCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph,
                                                        std::vector<ScheduleTask> &tasks,
                                                        const OptimizerOptions &options) {
  bool need_update_axis = false;
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(optimize_graph, need_update_axis));
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(Generate(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  for (size_t i = 0UL; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i]};
    task.has_load_store_conversion = HasLoadStoreConversion();
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    // [SK分段SIMT] 重定向形态（template 已是 SIMT 但图含 output workspace 边界）与
    // SK 走同一分区/恢复路径（分区后子图需重新执行模板归一）。
    const bool is_sk_template =
        indirect_load != nullptr &&
        (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSK ||
         (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimt &&
          ascir::IsSkSegmentedSimt(*indirect_load)));
    GE_CHK_STATUS_RET(PartitionTaskGraph(graph, indirect_load, is_sk_template, task.grouped_graphs),
                      "Failed to partition graph");
    GE_ASSERT_SUCCESS(RefreshTaskAxisSizes(task.grouped_graphs, need_update_axis, is_sk_template));
    GE_CHK_STATUS_RET(ReduceTaskGraphCount(task.grouped_graphs, options), "Failed to reduce graph count");
    GE_ASSERT_SUCCESS(FinalizeGroupedGraphLoweringMetadata(task.grouped_graphs));
    tasks.emplace_back(std::move(task));
  }
  return af::SUCCESS;
}

}  // namespace optimize
