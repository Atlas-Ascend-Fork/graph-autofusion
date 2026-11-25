/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_graph_unfolder.h"

#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/gnode_af.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "ascir_utils.h"
#include "ascgraph_info_complete.h"

namespace optimize {
namespace {
const af::AscTensorAttr *FindBoundaryTensorAttr(const af::AscGraph &graph, const int32_t index, const bool is_output) {
  // External anchor indices are mirrored by the internal Data/Output ir_attr index.
  for (const auto &node : graph.GetAllNodes()) {
    const bool is_expected_node =
        is_output ? af::ops::IsOps<af::ascir_op::Output>(node) : ScheduleUtils::IsDataInput(node);
    if (!is_expected_node) {
      continue;
    }
    int64_t node_index = -1;
    if (ScheduleUtils::GetNodeIrAttrIndex(node, node_index) != af::SUCCESS || node_index != index) {
      continue;
    }
    const auto tensors = is_output ? node->inputs() : node->outputs();
    if (tensors.empty()) {
      return nullptr;
    }
    return &tensors[0]->attr;
  }
  return nullptr;
}

}  // namespace

bool FusedGraphUnfolder::BuildGraphAxisMapping(const af::AscGraph &source_graph, const af::AscTensorAttr &source_attr,
                                               const af::AscGraph &target_graph, const af::AscTensorAttr &target_attr,
                                               const std::vector<size_t> &target_to_global,
                                               std::vector<size_t> &source_to_global) {
  const auto source_axes = source_graph.GetAllAxis();
  const auto target_axes = target_graph.GetAllAxis();
  if (target_axes.size() != target_to_global.size()) {
    return false;
  }
  std::map<af::AxisId, size_t> source_axis_to_index;
  std::map<af::AxisId, size_t> target_axis_to_index;
  if (!BuildAxisIndex(source_axes, source_axis_to_index) || !BuildAxisIndex(target_axes, target_axis_to_index)) {
    return false;
  }
  return ComposeGraphAxisMapping(source_attr, target_attr, source_axis_to_index, target_axis_to_index, target_to_global,
                                 source_to_global);
}

//  AxisId -> graph axis index
bool FusedGraphUnfolder::BuildAxisIndex(const std::vector<af::AxisPtr> &axes,
                                        std::map<af::AxisId, size_t> &axis_to_index) {
  for (size_t index = 0UL; index < axes.size(); ++index) {
    if (axes[index] == nullptr || !axis_to_index.emplace(axes[index]->id, index).second) {
      return false;
    }
  }
  return true;
}

// 把局部 tensor 映射组合成图级映射
bool FusedGraphUnfolder::ComposeGraphAxisMapping(const af::AscTensorAttr &source_attr,
                                                 const af::AscTensorAttr &target_attr,
                                                 const std::map<af::AxisId, size_t> &source_axis_to_index,
                                                 const std::map<af::AxisId, size_t> &target_axis_to_index,
                                                 const std::vector<size_t> &target_to_global,
                                                 std::vector<size_t> &source_to_global) {
  AxisMappingResult local_mapping;
  if (BuildLocalAxisMapping(source_attr, target_attr, local_mapping) != af::SUCCESS) {
    return false;
  }
  const auto unmapped_axis = std::numeric_limits<size_t>::max();
  source_to_global.assign(source_axis_to_index.size(), unmapped_axis);
  std::set<size_t> mapped_global_axes;
  // Tensor axes are local to a graph. Convert through the target graph before using global positions.
  for (size_t index = 0UL; index < source_attr.axis.size(); ++index) {
    const auto source_iter = source_axis_to_index.find(source_attr.axis[index]);
    const auto target_iter = target_axis_to_index.find(target_attr.axis[local_mapping.old_to_global[index]]);
    if (source_iter == source_axis_to_index.end() || target_iter == target_axis_to_index.end()) {
      return false;
    }
    const auto global_index = target_to_global[target_iter->second];
    if (source_to_global[source_iter->second] != unmapped_axis &&
        source_to_global[source_iter->second] != global_index) {
      return false;
    }
    if (!mapped_global_axes.emplace(global_index).second && source_to_global[source_iter->second] == unmapped_axis) {
      return false;
    }
    source_to_global[source_iter->second] = global_index;
  }
  return true;
}

// 合并同一个 AscGraph 从多条边得到的映射约束
bool FusedGraphUnfolder::MergeGraphAxisMapping(std::vector<size_t> &existing_mapping,
                                               const std::vector<size_t> &new_mapping) {
  const auto unmapped_axis = std::numeric_limits<size_t>::max();
  if (existing_mapping.empty()) {
    existing_mapping = new_mapping;
    return true;
  }
  if (existing_mapping.size() != new_mapping.size()) {
    return false;
  }
  for (size_t index = 0UL; index < existing_mapping.size(); ++index) {
    if (new_mapping[index] == unmapped_axis) {
      continue;
    }
    if (existing_mapping[index] != unmapped_axis && existing_mapping[index] != new_mapping[index]) {
      return false;
    }
    existing_mapping[index] = new_mapping[index];
  }
  return true;
}

// 检查图级映射是否还有未映射轴
bool FusedGraphUnfolder::IsGraphAxisMappingComplete(const std::vector<size_t> &mapping) {
  const auto unmapped_axis = std::numeric_limits<size_t>::max();
  return std::none_of(mapping.begin(), mapping.end(),
                      [unmapped_axis](const size_t index) { return index == unmapped_axis; });
}

// 校验 AscTensorAttr 的基本结构是否一致
bool FusedGraphUnfolder::IsTensorAttrValid(const af::AscTensorAttr &attr) {
  return attr.axis.size() == attr.repeats.size() && attr.axis.size() == attr.strides.size();
}

// 判断 source 某一维能否匹配 target 某一维
// repeats 必须静态相等。
// 如果 source repeat 静态等于 1，则 stride 可以不同，因为该维索引恒为 0，不影响地址。
// 如果 source repeat 不是 1，则 stride 也必须静态相等。
bool FusedGraphUnfolder::IsAxisMatch(const af::AscTensorAttr &source_attr, const size_t source_index,
                                     const af::AscTensorAttr &target_attr, const size_t target_index) {
  if (af::SymbolicUtils::StaticCheckEq(source_attr.repeats[source_index], target_attr.repeats[target_index]) !=
      af::TriBool::kTrue) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckEq(source_attr.repeats[source_index], af::sym::kSymbolOne) ==
             af::TriBool::kTrue ||
         af::SymbolicUtils::StaticCheckEq(source_attr.strides[source_index], target_attr.strides[target_index]) ==
             af::TriBool::kTrue;
}

// 受限 DFS，搜索 source tensor 维度到 target tensor 维度的保序映射：
void FusedGraphUnfolder::SearchLocalAxisMappings(const af::AscTensorAttr &source_attr,
                                                 const af::AscTensorAttr &target_attr, const size_t source_index,
                                                 const size_t target_index, std::vector<size_t> &mapping,
                                                 std::vector<std::vector<size_t>> &candidates) {
  // 找到第二个候选即可判定歧义，避免组合爆炸
  // 剩余 source 维度数 > 剩余 target 维度数 也直接剪枝返回
  if (candidates.size() == 2UL || source_attr.axis.size() - source_index > target_attr.axis.size() - target_index) {
    return;
  }

  // source_index 到达末尾 → 检查剩余 target 维度均为 size-1，若是则记录一个候选
  if (source_index == source_attr.axis.size()) {
    for (size_t index = target_index; index < target_attr.axis.size(); ++index) {
      if (af::SymbolicUtils::StaticCheckEq(target_attr.repeats[index], af::sym::kSymbolOne) != af::TriBool::kTrue) {
        return;
      }
    }
    candidates.push_back(mapping);
    return;
  }
  if (target_index == target_attr.axis.size()) {
    return;
  }
  if (IsAxisMatch(source_attr, source_index, target_attr, target_index)) {
    mapping.push_back(target_index);
    SearchLocalAxisMappings(source_attr, target_attr, source_index + 1UL, target_index + 1UL, mapping, candidates);
    mapping.pop_back();
  }
  if (af::SymbolicUtils::StaticCheckEq(target_attr.repeats[target_index], af::sym::kSymbolOne) == af::TriBool::kTrue) {
    // Only unit target axes may be inserted without changing the source address calculation.
    SearchLocalAxisMappings(source_attr, target_attr, source_index, target_index + 1UL, mapping, candidates);
  }
}

// 局部映射的完整入口，将 DFS 搜索结果转换为状态枚举。
Status FusedGraphUnfolder::BuildLocalAxisMapping(const af::AscTensorAttr &source_attr,
                                                 const af::AscTensorAttr &target_attr, AxisMappingResult &result) {
  result = {};
  if (!IsTensorAttrValid(source_attr) || !IsTensorAttrValid(target_attr)) {
    result.reason = AxisMappingFailureReason::kInvalidTensorAttr;
    return af::FAILED;
  }
  if (source_attr.axis.size() > target_attr.axis.size()) {
    result.reason = AxisMappingFailureReason::kInvalidRank;
    return af::FAILED;
  }

  std::vector<std::vector<size_t>> candidates;
  std::vector<size_t> mapping;
  SearchLocalAxisMappings(source_attr, target_attr, 0UL, 0UL, mapping, candidates);

  if (candidates.empty()) {
    result.reason = AxisMappingFailureReason::kNonUnitInsertedAxis;
    return af::FAILED;
  }
  if (candidates.size() > 1UL) {
    result.status = AxisMappingStatus::kAmbiguous;
    result.reason = AxisMappingFailureReason::kMultipleMappings;
    return af::FAILED;
  }
  result.status = AxisMappingStatus::kSuccess;
  result.old_to_global = std::move(candidates.front());
  result.inserted_axes.assign(target_attr.axis.size(), true);
  for (const auto index : result.old_to_global) {
    result.inserted_axes[index] = false;
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::RemoveUnusedNode(const af::ComputeGraphPtr &graph, const af::NodePtr &node,
                                            const bool force) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(node);
  if (force && !node->GetAllOutDataAnchors().empty()) {
    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    out_anchor->UnlinkAll();
  }
  if (node->GetOutNodes().empty()) {
    GELOGI("%s node [%s] is unused, remove it.", node->GetTypePtr(), node->GetNamePtr());
    af::NodeUtils::UnlinkAll(*node);
    af::GraphUtils::RemoveNodeWithoutRelink(graph, node);
    return af::SUCCESS;
  }
  GELOGD("%s node [%s] has %zu outputs, keep it.", node->GetTypePtr(), node->GetNamePtr(), node->GetOutNodes().size());
  return af::SUCCESS;
}

Status FusedGraphUnfolder::TransferInControlEdges(const std::set<af::NodePtr> &src_nodes,
                                                  af::Node *const &asc_backend) {
  for (auto &src_node : src_nodes) {
    const auto &in_nodes = src_node->GetInAllNodes();
    const std::set<af::NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (auto &ctrl_node : asc_backend->GetInControlNodes()) {
      GE_CHECK_NOTNULL(ctrl_node);
      if (in_node_set.count(ctrl_node) > 0UL) {
        continue;
      }
      GELOGI("[%s] Restore control edge to [%s]", ctrl_node->GetName().c_str(), src_node->GetName().c_str());
      // 添加连边
      af::GraphUtils::AddEdge(ctrl_node->GetOutControlAnchor(), src_node->GetInControlAnchor());
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MergeInputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend) {
  GE_CHECK_NOTNULL(asc_backend);
  std::set<af::NodePtr> src_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != af::DATA) {
      if (node->GetInAllNodes().empty()) {
        (void)src_nodes.emplace(node);
      }
      continue;
    }
    int64_t index = -1;
    (void)ScheduleUtils::GetNodeIrAttrIndex(node, index);
    GE_ASSERT_TRUE(index >= 0, "Ir attr index:[%ld] is invalid, node:[%s], expect >= 0.", index, node->GetNamePtr());
    const auto parent_node_in_anchor = asc_backend->GetInDataAnchor(static_cast<int32_t>(index));
    GE_CHECK_NOTNULL(parent_node_in_anchor, " Parent output anchor is nullptr, data:[%s], index:[%d].",
                     node->GetNamePtr(), static_cast<int32_t>(index));
    const auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
    if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNodeBarePtr() == nullptr)) {
      continue;
    }
    parent_node_in_anchor->UnlinkAll();
    // link src to outputs of DataNode
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto &in_nodes = dst_node->GetInDataNodes();
        if (std::all_of(in_nodes.begin(), in_nodes.end(),
                        [](const af::NodePtr &n) { return n->GetType() == af::DATA; })) {
          (void)src_nodes.emplace(dst_node);
        }
        GE_CHK_STATUS_RET(af::GraphUtils::ReplaceEdgeSrc(out_data_anchor, peer_in_anchor, src_out_anchor),
                          "[Replace][DataEdge] failed");
      }
    }
    // when unfold AscBackend, if data have control edges, which will be left in final graph
    // which cause topo sort failed.
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_CHECK_NOTNULL(out_control_anchor);
    out_control_anchor->UnlinkAll();
    // remove isolated data node
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveNodeWithoutRelink(graph, node), "Failed to remove data node [%s].",
                      node->GetNamePtr());
  }

  // transfer in control edges to all root nodes
  GE_CHK_STATUS_RET(TransferInControlEdges(src_nodes, asc_backend), "TransferInControlEdges failed.");

  asc_backend->GetInControlAnchor()->UnlinkAll();
  return af::SUCCESS;
}

bool FusedGraphUnfolder::IsSameLoadNode(const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
  // 如果不都是load节点，则返回
  if (!af::ops::IsOps<af::ascir_op::Load>(lhs) || !af::ops::IsOps<af::ascir_op::Load>(rhs)) {
    return false;
  }
  if (lhs->attr.sched.axis != rhs->attr.sched.axis) {
    return false;
  }

  af::Expression cur_load_offset;
  (void)ScheduleUtils::GetNodeIrAttrOffset(rhs, cur_load_offset);
  af::Expression old_load_offset;
  (void)ScheduleUtils::GetNodeIrAttrOffset(lhs, old_load_offset);
  if (af::SymbolicUtils::StaticCheckEq(cur_load_offset, old_load_offset) != af::TriBool::kTrue) {
    return false;
  }

  const auto &lhs_repeats = lhs->outputs[0].attr.repeats;
  const auto &lhs_strides = lhs->outputs[0].attr.strides;
  const auto &rhs_repeats = rhs->outputs[0].attr.repeats;
  const auto &rhs_strides = rhs->outputs[0].attr.strides;

  if ((lhs_repeats.size() != rhs_repeats.size()) || (lhs_strides.size() != rhs_strides.size())) {
    return false;
  }
  for (size_t i = 0UL; i < lhs_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_repeats[i], rhs_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }

  for (size_t i = 0UL; i < lhs_strides.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(lhs_strides[i], rhs_strides[i]) != af::TriBool::kTrue) {
      return false;
    }
  }

  GELOGI("Node(%s) and Node(%s) are same load node.", lhs->GetNamePtr(), rhs->GetNamePtr());
  return true;
}

Status FusedGraphUnfolder::DoSameLoadCse(const af::ComputeGraphPtr &fused_graph) {
  for (const auto &node : fused_graph->GetDirectNodePtr()) {
    GE_CHECK_NOTNULL(node);
    if ((node->GetType() != af::DATA) || (node->GetOutDataNodesSize() <= 1UL)) {
      continue;
    }
    auto output_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchorsPtr();
    auto pre_load_anchor = peer_in_anchors[0UL];
    GE_ASSERT_NOTNULL(pre_load_anchor);
    auto pre_load_node = std::dynamic_pointer_cast<af::AscNode>(pre_load_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(pre_load_node);
    for (size_t i = 1UL; i < peer_in_anchors.size(); ++i) {
      auto rear_load_anchor = peer_in_anchors[i];
      GE_ASSERT_NOTNULL(rear_load_anchor);
      auto rear_load_node = std::dynamic_pointer_cast<af::AscNode>(rear_load_anchor->GetOwnerNode());
      GE_ASSERT_NOTNULL(rear_load_node);
      const bool is_load_same = IsSameLoadNode(pre_load_node, rear_load_node);
      if (is_load_same) {
        // relink load's output
        auto rear_load_out_anchor = rear_load_node->GetOutDataAnchor(0);
        GE_ASSERT_NOTNULL(rear_load_out_anchor);
        auto pre_load_out_anchor = pre_load_node->GetOutDataAnchor(0);
        for (const auto &cur_load_next_in_anchor : rear_load_out_anchor->GetPeerInDataAnchors()) {
          GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(rear_load_out_anchor, cur_load_next_in_anchor));
          GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(pre_load_out_anchor, cur_load_next_in_anchor));
        }
        // Remove cur load
        af::NodeUtils::UnlinkAll(*rear_load_node);
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveNodeWithoutRelink(fused_graph, rear_load_node),
                          "Failed to remove load node [%s].", rear_load_node->GetNamePtr());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::RemoveRedundantLoads(const af::ComputeGraphPtr &graph) {
  for (auto &load_node : graph->GetAllNodes()) {
    // step1: find Load node
    GE_ASSERT_NOTNULL(load_node);
    if (!af::ops::IsOps<af::ascir_op::Load>(load_node)) {
      continue;
    }
    GE_ASSERT_TRUE(load_node->GetInDataNodesSize() == 1UL);  // Load node has only one input.

    // step2: find Output node
    auto output_node = load_node->GetInDataNodes().at(0);
    GE_ASSERT_NOTNULL(output_node);
    if (!af::ops::IsOps<af::ascir_op::Output>(output_node)) {
      continue;
    }
    GE_ASSERT_TRUE(output_node->GetInDataNodesSize() == 1UL);  // Output node has only one input.

    // step3: find Store node
    auto store_node = output_node->GetInDataNodes().at(0);
    GE_ASSERT_NOTNULL(store_node);
    if (!af::ops::IsOps<af::ascir_op::Store>(store_node)) {
      GELOGW("The input of Output node[%s] is %s, not Store node, ", output_node->GetNamePtr(),
             store_node->GetNamePtr());
      continue;
    }
    GE_ASSERT_TRUE(store_node->GetInDataNodesSize() == 1UL);  // Store node has only one input.

    // step4: Pattern like Store+Output+Load can be optimized.
    GELOGD("Find Store+Output+Load pattern: [%s]+[%s]+[%s]", store_node->GetNamePtr(), output_node->GetNamePtr(),
           load_node->GetNamePtr());
    // find the input node of Store
    auto store_in_anchor = store_node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(store_in_anchor);
    auto pre_node_out_anchor = store_in_anchor->GetPeerOutAnchor();  // the input of Store node
    GE_ASSERT_NOTNULL(pre_node_out_anchor);
    auto pre_node = pre_node_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(pre_node);
    for (const auto &load_out_anchor : load_node->GetAllOutDataAnchors()) {
      GE_ASSERT_NOTNULL(load_out_anchor);
      // disconnect Load from its output nodes; And relink input of Store to the output of Load
      for (const auto &peer_in_anchor : load_out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_anchor);
        auto load_output_node = peer_in_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(load_output_node);
        GELOGD("Disconnect %s[%d] and %s[%d], Relink to %s[%d]", load_node->GetNamePtr(), load_out_anchor->GetIdx(),
               load_output_node->GetNamePtr(), peer_in_anchor->GetIdx(), pre_node->GetNamePtr(),
               pre_node_out_anchor->GetIdx());
        GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(load_out_anchor, peer_in_anchor, pre_node_out_anchor));
      }
    }

    // step5: Remove redundant Load, Output and Store
    RemoveUnusedNode(graph, load_node);
    RemoveUnusedNode(graph, output_node);
    RemoveUnusedNode(graph, store_node);
  }
  // step6: Remove NetOutput. Cannot merge two for loop because this depends on the results of previous loop.
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == af::NETOUTPUT) {
      RemoveUnusedNode(graph, node);
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MergeOutputNodes(const af::ComputeGraphPtr &graph, af::Node *const &asc_backend) {
  GE_CHECK_NOTNULL(asc_backend);
  for (auto &output_node : graph->GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Output>(output_node)) {
      continue;
    }
    int64_t index = -1;
    (void)ScheduleUtils::GetNodeIrAttrIndex(output_node, index);
    GE_ASSERT_TRUE((index >= 0), "Get invalid attr index [%ld], node = %s[%s]", index, output_node->GetNamePtr(),
                   output_node->GetTypePtr());

    // Skip relinking when an ascbc node does not have a peer_in_data_anchor, which means that the data will be the
    // output in the ascgraph.
    const af::OutDataAnchorPtr &parent_out_anchor = asc_backend->GetOutDataAnchor(static_cast<int32_t>(index));
    GE_CHECK_NOTNULL(parent_out_anchor, " Parent output anchor is null, output_node:[%s], index:[%d].",
                     output_node->GetNamePtr(), static_cast<int32_t>(index));
    auto peer_in_anchor_in_parent = parent_out_anchor->GetPeerInDataAnchors();

    auto output_anchor = output_node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    for (af::InDataAnchorPtr &dst_in_anchor : peer_in_anchor_in_parent) {
      GE_CHK_STATUS_RET(af::GraphUtils::ReplaceEdgeSrc(parent_out_anchor, dst_in_anchor, output_anchor),
                        "[Replace][DataEdge] failed");
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::UnfoldAscbcNode(af::Node *const &ascbc_node, const af::AscGraph &asc_graph,
                                           const af::ComputeGraphPtr &target_computer_graph) {
  auto graph = af::AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(graph);
  GE_CHK_STATUS_RET(MergeInputNodes(graph, ascbc_node),
                    "[Invoke][MergeInputNodes] Merge data nodes for graph %s failed", graph->GetName().c_str());
  GE_CHK_STATUS_RET(MergeOutputNodes(graph, ascbc_node),
                    "[Invoke][MergeNetOutputNode] Merge net output nodes for graph %s failed",
                    graph->GetName().c_str());
  GELOGI("[%s] Merging graph inputs and outputs successfully", graph->GetName().c_str());
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    (void)target_computer_graph->AddNode(node);
    GELOGI("[%s::%s] added to target graph: [%s].", graph->GetName().c_str(), node->GetName().c_str(),
           target_computer_graph->GetName().c_str());
    (void)node->SetOwnerComputeGraph(target_computer_graph);
  }

  GELOGI("[%s] Done merging graph. remove it from root graph", graph->GetName().c_str());
  GE_LOGW_IF(
      af::GraphUtils::RemoveNodeWithoutRelink(target_computer_graph, ascbc_node->shared_from_this()) != af::SUCCESS,
      "Remove node %s failed, graph:%s.", ascbc_node->GetName().c_str(), target_computer_graph->GetName().c_str())
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ReAssembleOutputIndex(const af::ComputeGraphPtr &fused_graph) {
  for (const auto &node : fused_graph->GetAllNodes()) {
    if (node->GetType() == af::NETOUTPUT) {
      int64_t index = 0;
      for (auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
        GE_ASSERT_NOTNULL(in_data_anchor);
        auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
        if (peer_out_anchor != nullptr) {
          auto asc_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
          GE_ASSERT_NOTNULL(asc_node, "In anchor [%ld]'s peer out anchor [%d] does not have owner node.", index,
                            peer_out_anchor->GetIdx());
          GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Output>(asc_node),
                         "Only output nodes can be directly connected to the netoutput.");
          GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
          auto ir_attr = asc_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
          GE_ASSERT_NOTNULL(ir_attr);
          GE_ASSERT_SUCCESS(ir_attr->SetIndex(index));
          GELOGD("Output node [%s] has been set ir attr index [%ld].", node->GetNamePtr(), index);
          ++index;
        }
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ReAssembleDataIrAttr(const af::ComputeGraphPtr &fused_graph,
                                                const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph) {
  for (const auto &node : fused_graph->GetAllNodes()) {
    if (node->GetType() != af::DATA) {
      continue;
    }
    auto output_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    auto peer_in_anchor = output_anchor->GetPeerInDataAnchorsPtr();
    GE_ASSERT_TRUE(!peer_in_anchor.empty());
    auto peer_first_data_anchor = peer_in_anchor[0UL];
    auto iter = asc_backend_to_asc_graph.find(peer_first_data_anchor->GetOwnerNodeBarePtr());
    GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find AscGraph for data [%s].", node->GetNamePtr());
    auto data_index = peer_first_data_anchor->GetIdx();
    // 存在geir和ascir构图两种可能性
    auto node_attr = node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    auto tensor_attr = af::AscTensorAttr::GetTensorAttrPtr(*output_anchor);
    GE_ASSERT_NOTNULL(tensor_attr);
    int64_t ir_index = -1;
    (void)af::AttrUtils::GetInt(node->GetOpDescBarePtr(), "_parent_node_index", ir_index);
    if (node_attr->ir_attr != nullptr) {
      (void)node_attr->ir_attr->GetAttrValue("index", ir_index);
    }
    GE_ASSERT_TRUE(ir_index >= 0, "Cannot find ir attr index from data node [%s].", node->GetNamePtr());

    for (const auto &sub_data : iter->second.GetAllNodes()) {
      if (!ScheduleUtils::IsDataInput(sub_data)) {
        continue;
      }
      int64_t sub_index = -1;
      (void)ScheduleUtils::GetNodeIrAttrIndex(sub_data, sub_index);
      GE_ASSERT_TRUE(sub_index >= 0, "Cannot find ir attr index for node [%s].", sub_data->GetNamePtr());
      if (sub_index == data_index) {
        *node_attr = sub_data->attr;
        *tensor_attr = sub_data->outputs[0].attr;
        GELOGD("Data node [%s] use attr from [%s].", node->GetNamePtr(), sub_data->GetNamePtr());
      }
    }
    GE_ASSERT_NOTNULL(node_attr->ir_attr);
    GELOGD("Data node [%s] has been set ir attr index [%ld].", node->GetNamePtr(), ir_index);
    auto ir_attr = node_attr->ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    GE_ASSERT_NOTNULL(ir_attr);
    GE_ASSERT_SUCCESS(ir_attr->SetIndex(ir_index));
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::UnfoldFusedGraph(const af::ComputeGraphPtr &fused_graph,
                                            std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                            af::AscGraph &unfolded_asc_graph) {
  // step1 verify and choose loop
  std::vector<af::AxisPtr> new_loop_axes;
  GE_CHK_STATUS_RET(SelectCommonLoopAxis(fused_graph, asc_backend_to_asc_graph, new_loop_axes),
                    "The loop axis verification failed. Please confirm whether the fused graph [%s] is legitimate.",
                    fused_graph->GetName().c_str());
  // set loop and convert to ascgraph
  auto graph_attr = fused_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_CHECK_NOTNULL(graph_attr);
  graph_attr->axis = new_loop_axes;

  // reset data ir attr
  GE_ASSERT_SUCCESS(ReAssembleDataIrAttr(fused_graph, asc_backend_to_asc_graph),
                    "ReAssembleDataIrAttr failed, graph:[%s].", fused_graph->GetName().c_str());
  // Log the fused graph name to identify which fused graph the following subgraphs belong to.
  GELOGI("Start unfolding fused graph [%s] with [%zu] direct nodes.", fused_graph->GetName().c_str(),
         fused_graph->GetDirectNodePtr().size());
  // step2 do graph unfold
  for (const auto &node : fused_graph->GetDirectNodePtr()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() == kAscGraphNodeType) {
      auto iter = asc_backend_to_asc_graph.find(node);
      GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end());
      ascir::utils::DumpGraph(iter->second, "BeforeUnfoldAscBackend");
      GE_CHK_STATUS_RET(UnfoldAscbcNode(node, iter->second, fused_graph),
                        "Unfold ascgraph node [%s] to fused graph [%s] failed.", node->GetNamePtr(),
                        fused_graph->GetName().c_str());
      ascir::utils::DumpGraph(iter->second, "AfterUnfoldAscBackend");
    }
  }
  ascir::utils::DumpComputeGraph(fused_graph, "FusedGraphAfterUnfold");

  // step3 Do load cse
  GE_CHK_STATUS_RET(DoSameLoadCse(fused_graph),
                    "[Invoke][RemoveSameIndexData] Remove same index node for graph %s failed",
                    fused_graph->GetName().c_str());
  ascir::utils::DumpComputeGraph(fused_graph, "AfterDoSameLoadCse");
  // step4 reassemble io index by fused graph io nodes.
  GE_ASSERT_SUCCESS(ReAssembleOutputIndex(fused_graph), "ReAssembleOutputIndex failed, graph:[%s].",
                    fused_graph->GetName().c_str());

  // step5 Remove redundant Load, Output and Store nodes
  GE_CHK_STATUS_RET(RemoveRedundantLoads(fused_graph),
                    "[Invoke][RemoveRedundantLoads] Remove redundant Loads for graph %s failed",
                    fused_graph->GetName().c_str());
  ascir::utils::DumpComputeGraph(fused_graph, "AfterRemoveRedundantLoads");

  GE_CHK_STATUS_RET(fused_graph->TopologicalSorting(), "Failed to do topological sorting for graph:[%s].",
                    fused_graph->GetName().c_str());

  GE_ASSERT_GRAPH_SUCCESS(af::AscGraphUtils::ConvertComputeGraphToAscGraph(fused_graph, unfolded_asc_graph));
  return af::SUCCESS;
}

Status FusedGraphUnfolder::CloneAscGraphs(const std::map<af::Node *, af::AscGraph> &source_graphs,
                                          std::map<af::Node *, af::AscGraph> &cloned_graphs) {
  // AxisPtr instances must not be shared with the original graphs before commit.
  for (const auto &iter : source_graphs) {
    std::string serialized_graph;
    GE_CHK_STATUS_RET(af::AscGraphUtils::SerializeToReadable(iter.second, serialized_graph),
                      "Serialize graph:[%s] to readable format failed.", iter.second.GetName().c_str());
    af::AscGraph cloned_graph(iter.second.GetName().c_str());
    GE_CHK_STATUS_RET(af::AscGraphUtils::DeserializeFromReadable(serialized_graph, cloned_graph),
                      "Deserialize graph:[%s] from readable format failed.", cloned_graph.GetName().c_str());
    GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(cloned_graph), "Complete api info for graph:[%s] failed.",
                      cloned_graph.GetName().c_str());
    cloned_graphs.emplace(iter.first, std::move(cloned_graph));
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::FindConcatContext(const af::ComputeGraphPtr &fused_graph,
                                             const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                             af::Node *&concat_ascbc_node, std::vector<af::AxisPtr> &new_loop_axes,
                                             std::vector<af::AxisId> &loop_axis_ids, size_t &concat_dim) {
  for (const auto &graph_node : fused_graph->GetDirectNodePtr()) {
    const auto graph_iter = asc_backend_to_asc_graph.find(graph_node);
    if (graph_iter == asc_backend_to_asc_graph.end()) {
      continue;
    }
    for (const auto &node : graph_iter->second.GetAllNodes()) {
      if (!af::ops::IsOps<af::ascir_op::Concat>(node)) {
        continue;
      }
      GE_ASSERT_TRUE(concat_ascbc_node == nullptr, "Only one concat is supported in a fused graph.");
      GE_CHK_STATUS_RET(ScheduleUtils::GetConcatDim(node, concat_dim), "Get concat dim for node:[%s] failed.",
                        node->GetNamePtr());
      concat_ascbc_node = graph_node;
      new_loop_axes = graph_iter->second.GetAllAxis();
      for (const auto &axis : new_loop_axes) {
        GE_ASSERT_NOTNULL(axis);
        loop_axis_ids.push_back(axis->id);
      }
    }
  }
  GE_ASSERT_NOTNULL(concat_ascbc_node);
  GE_ASSERT_TRUE(concat_dim < new_loop_axes.size(), "Concat dim [%zu] is invalid, max dim [%zu].", concat_dim,
                 new_loop_axes.size());
  return af::SUCCESS;
}

Status FusedGraphUnfolder::CollectPreConcatMappings(const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                                    af::Node *concat_ascbc_node,
                                                    const std::vector<af::AxisId> &loop_axis_ids,
                                                    std::map<af::Node *, std::vector<size_t>> &pre_concat_mappings) {
  std::queue<af::Node *> pending_nodes;
  pre_concat_mappings.emplace(concat_ascbc_node, std::vector<size_t>(loop_axis_ids.size()));
  std::iota(pre_concat_mappings[concat_ascbc_node].begin(), pre_concat_mappings[concat_ascbc_node].end(), 0UL);
  pending_nodes.push(concat_ascbc_node);
  while (!pending_nodes.empty()) {
    const auto target_node = pending_nodes.front();
    pending_nodes.pop();
    const auto target_iter = asc_backend_to_asc_graph.find(target_node);
    GE_ASSERT_TRUE(target_iter != asc_backend_to_asc_graph.end(), "Cannot find target ascgraph.");
    for (const auto &in_anchor : target_node->GetAllInDataAnchorsPtr()) {
      const auto source_out_anchor = in_anchor->GetPeerOutAnchor();
      if (source_out_anchor == nullptr ||
          asc_backend_to_asc_graph.count(source_out_anchor->GetOwnerNodeBarePtr()) == 0UL) {
        continue;
      }
      const auto source_node = source_out_anchor->GetOwnerNodeBarePtr();
      const auto source_iter = asc_backend_to_asc_graph.find(source_node);
      const auto source_attr = FindBoundaryTensorAttr(source_iter->second, source_out_anchor->GetIdx(), true);
      const auto target_attr = FindBoundaryTensorAttr(target_iter->second, in_anchor->GetIdx(), false);
      GE_ASSERT_NOTNULL(source_attr);
      GE_ASSERT_NOTNULL(target_attr);
      std::vector<size_t> mapping;
      if (!BuildGraphAxisMapping(source_iter->second, *source_attr, target_iter->second, *target_attr,
                                 pre_concat_mappings.at(target_node), mapping)) {
        GELOGW("Cannot map pre-concat for source node [%s], falling back to merged loop axis.",
               source_node->GetNamePtr());
        continue;
      }
      const auto mapping_iter = pre_concat_mappings.find(source_node);
      if (mapping_iter == pre_concat_mappings.end()) {
        pre_concat_mappings.emplace(source_node, std::move(mapping));
        pending_nodes.push(source_node);
      } else {
        const auto previous_mapping = mapping_iter->second;
        GE_ASSERT_TRUE(MergeGraphAxisMapping(mapping_iter->second, mapping), "Conflicting boundary mappings.");
        // Newly constrained target axes may unlock a complete mapping for its upstream producers.
        if (mapping_iter->second != previous_mapping) {
          pending_nodes.push(source_node);
        }
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ApplyPreConcatMappings(
    const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
    const std::map<af::Node *, af::AscGraph> &post_concat_node_to_asc_graph,
    const std::map<af::Node *, std::vector<size_t>> &pre_concat_mappings, const std::vector<af::AxisPtr> &new_loop_axes,
    const std::vector<af::AxisId> &loop_axis_ids, const size_t concat_dim) {
  for (const auto &iter : asc_backend_to_asc_graph) {
    if (post_concat_node_to_asc_graph.count(iter.first) != 0UL) {
      continue;
    }
    const auto mapping_iter = pre_concat_mappings.find(iter.first);
    if (mapping_iter == pre_concat_mappings.end()) {
      GE_CHK_STATUS_RET(ApplyMergedLoopAxis(iter.second, new_loop_axes, loop_axis_ids, concat_dim),
                        "Apply merged loop axis for graph:[%s] failed.", iter.second.GetName().c_str());
      continue;
    }
    GE_ASSERT_TRUE(IsGraphAxisMappingComplete(mapping_iter->second), "Pre-concat graph has unmapped axes.");
    GE_CHK_STATUS_RET(ApplyMappedLoopAxis(iter.second, new_loop_axes, loop_axis_ids, mapping_iter->second),
                      "Apply mapped loop axis for graph:[%s] failed.", iter.second.GetName().c_str());
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::SelectCommonLoopAxis(const af::ComputeGraphPtr &fused_graph,
                                                std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                                std::vector<af::AxisPtr> &new_loop_axes) {
  GE_ASSERT_TRUE(!asc_backend_to_asc_graph.empty(), "The map is empty after deserialization.");
  std::map<af::Node *, af::AscGraph> cloned_graphs;
  GE_CHK_STATUS_RET(CloneAscGraphs(asc_backend_to_asc_graph, cloned_graphs),
                    "Clone asc graphs of %zu backend node(s) failed.", asc_backend_to_asc_graph.size());
  size_t concat_dim = 0UL;
  af::Node *concat_ascbc_node = nullptr;
  std::vector<af::AxisId> loop_axis_ids;
  GE_CHK_STATUS_RET(
      FindConcatContext(fused_graph, cloned_graphs, concat_ascbc_node, new_loop_axes, loop_axis_ids, concat_dim),
      "Find concat context failed.");
  std::map<af::Node *, af::AscGraph> post_concat_graphs;
  GE_CHK_STATUS_RET(
      CollectPostConcatAscGraphs(concat_ascbc_node, cloned_graphs, new_loop_axes, loop_axis_ids, post_concat_graphs),
      "Collect post-concat graphs failed.");
  std::map<af::Node *, std::vector<size_t>> pre_concat_mappings;
  GE_CHK_STATUS_RET(CollectPreConcatMappings(cloned_graphs, concat_ascbc_node, loop_axis_ids, pre_concat_mappings),
                    "Collect pre-concat mappings failed.");
  GE_CHK_STATUS_RET(ApplyPreConcatMappings(cloned_graphs, post_concat_graphs, pre_concat_mappings, new_loop_axes,
                                           loop_axis_ids, concat_dim),
                    "Apply pre-concat mappings failed.");
  // The original map remains untouched until every clone has been mapped successfully.
  asc_backend_to_asc_graph.swap(cloned_graphs);
  return af::SUCCESS;
}

Status FusedGraphUnfolder::CollectPostConcatAscGraphs(
    af::Node *concat_ascbc_node, const std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
    const std::vector<af::AxisPtr> &new_loop_axes, const std::vector<af::AxisId> &loop_axis_ids,
    std::map<af::Node *, af::AscGraph> &post_concat_node_to_asc_graph) {
  std::queue<af::Node *> queue;
  std::set<af::Node *> seen_nodes;
  queue.push(concat_ascbc_node);
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    auto iter = asc_backend_to_asc_graph.find(node);
    if (seen_nodes.count(node) == 0UL && iter != asc_backend_to_asc_graph.end()) {
      GE_ASSERT_SUCCESS(DoAxisMappingForConstPostAscGraph(iter->second, new_loop_axes, loop_axis_ids),
                        "Failed to do axis mapping for graph [%s], asc_node:[%s].", iter->second.GetName().c_str(),
                        iter->first->GetNamePtr());
      post_concat_node_to_asc_graph.emplace(node, iter->second);
    }
    seen_nodes.insert(node);
    for (auto out_node : node->GetOutDataNodesPtr()) {
      queue.push(out_node);
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MarkAllOutputAxisId(
    af::Node *concat_ascbc_node, std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
    const af::AxisId &axis_id, std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
    std::set<af::Node *> &seen_node) {
  std::queue<af::Node *> que;
  que.emplace(concat_ascbc_node);
  while (!que.empty()) {
    auto top = que.front();
    auto iter = asc_backend_to_asc_graph.find(top);
    GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find AscGraph for node [%s].", top->GetNamePtr());
    seen_graph_to_changed_axis_id[&iter->second] = axis_id;
    GELOGD("Mark graph [%s] with id [%ld].", iter->second.GetName().c_str(), axis_id);
    seen_node.emplace(top);
    que.pop();
    for (auto &output_node : top->GetOutDataNodes()) {
      if (output_node->GetType() == kAscGraphNodeType && seen_node.count(output_node.get()) == 0UL) {
        que.emplace(output_node.get());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::MarkAllInputAxisId(af::Node *concat_input_node,
                                              std::map<af::Node *, af::AscGraph> &asc_backend_to_asc_graph,
                                              const af::AxisId &axis_id,
                                              std::map<const af::AscGraph *, af::AxisId> &seen_graph_to_changed_axis_id,
                                              std::set<af::Node *> &seen_node) {
  std::queue<af::Node *> que;
  que.emplace(concat_input_node);
  while (!que.empty()) {
    auto top = que.front();
    if (top->GetType() == kAscGraphNodeType && seen_node.count(top) == 0U) {
      auto iter = asc_backend_to_asc_graph.find(top);
      GE_ASSERT_TRUE(iter != asc_backend_to_asc_graph.end(), "Cannot find AscGraph for node [%s].", top->GetNamePtr());
      seen_graph_to_changed_axis_id[&iter->second] = axis_id;
      GELOGD("Mark graph [%s] with id [%ld].", iter->second.GetName().c_str(), axis_id);
    }
    seen_node.emplace(top);
    que.pop();
    for (auto &in_node : top->GetInDataNodes()) {
      if (in_node->GetType() == kAscGraphNodeType && seen_node.count(in_node.get()) == 0UL) {
        que.emplace(in_node.get());
      }
    }
    for (auto &output_node : top->GetOutDataNodes()) {
      if (output_node->GetType() == kAscGraphNodeType && seen_node.count(output_node.get()) == 0UL) {
        que.emplace(output_node.get());
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ApplyMergedLoopAxis(const af::AscGraph &graph, const std::vector<af::AxisPtr> &new_loop_axes,
                                               const std::vector<af::AxisId> &loop_axis_ids, const size_t concat_dim) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  bool need_expand = false;
  auto old_axis = graph_attr->axis;
  if (old_axis.size() != loop_axis_ids.size()) {
    // 补轴场景只支持补concat_dim轴
    need_expand = true;
    GE_ASSERT_TRUE(old_axis.size() + 1UL == loop_axis_ids.size(), "Only expand concat_dim axis is supported.");
  }
  graph_attr->axis = new_loop_axes;

  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    node->attr.sched.axis = loop_axis_ids;
    for (auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      output->attr.axis = loop_axis_ids;
      if (!need_expand) {
        continue;
      }
      if (concat_dim == output->attr.repeats.size()) {
        output->attr.repeats.push_back(af::sym::kSymbolOne);
        output->attr.strides.push_back(af::sym::kSymbolZero);
      } else {
        GE_ASSERT_TRUE(concat_dim < output->attr.repeats.size(),
                       "concat dim:[%zu] is invalid, valid range is [0, %zu), node name:[%s]", concat_dim,
                       output->attr.repeats.size(), node->GetNamePtr());
        GE_ASSERT_TRUE(concat_dim < output->attr.strides.size());
        output->attr.repeats.insert(output->attr.repeats.begin() + static_cast<int64_t>(concat_dim),
                                    af::sym::kSymbolOne);
        output->attr.strides.insert(output->attr.strides.begin() + static_cast<int64_t>(concat_dim),
                                    af::sym::kSymbolZero);
      }
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::ApplyMappedLoopAxis(const af::AscGraph &graph, const std::vector<af::AxisPtr> &new_loop_axes,
                                               const std::vector<af::AxisId> &loop_axis_ids,
                                               const std::vector<size_t> &old_to_global) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  const auto old_axis = graph_attr->axis;
  GE_ASSERT_TRUE(old_axis.size() == old_to_global.size(), "Axis mapping rank mismatch, graph:[%s].",
                 graph.GetName().c_str());
  std::map<af::AxisId, size_t> old_axis_to_global;
  std::set<size_t> mapped_global_axes;
  for (size_t index = 0UL; index < old_axis.size(); ++index) {
    GE_ASSERT_NOTNULL(old_axis[index]);
    GE_ASSERT_TRUE(old_to_global[index] < loop_axis_ids.size(),
                   "Axis mapping index:[%zu] is invalid, valid range is [0, %zu), graph:[%s].", old_to_global[index],
                   loop_axis_ids.size(), graph.GetName().c_str());
    GE_ASSERT_TRUE(mapped_global_axes.emplace(old_to_global[index]).second, "Axis mapping is not injective.");
    GE_ASSERT_TRUE(old_axis_to_global.emplace(old_axis[index]->id, old_to_global[index]).second,
                   "Graph has duplicate axis id.");
  }
  graph_attr->axis = new_loop_axes;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    node->attr.sched.axis = loop_axis_ids;
    if (node->attr.sched.loop_axis != af::kIdNone) {
      const auto loop_iter = old_axis_to_global.find(node->attr.sched.loop_axis);
      GE_ASSERT_TRUE(loop_iter != old_axis_to_global.end(), "Cannot map loop axis, node:[%s].", node->GetNamePtr());
      node->attr.sched.loop_axis = loop_axis_ids[loop_iter->second];
    }
    for (auto &output : node->outputs()) {
      GE_ASSERT_NOTNULL(output);
      GE_CHK_STATUS_RET(RewriteTensorAxis(node, output->attr, old_axis_to_global, loop_axis_ids),
                        "Rewrite tensor axis failed.");
    }
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::RewriteTensorAxis(const af::AscNodePtr &node, af::AscTensorAttr &tensor_attr,
                                             const std::map<af::AxisId, size_t> &old_axis_to_global,
                                             const std::vector<af::AxisId> &loop_axis_ids) {
  GE_ASSERT_TRUE(IsTensorAttrValid(tensor_attr), "Tensor axis attr size mismatch, node:[%s].", node->GetNamePtr());
  std::vector<af::Expression> new_repeats(loop_axis_ids.size(), af::sym::kSymbolOne);
  std::vector<af::Expression> new_strides(loop_axis_ids.size(), af::sym::kSymbolZero);
  for (size_t index = 0UL; index < tensor_attr.axis.size(); ++index) {
    const auto axis_iter = old_axis_to_global.find(tensor_attr.axis[index]);
    GE_ASSERT_TRUE(axis_iter != old_axis_to_global.end(), "Cannot map tensor axis, node:[%s].", node->GetNamePtr());
    new_repeats[axis_iter->second] = tensor_attr.repeats[index];
    new_strides[axis_iter->second] = tensor_attr.strides[index];
  }
  tensor_attr.axis = loop_axis_ids;
  // Preserve local repeat/stride at mapped positions; missing global axes are unit/zero-stride.
  tensor_attr.repeats = std::move(new_repeats);
  tensor_attr.strides = std::move(new_strides);
  for (auto &axis_id : tensor_attr.vectorized_axis) {
    const auto axis_iter = old_axis_to_global.find(axis_id);
    GE_ASSERT_TRUE(axis_iter != old_axis_to_global.end(), "Cannot map vectorized axis, node:[%s].", node->GetNamePtr());
    axis_id = loop_axis_ids[axis_iter->second];
  }
  return af::SUCCESS;
}

Status FusedGraphUnfolder::DoAxisMappingForConstPostAscGraph(const af::AscGraph &graph,
                                                             const std::vector<af::AxisPtr> &new_loop_axes,
                                                             const std::vector<af::AxisId> &loop_axis_ids) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  auto old_axis = graph_attr->axis;
  if (old_axis.size() == loop_axis_ids.size()) {
    return af::SUCCESS;
  }
  std::map<size_t, size_t> new_idx_to_old_idx;
  size_t old_idx = 0UL;
  const size_t old_size = old_axis.size();
  const size_t new_size = new_loop_axes.size();
  for (size_t new_idx = 0UL; new_idx < new_size; ++new_idx) {
    if (old_idx < old_size &&
        af::SymbolicUtils::StaticCheckEq(new_loop_axes[new_idx]->size, old_axis[old_idx]->size) == af::TriBool::kTrue) {
      new_idx_to_old_idx.emplace(new_idx, old_idx);
      old_idx++;
    }
  }

  GE_ASSERT_TRUE(old_idx == old_size, "Axes mapping failed, only expansion scenarios are supported, graph:[%s].",
                 graph.GetName().c_str());
  graph_attr->axis = new_loop_axes;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    node->attr.sched.axis = loop_axis_ids;
    for (auto &output : node->outputs()) {
      output->attr.axis = loop_axis_ids;
      std::vector<af::Expression> new_repeats;
      std::vector<af::Expression> new_strides;
      const size_t axis_size = loop_axis_ids.size();
      for (size_t idx = 0UL; idx < axis_size; ++idx) {
        auto iter = new_idx_to_old_idx.find(idx);
        if (iter != new_idx_to_old_idx.end()) {
          const size_t old_index = iter->second;
          GE_ASSERT_TRUE(old_index < output->attr.repeats.size(),
                         "Index out of bounds, index:[%zu], repeats size:[%zu], node:[%s].",
                         static_cast<size_t>(old_index), output->attr.repeats.size(), node->GetNamePtr());
          GE_ASSERT_TRUE(old_index < output->attr.strides.size(),
                         "Index out of bounds, index:[%zu], strides size:[%zu], node:[%s].",
                         static_cast<size_t>(old_index), output->attr.strides.size(), node->GetNamePtr());
          new_repeats.push_back(output->attr.repeats[old_index]);
          new_strides.push_back(output->attr.strides[old_index]);
        } else {
          new_repeats.push_back(af::sym::kSymbolOne);
          new_strides.push_back(af::sym::kSymbolZero);
        }
      }
      output->attr.repeats = std::move(new_repeats);
      output->attr.strides = std::move(new_strides);
    }
  }

  return af::SUCCESS;
}
}  // namespace optimize
