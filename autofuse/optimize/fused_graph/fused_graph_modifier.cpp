/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_graph_modifier.h"
#include <queue>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

namespace {
const char *const kAscBackendType = "AscBackend";
const char *const kWorkspacePrefix = "fused_workspace";
}  // namespace
namespace optimize {
Status FusedGraphModifier::InitAscbcOutAnchorAttr(
    const af::ComputeGraphPtr &fused_graph,
    std::map<const af::Node *, std::map<int64_t, OutAnchorAttr>> &nodes_to_out_anchor_idx_to_attr) {
  for (auto &node : fused_graph->GetDirectNodePtr()) {
    if (node->GetType() != kAscBackendType) {
      continue;
    }
    for (const auto &out_anchor : node->GetAllOutAnchorsPtr()) {
      GE_ASSERT_NOTNULL(out_anchor);
      auto peer_in_anchors = out_anchor->GetPeerAnchorsPtr();
      nodes_to_out_anchor_idx_to_attr[node][out_anchor->GetIdx()].depends =
          static_cast<int64_t>(peer_in_anchors.size());
      for (auto &in_anchor : peer_in_anchors) {
        GE_ASSERT_NOTNULL(in_anchor);
        if (in_anchor->GetOwnerNode()->GetType() == af::ascir_op::Output::Type) {
          auto op_desc = in_anchor->GetOwnerNode()->GetOpDescBarePtr();
          GE_ASSERT_NOTNULL(op_desc);
          auto node_attr = op_desc->GetAttrsGroup<af::AscNodeAttr>();
          GE_ASSERT_NOTNULL(node_attr);
          GE_ASSERT_NOTNULL(node_attr->ir_attr);
          int64_t ir_idx = -1;
          (void)node_attr->ir_attr->GetAttrValue("index", ir_idx);
          GE_ASSERT_TRUE(ir_idx >= 0, "Get invalid ir_index:[%ld] from node:[%s], expect >= 0.", ir_idx,
                         op_desc->GetNamePtr());
          nodes_to_out_anchor_idx_to_attr[node][out_anchor->GetIdx()].linked_output_ir_idx = ir_idx;
        }
      }
    }
  }
  return af::SUCCESS;
}

int64_t FusedGraphModifier::ReuseWorkspaceId(std::set<int64_t> &free_ws_ids, const std::set<int64_t> &data_used_ids,
                                             int64_t &max_workspace_num) {
  for (auto it = free_ws_ids.begin(); it != free_ws_ids.end(); ++it) {
    if (data_used_ids.find(*it) == data_used_ids.end()) {
      int64_t result = *it;
      free_ws_ids.erase(it);
      return result;
    }
  }
  return max_workspace_num++;
}

Status FusedGraphModifier::ProcessOutputNodes(const af::AscNodePtr &sub_out_node, const af::Node *const ascbc_node,
                                              af::AscGraph &asc_graph, ProcessNodesContext &context,
                                              int64_t &max_workspace_num) {
  int64_t ir_index = -1;
  (void)ScheduleUtils::GetNodeIrAttrIndex(sub_out_node, ir_index);
  GE_ASSERT_TRUE(ir_index >= 0, "Cannot get ir_index from output node [%s].", sub_out_node->GetNamePtr());
  auto &out_attr = context.nodes_to_out_anchor_idx_to_attr[ascbc_node][ir_index];
  if (out_attr.linked_output_ir_idx >= 0) {
    GE_ASSERT_NOTNULL(sub_out_node->attr.ir_attr);
    auto ir_attr = sub_out_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    GE_ASSERT_NOTNULL(ir_attr);
    (void)ir_attr->SetIndex(out_attr.linked_output_ir_idx);
    GELOGD("Node [%s] connect to output directly, set with index [%ld].", sub_out_node->GetNamePtr(),
           out_attr.linked_output_ir_idx);
    return af::SUCCESS;
  }
  out_attr.used_ws_idx = ReuseWorkspaceId(context.free_workspace_id, context.data_used_ids, max_workspace_num);
  std::string ws_name = kWorkspacePrefix + std::to_string(out_attr.used_ws_idx);
  af::ascir_op::Workspace ws(ws_name.c_str());
  auto ws_node = asc_graph.AddNode(ws);
  GE_ASSERT_NOTNULL(ws_node);
  ws_node->attr = sub_out_node->attr;
  ws_node->outputs[0].attr = sub_out_node->outputs[0].attr;
  auto sub_in_anchor = sub_out_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(sub_in_anchor);
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(sub_in_anchor->GetPeerOutAnchor(), ws_node->GetInDataAnchor(0)));
  af::NodeUtils::UnlinkAll(*sub_out_node);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(asc_graph);
  af::GraphUtils::RemoveNodeWithoutRelink(compute_graph, sub_out_node);
  return af::SUCCESS;
}

Status FusedGraphModifier::ProcessDataNodes(const af::AscNodePtr &sub_data_node, const af::Node *const ascbc_node,
                                            af::AscGraph &asc_graph, ProcessNodesContext &context) {
  int64_t index = -1;
  (void)ScheduleUtils::GetNodeIrAttrIndex(sub_data_node, index);
  auto in_anchor = ascbc_node->GetInDataAnchor(static_cast<int32_t>(index));
  GE_ASSERT_NOTNULL(in_anchor);
  GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor(), "Cannot get peer out anchor from [%s]'s [%ld]'th input node.",
                    ascbc_node->GetNamePtr(), index);
  auto peer_out_node = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(peer_out_node);
  if (ScheduleUtils::IsDataInput(peer_out_node)) {
    auto op_desc = peer_out_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<af::AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    GE_ASSERT_NOTNULL(node_attr->ir_attr);
    int64_t ir_idx = -1;
    (void)node_attr->ir_attr->GetAttrValue("index", ir_idx);
    GE_ASSERT_TRUE(ir_idx >= 0, "Get invalid ir_index:[%ld] from node:[%s], expect >= 0.", ir_idx,
                   op_desc->GetNamePtr());
    auto ir_attr = sub_data_node->attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    (void)ir_attr->SetIndex(ir_idx);
    return af::SUCCESS;
  }

  // data 改成workspace、output
  auto out_idx = in_anchor->GetPeerOutAnchor()->GetIdx();
  auto &out_attr = context.nodes_to_out_anchor_idx_to_attr[peer_out_node][out_idx];
  if (out_attr.linked_output_ir_idx >= 0) {
    std::string ws_name = kWorkspacePrefix + sub_data_node->GetName();
    af::ascir_op::Output tmp_out(ws_name.c_str());
    auto tmo_output_node = asc_graph.AddNode(tmp_out);
    tmo_output_node->attr = sub_data_node->attr;
    tmo_output_node->outputs[0].attr = sub_data_node->outputs[0].attr;
    auto ir_attr = tmp_out.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
    (void)ir_attr->SetIndex(out_attr.linked_output_ir_idx);
    GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceNodeDataAnchors(tmo_output_node, sub_data_node, {0}, {0}));
  } else {
    context.data_used_ids.emplace(out_attr.used_ws_idx);
    std::string ws_name = kWorkspacePrefix + std::to_string(out_attr.used_ws_idx);
    af::ascir_op::Workspace ws(ws_name.c_str());
    auto ws_node = asc_graph.AddNode(ws);
    ws_node->attr = sub_data_node->attr;
    ws_node->outputs[0].attr = sub_data_node->outputs[0].attr;
    GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceNodeDataAnchors(ws_node, sub_data_node, {0}, {0}));
    out_attr.depends--;
    if (out_attr.depends <= 0) {
      context.free_workspace_id.emplace(out_attr.used_ws_idx);
    }
  }
  af::NodeUtils::UnlinkAll(*sub_data_node);
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(compute_graph, sub_data_node), "Remove node [%s] failed.",
                    sub_data_node->GetNamePtr());
  return af::SUCCESS;
}

Status FusedGraphModifier::SubgraphConnectionsToWorkspace(const af::ComputeGraphPtr &fused_graph,
                                                          std::map<af::Node *, af::AscGraph> &asc_backend_to_ascgraph) {
  std::map<const af::Node *, std::map<int64_t, OutAnchorAttr>> nodes_to_out_anchor_idx_to_attr;
  GE_ASSERT_SUCCESS(InitAscbcOutAnchorAttr(fused_graph, nodes_to_out_anchor_idx_to_attr));

  int64_t max_workspace_num = 0;
  std::set<int64_t> free_workspace_id;
  for (auto &node : fused_graph->GetDirectNodePtr()) {
    if (node->GetType() != kAscBackendType) {
      continue;
    }
    auto iter = asc_backend_to_ascgraph.find(node);
    GE_ASSERT_TRUE(iter != asc_backend_to_ascgraph.end(), "Cannot find AscGraph for node [%s].", node->GetNamePtr());
    std::set<int64_t> data_used_ids;
    ProcessNodesContext context = {nodes_to_out_anchor_idx_to_attr, free_workspace_id, data_used_ids};

    for (const auto &sub_node : iter->second.GetAllNodes()) {
      // 处理输入
      if (ScheduleUtils::IsDataInput(sub_node)) {
        GE_ASSERT_SUCCESS(ProcessDataNodes(sub_node, node, iter->second, context));
      }
    }

    for (const auto &sub_node : iter->second.GetAllNodes()) {
      // 处理输出，考虑复用
      if (!af::ops::IsOps<af::ascir_op::Output>(sub_node) || sub_node->GetInDataNodesSize() == 0) {
        continue;
      }
      GE_ASSERT_SUCCESS(ProcessOutputNodes(sub_node, node, iter->second, context, max_workspace_num));
    }

    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(iter->second));
  }
  return af::SUCCESS;
}

Status FusedGraphModifier::ChangeStartingOutputToWorkspace(std::vector<ascir::ScheduleGroup> &schedule_groups) {
  for (auto &group : schedule_groups) {
    for (auto &graph : group.impl_graphs) {
      for (auto node : graph.GetAllNodes()) {
        if (!af::ops::IsOps<af::ascir_op::Output>(node) || !node->GetInNodes().empty()) {
          continue;
        }
        af::ascir_op::Workspace ws(node->GetNamePtr());
        auto ws_node = graph.AddNode(ws);
        ws_node->attr = node->attr;
        ws_node->outputs[0].attr = node->outputs[0].attr;
        GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceNodeDataAnchors(ws_node, node, {0}, {0}));
        af::NodeUtils::UnlinkAll(*node);

        auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
        GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(compute_graph, node));
        GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
      }
    }
  }
  return af::SUCCESS;
}

}  // namespace optimize
