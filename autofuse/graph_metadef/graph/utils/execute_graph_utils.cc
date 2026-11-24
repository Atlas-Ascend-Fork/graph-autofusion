/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/execute_graph_utils.h"

#include "common/checker.h"
#include "graph/fast_graph/fast_graph_impl.h"
#include "graph/fast_graph/fast_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace af {
namespace {
using InFastNodesToOut = std::map<FastNode *, std::vector<FastNode *>, FastNodeCompareKey>;
InFastNodesToOut GetFullConnectIONodes(const FastNode *fast_node) {
  GE_ASSERT_NOTNULL(fast_node);
  InFastNodesToOut in_nodes_to_out;
  const auto &in_nodes = fast_node->GetAllInNodes();
  const auto &out_nodes = fast_node->GetAllOutNodes();
  for (const auto &in_node : in_nodes) {
    (void)in_nodes_to_out.emplace(in_node, out_nodes);
  }
  return in_nodes_to_out;
}

graphStatus ReplaceOutDataEdges(FastNode *new_node, const FastNode *old_node, const std::vector<int32_t> &outputs_map,
                                ExecuteGraph *graph) {
  const auto &new_outs = new_node->GetAllOutDataEdgesRef();
  const auto new_out_size = new_outs.size();
  GE_ASSERT_TRUE(new_out_size >= outputs_map.size(),
                 "Failed to replace out data edge, the actual size %zu is less than the mapping size %zu", new_out_size,
                 outputs_map.size());
  const auto &old_outs = old_node->GetAllOutDataEdgesRef();
  for (size_t i = 0U; i < new_out_size; ++i) {
    if (i >= outputs_map.size()) {
      return GRAPH_SUCCESS;
    }
    const auto old_index = outputs_map[i];
    if ((old_index < 0) || (static_cast<size_t>(old_index) >= old_outs.size())) {
      continue;
    }

    for (const auto old_edge : old_outs[old_index]) {
      if (old_edge == nullptr) {
        continue;
      }
      const auto dst_node = old_edge->dst;
      GE_ASSERT_NOTNULL(dst_node);
      const auto dst_input = old_edge->dst_input;
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_edge), "Remove edge %s:%d->%s:%d failed", old_node->GetNamePtr(),
                              old_index, dst_node->GetNamePtr(), dst_input);
      GE_ASSERT_NOTNULL(graph->AddEdge(new_node, i, dst_node, dst_input), "Add edge %s:%d->%s:%d failed",
                        new_node->GetNamePtr(), i, dst_node->GetNamePtr(), dst_input);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceInDataEdges(FastNode *new_node, const FastNode *old_node, const std::vector<int32_t> &inputs_map,
                               ExecuteGraph *graph) {
  const auto &new_ins = new_node->GetAllInDataEdgesRef();
  const auto new_in_size = new_ins.size();
  GE_ASSERT_TRUE(new_in_size >= inputs_map.size(),
                 "Failed to replace in data edge, the actual size %zu is less than the mapping size %zu", new_in_size,
                 inputs_map.size());
  const auto &old_ins = old_node->GetAllInDataEdgesRef();
  for (size_t i = 0U; i < new_in_size; ++i) {
    if (i >= inputs_map.size()) {
      return GRAPH_SUCCESS;
    }
    const auto old_index = inputs_map[i];
    if ((old_index < 0) || (static_cast<size_t>(old_index) >= old_ins.size())) {
      continue;
    }

    const auto old_edge = old_ins[old_index];
    if (old_edge == nullptr) {
      continue;
    }
    const auto src_node = old_edge->src;
    GE_ASSERT_NOTNULL(src_node);
    const auto src_output = old_edge->src_output;
    GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_edge), "Remove edge %s:%d->%s:%d failed", src_node->GetNamePtr(),
                            src_output, old_node->GetNamePtr(), old_index);
    GE_ASSERT_NOTNULL(graph->AddEdge(src_node, src_output, new_node, i), "Add edge %s:%d->%s:%d failed",
                      src_node->GetNamePtr(), src_output, new_node->GetNamePtr(), i);
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceControlEdges(FastNode *new_node, const FastNode *old_node, ExecuteGraph *graph) {
  const auto &new_control_in_edges = new_node->GetAllInControlEdges();
  const auto exist_control_in_edges =
      std::unordered_set<Edge<FastNode> *>(new_control_in_edges.begin(), new_control_in_edges.end());
  for (const auto old_control_in_edge : old_node->GetAllInControlEdgesRef()) {
    if ((old_control_in_edge == nullptr) || (exist_control_in_edges.count(old_control_in_edge) > 0U)) {
      continue;
    }
    const auto src_node = old_control_in_edge->src;
    GE_ASSERT_NOTNULL(src_node);
    GE_ASSERT_NOTNULL(graph->AddEdge(src_node, kControlEdgeIndex, new_node, kControlEdgeIndex),
                      "Add control edge %s->%s failed", src_node->GetNamePtr(), new_node->GetNamePtr());
  }

  const auto &new_control_out_edges = new_node->GetAllOutControlEdges();
  const auto exist_control_out_edges =
      std::unordered_set<Edge<FastNode> *>(new_control_out_edges.begin(), new_control_out_edges.end());
  for (const auto old_control_out_edge : old_node->GetAllOutControlEdgesRef()) {
    if ((old_control_out_edge == nullptr) || (exist_control_out_edges.count(old_control_out_edge) > 0U)) {
      continue;
    }
    const auto dst_node = old_control_out_edge->dst;
    GE_ASSERT_NOTNULL(dst_node);
    GE_ASSERT_NOTNULL(graph->AddEdge(new_node, kControlEdgeIndex, dst_node, kControlEdgeIndex),
                      "Add control edge %s->%s failed", new_node->GetNamePtr(), dst_node->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}

graphStatus RelinkDataIO(ExecuteGraph *graph, FastNode *node, const std::vector<int32_t> &io_map,
                         InFastNodesToOut &in_nodes_to_out) {
  const size_t out_data_endpoint_size = node->GetDataOutNum();
  const size_t in_data_endpoint_size = node->GetDataInNum();
  GE_ASSERT_TRUE(out_data_endpoint_size >= io_map.size(),
                 "The io_map specified for node %s type %s is larger %zu than the actual size %zu",
                 node->GetName().c_str(), node->GetType().c_str(), io_map.size(), out_data_endpoint_size);
  const auto &all_in_data_edges = node->GetAllInDataEdgesRef();
  for (size_t i = 0U; i < out_data_endpoint_size; ++i) {
    int32_t in_index = (i < io_map.size()) ? io_map[i] : -1;
    if (in_index < 0) {
      for (const auto old_out_edge : node->GetOutEdgesRefByIndex(i)) {
        if (old_out_edge != nullptr) {
          GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_out_edge), "Remove out data edge for %s:%d failed.",
                                  node->GetNamePtr(), i);
        }
      }
    } else {
      GE_ASSERT_TRUE(in_index < static_cast<int32_t>(in_data_endpoint_size),
                     "Failed to relink for node %s type %s, invalid index %d specified for input(%zu)",
                     node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_endpoint_size);
      const auto old_in_edge = all_in_data_edges[in_index];
      if (old_in_edge == nullptr) {
        continue;
      }
      const auto src_node = old_in_edge->src;
      GE_ASSERT_NOTNULL(src_node);
      const auto src_output = old_in_edge->src_output;
      GE_ASSERT_GRAPH_SUCCESS(
          graph->RemoveEdge(old_in_edge),
          "Failed relink node %s type %s, failed to unlink the data link from %s(%d) to it at input-index %d",
          node->GetName().c_str(), node->GetType().c_str(), src_node->GetName().c_str(), src_output, in_index);

      for (const auto old_out_edge : node->GetOutEdgesRefByIndex(i)) {
        if (old_out_edge == nullptr) {
          continue;
        }
        const auto dst_node = old_out_edge->dst;
        GE_ASSERT_NOTNULL(dst_node);
        const auto dst_input = old_out_edge->dst_input;
        GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_out_edge), "Remove data edge %s:%d->%s:%d failed.",
                                node->GetNamePtr(), i, dst_node->GetNamePtr(), dst_input);
        GE_ASSERT_NOTNULL(graph->AddEdge(src_node, src_output, dst_node, dst_input),
                          "Add data edge %s:%d->%s:%d failed.", src_node->GetNamePtr(), src_output,
                          dst_node->GetNamePtr(), dst_input);
        in_nodes_to_out[src_node].emplace_back(dst_node);
      }
    }
  }

  for (const auto in_data_edge : all_in_data_edges) {
    if (in_data_edge != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(in_data_edge), "Remove in data edge for node:%s failed.",
                              node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RelinkControlNodeIfNeed(ExecuteGraph *graph, const InFastNodesToOut &in_nodes_to_out,
                                    InFastNodesToOut &connected_data_in_to_out) {
  for (const auto &in_node_to_out : in_nodes_to_out) {
    const auto in_node = in_node_to_out.first;
    GE_ASSERT_NOTNULL(in_node);
    const auto &connected_data_out = connected_data_in_to_out[in_node];
    const auto &out_control_nodes = in_node->GetOutControlNodes();
    const auto out_control_nodes_set =
        std::unordered_set<FastNode *>(out_control_nodes.begin(), out_control_nodes.end());
    for (const auto out_node : in_node_to_out.second) {
      GE_ASSERT_NOTNULL(out_node);
      if (std::find(connected_data_out.begin(), connected_data_out.end(), out_node) == connected_data_out.end()) {
        if (out_control_nodes_set.count(out_node) > 0) {
          continue;
        }
        GE_ASSERT_NOTNULL(graph->AddEdge(in_node, kControlEdgeIndex, out_node, kControlEdgeIndex),
                          "Add control edge %s->%s failed.", in_node->GetNamePtr(), out_node->GetNamePtr());
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ExecuteGraph *ExecuteGraphUtils::FindRootGraph(ExecuteGraph *exe_graph) {
  ExecuteGraph *result = nullptr;
  while (exe_graph != nullptr) {
    result = exe_graph;
    exe_graph = result->GetParentGraphBarePtr();
  }
  return result;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY FastNode *ExecuteGraphUtils::FindNodeFromAllNodes(
    ExecuteGraph *exe_graph, const char_t *const name) {
  GE_ASSERT_NOTNULL(exe_graph);
  GE_ASSERT_NOTNULL(exe_graph->graph_shared_);
  GE_ASSERT_NOTNULL(name);
  const auto root_graph = FindRootGraph(exe_graph);
  GE_ASSERT_NOTNULL(root_graph);

  const auto insert_func = [](const ExecuteGraph *const exe_graph, std::deque<FastNode *> &candidates) -> void {
    auto iter = exe_graph->graph_shared_->nodes_.end();
    while (iter != exe_graph->graph_shared_->nodes_.begin()) {
      --iter;
      (void)candidates.insert(candidates.begin(), &FastGraphUtils::GetNode(iter.element_));
    }
  };
  std::deque<FastNode *> candidates;
  insert_func(exe_graph, candidates);
  while (!candidates.empty()) {
    const auto fast_node = candidates.front();
    candidates.pop_front();
    if (fast_node == nullptr) {
      continue;
    }
    if (strcmp(fast_node->GetNamePtr(), name) == 0) {
      return fast_node;
    }
    const auto op_desc = fast_node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
      auto name_iter = subgraph_names.rbegin();
      while (name_iter != subgraph_names.rend()) {
        const auto subgraph = root_graph->GetSubGraph(*name_iter);
        if ((subgraph != nullptr) && (subgraph->graph_shared_ != nullptr)) {
          insert_func(subgraph, candidates);
        }
        ++name_iter;
      }
    }
  }
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<FastNode *> ExecuteGraphUtils::FindNodesByTypeFromAllNodes(
    ExecuteGraph *exe_graph, const char_t *const type) {
  GE_ASSERT_NOTNULL(exe_graph);
  GE_ASSERT_NOTNULL(type);
  const auto root_graph = FindRootGraph(exe_graph);
  GE_ASSERT_NOTNULL(root_graph);

  std::vector<FastNode *> nodes;
  for (const auto node : root_graph->GetAllNodes()) {
    if ((node != nullptr) && (strcmp(node->GetTypePtr(), type) == 0)) {
      nodes.emplace_back(node);
    }
  }
  return nodes;
}

FastNode *ExecuteGraphUtils::FindFirstNodeMatchType(ExecuteGraph *exe_graph, const char_t *const type) {
  GE_ASSERT_NOTNULL(exe_graph);
  GE_ASSERT_NOTNULL(exe_graph->graph_shared_);
  GE_ASSERT_NOTNULL(type);
  for (const auto &node : exe_graph->graph_shared_->nodes_) {
    if ((node != nullptr) && (strcmp(FastGraphUtils::GetNode(node).GetTypePtr(), type) == 0)) {
      return &FastGraphUtils::GetNode(node);
    }
  }
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::InsertNodeAfter(const EdgeSrcEndpoint &src, const std::vector<EdgeDstEndpoint> &dsts,
                                   FastNode *insert_node, const uint32_t input_index, const uint32_t output_index) {
  GE_ASSERT_NOTNULL(insert_node);
  const auto src_node = src.node;
  GE_ASSERT_NOTNULL(src_node);
  const auto src_extend_info = src_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:%s is null", src_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The own graph of src node:%s is null", src_node->GetNamePtr());
  GE_ASSERT_NOTNULL(insert_node->GetExtendInfo(), "The extend info of insert node:%s is null",
                    insert_node->GetNamePtr());
  GE_ASSERT_TRUE(graph == insert_node->GetExtendInfo()->GetOwnerGraphBarePtr(),
                 "src:%s and insert_node:%s do not exist in the same graph.", src_node->GetNamePtr(),
                 insert_node->GetNamePtr());

  const auto src_index = src.index;
  GE_ASSERT_NOTNULL(graph->AddEdge(src_node, src_index, insert_node, input_index), "[Add][Edge] %s:%d->%s:%d failed.",
                    src_node->GetNamePtr(), src_index, insert_node->GetNamePtr(), input_index);
  for (const auto &dst : dsts) {
    const auto dst_node = dst.node;
    GE_ASSERT_NOTNULL(dst_node);
    const auto dst_index = dst.index;
    const auto dst_extend_info = dst_node->GetExtendInfo();
    GE_ASSERT_NOTNULL(dst_extend_info, "The extend info of dst node:%s is null", dst_node->GetNamePtr());
    GE_ASSERT_TRUE(graph == dst_extend_info->GetOwnerGraphBarePtr(),
                   "[Check][Param] dst:%s and insert_node:%s do not exist in the same graph.", dst_node->GetNamePtr(),
                   insert_node->GetNamePtr());

    const auto old_edge = dst_node->GetInDataEdgeByIndex(dst_index);
    if (old_edge != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_edge), "Remove input edge %s:%d failed.", dst_node->GetNamePtr(),
                              dst_index);
    }
    GE_ASSERT_NOTNULL(graph->AddEdge(insert_node, output_index, dst_node, dst_index), "Add edge %s:%d->%s:%d failed.",
                      insert_node->GetNamePtr(), output_index, dst_node->GetNamePtr(), dst_index);
    for (const auto &old_ctrl_edge : src_node->GetAllOutControlEdgesRef()) {
      if (old_ctrl_edge == nullptr) {
        continue;
      }
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_ctrl_edge), "Remove out control edge for %s failed.",
                              src_node->GetNamePtr());
      GE_ASSERT_NOTNULL(graph->AddEdge(insert_node, kControlEdgeIndex, dst_node, kControlEdgeIndex),
                        "Add control edge %s->%s failed.", insert_node->GetNamePtr(), dst_node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::InsertNodeBefore(
    const EdgeDstEndpoint &dst, FastNode *insert_node, const uint32_t input_index, const uint32_t output_index) {
  GE_ASSERT_NOTNULL(insert_node);
  const auto dst_node = dst.node;
  GE_ASSERT_NOTNULL(dst_node);
  const auto dst_extend_info = dst_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(dst_extend_info, "The extend info of dst node:%s is null", dst_node->GetNamePtr());
  const auto graph = dst_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The own graph of dst node:%s is null", dst_node->GetNamePtr());
  GE_ASSERT_NOTNULL(insert_node->GetExtendInfo(), "The extend info of insert node:%s is null",
                    insert_node->GetNamePtr());
  GE_ASSERT_TRUE(graph == insert_node->GetExtendInfo()->GetOwnerGraphBarePtr(),
                 "[Check][Param] src:%s and insert_node:%s do not exist in the same graph.", dst_node->GetNamePtr(),
                 insert_node->GetNamePtr());

  const auto dst_index = dst.index;
  const auto old_edge = dst_node->GetInDataEdgeByIndex(dst_index);
  GE_ASSERT_NOTNULL(old_edge, "The input edge %s:%d is nullptr.", dst_node->GetNamePtr(), dst_index);
  const auto src_node = old_edge->src;
  GE_ASSERT_NOTNULL(src_node, "The src of %s:%d is nullptr.", dst_node->GetNamePtr(), dst_index);
  const auto src_index = old_edge->src_output;
  GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_edge), "Remove edge %s:%d->%s:%d failed.", src_node->GetNamePtr(),
                          src_index, dst_node->GetNamePtr(), dst_index);
  GE_ASSERT_NOTNULL(graph->AddEdge(src_node, src_index, insert_node, input_index), "Add edge %s:%d->%s:%d failed.",
                    src_node->GetNamePtr(), src_index, insert_node->GetNamePtr(), input_index);
  GE_ASSERT_NOTNULL(graph->AddEdge(insert_node, output_index, dst_node, dst_index), "Add edge %s:%d->%s:%d failed.",
                    insert_node->GetNamePtr(), output_index, dst_node->GetNamePtr(), dst_index);

  for (const auto old_ctrl_edge : dst_node->GetAllInControlEdgesRef()) {
    if (old_ctrl_edge == nullptr) {
      continue;
    }
    const auto src_ctrl_node = old_ctrl_edge->src;
    GE_ASSERT_NOTNULL(src_ctrl_node);
    GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(old_ctrl_edge), "Remove ctrl edge %s->%s failed.",
                            src_ctrl_node->GetNamePtr(), dst_node->GetNamePtr());
    GE_ASSERT_NOTNULL(graph->AddEdge(src_ctrl_node, kControlEdgeIndex, insert_node, kControlEdgeIndex),
                      "Add ctrl edge %s->%s failed.", src_ctrl_node->GetNamePtr(), insert_node->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::CopyInCtrlEdges(const FastNode *src_node,
                                                                                              FastNode *dst_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);

  const auto &src_ctrl_in_nodes = src_node->GetInControlNodes();
  if (src_ctrl_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }
  std::unordered_set<FastNode *> exist_in_ctrl_nodes_set;
  const auto &exist_in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!exist_in_ctrl_nodes.empty()) {
    exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
  }

  const auto src_extend_info = src_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:%s is null", src_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The graph of src node:%s is null", src_node->GetNamePtr());
  for (const auto in_node : src_ctrl_in_nodes) {
    if ((in_node != nullptr) && (exist_in_ctrl_nodes_set.count(in_node) == 0U)) {
      GE_ASSERT_NOTNULL(graph->AddEdge(in_node, kControlEdgeIndex, dst_node, kControlEdgeIndex),
                        "Add ctrl edge %s->%s failed.", in_node->GetNamePtr(), dst_node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::MoveInCtrlEdges(const FastNode *src_node,
                                                                                              FastNode *dst_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);
  GE_ASSERT_GRAPH_SUCCESS(CopyInCtrlEdges(src_node, dst_node), "Copy in ctrl edges failed, src_node:%s, dst_node:%s",
                          src_node->GetNamePtr(), dst_node->GetNamePtr());

  const auto src_extend_info = src_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:%s is null", src_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The graph of src node:%s is null", src_node->GetNamePtr());
  for (const auto src_in_ctrl_edge : src_node->GetAllInControlEdgesRef()) {
    if (src_in_ctrl_edge != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(src_in_ctrl_edge), "Remove in ctrl edge for %s failed.",
                              src_node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::CopyOutCtrlEdges(const FastNode *src_node,
                                                                                               FastNode *dst_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);

  const auto &out_ctrl_nodes = src_node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<FastNode *> exist_out_ctrl_nodes_set;
  const auto &exist_out_ctrl_nodes = dst_node->GetOutControlNodes();
  if (!exist_out_ctrl_nodes.empty()) {
    exist_out_ctrl_nodes_set.insert(exist_out_ctrl_nodes.begin(), exist_out_ctrl_nodes.end());
  }

  const auto src_extend_info = src_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:%s is null", src_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The graph of src node:%s is null", src_node->GetNamePtr());
  for (const auto out_node : out_ctrl_nodes) {
    if ((out_node != nullptr) && (exist_out_ctrl_nodes_set.count(out_node) == 0U)) {
      GE_ASSERT_NOTNULL(graph->AddEdge(dst_node, kControlEdgeIndex, out_node, kControlEdgeIndex),
                        "Add ctrl edge %s->%s failed.", dst_node->GetNamePtr(), out_node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::MoveOutCtrlEdges(const FastNode *src_node,
                                                                                               FastNode *dst_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);
  GE_ASSERT_GRAPH_SUCCESS(CopyOutCtrlEdges(src_node, dst_node), "Copy out ctrl edges failed, src_node:%s, dst_node:%s",
                          src_node->GetNamePtr(), dst_node->GetNamePtr());

  const auto src_extend_info = src_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(src_extend_info, "The extend info of src node:%s is null", src_node->GetNamePtr());
  const auto graph = src_extend_info->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "The graph of src node:%s is null", src_node->GetNamePtr());
  for (const auto src_out_ctrl_edge : src_node->GetAllOutControlEdgesRef()) {
    if (src_out_ctrl_edge != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(src_out_ctrl_edge), "Remove out ctrl edge for %s failed.",
                              src_node->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::MoveNodeToGraph(FastNode *node,
                                                                                              ExecuteGraph *dst_graph) {
  GE_ASSERT_GRAPH_SUCCESS(IsolateNode(node, {}));
  GE_ASSERT_NOTNULL(node->GetExtendInfo(), "ExtendInfo of node %s is null.", node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(RemoveNodeWithoutRelink(node->GetExtendInfo()->GetOwnerGraphBarePtr(), node));
  GE_ASSERT_NOTNULL(dst_graph->AddNode(node));
  GE_ASSERT_GRAPH_SUCCESS(node->GetExtendInfo()->SetOwnerGraph(dst_graph, node));
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::ReplaceNodeDataEdges(
    FastNode *new_node, FastNode *old_node, const std::initializer_list<int32_t> inputs_map,
    const std::initializer_list<int32_t> outputs_map, ExecuteGraph *graph) {
  return ReplaceNodeDataEdges(new_node, old_node, std::vector<int32_t>(inputs_map), std::vector<int32_t>(outputs_map),
                              graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::ReplaceNodeDataEdges(FastNode *new_node, FastNode *old_node, const std::vector<int32_t> &inputs_map,
                                        const std::vector<int32_t> &outputs_map, ExecuteGraph *graph) {
  GE_ASSERT_NOTNULL(new_node);
  GE_ASSERT_NOTNULL(old_node);
  if (graph == nullptr) {
    GE_ASSERT_NOTNULL(new_node->GetExtendInfo());
    graph = new_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(graph);
    GE_ASSERT_NOTNULL(old_node->GetExtendInfo());
    GE_ASSERT_TRUE(graph == old_node->GetExtendInfo()->GetOwnerGraphBarePtr());
  }

  GE_ASSERT_GRAPH_SUCCESS(ReplaceOutDataEdges(new_node, old_node, outputs_map, graph),
                          "Failed when replace node outputs from old node %s type %s to new node %s type %s",
                          old_node->GetNamePtr(), old_node->GetTypePtr(), new_node->GetNamePtr(),
                          new_node->GetTypePtr());
  GE_ASSERT_GRAPH_SUCCESS(ReplaceInDataEdges(new_node, old_node, inputs_map, graph),
                          "Failed when replace node inputs from old node %s type %s to new node %s type %s",
                          old_node->GetNamePtr(), old_node->GetTypePtr(), new_node->GetNamePtr(),
                          new_node->GetTypePtr());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ExecuteGraphUtils::ReplaceNodeEdges(
    FastNode *new_node, FastNode *old_node, const std::initializer_list<int32_t> inputs_map,
    const std::initializer_list<int32_t> outputs_map) {
  return ReplaceNodeEdges(new_node, old_node, std::vector<int32_t>(inputs_map), std::vector<int32_t>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::ReplaceNodeEdges(FastNode *new_node, FastNode *old_node, const std::vector<int32_t> &inputs_map,
                                    const std::vector<int32_t> &outputs_map) {
  GE_ASSERT_NOTNULL(new_node);
  GE_ASSERT_NOTNULL(old_node);
  GE_ASSERT_NOTNULL(new_node->GetExtendInfo());
  const auto graph = new_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(old_node->GetExtendInfo());
  GE_ASSERT_TRUE(graph == old_node->GetExtendInfo()->GetOwnerGraphBarePtr());
  GE_ASSERT_GRAPH_SUCCESS(ReplaceNodeDataEdges(new_node, old_node, inputs_map, outputs_map, graph),
                          "Replace data edges from %s to %s failed.", old_node->GetNamePtr(), new_node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(ReplaceControlEdges(new_node, old_node, graph), "Replace control edges from %s to %s failed.",
                          old_node->GetNamePtr(), new_node->GetNamePtr());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::IsolateNode(FastNode *node, const std::initializer_list<int32_t> &io_map) {
  return IsolateNode(node, std::vector<int32_t>(io_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::IsolateNode(FastNode *node, const std::vector<int32_t> &io_map) {
  GE_ASSERT_NOTNULL(node);
  const auto &in_nodes_to_out = GetFullConnectIONodes(node);
  GE_ASSERT_NOTNULL(node->GetExtendInfo());
  const auto graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  InFastNodesToOut data_in_to_out;
  GE_ASSERT_GRAPH_SUCCESS(RelinkDataIO(graph, node, io_map, data_in_to_out), "Relink data io failed for node %s",
                          node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(RelinkControlNodeIfNeed(graph, in_nodes_to_out, data_in_to_out),
                          "Relink control io failed for node %s", node->GetNamePtr());
  FastNodeUtils::UnlinkAll(node);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::ReplaceEdgeSrc(FastEdge *old_edge, const EdgeSrcEndpoint &new_src) {
  GE_ASSERT_NOTNULL(old_edge);
  const auto dst_node = old_edge->dst;
  GE_ASSERT_NOTNULL(dst_node);
  GE_ASSERT_NOTNULL(dst_node->GetExtendInfo());
  const auto graph = dst_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph, "Failed to replace edge source, node %s has null root graph", dst_node->GetNamePtr());
  if ((graph->RemoveEdge(old_edge) == GRAPH_SUCCESS) &&
      (graph->AddEdge(new_src.node, new_src.index, dst_node, old_edge->dst_input) != nullptr)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "Replace edge failed, dst_node:%s, new src node:%s, index:%d", dst_node->GetNamePtr(),
         new_src.node->GetNamePtr(), new_src.index);
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::RemoveSubgraphRecursively(ExecuteGraph *execute_graph, FastNode *remove_node) {
  GE_ASSERT_NOTNULL(execute_graph);
  GE_ASSERT_NOTNULL(remove_node);
  GE_ASSERT_NOTNULL(remove_node->GetOpDescBarePtr());
  if (remove_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
    GELOGD("Node %s has no subgraph.", remove_node->GetName().c_str());
    return GRAPH_SUCCESS;
  }

  const auto remove_extend_info = remove_node->GetExtendInfo();
  GE_ASSERT_NOTNULL(remove_extend_info);
  if (remove_extend_info->GetOwnerGraphBarePtr() == nullptr) {
    GELOGW("Node %s has an owner graph with null value.", remove_node->GetNamePtr());
    return GRAPH_SUCCESS;
  }

  if ((remove_extend_info->GetOwnerGraphBarePtr() != execute_graph) &&
      !execute_graph->CheckNodeIsInGraph(remove_node)) {
    GELOGW("Can not find node %s in graph %s.", remove_node->GetName().c_str(), execute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // find all subgraphs connecting to remove_node
  const auto root_graph = FindRootGraph(execute_graph);
  std::vector<ExecuteGraph *> subgraphs_to_remove;
  std::deque<const FastNode *> nodes_to_visit;
  nodes_to_visit.push_back(remove_node);
  const auto insert_func = [](const ExecuteGraph *const exe_graph, std::deque<const FastNode *> &candidates) -> void {
    auto iter = exe_graph->graph_shared_->nodes_.end();
    while (iter != exe_graph->graph_shared_->nodes_.begin()) {
      --iter;
      (void)candidates.insert(candidates.begin(), &FastGraphUtils::GetNode(iter.element_));
    }
  };
  while (!nodes_to_visit.empty()) {
    const auto curr_node = nodes_to_visit.front();
    nodes_to_visit.pop_front();
    const OpDesc *op_desc = curr_node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
      for (const auto &name : subgraph_names) {
        const auto subgraph = root_graph->GetSubGraph(name);
        if ((subgraph != nullptr) && (subgraph->graph_shared_ != nullptr)) {
          subgraphs_to_remove.emplace_back(subgraph);
          insert_func(subgraph, nodes_to_visit);
        }
      }
    }
  }

  // remove all subgraphs
  for (const auto &remove_graph : subgraphs_to_remove) {
    GE_ASSERT_GRAPH_SUCCESS(root_graph->RemoveSubGraph(remove_graph),
                            "[Remove][SubGraph] failed, sub graph name is %s, execute graph is %s.",
                            remove_node->GetNamePtr(), execute_graph->GetName().c_str());
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ExecuteGraphUtils::RemoveNodeWithoutRelink(ExecuteGraph *execute_graph, FastNode *node) {
  GE_ASSERT_NOTNULL(execute_graph);
  GE_ASSERT_NOTNULL(node, "param node is nullptr, check invalid.");
  // If the node save as input node, delete it
  (void)execute_graph->RemoveInputNode(node);

  // If the node save as output node, delete it
  (void)execute_graph->RemoveOutputNode(node);

  // If the node has sub-graphs, delete them
  GE_ASSERT_GRAPH_SUCCESS(RemoveSubgraphRecursively(execute_graph, node), "Remove subgraph of node %s failed.",
                          node->GetNamePtr());
  if (execute_graph->CheckNodeIsInGraph(node)) {
    GE_ASSERT_GRAPH_SUCCESS(execute_graph->RemoveJustNode(node), "Remove node %s failed.", node->GetNamePtr());
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::unordered_map<std::string, FastNode *>
ExecuteGraphUtils::GetNodeMapFromAllNodes(ExecuteGraph *exe_graph) {
  GE_ASSERT_NOTNULL(exe_graph);
  GE_ASSERT_NOTNULL(exe_graph->graph_shared_);
  const auto root_graph = FindRootGraph(exe_graph);
  GE_ASSERT_NOTNULL(root_graph);

  const auto insert_func = [](const ExecuteGraph *const exe_graph, std::deque<FastNode *> &candidates) -> void {
    auto iter = exe_graph->graph_shared_->nodes_.end();
    while (iter != exe_graph->graph_shared_->nodes_.begin()) {
      --iter;
      (void)candidates.insert(candidates.begin(), &FastGraphUtils::GetNode(iter.element_));
    }
  };
  std::deque<FastNode *> candidates;
  insert_func(exe_graph, candidates);
  std::unordered_map<std::string, FastNode *> node_name_to_nodes;
  while (!candidates.empty()) {
    const auto fast_node = candidates.front();
    candidates.pop_front();
    if ((fast_node == nullptr) || (fast_node->GetOpDescBarePtr() == nullptr)) {
      continue;
    }
    node_name_to_nodes.emplace(fast_node->GetName(), fast_node);
    const auto &subgraph_names = fast_node->GetOpDescBarePtr()->GetSubgraphInstanceNames();
    auto name_iter = subgraph_names.rbegin();
    while (name_iter != subgraph_names.rend()) {
      const auto subgraph = root_graph->GetSubGraph(*name_iter);
      if ((subgraph != nullptr) && (subgraph->graph_shared_ != nullptr)) {
        insert_func(subgraph, candidates);
      }
      ++name_iter;
    }
  }
  return node_name_to_nodes;
}
}  // namespace af
