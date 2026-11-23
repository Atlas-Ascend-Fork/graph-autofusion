/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/execute_graph_adapter.h"
#include "common/checker.h"
#include "fast_graph/fast_graph_impl.h"
#include "graph/compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"

namespace af {
namespace {
constexpr int32_t kHybridSubgraphRecursion = 32;
}  // namespace

ComputeGraphPtr ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(ExecuteGraph *src_graph) {
  GE_ASSERT_NOTNULL(src_graph);
  const auto dst_graph = ComGraphMakeShared<ComputeGraph>(src_graph->GetName());
  GE_ASSERT_NOTNULL(dst_graph);
  const int32_t depth = 0;
  GE_ASSERT_GRAPH_SUCCESS(ConvertExecuteGraphToComputeGraph(src_graph, dst_graph, depth),
                          "Convert execute graph:%s to compute graph failed.", src_graph->GetName().c_str());
  return dst_graph;
}

graphStatus ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(ExecuteGraph *src_graph,
                                                                   const ComputeGraphPtr &dst_graph,
                                                                   const int32_t depth) {
  GE_ASSERT_TRUE(depth <= kHybridSubgraphRecursion, "param depth:%d larger than %d(allow max subgraphs).", depth,
                 kHybridSubgraphRecursion);
  std::unordered_map<FastNode *, Node *> all_new_nodes;
  GE_ASSERT_GRAPH_SUCCESS(CopyOpAndSubgraph(src_graph, dst_graph, all_new_nodes, depth),
                          "Copy op and subgraph from %s to %s failed.", src_graph->GetName().c_str(),
                          dst_graph->GetName().c_str());

  for (const auto &n : src_graph->graph_shared_->nodes_) {
    GE_ASSERT_NOTNULL(n);
    GE_ASSERT_GRAPH_SUCCESS(RelinkGraphEdges(&FastGraphUtils::GetNode(n), all_new_nodes),
                            "Relink edge for node %s failed.", FastGraphUtils::GetNode(n).GetNamePtr());
  }

  std::vector<ComputeGraphPtr> new_subgraphs;
  const auto &old_subgraphs = src_graph->GetAllSubgraphs();
  for (const auto &sub_graph : old_subgraphs) {
    const auto new_subgraph = dst_graph->GetSubgraph(sub_graph->GetName());
    GE_CHECK_NOTNULL(new_subgraph);
    new_subgraphs.emplace_back(new_subgraph);
  }
  dst_graph->SetAllSubgraphs(new_subgraphs);

  GE_ASSERT_GRAPH_SUCCESS(CopyMembers(src_graph, dst_graph, all_new_nodes));

  // inherit all attr from old graph to new graph
  InheritOriginalAttr(src_graph, dst_graph);
  return GRAPH_SUCCESS;
}

graphStatus ExecuteGraphAdapter::CopyOpAndSubgraph(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph,
                                                   std::unordered_map<FastNode *, Node *> &all_new_nodes,
                                                   const int32_t depth) {
  const auto src_root_graph = ExecuteGraphUtils::FindRootGraph(src_graph);
  GE_ASSERT_NOTNULL(src_root_graph);
  const auto dst_root_graph = GraphUtils::FindRootGraph(dst_graph);
  GE_ASSERT_NOTNULL(dst_root_graph);
  for (const auto &src_node : src_graph->graph_shared_->nodes_) {
    GE_ASSERT_NOTNULL(src_node);
    // 复用原图的OpDesc对象，原图不能释放
    const auto &op_desc = FastGraphUtils::GetNode(src_node).GetOpDescPtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto &dst_node = dst_graph->AddNode(op_desc, op_desc->GetId());
    GE_ASSERT_NOTNULL(dst_node, "Add node:%s for dst graph failed.", op_desc->GetName().c_str());
    all_new_nodes[&FastGraphUtils::GetNode(src_node)] = dst_node.get();

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    const auto subgraph_num = subgraph_names.size();
    for (size_t subgrah_idx = 0U; subgrah_idx < subgraph_num; ++subgrah_idx) {
      const auto &subgraph_name = subgraph_names[subgraph_num - 1U - subgrah_idx];
      const auto src_subgraph = src_root_graph->GetSubGraph(subgraph_name);
      if ((src_subgraph == nullptr) && subgraph_name.empty()) {
        continue;
      }
      GE_ASSERT_NOTNULL(src_subgraph);
      const auto dst_subgraph = ComGraphMakeShared<ComputeGraph>(src_subgraph->GetName());
      GE_ASSERT_NOTNULL(dst_subgraph);
      dst_subgraph->SetParentGraph(dst_root_graph);
      GE_ASSERT_GRAPH_SUCCESS(ConvertExecuteGraphToComputeGraph(src_subgraph, dst_subgraph, depth + 1),
                              "Copy subgraph from %s to %s failed.", src_subgraph->GetName().c_str(),
                              dst_subgraph->GetName().c_str());
      (void)dst_root_graph->AddSubGraph(dst_subgraph);
      dst_subgraph->SetParentNode(dst_node);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ExecuteGraphAdapter::RelinkGraphEdges(FastNode *old_node,
                                                  const std::unordered_map<FastNode *, Node *> &all_new_nodes) {
  const auto &iter = all_new_nodes.find(old_node);
  GE_ASSERT_TRUE(iter != all_new_nodes.end(), "all_new_nodes does not contain %s", old_node->GetNamePtr());
  const auto &new_node = iter->second;
  GE_ASSERT_NOTNULL(new_node);
  const auto &old_out_edges = old_node->GetAllOutDataEdgesRef();
  for (size_t out_i = 0; out_i < old_out_edges.size(); ++out_i) {
    for (const auto old_edge : old_out_edges[out_i]) {
      if (old_edge == nullptr) {
        continue;
      }
      const auto old_dst_node = old_edge->dst;
      GE_ASSERT_NOTNULL(old_dst_node);
      const auto dst_index = old_edge->dst_input;

      const auto &dst_iter = all_new_nodes.find(old_dst_node);
      if (dst_iter != all_new_nodes.end()) {
        const auto &new_dst_node = dst_iter->second;
        GE_ASSERT_NOTNULL(new_dst_node);
        GE_ASSERT_GRAPH_SUCCESS(
            GraphUtils::AddEdge(new_node->GetOutDataAnchor(out_i), new_dst_node->GetInDataAnchor(dst_index)),
            "Add edge %s:%d -> %s:%d failed.", new_node->GetName().c_str(), out_i, new_dst_node->GetName().c_str(),
            dst_index);
      }
    }
  }

  for (const auto old_control_out_edge : old_node->GetAllOutControlEdgesRef()) {
    if (old_control_out_edge == nullptr) {
      continue;
    }
    const auto old_dst_node = old_control_out_edge->dst;
    GE_ASSERT_NOTNULL(old_dst_node);

    auto dst_iter = all_new_nodes.find(old_dst_node);
    if (dst_iter != all_new_nodes.end()) {
      const auto &new_dst_node = dst_iter->second;
      GE_ASSERT_NOTNULL(new_dst_node);
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), new_dst_node->GetInControlAnchor()),
                              "Add control edge %s -> %s failed.", new_node->GetName().c_str(),
                              new_dst_node->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ExecuteGraphAdapter::CopyMembers(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph,
                                             const std::unordered_map<FastNode *, Node *> &all_new_nodes) {
  GE_ASSERT_NOTNULL(src_graph);
  GE_ASSERT_NOTNULL(dst_graph);
  GE_ASSERT_NOTNULL(src_graph->graph_shared_);
  GE_ASSERT_NOTNULL(dst_graph->impl_);

  // copy info of output nodes from old graph to new graph.
  const auto &out_nodes_info = src_graph->graph_shared_->GetAllOutNodeInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  // ExecuteGraph未开放OutNodeInfo的接口，只是预埋了实现，以下流程暂时业务流程走不到
  for (const auto &info : out_nodes_info) {
    GE_ASSERT_NOTNULL(info.first);
    const auto it = all_new_nodes.find(info.first);
    if (it != all_new_nodes.end()) {
      new_out_nodes_info.emplace_back(std::shared_ptr<af::Node>(it->second), info.second);
    }
  }
  GE_ASSERT_SUCCESS(dst_graph->SetGraphOutNodesInfo(new_out_nodes_info));

  // copy info of input nodes from old graph to new graph.
  const auto &input_nodes = src_graph->graph_shared_->GetAllInputNodeInfo();
  for (const auto &node : input_nodes) {
    GE_ASSERT_NOTNULL(node);
    const auto &it = all_new_nodes.find(node);
    if (it != all_new_nodes.end()) {
      (void)dst_graph->AddInputNode(it->second->shared_from_this());
    }
  }

  // ExecuteGraph没有target信息 & other members
  // graph属性序列化
  dst_graph->impl_->attrs_ = src_graph->attrs_;
  return GRAPH_SUCCESS;
}

void ExecuteGraphAdapter::InheritOriginalAttr(ExecuteGraph *src_graph, const ComputeGraphPtr &dst_graph) {
  const auto &original_attrs = AttrUtils::GetAllAttrs(src_graph);
  for (const auto &attr_iter : original_attrs) {
    if (dst_graph->TrySetAttr(attr_iter.first, attr_iter.second) != GRAPH_SUCCESS) {
      GELOGW("Set inherit original attr[%s] failed, Please Check.", attr_iter.first.c_str());
    }
  }
  // copy ExtAttr to dst_graph
  dst_graph->CopyFrom(*src_graph);
}
}  // namespace af
