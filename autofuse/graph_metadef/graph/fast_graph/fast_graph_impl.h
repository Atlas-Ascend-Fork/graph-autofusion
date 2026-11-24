/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAST_GRAPH_FAST_GRAPH_IMPL_H
#define FAST_GRAPH_FAST_GRAPH_IMPL_H

#include <unordered_map>
#include "graph/fast_graph/fast_node.h"
#include "quick_list.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/utils/ge_ir_utils.h"
#include "common/ge_common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "fast_graph_utils.h"

namespace af {
namespace {
constexpr int32_t kInvalidIndex = -1;
}  // namespace
template <class NodeT, class GraphT>
class FastGraphImpl {
 public:
  explicit FastGraphImpl(const std::string &name)
      : name_(std::move(name)), parent_graph_(nullptr), parent_node_(nullptr) {}
  FastGraphImpl() = default;

  ~FastGraphImpl() {
    for (auto iter = nodes_.begin(); iter != nodes_.end();) {
      auto quick_node = *iter;
      iter = nodes_.erase(iter);
      if (quick_node == nullptr) {
        continue;
      }
      // for add extra ref count.
      // if not the line, quick_node maybe release in SetNodePtr(nullptr).
      if (FastGraphUtils::GetNode(quick_node).GetNodeBarePtr() != nullptr) {
        auto node_ptr = FastGraphUtils::GetNode(quick_node).GetNodePtr();
        FastGraphUtils::GetNode(quick_node).ClearNodePtr();
      } else {
        free_nodes_.push_back(quick_node, ListMode::kFreeMode);
        ClearNodeRelateInfo(quick_node);
      }
    }

    for (auto iter = free_nodes_.begin(); iter != free_nodes_.end();) {
      auto quick_node = *iter;
      iter = free_nodes_.erase(iter);
      if (quick_node != nullptr) {
        delete quick_node;
      }
    }

    for (auto iter = free_edges_.begin(); iter != free_edges_.end();) {
      auto edge = *iter;
      iter = free_edges_.erase(iter);
      if (edge != nullptr) {
        delete edge;
      }
    }

    for (auto iter = sub_graphs_.begin(); iter != sub_graphs_.end();) {
      auto item = *iter;
      iter = sub_graphs_.erase(iter);
      if (item != nullptr) {
        delete item;
      }
    }

    for (auto iter = free_sub_graphs_.begin(); iter != free_sub_graphs_.end();) {
      auto item = *iter;
      iter = free_sub_graphs_.erase(iter);
      if (item != nullptr) {
        delete item;
      }
    }
  }

  void SetOwnerGraph(GraphT *graph) {
    owner_graph_ = graph;
  }

  /**
   * The function Set edge owner to the edges_.
   */
  void MoveEdgeToGraph(const FastEdge *const edge) {
    ListElement<Edge<FastNode>> *element = FastGraphUtils::GetListElementAddr(edge);
    if ((FastGraphUtils::GetOwner(element) != &edges_) && (FastGraphUtils::GetOwner(element) != nullptr)) {
      FastGraphUtils::GetOwner(element)->erase(element);
    }

    if (FastGraphUtils::GetOwner(element) == &edges_) {
      return;
    }

    edges_.push_back(element, ListMode::kWorkMode);
  }

  /**
   * The function provide the deep copy of graph to other graph.
   */
  graphStatus DeepCopy(const FastGraphImpl &graph) {
    std::unordered_map<const NodeT *, NodeT *> origin_node_to_copy_node;
    DeepCopyNodes(graph, origin_node_to_copy_node);
    DeepCopyEdges(graph, origin_node_to_copy_node);
    DeepCopySubGraphs(graph);
    DeepCopyInputNodes(graph, origin_node_to_copy_node);
    DeepCopyOutputNodes(graph, origin_node_to_copy_node);

    name_ = graph.name_;
    graph_id_ = graph.graph_id_;
    parent_graph_ = graph.parent_graph_;
    parent_node_ = graph.parent_node_;
    graph_netoutput_ = graph.graph_netoutput_;
    extend_info_ = graph.extend_info_;
    return GRAPH_SUCCESS;
  }

  graphStatus SetNodesAfterSorting(const std::vector<FastNode *> &nodes) {
    nodes_.clear();
    for (size_t i = 0UL; i < nodes.size(); i++) {
      auto node = nodes[i];
      if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
        REPORT_INNER_ERR_MSG("E18888", "The node ptr or op_desc should not be null.");
        GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op_desc should not be null.");
        return PARAM_INVALID;
      }
      node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));
      PushBackToNodeList(FastGraphUtils::GetListElementAddr(node));
    }
    return GRAPH_SUCCESS;
  }

  /**
   * The function is provide:
   * 1. set node to nodes_
   * 2. if it is data, set it to input_nodes_
   */
  graphStatus SetNodes(const std::vector<FastNode *> &nodes) {
    nodes_.clear();
    for (size_t i = 0UL; i < nodes.size(); i++) {
      auto node = nodes[i];
      if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
        REPORT_INNER_ERR_MSG("E18888", "The node ptr or op_desc should not be null.");
        GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op_desc should not be null.");
        return PARAM_INVALID;
      }
      RecordNodeAndInputDataToGraph(node);
    }
    return GRAPH_SUCCESS;
  }

  NodeT *AddInputNode(NodeT *const node) {
    if (node->GetExtendInfo() == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The node extend info should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node extend info should not be null.");
      return nullptr;
    }

    auto input_idx = node->GetExtendInfo()->GetInputIndex();
    auto already_exist_flag = (input_idx >= 0) && (input_idx < static_cast<int32_t>(input_nodes_.size())) &&
                              (node == input_nodes_[input_idx]);
    if (!already_exist_flag) {
      node->GetExtendInfo()->SetInputIndex(input_nodes_.size());
      input_nodes_.push_back(node);
    }

    if (CheckNodeIsInGraph(node)) {
      return node;
    }

    return AddNode(node);
  }

  graphStatus RemoveInputNode(NodeT *const node) {
    if ((node == nullptr) || (node->GetExtendInfo() == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null, graph:%s.", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return GRAPH_FAILED;
    }

    auto input_idx = node->GetExtendInfo()->GetInputIndex();
    if ((input_idx >= 0) && (input_idx < static_cast<int32_t>(input_nodes_.size())) &&
        (node == input_nodes_[input_idx])) {
      input_nodes_[input_idx] = nullptr;
      node->GetExtendInfo()->SetInputIndex(kInvalidIndex);
      return SUCCESS;
    }

    GELOGW("[Remove][Node] Failed to remove input node.");
    return GRAPH_FAILED;
  }

  NodeT *AddOutputNodeByIndex(NodeT *const node, const int32_t index) {
    if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr) || node->GetExtendInfo() == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr or opdesc should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or opdesc should not be null.");
      return nullptr;
    }

    bool already_have = false;
    NodeT *result = node;
    // [output_nodes_info_ : should not be null]
    auto &output_idxs = node->GetExtendInfo()->GetOutputIndex();
    for (auto item : output_idxs) {
      auto flag = (item >= 0) && (item < static_cast<int32_t>(output_nodes_.size())) &&
                  (node == output_nodes_[item].first) && (output_nodes_[item].second == item);
      if (flag) {
        already_have = true;
        result = output_nodes_[item].first;
        break;
      }
    }

    if (!already_have) {
      node->GetExtendInfo()->AddOneOutputIndex(output_nodes_.size());
      output_nodes_.emplace_back(std::make_pair(node, index));
      GELOGI("Push back node name:%s, index:%d, into output_nodes_info_.", node->GetName().c_str(), index);
    }

    if (!CheckNodeIsInGraph(node)) {
      GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "[Add][Node] failed");
    }

    return result;
  }

  const std::vector<NodeT *> &GetAllInputNodeInfo() const {
    return input_nodes_;
  }

  const std::vector<std::pair<NodeT *, int32_t>> &GetAllOutNodeInfo() const {
    return output_nodes_;
  }

  std::vector<NodeT *> GetInputNodes() const {
    return input_nodes_;
  }

  std::vector<std::pair<NodeT *, int32_t>> GetAllOutNodes() const {
    return output_nodes_;
  }

  void SetGraphOutNodesInfo(const std::vector<std::pair<NodeT *, int32_t>> &out_nodes_info) {
    output_nodes_ = out_nodes_info;
  }

  void AppendGraphOutNodesInfo(std::vector<std::pair<NodeT *, int32_t>> &out_nodes_info) {
    (void)output_nodes_.insert(output_nodes_.cend(), out_nodes_info.cbegin(), out_nodes_info.cend());
  }

  graphStatus RemoveOutputNode(const NodeT *const node) {
    if ((node == nullptr) || (node->GetExtendInfo() == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null, graph:%s.", name_.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return GRAPH_FAILED;
    }

    bool find_node = false;
    // [output_nodes_info_ : should not be null]
    auto &output_idxs = node->GetExtendInfo()->GetOutputIndex();
    auto iter = output_idxs.begin();
    while (iter != output_idxs.end()) {
      auto item = *iter;
      auto flag =
          (item >= 0) && (item < static_cast<int32_t>(output_nodes_.size())) && (node == output_nodes_[item].first);
      if (flag) {
        output_nodes_[item] = {nullptr, -1};
        find_node = true;
        iter = output_idxs.erase(iter);
      } else {
        ++iter;
      }
    }

    GE_IF_BOOL_EXEC(!find_node, return GRAPH_FAILED);
    return GRAPH_SUCCESS;
  }

  NodeT *AddNode(NodeT *const fast_node) {
    if ((fast_node == nullptr) || (fast_node->GetOpDescBarePtr() == nullptr) ||
        (fast_node->GetExtendInfo() == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "the node ptr or op desc ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc ptr should not be null.");
      return nullptr;
    }

    fast_node->GetExtendInfo()->SetHostNode(extend_info_.is_valid_flag_);
    fast_node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(GetDirectNodesSize()));
    RecordNodeAndInputDataToGraph(fast_node);
    return fast_node;
  }

  NodeT *AddNode(const OpDescPtr &op) {
    NodeT *node_ptr = CreateOneNode(op, static_cast<int64_t>(GetDirectNodesSize()));
    return AddNode(node_ptr);
  }

  NodeT *AddNode(const OpDescPtr &op, const int64_t id) {
    NodeT *node_ptr = CreateOneNode(op, id);
    if ((node_ptr == nullptr) || (node_ptr->GetExtendInfo() == nullptr)) {
      return nullptr;
    }
    node_ptr->GetExtendInfo()->SetHostNode(extend_info_.is_valid_flag_);
    RecordNodeAndInputDataToGraph(node_ptr);
    return node_ptr;
  }

  NodeT *AddNodeFront(NodeT *const fast_node) {
    if ((fast_node == nullptr) || (fast_node->GetOpDescBarePtr() == nullptr) ||
        (fast_node->GetExtendInfo() == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr or op desc should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc should not be null.");
      return nullptr;
    }

    fast_node->GetExtendInfo()->SetHostNode(extend_info_.is_valid_flag_);
    fast_node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(GetDirectNodesSize()));

    auto quick_node = FastGraphUtils::GetListElementAddr(fast_node);
    if (FastGraphUtils::GetOwner(quick_node) != nullptr) {
      FastGraphUtils::GetOwner(quick_node)->erase(quick_node);
    }

    auto pos = nodes_.begin();
    if (*pos == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The node begin ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node begin ptr should not be null.");
      return nullptr;
    }

    if ((GetDirectNodesSize() > 0UL) && ((*pos)->data.GetType() == DATA)) {
      pos = std::next(nodes_.begin());
    }

    GELOGD("[insert][NodeT] node = %p.", quick_node);
    nodes_.insert(pos, quick_node, ListMode::kWorkMode);
    fast_node->GetExtendInfo()->SetOwnerGraph(owner_graph_, fast_node);
    CheckAndRecordInputNode(fast_node);
    return fast_node;
  }

  NodeT *AddNodeFront(const OpDescPtr &op) {
    NodeT *node_ptr = CreateOneNode(op, static_cast<int64_t>(GetDirectNodesSize()));
    return AddNodeFront(node_ptr);
  }

  graphStatus RemoveJustNode(ListElement<NodeT> *node_ptr) {
    if ((node_ptr == nullptr) || (node_ptr->owner == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return GRAPH_FAILED;
    }

    if (FastGraphUtils::GetOwner(node_ptr) != &nodes_) {
      if ((FastGraphUtils::GetMode(node_ptr) == ListMode::kFreeMode) &&
          (FastGraphUtils::GetOwner(node_ptr) != nullptr)) {
        return GRAPH_SUCCESS;
      }
      /* already add to other graph, so it can`t remove from current graph. */
      GELOGW("[Remove][Node] The node is not in the graph, please check the node.");
      return GRAPH_NOT_CHANGED;
    }

    (void)nodes_.erase(node_ptr);
    if (FastGraphUtils::GetNode(node_ptr).GetNodeBarePtr() == nullptr) {
      free_nodes_.push_back(node_ptr, ListMode::kFreeMode);
    }

    return GRAPH_SUCCESS;
  }

  graphStatus RecycleQuickNode(ListElement<NodeT> *const quick_node) {
    if (quick_node == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return GRAPH_FAILED;
    }

    if (FastGraphUtils::GetOwner(quick_node) != nullptr) {
      FastGraphUtils::GetOwner(quick_node)->erase(quick_node);
    }

    free_nodes_.push_back(quick_node, ListMode::kFreeMode);
    return GRAPH_SUCCESS;
  }

  graphStatus RecycleQuickEdge(ListElement<Edge<NodeT>> *const list_edge) {
    if (list_edge == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return GRAPH_FAILED;
    }

    if (FastGraphUtils::GetOwner(list_edge) != nullptr) {
      FastGraphUtils::GetOwner(list_edge)->erase(list_edge);
    }

    free_edges_.push_back(list_edge, ListMode::kFreeMode);
    return GRAPH_SUCCESS;
  }

  graphStatus UpdateNodePos(ListElement<NodeT> *const need_move_node, ListElement<NodeT> *const dst_node,
                            bool before_insert) {
    return nodes_.move(need_move_node, dst_node, before_insert);
  }

  FastEdge *AddEdge(NodeT *const src, int32_t src_index, NodeT *const dst, int32_t dst_index) {
    if ((src == nullptr) || (dst == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
      return nullptr;
    }

    auto io_index_valid_flag = ((src_index == kControlEdgeIndex) && (dst_index != kControlEdgeIndex)) ||
                               ((src_index != kControlEdgeIndex) && (dst_index == kControlEdgeIndex));
    if (io_index_valid_flag) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to check output index [%d] or input index [%d].", src_index, dst_index);
      GELOGE(GRAPH_FAILED, "[check][index] Failed to check output index [%d] or input index [%d].", src_index,
             dst_index);
      return nullptr;
    }

    ListElement<Edge<NodeT>> *edge = nullptr;
    if (free_edges_.empty()) {
      edge = new (std::nothrow) ListElement<Edge<NodeT>>;
    } else {
      auto iter = free_edges_.begin();
      edge = *iter;
      free_edges_.erase(iter);
    }
    if (edge == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to malloc memory for edge.");
      GELOGE(GRAPH_FAILED, "[malloc][edge] Failed to malloc memory for edge.");
      return nullptr;
    }

    FastGraphUtils::GetEdgeSrc(edge) = src;
    FastGraphUtils::GetEdgeDst(edge) = dst;
    FastGraphUtils::GetEdgeSrcOutput(edge) = src_index;
    FastGraphUtils::GetEdgeDstInput(edge) = dst_index;

    auto ret = src->RecordEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionOutType);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to record edge in the output node.");
      GELOGE(GRAPH_FAILED, "[malloc][edge] Failed to record edge in the output node.");
      free_edges_.push_back(edge, ListMode::kFreeMode);
      return nullptr;
    }

    ret = dst->RecordEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionInType);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to record edge in the input node.");
      GELOGE(GRAPH_FAILED, "[malloc][edge] Failed to record edge in the input node.");
      src->EraseEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionOutType);
      free_edges_.push_back(edge, ListMode::kFreeMode);
      return nullptr;
    }

    edges_.push_back(edge, ListMode::kWorkMode);
    return &FastGraphUtils::GetEdge(edge);
  }

  /**
   * currently, The function only remove edge belongs to self graph.
   */
  graphStatus RemoveEdge(ListElement<Edge<NodeT>> *const edge) {
    if (edge == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The edge ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The edge ptr should not be null.");
      return GRAPH_FAILED;
    }

    if (FastGraphUtils::GetOwner(edge) != &edges_) {
      if ((FastGraphUtils::GetMode(edge) == ListMode::kFreeMode) && (FastGraphUtils::GetOwner(edge) != nullptr)) {
        return GRAPH_SUCCESS;
      }

      GELOGW("[Remove][Edge] The edge is not in the graph, please check the edge.");
      return GRAPH_NOT_CHANGED;
    }

    if (FastGraphUtils::GetEdgeSrc(edge) != nullptr) {
      FastGraphUtils::GetEdgeSrc(edge)->EraseEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionOutType);
      FastGraphUtils::GetEdgeSrc(edge) = nullptr;
    }

    if (FastGraphUtils::GetEdgeDst(edge) != nullptr) {
      FastGraphUtils::GetEdgeDst(edge)->EraseEdge(&FastGraphUtils::GetEdge(edge), DirectionType::kDirectionInType);
      FastGraphUtils::GetEdgeDst(edge) = nullptr;
    }

    edges_.erase(edge);
    free_edges_.push_back(edge, ListMode::kFreeMode);
    return GRAPH_SUCCESS;
  }

  ListElement<GraphT *> *AddSubGraph(GraphT *const sub_graph) {
    ListElement<GraphT *> *quick_graph = nullptr;
    if (free_sub_graphs_.empty()) {
      quick_graph = new (std::nothrow) ListElement<GraphT *>;
    } else {
      auto iter = free_sub_graphs_.begin();
      quick_graph = *iter;
      free_sub_graphs_.erase(iter);
    }

    if (quick_graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to create a subgraph.");
      GELOGE(GRAPH_FAILED, "[Create][SubGraph] Failed to create a subgraph.");
      return nullptr;
    }

    quick_graph->data = sub_graph;
    sub_graphs_.push_back(quick_graph, ListMode::kWorkMode);
    return quick_graph;
  }

  graphStatus RemoveSubGraph(ListElement<GraphT *> *const sub_graph) {
    if (FastGraphUtils::GetOwner(sub_graph) != &sub_graphs_) {
      if ((FastGraphUtils::GetMode(sub_graph) == ListMode::kFreeMode) &&
          (FastGraphUtils::GetOwner(sub_graph) != nullptr)) {
        return GRAPH_SUCCESS;
      }

      /* already add to other graph, so it can`t remove from current graph. */
      GELOGW("[Remove][SubGraph] The sub graph is not in the graph, please check the sub graph.");
      return GRAPH_NOT_CHANGED;
    }

    sub_graphs_.erase(sub_graph);
    free_sub_graphs_.push_back(sub_graph, ListMode::kFreeMode);
    return GRAPH_SUCCESS;
  }

  void ClearAllSubGraph() {
    ClearGraphs();
  }

  void SetGraphId(size_t graph_id) {
    graph_id_ = graph_id;
  }

  size_t GetGraphId(void) const {
    return graph_id_;
  }

  void SetParentNode(NodeT *const parent_node) {
    parent_node_ = parent_node;
  }

  NodeT *GetParentNode(void) const {
    return parent_node_;
  }

  void SetParentGraph(GraphT *const parent_graph) {
    parent_graph_ = parent_graph;
  }

  const GraphT *GetParentGraph(void) const {
    return parent_graph_;
  }

  GraphT *GetParentGraph(void) {
    return parent_graph_;
  }

  const QuickList<NodeT> &GetAllNodeInfo(void) const {
    return nodes_;
  }

  QuickList<NodeT> &GetAllNodeInfoForModify(void) {
    return nodes_;
  }

  void SetAllInputNodeInfo(const std::vector<NodeT *> &inputs) {
    input_nodes_.swap(inputs);
  }

  size_t GetAllSubGraphSize(void) const {
    return sub_graphs_.size();
  }

  void SetNetOutputNode(NodeT *const netoutput_node) {
    graph_netoutput_ = netoutput_node;
  }

  void SetName(const std::string &name) {
    name_ = name;
  }
  std::string GetName() const {
    return name_;
  }

  size_t GetDirectNodesSize() const {
    return nodes_.size();
  }

  const QuickList<NodeT> &GetRawDirectNode() const {
    return nodes_;
  }

  QuickList<NodeT> &GetDirectNodeToModify() const {
    return nodes_;
  }

  QuickList<Edge<NodeT>> &GetRawAllEdges() const {
    return edges_;
  }

  QuickList<GraphT *> &GetRawAllSubgraphs() const {
    return sub_graphs_;
  }

  std::vector<NodeT *> GetDirectNode() const {
    return nodes_.CollectAllPtrItemToVector();
  }

  std::vector<Edge<NodeT> *> GetAllEdges() const {
    return edges_.CollectAllPtrItemToVector();
  }

  std::vector<GraphT *> GetAllSubgraphs() const {
    return sub_graphs_.CollectAllItemToVector();
  }

  const ListElement<NodeT> *FindNode(size_t token) const {
    for (const auto &node : nodes_) {
      if (node == nullptr) {
        continue;
      }
      if (FastGraphUtils::GetConstNode(node).GetNodeToken() == token) {
        return node;
      }
    }
    return nullptr;
  }

  bool IsValid() const {
    return extend_info_.is_valid_flag_;
  }

  void SetValidFlag(bool flag) {
    extend_info_.is_valid_flag_ = flag;
  }

  void InValid() {
    extend_info_.is_valid_flag_ = false;
  }

  bool operator==(const FastGraphImpl<NodeT, GraphT> &r_graph) const {
    return (IsEqual(this->name_, r_graph.name_, "graph.name") &&
            IsEqual(this->graph_id_, r_graph.graph_id_, "graph.graph_id") &&
            IsEqual(this->GetDirectNodesSize(), r_graph.GetDirectNodesSize(), "graph.nodes.size()") &&
            IsEqual(this->edges_.size(), r_graph.edges_.size(), "graph.edge.size()") &&
            IsEqual(this->sub_graphs_.size(), r_graph.sub_graphs_.size(), "graph.sub_graph.size()") &&
            IsEqual(this->parent_graph_, r_graph.parent_graph_, "graph.parent_graph") &&
            IsEqual(this->parent_node_, r_graph.parent_node_, "graph.parent_node") &&
            IsEqual(this->graph_netoutput_, r_graph.graph_netoutput_, "graph.graph_netoutput") &&
            IsEqual(this->extend_info_.is_valid_flag_, r_graph.extend_info_.is_valid_flag_, "graph.is_valid_flag_"));
  }

  void Swap(FastGraphImpl<NodeT, GraphT> &graph) {
    name_.swap(graph.name_);
    std::swap(graph_id_, graph.graph_id_);
    nodes_.swap(graph.nodes_);
    edges_.swap(graph.edges_);
    input_nodes_.swap(graph.input_nodes_);
    output_nodes_.swap(graph.output_nodes_);
    sub_graphs_.swap(graph.sub_graphs_);

    std::swap(parent_graph_, graph.parent_graph_);
    std::swap(parent_node_, graph.parent_node_);
    std::swap(graph_netoutput_, graph.graph_netoutput_);
    std::swap(extend_info_.is_valid_flag_, graph.extend_info_.is_valid_flag_);
  }

  void SetSubGraph(const std::vector<GraphT *> &sub_graphs) {
    ClearGraphs();
    std::for_each(sub_graphs.begin(), sub_graphs.end(), [this](GraphT *graph) {
      if (graph != nullptr) {
        AddSubGraph(graph);
      }
    });
  }

  bool CheckNodeIsInGraph(const FastNode *const node) {
    ListElement<NodeT> *quick_node = FastGraphUtils::GetListElementAddr(node);
    return FastGraphUtils::GetOwner(quick_node) == &nodes_;
  }

  bool CheckEdgeIsInGraph(const FastEdge *const edge) {
    ListElement<Edge<NodeT>> *element = FastGraphUtils::GetListElementAddr(edge);
    return FastGraphUtils::GetOwner(element) == &edges_;
  }

  graphStatus ClearNode(const std::function<graphStatus(QuickNode *quick_node)> &clear_oper) {
    auto iter = nodes_.begin();
    while (iter != nodes_.end()) {
      QuickNode *quick_node = *iter;
      ++iter;
      auto ret = clear_oper(quick_node);
      if (ret != GRAPH_SUCCESS) {
        return ret;
      }
    }
    return GRAPH_SUCCESS;
  }

  void ReorderByNodeId() {
    nodes_.sort([](const ListElement<NodeT> *lhs, const ListElement<NodeT> *rhs) {
      return FastGraphUtils::GetConstNode(lhs).GetOpDescBarePtr()->GetId() <
             FastGraphUtils::GetConstNode(rhs).GetOpDescBarePtr()->GetId();
    });
  }

  /**
   * The Function is used to delete edge which is not in the graph.
   * it can used for the following scenarios:
   * 1. Remove node a in graph A;
   * 2. Add node a in graph B;
   * 3. remove edges of all node a in graph A.
   */
  void ForceDeleteEdge(FastEdge *const e) {
    if (e == nullptr) {
      return;
    }
    auto quick_edge = FastGraphUtils::GetListElementAddr(e);
    if (FastGraphUtils::GetEdgeDst(quick_edge) != nullptr) {
      FastGraphUtils::GetEdgeDst(quick_edge)->EraseEdge(e, DirectionType::kDirectionInType);
      FastGraphUtils::GetEdgeDst(quick_edge) = nullptr;
    }

    if (FastGraphUtils::GetEdgeSrc(quick_edge) != nullptr) {
      FastGraphUtils::GetEdgeSrc(quick_edge)->EraseEdge(e, DirectionType::kDirectionOutType);
      FastGraphUtils::GetEdgeSrc(quick_edge) = nullptr;
    }

    if (FastGraphUtils::GetOwner(quick_edge) != nullptr) {
      FastGraphUtils::GetOwner(quick_edge)->erase(quick_edge);
      free_edges_.push_back(quick_edge, ListMode::kFreeMode);
    } else {
      // The rt2 move edge of two nodes into same graph.
      // it modify the owner of nodes, but not modify the owner of edges.
      // it will be result in the inability to delete the edge.
      // Therefore, if check the edge is nullptr, it push to the free edges
      free_edges_.push_back(quick_edge, ListMode::kFreeMode);
    }
  }

 private:
  graphStatus ClearNodeRelateInfo(ListElement<NodeT> *const node_ptr) {
    FastGraphUtils::GetNode(node_ptr).RemoveAllEdge([this](FastEdge *e) { ForceDeleteEdge(e); });
    return GRAPH_SUCCESS;
  }

  void CheckAndRecordInputNode(NodeT *const node) {
    if ((node == nullptr) || (node->GetExtendInfo() == nullptr)) {
      return;
    }

    auto input_idx = node->GetExtendInfo()->GetInputIndex();
    auto already_exist_flag = (input_idx >= 0) && (input_idx < static_cast<int32_t>(input_nodes_.size())) &&
                              (node == input_nodes_[input_idx]);
    if (OpTypeUtils::IsDataNode(node->GetType()) && (!already_exist_flag)) {
      node->GetExtendInfo()->SetInputIndex(input_nodes_.size());
      input_nodes_.push_back(node);
    }
  }

  /**
   * the input parameter (node) can`t be empty.
   * it need to check in upper-layer functions.
   */
  void PushBackToNodeList(ListElement<NodeT> *const node) {
    if (FastGraphUtils::GetOwner(node) != nullptr) {
      node->owner->erase(node);
    }
    GELOGD("[Add][NodeT] node = %p.", node);
    nodes_.push_back(node, ListMode::kWorkMode);
  }

  void RecordNodeAndInputDataToGraph(NodeT *const node) {
    PushBackToNodeList(FastGraphUtils::GetListElementAddr(node));
    node->GetExtendInfo()->SetOwnerGraph(owner_graph_, node);
    CheckAndRecordInputNode(node);
  }

  void DeepCopyNodes(const FastGraphImpl &other_graph,
                     std::unordered_map<const NodeT *, NodeT *> &origin_node_to_copy_node) {
    ClearNodes();
    for (auto iter = other_graph.nodes_.begin(); iter != other_graph.nodes_.end(); ++iter) {
      auto origin_node = *iter;
      if (origin_node == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "The node is nullptr in src graph.");
        GELOGW("[DeepCopyNodes] The node is nullptr in src graph, skip it.");
        continue;
      }

      OpDescPtr opdesc_ptr = std::make_shared<OpDesc>(*(FastGraphUtils::GetConstNode(origin_node).GetOpDescPtr()));
      auto copy_node = AddNode(opdesc_ptr);
      if (copy_node == nullptr) {
        continue;
      }
      origin_node_to_copy_node.insert(std::make_pair(&FastGraphUtils::GetConstNode(origin_node), copy_node));
    }
  }

  void DeepCopyEdges(const FastGraphImpl &other_graph,
                     const std::unordered_map<const NodeT *, NodeT *> &origin_node_to_copy_node) {
    ClearEdges();
    for (auto origin_edge : other_graph.edges_) {
      if (origin_edge == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "The edge is nullptr in src graph.");
        GELOGW("[DeepCopyEdges] The edge is nullptr in src graph, skip it.");
        continue;
      }

      NodeT *new_src_node = nullptr;
      NodeT *new_dst_node = nullptr;
      auto map_iter = origin_node_to_copy_node.find(FastGraphUtils::GetConstEdgeSrc(origin_edge));
      if (map_iter != origin_node_to_copy_node.end()) {
        new_src_node = map_iter->second;
      } else {
        // 这里不需要报错，是为了支持部分拷贝的情况
        GELOGI("[DeepCopyEdges] Can not find src node to add edge, skip it.");
        continue;
      }

      map_iter = origin_node_to_copy_node.find(FastGraphUtils::GetConstEdgeDst(origin_edge));
      if (map_iter != origin_node_to_copy_node.end()) {
        new_dst_node = map_iter->second;
      } else {
        // 这里不需要报错，是为了支持部分拷贝的情况
        GELOGI("[DeepCopyEdges] Can not find dst node to add edge, skip it.");
        continue;
      }

      auto copy_edge = AddEdge(new_src_node, FastGraphUtils::GetConstEdgeSrcOutput(origin_edge), new_dst_node,
                               FastGraphUtils::GetConstEdgeDstInput(origin_edge));
      if (copy_edge == nullptr) {
        continue;
      }
    }
  }

  void DeepCopySubGraphs(const FastGraphImpl &other_graph) const {
    owner_graph_->ClearAllSubGraph();
    for (auto iter = other_graph.sub_graphs_.begin(); iter != other_graph.sub_graphs_.end(); ++iter) {
      const ListElement<GraphT *> *origin_graph_listnode = *iter;
      if ((origin_graph_listnode == nullptr) || (origin_graph_listnode->data == nullptr)) {
        REPORT_INNER_ERR_MSG("E18888", "The sub graph is nullptr in src graph.");
        GELOGW("[DeepCopySubGraphs] The sub graph is nullptr in src graph, skip it.");
        continue;
      }

      auto name = origin_graph_listnode->data->GetName();
      std::shared_ptr<GraphT> copy_sub_graph = std::make_shared<GraphT>(name);
      if (copy_sub_graph == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "Failed to new sub graph.");
        GELOGW("[DeepCopySubGraphs] Failed to new sub graph [%s], skip it.", name.c_str());
        continue;
      }
      copy_sub_graph->CompleteCopy(*(origin_graph_listnode->data));
      owner_graph_->AddSubGraph(copy_sub_graph, name);
    }
  }

  void DeepCopyInputNodes(const FastGraphImpl &other_graph,
                          const std::unordered_map<const NodeT *, NodeT *> &origin_node_to_copy_node) {
    input_nodes_.clear();
    if (other_graph.input_nodes_.empty()) {
      return;
    }
    for (auto &item : other_graph.input_nodes_) {
      auto origin_node = item;
      auto old_iter = origin_node_to_copy_node.find(origin_node);
      if (old_iter != origin_node_to_copy_node.end()) {
        input_nodes_.push_back(old_iter->second);
      }
    }
  }

  void DeepCopyOutputNodes(const FastGraphImpl &other_graph,
                           const std::unordered_map<const NodeT *, NodeT *> &origin_node_to_copy_node) {
    output_nodes_.clear();
    if (other_graph.output_nodes_.empty()) {
      return;
    }
    for (auto &item : other_graph.output_nodes_) {
      auto origin_node = item.first;

      auto old_iter = origin_node_to_copy_node.find(origin_node);
      if (old_iter != origin_node_to_copy_node.end()) {
        output_nodes_.push_back(std::make_pair(old_iter->second, item.second));
      }
    }
  }

  void ClearNodes() {
    auto iter = nodes_.begin();
    while (iter != nodes_.end()) {
      auto quick_node = *iter;
      iter = nodes_.erase(iter);
      free_nodes_.push_back(quick_node, ListMode::kFreeMode);
    }
  }

  void ClearEdges() {
    auto iter = edges_.begin();
    while (iter != edges_.end()) {
      auto quick_edge = *iter;
      iter = edges_.erase(iter);
      free_edges_.push_back(quick_edge, ListMode::kFreeMode);
    }
  }

  void ClearGraphs() {
    auto iter = sub_graphs_.begin();
    while (iter != sub_graphs_.end()) {
      auto quick_graph = *iter;
      iter = sub_graphs_.erase(iter);
      free_sub_graphs_.push_back(quick_graph, ListMode::kFreeMode);
    }
  }

  NodeT *CreateOneNode(const OpDescPtr &op, int64_t id) {
    if (op == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "The OpDesc ptr should not be null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
      return nullptr;
    }
    op->SetId(id);

    ListElement<NodeT> *node_ptr = nullptr;
    if (free_nodes_.empty()) {
      node_ptr = new (std::nothrow) ListElement<NodeT>();
    } else {
      auto iter = free_nodes_.begin();
      node_ptr = *iter;
      free_nodes_.erase(iter);
    }

    if (node_ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "create node failed.");
      GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!");
      return nullptr;
    }

    auto fast_node = &FastGraphUtils::GetNode(node_ptr);
    auto ret = fast_node->Init(op);
    if (ret != GRAPH_SUCCESS) {
      return nullptr;
    }
    return fast_node;
  }

 private:
  friend GraphUtils;
  friend ExecuteGraph;
  friend class ExecuteGraphAdapter;
  friend class ExecuteGraphUtils;

  GraphT *owner_graph_ = nullptr;
  std::string name_;
  size_t graph_id_ = 0UL;
  // node
  mutable QuickList<NodeT> nodes_;
  QuickList<NodeT> free_nodes_;
  // edge
  mutable QuickList<Edge<NodeT>> edges_;
  QuickList<Edge<NodeT>> free_edges_;
  // io
  std::vector<NodeT *> input_nodes_;
  std::vector<std::pair<NodeT *, int32_t>> output_nodes_;
  // subgraph
  mutable QuickList<GraphT *> sub_graphs_;
  QuickList<GraphT *> free_sub_graphs_;

  GraphT *parent_graph_ = nullptr;
  NodeT *parent_node_ = nullptr;
  NodeT *graph_netoutput_ = nullptr;

  GraphExtendInfo extend_info_;
};

}  // namespace af
#endif  // FAST_GRAPH_NEW_GRAPH_IMPL_H
