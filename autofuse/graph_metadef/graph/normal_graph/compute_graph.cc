/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/compute_graph.h"

#include <deque>
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "common/ge_common/string_util.h"
#include "common/ge_common/ge_types.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/mem_utils.h"
#include "graph/utils/op_type_utils.h"
#include <ge_local_context.h>

namespace af {
namespace {
const size_t OUTPUT_PARAM_SIZE = 2UL;
const std::string kMemoryPriority = "MemoryPriority";

TopoSortingMode GetTopoSortingStrategy() {
  std::string topo_sorting_mode_str;
  if ((af::GetContext().GetOption(af::OPTION_TOPOSORTING_MODE, topo_sorting_mode_str) == GRAPH_SUCCESS) &&
      (!topo_sorting_mode_str.empty())) {
    const int32_t base = 10;
    auto topo_sorting_mode = static_cast<TopoSortingMode>(std::strtol(topo_sorting_mode_str.c_str(), nullptr, base));
    if ((topo_sorting_mode >= TopoSortingMode::kBFS) && (topo_sorting_mode < TopoSortingMode::kInvalid)) {
      GELOGD("topo_sorting_mode: %s", GetTopoSortingModeStr(topo_sorting_mode));
      return topo_sorting_mode;
    } else {
      GELOGW("OPTION_TOPOSORTING_MODE = %s is invalid, valid values are [BFS, DFS]", topo_sorting_mode_str.c_str());
    }
  }

  if (af::GetContext().GetTrainGraphFlag()) {
    GELOGD("train flag is 1, use BFS.");
    return TopoSortingMode::kBFS;
  }

  GELOGD("train flag is 0, use DFS.");
  return TopoSortingMode::kDFS;
}

struct NodeStatus {
  size_t size = 0U;
  WalkStatus status;
};

void InitNodeStatus(const ConstComputeGraphPtr &compute_graph, std::vector<NodeStatus> &nodes_info) {
  nodes_info.clear();
  nodes_info.resize(compute_graph->GetDirectNodesSize());
  int64_t index = 0;
  for (const auto &node : compute_graph->GetDirectNode()) {
    nodes_info[index].size = 0;
    nodes_info[index].status = WalkStatus::kNotWalked;
    node->GetOpDesc()->SetId(index);
    index++;
  }
}

int64_t GetNodeOutputRealSize(const NodePtr &node, std::vector<NodeStatus> &nodes_info) {
  int64_t total_size = 0;
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    return total_size;
  }
  NodeStatus &reverse_dfs_node_info = nodes_info[static_cast<size_t>(node->GetOpDesc()->GetId())];
  total_size = reverse_dfs_node_info.size;
  if (total_size != 0) {
    return total_size;
  }
  for (const auto &out_desc : node->GetOpDescBarePtr()->GetAllOutputsDescPtr()) {
    if (out_desc == nullptr) {
      continue;
    }
    int64_t output_size = 0;
    (void)af::TensorUtils::CalcTensorMemSize(out_desc->GetShape(), out_desc->GetFormat(), out_desc->GetDataType(),
                                             output_size);
    total_size += output_size;
  }
  if (total_size != 0) {
    reverse_dfs_node_info.size = total_size;
  }
  return total_size;
}

// 使用节点的输出空间占用大小来排序
struct NodeCmp {
  explicit NodeCmp(std::vector<NodeStatus> *nodes_info) : nodes_info_(nodes_info) {}
  bool operator()(const NodePtr &lhs, const NodePtr &rhs) const {
    const auto lhs_size = GetNodeOutputRealSize(lhs, *nodes_info_);
    const auto rhs_size = GetNodeOutputRealSize(rhs, *nodes_info_);
    if (lhs_size == rhs_size) {
      const int name_cmp = strcmp(lhs->GetNamePtr(), rhs->GetNamePtr());
      if (name_cmp == 0) {
        GELOGW("Duplicate node name [%s] found in topological sorting, comparing by pointer address.",
               lhs->GetNamePtr());
        return lhs.get() < rhs.get();
      }
      return name_cmp > 0;
    }
    return lhs_size > rhs_size;
  }
  std::vector<NodeStatus> *nodes_info_;
};

int64_t GetNodeInputCount(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    return 0;
  }
  return static_cast<int64_t>(node->GetInDataNodesSize());
}

struct NodeCmpV2 {
  NodeCmpV2(std::vector<NodeStatus> *nodes_info, const std::vector<int64_t> *branch_widths)
      : nodes_info_(nodes_info), branch_widths_(branch_widths) {}
  bool operator()(const NodePtr &lhs, const NodePtr &rhs) const {
    const auto lhs_size = GetNodeOutputRealSize(lhs, *nodes_info_);
    const auto rhs_size = GetNodeOutputRealSize(rhs, *nodes_info_);
    if (lhs_size == rhs_size) {
      const auto lhs_branch = (*branch_widths_)[static_cast<size_t>(lhs->GetOpDesc()->GetId())];
      const auto rhs_branch = (*branch_widths_)[static_cast<size_t>(rhs->GetOpDesc()->GetId())];
      if (lhs_branch == rhs_branch) {
        const int name_cmp = strcmp(lhs->GetNamePtr(), rhs->GetNamePtr());
        if (name_cmp == 0) {
          GELOGW("Duplicate node name [%s] found in topological sorting, comparing by pointer address.",
                 lhs->GetNamePtr());
          return lhs.get() < rhs.get();
        }
        return name_cmp > 0;
      }
      return lhs_branch < rhs_branch;
    }
    return lhs_size > rhs_size;
  }
  std::vector<NodeStatus> *nodes_info_;
  const std::vector<int64_t> *branch_widths_;
};

template <typename Compare>
graphStatus RDFSCoreSort(std::vector<NodePtr> &node_vec, std::vector<NodeStatus> &nodes_info,
                         const ConstComputeGraphPtr &compute_graph, const Compare &cmp) {
  std::set<NodePtr, Compare> zero_out_nodes{cmp};
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetOutNodesSize() == 0U) {
      zero_out_nodes.insert(node);
    }
  }
  std::vector<NodePtr> stack(zero_out_nodes.begin(), zero_out_nodes.end());
  while (!stack.empty()) {
    const auto current = stack.back();
    NodeStatus &info = nodes_info[current->GetOpDesc()->GetId()];
    if (info.status == WalkStatus::kNotWalked) {
      info.status = WalkStatus::kWalking;
      const auto in_all_nodes = current->GetInAllNodes();
      std::set<NodePtr, Compare> input_nodes{in_all_nodes.begin(), in_all_nodes.end(), cmp};
      stack.insert(stack.end(), input_nodes.cbegin(), input_nodes.cend());
      continue;
    }
    stack.pop_back();
    if (info.status == WalkStatus::kWalking) {
      info.status = WalkStatus::kWalked;
      node_vec.emplace_back(current);
    }
  }
  return GRAPH_SUCCESS;
}

struct NodeOutInfo {
  NodeOutInfo(const NodePtr &node, std::vector<NodeStatus> *nodes_info)
      : num_out_data_nodes(node->GetOutDataNodesSize()),
        output_size(GetNodeOutputRealSize(node, *nodes_info)),
        node_name(node->GetName()) {}

  bool operator<(const NodeOutInfo &rhs) const {
    if (num_out_data_nodes < rhs.num_out_data_nodes) {
      return true;
    }
    if (num_out_data_nodes > rhs.num_out_data_nodes) {
      return false;
    }
    if (output_size < rhs.output_size) {
      return true;
    }
    if (output_size > rhs.output_size) {
      return false;
    }
    return node_name < rhs.node_name;
  }

  int64_t num_out_data_nodes;
  int64_t output_size;
  std::string node_name;
};

bool IsMemoryPriority() {
  std::string memory_optimization_policy;
  (void)af::GetContext().GetOption(af::MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  return (memory_optimization_policy == kMemoryPriority);
}

bool InputIsLongLifeTimeNode(const NodePtr &node, const ConstComputeGraphPtr &graph) {
  bool match = false;
  for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    if (in_data_anchor == nullptr) {
      continue;
    }
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    const auto &peer_node = peer_out_anchor->GetOwnerNode();
    if (peer_node == nullptr) {
      continue;
    }
    const bool is_io_data =
        ((graph.get() == peer_node->GetOwnerComputeGraphBarePtr()) &&
         (OpTypeUtils::IsDataNode(peer_node->GetType()) || OpTypeUtils::IsConstPlaceHolderNode(peer_node->GetType())));
    std::string op_type;
    if ((!NodeUtils::GetConstOpType(peer_node, op_type)) && (!OpTypeUtils::IsVariableNode(peer_node->GetType())) &&
        (!is_io_data)) {
      return false;
    } else {
      match = true;
    }
    GELOGD("Node:%s peer:%s type :%s", node->GetName().c_str(), peer_node->GetName().c_str(),
           peer_node->GetType().c_str());
  }
  return match;
}

///  variable  const
///      \    /
///   first node
///       |
///   middle node
///       |
///   last node
///     /  |
/// node1  node2
graphStatus GetOutNodeIndex(std::vector<NodePtr> &nodes, size_t &index, size_t &out_count,
                            const ConstComputeGraphPtr &graph) {
  if (nodes.empty()) {
    return GRAPH_FAILED;
  }

  // first node's inputs muse be long life time
  if ((nodes.size() == 1U) && (!InputIsLongLifeTimeNode(nodes.front(), graph))) {
    return GRAPH_FAILED;
  }

  const auto &node = nodes.back();
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // middle node must be single input
  if ((nodes.size() != 1U) && (node->GetAllInDataAnchorsSize() != 1U)) {
    return GRAPH_FAILED;
  }

  int64_t min_index = 0;
  Node *delay_node = nullptr;
  for (const auto &out_node : node->GetOutAllNodes()) {
    out_count++;
    GE_CHECK_NOTNULL(out_node);
    auto out_node_desc = out_node->GetOpDescBarePtr();
    GE_CHECK_NOTNULL(out_node_desc);
    GELOGD("Node:%s id:%ld peer node:%s id:%ld", node->GetName().c_str(), op_desc->GetId(),
           out_node_desc->GetName().c_str(), out_node_desc->GetId());
    if ((min_index == 0) || (out_node_desc->GetId() < min_index)) {
      min_index = out_node_desc->GetId();
      delay_node = out_node.get();
    }
  }

  if (delay_node != nullptr) {
    index = static_cast<size_t>(min_index);
    if (index > (static_cast<size_t>(op_desc->GetId()) + 1U)) {
      GELOGD("Node:%s id:%ld delay to:%s id:%zu", node->GetName().c_str(), op_desc->GetId(),
             delay_node->GetName().c_str(), index);
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

void DelayTopoSort(std::vector<NodePtr> &nodes, const ConstComputeGraphPtr &graph) {
  // pair.first:  this node can be delay or not
  // pair.second: delayed nodes to this node
  std::vector<std::pair<bool, std::vector<NodePtr>>> delay_nodes;
  delay_nodes.resize(nodes.size());

  // set init index
  for (size_t i = 0U; i < delay_nodes.size(); ++i) {
    nodes[i]->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));
    delay_nodes[i].first = true;
    delay_nodes[i].second.emplace_back(nodes[i]);
  }

  // move delayed node to fit node
  size_t delay_node_count = 0U;
  for (size_t i = 0U; i < delay_nodes.size(); ++i) {
    size_t delay_to_index = 0U;
    size_t out_count = 0U;
    if (delay_nodes[i].first &&
        (GetOutNodeIndex(delay_nodes[i].second, delay_to_index, out_count, graph) == GRAPH_SUCCESS) &&
        (delay_to_index < delay_nodes.size()) && (delay_to_index > (i + 1U))) {
      delay_nodes[delay_to_index].second.insert(delay_nodes[delay_to_index].second.begin(),
                                                delay_nodes[i].second.begin(), delay_nodes[i].second.end());
      if (out_count > 1U) {
        // last node can not be delay
        delay_nodes[delay_to_index].first = false;
      }
      delay_nodes[i].second.clear();
      delay_node_count++;
    }
  }
  if (delay_node_count > 0U) {
    nodes.clear();
    for (size_t i = 0U; i < delay_nodes.size(); ++i) {
      if (!delay_nodes[i].second.empty()) {
        nodes.insert(nodes.end(), delay_nodes[i].second.begin(), delay_nodes[i].second.end());
      }
    }
    GELOGI("Delay %zu nodes for %s.", delay_node_count, graph->GetName().c_str());
  }
}

class TopoSortStack {
 public:
  explicit TopoSortStack(std::vector<NodeStatus> *nodes_info, const bool is_mem_priority = false,
                         const bool is_dfs = false, const bool is_reverse_dfs = false)
      : is_mem_priority_(is_mem_priority), is_dfs_(is_dfs), is_reverse_dfs_(is_reverse_dfs), nodes_info_(nodes_info) {}

  NodePtr Pop() {
    if (is_mem_priority_ && (!is_reverse_dfs_)) {
      const auto &it = mem_priority_stack_.cbegin();
      const NodePtr node = it->second;
      (void)mem_priority_stack_.erase(it);
      return node;
    }
    const NodePtr node = normal_stack_.back();
    normal_stack_.pop_back();
    return node;
  }

  void Push(const NodePtr &node) {
    if (is_mem_priority_ && (!is_reverse_dfs_)) {
      (void)mem_priority_stack_.emplace(NodeOutInfo(node, nodes_info_), node);
      return;
    }
    if (is_dfs_) {
      (void)normal_stack_.insert(normal_stack_.end(), node);
    } else {
      (void)normal_stack_.insert(normal_stack_.begin(), node);
    }
  }

  bool Empty() {
    if (is_mem_priority_ && (!is_reverse_dfs_)) {
      return mem_priority_stack_.empty();
    }
    return normal_stack_.empty();
  }

 private:
  bool is_mem_priority_;
  bool is_dfs_;
  bool is_reverse_dfs_;
  std::vector<NodeStatus> *nodes_info_;
  std::vector<NodePtr> normal_stack_;
  std::map<NodeOutInfo, NodePtr> mem_priority_stack_;
};

void AssembleFuseFailReason(const std::vector<NodePtr> &nodes, const std::unordered_set<std::string> attr_set,
                            const std::string attr_key, std::string &reason_not_support) {
  std::stringstream failed_reason;
  failed_reason << "Fusion is not supported because there are multiple " << attr_key << " ";
  for (const auto &ele : attr_set) {
    failed_reason << "[" << ele << "]";
  }
  failed_reason << " between nodes ";
  for (const auto &node : nodes) {
    failed_reason << "[" << node->GetName() << "]";
  }
  reason_not_support += failed_reason.str();
  GELOGI("%s.", reason_not_support.c_str());
}

std::unordered_set<std::string> GetAttrStringSet(const std::vector<NodePtr> &nodes, const std::string attr_key) {
  std::unordered_set<std::string> attr_set;
  for (const auto &node : nodes) {
    const auto &op_desc = node->GetOpDesc();
    const std::string *attr_val_ptr = AttrUtils::GetStr(op_desc, attr_key);
    if (attr_val_ptr != nullptr) {
      attr_set.emplace(*attr_val_ptr);
    }
  }
  return attr_set;
}

std::unordered_set<std::string> GetUserStreamLabels(const std::vector<NodePtr> &nodes) {
  return GetAttrStringSet(nodes, public_attr::USER_STREAM_LABEL);
}
/**
 * 临时方案： 通过开放public属性对用户开放流编排
 * 正式方案： 后续通过属性组方案实现，该方案将属性进行分组，改图的时候由属性组决定属性处理策略。
 *          预计25年H1落地
 * @param ori_nodes
 * @param fusion_ops
 * @return
 */
graphStatus InheritUserSteamLabelFromOriginNodes(const std::vector<NodePtr> &ori_nodes,
                                                 const std::vector<OpDescPtr> &fusion_ops) {
  const std::unordered_set<std::string> origin_stream_labels = GetUserStreamLabels(ori_nodes);
  GE_WARN_ASSERT(origin_stream_labels.size() < 2U,
                 "Inherit user stream label failed, because origin nodes have multiple user stream label.");
  if (origin_stream_labels.empty()) {
    return GRAPH_SUCCESS;
  }
  for (const auto &op_desc : fusion_ops) {
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, public_attr::USER_STREAM_LABEL, *origin_stream_labels.begin()));
  }
  return GRAPH_SUCCESS;
}

graphStatus InheritSkFromOriginNodes(const std::vector<NodePtr> &ori_nodes, const std::vector<OpDescPtr> &fusion_ops) {
  const std::unordered_set<std::string> scopes = GetAttrStringSet(ori_nodes, ATTR_NAME_SUPER_KERNEL_SCOPE);
  const std::unordered_set<std::string> kernel_options = GetAttrStringSet(ori_nodes, ATTR_NAME_SUPER_KERNEL_OPTIONS);
  if (scopes.empty() && kernel_options.empty()) {
    return GRAPH_SUCCESS;
  }
  for (const auto &op_desc : fusion_ops) {
    if (!scopes.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_SUPER_KERNEL_SCOPE, *scopes.begin()));
      GELOGD("set _super_kernel_scope %s for op %s", scopes.begin()->c_str(), op_desc->GetNamePtr());
    }
    if (!kernel_options.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_SUPER_KERNEL_OPTIONS, *kernel_options.begin()));
      GELOGD("set _super_kernel_options %s for op %s", kernel_options.begin()->c_str(), op_desc->GetNamePtr());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus InheritCoreNumFromOriginNodes(const std::vector<NodePtr> &ori_nodes,
                                          const std::vector<OpDescPtr> &fusion_ops) {
  std::unordered_set<std::string> origin_ai_core_nums;
  std::unordered_set<std::string> origin_vector_core_nums;

  for (const auto &node : ori_nodes) {
    const auto &op_desc = node->GetOpDesc();
    const std::string *user_ai_core_num_op_ptr = AttrUtils::GetStr(op_desc, public_attr::OP_AI_CORE_NUM);
    if (user_ai_core_num_op_ptr != nullptr) {
      origin_ai_core_nums.emplace(*user_ai_core_num_op_ptr);
    }
    const std::string *user_vector_core_num_op_ptr = AttrUtils::GetStr(op_desc, public_attr::OP_VECTOR_CORE_NUM);
    if (user_vector_core_num_op_ptr != nullptr) {
      origin_vector_core_nums.emplace(*user_vector_core_num_op_ptr);
    }
  }

  // 如果所有原始节点都没有设置核数，则不需要继承
  if (origin_ai_core_nums.empty() && origin_vector_core_nums.empty()) {
    GELOGI("No need to inherit core num, because origin nodes have no core num.");
    return GRAPH_SUCCESS;
  }

  GELOGI("Begin to set core num for fusion ops.");
  for (const auto &op_desc : fusion_ops) {
    if (!origin_ai_core_nums.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, public_attr::OP_AI_CORE_NUM, *origin_ai_core_nums.begin()));
      GELOGD("set ai core num %s for op %s", origin_ai_core_nums.begin()->c_str(), op_desc->GetName().c_str());
    }
    if (!origin_vector_core_nums.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, public_attr::OP_VECTOR_CORE_NUM, *origin_vector_core_nums.begin()));
      GELOGD("set vector core num %s for op %s", origin_vector_core_nums.begin()->c_str(), op_desc->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

ComputeGraphImpl::ComputeGraphImpl(const std::string &name)
    : name_(name), nodes_(), input_nodes_(), sub_graph_(), is_valid_flag_(false), need_iteration_(false) {}

std::string ComputeGraphImpl::GetName() const {
  return name_;
}

void ComputeGraphImpl::SetName(const std::string &name) {
  name_ = name;
}

size_t ComputeGraphImpl::GetAllNodesSize(const ConstComputeGraphPtr &compute_graph) const {
  return GetAllNodes(compute_graph).size();
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetAllNodes(const ConstComputeGraphPtr &compute_graph) const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodes(subgraphs, compute_graph);
}

void ComputeGraphImpl::GetAllNodesFromOpdesc(const OpDesc &op_desc, const GraphFilter &graph_filter,
                                             std::deque<NodePtr> &candidates, const NodePtr node) const {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  auto name_iter = subgraph_names.rbegin();
  while (name_iter != subgraph_names.rend()) {
    const auto subgraph = GetSubgraph(*name_iter);
    if (subgraph != nullptr) {
      if ((graph_filter == nullptr) || graph_filter(*node, name_iter->c_str(), subgraph)) {
        auto subgraph_nodes = subgraph->GetDirectNode();
        (void)(candidates.insert(candidates.begin(), subgraph_nodes.begin(), subgraph_nodes.end()));
      }
    }
    ++name_iter;
  }
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetAllNodes(const NodeFilter &node_filter,
                                                                const GraphFilter &graph_filter,
                                                                const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;

  (void)candidates.insert(candidates.begin(), nodes_.begin(), nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    candidates.pop_front();

    if ((node_filter == nullptr) || node_filter(*node)) {
      all_nodes.emplace_back(node);
    }

    const auto op_desc = node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      GetAllNodesFromOpdesc(*op_desc, graph_filter, candidates, node);
    }
  }

  return Vistor<NodePtr>(compute_graph, all_nodes);
}

void inline ComputeGraphImpl::GetAllNodesFromOpdesc(std::vector<ComputeGraphPtr> &subgraphs, const OpDesc &op_desc,
                                                    std::deque<NodePtr> &candidates) const {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  auto name_iter = subgraph_names.rbegin();
  while (name_iter != subgraph_names.rend()) {
    auto subgraph = GetSubgraph(*name_iter);
    if (subgraph != nullptr) {
      subgraphs.emplace_back(subgraph);
      auto subgraph_nodes = subgraph->GetDirectNode();
      (void)candidates.insert(candidates.begin(), subgraph_nodes.begin(), subgraph_nodes.end());
    }
    ++name_iter;
  }
}

void inline ComputeGraphImpl::GetAllNodesPtrFromOpdesc(std::vector<ComputeGraphPtr> &subgraphs, const OpDesc &op_desc,
                                                       std::deque<Node *> &candidates) const {
  const auto &subgraph_names = op_desc.GetSubgraphInstanceNames();
  auto name_iter = subgraph_names.rbegin();
  while (name_iter != subgraph_names.rend()) {
    auto subgraph = GetSubgraph(*name_iter);
    if (subgraph != nullptr) {
      subgraphs.emplace_back(subgraph);
      auto subgraph_nodes = subgraph->GetDirectNodePtr();
      (void)candidates.insert(candidates.begin(), subgraph_nodes.begin(), subgraph_nodes.end());
    }
    ++name_iter;
  }
}

std::vector<Node *> ComputeGraphImpl::AllGraphNodesPtr(std::vector<ComputeGraphPtr> &subgraphs) const {
  std::vector<Node *> all_nodes;
  std::deque<Node *> candidates;

  for (const auto &node : nodes_) {
    (void)candidates.emplace_back(node.get());
  }
  while (!candidates.empty()) {
    Node *node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    const auto op_desc = node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      GetAllNodesPtrFromOpdesc(subgraphs, *op_desc, candidates);
    }
  }

  return all_nodes;
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs,
                                                                  const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;

  (void)candidates.insert(candidates.begin(), nodes_.begin(), nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    const auto op_desc = node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      GetAllNodesFromOpdesc(subgraphs, *op_desc, candidates);
    }
  }

  return Vistor<NodePtr>(compute_graph, all_nodes);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetNodes(const bool is_unknown_shape,
                                                             const ConstComputeGraphPtr &compute_graph) const {
  if (is_unknown_shape) {
    return GetDirectNode(compute_graph);
  } else {
    return GetAllNodes(compute_graph);
  }
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetNodes(const bool is_unknown_shape, const NodeFilter &node_filter,
                                                             const GraphFilter &graph_filter,
                                                             const ConstComputeGraphPtr &compute_graph) const {
  return is_unknown_shape ? GetDirectNode(compute_graph) : GetAllNodes(node_filter, graph_filter, compute_graph);
}

size_t ComputeGraphImpl::GetDirectNodesSize() const {
  return direct_nodes_size_;
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetDirectNode(const ConstComputeGraphPtr &compute_graph) const {
  return Vistor<NodePtr>(compute_graph, nodes_);
}

std::vector<Node *> ComputeGraphImpl::GetDirectNodePtr() const {
  std::vector<Node *> direct_nodes;
  direct_nodes.reserve(nodes_.size());
  for (const auto &node : nodes_) {
    (void)direct_nodes.emplace_back(node.get());
  }
  return direct_nodes;
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetInputNodes(const ConstComputeGraphPtr &compute_graph) const {
  return Vistor<NodePtr>(compute_graph, input_nodes_);
}

ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetOutputNodes(const ConstComputeGraphPtr &compute_graph) const {
  std::vector<NodePtr> result;
  auto iter = output_nodes_info_.begin();
  while (iter != output_nodes_info_.end()) {
    result.push_back(iter->first);
    ++iter;
  }
  return Vistor<NodePtr>(compute_graph, result);
}

NodePtr ComputeGraphImpl::FindNode(const std::string &name) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (NodeUtils::IsNameEqual(node, name.c_str())) {
      return node;
    }
  }
  return nullptr;
}

NodePtr ComputeGraphImpl::FindFirstNodeMatchType(const std::string &type) const {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    if (NodeUtils::IsTypeEqual(node, type.c_str())) {
      return node;
    }
  }
  return nullptr;
}

bool ComputeGraphImpl::GraphAttrsAreEqual(const ComputeGraphImpl &r_graph) const {
  // 整改前实现中，只比较了属性名字，没有比较属性内容，暂时维持这个玩法
  return attrs_.GetAllAttrNames() == r_graph.attrs_.GetAllAttrNames();
}

/// Since there may be different input nodes
/// chosen by user in the same graph, special judgment is needed
bool ComputeGraphImpl::VectorInputNodePtrIsEqual(const std::vector<NodePtr> &left_nodes,
                                                 const std::vector<NodePtr> &right_nodes) const {
  const auto left_nodes_size = left_nodes.size();
  const auto right_nodes_size = right_nodes.size();
  if (left_nodes_size != right_nodes_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Check failed with graph input_nodes_: "
                         "left inputNodes size %zu is different from right inputNodes size %zu.",
                         left_nodes_size, right_nodes_size);
    GELOGE(GRAPH_FAILED,
           "[Check][Param] failed with graph input_nodes_: "
           "left inputNodes size %zu is different from right inputNodes size %zu.",
           left_nodes_size, right_nodes_size);
    return false;
  }
  for (size_t j = 0UL; j < left_nodes_size; j++) {
    if ((left_nodes.at(j) == nullptr) || (right_nodes.at(j) == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "left_nodes.at(%zu) or right_nodes.at(%zu) is nullptr", j, j);
      GELOGE(GRAPH_FAILED, "[Check][Param] left_nodes.at(%zu) or right_nodes.at(%zu) is nullptr", j, j);
      return false;
    }
    const auto &left_input_name = left_nodes.at(j)->GetName();
    const auto &right_input_name = right_nodes.at(j)->GetName();
    if (left_input_name != right_input_name) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Check failed with graph input_nodes_: "
                           "left inputNode name %s is different from right inputNode name %s at inputNodes index %zu.",
                           left_input_name.c_str(), right_input_name.c_str(), j);
      GELOGE(GRAPH_FAILED,
             "[Check][Param] failed with graph input_nodes_: "
             "left inputNode name %s is different from right inputNode name %s at inputNodes index %zu.",
             left_input_name.c_str(), right_input_name.c_str(), j);
      return false;
    }
  }
  return true;
}

bool ComputeGraphImpl::GraphMembersAreEqual(const ComputeGraphImpl &r_graph) const {
  return (IsEqual(this->sub_graph_.size(), r_graph.sub_graph_.size(), "graph.subgraphs_.size()") &&
          IsEqual(this->GetDirectNodesSize(), r_graph.GetDirectNodesSize(), "graph.nodes_.size()") &&
          VectorInputNodePtrIsEqual(this->input_nodes_, r_graph.input_nodes_) &&
          IsEqual(this->name_, r_graph.name_, "graph.name_") &&
          IsEqual(this->is_valid_flag_, r_graph.is_valid_flag_, "graph.is_valid_flag_") &&
          IsEqual(this->need_iteration_, r_graph.need_iteration_, "graph.need_iteration_") &&
          IsEqual(this->params_share_map_, r_graph.params_share_map_, "graph.params_share_map_") &&
          IsEqual(this->out_nodes_map_, r_graph.out_nodes_map_, "graph.out_nodes_map_") &&
          IsEqual(this->inputs_order_, r_graph.inputs_order_, "graph.inputs_order_") &&
          IsEqual(this->output_size_, r_graph.output_size_, "graph.output_size_") &&
          IsEqual(this->input_size_, r_graph.input_size_, "graph.input_size_") &&
          IsEqual(this->output_nodes_info_, r_graph.output_nodes_info_, "graph.output_nodes_info_"));
}

bool ComputeGraphImpl::operator==(const ComputeGraphImpl &r_graph) const {
  // Firstly: Graph's members equal
  if ((!GraphMembersAreEqual(r_graph)) || (!GraphAttrsAreEqual(r_graph))) {
    return false;
  }

  // Secondly: Node equal means the link relationship between node and node itself equal
  for (const auto &left_node : nodes_) {
    if (left_node == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "left_node is nullptr, graph:%s", this->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] left_node is nullptr");
      return false;
    }
    const auto &node_name = left_node->GetName();
    // After TopologicalSorting, node order can change, so find node by name
    const auto &right_node = r_graph.FindNode(node_name);
    GE_IF_BOOL_EXEC(right_node == nullptr, REPORT_INNER_ERR_MSG("E18888", "left_node:%s was not found in r_graph:%s",
                                                                node_name.c_str(), r_graph.GetName().c_str());
                    GELOGE(GRAPH_FAILED, "[Check][Param] right_node is NULL!!!"); return false);
    if (!((*right_node) == (*left_node))) {
      REPORT_INNER_ERR_MSG("E18888", "Compare graph failed, node:%s not equal.", node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Compare][Graph] failed, node:%s not equal.", node_name.c_str());
      return false;
    }
  }

  // Thirdly: Recursively determine whether the sub graphs are equal
  for (size_t i = 0UL; i < this->sub_graph_.size(); i++) {
    if (!((*((this->sub_graph_)[i])) == (*((r_graph.sub_graph_)[i])))) {
      return false;
    }
  }
  return true;
}

NodePtr ComputeGraphImpl::AddNodeFront(const NodePtr node) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr or op desc should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  if ((GetDirectNodesSize() > 0UL) && ((*(nodes_.begin()))->GetType() == DATA)) {
    InsertToNodeList(next(nodes_.begin()), node);
  } else {
    InsertToNodeList(nodes_.begin(), node);
  }
  AddInputDataNode(node);
  return node;
}

NodePtr ComputeGraphImpl::AddNodeFront(const OpDescPtr &op, const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  const NodePtr node_ptr = std::shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS,
                  REPORT_INNER_ERR_MSG("E18888", "node %s init failed.", op->GetName().c_str());
                  GELOGE(GRAPH_FAILED, "node %s init failed.", op->GetName().c_str()); return nullptr);
  return AddNodeFront(node_ptr);
}

af::NodePtr ComputeGraphImpl::CreateNodeFromOpDesc(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph,
                                                   const int64_t topo_id) {
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetId(topo_id);
  const auto node = std::shared_ptr<Node>(new (std::nothrow) Node(op_desc, compute_graph));
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_GRAPH_SUCCESS(node->Init());
  node->SetHostNode(is_valid_flag_);
  AddInputDataNode(node);
  return node;
}

std::vector<NodePtr> ComputeGraphImpl::InsertNodes(const NodePtr &node, const std::vector<OpDescPtr> &insert_ops,
                                                   const ComputeGraphPtr &compute_graph) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto topo_id = node->GetOpDesc()->GetId();
  auto iter = std::find(nodes_.begin(), nodes_.end(), node);
  GE_ASSERT_TRUE(iter != nodes_.end(), "Cannot find before node: %s in graph: %s", node->GetName().c_str(),
                 compute_graph->GetName().c_str());
  iter = next(iter);
  std::vector<af::NodePtr> insert_nodes;
  for (const auto &op_desc : insert_ops) {
    auto insert_node = CreateNodeFromOpDesc(op_desc, compute_graph, topo_id);
    GE_ASSERT_NOTNULL(insert_node);
    InsertToNodeList(iter, insert_node);
    insert_nodes.emplace_back(insert_node);
  }
  return insert_nodes;
}

NodePtr ComputeGraphImpl::InsertNodeBefore(const NodePtr &node, const OpDescPtr &insert_op,
                                           const ComputeGraphPtr &compute_graph) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto topo_id = node->GetOpDesc()->GetId();
  auto iter = std::find(nodes_.begin(), nodes_.end(), node);
  GE_ASSERT_TRUE(iter != nodes_.end(), "Cannot find node: %s in graph: %s", node->GetNamePtr(),
                 compute_graph->GetName().c_str());
  auto insert_node = CreateNodeFromOpDesc(insert_op, compute_graph, topo_id);
  GE_ASSERT_NOTNULL(insert_node);
  InsertToNodeList(iter, insert_node);
  return insert_node;
}

NodePtr ComputeGraphImpl::InsertNode(const NodePtr &node, const OpDescPtr &insert_op,
                                     const ComputeGraphPtr &compute_graph) {
  std::vector<OpDescPtr> ops_vec = {insert_op};
  const auto node_vec = InsertNodes(node, ops_vec, compute_graph);
  GE_ASSERT_TRUE(!node_vec.empty());
  return node_vec.front();
}

bool ComputeGraphImpl::IsSupportFuse(const std::vector<NodePtr> &nodes, std::string &reason_not_support) {
  const std::unordered_set<std::string> origin_stream_labels = GetUserStreamLabels(nodes);
  if (origin_stream_labels.size() > 1U) {
    AssembleFuseFailReason(nodes, origin_stream_labels, public_attr::USER_STREAM_LABEL, reason_not_support);
    return false;
  }

  const std::unordered_set<std::string> sk_scopes = GetAttrStringSet(nodes, ATTR_NAME_SUPER_KERNEL_SCOPE);
  if (sk_scopes.size() > 1U) {
    AssembleFuseFailReason(nodes, sk_scopes, ATTR_NAME_SUPER_KERNEL_SCOPE, reason_not_support);
    return false;
  }

  const std::unordered_set<std::string> sk_options = GetAttrStringSet(nodes, ATTR_NAME_SUPER_KERNEL_OPTIONS);
  if (sk_options.size() > 1U) {
    AssembleFuseFailReason(nodes, sk_options, ATTR_NAME_SUPER_KERNEL_OPTIONS, reason_not_support);
    return false;
  }
  const std::unordered_set<std::string> aicore_num_options = GetAttrStringSet(nodes, public_attr::OP_AI_CORE_NUM);
  if (aicore_num_options.size() > 1U) {
    AssembleFuseFailReason(nodes, aicore_num_options, public_attr::OP_AI_CORE_NUM, reason_not_support);
    return false;
  }

  const std::unordered_set<std::string> vectorcore_num_options =
      GetAttrStringSet(nodes, public_attr::OP_VECTOR_CORE_NUM);
  if (vectorcore_num_options.size() > 1U) {
    AssembleFuseFailReason(nodes, vectorcore_num_options, public_attr::OP_VECTOR_CORE_NUM, reason_not_support);
    return false;
  }

  return true;
}

std::vector<NodePtr> ComputeGraphImpl::FuseNodeKeepTopo(const std::vector<NodePtr> &ori_nodes,
                                                        const std::vector<OpDescPtr> &fusion_ops,
                                                        const ComputeGraphPtr &compute_graph) {
  std::string failed_reason;
  GE_WARN_ASSERT(IsSupportFuse(ori_nodes, failed_reason), failed_reason.c_str());
  if (InheritUserSteamLabelFromOriginNodes(ori_nodes, fusion_ops) != GRAPH_SUCCESS) {
    GELOGD("Abandoned to fuse nodes because inherit user stream label failed.");
    return {};
  }
  if (InheritSkFromOriginNodes(ori_nodes, fusion_ops) != GRAPH_SUCCESS) {
    GELOGI("Abandoned to fuse nodes because inherit sk attrs failed.");
    return {};
  }
  if (InheritCoreNumFromOriginNodes(ori_nodes, fusion_ops) != GRAPH_SUCCESS) {
    return {};
  }

  auto min_id_node = NodeUtils::GetNodeWithMinimalId(ori_nodes);
  GE_ASSERT_NOTNULL(min_id_node);
  return InsertNodes(min_id_node, fusion_ops, compute_graph);
}

NodePtr ComputeGraphImpl::AddNode(const NodePtr node) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "the node ptr or op desc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or op desc ptr should not be null.");
    return nullptr;
  }
  node->SetHostNode(is_valid_flag_);
  node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  PushBackToNodeList(node);
  AddInputDataNode(node);
  return node;
}

NodePtr ComputeGraphImpl::AddNode(const OpDescPtr op, const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(static_cast<int64_t>(GetDirectNodesSize()));
  const NodePtr node_ptr = std::shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node_ptr == nullptr, REPORT_INNER_ERR_MSG("E18888", "create node failed.");
                  GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node_ptr->Init() != GRAPH_SUCCESS,
                  REPORT_INNER_ERR_MSG("E18888", "node:%s init failed.", op->GetName().c_str());
                  GELOGE(GRAPH_FAILED, "[Init][Node] node %s init failed.", op->GetName().c_str()); return nullptr);
  return AddNode(node_ptr);
}

NodePtr ComputeGraphImpl::AddNode(const OpDescPtr op, const int64_t id, const ComputeGraphPtr &compute_graph) {
  if (op == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  op->SetId(id);
  const NodePtr node = std::shared_ptr<Node>(new (std::nothrow) Node(op, compute_graph));
  GE_IF_BOOL_EXEC(node == nullptr, REPORT_INNER_ERR_MSG("E18888", "create node failed.");
                  GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!!!"); return nullptr);
  GE_IF_BOOL_EXEC(node->Init() != GRAPH_SUCCESS, REPORT_INNER_ERR_MSG("E18888", "node init failed.");
                  GELOGE(GRAPH_FAILED, "[Init][Node] node init failed."); return nullptr);
  node->SetHostNode(is_valid_flag_);
  PushBackToNodeList(node);
  AddInputDataNode(node);
  return node;
}

void ComputeGraphImpl::AddInputDataNode(const NodePtr &node) {
  if (OpTypeUtils::IsDataNode(node->GetType())) {
    if (std::find(input_nodes_.begin(), input_nodes_.end(), node) == input_nodes_.end()) {
      input_nodes_.push_back(node);
    }
  }
}

NodePtr ComputeGraphImpl::AddInputNode(const NodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return nullptr;
  }
  if (std::find(input_nodes_.begin(), input_nodes_.end(), node) == input_nodes_.end()) {
    input_nodes_.push_back(node);
  }
  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "[Add][Node] failed");
  }
  return node;
}

NodePtr ComputeGraphImpl::AddOutputNode(const NodePtr node) {
  return AddOutputNodeByIndex(node, 0);
}

NodePtr ComputeGraphImpl::AddOutputNodeByIndex(const NodePtr node, const int32_t index) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr or opdesc should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr or opdesc should not be null.");
    return nullptr;
  }

  bool already_have = false;
  NodePtr result = node;
  // [output_nodes_info_ : should not be null]
  for (const auto &item : output_nodes_info_) {
    if ((item.first->GetName() == node->GetName()) && (item.second == index)) {
      already_have = true;
      result = item.first;
      break;
    }
  }

  if (!already_have) {
    output_nodes_info_.emplace_back(std::make_pair(node, index));
    GELOGI("Push back node name:%s, index:%d, into output_nodes_info_.", node->GetName().c_str(), index);
  }

  if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
    GE_CHK_BOOL_EXEC(AddNode(node) != nullptr, return nullptr, "[Add][Node] failed");
  }
  return result;
}

graphStatus ComputeGraphImpl::RemoveConstInput(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if ((out_anchor == nullptr) || (out_anchor->GetOwnerNodeBarePtr() == nullptr)) {
      continue;
    }
    if ((out_anchor->GetOwnerNodeBarePtr()->GetType() == CONSTANT) ||
        (out_anchor->GetOwnerNodeBarePtr()->GetType() == CONSTANTOP)) {
      GE_CHK_BOOL_RET_STATUS(GraphUtils::RemoveEdge(out_anchor, in_anchor) == GRAPH_SUCCESS, GRAPH_FAILED,
                             "[Remove][Edge] from const op %s failed.", out_anchor->GetOwnerNode()->GetName().c_str());
      if (out_anchor->GetOwnerNode()->GetOutNodes().empty()) {
        GELOGI("Remove const op %s.", out_anchor->GetOwnerNode()->GetName().c_str());
        const auto iter = find(nodes_.begin(), nodes_.end(), out_anchor->GetOwnerNode());
        if (iter != nodes_.end()) {
          EraseFromNodeList(iter);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::RemoveNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // delete const op for this node
  (void)RemoveConstInput(node);

  // if the node save as input node, delete it
  (void)RemoveInputNode(node);

  // if the node save as output node, delete it
  (void)RemoveOutputNode(node);

  if (IsolateNode(node) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Isolate][Node] failed, node name: %s, graph:%s.", node->GetName().c_str(), name_.c_str());
    return GRAPH_FAILED;
  }

  const auto iter = find(nodes_.begin(), nodes_.end(), node);
  if (iter != nodes_.end()) {
    EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraphImpl::RemoveInputNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  const auto iter = find(input_nodes_.begin(), input_nodes_.end(), node);
  if (iter != input_nodes_.end()) {
    (void)input_nodes_.erase(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// Used in sub_graph scenes
graphStatus ComputeGraphImpl::RemoveOutputNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The node ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  auto target_iter = targets_.find(node);
  if (target_iter != targets_.end()) {
    targets_.erase(target_iter);
  }

  auto iter = output_nodes_info_.begin();
  bool find_node = false;
  // [output_nodes_info_ : should not be null]
  while (iter != output_nodes_info_.end()) {
    if (node->GetName() == iter->first->GetName()) {
      iter = output_nodes_info_.erase(iter);
      find_node = true;
    } else {
      ++iter;
    }
  }
  GE_IF_BOOL_EXEC(!find_node, return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

std::shared_ptr<ComputeGraph> ComputeGraphImpl::AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  if (sub_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The graph ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The graph ptr should not be null.");
    return nullptr;
  }
  sub_graph_.push_back(sub_graph);
  names_to_subgraph_[sub_graph->GetName()] = sub_graph;
  return sub_graph;
}

graphStatus ComputeGraphImpl::RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  if (sub_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The graph ptr should not be null, graph:%s.", name_.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The graph ptr should not be null.");
    return GRAPH_FAILED;
  }

  (void)names_to_subgraph_.erase(sub_graph->GetName());
  const auto iter = find(sub_graph_.begin(), sub_graph_.end(), sub_graph);
  if (iter != sub_graph_.end()) {
    (void)sub_graph_.erase(iter);
  } else {
    GELOGW("[Remove][Subgraph] find sub_graph failed");
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Try to add a null subgraph, name %s", name.c_str());
    GE_LOGE("[Check][Param] Try to add a null subgraph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto parent_graph = subgraph->GetParentGraph();
  if (parent_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Try to add subgraph without parent graph, name %s", name.c_str());
    GE_LOGE("[Get][Graph] Try to add subgraph without parent graph, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto parent_node = subgraph->GetParentNode();
  if (parent_node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Try to add a subgraph without parent node, name %s", name.c_str());
    GE_LOGE("[Get][Node] Try to add a subgraph without parent node, name %s", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (parent_node->GetOwnerComputeGraph() != parent_graph) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Try to add a subgraph which parent node's graph is not equal to "
                         "the subgraph's parent graph, subgraph name %s, parent node name %s",
                         subgraph->GetName().c_str(), parent_graph->GetName().c_str());
    GE_LOGE(
        "[Check][Param] Try to add a subgraph which parent node's graph is not equal to "
        "the subgraph's parent graph, subgraph name %s, parent node name %s",
        subgraph->GetName().c_str(), parent_graph->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  if (!this->parent_graph_.expired()) {
    GELOGW("[Add][Subgraph] The subgraphs should only be added to the root graph");
  }
  if (name != subgraph->GetName()) {
    GELOGW("[Add][Subgraph] The subgraph name %s is different from input %s", subgraph->GetName().c_str(),
           name.c_str());
  }
  if (names_to_subgraph_.find(name) != names_to_subgraph_.end()) {
    REPORT_INNER_ERR_MSG("E18888", "The subgraph %s existed", name.c_str());
    GE_LOGE("[Check][Param] The subgraph %s existed", name.c_str());
    return GRAPH_PARAM_INVALID;
  }
  sub_graph_.push_back(subgraph);
  names_to_subgraph_[name] = subgraph;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraphImpl::RemoveSubgraph(const std::string &name) {
  const std::map<std::string, af::ComputeGraphPtr>::const_iterator iter = names_to_subgraph_.find(name);
  if (iter == names_to_subgraph_.cend()) {
    return;
  }
  auto vec_iter = sub_graph_.begin();
  while (vec_iter != sub_graph_.end()) {
    if ((*vec_iter) == iter->second) {
      (void)sub_graph_.erase(vec_iter);
      break;
    }
    ++vec_iter;
  }
  (void)names_to_subgraph_.erase(iter);
}

std::shared_ptr<ComputeGraph> ComputeGraphImpl::GetSubgraph(const std::string &name) const {
  const std::shared_ptr<ComputeGraph> parent = parent_graph_.lock();
  if (parent == nullptr) {
    const auto iter = names_to_subgraph_.find(name);
    return (iter == names_to_subgraph_.end()) ? nullptr : iter->second;
  } else {
    return parent->GetSubgraph(name);
  }
}

std::vector<std::shared_ptr<ComputeGraph>> ComputeGraphImpl::GetAllSubgraphs() const {
  return sub_graph_;
}

void ComputeGraphImpl::SetAllSubgraphs(const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) {
  sub_graph_ = subgraphs;
}

shared_ptr<ComputeGraph> ComputeGraphImpl::GetParentGraph() const {
  return parent_graph_.lock();
}

const ComputeGraph *ComputeGraphImpl::GetParentGraphBarePtr() const {
  return parent_graph_bare_ptr_;
}

void ComputeGraphImpl::SetParentGraph(const std::shared_ptr<ComputeGraph> &parent) {
  parent_graph_ = parent;
  parent_graph_bare_ptr_ = parent_graph_.lock().get();
}

shared_ptr<Node> ComputeGraphImpl::GetParentNode() const {
  return parent_node_.lock();
}

const Node *ComputeGraphImpl::GetParentNodeBarePtr() const {
  return parent_node_bare_ptr_;
}

void ComputeGraphImpl::SetParentNode(const std::shared_ptr<Node> &parent) {
  parent_node_ = parent;
  parent_node_bare_ptr_ = parent_node_.lock().get();
}

shared_ptr<Node> ComputeGraphImpl::GetOrUpdateNetOutputNode() {
  auto graph_netoutput = graph_netoutput_.lock();
  if (graph_netoutput == nullptr || graph_netoutput->GetType() != NETOUTPUT) {
    graph_netoutput = FindFirstNodeMatchType(NETOUTPUT);
    SetNetOutputNode(graph_netoutput);
  }
  if (graph_netoutput == nullptr) {
    GELOGW("Graph %s has no netoutput node", GetName().c_str());
  }
  return graph_netoutput;
}

void ComputeGraphImpl::SetNetOutputNode(const std::shared_ptr<Node> &netoutput_node) {
  graph_netoutput_ = netoutput_node;
}

/// @brief Update input-mapping
/// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
/// @return graphStatus
graphStatus ComputeGraphImpl::UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &input : nodes_) {
    if (input->GetType() == DATA) {
      uint32_t cur_index = 0U;
      if (!af::AttrUtils::GetInt(input->GetOpDescBarePtr(), ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
        continue;
      }
      const auto iter = input_mapping.find(cur_index);
      if (iter == input_mapping.end()) {
        continue;
      }
      if (!af::AttrUtils::SetInt(input->GetOpDescBarePtr(), ATTR_NAME_PARENT_NODE_INDEX,
                                 static_cast<int64_t>(iter->second))) {
        REPORT_INNER_ERR_MSG("E18888", "set attr ATTR_NAME_PARENT_NODE_INDEX failed, op:%s.",
                             input->GetOpDescBarePtr()->GetName().c_str());
        GE_LOGE("[Call][SetInt] UpdateInputMapping failed: set attr ATTR_NAME_PARENT_NODE_INDEX failed, op:%s.",
                input->GetOpDescBarePtr()->GetName().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

/// @brief Update output-mapping
/// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
/// @return graphStatus
graphStatus ComputeGraphImpl::UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) const {
  const NodePtr net_output = FindFirstNodeMatchType(NETOUTPUT);
  if (net_output == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "UpdateOutputMapping failed: node type %s does not exist in graph.", NETOUTPUT);
    GE_LOGE("[Get][NodeType] UpdateOutputMapping failed: node type %s does not exist in graph.", NETOUTPUT);
    return GRAPH_FAILED;
  }
  const auto op_desc = net_output->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "net output's op desc ptr should not be null.");
    GE_LOGE("[Get][OpDesc] UpdateOutputMapping failed: op_desc is NULL.");
    return GRAPH_FAILED;
  }

  const size_t num = op_desc->GetAllInputsSize();
  for (size_t i = 0UL; i < num; i++) {
    GeTensorDesc tensor = op_desc->GetInputDesc(static_cast<uint32_t>(i));
    uint32_t cur_index = 0U;
    if (!af::AttrUtils::GetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, cur_index)) {
      continue;
    }
    const auto iter = output_mapping.find(cur_index);
    if (iter == output_mapping.end()) {
      continue;
    }
    if (!af::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(iter->second))) {
      REPORT_INNER_ERR_MSG("E18888", "op %s set %zu input tensor attr ATTR_NAME_PARENT_NODE_INDEX failed.",
                           op_desc->GetName().c_str(), i);
      GE_LOGE("[Set][Int] op %s set %zu input tensor attr ATTR_NAME_PARENT_NODE_INDEX failed.",
              op_desc->GetName().c_str(), i);
      return GRAPH_FAILED;
    }
    if (op_desc->UpdateInputDesc(static_cast<uint32_t>(i), tensor) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "op %s update %zu input_tensor failed.", op_desc->GetName().c_str(), i);
      GE_LOGE("[Update][InputDesc] UpdateOutputMapping failed: update %zu input_tensor failed.", i);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::ReorderEventNodes(const ConstComputeGraphPtr &compute_graph) {
  std::list<NodePtr> &node_list = nodes_;
  for (const auto &node : GetDirectNode(compute_graph)) {
    if ((strcmp(node->GetTypePtr(), RECV) == 0) || (strcmp(node->GetTypePtr(), RECV_NOTIFY) == 0)) {
      const auto iter = find(node_list.cbegin(), node_list.cend(), node);
      if (iter != node_list.cend()) {
        (void)node_list.erase(iter);
      }

      const auto dst_iter = find(node_list.cbegin(), node_list.cend(), node->GetOutControlNodes().at(0UL));
      (void)node_list.insert(dst_iter, node);
    }
    if ((strcmp(node->GetTypePtr(), SEND) == 0) || (strcmp(node->GetTypePtr(), SEND_NOTIFY) == 0)) {
      const auto iter = find(node_list.cbegin(), node_list.cend(), node);
      if (iter != node_list.cend()) {
        (void)node_list.erase(iter);
      }

      auto src_iter = find(node_list.cbegin(), node_list.cend(), node->GetInControlNodes().at(0UL));
      (void)node_list.insert(++src_iter, node);
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::InsertGraphEvents(const ConstComputeGraphPtr &compute_graph) {
  auto status = ReorderEventNodes(compute_graph);
  if (status != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Graph [%s] record event nodes failed, status:%u", name_.c_str(), status);
    GELOGE(status, "[Reorder][EventNodes] failed for Graph:%s, status:%u", name_.c_str(), status);
    return status;
  }

  // Partition subgraph
  for (const auto &graph : sub_graph_) {
    status = graph->ReorderEventNodes();
    if (status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "ReorderEventNodes failed for SubGraph:%s, status:%u", graph->GetName().c_str(),
                           status);
      GELOGE(status, "[Reorder][EventNodes] failed for SubGraph:%s, status:%u", graph->GetName().c_str(), status);
      return status;
    }
  }

  std::vector<ComputeGraphPtr> subgraphs;
  const auto nodes = AllGraphNodes(subgraphs, compute_graph);
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    const NodePtr node = nodes.at(i);                          // [node: should not be null]
    node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));  // [node->GetOpDescBarePtr(): should not be null]
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::DFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                                    const ConstComputeGraphPtr &compute_graph) const {
  GELOGI("Running_Dfs_Sort, reverse: %d, graph: %s", reverse, name_.c_str());
  std::vector<NodePtr> stack;
  std::map<NodePtr, uint32_t> map_in_edge_num;
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack, map_in_edge_num, compute_graph) == GRAPH_SUCCESS, return GRAPH_FAILED,
                   "sort nodes failed");
  const bool is_mem_priority = IsMemoryPriority();
  std::vector<NodeStatus> nodes_info;
  if (is_mem_priority) {
    InitNodeStatus(compute_graph, nodes_info);
  }
  TopoSortStack topo_sort_stack(&nodes_info, is_mem_priority, true, reverse);
  for (const auto &node : stack) {
    topo_sort_stack.Push(node);
  }
  std::vector<NodePtr> out_nodes;
  const auto stack_push = [&reverse, &topo_sort_stack](std::vector<NodePtr> &tmp_out_nodes) {
    if (reverse) {
      std::reverse(tmp_out_nodes.begin(), tmp_out_nodes.end());
    }
    for (const auto &node : tmp_out_nodes) {
      topo_sort_stack.Push(node);
    }
    tmp_out_nodes.clear();
  };
  // Only data nodes here
  while (!topo_sort_stack.Empty()) {
    const NodePtr node = topo_sort_stack.Pop();
    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDescBarePtr());
    GELOGD("node_vec.push_back %s", node->GetOpDescBarePtr()->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(anchor);
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchorsPtr()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        GetOutNodesFromAnchor(peer_in_anchor, map_in_edge_num, out_nodes);
      }
      stack_push(out_nodes);
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_CHECK_NOTNULL(peer_in_anchor);
        GetOutNodesFromAnchor(peer_in_anchor, map_in_edge_num, out_nodes);
      }
      stack_push(out_nodes);
    }
    GE_IF_BOOL_EXEC(
        node->GetOutControlAnchor() != nullptr, for (const AnchorPtr peer_in_anchor
                                                     : node->GetOutControlAnchor()->GetPeerAnchors()) {
          GE_CHECK_NOTNULL(peer_in_anchor);
          GetOutNodesFromAnchor(peer_in_anchor, map_in_edge_num, out_nodes);
        } stack_push(out_nodes);)
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::StableRDFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                                           const ConstComputeGraphPtr &compute_graph) const {
  (void)reverse;
  GELOGI("Running_Stable_Reverse_Dfs_Sort: %s", name_.c_str());
  std::vector<NodeStatus> nodes_info;
  InitNodeStatus(compute_graph, nodes_info);

  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_CHECK_NOTNULL(node->GetOpDescBarePtr());
    // id作为索引前面的init初始化保证了一定是有效的, walked的节点不入栈
    if (nodes_info[node->GetOpDesc()->GetId()].status == WalkStatus::kWalked) {
      continue;
    }
    // 按照原有的nodes的topo顺序来入栈
    std::vector<Node *> stack = {node.get()};
    while (!stack.empty()) {
      const auto current = stack.back();
      NodeStatus &reverse_dfs_node_info = nodes_info[current->GetOpDesc()->GetId()];
      if (reverse_dfs_node_info.status == WalkStatus::kNotWalked) {
        reverse_dfs_node_info.status = WalkStatus::kWalking;
        // 获取输入节点，反向遍历
        const auto in_all_nodes = current->GetInNodesPtr();
        if (in_all_nodes.empty()) {
          continue;
        }
        std::vector<Node *> in_nodes_has_not_been_walked;
        in_nodes_has_not_been_walked.reserve(in_all_nodes.size());
        for (const auto in_node : in_all_nodes) {
          if (nodes_info[in_node->GetOpDesc()->GetId()].status == WalkStatus::kNotWalked) {
            in_nodes_has_not_been_walked.push_back(in_node);
          }
        }

        auto cmp = [](const Node *lhs, const Node *rhs) {
          // not null
          return lhs->GetOpDescBarePtr()->GetId() > rhs->GetOpDescBarePtr()->GetId();
        };
        // 输入节点的排序使用原始的顺序，可以保证原有topo如果满足当前图的遍历关系，最大程度的保留下来
        std::set<Node *, decltype(cmp)> input_nodes{in_nodes_has_not_been_walked.begin(),
                                                    in_nodes_has_not_been_walked.end(), cmp};
        stack.insert(stack.end(), input_nodes.cbegin(), input_nodes.cend());
      } else {
        stack.pop_back();
        if (reverse_dfs_node_info.status != WalkStatus::kWalking) {
          continue;
        }
        reverse_dfs_node_info.status = WalkStatus::kWalked;
        node_vec.emplace_back(current->shared_from_this());
        GE_CHECK_NOTNULL(current->GetOpDescBarePtr());
        GELOGD("node_vec.push_back %s", current->GetOpDescBarePtr()->GetName().c_str());
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::RDFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                                     const ConstComputeGraphPtr &compute_graph) const {
  (void)reverse;
  GELOGI("Running_Reverse_Dfs_Sort: %s", name_.c_str());
  std::vector<NodeStatus> nodes_info;
  InitNodeStatus(compute_graph, nodes_info);
  NodeCmp cmp(&nodes_info);
  return RDFSCoreSort(node_vec, nodes_info, compute_graph, cmp);
}

graphStatus ComputeGraphImpl::RDFSTopologicalSortingV2(std::vector<NodePtr> &node_vec, const bool reverse,
                                                       const ConstComputeGraphPtr &compute_graph) const {
  (void)reverse;
  GELOGI("Running_Reverse_Dfs_Sort_V2: %s", name_.c_str());
  std::vector<NodeStatus> nodes_info;
  InitNodeStatus(compute_graph, nodes_info);
  NodeCmp cmp(&nodes_info);

  std::vector<NodePtr> initial_vec;
  if (RDFSCoreSort(initial_vec, nodes_info, compute_graph, cmp) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  for (size_t i = 0UL; i < initial_vec.size(); ++i) {
    initial_vec[i]->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));
  }

  std::vector<int64_t> branch_widths(initial_vec.size(), 0);
  for (const auto &node : initial_vec) {
    const auto id = static_cast<size_t>(node->GetOpDesc()->GetId());
    int64_t width = GetNodeInputCount(node);
    for (const auto &in_node : node->GetInAllNodes()) {
      const auto in_id = static_cast<size_t>(in_node->GetOpDesc()->GetId());
      width = std::max(width, branch_widths[in_id]);
    }
    branch_widths[id] = width;
  }

  std::fill(nodes_info.begin(), nodes_info.end(), NodeStatus{0U, WalkStatus::kNotWalked});

  const NodeCmpV2 cmp_v2(&nodes_info, &branch_widths);
  return RDFSCoreSort(node_vec, nodes_info, compute_graph, cmp_v2);
}

graphStatus ComputeGraphImpl::BFSTopologicalSorting(std::vector<NodePtr> &node_vec, const bool reverse,
                                                    const ConstComputeGraphPtr &compute_graph) const {
  GELOGI("Running_Bfs_Sort: %s", name_.c_str());
  (void)reverse;
  const bool is_mem_priority = IsMemoryPriority();
  std::vector<NodeStatus> nodes_info;
  if (is_mem_priority) {
    InitNodeStatus(compute_graph, nodes_info);
  }
  TopoSortStack topo_sort_stack(&nodes_info, is_mem_priority);
  std::vector<NodePtr> stack_input;
  std::map<std::string, NodePtr> breadth_node_map;
  std::map<NodePtr, uint32_t> map_in_edge_num;
  // Record the number of non data nodes but no input nodes
  GE_CHK_BOOL_EXEC(SortNodes(stack_input, map_in_edge_num, compute_graph) == GRAPH_SUCCESS, return GRAPH_FAILED,
                   "sort nodes failed");

  // Only data nodes here
  while ((!stack_input.empty()) || (!topo_sort_stack.Empty())) {
    NodePtr node = nullptr;
    if (!topo_sort_stack.Empty()) {
      node = topo_sort_stack.Pop();
    } else {
      node = stack_input.back();
      stack_input.pop_back();
    }

    node_vec.push_back(node);
    GE_CHECK_NOTNULL(node->GetOpDescBarePtr());
    GELOGD("node_vec.push_back %s", node->GetOpDescBarePtr()->GetName().c_str());
    (void)CollectBreadthOutNode(node, map_in_edge_num, breadth_node_map);

    for (const auto &name_node : breadth_node_map) {
      (void)topo_sort_stack.Push(name_node.second);
    }
    breadth_node_map.clear();
  }
  return GRAPH_SUCCESS;
}

const std::vector<std::pair<af::NodePtr, int32_t>> &ComputeGraphImpl::GetGraphOutNodesInfo() {
  return output_nodes_info_;
}

void ComputeGraphImpl::SetGraphOutNodesInfo(const std::vector<std::pair<af::NodePtr, int32_t>> &out_nodes_info) {
  output_nodes_info_ = out_nodes_info;
}

void ComputeGraphImpl::SetGraphTargetNodesInfo(const std::vector<af::NodePtr> &target_nodes_info) {
  target_nodes_info_ = target_nodes_info;
  old_targets_ = targets_;
  targets_.clear();
  for (auto &node : target_nodes_info_) {
    if (node == nullptr) {
      GELOGW("User pointed targets contain null node, ignore it!");
      continue;
    }
    targets_.insert(node);
  }
  GELOGI("User pointed targets size = %zu.", targets_.size());
}

graphStatus ComputeGraphImpl::CreateNetOutputNode(OpDescPtr &net_output_desc) const {
  // Only flush subgraph name
  std::string node_name =
      (GetParentGraph() != nullptr) ? (GetName() + "_" + NODE_NAME_NET_OUTPUT) : NODE_NAME_NET_OUTPUT;
  net_output_desc = MakeShared<OpDesc>(node_name, NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_desc, "New OpDesc failed");
  (void)AttrUtils::SetListStr(net_output_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                              std::move(std::vector<std::string>()));
  (void)AttrUtils::SetBool(net_output_desc, "_inner_net_output", true);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::GetRetvalOutputInfo(
    const af::NodePtr &node, std::map<int32_t, af::ComputeGraphImpl::RetvalInfo> &retval_node_index_map) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  int64_t output_index = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), RETVAL_ATTR_NAME_INDEX, output_index),
                 "Get Attr:%s from op:%s(%s) failed", RETVAL_ATTR_NAME_INDEX.c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
  GE_ASSERT_TRUE((retval_node_index_map.count(output_index) <= 0),
                 "Attr:%s from op:%s(%s), value:%ld duplicate with other node, check invalid",
                 RETVAL_ATTR_NAME_INDEX.c_str(), node->GetName().c_str(), node->GetType().c_str(), output_index);
  int32_t parent_node_index = -1;
  (void)AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index);
  InDataAnchorPtr in_data_anchor = node->GetInDataAnchor(0);
  GE_CHECK_NOTNULL(in_data_anchor);
  GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor());
  int32_t src_node_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
  NodePtr src_node_ptr = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
  retval_node_index_map[output_index] = {src_node_ptr, src_node_index, parent_node_index};
  GELOGD("retval node %s, index %d", src_node_ptr->GetNamePtr(), src_node_index);
  // if user targets include retval node,delete it from set and insert its input node instead
  // better to GetInNodes here
  const auto iter = targets_.find(node);
  if (iter != targets_.end()) {
    targets_.erase(iter);
    targets_.insert(src_node_ptr);
    GELOGI("Node [%s] is in user def targets, do not output result to user!", node->GetName().c_str());
  }
  is_include_special_node_ = true;
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::CollectOutputNode(const ComputeGraphPtr &compute_graph,
                                                std::vector<af::ComputeGraphImpl::RetvalInfo> &output_nodes_info) {
  for (NodePtr &node : GetDirectNode(compute_graph)) {
    Status ret = SUCCESS;
    if ((node->GetOpDesc() != nullptr) && (node->GetOpDesc()->HasAttr(RETVAL_ATTR_NAME_INDEX))) {
      /// Set the output according to the Retval operator,
      /// identify by whether there is an index parameter
      ret = GetRetvalOutputInfo(node, retval_node_index_map_);
    }
    GE_ASSERT_SUCCESS(ret, "[Get][RetvalOutputInfo] for node:%s failed", node->GetName().c_str());
  }
  GELOGI("Get retval node size:%zu.", retval_node_index_map_.size());
  std::vector<RetvalInfo> out_nodes_tmp;
  /// The Netoutput output is determined by Retval, and the input order
  /// of Netoutput is sorted according to the index value of Retval.
  for (auto &it : retval_node_index_map_) {
    out_nodes_tmp.push_back(it.second);
  }

  for (auto &ele : GetGraphOutNodesInfo()) {
    const auto iter = targets_.find(ele.first);
    if (iter != targets_.end()) {
      GELOGI("User set out node [%s] is found in user def targets, out node is prior!", ele.first->GetName().c_str());
      targets_.erase(iter);
    }

    auto op_desc = ele.first->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int32_t parent_index = -1;
    auto output_desc = op_desc->MutableOutputDesc(ele.second);
    GE_ASSERT_NOTNULL(output_desc, "[Get][OutputDesc] Can not find output tensor desc from node:%s, index %d",
                      op_desc->GetName().c_str(), ele.second);
    (void)af::AttrUtils::GetInt(output_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    output_nodes_info.push_back({ele.first, ele.second, parent_index});
  }
  GELOGI("Output node set by user or leaf node, size:%zu.", output_nodes_info.size());
  for (auto &ele : out_nodes_tmp) {
    // add member, no need to remove duplicated because we need to keep all edges
    output_nodes_info.push_back(ele);
  }
  GELOGI("Get output node, size:%zu.", output_nodes_info.size());

  GE_ASSERT_SUCCESS(CheckOutputNodeInfo(output_nodes_info));
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::CheckOutputNodeInfo(const std::vector<af::ComputeGraphImpl::RetvalInfo> &outputs) const {
  for (auto &item : outputs) {
    NodePtr node = item.output_node;
    GE_ASSERT_NOTNULL(node, "Param outputs has item which output_node is nullptr, check invalid");
    GE_ASSERT_NOTNULL(FindNode(node->GetName()), "Find node:%s from graph:%s failed", node->GetName().c_str(),
                      GetName().c_str());
    GE_CHECK_NOTNULL(node->GetOpDesc());
    int32_t out_size = node->GetOpDesc()->GetOutputsSize();
    int32_t index = item.node_output_index;
    GE_ASSERT_TRUE((index >= 0) && (index < out_size),
                   "Index:%d in param outputs item, < 0 or > output size:%d of node:%s(%s)", index, out_size,
                   node->GetName().c_str(), node->GetType().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddCtrlEdgeForTargets(const af::NodePtr &net_out_node) {
  GE_ASSERT_NOTNULL(net_out_node, "Param net_out_node is nullptr, check invalid");
  // Add ctrl edge for targets
  for (auto &node : targets_) {
    if (node == nullptr) {
      continue;
    }
    // no need to check null because have handled it in run SaveAndRemoveTargets function
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(node->GetOutControlAnchor(), net_out_node->GetInControlAnchor()),
                      "Add control edge between op:%s(%s) and op:%s(%s) failed", node->GetName().c_str(),
                      node->GetType().c_str(), net_out_node->GetName().c_str(), net_out_node->GetType().c_str());
    GELOGD("Add ctrl edge to netoutput node[%s] for target node [%s] success!", net_out_node->GetName().c_str(),
           node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddInOutForNetOutputOp(const OpDescPtr &net_output_desc,
                                                     std::vector<af::ComputeGraphImpl::RetvalInfo> &output_nodes_info) {
  std::vector<bool> is_input_const;
  for (auto iter = output_nodes_info.begin(); iter != output_nodes_info.end();) {
    NodePtr src_node = iter->output_node;
    if (src_node == nullptr) {
      continue;
    }
    // if src_node is in targets_, no need to Add in and out for netoutput
    const auto it = targets_.find(src_node);
    if (it != targets_.end()) {
      iter = output_nodes_info.erase(iter);
      GELOGD("node [%s] is in processed targets, do not add input for netoutput!", src_node->GetName().c_str());
      continue;
    }
    GE_ASSERT_TRUE((src_node != nullptr) && (src_node->GetOpDesc() != nullptr) && (net_output_desc != nullptr),
                   "Param output_nodes_info has RetvalInfo item, which src_node is invalid; "
                   "or Param net_output_desc is nullptr, check invalid");
    is_input_const.push_back(ConstantUtils::IsRealConst(src_node->GetOpDesc()));
    ++iter;
  }
  net_output_desc->SetIsInputConst(is_input_const);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddDataEdgesForNetOutput(
    const ComputeGraphPtr &compute_graph, const af::NodePtr &net_out_node,
    const std::vector<af::ComputeGraphImpl::RetvalInfo> &output_nodes_info) {
  int32_t net_input_index = 0;
  for (auto &item : output_nodes_info) {
    NodePtr src_node = item.output_node;
    auto src_output_index = item.node_output_index;
    GE_CHECK_NOTNULL(src_node);

    auto net_in_data_anchor = net_out_node->GetInDataAnchor(net_input_index);
    // 判断是否已经连过边，防止重复连边
    if ((net_in_data_anchor != nullptr) &&
        src_node->GetOutDataAnchor(src_output_index)->IsLinkedWith(net_in_data_anchor)) {
      net_input_index++;
      continue;
    }

    // 判断是否别的节点已经连边，如果连了，则断开
    if ((net_in_data_anchor != nullptr) && (net_in_data_anchor->GetPeerOutAnchor()) != nullptr) {
      GELOGI("Netoutput node input index %d has been linked, unlink it first.", net_input_index);
      GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(net_in_data_anchor->GetPeerOutAnchor(), net_in_data_anchor),
                        "[Remove][Edge] failed, netoutput %s input index %d.", net_out_node->GetName().c_str(),
                        net_input_index);
    }

    // 如果连着控制边，则断开，因为后面会连数据边
    const auto src_control_anchor = src_node->GetOutControlAnchor();
    const auto dst_control_anchor = net_out_node->GetInControlAnchor();
    if ((src_control_anchor != nullptr) && (dst_control_anchor != nullptr) &&
        src_control_anchor->IsLinkedWith(dst_control_anchor)) {
      GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(src_control_anchor, dst_control_anchor),
                        "[Remove][Edge] remove control edge failed, netoutput %s.", net_out_node->GetName().c_str());
    }

    GE_ASSERT_SUCCESS(net_out_node->AddLinkFrom(net_input_index, src_node, src_output_index),
                      "Add edge between op:%s(%s)(index:%u) and op:%s(%s)(index:%d) failed",
                      src_node->GetName().c_str(), src_node->GetType().c_str(), item.node_output_index,
                      net_out_node->GetName().c_str(), net_out_node->GetType().c_str(), net_input_index);

    GELOGD("AddEdge to output node, src name:%s, src index:%d, dst index:%d.", src_node->GetName().c_str(),
           item.node_output_index, net_input_index);
    net_input_index++;
  }

  // 对于其他多连的数据输入，断开连边
  for (int32_t i = net_input_index; i < static_cast<int32_t>(net_out_node->GetInDataNodesSize()); i++) {
    auto peer_out_anchor = net_out_node->GetInDataAnchor(i)->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      GELOGI("Netoutput node input index %d has been linked, unlink it.", i);
      GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(peer_out_anchor, net_out_node->GetInDataAnchor(i)),
                        "[Remove][Edge] failed, netoutput %s input index %d.", net_out_node->GetName().c_str(), i);
    }
  }
  GE_ASSERT_SUCCESS(RemoveUnusedRetvalNode(compute_graph), "[Remove][UnusedNode] from graph:%s failed.",
                    GetName().c_str());
  // Add true stream, netoutput is 0
  GE_IF_BOOL_EXEC(
      !af::AttrUtils::SetInt(net_out_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
                           net_out_node->GetName().c_str(), net_out_node->GetType().c_str());
      GELOGE(af::INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
             net_out_node->GetName().c_str(), net_out_node->GetType().c_str());
      return af::INTERNAL_ERROR);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::RemoveUnusedRetvalNode(const ComputeGraphPtr &compute_graph) {
  std::vector<af::NodePtr> node_to_delete;
  // Delete _Retval operator.
  for (auto &node : GetDirectNode(compute_graph)) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, GELOGW("Node OpDesc is nullptr"); continue);
    bool need_be_deleted = (node->GetInDataNodesSize() != 0) && (node->GetOutDataNodesSize() == 0) &&
                           (node->GetOpDesc()->HasAttr(RETVAL_ATTR_NAME_INDEX));
    if (need_be_deleted) {
      node_to_delete.push_back(node);
    }
  }
  for (NodePtr &node : node_to_delete) {
    const auto iter = targets_.find(node);
    if (iter != targets_.end()) {
      GELOGI("node[%s] is in user set targets, so do not remove!", node->GetName().c_str());
      continue;
    }
    GE_ASSERT_SUCCESS(RemoveNode(node), "Remove node:%s(%s) from graph:%s failed", node->GetName().c_str(),
                      node->GetType().c_str(), GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::AddNetOutputNodeToGraph(const ComputeGraphPtr &compute_graph, NodePtr &output_node) {
  if (GetAllNodesSize(compute_graph) == 0U) {
    return GRAPH_SUCCESS;
  }
  OpDescPtr net_output_desc = nullptr;
  GE_ASSERT_SUCCESS(CreateNetOutputNode(net_output_desc), "[Create][NetOutputNode] in graph:%s failed.",
                    GetName().c_str());

  std::vector<RetvalInfo> output_nodes_info;
  GE_ASSERT_SUCCESS(CollectOutputNode(compute_graph, output_nodes_info), "[Get][OutputNode] in graph:%s failed.",
                    GetName().c_str());
  GELOGI("OutNodesInfo size:%zu, Targets Size:%zu, is_include_special_node_:%d", output_nodes_info.size(),
         target_nodes_info_.size(), is_include_special_node_);

  // If user does not set out nodes and targets and no retval node, also add netoutput node
  if ((output_nodes_info_.empty()) && (target_nodes_info_.empty()) && !is_include_special_node_) {
    GELOGI("Both output, target and special nodes are empty! add net output node");
    output_node = AddNode(net_output_desc, compute_graph);
    GE_CHECK_NOTNULL(output_node);
    GE_ASSERT_TRUE(af::AttrUtils::SetInt(output_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0),
                   "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
                   output_node->GetName().c_str(), output_node->GetType().c_str());
    GELOGI("Add net output node succeed");
    return SUCCESS;
  }

  if (output_nodes_info.empty()) {
    // because retval node is contained by output_nodes_info, here means targets is non-empty
    output_node = AddNode(net_output_desc, compute_graph);
    GE_ASSERT_NOTNULL(output_node, "Add node:%s(%s) to graph:%s failed", net_output_desc->GetName().c_str(),
                      net_output_desc->GetType().c_str(), GetName().c_str());
    GE_CHK_STATUS_RET(AddCtrlEdgeForTargets(output_node), "[Add][CtrlEdge] for targets failed, output node:%s",
                      output_node->GetName().c_str());
    // Add true stream, netoutput is 0
    GE_IF_BOOL_EXEC(
        !af::AttrUtils::SetInt(output_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0),
        REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
                             output_node->GetName().c_str(), output_node->GetType().c_str());
        GELOGE(af::INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
               output_node->GetName().c_str(), output_node->GetType().c_str());
        return af::INTERNAL_ERROR);
    return GRAPH_SUCCESS;
  }

  GE_ASSERT_SUCCESS(AddInOutForNetOutputOp(net_output_desc, output_nodes_info));
  output_node = AddNode(net_output_desc, compute_graph);
  GE_ASSERT_NOTNULL(output_node, "Add node:%s(%s) to graph:%s failed", net_output_desc->GetName().c_str(),
                    net_output_desc->GetType().c_str(), GetName().c_str());
  GE_ASSERT_SUCCESS(AddDataEdgesForNetOutput(compute_graph, output_node, output_nodes_info),
                    "[Add][Edges] for net output node in graph:%s failed.", GetName().c_str());
  GE_ASSERT_SUCCESS(UpdateNetOutputParentNodeIndex(output_node, output_nodes_info),
                    "[Update][NetOutputInputDesc] for node %s failed", output_node->GetNamePtr());
  GE_ASSERT_SUCCESS(AddCtrlEdgeForTargets(output_node), "[Add][CtrlEdge] for targets failed, net_out_node:%s.",
                    output_node->GetName().c_str());

  GELOGI("Add NetOutput node success.");
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::UpdateNetOutput(const ComputeGraphPtr &compute_graph, const af::NodePtr &output_node,
                                              bool update_data_edge) {
  // update subgraph NetOutput name
  if ((output_node->GetName() == NODE_NAME_NET_OUTPUT) && (GetParentGraph() != nullptr)) {
    std::string node_name = GetName() + "_" + NODE_NAME_NET_OUTPUT;
    output_node->GetOpDesc()->SetName(node_name);
  }

  std::vector<RetvalInfo> output_nodes_info;
  GE_ASSERT_SUCCESS(CollectOutputNode(compute_graph, output_nodes_info), "[Get][OutputNode] in graph:%s failed.",
                    GetName().c_str());

  GE_ASSERT_SUCCESS(UnLinkAnchorsOfNetoutput(output_node),
                    "[UnLink][Connection] between netoutput node:%s and user set target node",
                    output_node->GetName().c_str());
  if (update_data_edge) {
    GE_ASSERT_SUCCESS(AddDataEdgesForNetOutput(compute_graph, output_node, output_nodes_info),
                      "[Add][Edges] for net output node in graph:%s failed.", GetName().c_str());
  }
  GE_ASSERT_SUCCESS(AddCtrlEdgeForTargets(output_node), "[Add][CtrlEdge] for targets failed, net_out_node:%s.",
                    output_node->GetName().c_str());
  GE_ASSERT_SUCCESS(UpdateNetOutputParentNodeIndex(output_node, output_nodes_info),
                    "[Update][NetOutputInputDesc] for node %s failed", output_node->GetNamePtr());
  GE_ASSERT_SUCCESS(UpdateNetOutputDesc(output_node), "[Update][NetOutputDesc] for node:%s failed.",
                    output_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::UpdateNetOutputParentNodeIndex(const af::NodePtr &net_output,
                                                             const std::vector<RetvalInfo> &output_nodes_info) const {
  int32_t net_input_index = 0;
  for (auto &item : output_nodes_info) {
    NodePtr src_node = item.output_node;
    GE_CHECK_NOTNULL(src_node);

    if (item.parent_node_index >= 0) {
      auto input_desc = net_output->GetOpDesc()->MutableInputDesc(net_input_index);
      GE_ASSERT_NOTNULL(input_desc, "Node:%s(%s) has no input desc at index:%d, check invalid",
                        net_output->GetName().c_str(), net_output->GetType().c_str(), net_input_index);
      int32_t parent_node_index = -1;
      if (AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index) && (parent_node_index >= 0)) {
        net_input_index++;
        continue;
      }
      GE_ASSERT_TRUE(AttrUtils::SetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, item.parent_node_index),
                     "Set Attr:%s to input:%d tensor of op:%s(%s) failed", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                     net_input_index, net_output->GetName().c_str(), net_output->GetType().c_str());
      GELOGD("Add parent node index %d for the netoutput input %d on graph %s", item.parent_node_index, net_input_index,
             GetName().c_str());
    }
    net_input_index++;
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::UpdateNetOutputDesc(const af::NodePtr &net_output) const {
  OpDescPtr net_output_desc = net_output->GetOpDesc();
  GE_ASSERT_NOTNULL(net_output_desc, "OpDesc in Param net_output is nullptr, check invalid");
  std::vector<bool> is_input_const;
  for (const auto in_anchor : net_output->GetAllInDataAnchorsPtr()) {
    auto index = static_cast<uint32_t>(in_anchor->GetIdx());
    GE_ASSERT_TRUE(index < net_output_desc->GetAllInputsSize(),
                   "Node:%s(%s) has in_anchor index:%u >= its input desc num:%zu, check invalid",
                   net_output_desc->GetName().c_str(), net_output_desc->GetType().c_str(), index,
                   net_output_desc->GetAllInputsSize());
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      GELOGD("Node %s index %d in anchor peer anchor is null, skip it", net_output->GetNamePtr(), index);
      continue;
    }
    is_input_const.push_back(ConstantUtils::IsRealConst(in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()));
    OpDescPtr src_op_desc = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    uint32_t peer_index = static_cast<uint32_t>(in_anchor->GetPeerOutAnchor()->GetIdx());
    GeTensorDesc output_in_desc = src_op_desc->GetOutputDesc(peer_index);
    uint32_t parent_index = 0U;
    if (AttrUtils::GetInt(net_output_desc->MutableInputDesc(index), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      AttrUtils::SetInt(output_in_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    }
    GELOGD("current desc, format:%s, data type:%s, index:%u.",
           TypeUtils::FormatToSerialString(output_in_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_in_desc.GetDataType()).c_str(), index);
  }
  net_output_desc->SetIsInputConst(is_input_const);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::UnLinkAnchorsOfNetoutput(const af::NodePtr &net_out_node) {
  GELOGI("Enter Unlink process.");
  GE_ASSERT_SUCCESS(UnLinkDataAnchorOfNetoutput(net_out_node), "UnLinkDataAnchorOfNetoutput process fail.");
  GE_ASSERT_SUCCESS(UnLinkControlAnchorOfNetoutput(net_out_node), "UnLinkControlAnchorOfNetoutput process fail.");
  return GRAPH_SUCCESS;
}

bool ComputeGraphImpl::CheckNodeIsInOutputNodes(const af::NodePtr &node) const {
  for (auto &ele : output_nodes_info_) {
    auto out_node = ele.first;
    if (node == out_node) {
      return true;
    }
  }
  return false;
}

graphStatus ComputeGraphImpl::UnLinkDataAnchorOfNetoutput(const af::NodePtr &net_out_node) {
  GE_ASSERT_NOTNULL(net_out_node, "Param net_out_node is nullptr, check invalid");
  Status ret = SUCCESS;
  // unlink all anchor to data anchor of netoutput
  for (auto in_data_anchor : net_out_node->GetAllInDataAnchorsPtr()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGI("PeerOutAnchor is null!");
      continue;
    }
    auto node = peer_out_anchor->GetOwnerNode();
    const auto iter = targets_.find(node);
    if (iter != targets_.end()) {
      if (!CheckNodeIsInOutputNodes(node)) {
        GE_ASSERT_SUCCESS(in_data_anchor->Unlink(peer_out_anchor),
                          "Op:%s(%s) out index:%d unlink from op:%s(%s) in index:%d failed",
                          net_out_node->GetName().c_str(), net_out_node->GetType().c_str(), in_data_anchor->GetIdx(),
                          node->GetName().c_str(), node->GetType().c_str(), peer_out_anchor->GetIdx());
      } else {
        targets_.erase(iter);
      }
    }
  }
  return ret;
}

graphStatus ComputeGraphImpl::UnLinkControlAnchorOfNetoutput(const af::NodePtr &net_out_node) {
  GE_ASSERT_NOTNULL(net_out_node, "Param net_out_node is nullptr, check invalid");
  Status ret = SUCCESS;
  auto in_control_anchor = net_out_node->GetInControlAnchor();
  GE_ASSERT_NOTNULL(in_control_anchor, "In control anchor of param net_out_node:%s(%s) is nullptr, check invalid",
                    net_out_node->GetName().c_str(), net_out_node->GetType().c_str());
  // unlink all data anchor to control anchor of netoutput
  for (auto &peer_out_data_anchor : in_control_anchor->GetPeerOutDataAnchors()) {
    if (peer_out_data_anchor == nullptr) {
      GELOGD("PeerOutControlAnchor is null!");
    } else {
      auto node = peer_out_data_anchor->GetOwnerNode();
      const auto iter = targets_.find(node);
      if (iter != targets_.end()) {
        if (!(CheckNodeIsInOutputNodes(node))) {
          GE_ASSERT_SUCCESS(in_control_anchor->Unlink(peer_out_data_anchor),
                            "Op:%s(%s) unlink control edge from op:%s(%s) failed", net_out_node->GetName().c_str(),
                            net_out_node->GetType().c_str(), node->GetName().c_str(), node->GetType().c_str());
        } else {
          targets_.erase(iter);
        }
      }
    }
  }
  /// check all control anchor to control anchor of netoutput and delete it from targets
  /// to avoid duplicated add control edge;
  for (auto &peer_out_control_anchor : in_control_anchor->GetPeerOutControlAnchors()) {
    if (peer_out_control_anchor == nullptr) {
      GELOGD("PeerOutControlAnchor is null");
    } else {
      auto node = peer_out_control_anchor->GetOwnerNode();
      const auto iter = targets_.find(node);
      if (iter != targets_.end()) {
        // 如果是Target又是Output，则不需要控制边，因为后续会连数据边
        if (CheckNodeIsInOutputNodes(node)) {
          GE_ASSERT_SUCCESS(in_control_anchor->Unlink(peer_out_control_anchor),
                            "Op:%s(%s) unlink control edge from op:%s(%s) failed", net_out_node->GetName().c_str(),
                            net_out_node->GetType().c_str(), node->GetName().c_str(), node->GetType().c_str());
        }
        targets_.erase(iter);
      } else if (old_targets_.find(node) != old_targets_.end()) {
        // 如果是旧的Target，则删除控制边
        GE_ASSERT_SUCCESS(in_control_anchor->Unlink(peer_out_control_anchor),
                          "Op:%s(%s) unlink control edge from op:%s(%s) failed", net_out_node->GetName().c_str(),
                          net_out_node->GetType().c_str(), node->GetName().c_str(), node->GetType().c_str());
      }
    }
  }
  return ret;
}

graphStatus ComputeGraphImpl::CreateOrUpdateNetoutput(const ComputeGraphPtr &compute_graph, bool update_data_edge) {
  GELOGI("Run.graph is [%s]", GetName().c_str());
  auto output_node = FindFirstNodeMatchType("Output");
  // 自动融合场景 AscGraph里使用Output，该场景下不需要创建NetOutput
  if (output_node != nullptr) {
    GELOGI("Graph %s has Output node, no need to create or update NetOutput node.", GetName().c_str());
    return GRAPH_SUCCESS;
  }
  NodePtr net_output_node = FindFirstNodeMatchType(NETOUTPUT);
  // If graph already has a netoutput node, doesn't need to create it again.
  if (net_output_node != nullptr) {
    (void)AttrUtils::SetListStr(net_output_node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES,
                                std::move(std::vector<std::string>()));
    GE_ASSERT_SUCCESS(UpdateNetOutput(compute_graph, net_output_node, update_data_edge),
                      "[Process][WithNetoutput] failed, output_node:%s, graph:%s.", net_output_node->GetName().c_str(),
                      GetName().c_str());
    // Add true stream, netoutput is 0
    GE_CHK_BOOL_RET_STATUS(af::AttrUtils::SetInt(net_output_node->GetOpDesc(), ATTR_NAME_TRUE_BRANCH_STREAM, 0), FAILED,
                           "Failed to set attr[%s] to op:%s(%s).", ATTR_NAME_TRUE_BRANCH_STREAM.c_str(),
                           net_output_node->GetName().c_str(), net_output_node->GetType().c_str());
  } else {
    GE_ASSERT_SUCCESS(AddNetOutputNodeToGraph(compute_graph, net_output_node),
                      "[Add][NetOutputNode] to graph:%s failed.", GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                    std::map<std::string, NodePtr> &breadth_node_map) const {
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchorsPtr()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end()) {
        --iter->second;
        if (iter->second == 0U) {
          (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNodeBarePtr()->GetName(),
                                         peer_in_anchor->GetOwnerNode());
        }
      }
    }

    for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end()) {
        --iter->second;
        if (iter->second == 0U) {
          (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNodeBarePtr()->GetName(),
                                         peer_in_anchor->GetOwnerNode());
        }
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto peer_in_anchor : node->GetOutControlAnchor()->GetPeerAnchorsPtr()) {
      const auto iter = map_in_edge_num.find(peer_in_anchor->GetOwnerNode());
      if (iter != map_in_edge_num.end()) {
        --iter->second;
        if (iter->second == 0U) {
          (void)breadth_node_map.emplace(peer_in_anchor->GetOwnerNodeBarePtr()->GetName(),
                                         peer_in_anchor->GetOwnerNode());
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

void ComputeGraphImpl::TopologicalSorting(const std::function<bool(const NodePtr &, const NodePtr &)> comp) {
  nodes_.sort(std::move(comp));
  int64_t num = 0;
  for (const NodePtr &node : nodes_) {
    node->GetOpDescBarePtr()->SetId(num++);  // node should not be null, node->GetOpDescBarePtr() should not be null]
  }
}

graphStatus ComputeGraphImpl::TopologicalSorting(const ComputeGraphPtr &const_graph_ptr,
                                                 const ConstComputeGraphPtr &const_compute_graph) {
  auto ret = TopologicalSortingGraph(const_compute_graph);
  if (ret != GRAPH_SUCCESS) {
    GE_DUMP(const_graph_ptr, "black_box" + name_);
    REPORT_INNER_ERR_MSG("E18888", "Graph [%s] topological sort failed, saved to file black_box", name_.c_str());
    GELOGW("[Sort][Graph] Graph [%s] topological sort failed, saved to file black_box", name_.c_str());
    return ret;
  }

  if (sub_graph_.empty()) {
    return GRAPH_SUCCESS;
  }

  // partition sub graph
  for (const auto &sub_graph : sub_graph_) {
    ret = sub_graph->TopologicalSortingGraph();
    if (ret != GRAPH_SUCCESS) {
      GE_DUMP(sub_graph, "black_box" + sub_graph->GetName());
      REPORT_INNER_ERR_MSG("E18888", "Sub graph[%s] topological sort failed, saved to file black_box",
                           sub_graph->GetName().c_str());
      GELOGW("[Sort][Graph] Sub graph[%s] topological sort failed, saved to file black_box",
             sub_graph->GetName().c_str());
      return ret;
    }
  }

  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  auto nodes = AllGraphNodes(subgraphs, const_compute_graph);
  for (size_t i = 0UL; i < nodes.size(); i++) {
    const NodePtr node = nodes.at(i);                          // [node: should not be null]
    node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));  // [node->GetOpDescBarePtr(): should not be null]
  }
  if (sub_graph_.size() != subgraphs.size()) {  // Graph Partition use subgraph, Keep original
    GELOGW("[TopoSort][CheckNodeSize] Keep original subgraph for graph size %zu not equal %zu.", sub_graph_.size(),
           subgraphs.size());
    return GRAPH_SUCCESS;
  }
  sub_graph_.swap(subgraphs);
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::DoTopologicalSorting(const ConstComputeGraphPtr &compute_graph,
                                                   TopoSortingMode sorting_mode, bool dfs_reverse) {
  using TopoSortingStrategy =
      std::function<graphStatus(ComputeGraphImpl *, std::vector<NodePtr> &, const bool, const ConstComputeGraphPtr &)>;
  static const std::map<TopoSortingMode, TopoSortingStrategy> topo_sorting_strategy{
      {TopoSortingMode::kBFS, &ComputeGraphImpl::BFSTopologicalSorting},
      {TopoSortingMode::kDFS, &ComputeGraphImpl::DFSTopologicalSorting},
      {TopoSortingMode::kRDFS, &ComputeGraphImpl::RDFSTopologicalSorting},
      {TopoSortingMode::kStableRDFS, &ComputeGraphImpl::StableRDFSTopologicalSorting},
      {TopoSortingMode::kRDFSV2, &ComputeGraphImpl::RDFSTopologicalSortingV2}};

  std::vector<NodePtr> node_vec;
  const auto it = topo_sorting_strategy.find(sorting_mode);
  if (it == topo_sorting_strategy.end()) {
    GELOGE(GRAPH_FAILED, "Can not find topo sorting strategy of %d.", static_cast<int32_t>(sorting_mode));
    return GRAPH_FAILED;
  }
  if (it->second(this, node_vec, dfs_reverse, compute_graph) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  // If they are not equal, there is a closed loop
  if (node_vec.size() != GetDirectNodesSize()) {
    std::set<Node *> itered_nodes_set;
    for (auto &node : node_vec) {
      (void)itered_nodes_set.insert(node.get());
    }
    REPORT_INNER_ERR_MSG("E18888",
                         "Failed to do topo sorting total %zu, iterated %zu, a closed loop exists in graph:%s",
                         GetDirectNodesSize(), node_vec.size(), name_.c_str());
    GELOGW("[Check][Param] Failed to do topo sorting total %zu, iterated %zu, a closed loop exists in graph.",
           GetDirectNodesSize(), node_vec.size());
    for (auto &node : nodes_) {
      if (itered_nodes_set.count(node.get()) == 0UL) {
        GELOGW("[Check][Param] The node %s is not iterated when topological sorting", node->GetName().c_str());
      }
    }
    return GRAPH_FAILED;
  }

  ClearNodeList();
  if ((IsMemoryPriority() && (sorting_mode != TopoSortingMode::kStableRDFS)) ||
      (sorting_mode == TopoSortingMode::kRDFS) || (sorting_mode == TopoSortingMode::kRDFSV2)) {
    DelayTopoSort(node_vec, compute_graph);
  }
  for (size_t i = 0UL; i < node_vec.size(); i++) {
    const NodePtr node = node_vec[i];                          // [node: should not be null]
    node->GetOpDescBarePtr()->SetId(static_cast<int64_t>(i));  // [node->GetOpDescBarePtr(): should not be null]
    PushBackToNodeList(node);
  }

  is_valid_flag_ = true;
  return GRAPH_SUCCESS;
}

graphStatus ComputeGraphImpl::TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph,
                                                      const bool dfs_reverse) {
  return DoTopologicalSorting(compute_graph, GetTopoSortingStrategy(), dfs_reverse);
}

graphStatus ComputeGraphImpl::TopologicalSortingGraph(const ConstComputeGraphPtr &compute_graph,
                                                      TopoSortingMode topo_sorting_mode) {
  return DoTopologicalSorting(compute_graph, topo_sorting_mode, false);
}

graphStatus ComputeGraphImpl::SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                        const ConstComputeGraphPtr &compute_graph) const {
  // Record the number of non data nodes but no input nodes
  uint32_t spec_node_size = 0U;
  for (const auto &node : GetDirectNode(compute_graph)) {
    GE_IF_BOOL_EXEC(node->GetOpDescBarePtr() == nullptr, continue);
    map_in_edge_num[node] = static_cast<uint32_t>(GetInEdgeSize(node));
    if (map_in_edge_num[node] == 0U) {
      if ((!OpTypeUtils::IsDataNode(node->GetOpDescBarePtr()->GetType())) &&
          (node->GetOpDescBarePtr()->GetType() != INPUT_TYPE) && (node->GetOpDescBarePtr()->GetType() != RECV) &&
          (node->GetOpDescBarePtr()->GetType() != SEND)) {
        (void)stack.insert(stack.begin(), node);
        spec_node_size++;
        continue;
      }
      // Need to insert the data nodes in reverse order
      (void)stack.insert(stack.begin() + static_cast<int64_t>(spec_node_size), node);
    }
  }

  /// Make sure the inputs order matches with user-designated
  /// 1. Get the index of two input nodes in the user-inputs-order(inputs_order_)
  /// 2. Compare two indices, if not match, swap the positions of two inputs
  /// *: Remind: stack is reverse-order
  for (size_t i = 0UL; i < stack.size(); ++i) {
    // If not found in 'inputs_order_', skip it
    const auto it_i = std::find(inputs_order_.begin(), inputs_order_.end(), stack[i]->GetName());
    GE_IF_BOOL_EXEC(it_i == inputs_order_.end(), continue);
    const auto inx_i = it_i - inputs_order_.begin();
    for (size_t j = i + 1UL; j < stack.size(); ++j) {
      // If not found in 'inputs_order_', skip it
      const auto it_j = std::find(inputs_order_.begin(), inputs_order_.end(), stack[j]->GetName());
      GE_IF_BOOL_EXEC(it_j == inputs_order_.end(), continue);

      // Compare index, swap them if it should be
      const auto inx_j = it_j - inputs_order_.begin();
      GE_IF_BOOL_EXEC(inx_i < inx_j, std::swap(stack[i], stack[j]));
    }
  }

  return GRAPH_SUCCESS;
}

size_t ComputeGraphImpl::GetInEdgeSize(const NodePtr &node) const {
  size_t in_edge_size = 0UL;
  if (node == nullptr) {
    return in_edge_size;
  }
  for (const auto &anchor : node->GetAllInDataAnchorsPtr()) {
    in_edge_size = in_edge_size + anchor->GetPeerAnchorsSize();
    // Break flow control data loop.
    const OutDataAnchorPtr out_anchor = anchor->GetPeerOutAnchor();
    if ((out_anchor != nullptr) && (out_anchor->GetOwnerNodeBarePtr() != nullptr)) {
      const auto out_node = out_anchor->GetOwnerNodeBarePtr();
      if ((out_node->GetType() == NEXTITERATION) || (out_node->GetType() == REFNEXTITERATION)) {
        GE_IF_BOOL_EXEC(in_edge_size == 0UL,
                        GELOGE(GRAPH_FAILED,
                               "[Check][Param] Node %s has in_edge_size 0 for NEXTITERATION output node %s, "
                               "reversing result is invalid",
                               node->GetName().c_str(), out_node->GetName().c_str());
                        return in_edge_size);
        in_edge_size -= 1UL;
      }
    }
  }
  if (node->GetInControlAnchor() != nullptr) {
    in_edge_size = in_edge_size + node->GetInControlAnchor()->GetPeerAnchorsSize();
  }
  return in_edge_size;
}

size_t ComputeGraphImpl::GetOutEdgeSize(const NodePtr &node) const {
  size_t out_edge_size = 0UL;
  if (node == nullptr) {
    return out_edge_size;
  }

  // Break flow control data loop.
  if ((node->GetType() != NEXTITERATION) && (node->GetType() != REFNEXTITERATION)) {
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      if (anchor != nullptr) {
        out_edge_size = out_edge_size + anchor->GetPeerAnchorsSize();
      }
    }
  }
  if (node->GetOutControlAnchor() != nullptr) {
    if (out_edge_size > (UINT64_MAX - node->GetOutControlAnchor()->GetPeerAnchorsSize())) {
      return 0UL;
    }
    out_edge_size = out_edge_size + node->GetOutControlAnchor()->GetPeerAnchorsSize();
  }
  return out_edge_size;
}

bool ComputeGraphImpl::IsValid() const {
  return is_valid_flag_;
}

void ComputeGraphImpl::InValid() {
  is_valid_flag_ = false;
}

void ComputeGraphImpl::Dump(const ConstComputeGraphPtr &graph) const {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return;
  }

  GELOGI("graph name = %s.", GetName().c_str());
  for (const auto &node : GetAllNodes(graph)) {
    GELOGD("node name = %s.", node->GetName().c_str());
    for (const auto &anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchorsPtr()) {
        GE_IF_BOOL_EXEC((peer_in_anchor != nullptr) && (peer_in_anchor->GetOwnerNode() != nullptr),
                        GELOGD("node name = %s, out data node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC((peer_in_anchor != nullptr) && (peer_in_anchor->GetOwnerNode() != nullptr),
                        GELOGD("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
    const auto out_control_anchor = node->GetOutControlAnchor();
    if (out_control_anchor != nullptr) {
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInControlAnchors()) {
        GE_IF_BOOL_EXEC((peer_in_anchor != nullptr) && (peer_in_anchor->GetOwnerNode() != nullptr),
                        GELOGD("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
      for (const auto &peer_in_anchor : out_control_anchor->GetPeerInDataAnchors()) {
        GE_IF_BOOL_EXEC((peer_in_anchor != nullptr) && (peer_in_anchor->GetOwnerNode() != nullptr),
                        GELOGD("node name = %s, out control node name = %s.", node->GetName().c_str(),
                               peer_in_anchor->GetOwnerNode()->GetName().c_str()));
      }
    }
  }
}

void ComputeGraphImpl::Swap(ComputeGraphImpl &graph) {
  origGraph_.swap(graph.origGraph_);

  name_.swap(graph.name_);
  std::swap(graph_id_, graph.graph_id_);
  attrs_.Swap(graph.attrs_);
  nodes_.swap(graph.nodes_);
  const auto tmp_size = direct_nodes_size_;
  direct_nodes_size_ = graph.direct_nodes_size_;
  graph.direct_nodes_size_ = tmp_size;
  all_nodes_infos_.swap(graph.all_nodes_infos_);
  target_nodes_info_.swap(graph.target_nodes_info_);

  input_nodes_.swap(graph.input_nodes_);
  inputs_order_.swap(graph.inputs_order_);
  std::swap(input_size_, graph.input_size_);
  out_nodes_map_.swap(graph.out_nodes_map_);
  std::swap(output_size_, graph.output_size_);
  output_nodes_info_.swap(graph.output_nodes_info_);

  sub_graph_.swap(graph.sub_graph_);
  names_to_subgraph_.swap(graph.names_to_subgraph_);
  parent_graph_.swap(graph.parent_graph_);
  parent_graph_bare_ptr_ = parent_graph_.lock().get();
  parent_node_.swap(graph.parent_node_);
  parent_node_bare_ptr_ = parent_node_.lock().get();
  graph_netoutput_.swap(graph.graph_netoutput_);

  // the members followed should not in the ComputeGraphImpl class
  std::swap(is_valid_flag_, graph.is_valid_flag_);
  std::swap(is_summary_graph_, graph.is_summary_graph_);
  std::swap(need_iteration_, graph.need_iteration_);
  params_share_map_.swap(graph.params_share_map_);
  op_name_map_.swap(graph.op_name_map_);
  std::swap(session_id_, graph.session_id_);
  std::swap(data_format_, graph.data_format_);
}

void ComputeGraphImpl::SetNodesOwner(const ComputeGraphPtr &compute_graph) {
  for (const auto &node : nodes_) {
    if (node == nullptr) {
      continue;
    }
    (void)node->SetOwnerComputeGraph(compute_graph);
  }
}

void ComputeGraphImpl::SetTopParentGraph(const ComputeGraphPtr &compute_graph) {
  for (const auto &sub_graph : sub_graph_) {
    if ((sub_graph == nullptr) || (sub_graph->GetParentGraph() == nullptr) ||
        (sub_graph->GetParentGraph()->GetParentGraph() != nullptr)) {
      continue;
    }
    (void)sub_graph->SetParentGraph(compute_graph);
  }
}

graphStatus ComputeGraphImpl::IsolateNode(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  const auto next_nodes = node->GetOutAllNodes();
  // If there is input data side
  for (size_t i = 0UL; i < node->GetAllInDataAnchorsSize(); i++) {
    const auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto pre_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (pre_out_data_anchor != nullptr) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_data_anchor, in_data_anchor) == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                            pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                            in_data_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                              pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                              in_data_anchor->GetOwnerNode()->GetName().c_str());
      GE_IF_BOOL_EXEC((pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANT) ||
                          (pre_out_data_anchor->GetOwnerNode()->GetType() == CONSTANTOP),
                      continue);
      for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
        for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                                out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                                next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                                  out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                  next_in_data_anchor->GetOwnerNode()->GetName().c_str());
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                           REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                                pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                                next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                                  pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                  next_in_data_anchor->GetOwnerNode()->GetName().c_str());
        }
        for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
          GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                                out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                                  out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                  next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
          GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                           REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                                pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                           return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                                  pre_out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                  next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        }
      }
      const auto out_ctrl_anchor = node->GetOutControlAnchor();
      GE_CHECK_NOTNULL(out_ctrl_anchor);
      const auto pre_out_ctrl_anchor = pre_out_data_anchor->GetOwnerNodeBarePtr()->GetOutControlAnchor();
      GE_CHECK_NOTNULL(pre_out_ctrl_anchor);
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                              out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                                out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                              pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                                pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }

  // If there is an input control side
  const auto in_ctrl_anchor = node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  for (const auto &pre_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(pre_out_ctrl_anchor, in_ctrl_anchor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                          pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                          in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                     return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                            pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                            in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                              out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                                out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                              pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                                pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
    const auto out_ctrl_anchor = node->GetOutControlAnchor();
    if (out_ctrl_anchor != nullptr) {
      for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
        GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                              out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                                out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(pre_out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                         REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                              pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                         return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                                pre_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
      }
    }
  }

  for (const auto &out_peer_data_anchor : in_ctrl_anchor->GetPeerOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_peer_data_anchor, in_ctrl_anchor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                          out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                                          in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                     return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                            out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                            in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    for (const auto &next_node : next_nodes) {
      const auto next_in_control_anchor = next_node->GetInControlAnchor();
      GE_CHK_BOOL_EXEC(GraphUtils::AddEdge(out_peer_data_anchor, next_in_control_anchor) == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "add edge from %s to %s failed",
                                            out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                                            next_in_control_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Add][Edge] from %s to %s failed",
                              out_peer_data_anchor->GetOwnerNode()->GetName().c_str(),
                              next_in_control_anchor->GetOwnerNode()->GetName().c_str());
    }
  }

  return RemoveExtraOutEdge(node);
}

graphStatus ComputeGraphImpl::RemoveExtraOutEdge(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  // Remove redundant output edges
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    for (const auto &next_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_data_anchor) == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                            out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                            next_in_data_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                              out_data_anchor->GetOwnerNode()->GetName().c_str(),
                              next_in_data_anchor->GetOwnerNode()->GetName().c_str());
    }

    for (const auto &next_in_ctrl_anchor : out_data_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_data_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                            out_data_anchor->GetOwnerNode()->GetName().c_str(),
                                            next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                              out_data_anchor->GetOwnerNode()->GetName().c_str(),
                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    }
  }
  const auto out_ctrl_anchor = node->GetOutControlAnchor();
  if (out_ctrl_anchor != nullptr) {
    for (const auto &next_in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(GraphUtils::RemoveEdge(out_ctrl_anchor, next_in_ctrl_anchor) == GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "remove edge from %s to %s failed",
                                            out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                                            next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
                       return GRAPH_FAILED, "[Remove][Edge] from %s to %s failed",
                              out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                              next_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

ProtoAttrMap &ComputeGraphImpl::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &ComputeGraphImpl::GetAttrMap() const {
  return attrs_;
}

const std::map<OperatorImplPtr, NodePtr> &ComputeGraphImpl::GetAllNodesInfo() const {
  return all_nodes_infos_;
}

void ComputeGraphImpl::SetUserDefOutput(const std::string &output_name) {
  if (output_name.empty()) {
    return;
  }

  const std::vector<std::string> nodes = StringUtils::Split(output_name, ';');
  for (const std::string &node : nodes) {
    std::vector<std::string> item = StringUtils::Split(node, ':');
    if (item.size() != OUTPUT_PARAM_SIZE) {
      REPORT_INNER_ERR_MSG("W18888", "Check output param size failed, output_name:%s, current size:%zu, expected:%zu",
                           output_name.c_str(), item.size(), OUTPUT_PARAM_SIZE);
      GELOGW("[Check][Output] Check output param size failed, output_name:%s, current size:%zu, expected:%zu",
             output_name.c_str(), item.size(), OUTPUT_PARAM_SIZE);
      continue;
    }

    int32_t index;
    try {
      index = stoi(StringUtils::Trim(item[1UL]));
    } catch (const std::out_of_range &) {
      REPORT_INNER_ERR_MSG("W18888", "Catch out_of_range exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch out_of_range exception, output_name:%s", output_name.c_str());
      continue;
    } catch (const std::invalid_argument &) {
      REPORT_INNER_ERR_MSG("W18888", "Catch invalid_argument exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch invalid_argument exception, output_name:%s", output_name.c_str());
      continue;
    } catch (...) {
      REPORT_INNER_ERR_MSG("W18888", "Catch exception, output_name:%s", output_name.c_str());
      GELOGW("[Catch][Exception] Catch exception, output_name:%s", output_name.c_str());
      continue;
    }
    const auto iter = out_nodes_map_.find(item[0UL]);
    if (iter == out_nodes_map_.end()) {
      out_nodes_map_[item[0UL]] = std::vector<int32_t>(1UL, index);
    } else {
      const auto idx_iter = std::find(iter->second.begin(), iter->second.end(), index);
      if (idx_iter == iter->second.end()) {
        iter->second.push_back(index);
      }
    }
  }
}

const std::string ComputeGraphImpl::GetOutput() {
  static const int32_t resultDefaultSize = 2048;
  std::string result;
  result.reserve(static_cast<uint64_t>(resultDefaultSize));
  auto iter = out_nodes_map_.begin();
  while (iter != out_nodes_map_.end()) {
    const auto idxes = iter->second;
    for (const auto idx : idxes) {
      (void)result.append(iter->first).append(":").append(std::to_string(idx)).append(";");
    }
    ++iter;
  }

  return result.substr(0UL, result.length() - 1UL);
}

void ComputeGraphImpl::EraseFromNodeList(const std::list<NodePtr>::iterator &position) {
  (void)nodes_.erase(position);
  --direct_nodes_size_;
}

void ComputeGraphImpl::InsertToNodeList(const std::list<NodePtr>::iterator &position, const NodePtr &node) {
  (void)nodes_.insert(position, node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::PushBackToNodeList(const NodePtr &node) {
  (void)nodes_.push_back(node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::EmplaceBackToNodeList(const NodePtr &node) {
  (void)nodes_.emplace_back(node);
  ++direct_nodes_size_;
}

void ComputeGraphImpl::ClearNodeList() {
  (void)nodes_.clear();
  direct_nodes_size_ = 0UL;
}

void ComputeGraphImpl::ReorderByNodeId() {
  std::vector<NodePtr> node_vec(nodes_.begin(), nodes_.end());
  std::sort(node_vec.begin(), node_vec.end(), [](const NodePtr &lhs, const NodePtr &rhs) {
    return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
  });
  ClearNodeList();
  for (const auto &node : node_vec) {
    PushBackToNodeList(node);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const std::string &name)
    : enable_shared_from_this(), AttrHolder(), impl_(ComGraphMakeSharedAndThrow<ComputeGraphImpl>(name)) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const char_t *name)
    : ComputeGraph(std::string((name == nullptr) ? "" : name)) {}

ComputeGraph::~ComputeGraph() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(const ComputeGraph &compute_graph)
    : enable_shared_from_this(),
      AttrHolder(compute_graph),
      impl_(ComGraphMakeSharedAndThrow<ComputeGraphImpl>(*(compute_graph.impl_))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::ComputeGraph(ComputeGraph &&compute_graph)
    : enable_shared_from_this(),
      AttrHolder(std::move(compute_graph)),
      impl_(ComGraphMakeSharedAndThrow<ComputeGraphImpl>(std::move(*(compute_graph.impl_)))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string ComputeGraph::GetName() const {
  return impl_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetName(const std::string &name) {
  impl_->SetName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t ComputeGraph::GetAllNodesSize() const {
  return GetAllNodesPtr().size();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetAllNodes() const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodes(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> ComputeGraph::GetAllNodesPtr() const {
  std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
  return AllGraphNodesPtr(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetAllNodes(
    const NodeFilter &node_filter, const GraphFilter &graph_filter) const {
  return impl_->GetAllNodes(node_filter, graph_filter, shared_from_this());
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::AllGraphNodes(std::vector<ComputeGraphPtr> &subgraphs) const {
  return impl_->AllGraphNodes(subgraphs, shared_from_this());
}

std::vector<Node *> ComputeGraph::AllGraphNodesPtr(std::vector<ComputeGraphPtr> &subgraphs) const {
  return impl_->AllGraphNodesPtr(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetNodes(
    const bool is_unknown_shape) const {
  return impl_->GetNodes(is_unknown_shape, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetNodes(
    const bool is_unknown_shape, const NodeFilter &node_filter, const GraphFilter &graph_filter) const {
  return impl_->GetNodes(is_unknown_shape, node_filter, graph_filter, shared_from_this());
}

size_t ComputeGraph::GetDirectNodesSize() const {
  return impl_->GetDirectNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph::Vistor<NodePtr> ComputeGraph::GetDirectNode() const {
  return impl_->GetDirectNode(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> ComputeGraph::GetDirectNodePtr() const {
  return impl_->GetDirectNodePtr();
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetInputNodes() const {
  return impl_->GetInputNodes(shared_from_this());
}

ComputeGraph::Vistor<NodePtr> ComputeGraph::GetOutputNodes() const {
  return impl_->GetOutputNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::FindNode(const std::string &name) const {
  return impl_->FindNode(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr
ComputeGraph::FindFirstNodeMatchType(const std::string &name) const {
  return impl_->FindFirstNodeMatchType(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphAttrsAreEqual(
    const ComputeGraph &r_graph) const {
  return impl_->GraphAttrsAreEqual(*(r_graph.impl_));
}

/// Since there may be different input nodes
/// chosen by user in the same graph, special judgment is needed
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::VectorInputNodePtrIsEqual(
    const std::vector<NodePtr> &left_nodes, const std::vector<NodePtr> &right_nodes) const {
  return impl_->VectorInputNodePtrIsEqual(left_nodes, right_nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GraphMembersAreEqual(
    const ComputeGraph &r_graph) const {
  return impl_->GraphMembersAreEqual(*(r_graph.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::operator==(
    const ComputeGraph &r_compute_graph) const {
  return (*impl_) == (*(r_compute_graph.impl_));
}

ComputeGraph &ComputeGraph::operator=(ComputeGraph &compute_graph) {
  if (&compute_graph == this) {
    return *this;
  }
  AttrHolder::SwapBase(compute_graph);
  *impl_ = *(compute_graph.impl_);
  return *this;
}

NodePtr ComputeGraph::AddNodeFront(const NodePtr node) {
  return impl_->AddNodeFront(node);
}

NodePtr ComputeGraph::AddNodeFront(const OpDescPtr &op) {
  return impl_->AddNodeFront(op, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(const NodePtr node) {
  return impl_->AddNode(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<NodePtr> ComputeGraph::InsertNodes(
    const NodePtr &node, const std::vector<OpDescPtr> &insert_ops) {
  return impl_->InsertNodes(node, insert_ops, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::InsertNode(const NodePtr &node,
                                                                                const OpDescPtr &insert_op) {
  return impl_->InsertNode(node, insert_op, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::InsertNodeBefore(const NodePtr &node,
                                                                                      const OpDescPtr &insert_op) {
  return impl_->InsertNodeBefore(node, insert_op, shared_from_this());
}

bool ComputeGraph::IsSupportFuse(const std::vector<NodePtr> &origin_nodes, std::string &reason_not_support) const {
  return impl_->IsSupportFuse(origin_nodes, reason_not_support);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<NodePtr> ComputeGraph::FuseNodeKeepTopo(
    const std::vector<NodePtr> &ori_nodes, const std::vector<OpDescPtr> &fusion_ops) {
  return impl_->FuseNodeKeepTopo(ori_nodes, fusion_ops, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr ComputeGraph::AddNode(const OpDescPtr op) {
  return impl_->AddNode(op, shared_from_this());
}

NodePtr ComputeGraph::AddNode(const OpDescPtr op, const int64_t id) {  // for unserialize.
  return impl_->AddNode(op, id, shared_from_this());
}

NodePtr ComputeGraph::AddInputNode(const NodePtr node) {
  return impl_->AddInputNode(node);
}

NodePtr ComputeGraph::AddOutputNode(const NodePtr node) {
  return AddOutputNodeByIndex(node, 0);
}

NodePtr ComputeGraph::AddOutputNodeByIndex(const NodePtr node, const int32_t index) {
  return impl_->AddOutputNodeByIndex(node, index);
}

graphStatus ComputeGraph::RemoveConstInput(const NodePtr &node) {
  return impl_->RemoveConstInput(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::RemoveNode(const NodePtr &node) {
  return impl_->RemoveNode(node);
}

// Used in sub_graph scenes
graphStatus ComputeGraph::RemoveInputNode(const NodePtr &node) {
  return impl_->RemoveInputNode(node);
}

graphStatus ComputeGraph::RemoveOutputNode(const NodePtr &node) {
  return impl_->RemoveOutputNode(node);
}

std::shared_ptr<ComputeGraph> ComputeGraph::AddSubGraph(const std::shared_ptr<ComputeGraph> sub_graph) {
  return impl_->AddSubGraph(sub_graph);
}

graphStatus ComputeGraph::RemoveSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph) {
  return impl_->RemoveSubGraph(sub_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::string &name, const std::shared_ptr<ComputeGraph> &subgraph) {
  return impl_->AddSubgraph(name, subgraph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::AddSubgraph(const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  return AddSubgraph(subgraph->GetName(), subgraph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(const std::string &name) {
  return impl_->RemoveSubgraph(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::RemoveSubgraph(
    const std::shared_ptr<ComputeGraph> &subgraph) {
  if (subgraph != nullptr) {
    RemoveSubgraph(subgraph->GetName());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<ComputeGraph> ComputeGraph::GetSubgraph(
    const std::string &name) const {
  return impl_->GetSubgraph(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<std::shared_ptr<ComputeGraph>>
ComputeGraph::GetAllSubgraphs() const {
  return impl_->GetAllSubgraphs();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetAllSubgraphs(
    const std::vector<std::shared_ptr<ComputeGraph>> &subgraphs) {
  return impl_->SetAllSubgraphs(subgraphs);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::vector<std::string>, std::vector<std::string>> &
ComputeGraph::GetShareParamLayer() const {
  return impl_->GetShareParamLayer();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetShareParamLayer(
    const std::map<std::vector<std::string>, std::vector<std::string>> params_share_map) {
  impl_->SetShareParamLayer(params_share_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetInputsOrder(
    const std::vector<std::string> &inputs_order) {
  impl_->SetInputsOrder(inputs_order);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphOutNodes(
    const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
  impl_->SetGraphOutNodes(out_nodes_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::AppendGraphOutNodes(
    const std::map<std::string, std::vector<int32_t>> out_nodes_map) {
  impl_->AppendGraphOutNodes(out_nodes_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<ComputeGraph> ComputeGraph::GetParentGraph() {
  return impl_->GetParentGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const ComputeGraph *ComputeGraph::GetParentGraphBarePtr() const {
  return impl_->GetParentGraphBarePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentGraph(
    const std::shared_ptr<ComputeGraph> &parent) {
  impl_->SetParentGraph(parent);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<Node> ComputeGraph::GetParentNode() {
  return impl_->GetParentNode();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const Node *ComputeGraph::GetParentNodeBarePtr() const {
  return impl_->GetParentNodeBarePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetNetOutputNode(
    const std::shared_ptr<Node> &netoutput_node) {
  return impl_->SetNetOutputNode(netoutput_node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<Node> ComputeGraph::GetOrUpdateNetOutputNode() {
  return impl_->GetOrUpdateNetOutputNode();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetParentNode(const std::shared_ptr<Node> &parent) {
  return impl_->SetParentNode(parent);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, std::vector<int32_t>> &
ComputeGraph::GetGraphOutNodes() const {
  return impl_->GetGraphOutNodes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetOrigGraph(const ComputeGraphPtr orig_graph) {
  impl_->SetOrigGraph(orig_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr ComputeGraph::GetOrigGraph(void) {
  return impl_->GetOrigGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetOutputSize(const uint32_t size) {
  impl_->SetOutputSize(size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetOutputSize() const {
  return impl_->GetOutputSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetInputSize(const uint32_t size) {
  impl_->SetInputSize(size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetInputSize() const {
  return impl_->GetInputSize();
}

// false: known shape  true: unknown shape
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GetGraphUnknownFlag() const {
  bool is_unknown = false;
  (void)AttrUtils::GetBool(this, ATTR_NAME_GRAPH_UNKNOWN_FLAG, is_unknown);
  return is_unknown;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphUnknownFlag(const bool flag) {
  (void)AttrUtils::SetBool(this, ATTR_NAME_GRAPH_UNKNOWN_FLAG, flag);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetNeedIteration(const bool need_iteration) {
  impl_->SetNeedIteration(need_iteration);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::GetNeedIteration() const {
  return impl_->GetNeedIteration();
}

///
/// @brief Update input-mapping
/// @param [in] input_mapping : index_of_cur_graph_node_input -> index_of_new_graph_node_input
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  return impl_->UpdateInputMapping(input_mapping);
}

///
/// @brief Update output-mapping
/// @param [in] output_mapping : index_of_cur_graph_node_output -> index_of_new_graph_node_output
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::UpdateOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  return impl_->UpdateOutputMapping(output_mapping);
}

graphStatus ComputeGraph::ReorderEventNodes() {
  return impl_->ReorderEventNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::InsertGraphEvents() {
  return impl_->InsertGraphEvents(shared_from_this());
}

graphStatus ComputeGraph::DFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                const std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                const std::vector<NodePtr> &stack, const bool reverse) {
  (void)map_in_edge_num;
  (void)stack;
  return impl_->DFSTopologicalSorting(node_vec, reverse, shared_from_this());
}

graphStatus ComputeGraph::BFSTopologicalSorting(std::vector<NodePtr> &node_vec,
                                                const std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                const std::deque<NodePtr> &stack) {
  (void)map_in_edge_num;
  (void)stack;
  return impl_->BFSTopologicalSorting(node_vec, false, shared_from_this());
}

graphStatus ComputeGraph::CollectBreadthOutNode(const NodePtr &node, std::map<NodePtr, uint32_t> &map_in_edge_num,
                                                std::map<std::string, NodePtr> &breadth_node_map) {
  return impl_->CollectBreadthOutNode(node, map_in_edge_num, breadth_node_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::TopologicalSorting(
    const std::function<bool(const NodePtr &, const NodePtr &)> comp) {
  return impl_->TopologicalSorting(comp);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::TopologicalSorting() {
  return impl_->TopologicalSorting(shared_from_this(), shared_from_this());
}

graphStatus ComputeGraph::TopologicalSortingGraph(const bool dfs_reverse) {
  return impl_->TopologicalSortingGraph(shared_from_this(), dfs_reverse);
}

graphStatus ComputeGraph::SortNodes(std::vector<NodePtr> &stack, std::map<NodePtr, uint32_t> &map_in_edge_num) {
  return impl_->SortNodes(stack, map_in_edge_num, shared_from_this());
}

size_t ComputeGraph::GetInEdgeSize(const NodePtr &node) const {
  return impl_->GetInEdgeSize(node);
}

size_t ComputeGraph::GetOutEdgeSize(const NodePtr &node) const {
  return impl_->GetOutEdgeSize(node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::IsValid() const {
  return impl_->IsValid();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::InValid() {
  impl_->InValid();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Dump() const {
  return impl_->Dump(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::Swap(ComputeGraph &graph) {
  this->AttrHolder::SwapBase(graph);
  impl_->Swap(*(graph.impl_));

  // Update Node owner.
  SetNodesOwner();
  graph.SetNodesOwner();

  // Update parent graph of 'TOP subgraph'. 'TOP subgraph' refers to the direct subgraph of the root graph.
  SetTopParentGraph();
  graph.SetTopParentGraph();
}

void ComputeGraph::SetNodesOwner() {
  return impl_->SetNodesOwner(shared_from_this());
}

void ComputeGraph::SetTopParentGraph() {
  return impl_->SetTopParentGraph(shared_from_this());
}

void ComputeGraph::EraseFromNodeList(const std::list<NodePtr>::iterator position) {
  impl_->EraseFromNodeList(position);
}

void ComputeGraph::ClearNodeList() {
  impl_->ClearNodeList();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::IsolateNode(const NodePtr &node) {
  return impl_->IsolateNode(node);
}

graphStatus ComputeGraph::RemoveExtraOutEdge(const NodePtr &node) const {
  return impl_->RemoveExtraOutEdge(node);
}

ProtoAttrMap &ComputeGraph::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMap &ComputeGraph::GetAttrMap() const {
  return impl_->GetAttrMap();
}

const std::map<OperatorImplPtr, NodePtr> &ComputeGraph::GetAllNodesInfo() const {
  return impl_->GetAllNodesInfo();
}

void ComputeGraph::SetUserDefOutput(const std::string &output_name) {
  impl_->SetUserDefOutput(output_name);
}

const std::string ComputeGraph::GetOutput() {
  return impl_->GetOutput();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphOpName(
    const std::map<uint32_t, std::string> &op_name_map) {
  impl_->SetGraphOpName(op_name_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<uint32_t, std::string> &ComputeGraph::GetGraphOpName()
    const {
  return impl_->GetGraphOpName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetAllNodesInfo(
    const std::map<OperatorImplPtr, NodePtr> &nodes) {
  impl_->SetAllNodesInfo(nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus ComputeGraph::SetGraphOutNodesInfo(
    const std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info, bool update_data_edge) {
  impl_->SetGraphOutNodesInfo(out_nodes_info);
  return CreateOrUpdateNetoutput(update_data_edge);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::CreateOrUpdateNetoutput(bool update_data_edge) {
  return impl_->CreateOrUpdateNetoutput(shared_from_this(), update_data_edge);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::AppendGraphOutNodesInfo(
    std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info) {
  impl_->AppendGraphOutNodesInfo(out_nodes_info);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::pair<NodePtr, int32_t>> &
ComputeGraph::GetGraphOutNodesInfo() const {
  return impl_->GetGraphOutNodesInfo();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::SetGraphTargetNodesInfo(const std::vector<NodePtr> &target_nodes_info) {
  impl_->SetGraphTargetNodesInfo(target_nodes_info);
  return CreateOrUpdateNetoutput();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<NodePtr> &ComputeGraph::GetGraphTargetNodesInfo()
    const {
  return impl_->GetGraphTargetNodesInfo();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetSessionID(const uint64_t session_id) {
  impl_->SetSessionID(session_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint64_t ComputeGraph::GetSessionID() const {
  return impl_->GetSessionID();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetGraphID(const uint32_t graph_id) {
  impl_->SetGraphID(graph_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t ComputeGraph::GetGraphID() const {
  return impl_->GetGraphID();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SaveDataFormat(const ge::Format data_format) {
  impl_->SaveDataFormat(data_format);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ge::Format ComputeGraph::GetDataFormat() const {
  return impl_->GetDataFormat();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ComputeGraph::IsSummaryGraph() const {
  return impl_->IsSummaryGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::SetSummaryFlag(const bool is_summary_graph) {
  impl_->SetSummaryFlag(is_summary_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void ComputeGraph::ReorderByNodeId() {
  impl_->ReorderByNodeId();
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
ComputeGraph::TopologicalSorting(TopoSortingMode topo_sorting_mode) {
  return impl_->TopologicalSortingGraph(shared_from_this(), topo_sorting_mode);
}
}  // namespace af

af::ComputeGraphPtr GeApiWrapper_MakeComputeGraphPtr(const char *graph_name) {
  return af::MakeShared<af::ComputeGraph>(graph_name);
}

size_t GeApiWrapper_GetAllNodesSize(const af::ComputeGraphPtr &graph_ptr) {
  if (graph_ptr == nullptr) {
    return 0;
  }
  return graph_ptr->GetAllNodesSize();
}
