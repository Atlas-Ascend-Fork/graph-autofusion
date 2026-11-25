/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule_group_partitioner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "schedule_utils.h"
#include "task_generator/node_partition_utils.h"
#include "util/mem_utils.h"
#include <queue>
#include <set>

using namespace af;

namespace optimize {
namespace {
using Element = std::pair<size_t, size_t>;
using MinHeap = std::priority_queue<Element, std::vector<Element>, std::greater<>>;

template <typename Heap>
size_t GetMinMergeCost(const Heap &heap) {
  auto temp = heap;
  const auto e1 = temp.top();
  temp.pop();
  const auto e2 = temp.top();
  return e1.first + e2.first;
}

bool IsAxisEqual(const af::AxisPtr &lhs, const af::AxisPtr &rhs) {
  if (lhs->id != rhs->id || lhs->name != rhs->name) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckEq(lhs->size, rhs->size) == af::TriBool::kTrue;
}

bool IsAxisVecEqual(const std::vector<af::AxisPtr> &lhs, const std::vector<af::AxisPtr> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!IsAxisEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

size_t FindRoot(std::unordered_map<size_t, size_t> &idx_to_parent, size_t x) {
  while (idx_to_parent[x] != x) {
    x = idx_to_parent[x] = idx_to_parent[idx_to_parent[x]];
  }
  return x;
}
}  // namespace

Status ScheduleGroupGraphPartitioner::PartitionByConnectivity(const ::ascir::ImplGraph &optimize_graph,
                                                              std::vector<::ascir::ImplGraph> &sub_optimize_graphs,
                                                              std::vector<AscNodePtr> node_order) {
  std::vector<AscNodePtr> output_nodes;
  size_t num_nodes = 0;
  for (const auto &node : optimize_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOutDataNodes().empty()) {
      output_nodes.emplace_back(node);
      GELOGD("Output node: %s[%s]", node->GetName().c_str(), node->GetType().c_str());
    }
    ++num_nodes;
  }
  if (node_order.empty()) {
    node_order = std::move(output_nodes);
    std::sort(node_order.begin(), node_order.end(), CompareByNodeId);
  }
  int32_t index = 0;
  std::set<af::NodePtr> visited;
  for (const auto &node : node_order) {
    const auto &root_node = node;
    if (visited.find(root_node) == visited.cend()) {
      ::ascir::ImplGraph sub_graph((optimize_graph.GetName() + "_" + std::to_string(index)).c_str());
      GE_ASSERT_TRUE(sub_graph.CopyAttrFrom(optimize_graph));
      GE_CHK_STATUS_RET(AddConnectedNodes(root_node, sub_graph, visited),
                        "Failed to add connected nodes, root_node = %s", root_node->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(sub_graph));
      sub_optimize_graphs.emplace_back(sub_graph);
      ++index;
    }
  }
  if (sub_optimize_graphs.size() > 1U) {  // 切图后会产生冗余轴信息
    for (auto &sub_optimize_graph : sub_optimize_graphs) {
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveUnusedAxes(sub_optimize_graph), "Failed to remove unused axes");
    }
    GE_CHK_STATUS_RET(SortSubGraphsByDependency(sub_optimize_graphs), "Failed to sort subgraphs by dependency");
  }

  // 重名检查之前先合并子图内同名的 GM 源副本: 沿用源名字的 Workspace 副本与源节点同名,
  // 两者落在同一子图时属于冗余读而非切分出错, 合并掉才不会被 CheckDuplicateWorkspaceNodes
  // 当成切分错误清空全部子图。真正的错误(写入者与读取者同子图)留不到这里之后: 带输入的
  // 写入者不在合并之列, 仍会被重名检查拦下。
  for (auto &sub_optimize_graph : sub_optimize_graphs) {
    GE_CHK_STATUS_RET(MergeDuplicatedCopyNodesInSubGraph(sub_optimize_graph),
                      "Failed to merge duplicated copy nodes, graph = %s", sub_optimize_graph.GetName().c_str());
  }

  CheckDuplicateWorkspaceNodes(sub_optimize_graphs);

  if (visited.size() != num_nodes) {
    for (const auto &node : optimize_graph.GetAllNodes()) {
      if (visited.find(node) == visited.cend()) {
        GELOGE(af::FAILED, "node: %s[%s] not visited", node->GetName().c_str(), node->GetType().c_str());
      }
    }
    GELOGE(af::FAILED, "number of visited nodes = %zu, num_nodes = %zu", visited.size(), num_nodes);
    return af::FAILED;
  }

  GELOGI("Partition success, subgraph number = %zu", sub_optimize_graphs.size());
  return af::SUCCESS;
}

void ScheduleGroupGraphPartitioner::CheckDuplicateWorkspaceNodes(std::vector<::ascir::ImplGraph> &sub_optimize_graphs) {
  for (size_t i = 0UL; i < sub_optimize_graphs.size(); ++i) {
    std::string dup_name;
    if (!HasDuplicateWorkspaceNode(sub_optimize_graphs[i], dup_name)) {
      continue;
    }
    GELOGW("Duplicate workspace node[%s] found in subgraph[%s], partition may be incorrect", dup_name.c_str(),
           sub_optimize_graphs[i].GetName().c_str());
    ::ascir::utils::DumpImplGraphs(sub_optimize_graphs, "Wrong_DuplicateWorkspace");
    sub_optimize_graphs.clear();
    return;
  }
}

bool ScheduleGroupGraphPartitioner::HasDuplicateWorkspaceNode(const ::ascir::ImplGraph &sub_graph,
                                                              std::string &dup_name) {
  std::set<std::string> workspace_names;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (node->GetType() != "Workspace") {
      continue;
    }
    const auto &name = node->GetName();
    if (!workspace_names.insert(name).second) {
      dup_name = name;
      return true;
    }
  }
  return false;
}

Status ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(const ::ascir::ImplGraph &optimize_graph,
                                                          bool &need_refresh) {
  const auto concat_node = ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Concat>(optimize_graph);
  const auto split_node = ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Split>(optimize_graph);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_node == nullptr && split_node == nullptr, af::SUCCESS,
                                 "neither Concat nor Split Node was found");
  need_refresh = true;
  return af::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::CollectConnectedNodes(const af::AscNodePtr &root_node,
                                                            std::set<af::NodePtr> &visited,
                                                            std::vector<af::AscNodePtr> &asc_nodes) {
  std::list<af::NodePtr> next_nodes{root_node};
  visited.emplace(root_node);
  while (!next_nodes.empty()) {
    auto node = next_nodes.front();
    next_nodes.pop_front();
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_CHECK_NOTNULL(asc_node);
    asc_nodes.emplace_back(std::move(asc_node));
    for (auto &in_node : node->GetInNodes()) {
      if (visited.find(in_node) == visited.cend()) {
        next_nodes.emplace_back(in_node);
        visited.emplace(in_node);
      }
    }
    for (auto &out_node : node->GetOutNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return af::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::AddConnectedNodes(const af::AscNodePtr &root_node, ::ascir::ImplGraph &sub_graph,
                                                        std::set<af::NodePtr> &all_visited) {
  GELOGI("AddConnectedNodes in, root_node = %s[%s]", root_node->GetName().c_str(), root_node->GetType().c_str());
  af::NodeCloneMap node_clone_map;
  std::set<af::NodePtr> visited;
  std::vector<af::AscNodePtr> asc_nodes;
  GE_ASSERT_SUCCESS(CollectConnectedNodes(root_node, visited, asc_nodes));
  std::sort(asc_nodes.begin(), asc_nodes.end(), CompareByNodeId);
  for (const auto &asc_node : asc_nodes) {
    const auto &op_desc = af::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetName(asc_node->GetName());
    af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = sub_graph.AddNode(op);
    node_clone_map[asc_node.get()] = dst_new_node;
    GE_ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                   "DoCopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
  }

  for (const auto &src_node : visited) {
    GE_CHK_STATUS_RET(af::GraphUtils::RelinkGraphEdges(src_node, node_clone_map), "RelinkGraphEdges failed");
  }
  all_visited.insert(visited.cbegin(), visited.cend());
  return af::SUCCESS;
}

bool ScheduleGroupGraphPartitioner::CompareByNodeId(const AscNodePtr &lhs, const AscNodePtr &rhs) {
  return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
}

bool ScheduleGroupGraphPartitioner::HasDataDependency(const ::ascir::ImplGraph &producer,
                                                      const ::ascir::ImplGraph &consumer) {
  std::set<std::string> producer_output_names;
  for (const auto &node : producer.GetAllNodes()) {
    if (node->GetOutDataNodes().empty()) {
      producer_output_names.insert(node->GetName());
    }
  }
  if (producer_output_names.empty()) {
    return false;
  }
  for (const auto &node : consumer.GetAllNodes()) {
    for (const auto &in_node : node->GetInDataNodes()) {
      if (producer_output_names.find(in_node->GetName()) != producer_output_names.end()) {
        return true;
      }
    }
  }
  return false;
}

Status ScheduleGroupGraphPartitioner::SortSubGraphsByDependency(std::vector<::ascir::ImplGraph> &sub_optimize_graphs) {
  const size_t n = sub_optimize_graphs.size();
  if (n <= 1U) {
    return af::SUCCESS;
  }
  std::vector<std::vector<bool>> dep_matrix(n, std::vector<bool>(n, false));
  std::vector<size_t> in_degree(n, 0U);
  for (size_t i = 0U; i < n; ++i) {
    for (size_t j = 0U; j < n; ++j) {
      if (i != j) {
        dep_matrix[i][j] = HasDataDependency(sub_optimize_graphs[i], sub_optimize_graphs[j]);
        if (dep_matrix[i][j]) {
          ++in_degree[j];
        }
      }
    }
  }
  std::vector<size_t> sorted;
  sorted.reserve(n);
  std::queue<size_t> zero_in_degree_queue;
  for (size_t i = 0U; i < n; ++i) {
    if (in_degree[i] == 0U) {
      zero_in_degree_queue.push(i);
    }
  }
  while (!zero_in_degree_queue.empty()) {
    size_t idx = zero_in_degree_queue.front();
    zero_in_degree_queue.pop();
    sorted.emplace_back(idx);
    for (size_t j = 0U; j < n; ++j) {
      if (dep_matrix[idx][j]) {
        --in_degree[j];
        if (in_degree[j] == 0U) {
          zero_in_degree_queue.push(j);
        }
      }
    }
  }
  if (sorted.size() != n) {
    GELOGW("Cycle detected in subgraph dependency, sorted[%zu] != total[%zu]", sorted.size(), n);
    ::ascir::utils::DumpImplGraphs(sub_optimize_graphs, "Wrong_CycleSubgraphWorkspace");
    sub_optimize_graphs.clear();
    return af::SUCCESS;
  }
  std::vector<::ascir::ImplGraph> reordered;
  reordered.reserve(n);
  for (const auto idx : sorted) {
    reordered.emplace_back(std::move(sub_optimize_graphs[idx]));
  }
  sub_optimize_graphs = std::move(reordered);
  return af::SUCCESS;
}

Status ScheduleGroupGraphPartitioner::RecordAxisSizes(const std::vector<af::Expression> &repeats,
                                                      const std::vector<int64_t> &axis_ids,
                                                      std::map<af::AxisId, af::Expression> &axis_id_to_size) {
  for (size_t j = 0UL; j < repeats.size(); ++j) {
    if (af::SymbolicUtils::StaticCheckEq(repeats[j], ops::One) != af::TriBool::kTrue) {
      const auto axis_id = axis_ids[j];
      if (!axis_id_to_size.emplace(axis_id, repeats[j]).second) {
        GE_LOGW_IF(af::SymbolicUtils::StaticCheckEq(repeats[j], axis_id_to_size[axis_id]) != TriBool::kTrue,
                   "inconsistent axis size, id = %ld, size0 = %s, size1 = %s", axis_id,
                   SymbolicUtils::ToString(repeats[j]).c_str(),
                   SymbolicUtils::ToString(axis_id_to_size[axis_id]).c_str());
      }
    }
  }
  return SUCCESS;
}

Status ScheduleGroupGraphPartitioner::RefreshAxisSize(const ::ascir::ImplGraph &sub_graph) {
  GELOGI("RefreshAxisSize start, graph_name = %s", sub_graph.GetName().c_str());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(sub_graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  std::map<af::AxisId, af::Expression> axis_id_to_size;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto &repeats = node->outputs[i].attr.repeats;
      const auto &axis_ids = node->outputs[i].attr.axis;
      GE_ASSERT_TRUE(repeats.size() == axis_ids.size(), "node: %s:%u repeats.size() = %zu, but axis_ids.size() = %zu",
                     node->GetNamePtr(), i, repeats.size(), axis_ids.size());
      GE_ASSERT_SUCCESS(RecordAxisSizes(repeats, axis_ids, axis_id_to_size));
    }
  }
  for (const auto &axis : graph_attr->axis) {
    GE_ASSERT_NOTNULL(axis);
    const auto it = axis_id_to_size.find(axis->id);
    if ((it != axis_id_to_size.cend()) && (SymbolicUtils::StaticCheckEq(axis->size, it->second) != TriBool::kTrue)) {
      GELOGI("update axis, id = %ld, name = %s, from %s to %s", axis->id, axis->name.c_str(),
             SymbolicUtils::ToString(axis->size).c_str(), SymbolicUtils::ToString(it->second).c_str());
      axis->size = it->second;
    }
  }
  GELOGI("RefreshAxisSize end, graph_name = %s", sub_graph.GetName().c_str());
  return SUCCESS;
}

bool ScheduleGroupGraphPartitioner::IsSimpleComputeGraph(const ::ascir::ImplGraph &graph, size_t &node_count) {
  node_count = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    if (!ScheduleUtils::IsCompute(node)) {
      continue;
    }
    ++node_count;
    const auto compute_type = node->attr.api.compute_type;
    if (compute_type != af::ComputeType::kComputeLoad && compute_type != af::ComputeType::kComputeStore &&
        compute_type != af::ComputeType::kComputeElewise && compute_type != af::ComputeType::kComputeBroadcast) {
      return false;
    }
  }
  return true;
}

std::vector<MergeableGraphs> ScheduleGroupGraphPartitioner::FindMergeableGraphs(
    const std::vector<::ascir::ImplGraph> &grouped_graphs) {
  std::vector<MergeableGraphs> mergeable_groups;
  std::vector<bool> merged(grouped_graphs.size(), false);
  std::vector<size_t> node_counts(grouped_graphs.size());

  for (size_t i = 0; i < grouped_graphs.size(); ++i) {
    if (merged[i] || !IsSimpleComputeGraph(grouped_graphs[i], node_counts[i])) {
      continue;
    }

    MergeableGraphs group;
    group.graph_indices.push_back(i);
    group.node_counts.push_back(node_counts[i]);

    const auto graph_attr_i = AscGraphUtils::GetComputeGraph(grouped_graphs[i])->GetOrCreateAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(graph_attr_i);

    for (size_t j = i + 1; j < grouped_graphs.size(); ++j) {
      if (merged[j] || !IsSimpleComputeGraph(grouped_graphs[j], node_counts[j])) {
        continue;
      }

      const auto graph_attr_j =
          AscGraphUtils::GetComputeGraph(grouped_graphs[j])->GetOrCreateAttrsGroup<AscGraphAttr>();
      GE_ASSERT_NOTNULL(graph_attr_j);

      if (IsAxisVecEqual(graph_attr_i->axis, graph_attr_j->axis)) {
        group.graph_indices.push_back(j);
        group.node_counts.push_back(node_counts[j]);
        merged[j] = true;
      }
    }

    if (group.graph_indices.size() > 1) {
      mergeable_groups.push_back(std::move(group));
    }
  }
  for (const auto &mergeable_group : mergeable_groups) {
    GELOGD("mergeable_group: %s, node_num = %s", ToString(mergeable_group.graph_indices).c_str(),
           ToString(mergeable_group.node_counts).c_str());
  }

  return mergeable_groups;
}

Status ScheduleGroupGraphPartitioner::MergeGraphs(::ascir::ImplGraph &dst,
                                                  const std::vector<::ascir::ImplGraph> &grouped_graphs,
                                                  const std::vector<size_t> &group) {
  for (size_t idx : group) {
    const auto &src = grouped_graphs[idx];
    GELOGD("MergeGraphs: merging %s into %s", src.GetName().c_str(), dst.GetName().c_str());

    af::NodeCloneMap node_clone_map;
    for (const auto &node : src.GetAllNodes()) {
      const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      GE_CHECK_NOTNULL(asc_node);

      const auto &op_desc = af::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
      GE_CHECK_NOTNULL(op_desc);
      af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
      auto dst_new_node = dst.AddNode(op);
      GE_ASSERT_TRUE(AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                     "CopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
      node_clone_map[asc_node.get()] = std::move(dst_new_node);
    }

    for (const auto &node : src.GetAllNodes()) {
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      GE_CHK_STATUS_RET(af::GraphUtils::RelinkGraphEdges(asc_node, node_clone_map), "RelinkGraphEdges failed");
    }
  }
  GELOGI("MergeGraphs: merged %zu graph(s) into %s", group.size(), dst.GetName().c_str());

  GE_CHK_STATUS_RET(ScheduleUtils::TopologicalSorting(dst), "Failed to sort merged graph");
  return af::SUCCESS;
}

MergePlan ScheduleGroupGraphPartitioner::ResolveMergePlan(const std::vector<MergeableGraphs> &mergeable_groups,
                                                          size_t reductions_needed) {
  GELOGD("ResolveMergePlan: reductions_needed=%zu, mergeable_groups=%zu", reductions_needed, mergeable_groups.size());

  std::unordered_map<size_t, size_t> idx_to_parent;
  for (const auto &group : mergeable_groups) {
    for (size_t idx : group.graph_indices) {
      idx_to_parent[idx] = idx;
    }
  }

  std::vector<MinHeap> heaps(mergeable_groups.size());
  std::set<Element> global_heap;

  for (size_t g = 0; g < mergeable_groups.size(); ++g) {
    for (size_t i = 0; i < mergeable_groups[g].graph_indices.size(); ++i) {
      heaps[g].emplace(mergeable_groups[g].node_counts[i], mergeable_groups[g].graph_indices[i]);
    }
    global_heap.insert({GetMinMergeCost(heaps[g]), g});
  }

  size_t reductions = reductions_needed;
  while (reductions > 0 && !global_heap.empty()) {
    const auto g = global_heap.begin()->second;
    global_heap.erase(global_heap.begin());

    auto [cost1, idx1] = heaps[g].top();
    heaps[g].pop();
    auto [cost2, idx2] = heaps[g].top();
    heaps[g].pop();

    GELOGD("ResolveMergePlan: merge graph[%zu](%zu) and graph[%zu](%zu)", idx1, cost1, idx2, cost2);

    idx_to_parent[FindRoot(idx_to_parent, idx1)] = FindRoot(idx_to_parent, idx2);
    heaps[g].emplace(cost1 + cost2, FindRoot(idx_to_parent, idx2));

    if (heaps[g].size() > 1) {
      global_heap.insert({GetMinMergeCost(heaps[g]), g});
    }
    --reductions;
  }

  std::unordered_map<size_t, std::vector<size_t>> clusters;
  for (const auto &idx_and_parent : idx_to_parent) {
    clusters[FindRoot(idx_to_parent, idx_and_parent.first)].push_back(idx_and_parent.first);
  }

  MergePlan plan;
  for (auto &root_and_members : clusters) {
    if (root_and_members.second.size() > 1) {
      std::sort(root_and_members.second.begin(), root_and_members.second.end());
      plan.merge_groups.push_back(std::move(root_and_members.second));
    }
  }

  GELOGD("ResolveMergePlan: %zu merge groups, %zu reductions remaining", plan.merge_groups.size(), reductions);
  return plan;
}

Status ScheduleGroupGraphPartitioner::ReduceGraphCount(std::vector<::ascir::ImplGraph> &grouped_graphs,
                                                       size_t target_count) {
  if (grouped_graphs.size() <= target_count) {
    return af::SUCCESS;
  }

  GELOGI("ReduceGraphCount: reducing from %zu to %zu graphs", grouped_graphs.size(), target_count);

  const auto mergeable_groups = FindMergeableGraphs(grouped_graphs);
  if (mergeable_groups.empty()) {
    GELOGI("No mergeable graphs found, cannot reduce");
    return af::SUCCESS;
  }

  const auto plan = ResolveMergePlan(mergeable_groups, grouped_graphs.size() - target_count);
  if (plan.merge_groups.empty()) {
    GELOGI("No merge plan generated, cannot reduce");
    return af::SUCCESS;
  }

  GELOGD("ReduceGraphCount: merge plan has %zu groups", plan.merge_groups.size());
  for (size_t i = 0; i < plan.merge_groups.size(); ++i) {
    const auto &group = plan.merge_groups[i];
    GELOGD("  group[%zu]: sources=[", i);
    for (size_t j = 0; j < group.size(); ++j) {
      GELOGD("    graph[%zu]%s", group[j], (j < group.size() - 1) ? "," : "");
    }
    GELOGD("  ]");
  }

  // For each merge group, create a new empty graph and merge all sources into it
  for (const auto &group : plan.merge_groups) {
    // Create new empty graph with attributes from first source
    ::ascir::ImplGraph new_graph(("merged_" + std::to_string(group[0])).c_str());
    new_graph.CopyAttrFrom(grouped_graphs[group[0]]);

    GE_CHK_STATUS_RET(MergeGraphs(new_graph, grouped_graphs, group), "Failed to merge graphs");

    // Add new graph to grouped_graphs
    grouped_graphs.push_back(std::move(new_graph));
  }

  // Erase all source graphs (in reverse order to maintain index validity)
  std::set<size_t> to_erase;
  for (const auto &group : plan.merge_groups) {
    for (size_t idx : group) {
      to_erase.insert(idx);
    }
  }

  for (auto it = to_erase.rbegin(); it != to_erase.rend(); ++it) {
    GELOGD("  erasing graph[%zu]", *it);
    grouped_graphs.erase(grouped_graphs.begin() + static_cast<int32_t>(*it));
  }

  GELOGI("ReduceGraphCount: final count is %zu graphs", grouped_graphs.size());
  return af::SUCCESS;
}
}  // namespace optimize
