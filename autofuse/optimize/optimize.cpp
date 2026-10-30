/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize.h"
#include <utility>
#include <queue>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "autoschedule/autoschedule.h"
#include "graph_properties_cache.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "fused_graph/fused_graph_modifier.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "task_generator/schedule_task_generator.h"
#include "fusion/autofuse_attrs.h"
#include "buffer_allocate/buf_que_allocator.h"
#include "ascgraph_info_complete.h"
#include "indirect_load_utils.h"
#include "schedule_utils.h"
#include "common_utils.h"
#include "v35/optimize/partition/vector_func_partitioner.h"
#include "node_utils.h"
#include "optimize/graph_pass/pass_runner_handler.h"
#include "optimize/graph_pass/cast_reorder_pass.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "optimize/graph_completeness/dtype_consistency.h"
#include "pre_process/pre_process.h"
#include "static_ub_template_filter.h"
#include "common/l2_cache_hint_manager.h"

using namespace ascir;
using namespace optimize;
using namespace af::ascir_op;
using namespace af::ops;

namespace {
const char *const kAttrAscGraph = "ascgraph";
constexpr int64_t kInvalidNodeId = -1;
constexpr size_t kMaxGraphNameLength = 60UL;  // 截断后的最大graph name长度

// 截断 graph name，保留前 kMaxGraphNameLength 个字符
std::string TruncateGraphName(const std::string &name) {
  if (name.length() <= kMaxGraphNameLength) {
    return name;
  }
  return name.substr(0, kMaxGraphNameLength);
}

// graph_name添加索引格式为: original_name_S{result_idx}G{group_idx}C{impl_idx}
std::string GenerateIndexedGraphName(const std::string &original_name, size_t result_idx, size_t group_idx,
                                     size_t impl_idx) {
  std::ostringstream oss;
  oss << original_name << "_S" << result_idx << "G" << group_idx << "C" << impl_idx;
  return oss.str();
}

Status FinalizeIndexedGraphs(std::vector<ascir::ScheduledResult> &scheduled_results) {
  for (size_t result_idx = 0UL; result_idx < scheduled_results.size(); ++result_idx) {
    auto &scheduled_result = scheduled_results[result_idx];
    for (size_t group_idx = 0UL; group_idx < scheduled_result.schedule_groups.size(); ++group_idx) {
      auto &schedule_group = scheduled_result.schedule_groups[group_idx];
      for (size_t impl_idx = 0UL; impl_idx < schedule_group.impl_graphs.size(); ++impl_idx) {
        auto &impl_graph = schedule_group.impl_graphs[impl_idx];
        std::string old_name = impl_graph.GetName();
        std::string new_name = GenerateIndexedGraphName(old_name, result_idx, group_idx, impl_idx);
        // 修改 impl_graph 的 name
        auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
        GE_ASSERT_NOTNULL(compute_graph);
        compute_graph->SetName(new_name);
        GELOGD("Rename graph: [%s] -> [%s]", old_name.c_str(), new_name.c_str());

        // 如果有 score func，直接在 map 中更新 key（只查找一次）
        auto node = schedule_group.graph_name_to_score_funcs.extract(old_name);
        if (!node.empty()) {
          node.key() = std::move(new_name);
          const auto inserted = schedule_group.graph_name_to_score_funcs.insert(std::move(node));
          GELOGD("Update score func key: [%s] -> [%s]", old_name.c_str(), inserted.position->first.c_str());
        }
      }
    }
  }
  return af::SUCCESS;
}

bool IsAxisContinuous(const af::AscGraph &graph, const int64_t pre_id_idx, const int64_t post_id_idx) {
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    auto pre_id = node->attr.sched.axis[pre_id_idx];
    auto post_id = node->attr.sched.axis[post_id_idx];
    for (auto &out_tensor : node->outputs()) {
      auto pre_iter = std::find(out_tensor->attr.axis.begin(), out_tensor->attr.axis.end(), pre_id);
      auto post_iter = std::find(out_tensor->attr.axis.begin(), out_tensor->attr.axis.end(), post_id);
      if ((pre_iter == out_tensor->attr.axis.end()) || (post_iter == out_tensor->attr.axis.end())) {
        return false;
      }
      auto pre_idx = std::distance(out_tensor->attr.axis.begin(), pre_iter);
      auto post_idx = std::distance(out_tensor->attr.axis.begin(), post_iter);
      auto post_stride = out_tensor->attr.strides[post_idx] * out_tensor->attr.repeats[post_idx];
      if ((pre_idx + 1 != post_idx) ||
          (af::SymbolicUtils::StaticCheckEq(out_tensor->attr.strides[pre_idx], post_stride) != af::TriBool::kTrue)) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<int64_t>> MergeContinuousPairs(const std::vector<std::pair<int64_t, int64_t>> &potential_axis) {
  std::vector<std::vector<int64_t>> continuous_ids;
  if (potential_axis.empty()) {
    return continuous_ids;
  }

  std::vector<int64_t> current_chain;
  current_chain.push_back(potential_axis[0].first);
  current_chain.push_back(potential_axis[0].second);

  for (size_t i = 1UL; i < potential_axis.size(); ++i) {
    const auto &cur_pair = potential_axis[i];
    if (current_chain.back() == cur_pair.first) {
      current_chain.push_back(cur_pair.second);
    } else {
      continuous_ids.push_back(current_chain);
      current_chain.clear();
      current_chain.push_back(cur_pair.first);
      current_chain.push_back(cur_pair.second);
    }
  }
  continuous_ids.push_back(current_chain);

  return continuous_ids;
}

std::unordered_set<size_t> IdentifyZeroStrideAxisIndices(const ascir::ImplGraph &owner_graph) {
  std::vector<bool> is_zero_stride_axis;
  bool include_reduce = false;
  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!ScheduleUtils::IsBuffer(node) && is_zero_stride_axis.empty()) {
      is_zero_stride_axis.resize(node->attr.sched.axis.size(), true);
    }
    if (ScheduleUtils::IsReduce(node)) {
      include_reduce = true;
    }
  }

  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    // 当前节点包含_keep_first_axis属性且为true，或者图上包含reduce节点时，需要保留首轴
    bool keep_first_axis = false;
    (void)af::AttrUtils::GetBool(node->GetOpDesc(), "_keep_first_axis", keep_first_axis);
    keep_first_axis = keep_first_axis || include_reduce;

    const auto &loop_axes = node->attr.sched.axis;
    for (size_t loop_idx = 0UL; loop_idx < loop_axes.size(); ++loop_idx) {
      bool has_non_zero_stride = false;
      for (const auto &output : node->outputs()) {
        auto iter = std::find(output->attr.axis.begin(), output->attr.axis.end(), loop_axes[loop_idx]);
        GE_ASSERT_TRUE(iter != output->attr.axis.end());

        auto axis_index = static_cast<size_t>(std::distance(output->attr.axis.begin(), iter));
        GE_ASSERT_TRUE(axis_index < output->attr.strides.size(), "axis index [%zu] is out of range, max_size:[%zu].",
                       axis_index, output->attr.strides.size());

        if (af::SymbolicUtils::StaticCheckEq(output->attr.strides[axis_index], af::sym::kSymbolZero) !=
                af::TriBool::kTrue ||
            (keep_first_axis && (axis_index == 0))) {
          has_non_zero_stride = true;
          break;
        }
      }

      if (has_non_zero_stride) {
        is_zero_stride_axis[loop_idx] = false;
      }
    }
  }

  std::unordered_set<size_t> zero_stride_axis_indices;
  for (size_t i = 0UL; i < is_zero_stride_axis.size(); ++i) {
    if (is_zero_stride_axis[i]) {
      zero_stride_axis_indices.emplace(i);
    }
  }
  // 全0场景,不需要删除
  if (zero_stride_axis_indices.size() == is_zero_stride_axis.size()) {
    return {};
  }

  return zero_stride_axis_indices;
}

Status GetDirectFatherNode(const af::AscGraph &impl_graph, std::map<int64_t, int64_t> &dir_father_nodes) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    for (const auto &out_node : node->GetOutAllNodes()) {
      GE_CHECK_NOTNULL(out_node);
      GE_CHECK_NOTNULL(out_node->GetOpDesc());
      // 若输出节点为store或output，则认为其父节点即自己，否则所有节点的公共父节点都会搜索到store->output节点，保证并查集能够搜索到两个节点的公共计算节点。
      if (af::ops::IsOps<Store>(out_node) || af::ops::IsOps<Output>(out_node)) {
        dir_father_nodes.insert(std::make_pair(node->GetOpDesc()->GetId(), node->GetOpDesc()->GetId()));
        continue;
      }
      dir_father_nodes.insert(std::make_pair(node->GetOpDesc()->GetId(), out_node->GetOpDesc()->GetId()));
    }
  }
  return af::SUCCESS;
}

int64_t FindRoot(std::map<int64_t, int64_t> &dir_father_nodes, const int64_t node_id) {
  if (node_id == dir_father_nodes[node_id]) {
    return node_id;
  }
  return dir_father_nodes[node_id] = FindRoot(dir_father_nodes, dir_father_nodes[node_id]);
}

bool HasSameComputeNode(std::map<int64_t, int64_t> &dir_father_nodes, int64_t node0_id, int64_t node1_id) {
  node0_id = FindRoot(dir_father_nodes, node0_id);
  node1_id = FindRoot(dir_father_nodes, node1_id);
  return node0_id == node1_id;
}

Status GetMapFromQueId2LoadId(const af::AscGraph &impl_graph,
                              std::map<int64_t, vector<int64_t>> &loads_with_same_queid) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (!af::ops::IsOps<Load>(node)) {
      continue;
    }
    if (loads_with_same_queid.find(node->outputs[0].attr.que.id) != loads_with_same_queid.end()) {
      loads_with_same_queid[node->outputs[0].attr.que.id].emplace_back(node->GetOpDesc()->GetId());
      continue;
    }
    vector<int64_t> temp_load_ids = {node->GetOpDesc()->GetId()};
    loads_with_same_queid.insert(std::make_pair(node->outputs[0].attr.que.id, temp_load_ids));
  }
  return af::SUCCESS;
}

Status SearchNodesNeedForward(const af::AscGraph &impl_graph, std::map<int64_t, int64_t> &need_forward_nodes_id,
                              int64_t &first_load_id) {
  std::map<int64_t, int64_t> dir_father_nodes;  // 以节点id形式存储所有节点的直接父节点
  GE_ASSERT_SUCCESS(GetDirectFatherNode(impl_graph, dir_father_nodes));
  constexpr size_t load_thresh = 2UL;    // 能够参与调序的load节点个数阈值
  size_t num_of_load_need_adjust = 0UL;  // 用于记录需要调序的load节点个数，判断是否达到阈值
  size_t index_data_and_load = 0UL;
  std::map<int64_t, vector<int64_t>> loads_with_same_queid;  // 获取TQue的共复用情况
  GE_ASSERT_SUCCESS(GetMapFromQueId2LoadId(impl_graph, loads_with_same_queid));
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (num_of_load_need_adjust >= load_thresh) {
      GELOGD("The number of loads that need to be brought forward is %zu, which reaches threshold %zu.",
             num_of_load_need_adjust, load_thresh);
      break;
    }
    if (first_load_id == kInvalidNodeId && af::ops::IsOps<Load>(node)) {
      first_load_id = node->GetOpDesc()->GetId();
      num_of_load_need_adjust++;
      continue;
    }
    // 记录需要调序的input(data/workspace)和load节点：1. 非首load 2. 未被共复用 3. 和首load有公共计算节点 4.
    // 输出节点为非多输入节点
    if (first_load_id != kInvalidNodeId && af::ops::IsOps<Load>(node) && node->GetOpDesc()->GetId() > first_load_id &&
        loads_with_same_queid[node->outputs[0].attr.que.id].size() == 1UL &&
        HasSameComputeNode(dir_father_nodes, node->GetOpDesc()->GetId(), first_load_id)) {
      for (const auto &in_node : node->GetInAllNodes()) {
        GE_CHECK_NOTNULL(in_node);
        GE_CHECK_NOTNULL(in_node->GetOpDesc());
        if (ScheduleUtils::IsBuffer(std::dynamic_pointer_cast<af::AscNode>(in_node)) &&
            in_node->GetOpDesc()->GetId() > first_load_id) {
          GELOGD("Input node %s is after first load, needs to be advanced.", in_node->GetNamePtr());
          need_forward_nodes_id.insert(std::make_pair(in_node->GetOpDesc()->GetId(), index_data_and_load++));
        }
      }
      if (ScheduleUtils::IsOutNodeWithMultiInputs(node)) {
        continue;
      }
      GELOGD("Node %s needs to be advanced.", node->GetNamePtr());
      need_forward_nodes_id.insert(std::make_pair(node->GetOpDesc()->GetId(), index_data_and_load++));
      num_of_load_need_adjust++;
    }
  }
  return af::SUCCESS;
}

Status DoSeqAdjustment(const af::AscGraph &impl_graph, const std::map<int64_t, int64_t> &need_forward_nodes_id,
                       const int64_t &first_load_id) {
  const size_t need_forward_nodes_num = need_forward_nodes_id.size();
  if (need_forward_nodes_num <= 0UL || first_load_id == kInvalidNodeId) {
    return af::SUCCESS;
  }
  int64_t start_index_other_nodes = first_load_id + static_cast<int64_t>(need_forward_nodes_num) + 1;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetId() <= first_load_id) {
      continue;
    }
    if (node->GetOpDesc()->GetId() > need_forward_nodes_id.rbegin()->first) {
      GELOGD("No need to adjust node after node %" PRId64 ".", need_forward_nodes_id.rbegin()->first);
      break;
    }
    if (need_forward_nodes_id.find(node->GetOpDesc()->GetId()) != need_forward_nodes_id.end()) {
      GELOGD("Move node with id %" PRId64 " and name %s forward.", node->GetOpDesc()->GetId(), node->GetNamePtr());
      node->GetOpDesc()->SetId(first_load_id + need_forward_nodes_id.at(node->GetOpDesc()->GetId()) + 1);
      continue;
    }
    node->GetOpDesc()->SetId(start_index_other_nodes++);
  }
  return af::SUCCESS;
}

std::string RegisterScoreFuncInScheduleGroup(const autoschedule::AutoScheduleOutput &schedule_output,
                                             ScheduleGroup &schedule_group, const bool should_skip_registry = true) {
  if (schedule_output.score_func.empty()) {
    return "";
  }

  const std::string graph_name = schedule_output.scheduled_graph.GetName();

  if (should_skip_registry) {
    GELOGD("Not a valid case, skip register template score func of graph [%s].", graph_name.c_str());
    return "";
  }

  schedule_group.graph_name_to_score_funcs[graph_name] = schedule_output.score_func;
  GELOGD("The score func of template [%s] is [%s].", graph_name.c_str(), schedule_output.score_func.c_str());
  return graph_name;
}

af::Status CopyImplGraphs(const std::vector<autoschedule::AutoScheduleOutput> &schedule_outputs,
                          std::vector<ascir::ScheduledResult> &scheduled_results_cur) {
  for (size_t i = 0UL; i < scheduled_results_cur.size(); ++i) {
    auto &cur_result = scheduled_results_cur[i];
    ScheduleGroup cur_group;
    cur_group.impl_graphs.reserve(schedule_outputs.size());
    for (const auto &result : schedule_outputs) {
      ascir::ImplGraph copied_graph(result.scheduled_graph.GetName().c_str());
      GE_ASSERT_TRUE(copied_graph.CopyFrom(result.scheduled_graph));
      cur_group.impl_graphs.push_back(std::move(copied_graph));
      RegisterScoreFuncInScheduleGroup(result, cur_group);
    }
    cur_result.schedule_groups.emplace_back(std::move(cur_group));
  }
  return af::SUCCESS;
}

bool CanDoReMergeAxis(const af::AscGraph &impl_graph, ReduceTemplateType reduce_type) {
  GraphPropertiesCache cache(impl_graph);
  // 如果包含Gather、Cube类型节点，则不能重新合并轴
  // RCore类型中包含Reduce的图依赖固定的二维调度轴，不能重新合并轴
  const bool is_rcore_reduce = (reduce_type == ReduceTemplateType::kRCore) && cache.HasReduce();
  return !is_rcore_reduce && !cache.HasGather() && !cache.HasCube();
}

void FilterComplexTilingDataScoreFuncs(std::vector<::ascir::ScheduledResult> &scheduled_results,
                                       const std::set<std::string> &scored_graph_names) {
  // 如果没有需要清理的图名，直接返回
  if (scored_graph_names.empty()) {
    return;
  }
  // 只为单result单group场景注册打分函数
  if (scheduled_results.size() == 1UL && scheduled_results[0].schedule_groups.size() == 1UL) {
    GELOGD("Autoschedule score func for graph simple tiling data: %zu results, %zu groups", scheduled_results.size(),
           scheduled_results[0].schedule_groups.size());
    return;
  }

  for (auto &result : scheduled_results) {
    for (auto &group : result.schedule_groups) {
      // 遍历 score_funcs，检查是否在 scored_graph_names 中
      auto it = group.graph_name_to_score_funcs.begin();
      while (it != group.graph_name_to_score_funcs.end()) {
        if (scored_graph_names.count(it->first) == 0) {
          ++it;
          continue;
        }
        // 只删除在 scored_graph_names 中的打分函数
        GELOGD("Clear autoschedule score func for graph [%s] in complex tiling data: %zu results, %zu groups",
               it->first.c_str(), scheduled_results.size(), result.schedule_groups.size());
        it = group.graph_name_to_score_funcs.erase(it);
      }
    }
  }
}

Status CheckGraphValidity(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (size_t i = 0UL; i < node->outputs().size(); ++i) {
      const auto &tensor = node->outputs[i];
      const auto &axis = tensor.attr.axis;
      const auto &repeats = tensor.attr.repeats;
      const auto &strides = tensor.attr.strides;
      GE_ASSERT_TRUE(axis.size() == repeats.size() && repeats.size() == strides.size(),
                     "Output tensor[%zu] of node [%s] has mismatched sizes: axis=%zu, repeat=%zu, stride=%zu.", i,
                     node->GetNamePtr(), axis.size(), repeats.size(), strides.size());
    }
  }
  return af::SUCCESS;
}

bool IsRemovableDanglingNode(const af::NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  // 参照PassUtils::PruneGraph中的实现，保留输入输出节点
  if (IsOps<Data>(node) || IsOps<ScalarData>(node) || IsOps<Output>(node) || IsOps<Workspace>(node)) {
    return false;
  }
  return node->GetOutDataNodesSize() == 0U && node->GetOutControlNodesSize() == 0U;
}

bool IsDanglingInputNode(const af::NodePtr &node) {
  if (node == nullptr || (!IsOps<Data>(node) && !IsOps<ScalarData>(node))) {
    return false;
  }
  return node->GetOutDataNodesSize() == 0U && node->GetOutControlNodesSize() == 0U;
}

Status GetOutputNodes(const af::ComputeGraphPtr &graph, std::vector<af::NodePtr> &output_nodes) {
  GE_ASSERT_NOTNULL(graph);
  output_nodes.clear();
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (IsOps<Output>(node) || (IsOps<Workspace>(node) && node->GetOutDataNodesSize() == 0U)) {
      output_nodes.push_back(node);
    }
  }
  return af::SUCCESS;
}

Status LinkDanglingInputNodesToOutput(const af::ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::vector<af::NodePtr> output_nodes;
  GE_CHK_STATUS_RET(GetOutputNodes(graph, output_nodes), "Get output nodes failed, graph:[%s].",
                    graph->GetName().c_str());
  if (output_nodes.empty()) {
    GELOGW("Graph [%s] does not contain valid output nodes, skip linking dangling input nodes.",
           graph->GetName().c_str());
    return af::SUCCESS;
  }
  const auto &first_output_node = output_nodes.front();
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsDanglingInputNode(node)) {
      continue;
    }
    // Data/ScalarData保留输入签名，通过控制边挂到输出节点避免完全悬空
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::AddEdge(node->GetOutControlAnchor(), first_output_node->GetInControlAnchor()));
    GELOGD("Add extra control edge between input node[%s] and output node[%s] in graph[%s].", node->GetNamePtr(),
           first_output_node->GetNamePtr(), graph->GetName().c_str());
  }
  return af::SUCCESS;
}

Status RemoveDanglingNodes(const af::ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  size_t total_removed = 0UL;
  bool has_removed = false;
  do {
    has_removed = false;
    std::vector<af::NodePtr> nodes_to_remove;
    // 先收集再删除，避免遍历过程中修改图节点容器
    for (const auto &node : graph->GetDirectNode()) {
      GE_ASSERT_NOTNULL(node);
      if (IsRemovableDanglingNode(node)) {
        nodes_to_remove.push_back(node);
      }
    }
    for (const auto &node : nodes_to_remove) {
      GELOGW("Remove dangling node [%s], type [%s], graph [%s].", node->GetNamePtr(), node->GetTypePtr(),
             graph->GetName().c_str());
      GE_ASSERT_GRAPH_SUCCESS(graph->RemoveNode(node));
      ++total_removed;
      has_removed = true;
    }
    // 删除末端节点后，上游节点可能变成新的无消费者节点
  } while (has_removed);
  GELOGI("Remove dangling nodes end, graph [%s], removed node num [%zu].", graph->GetName().c_str(), total_removed);
  return af::SUCCESS;
}

Status RemoveDanglingNodes(af::AscGraph &graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  GE_CHK_STATUS_RET(RemoveDanglingNodes(compute_graph), "Remove dangling nodes failed, graph:[%s].",
                    graph.GetName().c_str());
  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() == "Unsupported") {
      const auto *error_msg = af::AttrUtils::GetStr(node->GetOpDescBarePtr(), "error_msg");
      GELOGE(af::FAILED, "Graph [%s] contains UnSupported node [%s]: %s", graph.GetName().c_str(), node->GetNamePtr(),
             error_msg != nullptr ? error_msg->c_str() : "unknown reason");
      return af::FAILED;
    }
  }
  GE_CHK_STATUS_RET(LinkDanglingInputNodesToOutput(compute_graph), "Link dangling input nodes failed, graph:[%s].",
                    graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return af::SUCCESS;
}
}  // namespace

Status Optimizer::ExpandReduceFirstStageResults(std::vector<autoschedule::AutoScheduleOutput> &schedule_outputs,
                                                std::vector<ascir::ScheduledResult> &scheduled_results_cur,
                                                size_t index, ScheduleTask &schedule_task) const {
  std::vector<ascir::ScheduledResult> scheduled_results_tmp;
  for (auto &schedule_output : schedule_outputs) {
    ScheduleGroup schedule_group = {{schedule_output.scheduled_graph}, {}};
    for (auto &d : scheduled_results_cur) {
      for (auto &existing_group : d.schedule_groups) {
        ScheduleGroup copied_group;
        copied_group.impl_graphs.reserve(existing_group.impl_graphs.size());
        for (const auto &impl_graph : existing_group.impl_graphs) {
          ascir::ImplGraph copied(impl_graph.GetName().c_str());
          GE_ASSERT_TRUE(copied.CopyFrom(impl_graph));
          copied_group.impl_graphs.push_back(std::move(copied));
        }
        copied_group.graph_name_to_score_funcs = existing_group.graph_name_to_score_funcs;
        existing_group = std::move(copied_group);
      }
      d.schedule_groups.emplace_back(schedule_group);
      RefreshGroupRelation(index, schedule_output.var_relations_, schedule_task, d);
      scheduled_results_tmp.emplace_back(d);
      d.schedule_groups.pop_back();
    }
    RegisterScoreFuncInScheduleGroup(schedule_output, schedule_group);
  }
  scheduled_results_tmp.swap(scheduled_results_cur);
  return af::SUCCESS;
}

Optimizer::Optimizer(const OptimizerOptions &options) : options_(options) {}

Status Optimizer::DeserializeAscGraphNodes(const af::ComputeGraphPtr &fused_graph,
                                           std::map<af::Node *, af::AscGraph> &asc_backend_to_ascgraph,
                                           SizeVarSet &original_var_set,
                                           std::vector<af::Expression> &frontend_shape_vars) {
  for (auto &node : fused_graph->GetDirectNodePtr()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() == kAscGraphNodeType) {
      const std::string *serialized_ascgraph = af::AttrUtils::GetStr(node->GetOpDescBarePtr(), kAttrAscGraph);
      GE_ASSERT_NOTNULL(serialized_ascgraph, "Failed to get serialized ascgraph attr from node:[%s].",
                        node->GetNamePtr());
      std::string graph_name = node->GetName() + "_ascgraph";
      af::AscGraph ascgraph(graph_name.c_str());
      GE_CHK_STATUS_RET(af::AscGraphUtils::DeserializeFromReadable(*serialized_ascgraph, ascgraph),
                        "DeserializeFromReadable failed, graph:[%s].", fused_graph->GetName().c_str());
      std::vector<af::Expression> graph_shape_vars;
      GE_CHK_STATUS_RET(AscGraphInfoComplete::CollectFrontendShapeVars(ascgraph, graph_shape_vars),
                        "Collect frontend shape vars failed, graph:[%s].", ascgraph.GetName().c_str());
      frontend_shape_vars.insert(frontend_shape_vars.end(), graph_shape_vars.begin(), graph_shape_vars.end());
      ascgraph.SetGraphType(af::AscGraphType::kImplGraph);
      GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(ascgraph), "CompleteApiInfo failed");
      AscGraphInfoComplete::AppendOriginalSizeVar(ascgraph, original_var_set);
      ascir::utils::DumpGraph(ascgraph, "AfterDeserialize");
      asc_backend_to_ascgraph.emplace(node, ascgraph);
    }
  }
  GE_ASSERT_TRUE(!asc_backend_to_ascgraph.empty(), "The fused graph [%s] is invalid, which has none AscBackend node.",
                 fused_graph->GetName().c_str());
  return af::SUCCESS;
}

Status Optimizer::Optimize(const af::ComputeGraphPtr &fused_graph,
                           ascir::FusedScheduledResult &fused_scheduled_result) {
  GELOGI("Fused graph optimize in, graph_name:[%s].", fused_graph->GetName().c_str());
  // RAII Guard，函数结束时自动清空 fused_graph_name
  ascir::utils::FusedGraphNameGuard guard(fused_graph->GetName());
  ascir::utils::DumpComputeGraph(fused_graph, "BaseFusedGraph");
  if (options_.graph_type == GraphType::kFusedAscBackend) {
    return OptimizeFusedAscBackend(fused_graph, fused_scheduled_result);
  }
  std::map<af::Node *, af::AscGraph> asc_backend_to_ascgraph;
  SizeVarSet original_var_set;
  std::vector<af::Expression> frontend_shape_vars;
  GE_CHK_STATUS_RET(
      DeserializeAscGraphNodes(fused_graph, asc_backend_to_ascgraph, original_var_set, frontend_shape_vars),
      "Deserialize ascgraph nodes failed, graph:[%s].", fused_graph->GetName().c_str());
  // If there is more than one Ascend backend on the fused graph, it is necessary to determine whether partial sub -
  // graphs can be merged based on the supported scenarios. If there are still more than one Ascend nodes after the
  // merging, it should be converted into multiple schedule groups.
  af::AscGraph hint_graph(fused_graph->GetName().c_str());
  if (asc_backend_to_ascgraph.size() > 1UL) {
    GE_CHK_STATUS_RET(FusedGraphUnfolder::UnfoldFusedGraph(fused_graph, asc_backend_to_ascgraph, hint_graph),
                      "Failed to unfold graph[%s].", fused_graph->GetName().c_str());
  } else {
    hint_graph = asc_backend_to_ascgraph.begin()->second;
  }
  auto owner_graph = af::AscGraphUtils::GetComputeGraph(hint_graph);
  GE_ASSERT_NOTNULL(owner_graph);
  owner_graph->SetName(ascgen_utils::GenValidName(fused_graph->GetName()));
  GE_ASSERT_SUCCESS(Optimize(hint_graph, fused_scheduled_result), "optimize failed, graph:[%s].",
                    hint_graph.GetName().c_str());
  GE_CHK_STATUS_RET(AscGraphInfoComplete::NormalizeFrontendShapeVars(frontend_shape_vars),
                    "Normalize frontend shape vars failed, graph:[%s].", fused_graph->GetName().c_str());
  fused_scheduled_result.frontend_shape_vars = std::move(frontend_shape_vars);
  fused_scheduled_result.frontend_shape_vars_collected = true;
  // modify origin var and fused_graph
  fused_scheduled_result.fused_graph_name = fused_graph->GetName().c_str();
  fused_scheduled_result.origin_vars.assign(original_var_set.begin(), original_var_set.end());
  return af::SUCCESS;
}

Status Optimizer::CollectAscBackendNodes(const af::ComputeGraphPtr &fused_graph,
                                         std::map<af::Node *, af::AscGraph> &asc_backend_to_ascgraph,
                                         SizeVarSet &original_var_set,
                                         std::vector<af::Expression> &frontend_shape_vars) const {
  for (auto &node : fused_graph->GetDirectNodePtr()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetType() == kAscBackendType) {
      const auto fuse_attr = node->GetOpDesc()->GetAttrsGroup<af::AutoFuseAttrs>();
      GE_ASSERT_NOTNULL(fuse_attr, "Node %s has no AutoFuseAttrs", node->GetName().c_str());
      auto fuse_asc_graph = fuse_attr->GetAscGraph();
      GE_ASSERT_NOTNULL(fuse_asc_graph, "Cannot get ascgraph from ascbc node:[%s].", node->GetNamePtr());
      GE_CHK_STATUS_RET(ScheduleUtils::ApplyGraphSchedAxisToNodes(*fuse_asc_graph),
                        "Failed to apply graph sched axis to graph:[%s].", fuse_asc_graph->GetName().c_str());
      ascir::utils::DumpGraph(*fuse_asc_graph, "AutoFuseBeforeRemoveDanglingNodes");
      std::vector<af::Expression> graph_shape_vars;
      GE_CHK_STATUS_RET(AscGraphInfoComplete::CollectFrontendShapeVars(*fuse_asc_graph, graph_shape_vars),
                        "Collect frontend shape vars failed, graph:[%s].", fuse_asc_graph->GetName().c_str());
      frontend_shape_vars.insert(frontend_shape_vars.end(), graph_shape_vars.begin(), graph_shape_vars.end());
      GE_CHK_STATUS_RET(RemoveDanglingNodes(*fuse_asc_graph), "Remove dangling nodes failed, graph:[%s].",
                        fuse_asc_graph->GetName().c_str());
      ::ascir::utils::DumpGraph(*fuse_asc_graph, "AutoFuseBeforeOptimize");
      AscGraphInfoComplete::AppendOriginalSizeVar(*fuse_asc_graph, original_var_set);
      asc_backend_to_ascgraph.emplace(node, *fuse_asc_graph);
    }
  }
  GE_ASSERT_TRUE(!asc_backend_to_ascgraph.empty(), "The fused graph [%s] is invalid, which has none AscBackend node.",
                 fused_graph->GetName().c_str());
  GE_CHK_STATUS_RET(AscGraphInfoComplete::NormalizeFrontendShapeVars(frontend_shape_vars),
                    "Normalize frontend shape vars failed, graph:[%s].", fused_graph->GetName().c_str());
  return af::SUCCESS;
}

Status Optimizer::OptimizeFusedAscBackend(const af::ComputeGraphPtr &fused_graph,
                                          ascir::FusedScheduledResult &fused_scheduled_result) const {
  std::map<af::Node *, af::AscGraph> asc_backend_to_ascgraph;
  SizeVarSet original_var_set;
  std::vector<af::Expression> frontend_shape_vars;
  GE_CHK_STATUS_RET(CollectAscBackendNodes(fused_graph, asc_backend_to_ascgraph, original_var_set, frontend_shape_vars),
                    "Collect ascbackend nodes failed, graph:[%s].", fused_graph->GetName().c_str());
  fused_scheduled_result.frontend_shape_vars = std::move(frontend_shape_vars);
  fused_scheduled_result.frontend_shape_vars_collected = true;

  GE_ASSERT_SUCCESS(FusedGraphModifier::SubgraphConnectionsToWorkspace(fused_graph, asc_backend_to_ascgraph),
                    "Failed to add workspace between ascgraphs.");
  fused_scheduled_result.fused_graph_name = fused_graph->GetName().c_str();

  for (auto &node : fused_graph->GetDirectNodePtr()) {
    if (node->GetType() == kAscBackendType) {
      auto it = asc_backend_to_ascgraph.find(node);
      GE_ASSERT_TRUE(it != asc_backend_to_ascgraph.end());
      std::vector<::ascir::ScheduledResult> sub_results;
      GE_ASSERT_SUCCESS(OptimizeForHintGraph(it->second, sub_results), "optimize failed, graph:[%s].",
                        it->second.GetName().c_str());
      fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(std::move(sub_results));
    }
  }
  fused_scheduled_result.origin_vars.assign(original_var_set.begin(), original_var_set.end());
  BufQueAllocator allocator;
  GE_CHK_STATUS_RET(allocator.PrepareImplGraphMemoryPlan(fused_scheduled_result));
  GE_CHK_STATUS_RET(StaticUbTemplateFilter().Filter(fused_scheduled_result));
  GE_CHK_STATUS_RET(allocator.CollectFusedIoNodes(fused_scheduled_result));
  GE_CHK_STATUS_RET(optimize::L2CacheHintManager::ParseGraph(*fused_graph, fused_scheduled_result));
  GELOGI("AllocBufQue end");
  TryEnableGroupParallel(fused_scheduled_result);
  for (auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (auto &result : scheduled_results) {
      GE_ASSERT_SUCCESS(FusedGraphModifier::ChangeStartingOutputToWorkspace(result.schedule_groups),
                        "Change starting output to workspace failed.");
    }
  }
  ascir::utils::DumpScheduleResult(fused_scheduled_result, "AutoFuseAfterOptimize");
  return af::SUCCESS;
}

Status Optimizer::BufQueAlloc(const ascir::HintGraph &graph, ascir::ImplGraph &impl_graph) const {
  (void)graph;
  FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.resize(1UL);
  fused_scheduled_result.node_idx_to_scheduled_results[0UL].resize(1UL);
  fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.resize(1UL);
  fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups[0].impl_graphs.emplace_back(
      impl_graph);
  GE_CHK_STATUS_RET(BufQueAllocator().AllocBufQue(fused_scheduled_result), "AllocBufQue failed");
  return af::SUCCESS;
}

Status Optimizer::BufQueAlloc(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &impl_graphs) const {
  for (auto &impl_graph : impl_graphs) {
    GE_CHK_STATUS_RET(this->BufQueAlloc(graph, impl_graph), "AllocBufQue failed");
  }
  return af::SUCCESS;
}

Status Optimizer::GraphPass(ascir::ImplGraph &impl_graph) const {
  return autoschedule::PassRunnerHandler::RunPasses(impl_graph);
}

Status Optimizer::GetNonContinuousAxisPairBySpecialRule(ascir::ImplGraph &impl_graph,
                                                        std::set<std::pair<int64_t, int64_t>> &non_continuous_pair) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsConcat(node) || (ScheduleUtils::IsSplit(node))) {
      const std::vector<af::Expression> &input_repeats = node->inputs[0].attr.repeats;
      const std::vector<af::Expression> &output_repeats = node->outputs[0].attr.repeats;
      GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                     "The output dim cnt [%zu] of concat mismatch with input dim cnt [%zu].", output_repeats.size(),
                     input_repeats.size());

      af::Expression pre_size = af::sym::kSymbolOne;
      uint32_t concat_dim{0U};
      for (uint32_t i = 0U; i < input_repeats.size(); ++i) {
        if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
          concat_dim = i;
          break;
        }
        pre_size = pre_size * input_repeats[i];
      }

      if ((concat_dim > 0U) &&
          (af::SymbolicUtils::StaticCheckEq(pre_size, af::sym::kSymbolOne) != af::TriBool::kTrue)) {
        non_continuous_pair.emplace(concat_dim - 1, concat_dim);
      }
      bool no_merge_first_axis = false;
      (void)af::AttrUtils::GetBool(node->GetOpDesc(), "_keep_first_axis", no_merge_first_axis);
      if (no_merge_first_axis) {
        non_continuous_pair.emplace(0, 1);
      }
    }

    if (ScheduleUtils::IsLoad(node)) {
      auto strides = node->outputs[0U].attr.strides;
      for (int64_t i = static_cast<int64_t>(strides.size() - 1); i >= 0; --i) {
        if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
          continue;
        }
        if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolOne) == af::TriBool::kTrue) {
          break;
        } else {
          GELOGD("Node [%s] is last axis load, axis:[%ld].", node->GetNamePtr(), i);
          non_continuous_pair.emplace(i - 1, i);
          break;
        }
      }
      continue;
    }

    if (ScheduleUtils::IsGather(node)) {
      int64_t attr_axis = -1;
      ScheduleUtils::GetNodeIrAttrValue(node, "axis", attr_axis);
      // scalar == input[0].repeat.size() - 1 --> gather尾轴场景
      if (static_cast<size_t>(attr_axis) == node->inputs[0].attr.repeats.size() - 1 && attr_axis != 0) {
        non_continuous_pair.emplace(attr_axis - 1, attr_axis);
      }
      // scalar != input[0].repeat.size() - 1 --> gather非尾轴场景
      if (static_cast<size_t>(attr_axis) != node->inputs[0].attr.repeats.size() - 1) {
        auto indices = node->inputs[1].attr.repeats;
        attr_axis = attr_axis + indices.size() - 1;
        non_continuous_pair.emplace(attr_axis, attr_axis + 1);
      }
    }

    if (node->GetType() == "Softmax") {
      auto axis_size = static_cast<int64_t>(node->inputs[0].attr.repeats.size());
      if (axis_size > 1) {
        non_continuous_pair.emplace(
            axis_size - 2, axis_size - 1);  // Softmax沿最后一个轴做归约，最后两个轴(axis - 2、axis - 1)不能合并
      }
    }
  }
  return af::SUCCESS;
}

Status Optimizer::RemoveAllZeroStrideLoopAxis(ascir::ImplGraph &owner_graph) {
  if (ScheduleUtils::HasComputeType(owner_graph, af::ComputeType::kComputeGather)) {
    return af::SUCCESS;
  }
  std::unordered_set<size_t> zero_stride_axis_indices = IdentifyZeroStrideAxisIndices(owner_graph);
  if (zero_stride_axis_indices.empty()) {
    return af::SUCCESS;
  }

  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    std::vector<int64_t> valid_axis_ids;
    const auto &original_axis_ids = node->attr.sched.axis;
    for (size_t i = 0UL; i < original_axis_ids.size(); ++i) {
      if (zero_stride_axis_indices.count(i) == 0UL) {
        valid_axis_ids.push_back(original_axis_ids[i]);
      }
    }
    node->attr.sched.axis = valid_axis_ids;

    for (const auto &output : node->outputs()) {
      std::vector<int64_t> new_axis_ids;
      std::vector<af::Expression> new_repeats;
      std::vector<af::Expression> new_strides;

      for (size_t i = 0UL; i < output->attr.axis.size(); ++i) {
        auto axis_id = output->attr.axis[i];
        if (std::find(valid_axis_ids.begin(), valid_axis_ids.end(), axis_id) != valid_axis_ids.end()) {
          new_axis_ids.push_back(axis_id);
          GE_ASSERT_TRUE(i < output->attr.strides.size());
          GE_ASSERT_TRUE(i < output->attr.repeats.size());
          new_strides.push_back(output->attr.strides[i]);
          new_repeats.push_back(output->attr.repeats[i]);
        }
      }

      output->attr.axis = new_axis_ids;
      output->attr.strides = new_strides;
      output->attr.repeats = new_repeats;
    }
  }
  return af::SUCCESS;
}

Status Optimizer::MergeContinuousAxis(ascir::ImplGraph &impl_graph, ascir::CubeTemplateType cube_type) {
  auto all_axis = impl_graph.GetAllAxis();
  if (all_axis.size() <= 1UL || ascgen_utils::indirect_load::FindIndirectLoadNode(impl_graph) != nullptr) {
    return af::SUCCESS;
  }
  // concat等场景,会有多套轴, 只能先用循环轴的index来生成潜在连续组, 后续根据连续组会找到多个连续轴
  std::vector<std::pair<int64_t, int64_t>> potential_axis_idx;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!ScheduleUtils::IsBuffer(node)) {
      auto axis_ids = node->attr.sched.axis;
      potential_axis_idx.reserve(axis_ids.size() - 1);
      for (size_t i = 0UL; i < axis_ids.size() - 1; ++i) {
        potential_axis_idx.emplace_back(i, i + 1);
      }
      break;
    }
  }

  // 剔除根据规则产生的不连续轴(后续reduce和brc考虑放规则里, 减少全局判断耗时
  std::set<std::pair<int64_t, int64_t>> non_continuous_pair;
  GE_ASSERT_SUCCESS(GetNonContinuousAxisPairBySpecialRule(impl_graph, non_continuous_pair));
  potential_axis_idx.erase(
      std::remove_if(potential_axis_idx.begin(), potential_axis_idx.end(),
                     [&non_continuous_pair](const auto &pair) { return non_continuous_pair.count(pair) > 0; }),
      potential_axis_idx.end());

  // 剩下的根据全图repeat/stride判断是否连续
  for (auto it = potential_axis_idx.rbegin(); it != potential_axis_idx.rend();) {
    if (!IsAxisContinuous(impl_graph, it->first, it->second)) {
      auto normal_it = it.base();
      ++it;
      potential_axis_idx.erase(normal_it - 1);
    } else {
      ++it;
    }
  }
  // 现根据id合并pair
  std::vector<std::vector<int64_t>> merged_axis_indexes = MergeContinuousPairs(potential_axis_idx);
  std::map<std::vector<af::AxisId>, af::AxisId> from_id_to_merged_id;
  std::vector<af::AxisPtr> new_merged_axes;
  // Do merge
  for (auto node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    std::vector<af::AxisId> node_merged_id;
    for (const auto &from_idx : merged_axis_indexes) {
      std::vector<af::AxisId> from_ids;
      for (const int64_t index : from_idx) {
        from_ids.emplace_back(node->attr.sched.axis[index]);
      }
      auto iter = from_id_to_merged_id.find(from_ids);
      if (iter == from_id_to_merged_id.end()) {
        auto merged_axis = impl_graph.MergeAxis(from_ids);
        new_merged_axes.push_back(merged_axis);
        node_merged_id.push_back(merged_axis->id);
        from_id_to_merged_id[from_ids] = merged_axis->id;
      } else {
        node_merged_id.push_back(iter->second);
      }
    }
    for (auto axis_id : node_merged_id) {
      GELOGD("Apply merged axis id [%ld] to node:[%s].", axis_id, node->GetNamePtr());
      GE_ASSERT_TRUE(impl_graph.ApplyMerge(node, axis_id), "Failed to apply merged axis id %ld to node:[%s].", axis_id,
                     node->GetNamePtr());
    }
  }

  // TTODO 需要考虑删轴等操作,后续优化
  int64_t attr_axis = -1;
  int64_t param_size = -1;
  bool has_gather = ScheduleUtils::GetGatherParams(impl_graph, attr_axis, param_size);
  if ((!has_gather) && (cube_type != ascir::CubeTemplateType::kUBFuse)) {
    // 此处合轴后的轴可以认为是original的
    for (const auto &axis : new_merged_axes) {
      axis->type = af::Axis::Type::kAxisTypeOriginal;
      axis->from.clear();
    }
    if (cube_type == ascir::CubeTemplateType::kUBFuse) {
      return af::GRAPH_SUCCESS;
    }
    GE_ASSERT_SUCCESS(ScheduleUtils::RemoveUnusedAxes(impl_graph), "Failed to remove unused axes");
  }

  return af::SUCCESS;
}

Status Optimizer::OptimizeForHintGraph(af::AscGraph &hint_graph,
                                       std::vector<ascir::ScheduledResult> &scheduled_results) const {
  ScheduleUtils::NormalizeAxisIds(hint_graph);
  hint_graph.SetGraphType(af::AscGraphType::kImplGraph);
  ascir::utils::DumpPyCode(hint_graph);
  GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(hint_graph), "CompleteApiInfo failed");
  GE_ASSERT_SUCCESS(CheckGraphValidity(hint_graph), "Check graph validity failed, graph:[%s].",
                    hint_graph.GetName().c_str());
  // 预处理：当前仅针对kFusedAscBackend流程生效，后续会全部放开
  if (options_.graph_type == GraphType::kFusedAscBackend) {
    GE_CHK_STATUS_RET(af::pre_process::PreProcess::Run(hint_graph), "Pre process failed");
    ascir::utils::DumpGraph(hint_graph, "AfterPreProcess");
  }
  // 截断 graph name
  std::string base_graph_name = TruncateGraphName(hint_graph.GetName());
  ascir::ImplGraph optimize_graph(base_graph_name.c_str());
  GE_ASSERT_TRUE(optimize_graph.CopyFrom(hint_graph));

  // dtype 兜底处理：针对算子实际支持的 dtype 与注册不一致的情况，插入必要的 Cast
  GE_CHK_STATUS_RET(DtypeConsistency::EnsureDtypeConsistency(optimize_graph), "Failed to ensure dtype consistency");
  ascir::utils::DumpGraph(optimize_graph, "AfterDtypeConsistency");

  // 重排 Cast 和 Transpose,
  // 针对GE前端的临时处理，后续如果和Inductor的升精度流程归一，统一放在schedule流程中，该pass可以删除
  GE_CHK_STATUS_RET(SwapCastWithPrecisionAgnosticOpsPass::Run(optimize_graph), "Failed to reorder Cast and Transpose");
  ascir::utils::DumpGraph(optimize_graph, "AfterSwapCastWithPrecisionAgnosticOps");

  GE_CHK_STATUS_RET(GraphPass(optimize_graph), "Run graph passes failed");

  GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(optimize_graph), "CompleteApiInfo failed");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(optimize_graph));
  utils::DumpGraph(optimize_graph, "AfterGraphPass");
  // cube拆分后再做合轴
  if (!ScheduleUtils::HasComputeType(optimize_graph, af::ComputeType::kComputeCube)) {
    // 前端显式 1D Arange -> Broadcast: 合轴前补齐退化前缀轴, 保持输入输出同轴数
    GE_ASSERT_SUCCESS(optimize::NormalizeArangeBroadcastViews(optimize_graph),
                      "Normalize explicit Arange Broadcast views failed.");
    // 这里concat已经打破了一套轴的约束
    GE_ASSERT_SUCCESS(RemoveAllZeroStrideLoopAxis(optimize_graph), "Remove All zero stride axis failed.");
    GE_ASSERT_SUCCESS(MergeContinuousAxis(optimize_graph), "Merge continuous axes failed.");
  }
  ascir::utils::DumpGraph(optimize_graph, "AfterMergeAxis");

  std::vector<ScheduleTask> schedule_tasks;
  GE_CHK_STATUS_RET(ScheduleTaskGenerator::GenerateTasks(optimize_graph, schedule_tasks, options_),
                    "Generate tasks failed");
  for (size_t i = 0U; i < schedule_tasks.size(); ++i) {
    GE_CHK_STATUS_RET(AutoScheduler(hint_graph, schedule_tasks[i], scheduled_results), "AutoScheduler task[%zu] failed",
                      i);
    GELOGI("AutoScheduler task[%zu] end", i);
  }

  // 最终处理：添加 SizeVar 和索引后缀
  GE_ASSERT_SUCCESS(FinalizeIndexedGraphs(scheduled_results));

  return af::SUCCESS;
}

/**
 * 用于调整load执行顺序，具体步骤如下：
 * 1. 获取节点之间的连边关系
 * 2. 按序遍历所有节点，从第二个load开始，先判断其是否存在内存共复用，若存在则跳过；
 *    若不存在则采用并查集判定其与首load是否具有公共计算节点，若有则前移；若无则不作处理；
 *    直到达到最大可调序load个数或遍历结束即退出
 * 3. 对load前移后的图进行reorder
 * @param impl_graph
 * @return
 */
Status Optimizer::LoadOpSeqAdjust(const af::AscGraph &impl_graph) {
  std::map<int64_t, int64_t> need_forward_nodes_id;
  int64_t first_load_id = kInvalidNodeId;
  GE_ASSERT_SUCCESS(SearchNodesNeedForward(impl_graph, need_forward_nodes_id, first_load_id));
  GE_ASSERT_SUCCESS(DoSeqAdjustment(impl_graph, need_forward_nodes_id, first_load_id));
  const auto &compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->ReorderByNodeId();
  return af::SUCCESS;
}

Status Optimizer::Optimize(af::AscGraph &hint_graph, FusedScheduledResult &fused_scheduled_result) {
  ascir::utils::DumpGraph(hint_graph, "AutoFuseBeforeRemoveDanglingNodes");
  GE_CHK_STATUS_RET(
      AscGraphInfoComplete::CollectFrontendShapeVars(hint_graph, fused_scheduled_result.frontend_shape_vars),
      "Collect frontend shape vars failed, graph:[%s].", hint_graph.GetName().c_str());
  fused_scheduled_result.frontend_shape_vars_collected = true;
  GE_CHK_STATUS_RET(AscGraphInfoComplete::NormalizeFrontendShapeVars(fused_scheduled_result.frontend_shape_vars),
                    "Normalize frontend shape vars failed, graph:[%s].", hint_graph.GetName().c_str());
  GE_CHK_STATUS_RET(RemoveDanglingNodes(hint_graph), "Remove dangling nodes failed, graph:[%s].",
                    hint_graph.GetName().c_str());
  ascir::utils::DumpGraph(hint_graph, "AutoFuseBeforeOptimize");
  fused_scheduled_result.node_idx_to_scheduled_results.resize(1UL);
  SizeVarSet original_var_set;
  AscGraphInfoComplete::AppendOriginalSizeVar(hint_graph, original_var_set);
  GE_ASSERT_SUCCESS(OptimizeForHintGraph(hint_graph, fused_scheduled_result.node_idx_to_scheduled_results[0UL]),
                    "Failed to optimize for graph:[%s].", hint_graph.GetName().c_str());
  // 内存分配
  BufQueAllocator allocator;
  GE_CHK_STATUS_RET(allocator.PrepareImplGraphMemoryPlan(fused_scheduled_result));
  if (options_.graph_type == GraphType::kAscGraph) {
    fused_scheduled_result.fused_graph_name = hint_graph.GetName().c_str();
    fused_scheduled_result.origin_vars.assign(original_var_set.begin(), original_var_set.end());
  }
  GE_CHK_STATUS_RET(StaticUbTemplateFilter().Filter(fused_scheduled_result));
  GE_CHK_STATUS_RET(allocator.CollectFusedIoNodes(fused_scheduled_result));
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(hint_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  GE_CHK_STATUS_RET(optimize::L2CacheHintManager::ParseGraph(*compute_graph, fused_scheduled_result));
  GELOGI("AllocBufQue end");
  TryEnableGroupParallel(fused_scheduled_result);
  ExecSeqAdvancedOfLoad(fused_scheduled_result);
  ascir::utils::DumpScheduleResult(fused_scheduled_result, "AutoFuseAfterOptimize");
  return af::SUCCESS;
}

// Reduce 多核切R轴场景，是否第一阶段
bool Optimizer::IsReduceFirstStage(size_t index, ScheduleTask &schedule_task) {
  if (schedule_task.reduce_type != ReduceTemplateType::kRCore) {
    return false;
  }
  auto iter = schedule_task.groups_relations_in.find(index);
  if (iter != schedule_task.groups_relations_in.end()) {
    return true;
  }
  return false;
}

void Optimizer::RefreshGroupRelation(size_t index, std::map<std::string, af::Expression> &var_relations,
                                     ScheduleTask &schedule_task, ScheduledResult &schedule_result) const {
  auto iter = schedule_task.groups_relations_in.find(index);
  if (iter == schedule_task.groups_relations_in.end()) {
    return;
  }
  for (auto dst : iter->second) {
    schedule_result.var_relations[dst][index] = var_relations;
  }
}

static Status ProcessCubeSchedules(std::vector<ascir::ScheduledResult> &scheduled_results_cur,
                                   ascir::ImplGraph &grouped_graph) {
  ascir::Graph optimize_graph(ascgen_utils::GenValidName(grouped_graph.GetName()).c_str());
  GE_ASSERT_TRUE(optimize_graph.CopyFrom(grouped_graph));
  ScheduleGroup schedule_group{{optimize_graph}, {}};
  std::for_each(scheduled_results_cur.begin(), scheduled_results_cur.end(),
                [&schedule_group](ascir::ScheduledResult &res) { res.schedule_groups.emplace_back(schedule_group); });
  return af::SUCCESS;
}

static Status ProcessNonReduceSchedules(const std::vector<autoschedule::AutoScheduleOutput> &schedule_outputs,
                                        std::vector<ascir::ScheduledResult> &scheduled_results_cur,
                                        ScheduleTask &schedule_task, std::set<std::string> &scored_graph_names) {
  ScheduleGroup schedule_group;
  schedule_group.impl_graphs.reserve(schedule_outputs.size());
  for (const auto &schedule_output : schedule_outputs) {
    schedule_group.impl_graphs.emplace_back(schedule_output.scheduled_graph);
    // 目前仅对elewise+brc/单group/单result开放nddma模板打分，单result单group场景过滤放在FilterComplexTilingDataScoreFuncs
    // transpose/split/concat有场景触发转为Load/Store
    std::string autoschedule_graph_name =
        RegisterScoreFuncInScheduleGroup(schedule_output, schedule_group, schedule_task.has_load_store_conversion);
    scored_graph_names.insert(autoschedule_graph_name);
  }
  for (auto &res : scheduled_results_cur) {
    res.schedule_groups.emplace_back(schedule_group);
  }
  return af::SUCCESS;
}

Status Optimizer::InitializeScheduledResults(std::vector<ascir::ScheduledResult> &scheduled_results_cur,
                                             ScheduleTask &schedule_task) {
  ::ascir::ScheduledResult schedule_result;
  schedule_result.score_func = schedule_task.score_func.c_str();
  schedule_result.is_reduce_mem_reuse = schedule_task.reduce_type == ReduceTemplateType::kAllLoad;
  schedule_result.cube_type = schedule_task.cube_type;
  schedule_result.semantic_name = schedule_task.semantic_name.empty()
                                      ? ascgen_utils::GetTemplateSemanticName(schedule_task.optimize_graph)
                                      : schedule_task.semantic_name;
  scheduled_results_cur.emplace_back(schedule_result);
  return af::SUCCESS;
}

Status Optimizer::AutoScheduler([[maybe_unused]] const HintGraph &hint_graph, ScheduleTask &schedule_task,
                                std::vector<ascir::ScheduledResult> &scheduled_results) const {
  size_t index = 0UL;
  std::vector<ascir::ScheduledResult> scheduled_results_cur;
  GE_ASSERT_SUCCESS(InitializeScheduledResults(scheduled_results_cur, schedule_task));

  // 记录注册打分函数的nddma模板名
  std::set<std::string> scored_graph_names;

  for (auto &grouped_graph : schedule_task.grouped_graphs) {
    GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(grouped_graph), "CompleteApiInfo failed");
    if (CanDoReMergeAxis(grouped_graph, schedule_task.reduce_type)) {
      GE_ASSERT_SUCCESS(RemoveAllZeroStrideLoopAxis(grouped_graph), "Remove All zero stride axis failed.");
      GE_ASSERT_SUCCESS(MergeContinuousAxis(grouped_graph, schedule_task.cube_type), "Merge continuous axes failed.");
    }
    // 图上全部原子符号加到size var上用于生成tiling data
    GE_ASSERT_SUCCESS(ScheduleUtils::ClearAllSizeVar(grouped_graph));
    SizeVarSet original_var_set;
    AscGraphInfoComplete::AppendOriginalSizeVar(grouped_graph, original_var_set);
    for (const auto &exp : original_var_set) {
      GE_ASSERT_GRAPH_SUCCESS(grouped_graph.CreateSizeVar(exp));
    }
    ascir::utils::DumpGraph(grouped_graph, "BeforeAutoSchedule");
    GELOGI("AutoScheduler start: %s", grouped_graph.GetName().c_str());
    if (ScheduleUtils::HasComputeType(grouped_graph, af::ComputeType::kComputeCube)) {
      GE_ASSERT_SUCCESS(ProcessCubeSchedules(scheduled_results_cur, grouped_graph));
      continue;
    }
    bool is_reduce_first_stage = IsReduceFirstStage(index, schedule_task);
    std::vector<autoschedule::AutoScheduleOutput> schedule_outputs;
    auto scheduler = autoschedule::AutoSchedule(grouped_graph, schedule_outputs, is_reduce_first_stage,
                                                schedule_task.reduce_type, schedule_task.cube_type);
    GE_CHK_STATUS_RET(scheduler.DoAutoSchedule(), "Failed to do schedule, graph:[%s].",
                      grouped_graph.GetName().c_str());
    GE_ASSERT_TRUE(!schedule_outputs.empty(), "Failed to gen tiling case for graph:[%s].",
                   grouped_graph.GetName().c_str());
    GELOGI("AutoScheduler end: %s, number of tiling cases = %zu", grouped_graph.GetName().c_str(),
           schedule_outputs.size());
    if (is_reduce_first_stage) {
      GE_ASSERT_SUCCESS(ExpandReduceFirstStageResults(schedule_outputs, scheduled_results_cur, index, schedule_task));
    } else {
      if (schedule_task.reduce_type == ReduceTemplateType::kRCore) {
        GE_ASSERT_SUCCESS(CopyImplGraphs(schedule_outputs, scheduled_results_cur));
      } else {
        GE_ASSERT_SUCCESS(
            ProcessNonReduceSchedules(schedule_outputs, scheduled_results_cur, schedule_task, scored_graph_names));
      }
    }
    index++;
  }
  scheduled_results.insert(scheduled_results.end(), scheduled_results_cur.begin(), scheduled_results_cur.end());

  // 过滤复杂tilingdata的nddma打分函数
  FilterComplexTilingDataScoreFuncs(scheduled_results, scored_graph_names);

  return af::SUCCESS;
}

void Optimizer::TryEnableGroupParallel(FusedScheduledResult &fused_scheduled_result) {
  if (fused_scheduled_result.node_idx_to_scheduled_results.size() == 1UL &&
      fused_scheduled_result.workspace_nodes.empty()) {  // 有workspace表示有依赖，不能使能enable_group_parallel
    for (auto &schedule_result : fused_scheduled_result.node_idx_to_scheduled_results.front()) {
      if (schedule_result.cube_type == CubeTemplateType::kCommon) {
        schedule_result.enable_group_parallel = false;
        return;
      }
      schedule_result.enable_group_parallel =
          schedule_result.schedule_groups.size() > 1UL && schedule_result.var_relations.empty();
    }
  }
}

void Optimizer::ExecSeqAdvancedOfLoad(const FusedScheduledResult &fused_scheduled_result) {
  for (auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (auto &schedule_result : scheduled_results) {
      if (schedule_result.cube_type == ascir::CubeTemplateType::kUBFuse) {
        continue;
      }
      for (auto &schedule_group : schedule_result.schedule_groups) {
        for (auto &impl_graph : schedule_group.impl_graphs) {
          LoadOpSeqAdjust(impl_graph);
        }
      }
    }
  }
}
