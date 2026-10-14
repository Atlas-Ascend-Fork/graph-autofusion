/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <queue>
#include <map>
#include <numeric>
#include <set>
#include "graph/utils/graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "autoschedule/axis_group.h"
#include "ascir_utils.h"
#include "common_utils.h"
#include "node_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_pass.h"
#include "reduce_schedule_case_generator.h"
#include "register/op_def_factory_af.h"
#include "base/err_msg.h"
#include "graph/symbolizer/symbolic.h"
#include "indirect_load_utils.h"

namespace optimize {
namespace {
size_t TWO = 2;
size_t kMaxFullLoadAxisSize = 3UL;
size_t NODE_COUNT_AFTER_REDUCE = 4UL;
constexpr int64_t kBroadcastTailRepeatWorkspaceThreshold = 22500L;
constexpr int64_t kBroadcastFirstAxisWorkspaceThreshold = 128L;
std::string GetNewNodeName(const af::AscNodePtr &src_node, const af::AscNodePtr &dst_node, const std::string &type,
                           int32_t idx) {
  return src_node->GetName() + "_to_" + dst_node->GetName() + "_" + type + "_" + to_string(idx);
}

// 打印 loop_start_end 的 [begin_idx, end_idx) 区段, tag 标识来源(FindNormLoops/AddNormLikeLoops)
void LogNormLoops(const char *tag, const std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &loop_start_end,
                  size_t begin_idx, size_t end_idx) {
  GELOGI("%s got %zu loop(s):", tag, end_idx - begin_idx);
  for (size_t i = begin_idx; i < end_idx; i++) {
    GELOGI("  loop[%zu] start=%s(id=%lld) end=%s(id=%lld)", i, loop_start_end[i].first->GetName().c_str(),
           loop_start_end[i].first->GetOpDescBarePtr()->GetId(), loop_start_end[i].second->GetName().c_str(),
           loop_start_end[i].second->GetOpDescBarePtr()->GetId());
  }
}

bool IsTailRepeatProductLargeEnough(const std::vector<ascir::SizeExpr> &repeats) {
  if (repeats.size() <= 1UL) {
    return false;
  }
  int64_t product = 1L;
  for (size_t i = 1UL; i < repeats.size(); ++i) {
    int64_t repeat = 0L;
    if (!repeats[i].GetConstValue(repeat) || repeat <= 0L) {
      return false;
    }
    if (product >= kBroadcastTailRepeatWorkspaceThreshold || repeat >= kBroadcastTailRepeatWorkspaceThreshold ||
        product > kBroadcastTailRepeatWorkspaceThreshold / repeat) {
      return true;
    }
    product *= repeat;
  }
  return product >= kBroadcastTailRepeatWorkspaceThreshold;
}

bool IsOnlyFirstAxisBroadcast(const af::AscNodePtr &pre_node, const af::AscNodePtr &broadcast_node) {
  GE_ASSERT_NOTNULL(pre_node);
  GE_ASSERT_NOTNULL(broadcast_node);
  GE_ASSERT_TRUE(!pre_node->outputs().empty());
  GE_ASSERT_TRUE(!broadcast_node->outputs().empty());
  const auto &pre_repeats = pre_node->outputs[0].attr.repeats;
  const auto &broadcast_repeats = broadcast_node->outputs[0].attr.repeats;
  if (pre_repeats.empty() || pre_repeats.size() != broadcast_repeats.size()) {
    return false;
  }
  if (af::SymbolicUtils::StaticCheckEq(pre_repeats[0], broadcast_repeats[0]) == af::TriBool::kTrue) {
    return false;
  }
  int64_t broadcast_first_axis = 0L;
  if (!broadcast_repeats[0].GetConstValue(broadcast_first_axis) ||
      broadcast_first_axis <= kBroadcastFirstAxisWorkspaceThreshold) {
    return false;
  }
  for (size_t i = 1UL; i < pre_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(pre_repeats[i], broadcast_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool HasOperatorBeforeBroadcast(const af::AscNodePtr &pre_node) {
  GE_ASSERT_NOTNULL(pre_node);
  return pre_node->GetInDataNodesSize() != 0UL && !af::ops::IsOps<af::ascir_op::Load>(pre_node);
}

bool NeedWorkspaceBeforeBroadcast(const af::AscNodePtr &broadcast_node) {
  if (!af::ops::IsOps<af::ascir_op::Broadcast>(broadcast_node) || broadcast_node->GetInDataNodesSize() != 1UL) {
    return false;
  }
  auto pre_node = std::dynamic_pointer_cast<af::AscNode>(broadcast_node->GetInDataNodes().at(0UL));
  GE_ASSERT_NOTNULL(pre_node);
  if (!IsOnlyFirstAxisBroadcast(pre_node, broadcast_node)) {
    return false;
  }
  if (!HasOperatorBeforeBroadcast(pre_node)) {
    return false;
  }
  return IsTailRepeatProductLargeEnough(broadcast_node->outputs[0].attr.repeats);
}

bool IsLegacyFullLoadReduce(const std::vector<ascir::SizeExpr> &input_repeats,
                            const std::vector<ascir::SizeExpr> &output_repeats) {
  if (output_repeats.size() > kMaxFullLoadAxisSize) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckEq(input_repeats[0], output_repeats[0]) == af::TriBool::kTrue;
}

bool IsTailReduceFullLoad(const af::AscNodePtr &node, const std::vector<ascir::SizeExpr> &input_repeats,
                          const std::vector<ascir::SizeExpr> &output_repeats) {
  if (input_repeats.size() < kMaxFullLoadAxisSize) {
    return false;
  }
  const auto tail_index = input_repeats.size() - 1U;
  for (size_t i = 0U; i < tail_index; ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }
  if (node != nullptr && node->GetType() == "Softmax") {
    GELOGD("IsTailReduceFullLoad node=%s is Softmax, return true.", node->GetName().c_str());
    return true;
  }
  if (!ScheduleUtils::IsReduceOnTailAxis(node)) {
    return false;
  }
  return true;
}

Status DoCopyAscNodeTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node) {
  auto op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  for (size_t i = 0U; i < src_node->outputs().size(); i++) {
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(i));
    auto tensor_attr_group = op_desc->MutableOutputDesc(i)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = src_node->outputs[i].attr;
  }
  return af::SUCCESS;
}

Status DoCopyAscNodeOutputTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                     const uint32_t src_output_index) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);
  GE_ASSERT_TRUE(src_output_index < src_node->outputs().size());
  GE_ASSERT_TRUE(!dst_node->outputs().empty());
  auto op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  const auto output_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(output_desc);
  auto tensor_attr_group = output_desc->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr_group);
  *tensor_attr_group = src_node->outputs[src_output_index].attr;
  return af::SUCCESS;
}

Status DoCopyWorkspaceTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &workspace_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(workspace_node);
  GE_ASSERT_TRUE(!src_node->outputs().empty());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty());
  workspace_node->outputs[0].attr.dtype = src_node->outputs[0].attr.dtype;
  return af::SUCCESS;
}

const std::unordered_map<std::string, std::function<ReduceType(const char *)>> reducers = {
    {"Max", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Max>{}, n}; }},
    {"ArgMax",
     [](const char *n) {
       return ReduceType{std::in_place_type_t<af::ascir_op::ArgMaxMultiRPhase2>{}, n};
     }},  // ArgMax 二阶段使用 ArgMaxMultiRPhase2
    {"ArgMaxMultiRPhase1",
     [](const char *n) {
       return ReduceType{std::in_place_type_t<af::ascir_op::ArgMaxMultiRPhase2>{}, n};
     }},  // ArgMaxMultiRPhase1 二阶段使用 ArgMaxMultiRPhase2
    {"ArgMaxMultiRPhase2",
     [](const char *n) {
       return ReduceType{std::in_place_type_t<af::ascir_op::ArgMaxMultiRPhase2>{}, n};
     }},  // ArgMaxMultiRPhase2 二阶段使用 ArgMaxMultiRPhase2
    {"Sum", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Sum>{}, n}; }},
    {"Mean",
     [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Sum>{}, n}; }},  // Mean 二阶段使用 Sum
    {"Min", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Min>{}, n}; }},
    {"Prod", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Prod>{}, n}; }},
    {"Any", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Any>{}, n}; }},
    {"All", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::All>{}, n}; }},
    {"Softmax", [](const char *n) { return ReduceType{std::in_place_type_t<af::ascir_op::Softmax>{}, n}; }}};

bool IsNotPartitionReduce(const af::AscNodePtr &reduce_node, size_t threshold) {
  std::queue<af::NodePtr> node_queue;
  size_t node_count = 0UL;
  std::unordered_set<const af::Node *> visited;
  visited.insert(reduce_node.get());
  for (const auto &reduce_out_node : reduce_node->GetOutNodes()) {
    node_queue.emplace(reduce_out_node);
    visited.insert(reduce_out_node.get());
  }
  while (!node_queue.empty()) {
    const auto current_node = node_queue.front();
    node_queue.pop();

    if (current_node->GetInDataNodesSize() > 1UL) {
      for (const auto &in_current_node : current_node->GetInDataNodes()) {
        if (visited.find(in_current_node.get()) == visited.end()) {
          GELOGW("Node [%s] has multiple inputs, with input node [%s] not being a post-reduction node.",
                 current_node->GetNamePtr(), in_current_node->GetNamePtr());
          return false;
        }
      }
    }

    node_count += 1UL;
    if (node_count > threshold) {
      GELOGW(
          "The total count of nodes after the reduce node[%s](including the store node) is above the threshold[%zu].",
          reduce_node->GetNamePtr(), threshold);
      return false;
    }
    const auto &asc_current_node = std::dynamic_pointer_cast<af::AscNode>(current_node);
    GE_ASSERT_NOTNULL(asc_current_node);
    if (af::ops::IsOps<af::ascir_op::Store>(asc_current_node) ||
        asc_current_node->attr.api.type == af::ApiType::kAPITypeBuffer) {
      continue;
    }

    if (!ScheduleUtils::IsElewise(asc_current_node)) {
      GELOGW("The node[%s] after the reduce node[%s] is not elewise type.", asc_current_node->GetNamePtr(),
             reduce_node->GetNamePtr());
      return false;
    }

    if (node_count > threshold - 1UL) {
      GELOGW("The count of nodes after the reduce node[%s] is above the threshold[%zu].", reduce_node->GetNamePtr(),
             threshold - 1UL);
      return false;
    }

    for (const auto &next_node : current_node->GetOutAllNodes()) {
      if (visited.find(next_node.get()) == visited.end()) {
        visited.insert(next_node.get());
        node_queue.emplace(next_node);
      }
    }
  }
  return true;
}

// 判断 src 到 dst 之间是否仍存在数据边(该边被切分改接后即消失), 用于检测引用是否已被切分
bool HasDataEdgeBetween(const af::AscNodePtr &src_node, const af::AscNodePtr &dst_node) {
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        return true;
      }
    }
  }
  return false;
}

// 获取 post 侧 workspace(输出连向 load 的 workspace)所供给的 load, 按 workspace 名索引:
// ws_pre 与 ws_post 同名标识同一 workspace 缓冲, store 链凭 ws_pre 的名字索引到 post 侧
// 供给的 load, 对应的 ws_post 可由该 load 的输入生产者结构定位
std::map<std::string, af::AscNodePtr> GetPostWorkspaceLoadMap(const ascir::ImplGraph &impl_graph) {
  std::map<std::string, af::AscNodePtr> post_ws_load_map;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Workspace>(node)) {
      continue;
    }
    af::OutDataAnchorPtr out_anchor = node->GetOutDataAnchor(0);
    if (out_anchor == nullptr || out_anchor->GetPeerInDataAnchors().empty()) {
      continue;
    }
    af::AscNodePtr load_node =
        std::dynamic_pointer_cast<af::AscNode>(out_anchor->GetPeerInDataAnchors().at(0UL)->GetOwnerNode());
    if (load_node != nullptr && af::ops::IsOps<af::ascir_op::Load>(load_node)) {
      post_ws_load_map[node->GetName()] = load_node;
    }
  }
  return post_ws_load_map;
}

// 连通判断: 从 from 出发沿数据边双向(输入/输出方向)搜索 target; 遇到 pre_workspace(输入侧
// workspace)时视作与其同名 post_workspace 连通, 即允许越过切分从 pre 侧继续到 post 侧
// (由同名 workspace 供给的 load 承接); 反向不成立——到达 post_workspace 后不再跳回 pre 侧,
// 视作到头。excluded 非空时该节点不可经过(用于排除多输入节点自身, 避免绕经它使另一侧输入
// 也被误判为连通)
bool IsConnectedThroughWorkspace(const af::NodePtr &from, const af::NodePtr &target,
                                 const std::map<std::string, af::AscNodePtr> &post_ws_load_map,
                                 const af::NodePtr &excluded) {
  if (from == nullptr || target == nullptr) {
    return false;
  }
  std::set<const af::Node *> visited;
  std::vector<af::NodePtr> pending;
  visited.insert(from.get());
  pending.emplace_back(from);
  while (!pending.empty()) {
    const af::NodePtr current = pending.back();
    pending.pop_back();
    if (current == target) {
      return true;
    }
    auto push_node = [&pending, &visited, &excluded](const af::NodePtr &node) {
      if (node == nullptr || node == excluded || !visited.insert(node.get()).second) {
        return;
      }
      pending.emplace_back(node);
    };
    for (const auto &in_node : current->GetInDataNodes()) {
      push_node(in_node);
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      push_node(out_node);
    }
    if (af::ops::IsOps<af::ascir_op::Workspace>(current)) {
      const auto load_iter = post_ws_load_map.find(current->GetName());
      if (load_iter != post_ws_load_map.cend()) {
        push_node(load_iter->second);
      }
    }
  }
  return false;
}

// 固有汇点: 本来就不应该有输出的节点——图输出 Output 与 workspace 写侧(store->ws_pre,
// 有数据输入边); 其余节点没有数据输出了即可视作悬空节点摘除
bool IsInherentSink(const af::AscNodePtr &node) {
  if (af::ops::IsOps<af::ascir_op::Output>(node)) {
    return true;
  }
  return af::ops::IsOps<af::ascir_op::Workspace>(node) && node->GetInDataNodesSize() != 0UL;
}

// 判断节点在图内是否仍有数据消费者: 只认还在图中的消费者, 排除已摘除但锚点引用仍残留的
// 幽灵消费者(GetOutDataNodesSize 会把幽灵消费者也统计进去)
bool HasConsumerInGraph(const af::AscNodePtr &node, const std::set<const af::Node *> &nodes_in_graph) {
  for (const auto &out_node : node->GetOutDataNodes()) {
    if (nodes_in_graph.count(out_node.get()) != 0U) {
      return true;
    }
  }
  return false;
}

// 手工注释：合并子图内同名的比如copy_from_scalar节点，避免codegen阶段检测失败
// 合并同一子图内的同名 copy_from_* 副本(在 PartitionByConnectivity 划分子图之后逐子图执行):
// 同一子图即同一连通分量/同一 kernel, 同名同类型副本必然同源, 保留 id 最小者为基准, 其余
// 副本的消费者改接到基准后摘除整节点; 副本摘除后其输入副本(Data/Workspace)可能失去唯一
// 消费者, 循环摘除直至无死副本。子图划分已定型, 该合并不影响连通性。
Status MergeDuplicatedCopyNodesInSubGraph(ascir::ImplGraph &sub_graph) {
  std::map<std::pair<std::string, std::string>, std::vector<af::AscNodePtr>> copies_by_key;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (node->GetName().rfind("copy_from_", 0U) != 0U) {
      continue;
    }
    copies_by_key[std::make_pair(node->GetName(), node->GetType())].push_back(node);
  }
  for (auto &item : copies_by_key) {
    std::vector<af::AscNodePtr> &copies = item.second;
    if (copies.size() <= 1UL) {
      continue;
    }
    std::sort(copies.begin(), copies.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });
    const af::AscNodePtr &canonical = copies.front();
    for (size_t i = 1UL; i < copies.size(); ++i) {
      GELOGI("Merge duplicated copy node [%s](id=%lld) into [%s](id=%lld), sub graph [%s].", copies[i]->GetNamePtr(),
             copies[i]->GetOpDescBarePtr()->GetId(), canonical->GetNamePtr(), canonical->GetOpDescBarePtr()->GetId(),
             sub_graph.GetName().c_str());
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveNode(sub_graph, copies[i], canonical->GetOutDataAnchor(0UL)),
                        "Merge duplicated copy node [%s] failed.", copies[i]->GetName().c_str());
    }
  }
  bool has_removed = false;
  do {
    has_removed = false;
    std::vector<af::AscNodePtr> dead_copies;
    for (const auto &node : sub_graph.GetAllNodes()) {
      if (node->GetName().rfind("copy_from_", 0U) == 0U && node->GetOutDataNodes().empty()) {
        dead_copies.push_back(node);
      }
    }
    for (const auto &node : dead_copies) {
      GELOGD("Remove dead copy node [%s](id=%lld), sub graph [%s].", node->GetNamePtr(),
             node->GetOpDescBarePtr()->GetId(), sub_graph.GetName().c_str());
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveNode(sub_graph, node, node->GetOutDataAnchor(0UL)),
                        "Remove dead copy node [%s] failed.", node->GetName().c_str());
      has_removed = true;
    }
  } while (has_removed);
  return af::SUCCESS;
}

}  // namespace

Status ReducePartitionCaseGenerator::GeneratorGeneralTask(ascir::HintGraph &optimize_graph,
                                                          std::vector<ScheduleTask> &tasks) {
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(GenerateGeneralCase(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {}, ReduceTemplateType::kCommon};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs, node_order_),
                      "Failed to partition graph");
    if (task.grouped_graphs.empty()) {
      GELOGW("Graph[%s] has no subgraphs after partition, skip task", graph.GetName().c_str());
      continue;
    }
    // 子图划分完成后, 在各自子图内合并重复的 copy_from_* 副本(子图已定型, 不影响连通性)
    for (auto &grouped_graph : task.grouped_graphs) {
      GE_CHK_STATUS_RET(MergeDuplicatedCopyNodesInSubGraph(grouped_graph), "Failed to merge duplicated copy nodes");
    }
    tasks.emplace_back(std::move(task));
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorAllLoadTask(ascir::HintGraph &optimize_graph,
                                                          std::vector<ScheduleTask> &tasks) {
  if (ascgen_utils::GetAscIrCodegenImpl("Softmax") != nullptr) {
    SoftmaxPatternFusionPass softmax_pattern_fusion_pass;
    GE_CHK_STATUS_RET(softmax_pattern_fusion_pass.RunPass(optimize_graph));
  }

  if (!CanFullLoadReduceFuse(optimize_graph)) {
    GELOGD("Graph %s does not support FullLoadReduceFuse, skip AllLoad task generation.",
           optimize_graph.GetName().c_str());
    return ge::GRAPH_SUCCESS;
  }
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(GenerateAllLoadCase(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {}, ReduceTemplateType::kAllLoad};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs, node_order_),
                      "Failed to partition graph");
    tasks.emplace_back(std::move(task));
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorRCoreTask(ascir::HintGraph &optimize_graph,
                                                        std::vector<ScheduleTask> &tasks) const {
  std::vector<ScheduleTask> new_tasks;
  for (const auto &task : tasks) {
    if (task.reduce_type != ReduceTemplateType::kCommon) {
      continue;
    }
    if (HasArgMaxReduce(task)) {
      GELOGI("Task contains ArgMax reduce op, skip R-core template generation");
      continue;
    }
    std::vector<::ascir::ImplGraph> new_task_grouped_graphs;
    std::map<size_t, std::vector<size_t>> map;
    size_t phase_2_graph_size = 0;
    for (size_t i = 0; i < task.grouped_graphs.size(); i++) {
      GE_ASSERT_TRUE(IsOnlyHasOneOrLessReduce(task.grouped_graphs[i]));
      if (!HasReduce(task.grouped_graphs[i])) {
        ::ascir::ImplGraph graph((task.grouped_graphs[i].GetName() + "_r_multicore").c_str());
        graph.CopyFrom(task.grouped_graphs[i]);
        new_task_grouped_graphs.emplace_back(std::move(graph));
        continue;
      }
      ascir::ImplGraph phase_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_graph").c_str());
      phase_graph.CopyFrom(task.grouped_graphs[i]);
      af::AscNodePtr reduce_node;
      for (auto node : phase_graph.GetAllNodes()) {
        if (ScheduleUtils::IsReduce(node)) {
          reduce_node = node;
          break;
        }
      }
      GE_CHECK_NOTNULL(reduce_node);
      GE_ASSERT_TRUE(IsNotPartitionReduce(reduce_node, NODE_COUNT_AFTER_REDUCE));
      ascir::ImplGraph phase_1_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_1_graph").c_str());
      ascir::ImplGraph phase_2_graph((task.grouped_graphs[i].GetName() + "_r_multicore_phase_2_graph").c_str());
      GE_CHK_STATUS_RET(RMulticorePhase2Graph(phase_2_graph, phase_1_graph, phase_graph, reduce_node).Construct());
      ascir::utils::DumpGraph(phase_2_graph, "phase2graph_construct");
      new_task_grouped_graphs.emplace_back(std::move(phase_1_graph));
      new_task_grouped_graphs.emplace_back(std::move(phase_2_graph));
      map[i + phase_2_graph_size] = {i + phase_2_graph_size + 1};
      phase_2_graph_size++;
    }
    if (phase_2_graph_size == 0) {
      continue;
    }
    ScheduleTask new_task{optimize_graph, new_task_grouped_graphs, task.score_func, map, ReduceTemplateType::kRCore};
    new_tasks.push_back(new_task);
  }
  tasks.insert(tasks.end(), new_tasks.begin(), new_tasks.end());
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                                                   const OptimizerOptions &options) {
  (void)options;
  const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(optimize_graph);
  if (indirect_load != nullptr) {
    GELOGI("Graph %s has indirect load, skip reduce task generation; handled by IndirectLoad tasks.",
           optimize_graph.GetName().c_str());
    return ge::GRAPH_SUCCESS;
  }
  const bool force_all_load = ShouldForceAllLoad(optimize_graph);
  GELOGI("Graph %s force AllLoad = %d, begin to generate reduce tasks.", optimize_graph.GetName().c_str(),
         static_cast<int32_t>(force_all_load));
  GE_CHK_STATUS_RET(GeneratorGeneralTask(optimize_graph, tasks));
  GELOGI("After GeneralTask, graph %s has %zu task(s).", optimize_graph.GetName().c_str(), tasks.size());
  if (!force_all_load) {
    GE_CHK_STATUS_RET(GeneratorRCoreTask(optimize_graph, tasks));
    GELOGI("After RCoreTask, graph %s has %zu task(s).", optimize_graph.GetName().c_str(), tasks.size());
  }
  GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));
  GELOGI("After AllLoadTask, graph %s has %zu task(s).", optimize_graph.GetName().c_str(), tasks.size());
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::Generate([[maybe_unused]] ascir::HintGraph &graph,
                                              [[maybe_unused]] std::vector<ascir::ImplGraph> &graphs,
                                              [[maybe_unused]] std::vector<std::string> &score_functions) {
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::ShouldForceAllLoad(ascir::HintGraph &graph) {
  if (!IsOnlyHasOneOrLessReduce(graph)) {
    return true;
  }

  ascir::ImplGraph impl_graph(graph.GetName().c_str());
  impl_graph.CopyFrom(graph);
  node_order_.clear();
  auto loop_start_end = FindNormLoops(impl_graph);
  if (!loop_start_end.empty()) {
    return true;
  }

  return false;
}

Status ReducePartitionCaseGenerator::GenerateGeneralCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                                         std::vector<std::string> &score_functions) {
  if (!HasReduce(graph)) {
    return ge::GRAPH_SUCCESS;
  }
  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);

  node_order_.clear();
  auto loop_start_end = FindNormLoops(optimize_graph);
  std::sort(
      loop_start_end.begin(), loop_start_end.end(),
      [](const std::pair<af::AscNodePtr, af::AscNodePtr> &lhs, const std::pair<af::AscNodePtr, af::AscNodePtr> &rhs) {
        return lhs.second->GetOpDescBarePtr()->GetId() < rhs.second->GetOpDescBarePtr()->GetId();
      });

  LogNormLoops("FindNormLoops", loop_start_end, 0UL, loop_start_end.size());

  // reduce 后融合切分
  GE_CHK_STATUS_RET(ReducePartitionPostFusion(optimize_graph));
  ascir::utils::DumpGraph(optimize_graph, "after_post_fusion");

  // 按照前面获取的环路起点、终点，进行norm的切分
  GE_CHK_STATUS_RET(PartitionNorm(optimize_graph, loop_start_end));
  ascir::utils::DumpGraph(optimize_graph, "after_partition_norm");

  // reduce 多引用结构需要切分的，切分开
  GE_CHK_STATUS_RET(ReducePartitionMultipleCitations(optimize_graph));
  ascir::utils::DumpGraph(optimize_graph, "after_multiple_citations");

  // 补充类 Norm 结构 loops: 上述切分动作建立 workspace 边界后才能探测到被边界绕过的类 Norm
  // 结构, 对新增 loops 再执行一次 norm 切分
  const size_t norm_loop_count = loop_start_end.size();
  GE_CHK_STATUS_RET(AddNormLikeLoops(optimize_graph, loop_start_end));
  LogNormLoops("AddNormLikeLoops", loop_start_end, norm_loop_count, loop_start_end.size());
  if (loop_start_end.size() > norm_loop_count) {
    std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> norm_like_loops(
        loop_start_end.begin() + static_cast<int64_t>(norm_loop_count), loop_start_end.end());
    GE_CHK_STATUS_RET(PartitionNorm(optimize_graph, norm_like_loops));
    ascir::utils::DumpGraph(optimize_graph, "after_partition_norm_like");
  }

  // 公共动作: 摘除三个切分步骤(PostFusion/PartitionNorm/多引用切分)产生的悬空节点
  GE_CHK_STATUS_RET(RemoveDanglingNodes(optimize_graph));
  ascir::utils::DumpGraph(optimize_graph, "after_remove_dangling_nodes");

  if (partition_) {
    std::sort(node_order_.begin(), node_order_.end(), [](const af::AscNodePtr &lhs, af::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });

    graphs.emplace_back(optimize_graph);
    score_functions.resize(graphs.size());
  } else {
    node_order_.clear();
    graphs.emplace_back(graph);
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GenerateBroadcastWorkspaceCase(ascir::HintGraph &graph,
                                                                    std::vector<ascir::ImplGraph> &graphs) {
  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);
  partition_ = false;
  for (const auto &node : optimize_graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty()) {
      node_order_.emplace_back(node);
    }
  }
  std::vector<af::AscNodePtr> broadcast_nodes;
  for (const auto &node : optimize_graph.GetAllNodes()) {
    if (NeedWorkspaceBeforeBroadcast(node)) {
      broadcast_nodes.emplace_back(node);
    }
  }
  for (auto &broadcast_node : broadcast_nodes) {
    auto pre_node = std::dynamic_pointer_cast<af::AscNode>(broadcast_node->GetInDataNodes().at(0UL));
    GE_ASSERT_NOTNULL(pre_node);
    GE_CHK_STATUS_RET(PartitionByNode(pre_node, broadcast_node, optimize_graph));
  }
  if (partition_) {
    std::sort(node_order_.begin(), node_order_.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });
    graphs.emplace_back(optimize_graph);
  } else {
    node_order_.clear();
    graphs.emplace_back(graph);
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::GenerateAllLoadCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                                         const std::vector<std::string> &score_functions) {
  (void)score_functions;
  if (!HasReduce(graph)) {
    return ge::GRAPH_SUCCESS;
  }
  node_order_.clear();
  return GenerateBroadcastWorkspaceCase(graph, graphs);
}

Status ReducePartitionCaseGenerator::ReducePartitionMultipleCitations(ascir::ImplGraph &impl_graph) {
  if (IsOnlyHasOneOrLessReduce(impl_graph)) {
    return ge::GRAPH_SUCCESS;
  }
  CitationGroups citation_groups;
  GE_CHK_STATUS_RET(CollectCitationGroups(impl_graph, citation_groups));
  std::vector<size_t> parent(citation_groups.size());
  std::map<size_t, af::AscNodePtr> group_anchors;
  BuildCitationGroupAnchors(citation_groups, parent, group_anchors);
  return PartitionCitationGroups(impl_graph, citation_groups, parent, group_anchors);
}

// 手工注释：解决有时会产生后续子图内只有一个节点的情况
// 公共动作(在 ReducePartitionPostFusion / PartitionNorm / ReducePartitionMultipleCitations
// 三个切分步骤之后、子图划分之前执行): 反复摘除"没有数据输出且非固有汇点"的节点, 直到没有
// 新节点被摘除——末端节点摘除后上游可能失去最后一个消费者, 因此需要重复处理。控制边不计入
// "有输出"(它只是保活/签名用途, 随节点摘除一并清理, RemoveNode 会处理 out/in 控制边)。
// 固有汇点为本来就不应该有输出的节点: 图输出 Output 与 workspace 写侧(store->ws_pre)。
// 重复切分残留的悬空副本(含同名副本, 不限类型)、失去消费者的源节点、死分支均在摘除范围内;
// 存活的链与副本各自保留。
Status ReducePartitionCaseGenerator::RemoveDanglingNodes(ascir::ImplGraph &impl_graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  GE_CHECK_NOTNULL(compute_graph);
  std::set<const af::Node *> nodes_in_graph;
  bool has_removed = false;
  do {
    has_removed = false;
    std::vector<af::NodePtr> nodes_to_remove;
    nodes_in_graph.clear();
    for (const auto &node : impl_graph.GetAllNodes()) {
      nodes_in_graph.insert(node.get());
    }
    // 先收集再删除, 避免遍历过程中修改图节点容器
    for (const auto &node : impl_graph.GetAllNodes()) {
      if (IsInherentSink(node) || HasConsumerInGraph(node, nodes_in_graph)) {
        continue;
      }
      nodes_to_remove.push_back(node);
    }
    for (const auto &node : nodes_to_remove) {
      GELOGD("Remove dangling node [%s], type [%s], graph [%s].", node->GetNamePtr(), node->GetTypePtr(),
             impl_graph.GetName().c_str());
      GE_ASSERT_GRAPH_SUCCESS(compute_graph->RemoveNode(node));
      has_removed = true;
    }
  } while (has_removed);
  // 被摘除的节点若残留在切分根序列中, PartitionByConnectivity 会按根重建出孤立子图
  node_order_.erase(
      std::remove_if(node_order_.begin(), node_order_.end(),
                     [&nodes_in_graph](const af::AscNodePtr &node) { return nodes_in_graph.count(node.get()) == 0U; }),
      node_order_.end());
  return af::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::CollectCitationGroups(ascir::ImplGraph &impl_graph,
                                                           CitationGroups &citation_groups) {
  for (auto node : impl_graph.GetAllNodes()) {
    if (node->GetOutDataNodes().size() > 1UL) {
      std::vector<Citation> citations;
      for (const auto &output_node : node->GetOutDataNodes()) {
        auto citation = std::dynamic_pointer_cast<af::AscNode>(output_node);
        GE_CHECK_NOTNULL(citation);
        af::AscNodePtr reduce;
        if (FindOutputReduce(citation, reduce)) {
          citations.push_back({node, citation, reduce});
        }
      }
      if (!citations.empty()) {
        std::sort(citations.begin(), citations.end(), [](const Citation &lhs, const Citation &rhs) {
          if (lhs.reduce->GetOpDescBarePtr()->GetId() != rhs.reduce->GetOpDescBarePtr()->GetId()) {
            return lhs.reduce->GetOpDescBarePtr()->GetId() < rhs.reduce->GetOpDescBarePtr()->GetId();
          }
          return lhs.citation->GetOpDescBarePtr()->GetId() < rhs.citation->GetOpDescBarePtr()->GetId();
        });
        citation_groups.emplace_back(std::move(citations));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

void ReducePartitionCaseGenerator::BuildCitationGroupAnchors(const CitationGroups &citation_groups,
                                                             std::vector<size_t> &parent,
                                                             std::map<size_t, af::AscNodePtr> &group_anchors) {
  std::iota(parent.begin(), parent.end(), 0UL);
  auto find_root = [&parent](size_t index) {
    while (parent[index] != index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  for (size_t i = 0UL; i < citation_groups.size(); ++i) {
    for (size_t j = i + 1UL; j < citation_groups.size(); ++j) {
      bool shared_reduce = false;
      for (const auto &lhs : citation_groups[i]) {
        for (const auto &rhs : citation_groups[j]) {
          if (lhs.reduce == rhs.reduce) {
            shared_reduce = true;
            break;
          }
        }
        if (shared_reduce) {
          break;
        }
      }
      if (shared_reduce) {
        parent[find_root(j)] = find_root(i);
      }
    }
  }

  for (size_t i = 0UL; i < citation_groups.size(); ++i) {
    const auto root = find_root(i);
    for (const auto &citation : citation_groups[i]) {
      auto anchor = group_anchors.find(root);
      if (anchor == group_anchors.end() ||
          citation.reduce->GetOpDescBarePtr()->GetId() < anchor->second->GetOpDescBarePtr()->GetId()) {
        group_anchors[root] = citation.reduce;
      }
    }
  }
}

Status ReducePartitionCaseGenerator::PartitionCitationGroups(ascir::ImplGraph &impl_graph,
                                                             const CitationGroups &citation_groups,
                                                             const std::vector<size_t> &parent,
                                                             const std::map<size_t, af::AscNodePtr> &group_anchors) {
  auto find_root = [&parent](size_t index) {
    while (parent[index] != index) {
      index = parent[index];
    }
    return index;
  };
  // 多引用切分:
  // 同一 source 的多条引用可能到达同一个非锚点 reduce(如 S->C1->...->R2 与 S->C2->...->R2),
  // 若只切分第一条会残留 source->C2 直连边, source 无法与该 reduce 子图解耦(连通性分组时
  // 整体仍归并为一个组, 切分失效)。因此按 (source, citation, reduce) 逐条检测切分: 消费者与
  // source 间数据边仍存在才切分, 边已消失(重复收集的引用或消费者多次消费 source 时首次切分
  // 已改接全部边)即跳过。逐条独立切分产生的副本链各自保留, 不做合并, 避免误并已切分的
  // workspace 边界。
  for (size_t i = 0UL; i < citation_groups.size(); ++i) {
    const size_t root = find_root(i);
    const af::AscNodePtr &anchor = group_anchors.at(root);
    for (const auto &citation : citation_groups[i]) {
      if (citation.reduce == anchor) {
        continue;  // 锚点reduce的引用保留与source直连, 作为融合基准
      }
      if (!HasDataEdgeBetween(citation.source, citation.citation)) {
        continue;  // 该引用已被切分改接, 无需重复处理
      }
      af::AscNodePtr source = citation.source;
      af::AscNodePtr citation_node = citation.citation;
      GE_CHK_STATUS_RET(PartitionByNode(source, citation_node, impl_graph));
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::FindOutputReduce(const af::AscNodePtr &node, af::AscNodePtr &reduce_node) {
  if (ScheduleUtils::IsReduce(node)) {
    reduce_node = node;
    return true;
  }
  bool output_has_reduce = false;
  if (node->GetOutDataNodes().empty()) {
    return output_has_reduce;
  }
  for (const auto &output_node : node->GetOutDataNodes()) {
    auto output_asc_node = std::dynamic_pointer_cast<af::AscNode>(output_node);
    output_has_reduce = output_has_reduce || FindOutputReduce(output_asc_node, reduce_node);
  }
  return output_has_reduce;
}

Status ReducePartitionCaseGenerator::PartitionReduceNode(af::AscNodePtr &src_node, ascir::ImplGraph &impl_graph) {
  partition_ = true;
  node_order_.emplace_back(src_node);
  af::ascir_op::Workspace workspace_pre((src_node->GetName() + "_Workspace").c_str());
  af::ascir_op::Workspace workspace_post((src_node->GetName() + "_Workspace").c_str());
  af::ascir_op::Load load((src_node->GetName() + "_Load").c_str());
  af::ascir_op::Store store((src_node->GetName() + "_Store").c_str());
  auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
  auto workspace_post_node = impl_graph.AddNode(workspace_post);
  auto load_node = impl_graph.AddNode(load);
  auto store_node = impl_graph.AddNode(store);
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, load_node));
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, store_node));
  GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      auto dst_node = peer_in_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dst_node, "peer node is null, src node: %s", src_node->GetNamePtr());
      // remove src->dst
      GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                   dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
    }
  }
  // add src->store->workspace_pre_node
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(src_node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
  // add workspace_post_node->load->dst
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::ReducePartitionPostFusion(ascir::ImplGraph &impl_graph) {
  for (auto node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      if (IsNotPartitionReduce(node, NODE_COUNT_AFTER_REDUCE)) {
        continue;
      }
      GE_CHK_STATUS_RET(PartitionReduceNode(node, impl_graph));
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionLoadNode(af::AscNodePtr &src_load_node, af::AscNodePtr &dst_node,
                                                       ascir::ImplGraph &impl_graph) {
  auto load_input_node = src_load_node->GetInNodes().at(0UL);
  auto load_input_asc_node = std::dynamic_pointer_cast<af::AscNode>(load_input_node);
  GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(load_input_asc_node) ||
                 af::ops::IsOps<af::ascir_op::Workspace>(load_input_asc_node));
  af::ascir_op::Load load(("copy_from_" + src_load_node->GetName()).c_str());

  af::AscNodePtr new_load_input_node;
  if (af::ops::IsOps<af::ascir_op::Data>(load_input_asc_node)) {
    af::ascir_op::Data data(("copy_from_" + load_input_asc_node->GetName()).c_str());
    new_load_input_node = impl_graph.AddNode(data);
  } else {
    af::ascir_op::Workspace workspace(("copy_from_" + load_input_asc_node->GetName()).c_str());
    new_load_input_node = impl_graph.AddNode(workspace);
  }
  auto load_node = impl_graph.AddNode(load);
  DoCopyAscNodeTensorAttr(load_input_asc_node, new_load_input_node);
  DoCopyAscNodeTensorAttr(src_load_node, load_node);
  // dst_node 的多个输入可能来自同一个 src_load_node（如 Mul 的两个输入都是同一个 Load），
  // 需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  // new_load_input -> load_node 的边只需建一次，后续匹配的 peer 边共用这条链路。
  bool found = false;
  for (const auto &out_anchor : src_load_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_load_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_load_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        if (!found) {
          found = true;
          GE_CHK_STATUS_RET(
              af::GraphUtils::AddEdge(new_load_input_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
        }
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionScalarNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                                         ascir::ImplGraph &impl_graph) {
  af::AscNodePtr scalar_node;
  if (af::ops::IsOps<af::ascir_op::ScalarData>(src_node)) {
    af::ascir_op::ScalarData scalar_data(("copy_from_" + src_node->GetName()).c_str());
    scalar_node = impl_graph.AddNode(scalar_data);
  } else {
    af::ascir_op::Scalar scalar(("copy_from_" + src_node->GetName()).c_str());
    scalar_node = impl_graph.AddNode(scalar);
  }
  DoCopyAscNodeTensorAttr(src_node, scalar_node);
  // dst_node 的多个输入可能来自同一个 src_node，需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(scalar_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionDataNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                                       ascir::ImplGraph &impl_graph) {
  // Data 是 GM 输入占位符（kAPITypeBuffer），值直接来自 kernel 输入参数，不能走通用路径用 Store 物化
  // （Data 直连 Store 会生成 DataCopyPadExtend(GlobalTensor, ...) 无匹配重载的非法调用）。
  // 复制 Data 节点让 citation 侧子图重读同一 GM 输入，ir_attr 携带的输入参数 index 随属性克隆保留。
  af::ascir_op::Data data(("copy_from_" + src_node->GetName()).c_str());
  auto data_node = impl_graph.AddNode(data);
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, data_node));
  // dst_node 的多个输入可能来自同一个 src_node，需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(data_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::PartitionByNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                                     ascir::ImplGraph &impl_graph) {
  partition_ = true;
  node_order_.emplace_back(src_node);
  if (ScheduleUtils::IsLoad(src_node)) {
    return PartitionLoadNode(src_node, dst_node, impl_graph);
  }
  if (ScheduleUtils::IsScalarLikeNode(src_node)) {
    return PartitionScalarNode(src_node, dst_node, impl_graph);
  };
  if (af::ops::IsOps<af::ascir_op::Data>(src_node)) {
    return PartitionDataNode(src_node, dst_node, impl_graph);
  }

  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    af::ascir_op::Workspace workspace_pre(
        GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    af::ascir_op::Workspace workspace_post(
        GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    af::ascir_op::Load load(GetNewNodeName(src_node, dst_node, "Load", out_anchor->GetIdx()).c_str());
    af::ascir_op::Store store(GetNewNodeName(src_node, dst_node, "Store", out_anchor->GetIdx()).c_str());
    auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
    auto workspace_post_node = impl_graph.AddNode(workspace_post);
    auto load_node = impl_graph.AddNode(load);
    auto store_node = impl_graph.AddNode(store);
    GE_CHK_STATUS_RET(DoCopyAscNodeOutputTensorAttr(src_node, load_node, out_anchor->GetIdx()));
    GE_CHK_STATUS_RET(DoCopyAscNodeOutputTensorAttr(src_node, store_node, out_anchor->GetIdx()));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        // remove src->dst
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
    // add src->store->workspace_pre_node
    GE_CHK_STATUS_RET(
        af::GraphUtils::AddEdge(src_node->GetOutAnchor(out_anchor->GetIdx()), store_node->GetInAnchor(0UL)));
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
    // add workspace_post_node->load->dst
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::IsInputNodePartitioned(const std::shared_ptr<af::Node> &start,
                                                          const std::shared_ptr<af::Node> &node) {
  // 向前找 start 节点，如找到则未经过切分，到根节点未找到则已切分或不需要切分
  if (node == start) {
    return false;
  }
  bool partitioned = true;
  if (node->GetInNodes().empty()) {
    return partitioned;
  }
  for (const auto &in_node : node->GetInNodes()) {
    partitioned = partitioned && IsInputNodePartitioned(start, in_node);
  }
  return partitioned;
}

std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> ReducePartitionCaseGenerator::FindNormLoops(
    ascir::ImplGraph &graph) {
  std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> loop_start_end;
  for (auto start_node : graph.GetAllNodes()) {
    if (start_node->GetOutDataNodes().empty()) {
      node_order_.emplace_back(start_node);
    }
    if (start_node->GetOutNodes().size() <= 1UL) {
      continue;
    }
    std::vector<af::AscNodePtr> loop_ends;
    FindNormLoop(start_node, loop_ends);
    for (const auto &end_node : loop_ends) {
      loop_start_end.emplace_back(start_node, end_node);
    }
  }
  return loop_start_end;
}

// 手工注释：该场景由交叉norm结构触发，即并不是直连的norm结构，而是间接连接形成的norm连通结构。（如A->B则AB连通，不看输入输出关系）
// 补充类 Norm 结构 loops: FindNormLoops 只沿输出方向从多输出节点展开, 这里在其结果基础上
// 补充非单输出方向的类 Norm 结构, 追加到 loop_start_end
Status ReducePartitionCaseGenerator::AddNormLikeLoops(
    ascir::ImplGraph &graph, std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &loop_start_end) {
  const std::map<std::string, af::AscNodePtr> post_ws_load_map = GetPostWorkspaceLoadMap(graph);
  if (post_ws_load_map.empty()) {
    return ge::GRAPH_SUCCESS;
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Workspace>(node) || node->GetInDataNodes().empty()) {
      continue;
    }
    const auto load_iter = post_ws_load_map.find(node->GetName());
    if (load_iter == post_ws_load_map.cend()) {
      continue;
    }
    // 被 workspace 切分的节点对: {pre_workspace 的输入, post_workspace 的输出}
    const af::AscNodePtr node_before_workspace = std::dynamic_pointer_cast<af::AscNode>(node->GetInDataNodes().at(0UL));
    const af::AscNodePtr node_after_workspace = load_iter->second;
    GE_ASSERT_NOTNULL(node_before_workspace);
    GE_ASSERT_NOTNULL(node_after_workspace);
    // 从 node_after_workspace 出发沿输出方向找多输入节点
    std::set<af::Node *> visited{node_after_workspace.get()};
    std::vector<af::AscNodePtr> pending{node_after_workspace};
    while (!pending.empty()) {
      const af::AscNodePtr current = pending.back();
      pending.pop_back();
      for (const auto &out_node : current->GetOutDataNodes()) {
        const af::AscNodePtr out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
        GE_ASSERT_NOTNULL(out_asc_node);
        if (!visited.insert(out_asc_node.get()).second) {
          continue;
        }
        pending.emplace_back(out_asc_node);
        if (out_asc_node->GetInDataNodesSize() <= 1UL) {
          continue;
        }
        // 多输入节点与 node_before_workspace 连通的输入节点, 与该多输入节点组成类 Norm 环路
        for (const auto &in_node : out_asc_node->GetInDataNodes()) {
          const af::AscNodePtr connected_input = std::dynamic_pointer_cast<af::AscNode>(in_node);
          GE_ASSERT_NOTNULL(connected_input);
          if (IsConnectedThroughWorkspace(connected_input, node_before_workspace, post_ws_load_map, out_asc_node)) {
            loop_start_end.emplace_back(connected_input, out_asc_node);
          }
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status ReducePartitionCaseGenerator::FindNormLoop(const af::AscNodePtr &start, std::vector<af::AscNodePtr> &ends) {
  std::set<af::NodePtr> visited{start};
  std::list<af::NodePtr> next_nodes{start};
  while (!next_nodes.empty()) {
    const auto cur_node = next_nodes.front();
    next_nodes.pop_front();
    for (auto &out_node : cur_node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      } else {
        auto asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
        ends.emplace_back(asc_node);
      }
    }
  }
  ends.erase(std::remove_if(ends.begin(), ends.end(),
                            [start, this](const af::AscNodePtr &end) { return !IsNorm(start, end); }),
             ends.end());
  return ge::GRAPH_SUCCESS;
}

void ReducePartitionCaseGenerator::FindAllPath(const af::AscNodePtr &start, const af::AscNodePtr &end,
                                               std::vector<af::AscNodePtr> &path,
                                               std::vector<std::vector<af::AscNodePtr>> &all_paths) {
  // 需保证图为有向无环图
  path.emplace_back(start);
  if (start == end) {
    all_paths.emplace_back(path);
  }
  for (const auto &out_node : start->GetOutNodes()) {
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    FindAllPath(asc_node, end, path, all_paths);
  }
  path.pop_back();
}

bool ReducePartitionCaseGenerator::IsNorm(const af::AscNodePtr &start, const af::AscNodePtr &end) {
  std::vector<af::AscNodePtr> path;
  std::vector<std::vector<af::AscNodePtr>> all_paths;
  FindAllPath(start, end, path, all_paths);
  bool is_norm = false;
  std::set<af::AscNodePtr> end_in_nodes;
  for (const auto &path_temp : all_paths) {
    for (const auto &node : path_temp) {
      if (ScheduleUtils::IsReduce(node)) {
        is_norm = true;
        break;
      }
    }
    end_in_nodes.insert(path_temp.at(path_temp.size() - TWO));
  }
  return is_norm && end_in_nodes.size() > 1;
}

Status ReducePartitionCaseGenerator::PartitionNorm(
    ascir::ImplGraph &impl_graph, std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &loop_start_end) {
  for (auto loop : loop_start_end) {
    for (auto &in_node : loop.second->GetInNodes()) {
      if (IsInputNodePartitioned(loop.first, in_node)) {
        continue;
      }
      af::AscNodePtr src_node = std::dynamic_pointer_cast<af::AscNode>(in_node);
      GELOGI("PartitionNorm: loop start=%s end=%s, partition src_node=%s -> dst(loop.end)=%s",
             loop.first->GetName().c_str(), loop.second->GetName().c_str(), src_node->GetName().c_str(),
             loop.second->GetName().c_str());
      GE_CHK_STATUS_RET(PartitionByNode(src_node, loop.second, impl_graph));
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool ReducePartitionCaseGenerator::HasReduce(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      return true;
    }
  }
  return false;
}

bool ReducePartitionCaseGenerator::HasArgMaxReduce(const ScheduleTask &task) {
  for (const auto &grouped_graph : task.grouped_graphs) {
    if (!HasReduce(grouped_graph)) {
      continue;
    }
    for (const auto &node : grouped_graph.GetAllNodes()) {
      if (ScheduleUtils::IsReduce(node) && node->GetType() == "ArgMax") {
        return true;
      }
    }
  }
  return false;
}

// 全载模板支持历史 AR/ARA 类短轴场景，并额外支持 AA...AR 尾轴 reduce 场景。
bool ReducePartitionCaseGenerator::CanFullLoadReduceFuse(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!ScheduleUtils::IsReduce(node)) {
      continue;
    }
    std::vector<ascir::SizeExpr> input_repeats = node->inputs[0].attr.repeats;
    std::vector<ascir::SizeExpr> output_repeats = node->outputs[0].attr.repeats;
    GE_ASSERT_TRUE(input_repeats.size() == output_repeats.size());
    if (output_repeats.empty()) {
      return false;
    }

    if (!IsLegacyFullLoadReduce(input_repeats, output_repeats) &&
        !IsTailReduceFullLoad(node, input_repeats, output_repeats)) {
      return false;
    }
  }

  return true;
}

bool ReducePartitionCaseGenerator::IsOnlyHasOneOrLessReduce(const ascir::ImplGraph &impl_graph) {
  int reduce_count = 0;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsReduce(node)) {
      reduce_count += 1;
    }
  }
  return reduce_count <= 1;
}

Status RMulticorePhase2Graph::Construct() {
  GE_ASSERT_TRUE(reducers.find(reduce_node->GetType()) != reducers.end());
  ReduceType phase2graph_reduce =
      reducers.find(reduce_node->GetType())
          ->second((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  std::visit([](auto &&reduce_op) { reduce_op.attr.sched.axis = {0, 1}; }, phase2graph_reduce);

  // 对于ArgMax，需要在调用CompletePhaseGraph之前保存输入dtype（value类型）
  // 因为CompletePhaseGraph会调用PartitionByReduce，后者会删除reduce_node
  af::AscTensorDataType argmax_input_dtype;
  if (reduce_node->GetType() == "ArgMax") {
    GE_ASSERT_TRUE(!reduce_node->inputs().empty(), "ArgMax node should have at least 1 input");
    argmax_input_dtype = reduce_node->inputs[0].attr.dtype;
  }

  GE_CHK_STATUS_RET(CompletePhaseGraph(phase2graph_reduce));
  GE_CHK_STATUS_RET(CreateVarAxis());

  auto workspace_node = phase2graph.FindNode((phase2graph.GetName() + "_workspace").c_str());
  GE_ASSERT_NOTNULL(workspace_node);
  workspace_node->attr.sched.axis = {0, 1};
  // 对于ArgMax，workspace存储的是value（输入类型），不是index（输出类型）
  af::AscTensorDataType workspace_dtype;
  if (reduce_node->GetType() == "ArgMax") {
    workspace_dtype = argmax_input_dtype;
  } else {
    workspace_dtype = reduce_node->outputs[0].attr.dtype;
  }
  GE_CHK_STATUS_RET(CompleteNodeAttr(workspace_node, true, workspace_dtype));
  auto load_node = phase2graph.FindNode((phase2graph.GetName() + "_load").c_str());
  GE_ASSERT_NOTNULL(load_node);
  load_node->attr.sched.axis = {0, 1};
  // load从workspace读取，dtype与workspace相同
  GE_CHK_STATUS_RET(CompleteNodeAttr(load_node, true, workspace_dtype));

  // ArgMax特殊处理：需要设置load_index_node和workspace_index_node的属性
  GE_CHK_STATUS_RET(SetupArgMaxIndexNodes(reduce_node, phase2graph));

  auto reduce_node_parse2graph =
      phase2graph.FindNode((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  GE_ASSERT_NOTNULL(reduce_node_parse2graph);
  std::set<af::NodePtr> visited{reduce_node_parse2graph};
  std::list<af::NodePtr> next_nodes{reduce_node_parse2graph};
  while (!next_nodes.empty()) {
    const auto node = next_nodes.front();
    next_nodes.pop_front();
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    asc_node->attr.sched.axis = {0, 1};
    GE_CHK_STATUS_RET(CompleteNodeAttr(asc_node, false, asc_node->outputs[0].attr.dtype));
    for (auto &out_node : node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CreateVarAxis() {
  // 创建符号：A轴符号：s1，R轴符号：s2
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(phase2graph);
  GE_ASSERT_NOTNULL(compute_graph);
  auto attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(attr);
  attr->axis.clear();
  attr->size_vars.clear();
  Rm_org_size = phase2graph.CreateSizeVar("Rm_org_size");
  A_org_size = phase2graph.CreateSizeVar("A_org_size");
  // 创建轴：a，r
  phase2graph.CreateAxis("Rm", Rm_org_size);
  phase2graph.CreateAxis("A", A_org_size);
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CompleteNodeAttr(af::AscNodePtr &node, bool before_reduce,
                                               const af::AscTensorDataType &data_type) {
  node->outputs[0].attr.dtype = data_type;
  node->outputs[0].attr.axis = {0, 1};
  if (before_reduce) {
    node->outputs[0].attr.strides = {A_org_size, af::ops::One};
    node->outputs[0].attr.repeats = {Rm_org_size, A_org_size};
  } else {
    node->outputs[0].attr.strides = {af::ops::Zero, af::ops::One};
    node->outputs[0].attr.repeats = {af::ops::One, A_org_size};
  }
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CompleteNodeAttrBeforeReduce(af::AscNodePtr &node) {
  node->outputs[0].attr.axis = {0, 1};
  node->outputs[0].attr.strides = {A_org_size, af::ops::One};
  node->outputs[0].attr.repeats = {Rm_org_size, A_org_size};
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::SetupArgMaxIndexNodes(const af::AscNodePtr &reduce_node, ascir::ImplGraph &phase2graph) {
  // ArgMax特殊处理：需要设置load_index_node和workspace_index_node的属性
  // 注意：虽然这些设置可能被后续的CopyFrom部分覆盖，但轴信息(strides, repeats等)需要保留
  if (reduce_node->GetType() == "ArgMax") {
    auto workspace_index_node = phase2graph.FindNode((phase2graph.GetName() + "_workspace_index").c_str());
    if (workspace_index_node != nullptr) {
      workspace_index_node->attr.sched.axis = {0, 1};
      GE_CHK_STATUS_RET(CompleteNodeAttrBeforeReduce(workspace_index_node));
    }

    auto index_load_node = phase2graph.FindNode((phase2graph.GetName() + "_load_index").c_str());
    if (index_load_node != nullptr) {
      index_load_node->attr.sched.axis = {0, 1};
      GE_CHK_STATUS_RET(CompleteNodeAttrBeforeReduce(index_load_node));
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::CompletePhaseGraph(ReduceType &phase2graph_reduce) {
  std::vector<af::AscNodePtr> node_order;
  GE_ASSERT_GRAPH_SUCCESS(PartitionByReduce(phase_graph, phase2graph_reduce, node_order));
  GE_ASSERT_GRAPH_SUCCESS(SetNodeOrder(node_order));
  std::vector<::ascir::ImplGraph> sub_optimize_graphs;
  GE_ASSERT_GRAPH_SUCCESS(
      ScheduleGroupGraphPartitioner::PartitionByConnectivity(phase_graph, sub_optimize_graphs, node_order));
  GE_ASSERT_EQ(sub_optimize_graphs.size(), 2UL);
  phase1graph.CopyFrom(sub_optimize_graphs[0]);
  phase2graph.CopyFrom(sub_optimize_graphs[1]);
  return ge::GRAPH_SUCCESS;
}

// 辅助函数：设置ArgMaxMultiRPhase1的输出属性
// ArgMaxMultiRPhase1有2个输出：output[0]=value(T类型), output[1]=index(DT_INT64)
// value的shape从原始ArgMax输出复制，dtype从输入复制
// index的shape从原始ArgMax输出复制，dtype固定为DT_INT64
static Status SetupArgMaxPhase1OutputAttrs(const af::AscNodePtr &phase1_node, const af::AscTensorAttr &input_attr,
                                           const af::AscTensorAttr &output_attr) {
  // 设置输出0（value）：shape从argmax_node输出复制，dtype从输入复制
  {
    auto op_desc = phase1_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    // 复制shape等属性，但dtype使用输入的类型
    *tensor_attr_group = output_attr;
    tensor_attr_group->dtype = input_attr.dtype;
  }

  // 设置输出1（index）：shape和格式等从输出0复制，dtype固定为DT_INT64
  {
    auto op_desc = phase1_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(1));
    auto tensor_attr_group = op_desc->MutableOutputDesc(1)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    // 使用之前保存的output_attr，而不是访问phase1_node->outputs[0]
    *tensor_attr_group = output_attr;
    tensor_attr_group->dtype = ge::DT_INT64;
  }
  return ge::GRAPH_SUCCESS;
}

// 辅助函数：复制节点级别的属性（sched、ir_attr等）
static void CopyNodeLevelAttrs(const af::AscNodePtr &dst_node, const af::AscNodePtr &src_node) {
  auto dst_op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = dst_op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_op_desc = src_node->GetOpDesc();
  auto src_asc_node_attr = src_op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
}

// R轴分核时，在阶段1将ArgMax替换为ArgMaxMultiRPhase1
Status ReplaceArgMaxInPhase1(ascir::ImplGraph &phase_graph, const af::AscNodePtr &argmax_node,
                             af::AscNodePtr &store_node, af::AscNodePtr &workspace_pre_node,
                             af::AscNodePtr &workspace_pre_index_node_out) {
  // 创建ArgMaxMultiRPhase1节点替换原始ArgMax
  af::ascir_op::ArgMaxMultiRPhase1 argmax_phase1((argmax_node->GetName() + "_phase1").c_str());
  auto new_argmax_phase1_node = phase_graph.AddNode(argmax_phase1);

  // 复制节点级别的属性（sched、ir_attr等）
  CopyNodeLevelAttrs(new_argmax_phase1_node, argmax_node);

  // 保存输入和输出属性，因为后续重定向边会导致inputs被清空
  const auto &saved_input_attr = argmax_node->inputs[0].attr;
  const auto &saved_output_attr = argmax_node->outputs[0].attr;

  // 为ArgMaxMultiRPhase1的两个输出设置正确的tensor属性
  (void)SetupArgMaxPhase1OutputAttrs(new_argmax_phase1_node, saved_input_attr, saved_output_attr);

  // 将ArgMax节点的所有输入边重定向到ArgMaxMultiRPhase1节点
  for (const auto &in_anchor : argmax_node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_anchor = in_anchor->GetPeerOutAnchor();
    if (src_anchor == nullptr) {
      continue;
    }
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_anchor, in_anchor));
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(src_anchor, new_argmax_phase1_node->GetInAnchor(in_anchor->GetIdx())));
  }

  // 为ArgMaxMultiRPhase1的第二个输出（index）创建额外的store节点
  // 注意：workspace_pre_index_node由调用方创建并传入，通过名称"_workspace_index"与Phase2的workspace_post_index关联
  af::ascir_op::Store store_index((argmax_node->GetName() + "_Store_index").c_str());
  auto new_store_index_node = phase_graph.AddNode(store_index);

  // 复制tensor属性
  // 注意：argmax_phase1_node有两个输出，需要正确设置属性

  // store_node接收argmax_phase1_node的输出0（value），类型应该是T（和输入相同）
  // 不能从argmax_node复制（argmax_node的输出是index，类型DT_INT64）
  // 注意：此时输入边已被重定向，argmax_node->inputs()可能为空，需要使用之前保存的属性
  {
    // 复制节点级别属性
    CopyNodeLevelAttrs(store_node, argmax_node);
    // 复制输出tensor属性：shape从output_attr复制，dtype从input_attr复制
    auto op_desc = store_node->GetOpDesc();
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = saved_output_attr;
    tensor_attr_group->dtype = saved_input_attr.dtype;
  }

  // store_index_node接收argmax_phase1_node的输出1（index），类型是DT_INT64
  // 使用之前保存的属性设置，因为此时输入边已被重定向
  {
    // 复制节点级别属性
    CopyNodeLevelAttrs(new_store_index_node, argmax_node);
    // 复制输出tensor属性：使用保存的output_attr
    auto op_desc = new_store_index_node->GetOpDesc();
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = saved_output_attr;  // index的shape和argmax输出相同
  }

  // workspace_pre_node连接到store_node，从store_node复制属性
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(new_store_index_node, workspace_pre_index_node_out));

  // 连接ArgMaxMultiRPhase1的第一个输出（value）到store节点
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(new_argmax_phase1_node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));

  // 连接ArgMaxMultiRPhase1的第二个输出（index）到store_index和workspace
  GE_CHK_STATUS_RET(
      af::GraphUtils::AddEdge(new_argmax_phase1_node->GetOutAnchor(1UL), new_store_index_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(
      af::GraphUtils::AddEdge(new_store_index_node->GetOutAnchor(0UL), workspace_pre_index_node_out->GetInAnchor(0UL)));

  // 删除原始的ArgMax节点，避免"not visited"错误
  // 输入边已被重定向到argmax_phase1_node
  // 输出边将在PartitionByReduce中重定向到new_reduce_node
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(phase_graph);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(compute_graph, argmax_node));

  return ge::GRAPH_SUCCESS;
}

// 已保存的节点属性（封装属性指针和属性值）
struct ReduceSavedNodeAttrInfo {
  const af::AscNodeAttr *attr_ptr;  // 属性指针
  af::AscNodeAttr attr_value;       // 属性值
};

// ArgMax的index路径节点
struct ArgMaxIndexNodes {
  af::AscNodePtr index_workspace_post_node;  // index workspace节点
  af::AscNodePtr index_load_node;            // index load节点
};

// Phase1节点集合结构体
struct ReducePhase1Nodes {
  af::AscNodePtr store_node;          // store节点
  af::AscNodePtr workspace_pre_node;  // workspace pre节点
};

// Phase2节点集合结构体
struct ReducePhase2Nodes {
  af::AscNodePtr workspace_post_node;  // workspace post节点
  af::AscNodePtr load_node;            // load节点
  af::AscNodePtr new_reduce_node;      // Phase2的reduce节点
};

// 图上下文结构体
struct ArgMaxPartitionGraphContext {
  ascir::ImplGraph &impl_graph;             // 图
  std::vector<af::AscNodePtr> &node_order;  // 节点顺序
  std::string graph_name;                   // 图名称（使用值而非引用，避免悬空引用）
};

// 设置ArgMax Phase2的所有属性
static Status SetupArgMaxPhase2Attrs(const ReduceSavedNodeAttrInfo &saved_attr, const ReducePhase2Nodes &phase2_nodes,
                                     const af::AscTensorAttr &reduce_input_attr,
                                     const af::AscTensorAttr &reduce_output_attr, const ArgMaxIndexNodes &index_nodes) {
  // 复制节点级别的属性到new_reduce_node
  if (saved_attr.attr_ptr != nullptr) {
    auto op_desc = phase2_nodes.new_reduce_node->GetOpDesc();
    auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
    if (dst_asc_node_attr != nullptr) {
      dst_asc_node_attr->sched = saved_attr.attr_value.sched;
      if (saved_attr.attr_value.ir_attr) {
        dst_asc_node_attr->ir_attr = saved_attr.attr_value.ir_attr->Clone();
      }
    }
  }

  // 设置输出（index）：shape和reduce输出相同，dtype为DT_INT64
  {
    auto op_desc = phase2_nodes.new_reduce_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = reduce_output_attr;
    tensor_attr_group->dtype = ge::DT_INT64;
  }

  // ArgMax特殊处理：先设置workspace和load的输出dtype
  {
    // 设置workspace_post和load的输出dtype为value类型
    auto op_desc = phase2_nodes.workspace_post_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    tensor_attr_group->dtype = reduce_input_attr.dtype;

    op_desc = phase2_nodes.load_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    tensor_attr_group->dtype = reduce_input_attr.dtype;
  }
  {
    // 设置workspace_post_index和load_index的输出dtype为DT_INT64
    auto op_desc = index_nodes.index_workspace_post_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    tensor_attr_group->dtype = ge::DT_INT64;

    op_desc = index_nodes.index_load_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
    tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    tensor_attr_group->dtype = ge::DT_INT64;
  }

  return ge::GRAPH_SUCCESS;
}

// 连接ArgMax Phase2的所有边
// 参数：1.输出锚点 2.输出边 3.Phase2节点集合 4.index workspace节点 5.index load节点
static Status ConnectArgMaxPhase2Edges(af::OutDataAnchorPtr argmax_out_anchor,
                                       const std::vector<af::InDataAnchorPtr> &argmax_out_edges,
                                       const ReducePhase2Nodes &phase2_nodes,
                                       const af::AscNodePtr &index_workspace_post_node,
                                       const af::AscNodePtr &index_load_node) {
  // 重定向原始ArgMax的输出边到new_reduce_node（ArgMaxMultiRPhase2）
  for (const auto &peer_in_anchor : argmax_out_edges) {
    // 移除原始边（虽然reduce_node已被删除，但移除边的操作应该还能执行）
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(argmax_out_anchor, peer_in_anchor));
    // 添加新边：从new_reduce_node的输出0到下游节点
    // ArgMaxMultiRPhase2只有1个输出（最终的index），所以输出索引是0
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(phase2_nodes.new_reduce_node->GetOutAnchor(0UL), peer_in_anchor));
  }

  // 连接Phase2的所有边
  // Phase2 value路径：workspace_post -> load -> new_reduce[输入0]
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(phase2_nodes.workspace_post_node->GetOutAnchor(0UL),
                                            phase2_nodes.load_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(phase2_nodes.load_node->GetOutAnchor(0UL),
                                            phase2_nodes.new_reduce_node->GetInAnchor(0UL)));

  // Phase2 index路径：workspace_post_index -> index_load -> new_reduce[输入1]
  GE_CHK_STATUS_RET(
      af::GraphUtils::AddEdge(index_workspace_post_node->GetOutAnchor(0UL), index_load_node->GetInAnchor(0UL)));
  GE_CHK_STATUS_RET(
      af::GraphUtils::AddEdge(index_load_node->GetOutAnchor(0UL), phase2_nodes.new_reduce_node->GetInAnchor(1UL)));

  return ge::GRAPH_SUCCESS;
}

// 处理ArgMax在R轴分核时的完整逻辑
// 包括：创建index路径节点、替换为Phase1算子、设置属性、连接边等
static Status HandleArgMaxPartition(const af::AscNodePtr &reduce_node, ReducePhase1Nodes &phase1_nodes,
                                    ReducePhase2Nodes &phase2_nodes, const ArgMaxPartitionGraphContext &ctx) {
  // ArgMax特殊处理：需要额外的workspace用于index
  af::ascir_op::Workspace index_workspace_pre((ctx.graph_name + "_workspace_index").c_str());
  af::ascir_op::Workspace index_workspace_post((ctx.graph_name + "_workspace_index").c_str());

  // 为index路径创建额外的workspace和load节点
  af::AscNodePtr index_workspace_pre_node = ctx.impl_graph.AddNode(index_workspace_pre);
  af::AscNodePtr index_workspace_post_node = ctx.impl_graph.AddNode(index_workspace_post);
  ctx.node_order.emplace_back(index_workspace_pre_node);

  af::ascir_op::Load index_load((ctx.graph_name + "_load_index").c_str());
  af::AscNodePtr index_load_node = ctx.impl_graph.AddNode(index_load);
  // 参考load_node的处理方式，不复制属性（避免从Phase1节点复制错误的sched.axis）

  const auto &saved_reduce_input_attr = reduce_node->inputs[0].attr;
  const auto &saved_reduce_output_attr = reduce_node->outputs[0].attr;

  // 保存ArgMax的输出边，用于后续重定向到new_reduce_node
  af::OutDataAnchorPtr argmax_out_anchor = reduce_node->GetOutDataAnchor(0UL);
  std::vector<af::InDataAnchorPtr> argmax_out_edges;
  if (argmax_out_anchor != nullptr) {
    for (const auto &peer_in_anchor : argmax_out_anchor->GetPeerInDataAnchors()) {
      GE_ASSERT_NOTNULL(peer_in_anchor);
      argmax_out_edges.push_back(peer_in_anchor);
    }
  }

  // 复制节点级别的属性（sched、ir_attr等）
  // 在调用ReplaceArgMaxInPhase1之前保存，因为调用后reduce_node将被删除
  auto src_asc_node_attr = reduce_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  af::AscNodeAttr saved_node_attr;
  if (src_asc_node_attr != nullptr) {
    saved_node_attr = *src_asc_node_attr;
  }

  // 在阶段1替换为ArgMaxMultiRPhase1，并处理额外的index输出
  GE_CHK_STATUS_RET(ReplaceArgMaxInPhase1(ctx.impl_graph, reduce_node, phase1_nodes.store_node,
                                          phase1_nodes.workspace_pre_node, index_workspace_pre_node));

  // 设置ArgMax Phase2的所有属性
  ReduceSavedNodeAttrInfo saved_attr_info = {src_asc_node_attr, saved_node_attr};
  ArgMaxIndexNodes index_nodes = {index_workspace_post_node, index_load_node};
  GE_CHK_STATUS_RET(SetupArgMaxPhase2Attrs(saved_attr_info, phase2_nodes, saved_reduce_input_attr,
                                           saved_reduce_output_attr, index_nodes));

  // 复制Phase2的workspace属性，与Phase1的workspace通过相同名称关联到同一buffer
  // 注意：在设置dtype之后调用，避免覆盖正确的dtype
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(phase2_nodes.load_node, phase2_nodes.workspace_post_node));
  GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(index_load_node, index_workspace_post_node));

  // 连接ArgMax Phase2的所有边
  GE_CHK_STATUS_RET(ConnectArgMaxPhase2Edges(argmax_out_anchor, argmax_out_edges, phase2_nodes,
                                             index_workspace_post_node, index_load_node));

  return ge::GRAPH_SUCCESS;
}

// R轴分核时，为ArgMax创建额外的workspace和load节点用于index
Status RMulticorePhase2Graph::PartitionByReduce(ascir::ImplGraph &impl_graph, ReduceType &phase2graph_reduce,
                                                std::vector<af::AscNodePtr> &node_order) {
  af::ascir_op::Workspace workspace_pre((phase2graph.GetName() + "_workspace").c_str());
  af::ascir_op::Workspace workspace_post((phase2graph.GetName() + "_workspace").c_str());
  af::ascir_op::Load load((phase2graph.GetName() + "_load").c_str());
  af::ascir_op::Store store((phase1graph.GetName() + "Store").c_str());

  af::AscNodePtr new_reduce_node;
  std::visit([&new_reduce_node, &impl_graph](auto &&reduce_op) { new_reduce_node = impl_graph.AddNode(reduce_op); },
             phase2graph_reduce);

  auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
  node_order.emplace_back(workspace_pre_node);
  auto workspace_post_node = impl_graph.AddNode(workspace_post);
  auto load_node = impl_graph.AddNode(load);
  GE_ASSERT_NOTNULL(load_node);
  auto store_node = impl_graph.AddNode(store);
  GE_ASSERT_NOTNULL(store_node);

  if (reduce_node->GetType() == "ArgMax") {
    // ArgMax特殊处理：调用专用函数处理双路径（value和index）逻辑
    ReducePhase1Nodes phase1_nodes = {store_node, workspace_pre_node};
    ReducePhase2Nodes phase2_nodes = {workspace_post_node, load_node, new_reduce_node};
    ArgMaxPartitionGraphContext ctx = {impl_graph, node_order, phase2graph.GetName()};
    GE_CHK_STATUS_RET(HandleArgMaxPartition(reduce_node, phase1_nodes, phase2_nodes, ctx));
  } else {
    // 普通reduce算子的处理逻辑
    GE_ASSERT_GRAPH_SUCCESS(DoCopyAscNodeTensorAttr(reduce_node, new_reduce_node));
    GE_ASSERT_GRAPH_SUCCESS(DoCopyAscNodeTensorAttr(reduce_node, store_node));
    GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(reduce_node, workspace_pre_node));
    GE_ASSERT_GRAPH_SUCCESS(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
    for (const auto &reduce_out_anchor : reduce_node->GetAllOutDataAnchors()) {
      GE_ASSERT_NOTNULL(reduce_out_anchor);
      for (const auto &peer_in_anchor : reduce_out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_anchor);
        auto reduce_out_node = peer_in_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(reduce_out_node);
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(reduce_node->GetOutAnchor(reduce_out_anchor->GetIdx()),
                                                     reduce_out_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(new_reduce_node->GetOutAnchor(reduce_out_anchor->GetIdx()),
                                                  reduce_out_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
    // add reduce->store->workspace_pre_node
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(reduce_node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
    // add workspace_post_node->load->new reduce
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL), new_reduce_node->GetInAnchor(0UL)));
  }

  return ge::GRAPH_SUCCESS;
}

Status RMulticorePhase2Graph::SetNodeOrder(std::vector<af::AscNodePtr> &node_order) {
  auto new_reduce_node =
      phase_graph.FindNode((phase2graph.GetName() + "_" + reduce_node->GetName() + "_reduce").c_str());
  GE_ASSERT_NOTNULL(new_reduce_node);
  std::set<af::NodePtr> visited{new_reduce_node};
  std::list<af::NodePtr> next_nodes{new_reduce_node};
  while (!next_nodes.empty()) {
    const auto node = next_nodes.front();
    GE_ASSERT_NOTNULL(node);
    next_nodes.pop_front();
    if (node->GetOutDataNodes().empty()) {
      node_order.emplace_back(std::dynamic_pointer_cast<af::AscNode>(node));
    }
    for (auto &out_node : node->GetOutDataNodes()) {
      if (visited.find(out_node) == visited.cend()) {
        next_nodes.emplace_back(out_node);
        visited.emplace(out_node);
      }
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
