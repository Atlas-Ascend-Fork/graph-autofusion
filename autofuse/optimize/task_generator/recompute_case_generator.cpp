/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "recompute_case_generator.h"
#include <algorithm>
#include <set>
#include <unordered_set>
#include <queue>
#include "schedule_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_utils.h"
#include "graph/utils/graph_utils.h"
#include "common_utils.h"
#include "task_generator/node_partition_utils.h"

namespace {
constexpr size_t kContinuousOneAxisNum = 2UL;
constexpr size_t kRecomputeNumThreshold = 5UL;
constexpr size_t kRecomputeGapThreshold = 5UL;
const af::Symbol kContinuousOneAxisSizeThreshold = af::Symbol(256);
constexpr int64_t kNonFirstAxisProductThreshold = 32768L;  // 32K, 按元素数计
constexpr int32_t kBroadcastWorkspaceSelectedScore = 1;    // 超阈值: 本模板优先
constexpr int32_t kBroadcastWorkspaceFallbackScore = -1;   // 未超阈值: 劣后给默认模板

bool IsSupportedComputeType(af::ComputeType compute_type) {
  static std::unordered_set<af::ComputeType> supported_types{
      af::ComputeType::kComputeElewise, af::ComputeType::kComputeBroadcast, af::ComputeType::kComputeLoad,
      af::ComputeType::kComputeStore};
  return supported_types.count(compute_type) > 0UL;
}

// Broadcast workspace 切分候选的局部结构过滤(全图连通性选点在 GenerateBroadcastWorkspaceTask 内完成):
// 1. Broadcast 单输入;
// 2. 上游必须是真实计算节点(与 reduce 侧 HasOperatorBeforeBroadcast 同语义): 排除直连无输入叶子
//    (Data/ScalarLike, 副本路径不产生 workspace 边界)与 Load(GM 搬运节点, 副本路径不建 ws 链,
//    且有输入类型断言与悬置退化子图两类崩溃风险);
// 3. 存在首轴 B 轴扩展(IsFirstAxisBroadcast 结构判定, 不限首轴大小与其余轴, 动态轴可识别)。
// 消费端不限类型。
bool IsBroadcastWorkspaceCandidate(const af::AscNodePtr &broadcast_node) {
  if (!af::ops::IsOps<af::ascir_op::Broadcast>(broadcast_node) || broadcast_node->GetInDataNodesSize() != 1UL) {
    return false;
  }
  auto pre_node = std::dynamic_pointer_cast<af::AscNode>(broadcast_node->GetInDataNodes().at(0UL));
  GE_ASSERT_NOTNULL(pre_node);
  if (pre_node->GetInDataNodesSize() == 0UL || optimize::ScheduleUtils::IsLoad(pre_node)) {
    return false;
  }
  return optimize::IsFirstAxisBroadcast(pre_node, broadcast_node);
}

// 断边连通判定: 断开 start->target 这条唯一数据边后, start 与 target 是否仍无向连通。
// 前提: start->target 至多一条数据边且无控制边(由 IsBroadcastWorkspaceCandidate 的单输入约束保证),
// 因此只需在 start 的出向扩展中跳过该邻接, 无需实际删边再恢复, 不会改写图结构。
// 从 start 沿 GetInNodes/GetOutNodes 双向 BFS(与 CollectConnectedNodes 同口径, 含控制边),
// 不做 workspace 同名跳转(同名对之间物理无边); 到达 target 提前终止。
// 等价于切分后同名 ws_pre/ws_post 是否落入同一子图。
bool IsConnectedAfterRemovingDirectEdge(const af::AscNodePtr &start, const af::AscNodePtr &target) {
  if (start == target) {
    return true;
  }
  std::set<af::NodePtr> visited;
  visited.emplace(start);
  std::queue<af::NodePtr> work_queue;
  work_queue.emplace(start);
  while (!work_queue.empty()) {
    const auto node = work_queue.front();
    work_queue.pop();
    for (const auto &in_node : node->GetInNodes()) {
      if (in_node.get() == target.get()) {
        return true;
      }
      if (visited.insert(in_node).second) {
        work_queue.push(in_node);
      }
    }
    for (const auto &out_node : node->GetOutNodes()) {
      // 仅在 start 处跳过 start->target: 其他节点指向 target 的边是真实残余路径, 不能跳
      if (node.get() == start.get() && out_node.get() == target.get()) {
        continue;
      }
      if (out_node.get() == target.get()) {
        return true;
      }
      if (visited.insert(out_node).second) {
        work_queue.push(out_node);
      }
    }
  }
  return false;
}

af::Status CopyRecomputeNode(af::Node *ori_node, af::AscGraph &graph, af::AscNodePtr &new_node) {
  const auto &op_desc = af::GraphUtils::CopyOpDesc(ori_node->GetOpDesc(), nullptr);
  GE_CHECK_NOTNULL(op_desc);
  op_desc->SetName(ori_node->GetName() + "_recompute");
  af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  new_node = graph.AddNode(op);
  GE_ASSERT_NOTNULL(new_node);
  auto src_asc_node = std::dynamic_pointer_cast<af::AscNode>(ori_node->shared_from_this());
  GE_ASSERT_NOTNULL(src_asc_node);
  GE_ASSERT_TRUE(af::AscGraph::CopyAscNodeTensorAttr(src_asc_node, new_node),
                 "DoCopyAscNodeTensorAttr failed, node = %s[%s]", ori_node->GetNamePtr(), ori_node->GetTypePtr());
  return af::SUCCESS;
}

size_t CountComputeNodes(const std::unordered_set<af::Node *> &nodes) {
  size_t count = 0UL;
  for (const auto &node : nodes) {
    auto node_attr = node->GetOpDescBarePtr()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
    if (node_attr != nullptr && node_attr->api.type == af::ApiType::kAPITypeCompute) {
      ++count;
    }
  }
  return count;
}
}  // namespace

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;
Status RecomputeCaseGenerator::GeneratorTask(ascir::HintGraph &hint_graph, std::vector<ScheduleTask> &tasks,
                                             const OptimizerOptions &options) {
  (void)options;
  auto graph_axis = hint_graph.GetAllAxis();
  is_static_graph_ = std::all_of(graph_axis.begin(), graph_axis.end(),
                                 [](const af::AxisPtr &axis) { return axis->size.IsConstExpr(); });

  std::vector<af::AscNodePtr> potential_store;
  bool has_unsupported_type{false};
  for (const auto &node : hint_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    if (!IsSupportedComputeType(node->attr.api.compute_type)) {
      has_unsupported_type = true;
      break;
    }
    if (ScheduleUtils::IsStore(node) && IsRecomputableNode(hint_graph, node.get())) {
      GELOGD("Store node [%s] may be recomputable.", node->GetNamePtr());
      potential_store.push_back(node);
    }
  }

  // Broadcast workspace 候选须基于未做重计算改写的 hint_graph 拷贝生成, 所有分支均需要该候选,
  // 在首个分支返回前统一生成一次; 后续 DoTaskGenerator 会原地改写 hint_graph
  GE_ASSERT_SUCCESS(GenerateBroadcastWorkspaceTask(hint_graph, tasks));

  if (has_unsupported_type || potential_store.empty()) {
    return GroupPartitionAndGenTasks(hint_graph, tasks);
  }

  for (const auto &store_node : potential_store) {
    GE_ASSERT_SUCCESS(AnalyzeSplittablePath(hint_graph, store_node));
  }
  MergeAllPaths();

  if (result_.merged_path_nodes.empty()) {
    return GroupPartitionAndGenTasks(hint_graph, tasks);
  }

  GE_ASSERT_SUCCESS(DoTaskGenerator(hint_graph, tasks));
  return af::SUCCESS;
}

Status RecomputeCaseGenerator::GenerateBroadcastWorkspaceTask(ascir::HintGraph &hint_graph,
                                                              std::vector<ScheduleTask> &tasks) {
  // 类比 ReducePartitionCaseGenerator::GenerateBroadcastWorkspaceCase: 对识别出的首轴 Broadcast 建立
  // workspace 切分(pre -> store -> ws_pre ⫽ ws_post -> load -> broadcast), 切分前的小张量经 workspace
  // 传递, B 轴扩展推迟到消费侧 kernel 内完成, 作为独立调度候选。
  ascir::ImplGraph optimize_graph(hint_graph.GetName().c_str());
  std::vector<af::AscNodePtr> node_order;
  std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> broadcast_candidates;  // (broadcast, pre)
  GE_CHK_STATUS_RET(CollectBroadcastCandidates(hint_graph, optimize_graph, node_order, broadcast_candidates));
  if (broadcast_candidates.empty()) {
    return af::SUCCESS;
  }
  bool partitioned = false;
  GE_CHK_STATUS_RET(SelectBroadcastPartitions(optimize_graph, broadcast_candidates, node_order, partitioned));
  if (!partitioned) {
    return af::SUCCESS;
  }
  ascir::utils::DumpGraph(optimize_graph, "AfterBroadcastPartition");
  std::sort(node_order.begin(), node_order.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
    return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
  });
  ScheduleTask task{optimize_graph, {}, {}};
  GE_CHK_STATUS_RET(
      ScheduleGroupGraphPartitioner::PartitionByConnectivity(optimize_graph, task.grouped_graphs, node_order),
      "Failed to partition graph [%s].", optimize_graph.GetName().c_str());
  // 同名 Workspace 落入同一子图或子图依赖成环时 PartitionByConnectivity 会清空全部子图并返回 SUCCESS,
  // 空分组下动态 shape 打分断言硬失败, 须跳过本模板降级(与 reduce 侧 GeneratorGeneralTask 一致)
  if (task.grouped_graphs.empty()) {
    GELOGW("Graph[%s] has no subgraphs after broadcast workspace partition, skip task.",
           optimize_graph.GetName().c_str());
    return af::SUCCESS;
  }
  // 打分函数参与 tiling 期多个调度模板间的竞争, 默认模板/重计算模板未设置打分时默认0分:
  // 原始图非首轴乘积 > 32K 打正分使本模板优先, 否则打负分劣后给默认模板(平分时 result 序号小者
  // 胜, 本模板恒为 result0, 返回 0 分无法落选); 生成须在切分后进行, 以便按 size var 定位所在 group
  std::string score_func;
  GE_CHK_STATUS_RET(BroadcastWorkspaceScoreFunctionGenerator(hint_graph, task.grouped_graphs).Generate(score_func),
                    "Failed to generate broadcast workspace score func for graph[%s].",
                    optimize_graph.GetName().c_str());
  task.score_func = score_func;
  GELOGI("Graph[%s] generate broadcast workspace task with %zu subgraph(s).", optimize_graph.GetName().c_str(),
         task.grouped_graphs.size());
  tasks.emplace_back(std::move(task));
  return af::SUCCESS;
}

// 收集切分候选: 原图快速预检无候选时直接返回, 避免整图拷贝; node_order 收集无数据输出节点作为
// 连通性分组种子, broadcast_candidates 收集候选对 (broadcast, pre)。
Status RecomputeCaseGenerator::CollectBroadcastCandidates(
    const ascir::HintGraph &hint_graph, ascir::ImplGraph &optimize_graph, std::vector<af::AscNodePtr> &node_order,
    std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &broadcast_candidates) {
  bool has_broadcast_candidate = false;
  for (const auto &node : hint_graph.GetAllNodes()) {
    if (IsBroadcastWorkspaceCandidate(node)) {
      has_broadcast_candidate = true;
      break;
    }
  }
  if (!has_broadcast_candidate) {
    GELOGD("Graph[%s] has no first-axis broadcast workspace candidate, skip broadcast workspace task.",
           hint_graph.GetName().c_str());
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(optimize_graph.CopyFrom(hint_graph), "Failed to copy graph from [%s].", hint_graph.GetName().c_str());
  for (const auto &node : optimize_graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty()) {
      node_order.emplace_back(node);
    }
    if (IsBroadcastWorkspaceCandidate(node)) {
      auto pre_node = std::dynamic_pointer_cast<af::AscNode>(node->GetInDataNodes().at(0UL));
      GE_ASSERT_NOTNULL(pre_node);
      broadcast_candidates.emplace_back(node, pre_node);
    }
  }
  GELOGI("Graph[%s] has %zu first-axis broadcast workspace candidate(es), begin connectivity selection.",
         optimize_graph.GetName().c_str(), broadcast_candidates.size());
  return af::SUCCESS;
}

// 断边连通性选点: 候选边断开后 pre 与 brc 仍无向连通则拒绝该切分(残余连通会使同名 ws_pre/ws_post
// 落入同一子图, 命中 CheckDuplicateWorkspaceNodes 清空全部子图)。逐候选独立判定, 无有效切分时不产生候选。
Status RecomputeCaseGenerator::SelectBroadcastPartitions(
    ascir::ImplGraph &optimize_graph,
    const std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &broadcast_candidates,
    std::vector<af::AscNodePtr> &node_order, bool &partitioned) {
  size_t partition_count = 0UL;
  for (const auto &candidate : broadcast_candidates) {
    const bool connected = IsConnectedAfterRemovingDirectEdge(candidate.second, candidate.first);
    if (connected) {
      GELOGD("Graph[%s] broadcast node [%s] keeps residual connectivity, skip.", optimize_graph.GetName().c_str(),
             candidate.first->GetNamePtr());
      continue;
    }
    // PartitionByNode 形参为非 const 引用, 拷贝出局部变量再传参
    af::AscNodePtr broadcast_node = candidate.first;
    af::AscNodePtr pre_node = candidate.second;
    GELOGI("Broadcast workspace partition in graph[%s]: pre_node=%s -> broadcast=%s.", optimize_graph.GetName().c_str(),
           pre_node->GetNamePtr(), broadcast_node->GetNamePtr());
    GE_CHK_STATUS_RET(PartitionByNode(pre_node, broadcast_node, optimize_graph, partitioned, node_order),
                      "Failed to partition broadcast node [%s].", broadcast_node->GetNamePtr());
    ++partition_count;
  }
  GELOGI("Graph[%s] connectivity selection kept %zu broadcast partition(s) of %zu candidate(es).",
         optimize_graph.GetName().c_str(), partition_count, broadcast_candidates.size());
  return af::SUCCESS;
}

Status RecomputeCaseGenerator::AnalyzeSplittablePath(ascir::HintGraph &hint_graph,
                                                     const af::AscNodePtr &potential_store) {
  SplitResultInOnePath single_path_results;
  auto out_nodes = potential_store->GetOutDataNodes();
  GE_ASSERT_TRUE(out_nodes.size() == 1UL);
  auto output_node = out_nodes.at(0UL).get();
  single_path_results.valid_path.emplace(output_node);
  single_path_results.output_nodes.emplace(output_node);

  std::set<af::Node *> visited_nodes;
  visited_nodes.emplace(potential_store.get());
  std::queue<af::AscNode *> head_nodes;
  head_nodes.emplace(potential_store.get());
  while (!head_nodes.empty()) {
    const auto top_node = head_nodes.front();
    head_nodes.pop();
    GE_ASSERT_NOTNULL(top_node);
    bool current_valid = IsRecomputableNode(hint_graph, top_node);
    if (!current_valid) {
      GELOGD("Store node [%s] cannot be split for node:[%s].", top_node->GetNamePtr(), potential_store->GetNamePtr());
      return af::SUCCESS;
    }

    GELOGD("Add path node [%s] in path of split node:[%s].", potential_store->GetNamePtr(), top_node->GetNamePtr());
    single_path_results.valid_path.emplace(top_node);
    bool is_all_in_path = true;
    for (const auto &out_node : top_node->GetOutDataNodes()) {
      if (single_path_results.valid_path.count(out_node.get()) == 0UL ||
          single_path_results.recompute_nodes.count(out_node.get()) > 0UL) {
        is_all_in_path = false;
        break;
      }
    }

    if (!is_all_in_path) {
      GELOGD("Add recompute node [%s] in path of split node:[%s].", potential_store->GetNamePtr(),
             top_node->GetNamePtr());
      single_path_results.recompute_nodes.emplace(top_node);
    }
    // add inputs
    for (auto &in_node : top_node->GetInDataNodes()) {
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(in_node);
      GE_ASSERT_NOTNULL(asc_node);
      if (visited_nodes.insert(asc_node.get()).second) {
        head_nodes.push(asc_node.get());
      }
    }
  }
  result_.valid_path_results.emplace_back(std::move(single_path_results));

  return af::SUCCESS;
}

void RecomputeCaseGenerator::MergeAllPaths() {
  // 预留动态评估单个path被拆分的代价
  for (const auto &path : result_.valid_path_results) {
    result_.merged_output_nodes.insert(path.output_nodes.begin(), path.output_nodes.end());
    result_.merged_recompute_nodes.insert(path.recompute_nodes.begin(), path.recompute_nodes.end());
    result_.merged_path_nodes.insert(path.valid_path.begin(), path.valid_path.end());
  }
  auto it = result_.merged_recompute_nodes.begin();
  while (it != result_.merged_recompute_nodes.end()) {
    bool is_all_in_path = true;
    for (const auto &out_node : (*it)->GetOutDataNodes()) {
      if (result_.merged_path_nodes.count(out_node.get()) == 0 ||
          result_.merged_recompute_nodes.count(out_node.get()) > 0UL) {
        is_all_in_path = false;
        break;
      }
    }

    if (is_all_in_path) {
      GELOGD("Node [%s]'s outputs are all in path, no need to do recompute.", (*it)->GetNamePtr());
      it = result_.merged_recompute_nodes.erase(it);
    } else {
      ++it;
    }
  }
  GELOGD("After merging path, [%zu] path nodes, [%zu] recompute_nodes [%zu] output nodes remaining.",
         result_.merged_path_nodes.size(), result_.merged_recompute_nodes.size(), result_.merged_output_nodes.size());
}

Status RecomputeCaseGenerator::DoTaskGenerator(ascir::ImplGraph &impl_graph, std::vector<ScheduleTask> &tasks) const {
  if (!IsRecomputableAlwaysBetter()) {
    ascir::ImplGraph origin_graph(impl_graph.GetName().c_str());
    GE_ASSERT_TRUE(origin_graph.CopyFrom(impl_graph), "Failed to copy graph from [%s].", impl_graph.GetName().c_str());
    GE_ASSERT_SUCCESS(GroupPartitionAndGenTasks(origin_graph, tasks));
  }

  // Gen recompute task
  auto cg = af::AscGraphUtils::GetComputeGraph(impl_graph);
  GE_ASSERT_NOTNULL(cg);
  cg->SetName(cg->GetName() + "_recompute_case");
  GE_ASSERT_SUCCESS(DoGraphSplit(impl_graph), "Failed to split recompute graph from [%s].",
                    impl_graph.GetName().c_str());
  GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticNameForGraph(impl_graph, semantic::kPointwiseRecompute));
  GE_ASSERT_SUCCESS(GroupPartitionAndGenTasks(impl_graph, tasks));

  return af::SUCCESS;
}

Status RecomputeCaseGenerator::DoGraphSplit(af::AscGraph &graph) const {
  // copy recompute_nodes
  std::unordered_map<af::Node *, af::Node *> recompute_to_new;
  for (auto recompute_node : result_.merged_recompute_nodes) {
    af::AscNodePtr new_node;
    GE_ASSERT_SUCCESS(CopyRecomputeNode(recompute_node, graph, new_node), "Failed to copy recompute node [%s].",
                      recompute_node->GetNamePtr());
    recompute_to_new[recompute_node] = new_node.get();
  }

  std::queue<af::Node *> bfs_queue;
  std::unordered_set<af::Node *> visited;
  for (auto &out_node : result_.merged_output_nodes) {
    bfs_queue.emplace(out_node);
    visited.emplace(out_node);
  }

  while (!bfs_queue.empty()) {
    af::Node *current = bfs_queue.front();
    bfs_queue.pop();
    auto cur_iter = recompute_to_new.find(current);
    bool is_cur_recompute = cur_iter != recompute_to_new.end();
    for (const auto &in_anchor : current->GetAllInDataAnchors()) {
      GE_ASSERT_NOTNULL(in_anchor);
      auto peer_out = in_anchor->GetPeerOutAnchor();
      if (peer_out == nullptr) {
        continue;
      }
      auto input_node = peer_out->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(input_node);
      auto in_iter = recompute_to_new.find(input_node);
      bool is_in_recompute = in_iter != recompute_to_new.end();
      if (is_cur_recompute) {
        GE_ASSERT_TRUE(is_in_recompute);
        GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(in_iter->second->GetOutDataAnchor(peer_out->GetIdx()),
                                                  cur_iter->second->GetInDataAnchor(in_anchor->GetIdx())));
      } else {
        if (is_in_recompute) {
          GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(peer_out, in_anchor));
          GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(in_iter->second->GetOutDataAnchor(peer_out->GetIdx()), in_anchor));
        }
      }

      if (visited.count(input_node) == 0UL) {
        visited.insert(input_node);
        bfs_queue.push(input_node);
      }
    }
  }

  return af::SUCCESS;
}

bool RecomputeCaseGenerator::IsRecomputableNode(ascir::HintGraph &hint_graph, af::AscNode *node) const {
  if (node->attr.api.type == af::ApiType::kAPITypeBuffer) {
    return true;
  }
  auto check_static_tensor = [&hint_graph](af::AscTensorAttr &tensor_attr) -> bool {
    af::Expression axis_size_product = af::sym::kSymbolOne;
    const auto &repeats = tensor_attr.repeats;
    const auto &axes = tensor_attr.axis;

    for (size_t i = 0UL; i < repeats.size() && i < axes.size(); ++i) {
      if (af::SymbolicUtils::StaticCheckNe(repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue) {
        break;
      }
      auto axis = hint_graph.FindAxis(axes[i]);
      if (axis == nullptr || af::SymbolicUtils::StaticCheckEq(axis->size, af::sym::kSymbolOne) == af::TriBool::kTrue) {
        break;
      }
      auto last_threshold = kContinuousOneAxisSizeThreshold / axis->size;
      if (af::SymbolicUtils::StaticCheckGt(axis_size_product, last_threshold) == af::TriBool::kTrue) {
        axis_size_product = kContinuousOneAxisSizeThreshold + af::sym::kSymbolOne;
        break;
      }
      axis_size_product = axis_size_product * axis->size;
    }
    return af::SymbolicUtils::StaticCheckGt(axis_size_product, kContinuousOneAxisSizeThreshold) == af::TriBool::kTrue;
  };

  auto check_dynamic_tensor = [](af::AscTensorAttr &tensor_attr) -> bool {
    size_t continuous_ones_count = 0UL;
    for (const auto &repeat : tensor_attr.repeats) {
      if (af::SymbolicUtils::StaticCheckEq(repeat, af::sym::kSymbolOne) == af::TriBool::kTrue) {
        ++continuous_ones_count;
      } else {
        break;
      }
    }
    return continuous_ones_count > kContinuousOneAxisNum;
  };

  auto output_tensors = node->outputs();
  if (is_static_graph_) {
    return std::all_of(output_tensors.begin(), output_tensors.end(),
                       [&check_static_tensor](af::AscTensor *&tensor) { return check_static_tensor(tensor->attr); });
  }
  return std::all_of(output_tensors.begin(), output_tensors.end(),
                     [&check_dynamic_tensor](af::AscTensor *&tensor) { return check_dynamic_tensor(tensor->attr); });
}

bool RecomputeCaseGenerator::IsRecomputableAlwaysBetter() const {
  if (!is_static_graph_) {
    return false;
  }

  const size_t recompute_valid_size = CountComputeNodes(result_.merged_recompute_nodes);
  if (recompute_valid_size >= kRecomputeNumThreshold) {
    return false;
  }

  const size_t path_valid_size = CountComputeNodes(result_.merged_path_nodes);
  const bool is_second_condition_met = (recompute_valid_size * kRecomputeGapThreshold) < path_valid_size;
  return is_second_condition_met;
}

BroadcastWorkspaceScoreFunctionGenerator::BroadcastWorkspaceScoreFunctionGenerator(
    const ascir::HintGraph &graph, const std::vector<ascir::ImplGraph> &grouped_graphs)
    : graph_(&graph), grouped_graphs_(&grouped_graphs) {}

Status BroadcastWorkspaceScoreFunctionGenerator::Generate(std::string &score_func) {
  GE_CHK_STATUS_RET(ParseNonFirstAxisProduct());
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  if (product_.IsConstExpr()) {
    // 静态分支: 全部非首轴 size 为常量, 编译期直接求乘积生成常量分
    int64_t product_value = 0L;
    GE_ASSERT_TRUE(product_.GetConstValue(product_value), "Failed to get int value, expr = %s", product_.Str().get());
    GenerateReturnValue(product_value > kNonFirstAxisProductThreshold ? kBroadcastWorkspaceSelectedScore
                                                                      : kBroadcastWorkspaceFallbackScore);
    score_func = ss_.str();
    return af::SUCCESS;
  }
  // 动态分支: CalcScore 在 DoTiling 前调用, 求解变量不可用, 只能引用输入变量(非首轴 size var 均为
  // 输入变量), 生成运行时代码从 tiling data 嵌套结构读实际值
  GE_CHK_STATUS_RET(GenerateForDynamic());
  score_func = ss_.str();
  return af::SUCCESS;
}

Status BroadcastWorkspaceScoreFunctionGenerator::ParseNonFirstAxisProduct() {
  // 原始图 shape 的非首轴乘积: GetAllAxis 按创建序返回(Loops 顺序, axes[0] 即 B 轴);
  // 一维/空轴图空乘积记为 1, 恒不超过阈值
  const auto axes = graph_->GetAllAxis();
  if (axes.size() <= 1UL) {
    product_ = af::Symbol(int64_t(1L));
    return af::SUCCESS;
  }
  product_ = axes[1UL]->size;
  for (size_t i = 2UL; i < axes.size(); ++i) {
    product_ = product_ * axes[i]->size;
  }
  GELOGD("Broadcast workspace score product expr = %s", product_.Str().get());
  return af::SUCCESS;
}

Status BroadcastWorkspaceScoreFunctionGenerator::GenerateForDynamic() {
  // 嵌套结构 InputParams 按 schedule group 定义(仅含该 group 子图用到的 size var), 逐自由符号
  // 定位实际所在 group 后生成跨 group 引用; CalcScore 入参为整个 AutofuseTilingData,
  // graph0_result0_g{N}_tiling_data 任意组合均可引用(本模板恒为 result0)
  std::vector<SizeVarSet> group_vars(grouped_graphs_->size());
  for (size_t g = 0UL; g < grouped_graphs_->size(); ++g) {
    AscGraphInfoComplete::AppendOriginalSizeVar((*grouped_graphs_)[g], group_vars[g]);
  }
  std::vector<std::pair<af::Expression, af::Expression>> replacements;
  for (const auto &sym : product_.FreeSymbols()) {
    size_t group_idx = grouped_graphs_->size();
    for (size_t g = 0UL; g < grouped_graphs_->size(); ++g) {
      if (group_vars[g].count(sym) > 0UL) {
        group_idx = g;
        break;
      }
    }
    // 切分覆盖全部节点且 tensor attr 原样拷贝, 原始轴符号必被某个 group 引用; 断言失败说明
    // 切分/拷贝不变量被破坏, 生成裸符号引用会造成 tiling 代码编译失败(字段不存在), 须立即暴露
    GE_ASSERT_TRUE(group_idx < grouped_graphs_->size(), "size var [%s] is not used by any schedule group",
                   sym.Str().get());
    const std::string field =
        std::string("tiling_data.graph0_result0_g") + std::to_string(group_idx) + "_tiling_data." + sym.Str().get();
    replacements.emplace_back(sym, af::Symbol(field.c_str()));
    GELOGD("Broadcast workspace score var [%s] located in schedule group[%zu].", sym.Str().get(), group_idx);
  }
  ss_ << "  auto non_first_axis_product = static_cast<int64_t>(" << product_.Replace(replacements).Str().get() << ");"
      << std::endl;
  ss_ << "  return non_first_axis_product > " << kNonFirstAxisProductThreshold << " ? "
      << kBroadcastWorkspaceSelectedScore << " : " << kBroadcastWorkspaceFallbackScore << ";" << std::endl;
  ss_ << "}" << std::endl;
  GELOGI("Broadcast workspace score code generated: %s", ss_.str().c_str());
  return af::SUCCESS;
}

void BroadcastWorkspaceScoreFunctionGenerator::GenerateReturnValue(const int32_t score) {
  ss_ << "  return " << score << ";" << std::endl;
  ss_ << "}" << std::endl;
}
}  // namespace optimize
