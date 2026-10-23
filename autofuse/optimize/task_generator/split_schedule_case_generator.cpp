/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_schedule_case_generator.h"

#include <queue>

#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"

#include "ascir/meta/ascir_utils.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "optimize/schedule_utils.h"
#include "optimize/task_generator/split_group_partitioner.h"
#include "platform/platform_factory.h"
#include "common_utils.h"

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;
namespace {
constexpr uint32_t kMaxOutputNum = 48U;
constexpr int32_t kAlignment = 32;
}  // namespace

Status SplitFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                          std::vector<std::string> &score_functions) {
  auto split_nodes = FindSplitNodes(graph);
  if (split_nodes.empty()) {
    return af::SUCCESS;
  }

  auto split_node = split_nodes.front();
  split_node_ = split_node;
  bool is_first_dim = false;
  size_t split_dim = 0;
  GE_CHK_STATUS_RET(ResolveSplitDim(split_node, split_dim, is_first_dim), "ResolveSplitDim failed");
  // case1. 首轴split，转load
  if (is_first_dim) {
    // gm split
    GE_CHK_STATUS_RET(ConvertSplitToLoads(graph, split_node, split_dim), "ConvertSplitToLoads failed");
    graphs.emplace_back(graph);
    GELOGI("split on first dim, num_inputs = %u, 1 template was generated", split_node->inputs.Size());
    return af::SUCCESS;
  }
  auto platform = PlatformFactory::GetInstance().GetPlatform();
  GE_ASSERT_NOTNULL(platform);

  // 改轴之前，备份图用于ub不全载case
  ascir::ImplGraph optimized_graph((graph.GetName() + "_splitv_group_split").c_str());
  optimized_graph.CopyFrom(graph);

  // case2. 生成ub内split的case
  if (split_node->outputs().size() <= kMaxOutputNum) {
    graphs.emplace_back(graph);
  }

  // case3. 在ub不全载的case的图中, 对split进行分组
  split_node = FindSplitNodes(optimized_graph).front();
  bool split = false;
  GE_CHK_STATUS_RET(SplitSplits(optimized_graph, split_node, split_dim, split), "SplitSplits failed");
  GELOGI("Split on non-first dim, splits are grouped and templates generated, split = %d", static_cast<int32_t>(split));

  GE_CHK_STATUS_RET(ConvertSplitToLoads(optimized_graph, split_node, split_dim), "ConvertSplitToLoads failed");
  graphs.emplace_back(optimized_graph);

  // 多模板，保持 score_functions 与模板数量对齐，但默认不生成 Split 打分函数。
  if ((graphs.size() > 1U)) {
    score_functions.resize(2U);  // ub_split + split_2_load
  }
  return af::SUCCESS;
}

std::vector<af::AscNodePtr> SplitFusionCaseGenerator::FindSplitNodes(const ascir::HintGraph &owner_graph) {
  std::vector<af::AscNodePtr> split_nodes;
  for (const auto &node : owner_graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Split>(node)) {
      split_nodes.emplace_back(node);
    }
  }
  return split_nodes;
}

Status SplitFusionCaseGenerator::ResolveSplitDim(const af::AscNodePtr &split_node, size_t &split_dim,
                                                 bool &is_first_dim) {
  const auto &input_repeats = split_node->inputs[0].attr.repeats;
  const auto &output_repeats = split_node->outputs[0].attr.repeats;
  GE_ASSERT_TRUE(input_repeats.size() == output_repeats.size(),
                 "input_repeats.size() = %zu, mismatches output_repeats.size() = %zu", input_repeats.size(),
                 output_repeats.size());
  size_t non_one_count = 0U;
  for (size_t i = 0U; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      split_dim = i;
      is_first_dim = (non_one_count == 0);
      break;
    }
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], af::ops::One) != af::TriBool::kTrue) {
      ++non_one_count;
    }
  }
  is_first_dim = (is_first_dim || (split_dim == 0UL));  // 单输入时，当成首轴转store处理
  GELOGI("node:%s input_shape = %s, output_shape = %s, is_first_dim_split = %d", split_node->GetName().c_str(),
         af::ToString(input_repeats).c_str(), af::ToString(output_repeats).c_str(), is_first_dim);
  return af::SUCCESS;
}

Status SplitFusionCaseGenerator::ConvertSplitToLoads(ascir::HintGraph &owner_graph, const af::AscNodePtr &split_node,
                                                     size_t split_dim) {
  GE_CHK_STATUS_RET(Prepare(split_node, split_dim), "Prepare failed");
  SplitGroupPartitioner partitioner(split_node, split_dim);
  GE_ASSERT_SUCCESS(partitioner.RecomputeDiffAxes());
  const auto &all_out_data_anchors = split_node->GetAllOutDataAnchors();
  // 逆序遍历，防止RemoveEdge改变下标
  for (size_t i = 0UL; i < all_out_data_anchors.size(); ++i) {
    const auto out_index = all_out_data_anchors.size() - i - 1UL;
    const auto &split_out_anchor = all_out_data_anchors.at(out_index);
    GE_CHK_STATUS_RET(ReplaceWithLoad(owner_graph, split_node, split_out_anchor), "ReplaceWithLoad failed");
  }
  GE_CHK_STATUS_RET(RemoveUnusedNodes(split_node), "RemoveUnusedNodes failed");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(owner_graph));
  GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticNameForGraph(owner_graph, semantic::kSplitOffsetLoad));
  ascir::utils::DumpGraph(owner_graph, "AfterConvertSplitToLoad");
  return af::SUCCESS;
}

Status SplitFusionCaseGenerator::SplitSplits(const ascir::HintGraph &owner_graph, const af::AscNodePtr &split_node,
                                             size_t split_dim, const bool &split) {
  (void)owner_graph;
  (void)split;
  GE_CHK_STATUS_RET(Prepare(split_node, split_dim), "Prepare failed");
  std::vector<SplitGroupPartitioner::SplitGroup> groups;
  SplitGroupPartitioner partitioner(split_node, split_dim);
  GE_ASSERT_SUCCESS(partitioner.PartitionGroups(groups));
  if ((groups.size() <= 1U) || (groups.size() == split_node->outputs().size())) {
    return af::SUCCESS;
  }
  return af::SUCCESS;
}

Status SplitFusionCaseGenerator::Prepare(const af::AscNodePtr &split_node, size_t split_dim) {
  const auto &split_in_data_nodes = split_node->GetInDataNodes();
  GE_ASSERT_TRUE(split_in_data_nodes.size() == 1UL, "Split node:%s links to %zu nodes", split_node->GetNamePtr(),
                 split_in_data_nodes.size());
  ori_load_node_ = std::dynamic_pointer_cast<af::AscNode>(split_in_data_nodes.at(0U));
  GE_CHECK_NOTNULL(ori_load_node_, "ori_store_node is nullptr or not an AscNode");
  GE_ASSERT_TRUE(ori_load_node_->GetType() == af::ascir_op::Load::Type, "Split node:%s links to %s:%s, not a Load node",
                 split_node->GetNamePtr(), ori_load_node_->GetNamePtr(), ori_load_node_->GetTypePtr());
  const auto &load_in_data_nodes = ori_load_node_->GetInDataNodes();
  GE_ASSERT_TRUE(load_in_data_nodes.size() == 1UL, "Split node:%s links to %zu nodes", split_node->GetNamePtr(),
                 load_in_data_nodes.size());
  ori_in_data_node_ = std::dynamic_pointer_cast<af::AscNode>(load_in_data_nodes.at(0U));
  GE_CHECK_NOTNULL(ori_in_data_node_, "ori_output_node is nullptr or not an AscNode");
  GE_ASSERT_TRUE(ScheduleUtils::IsDataInput(ori_in_data_node_), "Store node:%s links to %s:%s, not a Output node",
                 ori_load_node_->GetNamePtr(), ori_in_data_node_->GetNamePtr(), ori_in_data_node_->GetTypePtr());
  af::Expression dim_offset = af::ops::Zero;
  for (const auto &out_anchor : split_node->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_anchor);
    const auto &offset_expr = split_node->inputs[0].attr.strides[split_dim] * dim_offset;
    offsets_.emplace_back(offset_expr);
    dim_offset = dim_offset + split_node->outputs[static_cast<uint32_t>(out_anchor->GetIdx())].attr.repeats[split_dim];
  }
  split_axis_id_ = split_node->attr.sched.axis[split_dim];
  split_dim_ = split_dim;
  return af::SUCCESS;
}

Status SplitFusionCaseGenerator::ReplaceWithLoad(::ascir::ImplGraph &owner_graph, const af::AscNodePtr &split_node,
                                                 const af::OutDataAnchorPtr &split_out_anchor) {
  const auto out_index = split_out_anchor->GetIdx();
  GELOGD("Split node = %s, find_node is %s", split_node->GetNamePtr(), split_out_anchor->GetOwnerNode()->GetNamePtr());
  GE_ASSERT_TRUE(split_out_anchor->GetPeerInDataAnchors().size() != 0,
                 "peer_in_anchor is null, Split node = %s, input index = %u", split_node->GetNamePtr(), out_index);
  const auto data_out_anchor_peer = split_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(data_out_anchor_peer, "peer InDataAnchors is nullptr, Split node = %s, input index = %u",
                   split_node->GetNamePtr(), out_index);
  auto peer_in_node = dynamic_cast<af::AscNode *>(data_out_anchor_peer->GetOwnerNodeBarePtr());
  GE_CHECK_NOTNULL(peer_in_node, "peer node is nullptr, Split node = %s, input index = %u", split_node->GetNamePtr(),
                   out_index);

  af::ascir_op::Load load_op((ori_load_node_->GetName() + "_" + std::to_string(out_index)).c_str());
  auto load_attr = ori_load_node_->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_CHECK_NOTNULL(load_attr, "Node attr is null, node = %s", ori_load_node_->GetNamePtr());
  load_op.attr = *load_attr;
  load_op.attr.sched.axis = split_node->attr.sched.axis;
  const auto &output_tensor_attr = ori_load_node_->outputs[0];
  load_op.y.dtype = static_cast<ge::DataType>(output_tensor_attr.attr.dtype);
  *load_op.y.axis = split_node->outputs[out_index].attr.axis;
  *load_op.y.strides = output_tensor_attr.attr.strides;
  *load_op.y.repeats = split_node->outputs[out_index].attr.repeats;
  auto load_node = owner_graph.AddNode(load_op);
  GE_CHECK_NOTNULL(load_node, "Failed to create load node");

  // 如果输出的repeat为1需要将stride置为1
  if (af::SymbolicUtils::StaticCheckEq(load_node->outputs[0].attr.repeats[split_dim_], af::ops::One) ==
      af::TriBool::kTrue) {
    load_node->outputs[0].attr.strides[split_dim_] = af::ops::Zero;
  }
  // no member, safe to cast
  auto ir_attr = load_node->attr.ir_attr->DownCastTo<af::ascir_op::Load::AscLoadIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr);
  GE_CHK_STATUS_RET(ir_attr->SetOffset(offsets_[out_index]), "Failed to set offset");
  for (auto peer_in_anchor : split_out_anchor->GetPeerInDataAnchors()) {
    GELOGI("Store node: %s added, peer_node = %s:%d, offset = %s", load_node->GetName().c_str(),
           peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx(),
           offsets_[out_index].Serialize().get());
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(split_out_anchor, peer_in_anchor), "Failed to RemoveEdge");
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), peer_in_anchor), "Failed to AddEdge");
  }

  /*  根据oriload节点查找data节点，连边 */
  GE_CHK_STATUS_RET(SplitDataForConvertLoad(owner_graph, split_node, split_out_anchor, load_node),
                    "Failed to SplitData");
  std::vector<af::AscNodePtr> nodes;
  af::AscNodePtr broadcast_node;
  GE_CHK_STATUS_RET(CollectBackwardNodes(load_node, nodes, broadcast_node), "Failed to collect backward nodes");
  GE_CHK_STATUS_RET(SplitOutReplaceAxis(owner_graph, nodes, load_node, out_index, broadcast_node),
                    "Failed to replace axis");
  return af::SUCCESS;
}

af::Status SplitFusionCaseGenerator::SplitDataForConvertLoad(ascir::ImplGraph &owner_graph,
                                                             const af::AscNodePtr &split_node,
                                                             const af::OutDataAnchorPtr &split_out_anchor,
                                                             af::AscNodePtr &new_load_node) {
  (void)split_node;
  const auto out_index = split_out_anchor->GetIdx();
  std::string node_name =
      ori_in_data_node_->GetName() + std::string("_splitforconvertload") + std::to_string(out_index);
  af::ascir_op::Data data(node_name.c_str());
  auto data_node = owner_graph.AddNode(data);
  GE_ASSERT_NOTNULL(data_node);
  data_node->attr = ori_in_data_node_->attr;
  data_node->outputs[0].attr = ori_in_data_node_->outputs[0].attr;
  auto new_out_anchor = data_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(new_out_anchor);

  /* 将新创建的load节点和data节点，连边 */
  GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(new_out_anchor, new_load_node->GetInDataAnchor(0)), "Failed to AddEdge");
  return af::SUCCESS;
}

void SplitFusionCaseGenerator::IsBroadcastNode(const af::NodePtr &origin_node, af::AscNodePtr &broadcast_node,
                                               bool &has_broadcast_node) const {
  auto asc_node = std::dynamic_pointer_cast<af::AscNode>(origin_node);
  if (af::ops::IsOps<af::ascir_op::Broadcast>(asc_node)) {
    broadcast_node = asc_node;
    has_broadcast_node = true;
  }
  return;
}

af::Status SplitFusionCaseGenerator::CollectBackwardNodes(const af::AscNodePtr &load_node,
                                                          std::vector<af::AscNodePtr> &nodes,
                                                          af::AscNodePtr &broadcast_node) const {
  std::set<af::Node *> visited_nodes{load_node.get()};
  std::queue<af::NodePtr> next_nodes;
  bool has_broadcast_node = false;
  for (const auto &out_data_node : load_node->GetOutDataNodes()) {
    if (visited_nodes.emplace(out_data_node.get()).second) {
      next_nodes.push(out_data_node);
      IsBroadcastNode(out_data_node, broadcast_node, has_broadcast_node);
    }
  }
  if (has_broadcast_node == false) {
    while (!next_nodes.empty()) {
      auto &top = next_nodes.front();
      auto asc_node = std::dynamic_pointer_cast<af::AscNode>(top);
      GE_ASSERT_NOTNULL(asc_node);
      nodes.emplace_back(asc_node);
      for (const auto &in_node : top->GetInDataNodes()) {
        if (visited_nodes.emplace(in_node.get()).second) {
          next_nodes.emplace(in_node);
          IsBroadcastNode(in_node, broadcast_node, has_broadcast_node);
        }
      }
      if (has_broadcast_node == true) {
        break;
      }
      for (const auto &out_node : top->GetOutDataNodes()) {
        if (visited_nodes.emplace(out_node.get()).second) {
          next_nodes.emplace(out_node);
          IsBroadcastNode(out_node, broadcast_node, has_broadcast_node);
        }
      }
      if (has_broadcast_node == true) {
        break;
      }
      next_nodes.pop();
    }
  }

  std::sort(nodes.begin(), nodes.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) -> bool {
    return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
  });
  return af::SUCCESS;
}

af::Status SplitFusionCaseGenerator::SplitOutReplaceAxis(ascir::ImplGraph &owner_graph,
                                                         std::vector<af::AscNodePtr> &nodes,
                                                         const af::AscNodePtr &load_node_new, int32_t out_index,
                                                         af::AscNodePtr &broadcast_node) {
  ascir::Axis split_axis;
  ascir::Axis new_split_axis;
  split_axis = *(owner_graph.GetAllAxis().at(split_axis_id_));
  if (broadcast_node == nullptr) {
    const auto &output_repeats = split_node_->outputs[out_index].attr.repeats;
    new_split_axis =
        owner_graph.CreateAxis(split_axis.name + "_ss_" + std::to_string(out_index), output_repeats[split_dim_]);
  } else {
    auto broadcast_axisid = broadcast_node->attr.sched.axis[split_dim_];
    new_split_axis = *(owner_graph.GetAllAxis().at(broadcast_axisid));
  }

  GELOGD("New axis %s, size = %s", new_split_axis.name.c_str(),
         af::SymbolicUtils::ToString(new_split_axis.size).c_str());
  owner_graph.TryApplyAxisReplace(load_node_new, split_axis, new_split_axis);
  GELOGD("Replace axis for node: %s(%s) success", load_node_new->GetNamePtr(), load_node_new->GetTypePtr());
  for (const auto &asc_node : nodes) {
    if (!ScheduleUtils::IsBuffer(asc_node)) {
      owner_graph.TryApplyAxisReplace(asc_node, split_axis, new_split_axis);
      GELOGD("Replace axis for node: %s(%s) success", asc_node->GetNamePtr(), asc_node->GetTypePtr());
    }
  }
  return af::SUCCESS;
}

Status SplitFusionCaseGenerator::RemoveUnusedNodes(const af::AscNodePtr &split_node) const {
  auto owner_compute_graph = split_node->GetOwnerComputeGraph();
  GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(ori_load_node_->GetOutDataAnchor(0), split_node->GetInDataAnchor(0)),
                    "Failed to RemoveEdge");
  GE_ASSERT_NOTNULL(owner_compute_graph);
  GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(split_node), "Failed to remove node: %s", split_node->GetNamePtr());
  auto load_out_data_anchor = ori_load_node_->GetOutDataAnchor(0);
  if (load_out_data_anchor->GetPeerInDataAnchors().empty()) {
    /* 先删除data与load的边 */
    GE_CHK_STATUS_RET(
        af::GraphUtils::RemoveEdge(ori_in_data_node_->GetOutDataAnchor(0), ori_load_node_->GetInDataAnchor(0)),
        "Failed to RemoveEdge");
    GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(ori_load_node_), "Failed to remove node: %s",
                      ori_load_node_->GetNamePtr());
    auto data_node_data_anchor = ori_in_data_node_->GetOutDataAnchor(0);
    if (data_node_data_anchor->GetPeerInDataAnchors().empty()) {
      GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(ori_in_data_node_), "Failed to remove node: %s",
                        ori_in_data_node_->GetNamePtr());
    }
  }

  return af::SUCCESS;
}

}  // namespace optimize
