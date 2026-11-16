/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/cube_schedule_case_generator.h"
#include <queue>
#include <limits>
#include "common_utils.h"
#include "ascgraph_info_complete.h"
#include <unordered_set>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"
#include "ascir/meta/ascir_utils.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "optimize/schedule_utils.h"
#include "platform/platform_factory.h"
#include "util/mem_utils.h"
#include "base/err_msg.h"

namespace optimize {
namespace {
const af::Expression kSymbolZero = af::Symbol(0);
const af::Expression kSymbolOne = af::Symbol(1);

std::string GetNewNodeName(const af::AscNodePtr &src_node, const std::string &type) {
  return src_node->GetName() + "_" + type;
}

Status DoCopyAscNodeTensorAttr(const af::AscNodePtr &cube_node, af::AscNodePtr &store_node) {
  GE_ASSERT_NOTNULL(cube_node);
  GE_ASSERT_NOTNULL(store_node);
  auto op_desc = store_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = cube_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if ((src_asc_node_attr != nullptr) && (dst_asc_node_attr != nullptr)) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
  }
  GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
  auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_CHECK_NOTNULL(tensor_attr_group);
  const auto &output_attr = cube_node->outputs[0].attr;
  tensor_attr_group->dtype = static_cast<ge::DataType>(output_attr.dtype);
  tensor_attr_group->axis = output_attr.axis;
  tensor_attr_group->repeats = output_attr.repeats;
  tensor_attr_group->strides = output_attr.strides;
  return af::SUCCESS;
}

Status DoCopyInputDataTensorAttr(const af::AscNodePtr &producer_node, af::AscNodePtr &data_node) {
  GE_ASSERT_NOTNULL(producer_node);
  GE_ASSERT_NOTNULL(data_node);
  auto op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(producer_node->GetOpDesc());
  GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
  auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_CHECK_NOTNULL(tensor_attr_group);
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = producer_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if ((src_asc_node_attr != nullptr) && (dst_asc_node_attr != nullptr)) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
  }

  const auto &output_attr = producer_node->outputs[0].attr;
  tensor_attr_group->dtype = output_attr.dtype;
  tensor_attr_group->axis = output_attr.axis;
  tensor_attr_group->repeats = output_attr.repeats;
  tensor_attr_group->strides = output_attr.strides;
  return af::SUCCESS;
}

Status DoCopyAscNodeTensorAttr(const af::AscNodePtr &cube_node, const af::AscNodePtr &cube_next_node,
                               af::AscNodePtr &load_node) {
  GE_ASSERT_NOTNULL(cube_next_node);
  GE_ASSERT_NOTNULL(load_node);
  GE_ASSERT_NOTNULL(cube_node);
  auto op_desc = load_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = cube_next_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if ((src_asc_node_attr != nullptr) && (dst_asc_node_attr != nullptr)) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(0));
  auto tensor_attr_group = op_desc->MutableOutputDesc(0)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_CHECK_NOTNULL(tensor_attr_group);
  const auto &cube_next_output_attr = cube_next_node->outputs[0].attr;
  tensor_attr_group->axis = cube_next_output_attr.axis;

  const auto &cube_output_attr = cube_node->outputs[0].attr;
  tensor_attr_group->repeats = cube_output_attr.repeats;
  tensor_attr_group->strides = cube_output_attr.strides;
  tensor_attr_group->dtype = cube_output_attr.dtype;

  for (size_t i = 0U; i < cube_next_output_attr.axis.size(); i++) {
    auto it = std::find(cube_output_attr.axis.begin(), cube_output_attr.axis.end(), cube_next_output_attr.axis[i]);
    if (it == cube_output_attr.axis.end()) {
      // 给 tensor 补齐轴
      tensor_attr_group->repeats.insert(tensor_attr_group->repeats.begin() + i, kSymbolOne);
      tensor_attr_group->strides.insert(tensor_attr_group->strides.begin() + i, kSymbolZero);
    }
  }

  return af::SUCCESS;
}

Status DoCopyWorkspaceTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &workspace_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(workspace_node);
  GE_ASSERT_TRUE(src_node->outputs().size() > 0UL);
  GE_ASSERT_TRUE(workspace_node->outputs().size() > 0UL);
  workspace_node->outputs[0].attr.dtype = src_node->outputs[0].attr.dtype;
  return af::SUCCESS;
}

Status UpdateHintGraphAttr(const ::ascir::ImplGraph &graph) {
  const auto &compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph, "compute_graph is null");
  const auto compute_graph_attr = compute_graph->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_CHECK_NOTNULL(compute_graph_attr, "compute_graph_attr is null");

  for (auto node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsCube(node)) {
      continue;
    }
    const auto &cube_output_attr = node->outputs[0].attr;

    std::vector<af::AxisPtr> new_axis;
    for (size_t i = 0U; i < compute_graph_attr->axis.size(); i++) {
      auto it = std::find(cube_output_attr.axis.begin(), cube_output_attr.axis.end(), compute_graph_attr->axis[i]->id);
      if (it != cube_output_attr.axis.end()) {
        std::shared_ptr<af::Axis> axis = af::MakeShared<af::Axis>();
        GE_CHECK_NOTNULL(axis, "create axis failed");
        axis->id = compute_graph_attr->axis[i]->id;
        axis->name = compute_graph_attr->axis[i]->name;
        axis->type = compute_graph_attr->axis[i]->type;
        axis->size = compute_graph_attr->axis[i]->size;
        axis->from = compute_graph_attr->axis[i]->from;
        new_axis.push_back(std::move(axis));
      }
    }
    compute_graph_attr->axis = new_axis;
  }
  return af::SUCCESS;
}

bool IsCubeFixpip(const af::AscNodePtr &cube_node) {
  for (auto out_node : cube_node->GetOutNodes()) {
    GE_ASSERT_NOTNULL(out_node);
    auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(out_asc_node);
    if (!ScheduleUtils::IsStore(out_asc_node)) {
      return false;
    }
  }
  return true;
}

bool IsReachableToCube(const af::AscNodePtr &start_node) {
  GE_ASSERT_NOTNULL(start_node);
  std::queue<af::AscNodePtr> node_queue;
  std::unordered_set<const af::Node *> visited;
  node_queue.push(start_node);
  visited.insert(start_node.get());
  while (!node_queue.empty()) {
    auto current_node = node_queue.front();
    node_queue.pop();
    GE_ASSERT_NOTNULL(current_node);
    if (ScheduleUtils::IsCube(current_node)) {
      return true;
    }
    for (const auto &out_node : current_node->GetOutDataNodes()) {
      auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (out_asc_node == nullptr || visited.find(out_asc_node.get()) != visited.end()) {
        continue;
      }
      visited.insert(out_asc_node.get());
      node_queue.push(out_asc_node);
    }
  }
  return false;
}

std::string GetUniqueSplitDataName(af::AscGraph &graph, const std::string &base_name) {
  std::string split_name = base_name + "_shared_split_data";
  size_t suffix = 0UL;
  while (graph.FindNode(split_name.c_str()) != nullptr) {
    split_name = base_name + "_shared_split_data_" + std::to_string(++suffix);
  }
  return split_name;
}

Status CopyDataNodeForSharedSplit(af::AscGraph &graph, const af::AscNodePtr &src_data_node,
                                  const std::string &split_name, af::AscNodePtr &split_data_node) {
  GE_ASSERT_NOTNULL(src_data_node);
  af::ascir_op::Data data(split_name.c_str());
  data.attr = src_data_node->attr;
  data.y.dtype = src_data_node->outputs[0].attr.dtype;
  *data.y.axis = src_data_node->outputs[0].attr.axis;
  *data.y.repeats = src_data_node->outputs[0].attr.repeats;
  *data.y.strides = src_data_node->outputs[0].attr.strides;
  split_data_node = graph.AddNode(data);
  GE_ASSERT_NOTNULL(split_data_node);
  split_data_node->attr = src_data_node->attr;
  split_data_node->outputs[0].attr = src_data_node->outputs[0].attr;
  return af::SUCCESS;
}

Status SplitSharedDataForCubeAndVector(ascir::ImplGraph &graph,
                                       std::vector<std::pair<std::string, std::string>> &renamed_data_nodes) {
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    GE_ASSERT_NOTNULL(node);
    if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    auto data_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_ASSERT_NOTNULL(data_node);
    std::vector<af::AscNodePtr> cube_load_nodes;
    std::vector<af::AscNodePtr> vector_load_nodes;
    for (const auto &out_node : data_node->GetOutDataNodes()) {
      auto load_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (load_node == nullptr || !ScheduleUtils::IsLoad(load_node)) {
        continue;
      }
      if (IsReachableToCube(load_node)) {
        cube_load_nodes.emplace_back(load_node);
      } else {
        vector_load_nodes.emplace_back(load_node);
      }
    }
    if (cube_load_nodes.empty() || vector_load_nodes.empty()) {
      continue;
    }

    const auto original_name = data_node->GetName();
    const auto split_name = GetUniqueSplitDataName(graph, original_name);
    af::AscNodePtr split_data_node;
    GE_CHK_STATUS_RET(CopyDataNodeForSharedSplit(graph, data_node, split_name, split_data_node));
    for (const auto &load_node : vector_load_nodes) {
      GE_ASSERT_NOTNULL(load_node);
      auto in_anchor = load_node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(in_anchor);
      auto src_anchor = data_node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(src_anchor);
      auto new_src_anchor = split_data_node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(new_src_anchor);
      GE_CHK_STATUS_RET(af::GraphUtils::ReplaceEdgeSrc(src_anchor, in_anchor, new_src_anchor));
    }
    renamed_data_nodes.emplace_back(split_name, original_name);
  }
  return af::SUCCESS;
}

Status RestoreSplitDataNames(std::vector<ascir::ImplGraph> &grouped_graphs,
                             const std::vector<std::pair<std::string, std::string>> &renamed_data_nodes) {
  for (auto &grouped_graph : grouped_graphs) {
    for (const auto &name_pair : renamed_data_nodes) {
      const auto &split_name = name_pair.first;
      const auto &original_name = name_pair.second;
      auto split_node = grouped_graph.FindNode(split_name.c_str());
      if (split_node == nullptr) {
        continue;
      }
      auto original_node = grouped_graph.FindNode(original_name.c_str());
      GE_ASSERT_TRUE(original_node == nullptr, "Graph %s already has Data node %s, cannot restore split Data name.",
                     grouped_graph.GetName().c_str(), original_name.c_str());
      GE_CHECK_NOTNULL(split_node->GetOpDesc());
      split_node->GetOpDesc()->SetName(original_name.c_str());
    }
  }
  return af::SUCCESS;
}

Status GetPrioritySequence(const af::AscGraph &graph, std::unordered_set<af::Node *> &priority_sequences,
                           std::unordered_set<af::Node *> &store_sequences) {
  std::unordered_set<const af::Node *> visited;
  std::queue<af::NodePtr> node_queue;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName().find("Cube_Load_") == std::string::npos) {
      continue;
    }
    visited.insert(node.get());
    for (const auto &cube_output_node : node->GetOutNodes()) {
      GE_ASSERT_NOTNULL(cube_output_node);
      if (af::ops::IsOps<af::ascir_op::Store>(cube_output_node)) {
        store_sequences.insert(cube_output_node.get());
        continue;
      }
      node_queue.emplace(cube_output_node);
      visited.insert(cube_output_node.get());
    }
    while (!node_queue.empty()) {
      const auto current_node = node_queue.front();
      node_queue.pop();
      priority_sequences.insert(current_node.get());
      for (const auto &next_node : current_node->GetOutAllNodes()) {
        if (visited.find(next_node.get()) == visited.end()) {
          visited.insert(next_node.get());
          node_queue.emplace(next_node);
        }
      }
    }
  }
  return af::GRAPH_SUCCESS;
}

Status TopoSortByCubePriority(af::AscGraph &graph) {
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  std::unordered_set<af::Node *> priority_sequences;
  std::unordered_set<af::Node *> store_sequences;
  GE_ASSERT_GRAPH_SUCCESS(GetPrioritySequence(graph, priority_sequences, store_sequences),
                          "Get priority sequence failed.");
  if (priority_sequences.empty()) {
    return af::GRAPH_SUCCESS;
  }

  auto vec_node = *priority_sequences.begin();
  GE_ASSERT_NOTNULL(vec_node);
  GE_ASSERT_NOTNULL(vec_node->GetOpDesc());
  auto vec_node_id = vec_node->GetOpDesc()->GetId();
  int64_t store_id_min = std::numeric_limits<int64_t>::max();
  ;
  af::Node *store_node_id_min = nullptr;
  for (auto store_node : store_sequences) {
    GE_ASSERT_NOTNULL(store_node);
    GE_ASSERT_NOTNULL(store_node->GetOpDesc());
    auto store_id = store_node->GetOpDesc()->GetId();
    if (store_id < store_id_min) {
      store_id_min = store_id;
      store_node_id_min = store_node;
    }
  }
  if (store_id_min < vec_node_id) {
    GE_ASSERT_NOTNULL(store_node_id_min);
    GE_ASSERT_NOTNULL(store_node_id_min->GetOpDesc());
    vec_node->GetOpDesc()->SetId(store_id_min);
    store_node_id_min->GetOpDesc()->SetId(vec_node_id);
  }

  const auto func = [&priority_sequences](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    bool is_node1_in_priority_seq = priority_sequences.find(node1.get()) != priority_sequences.end();
    bool is_node2_in_priority_seq = priority_sequences.find(node2.get()) != priority_sequences.end();
    if (is_node1_in_priority_seq && !is_node2_in_priority_seq) {
      return true;
    } else {
      return node1->GetOpDescBarePtr()->GetId() < node2->GetOpDescBarePtr()->GetId();
    }
  };

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->TopologicalSorting(func);
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return af::SUCCESS;
}
bool HasBroadCastNode(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type == af::ComputeType::kComputeBroadcast && !ScheduleUtils::IsScalarBrc(node) &&
        ascir::GetCvBroadcastShape(*node).empty()) {
      return true;
    }
  }
  return false;
}

// Keep the final Broadcast visible to UB allocation and tiling.
Status MarkCubeOutputBroadcast(af::AscGraph &graph, const af::AscNodePtr &load, bool &requires_common) {
  if (load == nullptr || load->outputs().size() != 1UL || load->inputs().size() != 1UL) {
    return af::UNSUPPORTED;
  }
  const auto producer = load->GetInDataAnchor(0)->GetPeerOutAnchor();
  if (producer == nullptr || !af::ops::IsOps<af::ascir_op::Workspace>(producer->GetOwnerNode())) {
    return af::UNSUPPORTED;
  }
  std::vector<af::AscNodePtr> broadcasts;
  std::vector<af::AscNodePtr> chain;
  constexpr int64_t kMaxBroadcastCastElementBytes = 4;
  auto next = load;
  while (next->GetOutDataNodesSize() == 1UL) {
    next = std::dynamic_pointer_cast<af::AscNode>(next->GetOutDataNodes().at(0));
    if (next == nullptr || next->outputs().size() != 1UL || next->inputs().size() != 1UL) {
      break;
    }
    if (af::ops::IsOps<af::ascir_op::Cast>(next) &&
        af::GetSizeByDataType(next->outputs[0].attr.dtype) <= kMaxBroadcastCastElementBytes) {
      chain.push_back(next);
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Broadcast>(next)) {
      break;
    }
    broadcasts.push_back(next);
    chain.push_back(next);
  }
  if (broadcasts.empty()) {
    return af::UNSUPPORTED;
  }
  const auto &src = load->outputs[0].attr;
  const auto &dst = broadcasts.back()->outputs[0].attr;
  if (src.axis.size() != 2UL || src.axis != dst.axis || src.repeats.size() != 2UL || dst.repeats.size() != 2UL) {
    return af::UNSUPPORTED;
  }
  const auto eq = [](const af::Expression &lhs, const af::Expression &rhs) {
    return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kTrue;
  };
  bool has_broadcast = false;
  for (size_t i = 0UL; i < 2UL; ++i) {
    if (!eq(src.repeats[i], dst.repeats[i])) {
      if (!eq(src.repeats[i], kSymbolOne)) {
        return af::UNSUPPORTED;
      }
      has_broadcast = true;
    }
  }
  if (!has_broadcast) {
    return af::UNSUPPORTED;
  }
  if (!eq(src.repeats[0], dst.repeats[0])) {
    int64_t full_m = 0;
    int64_t full_n = 0;
    if (!dst.repeats[0].GetConstValue(full_m) || !dst.repeats[1].GetConstValue(full_n) || full_m <= 0 ||
        full_m > std::numeric_limits<uint16_t>::max() || full_n <= 0) {
      requires_common = true;
      return af::UNSUPPORTED;
    }
  }
  if (std::find(broadcasts.back()->attr.sched.axis.begin(), broadcasts.back()->attr.sched.axis.end(), dst.axis[1]) ==
      broadcasts.back()->attr.sched.axis.end()) {
    return af::UNSUPPORTED;  // No searchable N loop remains after singleton-axis normalization.
  }
  if (broadcasts.size() > 1UL) {
    for (const auto &node : chain) {
      if (node->GetInControlNodesSize() != 0UL || node->GetOutControlNodesSize() != 0UL) {
        return af::UNSUPPORTED;
      }
    }
  }
  // Move single-use Casts before the final Broadcast to avoid a padded [M,1] intermediate.
  const auto final_broadcast = broadcasts.back();
  for (const auto &node : chain) {
    if (node == final_broadcast) {
      break;
    }
    if (af::ops::IsOps<af::ascir_op::Cast>(node)) {
      node->outputs[0].attr.axis = src.axis;
      node->outputs[0].attr.repeats = src.repeats;
      node->outputs[0].attr.strides = src.strides;
    } else {
      const auto input = node->GetInDataAnchor(0)->GetPeerOutAnchor();
      if (ScheduleUtils::RemoveNode(graph, node, input) != af::SUCCESS ||
          graph.FindNode(node->GetNamePtr()) != nullptr) {
        return af::FAILED;
      }
    }
  }
  if (broadcasts.size() > 1UL && ascgen_utils::SetTemplateSemanticTransform(
                                     final_broadcast, ascgen_utils::template_semantic::kBroadcastFold) != af::SUCCESS) {
    return af::FAILED;
  }
  for (const auto &node : {load, final_broadcast}) {
    if (ascir::SetCvBroadcastShape(node, dst.repeats, dst.axis[1], src.repeats, dst.axis[0]) != af::SUCCESS) {
      return af::FAILED;
    }
  }
  return af::SUCCESS;
}

bool IsSupportedOutputBroadcastGraph(const ascir::ImplGraph &graph) {
  std::vector<af::Expression> shape;
  for (const auto &node : graph.GetAllNodes()) {
    const auto tagged_shape = ascir::GetCvBroadcastShape(*node);
    if (!tagged_shape.empty()) {
      shape = tagged_shape;
      break;
    }
  }
  if (shape.empty()) {
    return true;
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node) || ScheduleUtils::IsScalarLikeNode(node) || ScheduleUtils::IsScalarBrc(node) ||
        !ascir::GetCvBroadcastShape(*node).empty()) {
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node)) {
      // The CV callback reconstructs GM addresses from tile coordinates.
      af::Expression offset = kSymbolZero;
      if (node->attr.ir_attr != nullptr) {
        (void)node->attr.ir_attr->GetAttrValue("offset", offset);
      }
      if (af::SymbolicUtils::StaticCheckEq(offset, kSymbolZero) != af::TriBool::kTrue) {
        return false;
      }
      const auto &attr = node->outputs[0].attr;
      if (attr.repeats.size() != 2UL || attr.strides.size() != 2UL) {
        return false;
      }
      const auto eq = [](const af::Expression &lhs, const af::Expression &rhs) {
        return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kTrue;
      };
      if (!eq(attr.repeats[0], shape[0]) || !eq(attr.repeats[1], shape[1]) || !eq(attr.strides[0], shape[1]) ||
          !eq(attr.strides[1], kSymbolOne)) {
        return false;
      }
      continue;
    }
    if (node->attr.api.compute_type != af::ComputeType::kComputeElewise) {
      return false;
    }
  }
  return true;
}

Status FillInputDataAttrForCvGraph(af::AscGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    const auto &output_attr = node->outputs[0].attr;
    if (!output_attr.repeats.empty() && !output_attr.strides.empty()) {
      continue;
    }
    for (const auto &out_node : node->GetOutNodes()) {
      auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (out_asc_node == nullptr || !ScheduleUtils::IsLoad(out_asc_node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(DoCopyInputDataTensorAttr(out_asc_node, node));
      break;
    }
  }
  return af::SUCCESS;
}
}  // namespace

Status CubeFusionCaseGenerator::GenerateGeneralCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs) {
  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);

  for (const auto &node : optimize_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (node->GetOutDataNodes().empty()) {
      node_order_.emplace_back(node);
    }
    if (!ScheduleUtils::IsCube(node)) {
      continue;
    }
    if (IsCubeFixpip(node)) {
      GELOGI("graph %s is fixpip, generate fixpip task.", optimize_graph.GetName().c_str());
      break;
    }
    node_order_.emplace_back(node);
    partition_ = true;
    bool first = true;
    af::AscNodePtr workspace_pre_node;
    af::AscNodePtr workspace_post_node;
    af::AscNodePtr load_node;
    af::AscNodePtr store_node;
    af::ascir_op::Workspace workspace_pre(GetNewNodeName(node, "Workspace").c_str());
    af::ascir_op::Workspace workspace_post(GetNewNodeName(node, "Workspace").c_str());
    af::ascir_op::Load load(("Cube_Load_" + GetNewNodeName(node, "Load")).c_str());
    af::ascir_op::Store store(GetNewNodeName(node, "Store").c_str());
    for (const auto &cube_output_node : node->GetOutNodes()) {
      GE_CHECK_NOTNULL(cube_output_node);
      af::AscNodePtr cube_node = std::dynamic_pointer_cast<af::AscNode>(cube_output_node);
      GE_ASSERT_NOTNULL(cube_node);
      if (first) {
        workspace_pre_node = optimize_graph.AddNode(workspace_pre);
        GE_ASSERT_NOTNULL(workspace_pre_node);
        workspace_post_node = optimize_graph.AddNode(workspace_post);
        GE_ASSERT_NOTNULL(workspace_post_node);
        load_node = optimize_graph.AddNode(load);
        GE_ASSERT_NOTNULL(load_node);
        store_node = optimize_graph.AddNode(store);
        GE_ASSERT_NOTNULL(store_node);
        GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(node, cube_node, load_node));
        GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(node, store_node));
        GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
        GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
      }
      for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
        GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                         REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", node->GetName().c_str());
                         return af::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                                node->GetName().c_str());
        for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
          GE_CHECK_NOTNULL(peer_in_anchor);
          GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                           REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", node->GetName().c_str());
                           return af::GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
          if (peer_in_anchor->GetOwnerNodeBarePtr()->GetName() == cube_node->GetName()) {
            // remove src->dst
            GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(node->GetOutAnchor(out_anchor->GetIdx()),
                                                         cube_node->GetInAnchor(peer_in_anchor->GetIdx())));
            // load->dst
            GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                      cube_node->GetInAnchor(peer_in_anchor->GetIdx())));
            break;  // 这里加的break可能会导致add两个输入都来自cube时切图只切了一半
          }
        }
      }
      if (first) {
        // add src->store->workspace_pre_node
        GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(node->GetOutAnchor(0UL), store_node->GetInAnchor(0UL)));
        GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
        // add workspace_post_node->load
        GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
      }
      first = false;
    }
  }
  ascir::utils::DumpGraph(graph, "before_partition");
  ascir::utils::DumpGraph(optimize_graph, "after_partition");
  split_data_names_.clear();
  GE_ASSERT_SUCCESS(SplitSharedDataForCubeAndVector(optimize_graph, split_data_names_));
  // UB eligibility needs API metadata for the Workspace and its Load/Store.
  GE_ASSERT_SUCCESS(AscGraphInfoComplete::CompleteApiInfo(optimize_graph));
  GE_ASSERT_SUCCESS(FillInputDataAttrForCvGraph(optimize_graph));
  graphs.emplace_back(optimize_graph);
  return af::GRAPH_SUCCESS;
}

Status CubeFusionCaseGenerator::GenNddmaNode(const af::AscNodePtr &node_load, const af::AscNodePtr &node_brc,
                                             af::AscGraph &new_case) {
  GE_CHECK_NOTNULL(node_load);
  GE_CHECK_NOTNULL(node_load->GetOpDesc());
  GE_CHECK_NOTNULL(node_brc);
  // 继承自Load节点，把Load节点的offset信息和是否brc缓存信息同步继承过来
  node_load->GetOpDesc()->SetType("Nddma");
  node_load->attr.type = "Nddma";
  node_load->attr.sched = node_brc->attr.sched;
  node_load->outputs[0].attr = node_brc->outputs[0].attr;
  GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, std::dynamic_pointer_cast<af::AscNode>(node_brc),
                                              node_load->GetOutDataAnchor(0)));
  return af::SUCCESS;
}

af::Status CubeFusionCaseGenerator::SwapCastBrcAndGenNddma(const af::AscNodePtr &node_cast,
                                                           const af::AscNodePtr &node_load, af::AscGraph &new_case) {
  // 针对cast输出多引用的场景不做处理
  if (node_cast->GetOutNodesSize() != 1UL) {
    GELOGD("Node %s with single output and multiple refs, do not support gen nddma.", node_cast->GetNamePtr());
    return af::UNSUPPORTED;
  }
  // 判断是否为load-cast-brc场景
  auto cast_out_anchor = node_cast->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(cast_out_anchor);
  auto next_in_anchor = cast_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(next_in_anchor);
  const auto &next_node = std::dynamic_pointer_cast<af::AscNode>(next_in_anchor->GetOwnerNode());
  GE_CHECK_NOTNULL(next_node);
  if (!af::ops::IsOps<af::ascir_op::Broadcast>(next_node)) {
    GELOGD("The subgraph is not load-cast-brc, do not gen nddma.");
    return af::UNSUPPORTED;
  }
  node_cast->attr.sched = next_node->attr.sched;
  node_cast->outputs[0].attr = next_node->outputs[0].attr;
  next_node->outputs[0].attr.dtype = node_load->outputs[0].attr.dtype;
  auto load_out_anchor = node_load->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(load_out_anchor);
  auto cast_in_anchor = load_out_anchor->GetPeerInDataAnchors().at(0);
  GE_CHECK_NOTNULL(cast_in_anchor);
  auto brc_out_anchor = next_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(brc_out_anchor);
  // 将cast-->brc替换为load-->brc
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(cast_out_anchor, next_in_anchor, load_out_anchor));
  // 将load-->cast替换为brc-->cast
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(load_out_anchor, cast_in_anchor, brc_out_anchor));
  // 将brc-->others替换为cast-->others
  for (const auto &peer_in_anchor : brc_out_anchor->GetPeerInDataAnchors()) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    // 跳过cast_in_anchor这条边
    if (peer_in_anchor == cast_in_anchor) {
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(brc_out_anchor, peer_in_anchor, cast_out_anchor));
  }

  GE_ASSERT_SUCCESS(GenNddmaNode(node_load, std::dynamic_pointer_cast<af::AscNode>(next_node), new_case));
  return af::SUCCESS;
}

Status CubeFusionCaseGenerator::GeneratorUbTask(const std::vector<::ascir::ImplGraph> &grouped_graphs,
                                                ScheduleTask &ub_task, std::vector<ScheduleTask> &tasks) {
  std::vector<::ascir::ImplGraph> tmp_grouped_graphs;
  for (auto &grouped_graph : grouped_graphs) {
    ascir::ImplGraph optimize_graph((grouped_graph.GetName() + "_ub").c_str());
    if (!optimize_graph.CopyFrom(grouped_graph)) {
      GELOGW("Discard CV UB candidate because graph %s could not be copied.", grouped_graph.GetName().c_str());
      return af::SUCCESS;
    }
    if (ScheduleUtils::HasComputeType(optimize_graph, af::ComputeType::kComputeCube)) {
      tmp_grouped_graphs.emplace_back(optimize_graph);
      continue;
    }
    for (auto node : optimize_graph.GetAllNodes()) {
      GE_CHECK_NOTNULL(node);
      if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
        continue;
      }
      const auto input_anchor = node->GetInDataAnchor(0);
      if (input_anchor != nullptr && input_anchor->GetPeerOutAnchor() != nullptr &&
          af::ops::IsOps<af::ascir_op::Workspace>(input_anchor->GetPeerOutAnchor()->GetOwnerNode())) {
        bool requires_common = false;
        const auto status =
            MarkCubeOutputBroadcast(optimize_graph, std::dynamic_pointer_cast<af::AscNode>(node), requires_common);
        if (requires_common) {
          return af::SUCCESS;  // M broadcast without a valid full-M extent has no UB candidate.
        }
        if (status == af::SUCCESS) {
          continue;  // The CV callback consumes the Cube fragment directly.
        }
        if (status != af::UNSUPPORTED) {
          GELOGW("Discard CV UB candidate after broadcast rewrite failed in graph %s.",
                 optimize_graph.GetName().c_str());
          return af::SUCCESS;
        }
      }
      if (node->GetOutAllNodes().size() > 1UL) {
        GELOGD("Node %s with single output and multiple refs, do not support nddma.", node->GetNamePtr());
        continue;
      }
      auto load_out_anchor = node->GetOutDataAnchor(0);
      GE_CHECK_NOTNULL(load_out_anchor);
      auto peer_in_anchor = load_out_anchor->GetPeerInDataAnchors().at(0);
      GE_CHECK_NOTNULL(peer_in_anchor);
      const auto &out_node = std::dynamic_pointer_cast<af::AscNode>(peer_in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(out_node);
      if (af::ops::IsOps<af::ascir_op::Broadcast>(out_node)) {
        GE_ASSERT_SUCCESS(GenNddmaNode(node, std::dynamic_pointer_cast<af::AscNode>(out_node), optimize_graph),
                          "Generator nddma node failed.");
      }
      if (af::ops::IsOps<af::ascir_op::Cast>(out_node)) {
        auto ret = SwapCastBrcAndGenNddma(std::dynamic_pointer_cast<af::AscNode>(out_node), node, optimize_graph);
        if (ret == af::UNSUPPORTED) {
          continue;
        }
        if (ret != af::GRAPH_SUCCESS) {
          GELOGE(ret, "Swap cast and brc, generator nddma node failed.");
          return ret;
        }
      }
    }
    tmp_grouped_graphs.emplace_back(optimize_graph);
    if (HasBroadCastNode(optimize_graph) || !IsSupportedOutputBroadcastGraph(optimize_graph)) {
      GELOGW("The graph %s still contains broadcast nodes and does not support generating ub task.",
             grouped_graph.GetName().c_str());
      return af::GRAPH_SUCCESS;
    }
  }
  bool m_broadcast = false;
  for (const auto &graph : tmp_grouped_graphs) {
    for (const auto &node : graph.GetAllNodes()) {
      m_broadcast = m_broadcast || ascir::IsCvMBroadcast(*node);
    }
  }
  if (m_broadcast) {
    for (const auto &graph : tmp_grouped_graphs) {
      for (const auto &axis : graph.GetAllAxis()) {
        if (!axis->size.IsConstExpr()) {
          return af::SUCCESS;  // Keep the unmodified Common candidate.
        }
      }
      for (const auto &node : graph.GetAllNodes()) {
        for (const auto &output : node->outputs()) {
          for (const auto &dim : output->attr.repeats) {
            if (!dim.IsConstExpr()) {
              return af::SUCCESS;
            }
          }
        }
      }
    }
  }
  bool n_broadcast = false;
  size_t cube_count = 0UL;
  size_t vector_group_count = 0UL;
  for (const auto &tmp_graph : tmp_grouped_graphs) {
    bool has_cube = false;
    for (const auto &node : tmp_graph.GetAllNodes()) {
      if (ScheduleUtils::IsCube(node)) {
        ++cube_count;
        has_cube = true;
      }
      n_broadcast = n_broadcast || !ascir::GetCvBroadcastShape(*node).empty();
    }
    vector_group_count += has_cube ? 0UL : 1UL;
  }
  if (n_broadcast && (cube_count != 1UL || vector_group_count != 1UL)) {
    return af::GRAPH_SUCCESS;  // A callback owns one MM result and one vector domain.
  }
  for (const auto &tmp_graph : tmp_grouped_graphs) {
    ub_task.grouped_graphs.emplace_back(tmp_graph);
  }
  tasks.push_back(std::move(ub_task));
  return af::GRAPH_SUCCESS;
}

void MoveCubeGraphsToEnd(std::vector<::ascir::ImplGraph> &grouped_graphs) {
  (void)std::stable_partition(grouped_graphs.begin(), grouped_graphs.end(), [](const ::ascir::ImplGraph &graph) {
    // 返回true表示保留在前面，false表示移动到后面
    return !ScheduleUtils::HasComputeType(graph, af::ComputeType::kComputeCube);
  });
  // it现在指向第一个Cube类型元素的位置
}

Status CubeFusionCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                                              const OptimizerOptions &options) {
  (void)options;
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  if (!ScheduleUtils::HasComputeType(optimize_graph, af::ComputeType::kComputeCube)) {
    return af::GRAPH_SUCCESS;
  }
  GE_CHK_STATUS_RET(GenerateGeneralCase(optimize_graph, optimize_graphs), "GenerateScheduleCases failed");
  score_funcs.resize(optimize_graphs.size());
  if (partition_) {
    std::sort(node_order_.begin(), node_order_.end(), [](const af::AscNodePtr &lhs, af::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });
  } else {
    node_order_.clear();
  }
  for (size_t i = 0U; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    ScheduleTask task{graph, {}, score_funcs[i], {}, ReduceTemplateType::kDefault, ascir::CubeTemplateType::kFixpip};
    GE_CHK_STATUS_RET(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, task.grouped_graphs, node_order_),
                      "Failed to partition graph");
    GE_CHK_STATUS_RET(RestoreSplitDataNames(task.grouped_graphs, split_data_names_), "Restore split Data names failed");
    if (task.grouped_graphs.size() > 1U) {
      task.cube_type = ascir::CubeTemplateType::kCommon;
      MoveCubeGraphsToEnd(task.grouped_graphs);
      for (auto grouped_graph : task.grouped_graphs) {
        if (ScheduleUtils::HasComputeType(grouped_graph, af::ComputeType::kComputeCube)) {
          GE_CHK_STATUS_RET(UpdateHintGraphAttr(grouped_graph), "UpdateHintGraphAttr failed");
        } else {
          GE_ASSERT_GRAPH_SUCCESS(TopoSortByCubePriority(grouped_graph), "Failed to topo sort by cube priority.");
        }
      }
      ScheduleTask ub_task{
          graph, {}, score_funcs[i], {}, ReduceTemplateType::kDefault, ascir::CubeTemplateType::kUBFuse};
      GE_ASSERT_SUCCESS(GeneratorUbTask(task.grouped_graphs, ub_task, tasks), "Generator ub task failed.");
    }
    tasks.emplace_back(std::move(task));
  }
  return af::GRAPH_SUCCESS;
}

Status CubeFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                         std::vector<std::string> &score_functions) {
  (void)graph;
  (void)graphs;
  (void)score_functions;
  return af::GRAPH_SUCCESS;
}
}  // namespace optimize
