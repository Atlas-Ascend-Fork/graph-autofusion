/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pre_process/improve_precision.h"

#include <atomic>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ascir_ops.h"
#include "schedule_utils.h"
#include "graph_properties_cache.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "common/platform_context.h"
#include "pre_process/pre_process_config.h"
#include "ascgen_log.h"

namespace af::pre_process {
namespace {
using af::AscGraph;
using af::AscGraphUtils;
using af::AscNodeAttr;
using af::AscTensorAttr;
using af::ComputeGraphPtr;
using af::GeTensorDescPtr;
using af::GraphUtils;
using af::NodePtr;
using af::NodeUtils;
using af::OpDescBuilder;
using af::OpDescUtils;
using af::TypeUtils;
using ge::DataType;
using ge::DT_BF16;
using ge::DT_FLOAT;
using ge::DT_FLOAT16;
using ge::DT_INT4;
using ge::DT_INT8;
using ge::DT_UINT8;

bool IsLowPrecisionDataType(DataType dtype) {
  return dtype == DT_FLOAT16 || dtype == DT_BF16;
}

bool IsHighPrecisionDataType(DataType dtype) {
  return dtype == DT_FLOAT;
}

bool IsFloatDataType(DataType dtype) {
  return (IsLowPrecisionDataType(dtype) || IsHighPrecisionDataType(dtype));
}

bool IsUltraLowPrecisionDataType(DataType dtype) {
  return dtype == DT_INT4 || dtype == DT_INT8 || dtype == DT_UINT8;
}

bool IsFloatToUltraLowPrecision(DataType peer_output_dtype, DataType output_dtype) {
  return IsFloatDataType(peer_output_dtype) && IsUltraLowPrecisionDataType(output_dtype);
}

bool IsUltraLowToLowPrecision(DataType peer_output_dtype, DataType output_dtype) {
  return IsUltraLowPrecisionDataType(peer_output_dtype) && IsLowPrecisionDataType(output_dtype);
}

bool IsNodeTypeInPeerInNodes(const std::string &node_type, const std::vector<NodePtr> &peer_in_nodes) {
  for (const auto &peer_in_node : peer_in_nodes) {
    if (peer_in_node->GetType() == node_type) {
      return true;
    }
  }
  return false;
}

Status GetPeerOutNode(const NodePtr &node, NodePtr &peer_out_node, int32_t idx) {
  const auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node);
  return af::SUCCESS;
}

Status GetPeerOutNodes(const NodePtr &node, std::vector<NodePtr> &peer_out_nodes) {
  auto size = static_cast<int32_t>(node->GetAllInDataAnchorsSize());
  for (auto i = 0; i < size; i++) {
    const auto in_anchor = node->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto peer_out_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_out_node);
    peer_out_nodes.push_back(peer_out_node);
  }
  return af::SUCCESS;
}

Status GetPeerInNodes(const NodePtr &node, std::vector<NodePtr> &peer_in_nodes, int32_t out_data_idx) {
  const auto out_anchor = node->GetOutDataAnchor(out_data_idx);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(peer_in_anchor);
    const auto peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    peer_in_nodes.push_back(peer_in_node);
  }
  return af::SUCCESS;
}

Status GetOutputTensorDesc(const NodePtr &node, GeTensorDescPtr &output_tensor_desc) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  output_tensor_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  return af::SUCCESS;
}

Status GetNodeOutputDtype(const NodePtr &node, DataType &dtype) {
  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(GetOutputTensorDesc(node, output_tensor_desc));
  dtype = output_tensor_desc->GetDataType();
  return af::SUCCESS;
}

Status SetNodeOutputDtype(const NodePtr &node, DataType dtype) {
  const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_TRUE(!asc_node->outputs().empty());
  asc_node->outputs[0].attr.dtype = dtype;
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto output_desc = op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_desc);
  output_desc->SetDataType(dtype);
  return af::SUCCESS;
}

Status DelNode(AscGraph &asc_graph, const NodePtr &node) {
  const auto in_data_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_data_anchor);
  const auto out_data_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto src_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src_anchor, in_data_anchor));
  for (const auto &dst_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(dst_anchor);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(out_data_anchor, dst_anchor));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src_anchor, dst_anchor));
  }
  GELOGD("Remove node: %s(%s) from asc_graph:%s.", node->GetName().c_str(), node->GetType().c_str(),
         asc_graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(AscGraphUtils::GetComputeGraph(asc_graph), node));
  NodeUtils::UnlinkAll(*node);
  return af::SUCCESS;
}

Status UpdateTopoId(AscGraph &asc_graph, const NodePtr &node, int64_t topo_id_increment) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto topo_id = op_desc->GetId();
  auto compute_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  for (const auto &n : compute_graph->GetAllNodes()) {
    const auto &n_desc = n->GetOpDesc();
    GE_ASSERT_NOTNULL(n_desc);
    if (n_desc->GetId() > topo_id) {
      n_desc->SetId(n_desc->GetId() + topo_id_increment);
    }
  }
  return af::SUCCESS;
}

Status FromDtypeToOtherDtype(const NodePtr &node, DataType s_dtype, DataType d_dtype) {
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto node_output_desc_size = node_opdesc->GetAllOutputsDescSize();
  for (auto i = 0U; i < node_output_desc_size; i++) {
    const auto output_tensor_desc = node_opdesc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    if (output_tensor_desc->GetDataType() == s_dtype) {
      output_tensor_desc->SetDataType(d_dtype);
    }
  }
  return af::SUCCESS;
}

void TopologicalSorting(const ComputeGraphPtr &graph) {
  graph->TopologicalSorting(
      [](const af::NodePtr &a, const af::NodePtr &b) { return a->GetOpDesc()->GetId() < b->GetOpDesc()->GetId(); });
}

bool CheckCastDtype(DataType input_dtype, DataType output_dtype) {
  std::vector<DataType> input_dtypes = {input_dtype};
  std::vector<DataType> expect_output_dtypes = {output_dtype};
  return optimize::ScheduleUtils::CallAscirInferDataType<af::ascir_op::Cast>(input_dtypes, expect_output_dtypes) ==
         af::SUCCESS;
}

bool IsTransposeSupportedDtype(DataType dtype) {
  return optimize::ScheduleUtils::IsNodeSupportDataType<af::ascir_op::Transpose>(dtype);
}

std::atomic<int64_t> g_unique_number{0};

int64_t GenUniqueNumber() {
  return g_unique_number.fetch_add(1);
}

// ====================== Blacklist ======================

const std::unordered_map<std::string, std::string> kBlackList1 = {
    {af::ascir_op::Data::Type, af::ascir_op::Data::Type},
    {af::ascir_op::Load::Type, af::ascir_op::Load::Type},
    {af::ascir_op::Scalar::Type, af::ascir_op::Scalar::Type},
    {af::ascir_op::Store::Type, af::ascir_op::Store::Type},
    {af::ascir_op::Output::Type, af::ascir_op::Output::Type},
    {af::ascir_op::Broadcast::Type, af::ascir_op::Broadcast::Type},
    {af::ascir_op::Transpose::Type, af::ascir_op::Transpose::Type},
    {af::ascir_op::Concat::Type, af::ascir_op::Concat::Type},
    {af::ascir_op::Gather::Type, af::ascir_op::Gather::Type},
    {"Slice", "Slice"}};

bool IsInBlackList1(const NodePtr &node) {
  return kBlackList1.find(node->GetType()) != kBlackList1.end();
}

bool IsInBlackList2(const NodePtr &node, const std::unordered_set<std::string> &blacklist2) {
  return blacklist2.find(node->GetType()) != blacklist2.end();
}

Status CheckNodeDtype(const NodePtr &node) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(node, peer_out_nodes));
  std::vector<DataType> input_dtypes;
  for (const auto &peer_out_node : peer_out_nodes) {
    GeTensorDescPtr peer_output_tensor_desc;
    GE_ASSERT_SUCCESS(GetOutputTensorDesc(peer_out_node, peer_output_tensor_desc));
    input_dtypes.push_back(peer_output_tensor_desc->GetDataType());
  }
  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(GetOutputTensorDesc(node, output_tensor_desc));
  std::vector<DataType> expect_output_dtypes = {output_tensor_desc->GetDataType()};
  std::string npu_arch;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
  if (af::ascir::CommonInferDtype(node->GetType(), input_dtypes, expect_output_dtypes, npu_arch) != af::SUCCESS) {
    GELOGE(af::FAILED,
           "Node %s(%s) with dtype(%s) is not supported. "
           "Do not configure it in autofuse_enhance_precision_blacklist",
           node->GetName().c_str(), node->GetType().c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor_desc->GetDataType()).c_str());
    return af::FAILED;
  }
  return af::SUCCESS;
}

const std::unordered_map<std::string, std::string> kTypeToGroup = {
    {af::ascir_op::Cast::Type, af::ascir_op::Cast::Type},
    {af::ascir_op::Load::Type, af::ascir_op::Load::Type},
    {af::ascir_op::Gather::Type, af::ascir_op::Gather::Type},
    {af::ascir_op::Scalar::Type, af::ascir_op::Scalar::Type},
    {af::ascir_op::Store::Type, af::ascir_op::Store::Type}};

struct CastChain {
  std::vector<NodePtr> nodes;
  std::vector<DataType> output_dtypes;
};

bool CanCollapseCast(DataType input_dtype, DataType output_dtype) {
  return input_dtype == output_dtype ||
         (IsFloatDataType(input_dtype) && IsFloatDataType(output_dtype) && CheckCastDtype(input_dtype, output_dtype));
}

struct CastChainPlan {
  std::vector<size_t> costs;
  std::vector<size_t> previous;
  std::vector<bool> requires_cast;
};

Status CollectLinearCastChain(const NodePtr &start, CastChain &chain) {
  chain.nodes.clear();
  chain.output_dtypes.clear();
  NodePtr current = start;
  while (current->GetType() == af::ascir_op::Cast::Type) {
    std::vector<NodePtr> consumers;
    GE_ASSERT_SUCCESS(GetPeerInNodes(current, consumers, 0));
    if (consumers.size() != 1U) {
      chain.nodes.clear();
      chain.output_dtypes.clear();
      return af::SUCCESS;
    }
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(current);
    GE_ASSERT_NOTNULL(asc_node);
    chain.nodes.push_back(current);
    chain.output_dtypes.push_back(asc_node->outputs[0].attr.dtype);
    current = consumers[0];
  }
  return af::SUCCESS;
}

void UpdateMinimumCastPath(const std::vector<DataType> &output_dtypes, size_t target, CastChainPlan &plan) {
  const auto unreachable = output_dtypes.size() + 1U;
  for (size_t source = 0U; source < target; ++source) {
    if (plan.costs[source] == unreachable || !CanCollapseCast(output_dtypes[source], output_dtypes[target])) {
      continue;
    }
    const auto transition_cost = output_dtypes[source] == output_dtypes[target] ? 0U : 1U;
    if (plan.costs[source] + transition_cost < plan.costs[target]) {
      plan.costs[target] = plan.costs[source] + transition_cost;
      plan.previous[target] = source;
      plan.requires_cast[target] = transition_cost == 1U;
    }
  }
}

CastChainPlan BuildMinimumCastPlan(DataType source_dtype, const std::vector<DataType> &output_dtypes) {
  const auto chain_size = output_dtypes.size();
  CastChainPlan plan{std::vector<size_t>(chain_size, chain_size + 1U), std::vector<size_t>(chain_size, chain_size),
                     std::vector<bool>(chain_size, false)};
  for (size_t target = 0U; target < chain_size; ++target) {
    if (CanCollapseCast(source_dtype, output_dtypes[target])) {
      plan.requires_cast[target] = source_dtype != output_dtypes[target];
      plan.costs[target] = plan.requires_cast[target] ? 1U : 0U;
    }
    UpdateMinimumCastPath(output_dtypes, target, plan);
  }
  return plan;
}

std::vector<bool> MarkRetainedCastNodes(const CastChainPlan &plan) {
  const auto chain_size = plan.previous.size();
  std::vector<bool> keep(chain_size, false);
  for (size_t index = chain_size - 1U; index < chain_size; index = plan.previous[index]) {
    keep[index] = plan.requires_cast[index];
    if (plan.previous[index] >= chain_size) {
      break;
    }
  }
  return keep;
}

Status DeleteUnretainedCastNodes(AscGraph &asc_graph, const CastChain &chain, const std::vector<bool> &keep,
                                 std::vector<NodePtr> &retained_nodes) {
  retained_nodes.clear();
  for (size_t i = 0U; i < chain.nodes.size(); ++i) {
    if (keep[i]) {
      retained_nodes.push_back(chain.nodes[i]);
    }
  }
  for (auto i = chain.nodes.size(); i > 0U; --i) {
    if (!keep[i - 1U]) {
      GE_ASSERT_SUCCESS(DelNode(asc_graph, chain.nodes[i - 1U]));
    }
  }
  return af::SUCCESS;
}

Status OptimizeLinearCastChain(AscGraph &asc_graph, const NodePtr &source, const CastChain &chain,
                               std::vector<NodePtr> &retained_nodes) {
  retained_nodes.clear();
  GE_ASSERT_TRUE(chain.nodes.size() > 1U);
  DataType source_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(source, source_dtype));
  if (source_dtype == chain.output_dtypes.back()) {
    for (auto i = chain.nodes.size(); i > 0U; --i) {
      GE_ASSERT_SUCCESS(DelNode(asc_graph, chain.nodes[i - 1U]));
    }
    return af::SUCCESS;
  }
  const auto plan = BuildMinimumCastPlan(source_dtype, chain.output_dtypes);
  if (plan.costs.back() == chain.nodes.size() + 1U) {
    retained_nodes = chain.nodes;
    return af::SUCCESS;
  }
  return DeleteUnretainedCastNodes(asc_graph, chain, MarkRetainedCastNodes(plan), retained_nodes);
}

Status CanOptimizeLinearCastChain(const NodePtr &source, const CastChain &chain, bool &can_optimize) {
  can_optimize = false;
  if (chain.nodes.size() <= 1U) {
    return af::SUCCESS;
  }
  DataType source_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(source, source_dtype));
  if (!IsFloatDataType(source_dtype)) {
    return af::SUCCESS;
  }
  can_optimize = std::all_of(chain.output_dtypes.begin(), chain.output_dtypes.end(),
                             [](const auto dtype) { return IsFloatDataType(dtype); });
  return af::SUCCESS;
}

Status DeleteIdentityCast(AscGraph &asc_graph, const NodePtr &source, const NodePtr &cast_node, bool &deleted) {
  DataType source_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(source, source_dtype));
  DataType cast_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(cast_node, cast_dtype));
  deleted = source_dtype == cast_dtype;
  if (deleted) {
    GE_ASSERT_SUCCESS(DelNode(asc_graph, cast_node));
  }
  return af::SUCCESS;
}

bool ShouldChangeDataType(const NodePtr &node, const std::vector<NodePtr> &peer_in_nodes, DataType peer_output_dtype,
                          DataType output_dtype) {
  (void)node;
  if (IsNodeTypeInPeerInNodes(af::ascir_op::Store::Type, peer_in_nodes)) {
    return false;
  }
  if (IsUltraLowToLowPrecision(peer_output_dtype, output_dtype) && !CheckCastDtype(peer_output_dtype, DT_FLOAT)) {
    return false;
  }
  return (output_dtype == DT_FLOAT16 || output_dtype == DT_BF16);
}

bool IsFloatToUltraLowNeedInsertCast(const NodePtr &peer_out_node, DataType peer_output_dtype, DataType output_dtype) {
  if (!IsFloatToUltraLowPrecision(peer_output_dtype, output_dtype)) {
    return false;
  }
  if (CheckCastDtype(DT_FLOAT, output_dtype)) {
    return false;
  }
  const auto &type = peer_out_node->GetType();
  if ((type == af::ascir_op::Load::Type || type == af::ascir_op::Gather::Type || type == af::ascir_op::Cast::Type) &&
      IsLowPrecisionDataType(peer_output_dtype)) {
    return false;
  }
  return true;
}

NodePtr BuildCastNode(AscGraph &asc_graph, const NodePtr &ref_node) {
  OpDescBuilder builder("Cast_" + ref_node->GetName() + "_" + std::to_string(GenUniqueNumber()),
                        af::ascir_op::Cast::Type);
  builder.AddInput("x");
  builder.AddOutput("y");
  auto cast_op_desc = builder.Build();
  GE_ASSERT_NOTNULL(cast_op_desc);
  cast_op_desc->AppendIrInput("x", af::kIrInputRequired);
  cast_op_desc->AppendIrOutput("y", af::kIrOutputRequired);
  auto op = std::make_shared<af::Operator>(OpDescUtils::CreateOperatorFromOpDesc(cast_op_desc));
  GE_ASSERT_NOTNULL(op);
  return asc_graph.AddNode(*op);
}

Status WireCastBeforeInput(AscGraph &asc_graph, const NodePtr &target, NodePtr &cast_node, int32_t input_idx) {
  cast_node = BuildCastNode(asc_graph, target);
  GE_ASSERT_NOTNULL(cast_node);
  GE_ASSERT_SUCCESS(cast_node->SetOwnerComputeGraph(AscGraphUtils::GetComputeGraph(asc_graph)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(cast_node, target, {input_idx}, {}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(cast_node->GetOutDataAnchor(0), target->GetInDataAnchor(input_idx)));
  return af::SUCCESS;
}

Status TransferNodeAttrs(const NodePtr &src_node, const NodePtr &dst_node) {
  GE_ASSERT_NOTNULL(src_node->GetOpDesc());
  const auto src_attr = src_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(src_attr);
  GE_ASSERT_NOTNULL(dst_node->GetOpDesc());
  auto dst_attr = dst_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(dst_attr);
  dst_attr->sched.axis = src_attr->sched.axis;
  dst_node->GetOpDesc()->SetId(src_node->GetOpDesc()->GetId());
  src_node->GetOpDesc()->SetId(src_node->GetOpDesc()->GetId() + 1);
  return af::SUCCESS;
}

Status ConfigureCastTensor(const NodePtr &src_node, const NodePtr &cast_node, const NodePtr &next_node,
                           bool is_increase) {
  // Determine output dtype first
  const auto c_opdesc = cast_node->GetOpDesc();
  GE_ASSERT_NOTNULL(c_opdesc);
  auto c_out_desc = c_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(c_out_desc);
  if (is_increase) {
    c_out_desc->SetDataType(DT_FLOAT);
  } else {
    const auto next_desc = next_node->GetOpDesc();
    GE_ASSERT_NOTNULL(next_desc);
    auto next_out = next_desc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(next_out);
    c_out_desc->SetDataType(IsLowPrecisionDataType(next_out->GetDataType()) ? next_out->GetDataType() : DT_FLOAT16);
  }
  // Copy tensor attrs from source
  auto c_o_attr = c_out_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(c_o_attr);
  // 当上游节点为 Scalar 时，其输出为标量形状，axis/repeats/strides 不包含目标张量的完整形状信息。
  // 前因：前端可能传入冗余连续 Cast（如 scalar->cast(FP32->FP32)->cast(FP32->FP16)->store），
  // ImprovePrecision 会删除冗余浮点 Cast，再在 Store 前重新插入新 Cast。
  // 若从 Scalar 复制输出信息，新 Cast 的 axis/repeats/strides 为空，
  // 导致后续 InsertBroadcast 补充的 broadcast 节点输出信息也丢失。
  // 因此当上游为 Scalar 时，从下游节点取 tensor 信息。
  GeTensorDescPtr src_tensor_desc;
  if (src_node->GetType() == af::ascir_op::Scalar::Type) {
    GE_ASSERT_SUCCESS(GetOutputTensorDesc(next_node, src_tensor_desc));
  } else {
    GE_ASSERT_SUCCESS(GetOutputTensorDesc(src_node, src_tensor_desc));
  }
  GE_ASSERT_NOTNULL(src_tensor_desc);
  const auto src_attr = src_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(src_attr);
  c_o_attr->axis = src_attr->axis;
  c_o_attr->repeats = src_attr->repeats;
  c_o_attr->strides = src_attr->strides;
  return af::SUCCESS;
}

Status ProcessCastNodePrecision(AscGraph &asc_graph, const NodePtr &node) {
  NodePtr peer_out_node;
  GE_ASSERT_SUCCESS(GetPeerOutNode(node, peer_out_node, 0));
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(GetPeerInNodes(node, peer_in_nodes, 0));

  GeTensorDescPtr output_tensor_desc;
  GE_ASSERT_SUCCESS(GetOutputTensorDesc(node, output_tensor_desc));
  DataType peer_output_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(peer_out_node, peer_output_dtype));
  const auto output_dtype = output_tensor_desc->GetDataType();
  if (IsFloatToUltraLowNeedInsertCast(peer_out_node, peer_output_dtype, output_dtype)) {
    GE_ASSERT_SUCCESS(UpdateTopoId(asc_graph, node, 1));
    NodePtr c_node = nullptr;
    GE_ASSERT_SUCCESS(WireCastBeforeInput(asc_graph, node, c_node, 0));
    NodePtr peer_out_of_cast;
    GE_ASSERT_SUCCESS(GetPeerOutNode(c_node, peer_out_of_cast, 0));
    GE_ASSERT_SUCCESS(ConfigureCastTensor(peer_out_of_cast, c_node, node, false));
    GE_ASSERT_SUCCESS(TransferNodeAttrs(node, c_node));
    return af::SUCCESS;
  }

  if (ShouldChangeDataType(node, peer_in_nodes, peer_output_dtype, output_dtype)) {
    output_tensor_desc->SetDataType(DT_FLOAT);
    return af::SUCCESS;
  }

  return af::SUCCESS;
}

Status ProcessCastChains(AscGraph &asc_graph, const std::vector<NodePtr> &nodes) {
  std::unordered_set<NodePtr> processed;
  std::vector<NodePtr> retained_nodes;
  for (const auto &node : nodes) {
    if (processed.find(node) != processed.end()) {
      continue;
    }
    NodePtr source;
    GE_ASSERT_SUCCESS(GetPeerOutNode(node, source, 0));
    CastChain chain;
    GE_ASSERT_SUCCESS(CollectLinearCastChain(node, chain));
    bool can_optimize = false;
    GE_ASSERT_SUCCESS(CanOptimizeLinearCastChain(source, chain, can_optimize));
    if (!can_optimize) {
      processed.insert(node);
      bool deleted = false;
      GE_ASSERT_SUCCESS(DeleteIdentityCast(asc_graph, source, node, deleted));
      if (!deleted) {
        retained_nodes.push_back(node);
      }
      continue;
    }
    std::vector<NodePtr> chain_retained_nodes;
    GE_ASSERT_SUCCESS(OptimizeLinearCastChain(asc_graph, source, chain, chain_retained_nodes));
    retained_nodes.insert(retained_nodes.end(), chain_retained_nodes.begin(), chain_retained_nodes.end());
    for (const auto &cast_node : chain.nodes) {
      processed.insert(cast_node);
    }
  }
  for (const auto &node : retained_nodes) {
    GE_ASSERT_SUCCESS(ProcessCastNodePrecision(asc_graph, node));
  }
  return af::SUCCESS;
}

Status IsNeedInsertCastAfterLoad(const NodePtr &node, bool &is_need_insert_cast) {
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(GetPeerInNodes(node, peer_in_nodes, 0));
  if (IsNodeTypeInPeerInNodes(af::ascir_op::Cast::Type, peer_in_nodes) ||
      IsNodeTypeInPeerInNodes(af::ascir_op::Store::Type, peer_in_nodes) ||
      (IsNodeTypeInPeerInNodes(af::ascir_op::Transpose::Type, peer_in_nodes) &&
       IsTransposeSupportedDtype(output_tensor_desc->GetDataType())) ||
      !(output_tensor_desc->GetDataType() == DT_FLOAT16 || output_tensor_desc->GetDataType() == DT_BF16)) {
    return af::SUCCESS;
  }
  is_need_insert_cast = true;
  return af::SUCCESS;
}

Status InsertCastToIncreasePrecision(AscGraph &asc_graph, const NodePtr &load_node, bool is_need_insert_cast) {
  if (!is_need_insert_cast) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(UpdateTopoId(asc_graph, load_node, 1));
  auto c_node = BuildCastNode(asc_graph, load_node);
  GE_ASSERT_NOTNULL(c_node);
  const auto load_out_anchor = load_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(load_out_anchor);
  GE_ASSERT_GRAPH_SUCCESS(AscGraphUtils::InsertNodeAfter(load_out_anchor, c_node));
  const auto c_opdesc = c_node->GetOpDesc();
  GE_ASSERT_NOTNULL(c_opdesc);
  const auto c_output_tensor_desc = c_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(c_output_tensor_desc);
  c_output_tensor_desc->SetDataType(DT_FLOAT);
  const auto load_opdesc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_opdesc);
  c_opdesc->SetId(load_opdesc->GetId() + 1);
  return af::SUCCESS;
}

Status IsNeedInsertCastBeforeOther(const NodePtr &other_node, bool &need_insert, std::vector<int32_t> &input_idxs) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(other_node, peer_out_nodes));
  for (auto idx = 0U; idx < peer_out_nodes.size(); idx++) {
    const auto &peer_out_node = peer_out_nodes[idx];
    DataType peer_dtype;
    GE_ASSERT_SUCCESS(GetNodeOutputDtype(peer_out_node, peer_dtype));
    const auto &type = peer_out_node->GetType();
    if (type == af::ascir_op::Cast::Type || type == af::ascir_op::Load::Type || type == af::ascir_op::Gather::Type ||
        (type == af::ascir_op::Transpose::Type && IsTransposeSupportedDtype(peer_dtype))) {
      if (IsLowPrecisionDataType(peer_dtype)) {
        need_insert = true;
        input_idxs.push_back(static_cast<int32_t>(idx));
      }
    }
  }
  return af::SUCCESS;
}

Status InsertCastBeforeNode(AscGraph &asc_graph, const NodePtr &other_node, bool is_need_insert_cast,
                            bool is_increase_precision, const std::vector<int32_t> &input_idxs) {
  if (!is_need_insert_cast) {
    return af::SUCCESS;
  }
  for (auto input_idx : input_idxs) {
    GE_ASSERT_SUCCESS(UpdateTopoId(asc_graph, other_node, 1));
    NodePtr c_node = nullptr;
    GE_ASSERT_SUCCESS(WireCastBeforeInput(asc_graph, other_node, c_node, input_idx));
    NodePtr peer_out_node;
    GE_ASSERT_SUCCESS(GetPeerOutNode(c_node, peer_out_node, 0));
    GE_ASSERT_SUCCESS(ConfigureCastTensor(peer_out_node, c_node, other_node, is_increase_precision));
    GE_ASSERT_SUCCESS(TransferNodeAttrs(other_node, c_node));
  }
  return af::SUCCESS;
}

Status IsNeedInsertCastBeforeStore(const NodePtr &store_node, bool &need_insert, bool &is_increase_precision) {
  need_insert = false;
  const auto in_anchor = store_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(in_anchor);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  const auto &peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node);
  const auto peer_op_desc = peer_out_node->GetOpDesc();
  GE_ASSERT_NOTNULL(peer_op_desc);
  // 多输出算子（如 Frexp）必须取 Store 实际连接的输出 anchor 的 dtype，不能固定取输出 0
  const auto peer_dtype = peer_op_desc->GetOutputDesc(static_cast<uint32_t>(peer_out_anchor->GetIdx())).GetDataType();
  DataType store_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(store_node, store_dtype));
  is_increase_precision = IsHighPrecisionDataType(store_dtype);
  if (peer_dtype == store_dtype) {
    return af::SUCCESS;
  }
  need_insert = true;
  return af::SUCCESS;
}

Status TryInsertCastBeforeTransposeForStore(AscGraph &asc_graph, const NodePtr &store_node,
                                            const NodePtr &transpose_node, DataType target_dtype, bool &inserted) {
  inserted = false;
  std::vector<NodePtr> consumers;
  GE_ASSERT_SUCCESS(GetPeerInNodes(transpose_node, consumers, 0));
  for (const auto &consumer : consumers) {
    DataType consumer_dtype;
    GE_ASSERT_SUCCESS(GetNodeOutputDtype(consumer, consumer_dtype));
    if (consumer->GetType() != af::ascir_op::Store::Type || consumer_dtype != target_dtype) {
      return af::SUCCESS;
    }
  }
  NodePtr transpose_input;
  GE_ASSERT_SUCCESS(GetPeerOutNode(transpose_node, transpose_input, 0));
  DataType transpose_input_dtype;
  GE_ASSERT_SUCCESS(GetNodeOutputDtype(transpose_input, transpose_input_dtype));
  if (!IsHighPrecisionDataType(transpose_input_dtype)) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(InsertCastBeforeNode(asc_graph, transpose_node, true, false, {0}));
  NodePtr cast_node;
  GE_ASSERT_SUCCESS(GetPeerOutNode(transpose_node, cast_node, 0));
  GE_ASSERT_SUCCESS(SetNodeOutputDtype(cast_node, target_dtype));
  GE_ASSERT_SUCCESS(SetNodeOutputDtype(transpose_node, target_dtype));
  inserted = true;
  GELOGD("Moved downcast before Transpose[%s] for Store[%s].", transpose_node->GetNamePtr(), store_node->GetNamePtr());
  return af::SUCCESS;
}
using TypeToNodesMap = std::unordered_map<std::string, std::vector<NodePtr>>;

bool ShouldSkipGraph(optimize::GraphPropertiesCache &cache, const AscGraph &asc_graph) {
  if (cache.HasCube() || cache.HasConcat() || cache.HasSplit()) {
    GELOGI("Graph %s is cube/concat/split type, skip precision improvement.", asc_graph.GetName().c_str());
    return true;
  }
  return false;
}

Status IsAllNodesInBlacklist(const AscGraph &asc_graph, bool &result) {
  const auto &blacklist2 = PreProcessConfig::Instance().GetImprovePrecisionBlacklist();
  const bool has_all = (blacklist2.find(PreProcessConfig::kAllNodeType) != blacklist2.end());
  result = true;
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (node->GetType() == af::ascir_op::Output::Type || node->GetType() == af::ascir_op::Data::Type) {
      continue;
    }
    if (has_all) {
      if (CheckNodeDtype(node) != af::SUCCESS) {
        result = false;
        return af::SUCCESS;
      }
    } else if (IsInBlackList1(node)) {
      continue;
    } else if (IsInBlackList2(node, blacklist2)) {
      GE_ASSERT_SUCCESS(CheckNodeDtype(node));
    } else {
      result = false;
      return af::SUCCESS;
    }
  }
  GELOGI("All nodes in graph %s are in the blacklist, skip precision improvement.", asc_graph.GetName().c_str());
  return af::SUCCESS;
}

TypeToNodesMap GroupNodesByType(const AscGraph &asc_graph) {
  TypeToNodesMap type_to_nodes;
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (node->GetType() == af::ascir_op::Output::Type || node->GetType() == af::ascir_op::Data::Type) {
      continue;
    }
    auto it = kTypeToGroup.find(node->GetType());
    const std::string &group_name = (it != kTypeToGroup.end()) ? it->second : "Other";
    type_to_nodes[group_name].push_back(node);
  }
  return type_to_nodes;
}

Status ProcessLoadGatherNodes(AscGraph &asc_graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    bool is_need = false;
    GE_ASSERT_SUCCESS(IsNeedInsertCastAfterLoad(node, is_need));
    GE_ASSERT_SUCCESS(InsertCastToIncreasePrecision(asc_graph, node, is_need));
  }
  return af::SUCCESS;
}

Status ProcessOtherComputeNodes(AscGraph &asc_graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    if (node->GetType() == af::ascir_op::Transpose::Type) {
      DataType node_dtype;
      GE_ASSERT_SUCCESS(GetNodeOutputDtype(node, node_dtype));
      if (IsTransposeSupportedDtype(node_dtype)) {
        continue;
      }
    }
    bool is_need = false;
    std::vector<int32_t> input_idxs;
    GE_ASSERT_SUCCESS(IsNeedInsertCastBeforeOther(node, is_need, input_idxs));
    GE_ASSERT_SUCCESS(InsertCastBeforeNode(asc_graph, node, is_need, true, input_idxs));
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(node, DT_BF16, DT_FLOAT));
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(node, DT_FLOAT16, DT_FLOAT));
  }
  return af::SUCCESS;
}

Status ProcessStoreNodes(AscGraph &asc_graph, const std::vector<NodePtr> &nodes) {
  for (const auto &node : nodes) {
    bool is_need = false;
    bool is_increase = true;
    GE_ASSERT_SUCCESS(IsNeedInsertCastBeforeStore(node, is_need, is_increase));
    NodePtr peer_out_node;
    GE_ASSERT_SUCCESS(GetPeerOutNode(node, peer_out_node, 0));
    const auto store_dtype = node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (IsLowPrecisionDataType(store_dtype) && IsTransposeSupportedDtype(store_dtype) &&
        peer_out_node->GetType() == af::ascir_op::Transpose::Type) {
      bool inserted = false;
      GE_ASSERT_SUCCESS(TryInsertCastBeforeTransposeForStore(asc_graph, node, peer_out_node, store_dtype, inserted));
      if (inserted) {
        continue;
      }
    }
    GE_ASSERT_SUCCESS(InsertCastBeforeNode(asc_graph, node, is_need, is_increase, {0}));
  }
  return af::SUCCESS;
}

Status ProcessNodeGroups(AscGraph &asc_graph, TypeToNodesMap &type_to_nodes) {
  GE_ASSERT_SUCCESS(ProcessCastChains(asc_graph, type_to_nodes[af::ascir_op::Cast::Type]));
  GE_ASSERT_SUCCESS(ProcessLoadGatherNodes(asc_graph, type_to_nodes[af::ascir_op::Load::Type]));
  GE_ASSERT_SUCCESS(ProcessLoadGatherNodes(asc_graph, type_to_nodes[af::ascir_op::Gather::Type]));
  for (const auto &node : type_to_nodes[af::ascir_op::Scalar::Type]) {
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(node, DT_BF16, DT_FLOAT));
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(node, DT_FLOAT16, DT_FLOAT));
  }
  GE_ASSERT_SUCCESS(ProcessOtherComputeNodes(asc_graph, type_to_nodes["Other"]));
  GE_ASSERT_SUCCESS(ProcessStoreNodes(asc_graph, type_to_nodes[af::ascir_op::Store::Type]));
  return af::SUCCESS;
}
}  // namespace

af::Status ImprovePrecisionForAscGraph(AscGraph &asc_graph) {
  optimize::GraphPropertiesCache cache(asc_graph);
  if (ShouldSkipGraph(cache, asc_graph)) {
    return af::SUCCESS;
  }
  bool all_in_blacklist = false;
  GE_ASSERT_SUCCESS(IsAllNodesInBlacklist(asc_graph, all_in_blacklist));
  if (all_in_blacklist) {
    return af::SUCCESS;
  }
  auto type_to_nodes = GroupNodesByType(asc_graph);
  GE_ASSERT_SUCCESS(ProcessNodeGroups(asc_graph, type_to_nodes));
  TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return af::SUCCESS;
}
}  // namespace af::pre_process
