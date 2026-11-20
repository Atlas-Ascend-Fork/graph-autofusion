/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include <google/protobuf/text_format.h>
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/ascendc_ir/core/ascendc_ir_impl.h"
#include "proto/ascendc_ir.pb.h"

namespace af {
namespace {
void RestoreInputDtypeSymbol(OpDesc &op_desc, const ascir::AscIrDef &asc_ir_def, const std::string &input_name,
                             IrInputType input_type) {
  const auto &symbol_name = asc_ir_def.GetInputIrSymName(input_name);
  if (symbol_name.empty()) {
    return;
  }
  op_desc.SetInputDtypeSymbol(input_name, input_type, symbol_name);
  const auto &soc_to_sym_store = asc_ir_def.GetSocToDataTypeSymbolStore();
  const auto &symbol_store =
      soc_to_sym_store.empty() ? asc_ir_def.GetDataTypeSymbolStore() : soc_to_sym_store.begin()->second;
  const auto &symbols = symbol_store.GetNamedSymbols();
  const auto symbol = symbols.find(symbol_name);
  if (symbol != symbols.end() && symbol->second != nullptr) {
    op_desc.DeclareDtypeSymbol(symbol_name, symbol->second->GetTensorType());
  }
}

graphStatus AddOpInputs(const ascendc_ir::proto::IrDef &ir_attr, const std::string &op_name,
                        OpDescBuilder &op_builder) {
  const auto &inputs_nums = ir_attr.input_nums();
  const auto &inputs_names = ir_attr.input_names();
  const auto &input_types = ir_attr.input_ir_type();
  GE_ASSERT_TRUE(inputs_nums.size() == input_types.size(),
                 "[Build][Op] for %s failed, inputs_nums[%zu] "
                 "input_types[%zu] not equal.",
                 op_name.c_str(), inputs_nums.size(), input_types.size());
  GE_ASSERT_TRUE(inputs_nums.size() == inputs_names.size(),
                 "[Build][Op] for %s failed, inputs_nums[%zu] "
                 "inputs_names[%zu] not equal.",
                 op_name.c_str(), inputs_nums.size(), inputs_names.size());
  for (int32_t ir_id = 0; ir_id < inputs_nums.size(); ir_id++) {
    if (input_types[ir_id] == IrInputType::kIrInputDynamic) {
      op_builder.AddDynamicInput(ir_attr.input_names(ir_id), inputs_nums[ir_id]);
      GELOGD("Add dynamic input[%s] of node[%s] success, input num[%ld].", inputs_names[ir_id].c_str(), op_name.c_str(),
             inputs_nums[ir_id]);
    } else if (input_types[ir_id] == IrInputType::kIrInputOptional && inputs_nums[ir_id] == 0) {
      op_builder.AddInput(inputs_names[ir_id], GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
      GELOGD("Add disconnected optional input[%s] of node[%s].", inputs_names[ir_id].c_str(), op_name.c_str());
    } else {
      op_builder.AddInput(inputs_names[ir_id]);
      GELOGD("Add input[%s] of node[%s] success.", inputs_names[ir_id].c_str(), op_name.c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AddOpOutputs(const ascendc_ir::proto::IrDef &ir_attr, const std::string &op_name,
                         OpDescBuilder &op_builder) {
  const auto &outputs_nums = ir_attr.output_nums();
  const auto &outputs_names = ir_attr.output_names();
  const auto &output_types = ir_attr.output_ir_type();
  GE_ASSERT_TRUE(outputs_nums.size() == output_types.size(),
                 "[Build][Op] for %s failed, outputs_nums[%zu] "
                 "output_types[%zu] not equal.",
                 op_name.c_str(), outputs_nums.size(), output_types.size());
  GE_ASSERT_TRUE(outputs_nums.size() == outputs_names.size(),
                 "[Build][Op] for %s failed, outputs_nums[%zu] "
                 "outputs_names[%zu] not equal.",
                 op_name.c_str(), outputs_nums.size(), outputs_names.size());
  for (int32_t ir_id = 0; ir_id < outputs_nums.size(); ir_id++) {
    if (output_types[ir_id] == IrOutputType::kIrOutputDynamic) {
      op_builder.AddDynamicOutput(ir_attr.output_names(ir_id), outputs_nums[ir_id]);
      GELOGD("Add dynamic output[%s] of node[%s] success, output num[%ld].", outputs_names[ir_id].c_str(),
             op_name.c_str(), outputs_nums[ir_id]);
    } else {
      op_builder.AddOutput(outputs_names[ir_id]);
      GELOGD("Add input[%s] of node[%s] success.", outputs_names[ir_id].c_str(), op_name.c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AddAscNode(const ascendc_ir::proto::AscNodeDef &asc_node, AscGraph &out_asc_graph) {
  auto &node_attr = asc_node.attr();
  auto &ir_attr = asc_node.ir_def();
  auto &op_name = node_attr.name();
  auto &op_type = ir_attr.type();
  OpDescBuilder op_builder(op_name, op_type);
  GE_ASSERT_GRAPH_SUCCESS(AddOpInputs(ir_attr, op_name, op_builder));
  GE_ASSERT_GRAPH_SUCCESS(AddOpOutputs(ir_attr, op_name, op_builder));

  auto op_desc = op_builder.Build();
  GE_ASSERT_NOTNULL(op_desc, "[Build][Op] for %s failed.", op_name.c_str());
  auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  out_asc_graph.AddNode(op);
  GELOGD("Add node[%s:%s] inputs size[%d] output size[%d] success.", op_name.c_str(), op_type.c_str(),
         ir_attr.input_names().size(), ir_attr.output_names().size());
  return GRAPH_SUCCESS;
}

int32_t GetInputAnchorIndex(const ascendc_ir::proto::IrDef &ir_attr, const int32_t connected_input_index) {
  int32_t anchor_index = 0;
  int32_t connected_index = 0;
  const auto &input_nums = ir_attr.input_nums();
  const auto &input_types = ir_attr.input_ir_type();
  for (int32_t ir_index = 0; ir_index < input_nums.size(); ++ir_index) {
    const int32_t input_num = input_nums[ir_index];
    if (input_types[ir_index] == IrInputType::kIrInputDynamic) {
      for (int32_t dynamic_index = 0; dynamic_index < input_num; ++dynamic_index) {
        if (connected_index == connected_input_index) {
          return anchor_index;
        }
        ++connected_index;
        ++anchor_index;
      }
      continue;
    }

    if (input_num != 0) {
      if (connected_index == connected_input_index) {
        return anchor_index;
      }
      ++connected_index;
    }
    ++anchor_index;
  }
  return -1;
}

graphStatus AddAscEdges(const ascendc_ir::proto::AscNodeDef &asc_node, const ComputeGraphPtr &compute_graph) {
  auto &node_attr = asc_node.attr();
  const auto &ir_attr = asc_node.ir_def();
  auto &dst_node_name = node_attr.name();
  int32_t connected_input_index = 0;
  for (const auto &input : asc_node.input_src()) {
    const int32_t dst_in_index = GetInputAnchorIndex(ir_attr, connected_input_index);
    GE_ASSERT_TRUE(dst_in_index >= 0, "[Get][InputAnchor] failed, node[%s], input[%d].", dst_node_name.c_str(),
                   connected_input_index);
    const auto &src_node_name = input.src_node_name();
    const auto &src_out_index = input.src_out_index();
    if (src_node_name.empty()) {
      GELOGW("[Get][SrcNodeName] failed of node [%s:%d]", node_attr.name().c_str(), dst_in_index);
      ++connected_input_index;
      continue;
    }
    const auto &src_node = compute_graph->FindNode(src_node_name);
    GE_ASSERT_NOTNULL(src_node, "[Find][SrcNode] %s failed, dst_node[%s].", src_node_name.c_str(),
                      node_attr.name().c_str());
    const auto &dst_node = compute_graph->FindNode(dst_node_name);
    GE_ASSERT_NOTNULL(dst_node, "[Find][DstNode] %s failed, dst_node[%s].", src_node_name.c_str(),
                      node_attr.name().c_str());
    GE_ASSERT_GRAPH_SUCCESS(
        GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_out_index), dst_node->GetInDataAnchor(dst_in_index)),
        "[Add][Edge] failed to add edge from node[%s:%d] to node[%s:%d], possible duplicate link.",
        src_node_name.c_str(), src_out_index, dst_node_name.c_str(), dst_in_index);
    GELOGD("[Add][Edge] from node[%s:%d] to node[%s:%d].", src_node_name.c_str(), src_out_index, dst_node_name.c_str(),
           dst_in_index);
    ++connected_input_index;
  }
  return GRAPH_SUCCESS;
}

graphStatus InheritAutofuseAttr(GeTensorDesc &src_tensor_desc, GeTensorDesc &insert_tensor_desc) {
  auto src_tensor_attr = src_tensor_desc.GetOrCreateAttrsGroup<AscTensorAttr>();
  auto insert_tensor_attr = insert_tensor_desc.GetOrCreateAttrsGroup<AscTensorAttr>();
  insert_tensor_attr->axis = src_tensor_attr->axis;
  insert_tensor_attr->repeats = src_tensor_attr->repeats;
  insert_tensor_attr->strides = src_tensor_attr->strides;
  return GRAPH_SUCCESS;
}

graphStatus EstablishAscNodeAndEdges(const ascendc_ir::proto::AscGraphDef &asc_graph_def, AscGraph &out_asc_graph) {
  auto &asc_nodes = asc_graph_def.asc_node();
  // 1. Add AscNodes to AscGraph
  for (const auto &asc_node : asc_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(AddAscNode(asc_node, out_asc_graph));
  }
  const auto &compute_graph = AscGraphUtils::GetComputeGraph(out_asc_graph);
  // 2. Add edge between all AscNodes
  for (const auto &asc_node : asc_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(AddAscEdges(asc_node, compute_graph));
  }
  GELOGD("Deserialize graph[%s] success, graph node size[%zu]", out_asc_graph.GetName().c_str(),
         compute_graph->GetDirectNodesSize());
  return GRAPH_SUCCESS;
}

graphStatus SerializeNodeInputs(const AscNodePtr &node, ascendc_ir::proto::AscNodeDef &node_def,
                                ascendc_ir::proto::IrDef &ir_def) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_index_map = OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  size_t ir_id = 0UL;
  size_t cur_input_id = 0UL;
  for (const auto &ir_input : op_desc->GetIrInputs()) {
    size_t start = cur_input_id;
    size_t end = cur_input_id;
    if (ir_input.second == IrInputType::kIrInputDynamic) {
      const auto &range_iter = ir_index_map.find(ir_id);
      if (range_iter != ir_index_map.cend()) {
        start = range_iter->second.first;
        end = range_iter->second.second;
      }
    } else {
      end = start + 1;
      if (ir_input.second == IrInputType::kIrInputOptional) {
        const auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(start));
        if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
          end = start;
        }
      }
    }
    GE_ASSERT_TRUE(start <= end);
    GE_ASSERT_TRUE(end <= op_desc->GetAllInputsSize());
    int64_t input_nums = static_cast<int64_t>(end - start);
    ir_def.add_input_nums(input_nums);
    GELOGD("Add input nums[%d, end:%zu, start:%zu] for node[%s:%s, %d]", input_nums, end, start, op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), ir_id);
    for (auto id = static_cast<int32_t>(start); id < static_cast<int32_t>(end); id++) {
      const auto input_src = node_def.add_input_src();
      const auto &in_anchor = node->GetInDataAnchor(id);
      GE_ASSERT_NOTNULL(in_anchor, "[Get][InDataAnchor] failed, node[%s]", node->GetNamePtr());
      const auto &peer_out = in_anchor->GetPeerOutAnchor();
      if (peer_out == nullptr) {
        GELOGW("[Get][PeerOut] failed of node [%s:%d]", node->GetNamePtr(), in_anchor->GetIdx());
        continue;
      }
      const auto &src_node = peer_out->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(src_node, "[Get][SrcNode] failed, node[%s]", node->GetNamePtr());
      input_src->set_src_node_name(src_node->GetName());
      input_src->set_src_out_index(peer_out->GetIdx());
      GELOGD("Set src node name[%s:%d] for node[%s:%s, %d]", src_node->GetName().c_str(), peer_out->GetIdx(),
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), id);
    }
    ir_id++;
    cur_input_id = ir_input.second == IrInputType::kIrInputDynamic ? end : start + 1UL;
  }
  return GRAPH_SUCCESS;
}

graphStatus SerializeNodeOutputs(const OpDescPtr &op_desc, ascendc_ir::proto::IrDef &ir_def) {
  const auto &ir_index_map_output = OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(op_desc);
  size_t output_ir_id = 0UL;
  size_t cur_output_id = 0UL;
  for (const auto &ir_output : op_desc->GetIrOutputs()) {
    size_t start = cur_output_id;
    size_t end = cur_output_id;
    if (ir_output.second == IrOutputType::kIrOutputDynamic) {
      const auto &range_iter = ir_index_map_output.find(output_ir_id);
      if (range_iter != ir_index_map_output.cend()) {
        start = range_iter->second.first;
        end = range_iter->second.second;
      }
    } else {
      end = start + 1;
    }
    GE_ASSERT_TRUE(start <= end);
    GE_ASSERT_TRUE(end <= op_desc->GetOutputsSize());
    int64_t output_nums = static_cast<int64_t>(end - start);
    ir_def.add_output_nums(output_nums);
    GELOGD("Add output nums[%d, end:%zu, start:%zu] for node[%s:%s, %d]", output_nums, end, start,
           op_desc->GetNamePtr(), op_desc->GetTypePtr(), output_ir_id);
    output_ir_id++;
    cur_output_id = end;
  }
  return GRAPH_SUCCESS;
}

graphStatus SerializeNodeOutputTensors(const OpDescPtr &op_desc, ascendc_ir::proto::AscNodeDef &node_def) {
  for (const auto &tensor : op_desc->GetAllOutputsDescPtr()) {
    const auto tensor_attr = tensor->GetOrCreateAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr, "[GetOrCreate][AscTensorAttr] failed, node[%s]", op_desc->GetNamePtr());
    const auto tensor_def = node_def.add_outputs();
    const auto tensor_attr_def = tensor_def->mutable_attr();
    GE_ASSERT_NOTNULL(tensor_attr_def, "[Get][TensorAttr] failed, node[%s]", op_desc->GetNamePtr());
    tensor_attr->SerializeAttr(*tensor_attr_def);
  }
  return GRAPH_SUCCESS;
}

graphStatus SerializeNode(const AscGraph &asc_graph, const AscNodePtr &node,
                          ascendc_ir::proto::AscGraphDef &asc_graph_def) {
  auto node_def = asc_graph_def.add_asc_node();
  GE_ASSERT_NOTNULL(node_def, "[Add][AscNode] proto failed, graph[%s]", asc_graph.GetName().c_str());
  const auto attr_def = node_def->mutable_attr();
  GE_ASSERT_NOTNULL(attr_def, "[Get][Attr] failed, graph[%s]", asc_graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(AscNodeSerializeUtils::SerializeAttrGroupsDef(*node, *attr_def),
                          "[Serialize][Attr] failed, graph[%s]", asc_graph.GetName().c_str());
  auto ir_def = node_def->mutable_ir_def();
  GE_ASSERT_NOTNULL(ir_def, "[Get][IrAttr] failed, graph[%s]", asc_graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(AscNodeSerializeUtils::SerializeIrDef(*node, *ir_def),
                          "[Serialize][IrAttr] failed, graph[%s]", asc_graph.GetName().c_str());
  GELOGD("[AscGraphUtils]Serialize ir attr node[%s:%s] success.", node->GetNamePtr(), ir_def->type().c_str());
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_GRAPH_SUCCESS(SerializeNodeInputs(node, *node_def, *ir_def));
  GE_ASSERT_GRAPH_SUCCESS(SerializeNodeOutputs(op_desc, *ir_def));
  GE_ASSERT_GRAPH_SUCCESS(SerializeNodeOutputTensors(op_desc, *node_def));
  GELOGD("Serialize node[%s:%s] success.", node->GetNamePtr(), node->GetTypePtr());
  return GRAPH_SUCCESS;
}
}  // namespace

NodePtr AscGraphUtils::InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                       const OpDescPtr &insert_op, const uint32_t input_index,
                                       const uint32_t output_index) {
  GE_ASSERT_NOTNULL(src);
  const NodePtr src_node = src->GetOwnerNode();
  GE_ASSERT_NOTNULL(src_node);
  auto compute_graph = src_node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(compute_graph);
  auto insert_node = compute_graph->InsertNode(src_node, insert_op);
  GE_ASSERT_GRAPH_SUCCESS(InsertNodeAfter(src, dsts, insert_node, input_index, output_index));
  return insert_node;
}

graphStatus AscGraphUtils::InsertNodeAfter(const OutDataAnchorPtr &src, const NodePtr &insert_node,
                                           const uint32_t input_index, const uint32_t output_index) {
  GE_CHECK_NOTNULL(src);
  const auto peer_in_anchor_range = src->GetPeerInDataAnchors();
  const std::vector<InDataAnchorPtr> peer_in_anchors(peer_in_anchor_range.begin(), peer_in_anchor_range.end());
  return InsertNodeAfter(src, peer_in_anchors, insert_node, input_index, output_index);
}

graphStatus AscGraphUtils::InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                           const NodePtr &insert_node, const uint32_t input_index,
                                           const uint32_t output_index) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(insert_node);

  const auto src_node = src->GetOwnerNodeBarePtr();
  GE_CHECK_NOTNULL(src_node);
  GE_ASSERT_TRUE(src_node->GetOwnerComputeGraph() == insert_node->GetOwnerComputeGraph(),
                 "src:%s and insert_node:%s do not exist in the same graph.", src_node->GetName().c_str(),
                 insert_node->GetName().c_str());

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src, insert_node->GetInDataAnchor(static_cast<int32_t>(input_index))));

  for (auto &dst : dsts) {
    const auto dst_node = dst->GetOwnerNodeBarePtr();
    GELOGI("Insert node(after) %s between %s->%s.", insert_node->GetName().c_str(), src_node->GetName().c_str(),
           dst_node->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src, dst));
    GE_ASSERT_GRAPH_SUCCESS(
        GraphUtils::AddEdge(insert_node->GetOutDataAnchor(static_cast<int32_t>(output_index)), dst));
  }
  insert_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>()->sched =
      src_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>()->sched;
  auto src_tensor_desc = src_node->GetOpDesc()->MutableOutputDesc(src->GetIdx());
  auto insert_tensor_desc = insert_node->GetOpDesc()->MutableOutputDesc(output_index);
  GE_ASSERT_GRAPH_SUCCESS(InheritAutofuseAttr(*src_tensor_desc, *insert_tensor_desc));
  return GRAPH_SUCCESS;
}

NodePtr AscGraphUtils::InsertNodeBefore(const InDataAnchorPtr &dst, const OpDescPtr &insert_op,
                                        const uint32_t input_index, const uint32_t output_index) {
  GE_ASSERT_NOTNULL(dst);
  const auto src_node_out_anchor = dst->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_node_out_anchor);
  const auto src_node = src_node_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(src_node);
  auto compute_graph = src_node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(compute_graph);
  auto insert_node = compute_graph->InsertNode(src_node, insert_op);
  GE_ASSERT_GRAPH_SUCCESS(InsertNodeBefore(dst, insert_node, input_index, output_index));
  return insert_node;
}

graphStatus AscGraphUtils::InsertNodeBefore(const InDataAnchorPtr &dst, const NodePtr &insert_node,
                                            const uint32_t input_index, const uint32_t output_index) {
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(insert_node);
  const auto dst_node = dst->GetOwnerNodeBarePtr();
  GE_CHECK_NOTNULL(dst_node);
  GE_ASSERT_TRUE(dst_node->GetOwnerComputeGraph() == insert_node->GetOwnerComputeGraph(),
                 "dst:%s and insert_node:%s do not exist in the same graph.", dst_node->GetName().c_str(),
                 insert_node->GetName().c_str());

  const auto src_node_out_anchor = dst->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_node_out_anchor);
  const auto src_node = src_node_out_anchor->GetOwnerNodeBarePtr();
  GE_CHECK_NOTNULL(src_node);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(src_node_out_anchor, dst));
  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::AddEdge(src_node_out_anchor, insert_node->GetInDataAnchor(static_cast<int32_t>(input_index))));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(insert_node->GetOutDataAnchor(static_cast<int32_t>(output_index)), dst));
  GELOGI("Insert node(before) %s between %s->%s", insert_node->GetName().c_str(), src_node->GetName().c_str(),
         dst_node->GetName().c_str());
  insert_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>()->sched =
      dst_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>()->sched;
  auto src_tensor_desc = src_node->GetOpDesc()->MutableOutputDesc(src_node_out_anchor->GetIdx());
  auto insert_tensor_desc = insert_node->GetOpDesc()->MutableOutputDesc(output_index);
  GE_ASSERT_GRAPH_SUCCESS(InheritAutofuseAttr(*src_tensor_desc, *insert_tensor_desc));
  return GRAPH_SUCCESS;
}

ComputeGraphPtr AscGraphUtils::GetComputeGraph(const AscGraph &asc_graph) {
  return asc_graph.impl_->GetComputeGraph();
}

Status AscGraphUtils::FromComputeGraph(const ComputeGraphPtr &compute_graph, AscGraph &graph) {
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_NOTNULL(graph.impl_);
  graph.impl_->compute_graph_ = compute_graph;
  return af::SUCCESS;
}

graphStatus AscGraphUtils::SerializeToProto(const AscGraph &asc_graph, ascendc_ir::proto::AscGraphDef &asc_graph_def) {
  const auto &ge_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(ge_graph);
  asc_graph_def.set_graph_name(asc_graph.GetName());
  // serialize asc graph attr
  auto asc_graph_attr_def = asc_graph_def.mutable_asc_graph_attr();
  GE_ASSERT_NOTNULL(asc_graph_attr_def);
  auto asc_graph_attr = ge_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(asc_graph_attr, "[GetOrCreate][AscGraphAttr] failed, graph[%s].", asc_graph.GetName().c_str());
  asc_graph_attr->SerializeAttr(*asc_graph_attr_def);
  // serialize asc nodes
  for (const auto &node : asc_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "[Get][Node] failed, graph[%s]", asc_graph.GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(SerializeNode(asc_graph, node, asc_graph_def));
  }
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::SerializeToBinary(const AscGraph &asc_graph, std::string &output) {
  ascendc_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_GRAPH_SUCCESS(SerializeToProto(asc_graph, asc_graph_def), "SerializeToProto failed.");
  GE_ASSERT_TRUE(asc_graph_def.SerializeToString(&output));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromBinary(const std::string &to_be_deserialized, AscGraph &out_asc_graph) {
  ascendc_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_TRUE(asc_graph_def.ParseFromString(to_be_deserialized));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeFromProto(asc_graph_def, out_asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::SerializeToReadable(const AscGraph &asc_graph, std::string &output) {
  ascendc_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_GRAPH_SUCCESS(SerializeToProto(asc_graph, asc_graph_def), "SerializeToProto failed.");
  GE_ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(asc_graph_def, &output), "SerializeToReadable failed.");
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromReadable(const std::string &to_be_deserialized, AscGraph &out_asc_graph) {
  ascendc_ir::proto::AscGraphDef asc_graph_def;
  GE_ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(to_be_deserialized, &asc_graph_def));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeFromProto(asc_graph_def, out_asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeGraphAttr(const AscGraph &asc_graph,
                                                const ascendc_ir::proto::AscGraphDef &asc_graph_def,
                                                const std::string &graph_name) {
  const auto &ge_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  const auto asc_graph_attr = ge_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(asc_graph_attr, "[GetOrCreate][AscGraphAttr] failed, graph[%s].", graph_name.c_str());
  GE_ASSERT_GRAPH_SUCCESS(asc_graph_attr->DeserializeAttr(asc_graph_def.asc_graph_attr()));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeNodeOutputAttrs(const OpDescPtr &op_desc,
                                                      const ascendc_ir::proto::AscNodeDef &asc_node_def) {
  int32_t index = 0;
  for (const auto &output_def : asc_node_def.outputs()) {
    const auto &output_attr = output_def.attr();
    const auto output_desc = op_desc->MutableOutputDesc(index);
    GE_ASSERT_NOTNULL(output_desc, "[Get][OutputDesc] failed of node[%s:%s, %d].", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str(), index);
    const auto output_tensor_attr = output_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(output_tensor_attr, "[GetOrCreate][OutputTensorAttr] failed of node[%s:%s, %d].",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
    output_tensor_attr->DeserializeAttr(output_attr, output_desc.get());
    index++;
  }
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeNodeInputAttrs(const AscNodePtr &asc_node, const OpDescPtr &op_desc,
                                                     const ascendc_ir::proto::AscNodeDef &asc_node_def,
                                                     const AscGraph &asc_graph) {
  int32_t connected_input_index = 0;
  for (const auto &input_def : asc_node_def.input_src()) {
    const int32_t index = GetInputAnchorIndex(asc_node_def.ir_def(), connected_input_index);
    GE_ASSERT_TRUE(index >= 0, "[Get][InputAnchor] failed, node[%s], input[%d].", asc_node->GetNamePtr(),
                   connected_input_index);
    const auto &src_node_name = input_def.src_node_name();
    const auto &src_node = asc_graph.FindNode(src_node_name.c_str());
    if (src_node == nullptr) {
      GELOGW("[Get][SrcNodeName] %s failed of node [%s:%d]", src_node_name.c_str(), asc_node->GetNamePtr(), index);
      ++connected_input_index;
      continue;
    }
    const auto src_out_index = input_def.src_out_index();
    const auto &src_op = src_node->GetOpDesc();
    GE_ASSERT_NOTNULL(src_op);
    const auto &out_tensor_desc = src_op->MutableOutputDesc(src_out_index);
    GE_ASSERT_NOTNULL(out_tensor_desc);
    const auto &out_tensor_attr = out_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(out_tensor_attr, "[GetOrCreate][InputTensorAttr] failed of node[%s:%s, %d].",
                      src_op->GetName().c_str(), src_op->GetType().c_str(), src_out_index);
    const auto input_desc = op_desc->MutableInputDesc(index);
    GE_ASSERT_NOTNULL(input_desc, "[Get][InputDesc] failed of node[%s:%s, %d].", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str(), index);
    const auto input_tensor_attr = input_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(input_tensor_attr, "[GetOrCreate][InputTensorAttr] failed of node[%s:%s, %d].",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), index);
    input_tensor_attr->dtype.tensor_desc_ = input_desc.get();
    *input_tensor_attr = *out_tensor_attr;
    ++connected_input_index;
  }
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeNodeDefinition(const AscNodePtr &asc_node,
                                                     const ascendc_ir::proto::AscGraphDef &asc_graph_def,
                                                     const int64_t node_index, const AscGraph &asc_graph) {
  const auto &asc_nodes = asc_graph_def.asc_node();
  GE_ASSERT_TRUE(node_index < asc_nodes.size(),
                 "[Deserialize][Node] failed, node_index[%ld] should be less than nodes size[%zu].", node_index,
                 asc_nodes.size());
  const auto &asc_node_def = asc_nodes[static_cast<int32_t>(node_index)];
  const auto op_desc = asc_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "[Get][OpDesc] failed, graph[%s]", asc_graph.GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(AscNodeDeserializeUtils::DeserializeAttrGroupsDef(asc_node_def.attr(), *asc_node));
  GE_ASSERT_GRAPH_SUCCESS(AscNodeDeserializeUtils::DeserializeIrDef(asc_node_def.ir_def(), *asc_node));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeNodeOutputAttrs(op_desc, asc_node_def));
  GE_ASSERT_GRAPH_SUCCESS(DeserializeNodeInputAttrs(asc_node, op_desc, asc_node_def, asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::DeserializeFromProto(const ascendc_ir::proto::AscGraphDef &asc_graph_def,
                                                AscGraph &asc_graph) {
  auto &graph_name = asc_graph_def.graph_name();
  // 1. Add AscGraph
  asc_graph.impl_->compute_graph_->SetName(graph_name);
  GE_ASSERT_GRAPH_SUCCESS(EstablishAscNodeAndEdges(asc_graph_def, asc_graph),
                          "EstablishAscNodeAndEdges of graph[%s] failed.", graph_name.c_str());
  // deserialize graph attr
  GE_ASSERT_GRAPH_SUCCESS(DeserializeGraphAttr(asc_graph, asc_graph_def, graph_name));
  // deserialize node attr, protobuf make sure the order as serializing
  int64_t node_index = 0L;
  for (const auto &asc_node : asc_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(asc_node, "[Get][Node] failed, graph[%s]", graph_name.c_str());
    // update asc node AscTensors
    asc_node->outputs();
    asc_node->inputs();
    GE_ASSERT_GRAPH_SUCCESS(DeserializeNodeDefinition(asc_node, asc_graph_def, node_index, asc_graph));
    node_index++;
  }
  return GRAPH_SUCCESS;
}

graphStatus AscGraphUtils::ConvertComputeGraphToAscGraph(const ComputeGraphPtr &compute_graph, AscGraph &asc_graph) {
  GE_ASSERT_NOTNULL(compute_graph);
  // 1. 如果外部没有指定名字, 共享名字
  if (asc_graph.GetName().empty()) {
    auto asc_compute_graph = GetComputeGraph(asc_graph);
    GE_ASSERT_NOTNULL(asc_compute_graph);
    asc_compute_graph->SetName(compute_graph->GetName());
  }
  // 2. 转换Node到AscNode, 共享OpDesc
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &node : compute_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = asc_graph.AddNode(op);
    GE_ASSERT_NOTNULL(dst_new_node);
    all_new_nodes[dst_new_node->GetName()] = dst_new_node;
  }
  // 3. 转换连边关系
  for (const auto &src_node : compute_graph->GetDirectNode()) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RelinkGraphEdges(src_node, "", all_new_nodes));
  }
  // 4. 转换graph上的属性组属性
  AscGraphImpl::DoCopyAscGraphAttrImpl(compute_graph, AscGraphUtils::GetComputeGraph(asc_graph));
  return GRAPH_SUCCESS;
}

graphStatus AscNodeSerializeUtils::SerializeIrDef(const AscNode &node, ascendc_ir::proto::IrDef &ir_def) {
  ir_def.set_type(node.GetType());
  const auto &op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (const auto &ir_input : op_desc->GetIrInputs()) {
    ir_def.add_input_names(ir_input.first);
    ir_def.add_input_ir_type(ir_input.second);
  }
  for (const auto &ir_output : op_desc->GetIrOutputs()) {
    ir_def.add_output_names(ir_output.first);
    ir_def.add_output_ir_type(ir_output.second);
  }
  GELOGD("Serialize ir def node[%s:%s] success.", node.GetNamePtr(), ir_def.type().c_str());
  return GRAPH_SUCCESS;
}

graphStatus AscNodeSerializeUtils::SerializeAttrGroupsDef(
    const AscNode &node, ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(asc_node_attr);
  return asc_node_attr->SerializeAttr(asc_node_attr_groups_def);
}

graphStatus AscNodeDeserializeUtils::DeserializeIrDef(const ascendc_ir::proto::IrDef &ir_def, AscNode &node) {
  const auto &type = ir_def.type();
  const auto &op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetType(type);
  GE_ASSERT_EQ(ir_def.input_names_size(), ir_def.input_ir_type_size());
  const auto &registry = ascir::AscirRegistry::GetInstance().GetAll();
  const auto asc_ir_def_iter = registry.find(type);
  const ascir::AscIrDef *asc_ir_def = (asc_ir_def_iter == registry.end()) ? nullptr : &asc_ir_def_iter->second;
  for (int64_t index = 0; index < ir_def.input_names_size(); ++index) {
    const auto &input_name = ir_def.input_names(index);
    const auto input_type = static_cast<IrInputType>(ir_def.input_ir_type(index));
    op_desc->AppendIrInput(input_name, input_type);
    if (asc_ir_def != nullptr) {
      RestoreInputDtypeSymbol(*op_desc, *asc_ir_def, input_name, input_type);
    }
  }
  GE_ASSERT_EQ(ir_def.output_names_size(), ir_def.output_ir_type_size());
  for (int64_t index = 0; index < ir_def.output_names_size(); ++index) {
    op_desc->AppendIrOutput(ir_def.output_names(index), static_cast<IrOutputType>(ir_def.output_ir_type(index)));
  }
  GELOGD("Deserialize ir def node[%s:%s] success.", node.GetNamePtr(), ir_def.type().c_str());
  return GRAPH_SUCCESS;
}
graphStatus AscNodeDeserializeUtils::DeserializeAttrGroupsDef(
    const ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_attr_groups_def, AscNode &node) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(asc_node_attr);
  GE_ASSERT_GRAPH_SUCCESS(asc_node_attr->DeserializeAttr(asc_node_attr_groups_def),
                          "[Deserialize][Attr] failed of node[%s:%s].", op_desc->GetName().c_str(),
                          op_desc->GetType().c_str());
  return GRAPH_SUCCESS;
}
graphStatus ExpressionSerializer::Serialize(const AnyValue &av, GeIrAttrDef &def) {
  Expression expression;
  GE_ASSERT_GRAPH_SUCCESS(av.GetValue(expression));
  def.set_expression(expression.Str().get());
  return GRAPH_SUCCESS;
}

graphStatus ExpressionSerializer::Deserialize(const GeIrAttrDef &def, AnyValue &av) {
  return av.SetValue(Expression::Deserialize(def.expression().c_str()));
}

REG_GEIR_SERIALIZER(expression_serializer, ExpressionSerializer, ge::GetTypeId<Expression>(), GeIrAttrDef::kExpression);
}  // namespace af
