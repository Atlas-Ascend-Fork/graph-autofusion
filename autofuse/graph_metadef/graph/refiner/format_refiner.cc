/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/refiner/format_refiner.h"

#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "graph/ref_relation.h"
#include "framework/common/debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "graph/types_af.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace af {
namespace {
const size_t kDimSizeOf4D = 4UL;
const std::set<std::string> kChangeDimNodes = {PERMUTE, EXPANDDIMS, SQUEEZE};
const char_t *const kIsGraphInferred = "_is_graph_inferred";
thread_local RefRelations reflection_builder;

static graphStatus ReflectionProcess(const std::unordered_set<RefCell, RefCellHash> &reflection,
                                     std::deque<NodePtr> &nodes, const Format to_be_set_format) {
  for (const auto &reflection_cell : reflection) {
    const auto &reflection_node = reflection_cell.node;
    const auto in_out_idx = reflection_cell.in_out_idx;
    GE_CHECK_NOTNULL(reflection_node);
    if (reflection_cell.in_out == NODE_IN) {
      auto desc = reflection_node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_out_idx));
      GE_CHECK_NOTNULL(desc);
      desc->SetOriginFormat(to_be_set_format);
      desc->SetFormat(to_be_set_format);
    } else {
      auto desc = reflection_node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(in_out_idx));
      GE_CHECK_NOTNULL(desc);
      desc->SetOriginFormat(to_be_set_format);
      desc->SetFormat(to_be_set_format);
    }
    nodes.push_back(reflection_cell.node);
  }

  return GRAPH_SUCCESS;
}

static graphStatus BiasAddFormatFixProcess(const NodePtr &graph_node_ptr) {
  // 5 meas dim num
  if ((graph_node_ptr->GetType() != "BiasAdd") && (graph_node_ptr->GetType() != "BiasAddGrad")) {
    return GRAPH_SUCCESS;
  }
  const std::unordered_map<std::string, Format> kTfFormatFix = {{"NHWC", FORMAT_NDHWC}, {"NCHW", FORMAT_NCDHW}};
  for (size_t i = 0UL; i < graph_node_ptr->GetOpDesc()->GetInputsSize(); i++) {
    const auto in_desc = graph_node_ptr->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(in_desc);
    const auto dim_num = in_desc->MutableShape().GetDimNum();
    if (dim_num == 5UL) {  // 5 means dim num
      const auto org_format = in_desc->GetOriginFormat();
      const auto key = TypeUtils::FormatToSerialString(org_format);
      const auto fixed_format = (kTfFormatFix.count(key) == 0UL) ? org_format : kTfFormatFix.at(key);
      in_desc->SetOriginFormat(fixed_format);
      in_desc->SetFormat(fixed_format);
      GELOGD("Fix the %zu'th input of node[%s]. Origin format is %s , after fixed it is %s", i,
             graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(org_format).c_str(),
             TypeUtils::FormatToSerialString(fixed_format).c_str());
    } else if (dim_num < 4UL) {
      in_desc->SetOriginFormat(FORMAT_ND);
      in_desc->SetFormat(FORMAT_ND);
      GELOGD("Fix the %zu'th input of node[%s]. Origin format is %s , after fixed it is %s", i,
             graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(FORMAT_ND).c_str(),
             TypeUtils::FormatToSerialString(FORMAT_ND).c_str());
    } else {
      // do nothing
    }
  }
  for (size_t i = 0UL; i < graph_node_ptr->GetOpDesc()->GetOutputsSize(); i++) {
    const auto out_desc = graph_node_ptr->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(out_desc);
    if (out_desc->MutableShape().GetDimNum() != 5UL) {  // 5 means dim num
      continue;
    }
    const auto org_format = out_desc->GetOriginFormat();
    const auto key = TypeUtils::FormatToSerialString(org_format);
    const auto fixed_format = (kTfFormatFix.count(key) == 0UL) ? org_format : kTfFormatFix.at(key);
    out_desc->SetOriginFormat(fixed_format);
    out_desc->SetFormat(fixed_format);
    GELOGD("fix the %zu'th output of node[%s]. Origin format is %s , after fixed it is %s", i,
           graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(org_format).c_str(),
           TypeUtils::FormatToSerialString(fixed_format).c_str());
  }
  return GRAPH_SUCCESS;
}

static bool JudgeNodeIsAllNd(const OpDescPtr &one_op_desc, const NodePtr &one_node_ptr,
                             std::vector<NodePtr> &anchor_data_nodes) {
  // consider special node save process
  // Pre-save data node (only main graph data) and default infer fail
  if (OpTypeUtils::IsDataNode(one_node_ptr->GetType())) {
    anchor_data_nodes.push_back(one_node_ptr);
  }

  // get all input desc format
  const auto input_size = static_cast<uint32_t>(one_op_desc->GetAllInputsSize());
  for (uint32_t i = 0U; i < input_size; i++) {
    // Operator pre-set format but not origin format
    const auto &input_desc = one_op_desc->MutableInputDesc(i);
    GE_IF_BOOL_EXEC(input_desc == nullptr, continue);
    const auto input_format = input_desc->GetFormat();
    if ((input_format != FORMAT_ND) && (input_format != FORMAT_RESERVED)) {
      return false;
    }
  }
  // Get all output desc format
  const auto output_size = static_cast<uint32_t>(one_op_desc->GetOutputsSize());
  for (uint32_t i = 0U; i < output_size; i++) {
    const auto &output_desc = one_op_desc->MutableOutputDesc(i);
    GE_IF_BOOL_EXEC(output_desc == nullptr, continue);
    const auto output_format = output_desc->GetFormat();
    if ((output_format != FORMAT_ND) && (output_format != FORMAT_RESERVED)) {
      return false;
    }
  }
  return true;
}

static graphStatus AnchorsInferProcess(std::deque<NodePtr> &nodes, const OutDataAnchorPtr &out_data_anchor,
                                       const Format to_be_set_format) {
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    GE_IF_BOOL_EXEC(peer_in_data_anchor == nullptr, continue);

    const auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(peer_in_data_node == nullptr, continue);
    const auto peer_in_data_opdesc = peer_in_data_node->GetOpDesc();
    GE_IF_BOOL_EXEC(peer_in_data_opdesc == nullptr, continue);

    // Check format whether have been set
    const int32_t idx = peer_in_data_anchor->GetIdx();
    // do peer_out_node name and index as key to lookup reflections
    const RefCell key(peer_in_data_node->GetName(), peer_in_data_node, NODE_IN, idx);
    std::unordered_set<RefCell, RefCellHash> reflection;
    auto ret_status = reflection_builder.LookUpRefRelations(key, reflection);
    if (ret_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "LookUpRefRelations failed! Node is [%s], the %d input edge",
                           (peer_in_data_node->GetName()).c_str(), idx);
      GELOGE(GRAPH_FAILED, "[Call][LookUpRefRelations] failed! Node is [%s], the %d input edge",
             (peer_in_data_node->GetName()).c_str(), idx);
      return GRAPH_FAILED;
    }

    bool format_locked = false;
    (void)AttrUtils::GetBool(peer_in_data_opdesc, ATTR_NAME_FORMAT_LOCKED, format_locked);
    GELOGD("Get format locked flag:%u (shape can not be changed while value is equal to 1) from peer in node:%s.",
           static_cast<uint32_t>(format_locked), peer_in_data_node->GetName().c_str());

    auto ge_tensor_desc = peer_in_data_opdesc->MutableInputDesc(static_cast<uint32_t>(idx));
    if (ge_tensor_desc == nullptr) {
      continue;
    }
    if ((ge_tensor_desc->GetOriginFormat() == FORMAT_ND) && (!format_locked)) {
      const auto dim_num = ge_tensor_desc->GetShape().GetDimNum();
      GE_IF_BOOL_EXEC(dim_num == 0UL, GELOGI("node name:%s idx:%d in is scalar. stop forward infer!",
                                             peer_in_data_node->GetName().c_str(), idx);
                      continue);

      /// Check whether node to change dims ()
      /// Because some node will calculate with 5D, C dim maybe multi meaning
      const auto peer_in_data_node_type = peer_in_data_node->GetType();
      const auto iter1 = kChangeDimNodes.find(peer_in_data_node_type);
      // 4 means dims num
      if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4UL)) {
        GELOGD("Node[%s] is change dim node. do not infer origin format", (peer_in_data_node->GetName()).c_str());
        continue;
      }

      if (reflection.empty()) {
        ge_tensor_desc->SetOriginFormat(to_be_set_format);
        ge_tensor_desc->SetFormat(to_be_set_format);

        /// Because netoutput node added before infer format ,so netoutput is end condition
        /// must set netoutput format , because saved result depend on format
        GE_IF_BOOL_EXEC(peer_in_data_node_type == NETOUTPUT, continue);

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Back]!Node is [%s] ", (peer_in_data_node->GetName()).c_str());
        ret_status = NodeUtilsEx::InferOriginFormat(peer_in_data_node);
        GE_IF_BOOL_EXEC(ret_status != GRAPH_SUCCESS,
                        GELOGE(GRAPH_FAILED, "[Infer][Format] failed, node:%s", (peer_in_data_node->GetName()).c_str());
                        return GRAPH_FAILED);
        nodes.push_back(peer_in_data_node);
      } else {
        const auto ret = ReflectionProcess(reflection, nodes, to_be_set_format);
        GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Reflect][Node] failed! status:%d", ret);
                        return GRAPH_FAILED);
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus FormatRefiner::RefreshConstantOutProcess(const ComputeGraphPtr &com_graph, const OpDescPtr &op_desc) {
  if ((op_desc->GetType() == CONSTANTOP) && (!IsGraphInferred(com_graph))) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(op_desc, "value", tensor_value)) {
      REPORT_INNER_ERR_MSG("E18888", "GetTensor failed, node name:%s.", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Tensor] failed, node name:%s.", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
    GE_CHECK_NOTNULL(tensor_value);
    (void)op_desc->UpdateOutputDesc(0U, tensor_value->GetTensorDesc());
  }
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::GetAnchorPoints(const ComputeGraphPtr &com_graph, std::vector<NodePtr> &anchor_points,
                                           std::vector<NodePtr> &anchor_data_nodes) {
  anchor_points.clear();
  // Get all anchor point nodes and switch nodes
  for (auto &one_node_ptr : com_graph->GetAllNodes()) {
    if (one_node_ptr == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "node ptr in graph(%s) should not be null", com_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node ptr in graph(%s) should not be null", com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    const auto &one_op_desc = one_node_ptr->GetOpDesc();
    if (one_op_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "node's opdesc is nullptr, graph:%s", com_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node's opdesc is nullptr, graph:%s", com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    graphStatus ret_status = RefreshConstantOutProcess(com_graph, one_op_desc);
    if (ret_status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][RefreshConstantOutProcess] failed! graph:%s, op:%s", com_graph->GetName().c_str(),
             one_op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }

    // check anchor point valid
    if (JudgeNodeIsAllNd(one_op_desc, one_node_ptr, anchor_data_nodes)) {
      continue;
    }
    // special process for biasAdd op
    // In tensorflow, biasAdd's format is alwayse NHWC even though set the arg
    // "data_format" to NDHWC or NCDHW.It will destroy our format-infer mechanism
    // so here do special process
    ret_status = BiasAddFormatFixProcess(one_node_ptr);
    if (ret_status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][BiasAddFormatFixProcess] failed! node:%s, graph:%s", one_node_ptr->GetName().c_str(),
             com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }

    GELOGD("Node[%s] is anchor point!", one_node_ptr->GetName().c_str());
    anchor_points.push_back(one_node_ptr);
  }
  GELOGI("anchor_points number is %zu", anchor_points.size());
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::AnchorProcess(const NodePtr &anchor_node) {
  std::deque<NodePtr> nodes;
  nodes.push_back(anchor_node);
  while (!nodes.empty()) {
    const NodePtr one_node = nodes.front();
    nodes.pop_front();
    GE_CHECK_NOTNULL(one_node);
    GE_CHECK_NOTNULL(one_node->GetOpDesc());
    graphStatus ret_status = BackInferProcess(nodes, one_node);
    if ((ret_status != GRAPH_SUCCESS) && (one_node != nullptr)) {
      GELOGE(ret_status, "[Back][InferProcess] failed! status:%d, node name [%s]", ret_status,
             one_node->GetName().c_str());
      return ret_status;
    }
    ret_status = ForwardInferProcess(nodes, one_node);
    if ((ret_status != GRAPH_SUCCESS) && (one_node != nullptr)) {
      GELOGE(ret_status, "[Forward][InferProcess] failed! status:%d, node name [%s]", ret_status,
             one_node->GetName().c_str());
      return ret_status;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::BackInferProcess(std::deque<NodePtr> &nodes, const NodePtr &node) {
  GELOGD("Enter back infer format for Node [%s]", node->GetName().c_str());
  for (const auto &in_anchor : node->GetAllInDataAnchorsPtr()) {
    const auto in_data_anchor_idx = in_anchor->GetIdx();
    GELOGD("Node [%s]:%d [B]", node->GetName().c_str(), in_data_anchor_idx);
    const auto input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor_idx));
    GE_IF_BOOL_EXEC(input_desc == nullptr, continue);
    const auto to_be_set_format = input_desc->GetOriginFormat();
    GE_IF_BOOL_EXEC(to_be_set_format == FORMAT_ND, GELOGD("Node [%s] format is ND.[B]", node->GetName().c_str());
                    continue);
    const auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);
    const auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
    const int32_t idx = peer_out_data_anchor->GetIdx();
    // do peer_out_node name and index as key to lookup reflections
    const RefCell key(peer_out_data_node->GetName(), peer_out_data_node, NODE_OUT, idx);
    std::unordered_set<RefCell, RefCellHash> reflection;
    auto status = reflection_builder.LookUpRefRelations(key, reflection);
    GE_IF_BOOL_EXEC(status != GRAPH_SUCCESS,
                    GELOGE(GRAPH_FAILED, "[Call][LookUpRefRelations] failed! Node is [%s], the %d out edge",
                           (peer_out_data_node->GetName()).c_str(), idx);
                    return GRAPH_FAILED);

    // Check format whether have been set
    // op_desc of node should not be null
    auto ge_tensor_desc = peer_out_data_node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(idx));

    bool format_locked = false;
    (void)AttrUtils::GetBool(peer_out_data_node->GetOpDesc(), ATTR_NAME_FORMAT_LOCKED, format_locked);
    GELOGD("Get format locked flag:%u (shape is locked if value is equal to 1) from peer out node:%s.",
           static_cast<uint32_t>(format_locked), peer_out_data_node->GetName().c_str());

    if ((ge_tensor_desc->GetOriginFormat() == FORMAT_ND) && (!format_locked)) {
      const auto dim_num = ge_tensor_desc->GetShape().GetDimNum();
      GE_IF_BOOL_EXEC(dim_num == 0UL, GELOGD("node name:%s idx:%d out is scalar. stop back infer!",
                                             peer_out_data_node->GetName().c_str(), idx);
                      continue);

      /// Check whether node to change dims ()
      /// Because some node will calculate with 5D, C dim maybe multi meaning
      const auto peer_out_data_node_type = peer_out_data_node->GetType();
      const auto iter1 = kChangeDimNodes.find(peer_out_data_node_type);
      // 4 means dims num
      if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4UL)) {
        GELOGD("Node[%s] is change dim node and shape is smaller than 4. do not modify format",
               (peer_out_data_node->GetName()).c_str());
        continue;
      }

      if (reflection.empty()) {
        ge_tensor_desc->SetOriginFormat(to_be_set_format);
        ge_tensor_desc->SetFormat(to_be_set_format);

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Back]!Node is [%s] ", (peer_out_data_node->GetName()).c_str());
        status = NodeUtilsEx::InferOriginFormat(peer_out_data_node);
        GE_IF_BOOL_EXEC(status != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Infer][Format] failed, Node:%s",
                                                        (peer_out_data_node->GetName()).c_str());
                        return GRAPH_FAILED);
        nodes.push_back(peer_out_data_node);
      } else {
        const auto ret = ReflectionProcess(reflection, nodes, to_be_set_format);
        GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Reflect][Node] failed! status:%d", ret);
                        return GRAPH_FAILED);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::ForwardInferProcess(std::deque<NodePtr> &nodes, const NodePtr &node) {
  GELOGD("Enter forward infer format for Node [%s]", node->GetName().c_str());
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
    const auto out_data_anchor_idx = out_data_anchor->GetIdx();
    GELOGD("Node [%s]:%d [F]", node->GetName().c_str(), out_data_anchor_idx);
    if (node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(out_data_anchor_idx)) == nullptr) {
      continue;
    }
    const auto to_be_set_format =
        node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(out_data_anchor_idx))->GetOriginFormat();
    if (to_be_set_format == FORMAT_ND) {
      GELOGD("Node [%s] format is ND.[F]", node->GetName().c_str());
      continue;
    }
    const auto ret = AnchorsInferProcess(nodes, out_data_anchor, to_be_set_format);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

void FormatRefiner::RefreshOriginFormatOfAnchor(const std::vector<NodePtr> &anchor_points) {
  for (const auto &node : anchor_points) {
    for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
      // single op support private format set, its origin format should not be override
      const auto ori_format = input_desc->GetOriginFormat();
      const auto format = input_desc->GetFormat();
      if (TypeUtilsInner::IsInternalFormat(format)) {
        continue;
      }
      if ((input_desc != nullptr) && ((ori_format == FORMAT_ND) || (ori_format == FORMAT_RESERVED))) {
        input_desc->SetOriginFormat(input_desc->GetFormat());
      }
    }
    for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      const auto ori_format = output_desc->GetOriginFormat();
      const auto format = output_desc->GetFormat();
      if (TypeUtilsInner::IsInternalFormat(format)) {
        continue;
      }
      if ((output_desc != nullptr) && ((ori_format == FORMAT_ND) || (ori_format == FORMAT_RESERVED))) {
        output_desc->SetOriginFormat(output_desc->GetFormat());
      }
    }
  }
}

graphStatus FormatRefiner::DataNodeFormatProcess(const ComputeGraphPtr &graph,
                                                 const std::vector<NodePtr> &anchor_data_nodes,
                                                 const Format data_format) {
  if (!(IsGraphInferred(graph) && (!TypeUtilsInner::IsInternalFormat(data_format)) && (data_format != FORMAT_ND))) {
    GELOGI("Not necessary to do DataNodeFormatProcess. is_graph_inferred:%d, data_format:%s",
           static_cast<int32_t>(IsGraphInferred(graph)), TypeUtils::FormatToSerialString(data_format).c_str());
    return GRAPH_SUCCESS;
  }
  GELOGD("Enter DataNodeFormatProcess");
  std::vector<NodePtr> uninferred_data_nodes;
  // Check and renew data nodes format
  for (const auto &data_node : anchor_data_nodes) {
    GE_CHECK_NOTNULL(data_node);
    const auto op_desc = data_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto input_desc = op_desc->MutableInputDesc(0U);
    const auto output_desc = op_desc->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHECK_NOTNULL(output_desc);

    const auto curr_format = output_desc->GetOriginFormat();
    if (curr_format != FORMAT_ND) {
      // Data format has been inferred, continue
      continue;
    }
    // Keep the data format as ND because the definition is missing when the input shape has fewer than four dims.
    if (input_desc->MutableShape().GetDimNum() < kDimSizeOf4D) {
      continue;
    }
    // Set the format for a data node that has not been inferred.
    input_desc->SetOriginFormat(data_format);
    input_desc->SetFormat(data_format);
    output_desc->SetOriginFormat(data_format);
    output_desc->SetFormat(data_format);
    uninferred_data_nodes.push_back(data_node);
  }
  // Reinfer the format from uninferred data nodes.
  for (const auto &node : uninferred_data_nodes) {
    if (node == nullptr) {
      continue;
    }
    GELOGD("data node [%s] start infer format process", node->GetName().c_str());
    const auto status = AnchorProcess(node);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][AnchorProcess] failed, status:%d, node:%s", status, node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  GELOGD("DataNodeFormatProcess success");
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::InferOrigineFormat(const ComputeGraphPtr &graph) {
  GELOGI("Enter InferOriginFormat process!");

  // True: inferred; false: not inferred.
  std::vector<NodePtr> anchor_points;
  std::vector<NodePtr> anchor_data_nodes;

  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] input graph is nullptr");
    return GRAPH_FAILED;
  }
  // build reflection relations of boundary
  (void)reflection_builder.Clear();
  auto status = reflection_builder.BuildRefRelations(*graph);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Call][BuildRefRelations] failed, graph:%s", graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // User set global net format
  status = GetAnchorPoints(graph, anchor_points, anchor_data_nodes);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "GetAnchorPoints Process Failed! graph:%s", graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // Refresh origin format of anchor point
  RefreshOriginFormatOfAnchor(anchor_points);
  // Infer format process
  for (const auto &anchor_node : anchor_points) {
    if (anchor_node == nullptr) {
      continue;
    }
    status = AnchorProcess(anchor_node);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][AnchorProcess] failed, node:%s", anchor_node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  /// According to discuss with sys-enginer, data node default format is ND.Its format
  /// should be set by inference. But if some data-node cannot be inferred, set the context's
  /// format for these data nodes.
  /// Notice: ignore 5D formats
  const auto data_format = graph->GetDataFormat();
  status = DataNodeFormatProcess(graph, anchor_data_nodes, data_format);

  (void)AttrUtils::SetBool(graph, kIsGraphInferred, true);

  return status;
}

bool FormatRefiner::IsGraphInferred(const ComputeGraphPtr &graph) {
  bool is_graph_inferred = false;
  return (AttrUtils::GetBool(graph, kIsGraphInferred, is_graph_inferred) && is_graph_inferred);
}
}  // namespace af
