/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/node_utils.h"
#include <stack>
#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/ge_context.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/constant_utils.h"
#include "common/checker.h"

namespace af {
const std::set<std::string> kConstOpTypes{"Const", "Constant"};

const std::set<std::string> kEnterOpTypes{"Enter", "RefEnter"};
const std::set<std::string> kMergeOpTypes{"Merge", "RefMerge"};
const std::set<std::string> kSwitchOpTypes{"Switch", "RefSwitch"};
const std::set<std::string> kNextIterationOpTypes{"NextIteration", "RefNextIteration"};
const std::set<std::string> kExitOpTypes{"Exit", "RefExit"};

const std::set<std::string> kIfOpTypes{"If", "_If", "StatelessIf"};
const std::set<std::string> kWhileOpTypes{"While", "_While", "StatelessWhile"};
const std::set<std::string> kCaseOpTypes{"Case"};
const std::set<std::string> kForOpTypes{"For"};

const char_t *const kRefIndex = "_parent_node_index";
const char_t *const kPartSrcGraph = "part_src_graph";

namespace {
constexpr int32_t kInvalidIndex = -1;

const std::unordered_set<std::string> kMultiBranchControlFlowOps = [] {
  std::unordered_set<std::string> s;
  s.insert(kIfOpTypes.begin(), kIfOpTypes.end());
  s.insert(kWhileOpTypes.begin(), kWhileOpTypes.end());
  s.insert(kForOpTypes.begin(), kForOpTypes.end());
  s.insert(kSwitchOpTypes.begin(), kSwitchOpTypes.end());
  return s;
}();

bool OpShapeIsUnknown(const OpDescPtr &desc) {
  for (const auto &ptr : desc->GetAllInputsDescPtr()) {
    const auto ge_shape = ptr->GetShape();
    auto dims = ge_shape.GetDims();
    if (std::any_of(dims.begin(), dims.end(),
                    [](const int64_t dim) { return ((dim == UNKNOWN_DIM) || (dim == (UNKNOWN_DIM_NUM))); })) {
      return true;
    }
  }
  for (const auto &ptr : desc->GetAllOutputsDescPtr()) {
    const auto ge_shape = ptr->GetShape();
    auto dims = ge_shape.GetDims();
    if (std::any_of(dims.begin(), dims.end(),
                    [](const int64_t dim) { return ((dim == UNKNOWN_DIM) || (dim == (UNKNOWN_DIM_NUM))); })) {
      return true;
    }
  }
  return false;
}

bool IsComputableOp(const NodePtr &node) {
  if ((node->GetType() == DATA) || (node->GetType() == NETOUTPUT)) {
    return false;
  }
  if (!node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
    return false;
  }
  return true;
}
}  // namespace

graphStatus NodeUtils::ClearInDataAnchor(const NodePtr &node_ptr, const InDataAnchorPtr &in_data_anchor) {
  GE_CHK_BOOL_EXEC((node_ptr != nullptr) && (node_ptr->impl_ != nullptr) && (in_data_anchor != nullptr),
                   REPORT_INNER_ERR_MSG("E18888", "param node or in_data_anchor is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] node or in_data_anchor is nullptr");
  bool find_flag = false;
  uint32_t index = 0U;
  std::vector<InDataAnchorPtr>::iterator it = node_ptr->impl_->in_data_anchors_.end();
  for (const auto &tmp : node_ptr->impl_->in_data_anchors_) {
    if (tmp == in_data_anchor) {
      find_flag = true;
      const auto iter = node_ptr->impl_->in_data_anchors_.begin() + static_cast<int64_t>(index);
      if (iter != node_ptr->impl_->in_data_anchors_.end()) {
        it = node_ptr->impl_->in_data_anchors_.erase(iter);
      }
      break;
    }
    index++;
  }
  while (it != node_ptr->impl_->in_data_anchors_.end()) {
    (*it)->SetIdx(static_cast<int32_t>(index));
    index++;
    ++it;
  }

  if (!find_flag) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::SetAllAnchorStatus(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node_ptr is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] node is nullptr");
  GE_CHK_BOOL_EXEC(SetAllAnchorStatus(*node_ptr) == GRAPH_SUCCESS,
                   REPORT_INNER_ERR_MSG("E18888", "SetAllAnchorStatus failed, node:%s", node_ptr->GetName().c_str());
                   return GRAPH_FAILED, "[Set][AllAnchorStatus] failed, node:%s", node_ptr->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::SetAllAnchorStatus(Node &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Param node impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node impl is nullptr.");
    return GRAPH_FAILED;
  }
  node.impl_->anchor_status_updated_ = true;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool NodeUtils::IsAnchorStatusSet(const NodePtr &node_ptr) {
  GE_CHK_BOOL_EXEC(node_ptr != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node_ptr is nullptr, check invalid");
                   return false, "[Check][Param] node is nullptr");
  return IsAnchorStatusSet(*node_ptr);
}

bool NodeUtils::IsAnchorStatusSet(const Node &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Param node impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node impl is nullptr.");
    return false;
  }
  return node.impl_->anchor_status_updated_;
}

graphStatus NodeUtils::MoveOutputEdges(const NodePtr &origin_node, const NodePtr &new_node) {
  if ((origin_node == nullptr) || (new_node == nullptr)) {
    return GRAPH_FAILED;
  }
  auto origin_out_data_anchors = origin_node->GetAllOutDataAnchors();
  const auto origin_out_data_anchors_size = origin_out_data_anchors.size();
  auto new_out_data_anchors = new_node->GetAllOutDataAnchors();
  if (origin_out_data_anchors_size != new_out_data_anchors.size()) {
    return GRAPH_FAILED;
  }

  for (size_t i = 0UL; i < origin_out_data_anchors_size; ++i) {
    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInDataAnchors()) {
      GE_CHK_BOOL_EXEC(
          origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS,
          REPORT_INNER_ERR_MSG("E18888", "unlink peer_dataanchor failed, node:%s", origin_node->GetName().c_str());
          continue, "[Unlink][PeerAnchor] failed, node:%s", origin_node->GetName().c_str());
      GE_CHK_BOOL_EXEC(
          new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS,
          REPORT_INNER_ERR_MSG("E18888", "LinkTo peer_dataanchor failed, node:%s", new_node->GetName().c_str());
          continue, "[LinkTo][PeerAnchor] failed, node:%s", new_node->GetName().c_str());
    }

    for (const auto &peer_anchor : origin_out_data_anchors.at(i)->GetPeerInControlAnchors()) {
      GE_CHK_BOOL_EXEC(
          origin_out_data_anchors.at(i)->Unlink(peer_anchor) == GRAPH_SUCCESS,
          REPORT_INNER_ERR_MSG("E18888", "unlink peer_control_anchor failed, node:%s", origin_node->GetName().c_str());
          continue, "[Unlink][PeerAnchor] failed, node:%s", origin_node->GetName().c_str());
      GE_CHK_BOOL_EXEC(
          new_out_data_anchors.at(i)->LinkTo(peer_anchor) == GRAPH_SUCCESS,
          REPORT_INNER_ERR_MSG("E18888", "LinkTo peer_control_anchor failed, node:%s", new_node->GetName().c_str());
          continue, "[LinkTo][PeerAnchor] failed, node:%s", new_node->GetName().c_str());
    }
  }

  const auto origin_out_control_anchor = origin_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(origin_out_control_anchor);
  const auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInControlAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E18888", "linkto peer_anchor from %s to %s failed.",
                                          new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                                          peer_anchor->GetOwnerNode()->GetName().c_str());
                     continue, "[LinkTo][PeerAnchor] from %s to %s failed",
                     new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_anchor->GetOwnerNode()->GetName().c_str());
  }
  for (const auto &peer_anchor : origin_out_control_anchor->GetPeerInDataAnchors()) {
    GE_CHK_BOOL_EXEC(new_out_control_anchor->LinkTo(peer_anchor) == GRAPH_SUCCESS,
                     REPORT_INNER_ERR_MSG("E18888", "linkto peer_anchor from %s to %s failed.",
                                          new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                                          peer_anchor->GetOwnerNode()->GetName().c_str());
                     continue, "[LinkTo][PeerAnchor] from %s to %s failed",
                     new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                     peer_anchor->GetOwnerNode()->GetName().c_str());
  }
  origin_out_control_anchor->UnlinkAll();

  return GRAPH_SUCCESS;
}

bool NodeUtils::IsConst(const Node &node) {
  const auto src_node_type = node.GetType();
  const bool is_const = ((src_node_type == CONSTANT) || (src_node_type == CONSTANTOP));
  return is_const;
}

void NodeUtils::UpdateIsInputConst(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node_ptr is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] node is null");
    return;
  }
  UpdateIsInputConst(*node_ptr);
}

/// update is_input_const
/// @param node
/// @return void
void NodeUtils::UpdateIsInputConst(Node &node) {
  std::vector<bool> is_input_const;
  const uint32_t anchor_num = node.GetAllInDataAnchorsSize();
  for (uint32_t i = 0UL; i < anchor_num; i++) {
    const auto in_anchor = node.GetInDataAnchor(static_cast<int32_t>(i));
    if (in_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    const auto src_node = peer_out_anchor->GetOwnerNodeBarePtr();
    if (src_node == nullptr) {
      is_input_const.push_back(false);
      continue;
    }
    if (IsConst(*(src_node))) {
      is_input_const.push_back(true);
    } else {
      is_input_const.push_back(false);
    }
  }
  if (node.GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node has no opdesc.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node get opdesc is nullptr");
    return;
  }
  node.GetOpDesc()->SetIsInputConst(is_input_const);
}

void NodeUtils::UnlinkAll(const Node &node) {
  for (const auto &anchor : node.GetAllOutAnchors()) {
    anchor->UnlinkAll();
  }
  for (const auto &anchor : node.GetAllInAnchors()) {
    anchor->UnlinkAll();
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AppendInputAnchor(const NodePtr &node,
                                                                                        const uint32_t num) {
  if ((node == nullptr) || (node->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const auto &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetAllInputsSize(); i < num; ++i) {
    if (op_desc->AddInputDesc(data_desc) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "AddInputDesc failed, op:%s", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed, op:%s", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  for (size_t i = node->impl_->in_data_anchors_.size(); i < num; ++i) {
    const auto anchor = ComGraphMakeShared<InDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Current in data anchor is null, make shared_ptr failed.");
      GELOGE(OUT_OF_MEMORY, "[Create][InDataAnchor] Current in data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->impl_->in_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool NodeUtils::ClearInputDesc(const OpDescPtr &op_desc,
                                                                              const uint32_t index) {
  GE_CHK_BOOL_EXEC((op_desc != nullptr) && (op_desc->impl_ != nullptr),
                   REPORT_INNER_ERR_MSG("E18888", "op_desc is nullptr, check invalid");
                   return false, "[Check][Param] op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->impl_->inputs_desc_.size(),
                   REPORT_INNER_ERR_MSG("E18888", "index %u is invalid, out of range(0, %zu).", index,
                                        op_desc->impl_->inputs_desc_.size());
                   return false, "[Check][Param] index %u is invalid, out of range(0, %zu).", index,
                          op_desc->impl_->inputs_desc_.size());

  const auto iter = op_desc->impl_->inputs_desc_.begin() + static_cast<int64_t>(index);
  if (iter < op_desc->impl_->inputs_desc_.end()) {
    (void)op_desc->impl_->inputs_desc_.erase(iter);
  } else {
    GELOGW("[Clear][InputDesc] index %u is out of range [0, %zu).", index, op_desc->impl_->inputs_desc_.size());
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool NodeUtils::ClearOutputDesc(const OpDescPtr &op_desc,
                                                                               const uint32_t index) {
  GE_CHK_BOOL_EXEC((op_desc != nullptr) && (op_desc->impl_ != nullptr),
                   REPORT_INNER_ERR_MSG("E18888", "param op_desc is nullptr, check invalid");
                   return false, "[Check][Param] op_desc is nullptr");
  GE_CHK_BOOL_EXEC(index < op_desc->impl_->outputs_desc_.size(),
                   REPORT_INNER_ERR_MSG("E18888", "index %u is invalid. out of range(0, %zu)", index,
                                        op_desc->impl_->outputs_desc_.size());
                   return false, "[Check][Param] index %u is invalid. out of range(0, %zu)", index,
                          op_desc->impl_->outputs_desc_.size());
  const auto iter = op_desc->impl_->outputs_desc_.begin() + static_cast<int64_t>(index);
  if (iter < op_desc->impl_->outputs_desc_.end()) {
    (void)op_desc->impl_->outputs_desc_.erase(iter);
  } else {
    GELOGW("[Clear][OutputDesc] index %u is out of range [0, %zu).", index, op_desc->impl_->outputs_desc_.size());
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::RemoveInputAnchor(const NodePtr &node,
                                                                                        const uint32_t num) {
  if ((node == nullptr) || (node->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param node is null, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  while (op_desc->GetInputsSize() > num) {
    if (!NodeUtils::ClearInputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }

  const auto input_names = op_desc->GetAllInputName();
  (void)op_desc->UpdateInputName(input_names);
  auto is_input_const = op_desc->GetIsInputConst();
  is_input_const.resize(static_cast<std::size_t>(num));
  op_desc->SetIsInputConst(is_input_const);

  while (node->impl_->in_data_anchors_.size() > num) {
    node->impl_->in_data_anchors_.pop_back();
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::AppendOutputAnchor(const NodePtr &node,
                                                                                         const uint32_t num) {
  if ((node == nullptr) || (node->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "Input node is null, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const GeTensorDesc data_desc(GeShape(), FORMAT_ND, DT_FLOAT);
  const OpDescPtr &op_desc = node->GetOpDesc();
  for (size_t i = op_desc->GetOutputsSize(); i < num; ++i) {
    if (op_desc->AddOutputDesc(data_desc) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Add output desc failed, op:%s", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputDesc] failed, op:%s", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  for (size_t i = node->impl_->out_data_anchors_.size(); i < num; ++i) {
    const auto anchor = ComGraphMakeShared<OutDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Current out data anchor is null, make shared_ptr failed.");
      GELOGE(OUT_OF_MEMORY, "[Create][OutDataAnchor] Current out data anchor is null, make shared_ptr failed.");
      return GRAPH_FAILED;
    }
    node->impl_->out_data_anchors_.push_back(anchor);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus NodeUtils::RemoveOutputAnchor(const NodePtr &node,
                                                                                         const uint32_t num) {
  if ((node == nullptr) || (node->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "Input node is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node is null");
    return GRAPH_FAILED;
  }

  const auto &op_desc = node->GetOpDesc();
  const auto output_names = op_desc->GetAllOutputName();
  while (op_desc->GetOutputsSize() > num) {
    if (!NodeUtils::ClearOutputDesc(op_desc, num)) {
      return GRAPH_FAILED;
    }
  }
  (void)op_desc->UpdateOutputName(output_names);

  while (node->impl_->out_data_anchors_.size() > num) {
    node->impl_->out_data_anchors_.pop_back();
  }

  return GRAPH_SUCCESS;
}

GeTensorDesc NodeUtils::GetOutputDesc(const Node &node, const uint32_t index) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return {};
  }
  return desc->GetOutputDesc(index);
}

graphStatus NodeUtils::GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow) {
  const auto desc = node.GetOpDesc();
  GE_CHECK_NOTNULL(desc);
  // check self
  is_unknow = OpShapeIsUnknown(desc);
  if (is_unknow) {
    return GRAPH_SUCCESS;
  }
  const auto sub_graph_names = desc->GetSubgraphInstanceNames();
  if (sub_graph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    const auto owner_graph = node.GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    // During graph splitting, get parent graph cannot be obtained in some scenarios,
    // but the root graph can be set use the attribute.
    ComputeGraphPtr src_graph = owner_graph->TryGetExtAttr(kPartSrcGraph, ComputeGraphPtr());
    if (src_graph == nullptr) {
      GELOGD("src graph is null, owner graph name is %s", owner_graph->GetName().c_str());
      src_graph = owner_graph;
    }
    GELOGD("src graph is %s, owner graph name is %s", src_graph->GetName().c_str(), owner_graph->GetName().c_str());
    const auto root_graph = GraphUtils::FindRootGraph(src_graph);
    if (root_graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "node:%s has no root graph.", node.GetName().c_str());
      GE_LOGE("[Get][Graph] Node %s gets null root graph", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    for (auto &sub_graph_name : sub_graph_names) {
      const auto sub_graph = root_graph->GetSubgraph(sub_graph_name);
      if (sub_graph == nullptr) {
        GELOGD("sub graph %s is empty", sub_graph_name.c_str());
        continue;
      }
      for (const auto &node_ptr : sub_graph->GetDirectNode()) {
        const auto status = GetNodeUnknownShapeStatus(*node_ptr, is_unknow);
        if (status != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E18888", "GetNodeUnknownShapeStatus failed, node:%s, status:%u",
                               node_ptr->GetName().c_str(), status);
          GE_LOGE("[Get][NodeUnknownShapeStatus] failed! node:%s, status:%u", node_ptr->GetName().c_str(), status);
          return status;
        }
        if (is_unknow) {
          return GRAPH_SUCCESS;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

std::string NodeUtils::GetNodeType(const Node &node) {
  if (node.GetType() != FRAMEWORKOP) {
    return node.GetType();
  }

  std::string type;
  const std::string *type_str = AttrUtils::GetStr(node.GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE);
  if (type_str != nullptr) {
    type = *type_str;
  }
  return type;
}

std::string NodeUtils::GetNodeType(const NodePtr &node) {
  return (node == nullptr) ? "" : GetNodeType(*node);
}

graphStatus NodeUtils::GetDirectSubgraphs(const NodePtr &node, std::vector<ComputeGraphPtr> &subgraphs) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "node or op_desc is null");
    GELOGE(GRAPH_FAILED, "[Check][Param] node or op_desc is null");
    return GRAPH_FAILED;
  }

  const auto &root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to find root graph from node %s ", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] Failed to find root graph from node %s ", node->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (const auto &graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
    const auto &graph = root_graph->GetSubgraph(graph_name);
    if (graph == nullptr) {
      GELOGW("[Get][Subgraph] subgraph %s of node %s is null", graph_name.c_str(), node->GetName().c_str());
      continue;
    }
    subgraphs.emplace_back(graph);
  }

  return GRAPH_SUCCESS;
}

ComputeGraphPtr NodeUtils::GetSubgraph(const Node &node, const uint32_t index) {
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }
  const auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    return nullptr;
  }
  return root_graph->GetSubgraph(op_desc->GetSubgraphInstanceName(index));
}

graphStatus NodeUtils::SetSubgraph(Node &node, const uint32_t index, const ComputeGraphPtr &subgraph) {
  if (subgraph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to set subgraph to node %s index %u, null subgraph", node.GetName().c_str(),
                         index);
    GE_LOGE("[Check][Param] Failed to set subgraph to node %s index %u, null subgraph", node.GetName().c_str(), index);
    return GRAPH_PARAM_INVALID;
  }
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto root_graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to add subgraph to node %s, null root graph", node.GetName().c_str());
    GE_LOGE("[Get][Graph] Failed to add subgraph to node %s, null root graph", node.GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  const auto ret = op_desc->SetSubgraphInstanceName(index, subgraph->GetName());
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to set subgraph to node %s index %u", node.GetName().c_str(), index);
    GE_LOGE("[Set][Name] Failed to set subgraph to node %s index %u", node.GetName().c_str(), index);
    return ret;
  }
  subgraph->SetParentNode(node.shared_from_this());
  subgraph->SetParentGraph(node.GetOwnerComputeGraph());
  return root_graph->AddSubgraph(subgraph);
}
graphStatus NodeUtils::AddSubgraph(Node &node, const std::string &subgraph_ir_name, const ComputeGraphPtr &subgraph) {
  GE_ASSERT_NOTNULL(subgraph);
  auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  // Will report inner warning if the Op is created using REG_OP format
  // because during REG_OP it has already registered subgraph IR name
  (void)op_desc->AddSubgraphName(subgraph_ir_name);
  auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  const auto &iter = subgraph_names_to_index.find(subgraph_ir_name);
  GE_ASSERT_TRUE(iter != subgraph_names_to_index.cend());

  return SetSubgraph(node, iter->second, subgraph);
}
graphStatus NodeUtils::AddSubgraph(const NodePtr &node_ptr, const std::string &subgraph_ir_name,
                                   const ComputeGraphPtr &subgraph) {
  GE_ASSERT_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto subgraph_ir_type = op_desc->GetSubgraphTypeByIrName(subgraph_ir_name);
  if (subgraph_ir_type == kSubgraphTypeEnd) {
    op_desc->RegisterSubgraphIrName(subgraph_ir_name, kStatic);
  } else {
    GE_ASSERT_EQ(kStatic, subgraph_ir_type);
  }
  auto &node = *node_ptr.get();
  return AddSubgraph(node, subgraph_ir_name, subgraph);
}
graphStatus NodeUtils::AddSubgraphs(const NodePtr &node_ptr, const std::string &subgraph_ir_name,
                                    const std::vector<ComputeGraphPtr> &subgraphs) {
  GE_ASSERT_NOTNULL(node_ptr);
  auto op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto subgraph_ir_type = op_desc->GetSubgraphTypeByIrName(subgraph_ir_name);
  if (subgraph_ir_type == kSubgraphTypeEnd) {
    op_desc->RegisterSubgraphIrName(subgraph_ir_name, kDynamic);
  } else {
    GE_ASSERT_EQ(kDynamic, subgraph_ir_type);
  }
  auto &node = *node_ptr.get();
  for (int64_t i = 0U; i < static_cast<int64_t>(subgraphs.size()); ++i) {
    const auto &subgraph = subgraphs[i];
    GE_ASSERT_SUCCESS(AddSubgraph(node, GenDynamicSubgraphName(subgraph_ir_name, i), subgraph));
  }
  return GRAPH_SUCCESS;
}
std::string NodeUtils::GenDynamicSubgraphName(const std::string &subgraph_ir_name, int64_t index) {
  return subgraph_ir_name + std::to_string(index);
}

/// Check if node is input of subgraph
/// @param [in] node
/// @return bool
bool NodeUtils::IsSubgraphInput(const NodePtr &node) {
  return IsSubgraphInput(node.get());
}

bool NodeUtils::IsSubgraphInput(const Node *const node) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr) ||
      (node->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr() == nullptr)) {
    return false;
  }

  const auto parent_op_desc = node->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr()->GetOpDescBarePtr();
  if (parent_op_desc == nullptr) {
    return false;
  }

  // dynamic shape unknown graph false
  // dynamic shape known graph with functional subgraph maybe true
  bool is_forced_unknown = false;
  if (AttrUtils::GetBool(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_forced_unknown) && is_forced_unknown) {
    if (node->GetOwnerComputeGraphBarePtr()->GetParentGraphBarePtr()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraphBarePtr()
              ->GetParentNodeBarePtr()
              ->GetOwnerComputeGraphBarePtr()
              ->GetParentNodeBarePtr() == nullptr) {
        return false;
      }
    }
  }

  return node->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

/// Check if node is output of subgraph
/// @param [in] node
/// @return bool
bool NodeUtils::IsSubgraphOutput(const NodePtr &node) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr) ||
      (node->GetOwnerComputeGraph()->GetParentNode() == nullptr) || (node->GetType() != NETOUTPUT)) {
    return false;
  }

  const auto parent_op_desc = node->GetOwnerComputeGraph()->GetParentNode()->GetOpDesc();
  if (parent_op_desc == nullptr) {
    return false;
  }

  bool is_forced_unknown = false;
  if (AttrUtils::GetBool(parent_op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_forced_unknown) && is_forced_unknown) {
    if (node->GetOwnerComputeGraph()->GetParentGraph()->GetGraphUnknownFlag()) {
      return false;
    } else {
      if (node->GetOwnerComputeGraph()->GetParentNode()->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
        return false;
      }
    }
  }

  for (const auto &tensor : node->GetOpDesc()->GetAllInputsDescPtr()) {
    if (AttrUtils::HasAttr(tensor, ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }

  return false;
}

/// @brief Get subgraph original input node.
/// @param [in] node
/// @return Node
NodePtr NodeUtils::GetParentInput(const Node &node) {
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node.GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return nullptr;
  }

  // Subgraph Data Node, check for constant input.
  const ComputeGraphPtr &graph = node.GetOwnerComputeGraph();
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr);

  const Node *parent_node = graph->GetParentNodeBarePtr();
  if (parent_node == nullptr) {
    GELOGW("Node {%s %s} has attr %s but has no parent node.", node.GetNamePtr(), node.GetTypePtr(),
           ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return nullptr;
  }

  const InDataAnchorPtr &in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
  GE_CHECK_NOTNULL_EXEC(in_anchor, return nullptr);

  const OutDataAnchorPtr &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL_EXEC(peer_out_anchor, return nullptr);

  auto peer_node = peer_out_anchor->GetOwnerNode();
  if (peer_node->GetType() == DATA) {
    if (peer_node->GetOpDesc() == nullptr) {
      return nullptr;
    }
    if (peer_node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
      return GetParentInput(peer_node);
    }
  }
  return peer_node;
}

NodePtr NodeUtils::GetParentInput(const NodePtr &node) {
  return (node == nullptr) ? node : GetParentInput(*node);
}

bool NodeUtils::IsWrapperNode(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  return !op_desc->GetSubgraphInstanceNames().empty();
}

NodePtr NodeUtils::GetParentNode(const Node &node) {
  const ComputeGraphPtr &graph = node.GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph, "Owner compute graph is null for DATA node: %s", node.GetNamePtr());
  NodePtr parent_node = graph->GetParentNode();
  return parent_node;
}

NodePtr NodeUtils::GetParentNode(const NodePtr &node) {
  return (node == nullptr) ? node : GetParentNode(*node);
}

InDataAnchorPtr NodeUtils::GetParentInDataAnchor(const NodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }

  // 1. 获取映射索引
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return nullptr;
  }

  // 2. 获取父图 Wrapper 节点
  const auto &graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    return nullptr;
  }

  const auto parent_node = graph->GetParentNode();
  if (parent_node == nullptr) {
    // 可能是根图，没有父节点
    return nullptr;
  }

  // 3. 获取 Wrapper 对应的输入锚点
  const auto parent_in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));

  return parent_in_anchor;
}

NodeToOutAnchor NodeUtils::GetParentInputAndAnchor(const NodePtr &node) {
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return {nullptr, nullptr};
  }

  // Subgraph Data Node, check for constant input.
  const ComputeGraphPtr &graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    return {nullptr, nullptr};
  }

  const Node *parent_node = graph->GetParentNodeBarePtr();
  if (parent_node == nullptr) {
    return {nullptr, nullptr};
  }

  const InDataAnchorPtr &in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
  if (in_anchor == nullptr) {
    return {nullptr, nullptr};
  }

  const OutDataAnchorPtr &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return {nullptr, nullptr};
  }

  return std::make_pair(peer_out_anchor->GetOwnerNode(), peer_out_anchor);
}

NodeToOutAnchor NodeUtils::GetParentInputAndAnchorCrossSubgraph(const NodePtr &node) {
  NodeToOutAnchor node_to_out_anchor = {nullptr, nullptr};
  std::stack<NodePtr> s;
  s.push(node);
  while (!s.empty()) {
    auto n = s.top();
    s.pop();
    node_to_out_anchor = GetParentInputAndAnchor(n);
    auto peer_node = node_to_out_anchor.first;
    if ((peer_node == nullptr) || (peer_node->GetType() != DATA)) {
      continue;
    }

    if ((peer_node->GetOpDesc() != nullptr) && peer_node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX)) {
      s.push(peer_node);
    }
  }
  return node_to_out_anchor;
}

/// @brief Get is dynamic shape graph from node.
/// @param [in] node
/// @return bool
bool NodeUtils::IsDynamicShape(const Node &node) {
  const auto graph = GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
  if (graph == nullptr) {
    return false;
  }

  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  return is_dynamic_shape;
}

bool NodeUtils::IsDynamicShape(const NodePtr &node) {
  return (node == nullptr) ? false : IsDynamicShape(*node);
}

/// @brief Check is varying_input for while node
/// @param [in] node: Data node for subgraph
/// @return bool
bool NodeUtils::IsWhileVaryingInput(const af::NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->GetType() != DATA) {
    return false;  // not input_node for subgraph
  }

  const Node *parent_node = node->GetOwnerComputeGraph()->GetParentNodeBarePtr();
  if (parent_node == nullptr) {
    return false;  // root graph
  }

  if (kWhileOpTypes.count(parent_node->GetType()) == 0U) {
    return false;  // not input_node for while subgraph
  }

  uint32_t index_i = 0U;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index_i)) {
    GELOGW("[Check][Attr] Node %s has no attr PARENT_NODE_INDEX.", node->GetName().c_str());
    return false;
  }
  bool varying_flag = true;
  for (const auto &item : node->GetOutDataNodesAndAnchors()) {
    if (item.first->GetType() != NETOUTPUT) {
      continue;
    }
    const OpDescPtr op_desc = item.first->GetOpDesc();
    uint32_t index_o = 0U;
    if ((op_desc == nullptr) || (!AttrUtils::GetInt(op_desc->GetInputDesc(static_cast<uint32_t>(item.second->GetIdx())),
                                                    ATTR_NAME_PARENT_NODE_INDEX, index_o))) {
      continue;  // input for while-cond subgraph
    }
    if (index_i != index_o) {
      continue;  // varying input for while-body subgraph
    }
    varying_flag = false;
    break;
  }
  return varying_flag;
}

/// @brief Get subgraph input is constant.
/// @param [in] node
/// @param [out] string
/// @return bool
bool NodeUtils::GetConstOpType(const NodePtr &node, std::string &type) {
  if (node == nullptr) {
    return false;
  }

  const auto node_type = node->GetType();
  if ((node_type == CONSTANT) || (node_type == CONSTANTOP) || (node_type == FILECONSTANT)) {
    type = node->GetType();
    return true;
  }

  if (node_type != DATA) {
    return false;  // not subgraph input node
  }

  const auto &parent = GetParentInput(node);
  return GetConstOpType(parent, type);
}

/// @brief Remove node-related subgraphs, including subgraphs of nodes in the subgraph.
/// @param [in] node
/// @return return GRAPH_SUCCESS if remove successfully, other for failed.
graphStatus NodeUtils::RemoveSubgraphsOnNode(const NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    return GRAPH_SUCCESS;
  } else {
    const auto owner_graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(owner_graph);
    const auto root_graph = GraphUtils::FindRootGraph(owner_graph);
    GE_CHECK_NOTNULL(root_graph);

    std::set<std::string> subgraph_to_remove;
    for (auto &subgraph_name : subgraph_names) {
      std::deque<std::string> queue;
      queue.push_back(subgraph_name);
      (void)subgraph_to_remove.insert(subgraph_name);
      op_desc->RemoveSubgraphInstanceName(subgraph_name);
      while (!queue.empty()) {
        const auto graph_name = queue.front();
        queue.pop_front();

        const auto subgraph = root_graph->GetSubgraph(graph_name);
        GE_CHECK_NOTNULL(subgraph);
        for (const auto &sub_node : subgraph->GetDirectNode()) {
          const auto sub_op_desc = sub_node->GetOpDesc();
          GE_CHECK_NOTNULL(sub_op_desc);
          const auto sub_names = sub_op_desc->GetSubgraphInstanceNames();
          // Subgraph and all nodes in it will be removed later,
          // no need to remove 'SubgraphInstanceName' in op desc here.
          for (auto &name : sub_names) {
            if (subgraph_to_remove.insert(name).second) {
              queue.push_back(name);
            }
          }
        }
      }
    }
    // Remove subgraph from root_graph
    for (const auto &name : subgraph_to_remove) {
      GELOGD("Remove subgraph:%s.", name.c_str());
      root_graph->RemoveSubgraph(name);
    }
  }

  return GRAPH_SUCCESS;
}

std::vector<NodePtr> NodeUtils::GetSubgraphDataNodesByIndex(const Node &node, const int32_t index) {
  std::vector<NodePtr> in_data_node_vec;
  const auto op_desc = node.GetOpDescBarePtr();
  GE_CHECK_NOTNULL_EXEC(op_desc, return in_data_node_vec);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    return in_data_node_vec;
  }
  const auto compute_graph = FindRootGraph(node);
  for (const std::string &instance_name : subgraph_names) {
    const auto subgraph = compute_graph->GetSubgraph(instance_name);
    if (subgraph == nullptr) {
      continue;
    }
    for (const auto &node_in_subgraph : subgraph->GetDirectNode()) {
      if (IsTypeEqual(node_in_subgraph, DATA)) {
        int32_t parent_index = -1;
        (void)AttrUtils::GetInt(node_in_subgraph->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index);
        if (parent_index == index) {
          in_data_node_vec.emplace_back(node_in_subgraph);
          break;
        }
      }
    }
  }
  return in_data_node_vec;
}
/// @brief Get subgraph input data node by index.
/// @param [in] node
/// @return Node
std::vector<NodePtr> NodeUtils::GetSubgraphOutputNodes(const Node &node) {
  std::vector<NodePtr> out_data_node_vec;
  const auto op_desc = node.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return out_data_node_vec);
  const auto subgraph_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_names.empty()) {
    GELOGI("Node %s is single node without sub graph.", node.GetName().c_str());
    return out_data_node_vec;
  }
  const auto compute_graph = FindRootGraph(node);
  for (const std::string &instance_name : subgraph_names) {
    const auto subgraph = compute_graph->GetSubgraph(instance_name);
    if (subgraph == nullptr) {
      continue;
    }
    out_data_node_vec.emplace_back(subgraph->GetOrUpdateNetOutputNode());
  }
  return out_data_node_vec;
}

NodePtr NodeUtils::GetInDataNodeByIndex(const Node &node, const int32_t index) {
  if (node.GetInDataAnchor(index) == nullptr) {
    return nullptr;
  }
  if (node.GetInDataAnchor(index)->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return node.GetInDataAnchor(index)->GetPeerOutAnchor()->GetOwnerNode();
}

std::vector<std::pair<InDataAnchorPtr, NodePtr>> NodeUtils::GetOutDataNodesWithAnchorByIndex(const Node &node,
                                                                                             const int32_t index) {
  std::vector<std::pair<InDataAnchorPtr, NodePtr>> out_data_nodes;
  const auto out_data_anchor = node.GetOutDataAnchor(index);
  if (out_data_anchor == nullptr) {
    return out_data_nodes;
  }

  for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (peer_in_anchor->GetOwnerNodeBarePtr() == nullptr) {
      continue;
    }
    out_data_nodes.emplace_back(peer_in_anchor, peer_in_anchor->GetOwnerNode());
  }
  return out_data_nodes;
}

std::string NodeUtils::GetInConstNodeTypeCrossSubgraph(const NodePtr &node) {
  const NodePtr input_node = GetInNodeCrossSubgraph(node);
  if (input_node == nullptr) {
    return "";
  }

  return input_node->GetType();
}

NodePtr NodeUtils::GetInNodeCrossSubgraph(const NodePtr &node) {
  NodePtr input_node = node;
  while (input_node != nullptr) {
    if (input_node->GetType() != DATA) {
      return input_node;
    }

    const auto owner_graph = input_node->GetOwnerComputeGraph();
    const auto parent_node = owner_graph->GetParentNodeBarePtr();
    if ((parent_node == nullptr) || (kWhileOpTypes.count(parent_node->GetType()) > 0UL)) {
      return input_node;  // not in subgraph or while subgraph.
    }

    input_node = GetParentInput(input_node);
  }

  return input_node;
}

NodePtr NodeUtils::CreatNodeWithoutGraph(const OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The OpDesc ptr should not be null.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The OpDesc ptr should not be null.");
    return nullptr;
  }
  auto node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op_desc, nullptr));
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create node failed.");
    GELOGE(GRAPH_FAILED, "[Create][Node] node_ptr is NULL!");
    return nullptr;
  }
  return node_ptr;
}

graphStatus NodeUtils::GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node) {
  int32_t peer_out_anchor_index = kInvalidIndex;
  return GetInNodeCrossPartionedCallNode(node, index, peer_node, peer_out_anchor_index);
}

graphStatus NodeUtils::GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node,
                                                       int32_t &peer_out_anchor_index) {
  GE_CHECK_NOTNULL(node);
  peer_out_anchor_index = kInvalidIndex;
  if ((node->GetAllInDataAnchorsSize() <= index) && (node->GetType() != DATA)) {
    return GRAPH_FAILED;
  }
  GELOGD("in node:%s index:%d", node->GetName().c_str(), index);
  peer_node = (node->GetType() == DATA) ? node : GetInDataNodeByIndex(*node, static_cast<int32_t>(index));
  if (peer_node == nullptr) {
    // A->B
    // Assuming A and B belongs to different engine, during graph partition, A will be set to B's extra attr as
    // parent node. when FE get parent node A from B, check A's in_anchor peer_out_anchor is null.
    return GRAPH_SUCCESS;
  }

  if (node->GetType() != DATA) {
    const auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(index));
    GE_CHECK_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    peer_out_anchor_index = peer_out_anchor->GetIdx();
  }
  while (!IsComputableOp(peer_node)) {
    if (peer_node->GetType() == DATA) {
      const auto parent_node_2_anchor = GetParentInputAndAnchor(peer_node);
      if ((parent_node_2_anchor.first == nullptr) || (parent_node_2_anchor.second == nullptr)) {
        GELOGW("Returned peer_out_node is nullptr because no attr[%s] on DATA[%s] node!", kRefIndex,
               peer_node->GetName().c_str());
        peer_node = nullptr;
        return GRAPH_SUCCESS;
      }
      peer_node = parent_node_2_anchor.first;
      peer_out_anchor_index = parent_node_2_anchor.second->GetIdx();
      continue;
    }

    if (peer_node->GetType() != PARTITIONEDCALL) {
      if (peer_node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
        GELOGI("Node [%s] type [%s], real peer in node [%s] type[%s].", node->GetName().c_str(),
               node->GetType().c_str(), peer_node->GetName().c_str(), peer_node->GetType().c_str());
        return GRAPH_SUCCESS;
      }
      // other subgraph(if,while,case) currently not support, return node and warn
      GELOGW("Node [%s] type [%s], real peer in node [%s] type[%s] has subgraph. Current operation is not supported.",
             node->GetName().c_str(), node->GetType().c_str(), peer_node->GetName().c_str(),
             peer_node->GetType().c_str());

      return GRAPH_SUCCESS;
    }
    // if peer node is PartionedCall, return owner graph's correspond node
    const auto sub_graph = GetSubgraph(*peer_node, 0U);
    if (sub_graph == nullptr) {
      GELOGW("SubGraph of node %s index 0 is null. Null is invalid.", peer_node->GetName().c_str());
      return ge::PARAM_INVALID;
    }
    const auto sub_graph_netoutput = sub_graph->GetOrUpdateNetOutputNode();
    GE_CHECK_NOTNULL(sub_graph_netoutput);

    for (const auto &in_data_anchor : sub_graph_netoutput->GetAllInDataAnchorsPtr()) {
      const auto in_desc =
          sub_graph_netoutput->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
      GE_CHECK_NOTNULL(in_desc);
      int32_t ref_o = 0;
      if (!AttrUtils::GetInt(in_desc, kRefIndex, ref_o)) {
        return GRAPH_FAILED;
      }
      if (peer_out_anchor_index != ref_o) {
        continue;
      }
      peer_node = NodeUtils::GetInDataNodeByIndex(*sub_graph_netoutput, in_data_anchor->GetIdx());
      GE_CHECK_NOTNULL(peer_node);
      GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor());
      peer_out_anchor_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
      GELOGD("in node[%s] peer_node[%s] type[%s] out anchor index[%d].", node->GetName().c_str(),
             peer_node->GetName().c_str(), peer_node->GetType().c_str(), peer_out_anchor_index);
      break;
    }
  }
  return GRAPH_SUCCESS;
}

bool NodeUtils::IsNodeInRootGraph(const NodePtr &node) {
  GE_ASSERT_NOTNULL(node);

  const auto owner_graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);

  const auto root_graph = af::GraphUtils::FindRootGraph(owner_graph);
  GE_ASSERT_NOTNULL(root_graph);

  if (owner_graph == root_graph) {
    GELOGD("Node [%s] is in root graph [%s].", node->GetName().c_str(), root_graph->GetName().c_str());
    return true;
  }
  GELOGD("Node [%s] is in sub graph [%s], root graph is [%s].", node->GetName().c_str(), owner_graph->GetName().c_str(),
         root_graph->GetName().c_str());
  return false;
}

bool NodeUtils::IsMultiBranchControlFlowOp(const NodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const std::string &type = node->GetType();
  return kMultiBranchControlFlowOps.count(type) > 0;
}

graphStatus NodeUtils::SetNodeParallelGroup(Node &node, const char_t *const group_name) {
  if (group_name == nullptr) {
    GE_LOGE("[Check][Parameter]Get nullptr when set parallel group on node:%s", node.GetName().c_str());
    REPORT_INNER_ERR_MSG("E18888", "Get nullptr when set parallel group on node:%s", node.GetName().c_str());
    return GRAPH_FAILED;
  }
  const std::string *current_group = AttrUtils::GetStr(node.GetOpDesc(), ATTR_NAME_PARALLEL_GROUP);
  const std::string new_group(group_name);
  if (current_group != nullptr) {
    if (new_group != *current_group) {
      GE_LOGE("[Compare][Attr]Failed to set parallel group name %s on node %s, group conflict with existing %s",
              new_group.c_str(), node.GetName().c_str(), group_name);
      REPORT_INNER_ERR_MSG("E18888", "Failed to set parallel group name %s on node %s, group conflict with existing %s",
                           new_group.c_str(), node.GetName().c_str(), group_name);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  if (!AttrUtils::SetStr(node.GetOpDesc(), ATTR_NAME_PARALLEL_GROUP, new_group)) {
    GE_LOGE("[Set][Attr] Failed to set parallel group name %s on node %s", group_name, node.GetName().c_str());
    REPORT_INNER_ERR_MSG("E18888", "Failed to set parallel group name %s on node %s", group_name,
                         node.GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::UpdateInputOriginalShapeAndShape(const Node &node, const uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto input_desc = desc->MutableInputDesc(index);
  if (input_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  input_desc->SetShape(shape);
  input_desc->SetOriginShape(shape);
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::UpdateOutputOriginalShapeAndShape(const Node &node, const uint32_t index, const GeShape &shape) {
  const auto desc = node.GetOpDesc();
  if (desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  const auto output_desc = desc->MutableOutputDesc(index);
  if (output_desc == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  output_desc->SetShape(shape);
  output_desc->SetOriginShape(shape);
  return GRAPH_SUCCESS;
}
std::pair<NodePtr, OutDataAnchorPtr> NodeUtils::GetInDataNodeAndAnchorByIndex(const Node &node, const int32_t index) {
  const auto dst_anchor = node.GetInDataAnchor(index);
  if (dst_anchor == nullptr) {
    GE_LOGE("Failed to get in data anchor from index %d for node %s", index, node.GetName().c_str());
    return {nullptr, nullptr};
  }
  auto src_anchor = dst_anchor->GetPeerOutAnchor();
  if (src_anchor == nullptr) {
    GE_LOGE("Failed to get peer out data anchor from index %i for node %s", index, node.GetName().c_str());
    return {nullptr, nullptr};
  }
  auto src_node = src_anchor->GetOwnerNode();
  if (src_node == nullptr) {
    GE_LOGE("Failed to get in data node from index %d for node %s", index, node.GetName().c_str());
    return {nullptr, nullptr};
  }
  return {src_node, src_anchor};
}

bool NodeUtils::IsDtResourceNode(const NodePtr &node) {
  for (const auto &in_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
    if (in_desc->GetDataType() == DT_RESOURCE) {
      return true;
    }
  }
  for (const auto &out_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
    if (out_desc->GetDataType() == DT_RESOURCE) {
      return true;
    }
  }
  return false;
}

bool NodeUtils::IsLikeAtomicClean(const NodePtr &node) {
  const auto node_type = NodeUtils::GetNodeType(node);
  return (node_type == ATOMICADDRCLEAN) || (node_type == MEMSET);
}

bool NodeUtils::IsIdentityUsefulForRWControl(const NodePtr &node_ptr) {
  GE_ASSERT_NOTNULL(node_ptr);
  if (!(OpTypeUtils::IsIdentityLikeNode(node_ptr->GetType()))) {
    return false;
  }
  Node &node = *(node_ptr.get());
  if (node.GetOutControlNodesSize() == 0U) {
    return false;
  }
  if (node.GetInDataNodesSize() != 1U) {
    return false;
  }
  if (node.GetOutDataNodesSize() == 0U) {
    return false;
  }
  const auto out_data_node = node.GetOutDataNodes().at(0U);
  GE_ASSERT_NOTNULL(node.GetInDataAnchor(0U));
  const auto &in_node_out_data_anchor = node.GetInDataAnchor(0U)->GetPeerOutAnchor();
  if (in_node_out_data_anchor == nullptr) {
    return false;
  }
  const auto in_node_ptr = in_node_out_data_anchor->GetOwnerNodeBarePtr();  // in_node_ptr must not be null
  for (const auto out_control_node_in_control_anchor : node.GetOutControlAnchor()->GetPeerInControlAnchorsPtr()) {
    const auto out_control_node =
        out_control_node_in_control_anchor->GetOwnerNodeBarePtr();  // out_control node must not be null
    for (const auto out_control_node_in_data_anchor : out_control_node->GetAllInDataAnchorsPtr()) {
      // out_control_node_in_data_anchor must not be null
      // out_control_node_in_data_anchor->GetOwnerNodeBarePtr() must not be null
      if (in_node_out_data_anchor->IsLinkedWith(out_control_node_in_data_anchor->shared_from_this())) {
        if ((OpTypeUtils::IsVarLikeNode(in_node_ptr->GetType())) &&
            (OpTypeUtils::IsAssignLikeNode(out_control_node->GetType()))) {
          GELOGD(
              "Node[%s %s] is useful for control relation, keep this node to ensure out data node[%s %s] read "
              "in_data_node [%s %s] firstly, then out control node [%s %s] write in_data_node",
              node.GetName().c_str(), node.GetType().c_str(), out_data_node->GetName().c_str(),
              out_data_node->GetType().c_str(), in_node_ptr->GetName().c_str(), in_node_ptr->GetType().c_str(),
              out_control_node->GetName().c_str(), out_control_node->GetType().c_str());
          return true;
        }
      }
    }
  }
  return false;
}

ComputeGraphPtr NodeUtils::FindRootGraph(const Node &node) {
  return GraphUtils::FindRootGraph(node.GetOwnerComputeGraph());
}

std::vector<NodePtr> NodeUtils::GetOutControlNodes(const Node &node, const NodeFilter &node_filter) {
  std::vector<NodePtr> out_ctrl_nodes;
  const auto &out_control = node.GetOutControlAnchor();
  if (out_control == nullptr) {
    return out_ctrl_nodes;
  }
  out_ctrl_nodes.reserve(node.GetOutControlNodesSize());
  for (const auto &in_anchor : out_control->GetPeerAnchorsPtr()) {
    const auto &peer_node = in_anchor->GetOwnerNode();
    if ((node_filter == nullptr) || node_filter(*peer_node)) {
      out_ctrl_nodes.push_back(peer_node);
    }
  }
  return out_ctrl_nodes;
}

std::vector<NodePtr> NodeUtils::GetOutDataNodes(const Node &node, const NodeFilter &node_filter) {
  std::vector<NodePtr> out_data_nodes;
  for (const auto &out_anchor : node.impl_->out_data_anchors_) {
    GE_ASSERT_NOTNULL(out_anchor);
    for (const auto &in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      const auto out_data_node = in_anchor->GetOwnerNode();
      if ((node_filter == nullptr) || node_filter(*out_data_node)) {
        out_data_nodes.push_back(out_data_node);
      }
    }
  }
  return out_data_nodes;
}

std::vector<NodePtr> NodeUtils::GetInControlNodes(const Node &node, const NodeFilter &node_filter) {
  std::vector<NodePtr> in_ctrl_nodes;
  const auto &in_control = node.GetInControlAnchor();
  if (in_control == nullptr) {
    return in_ctrl_nodes;
  }
  in_ctrl_nodes.reserve(node.GetInControlNodesSize());
  for (const auto out_anchor : in_control->GetPeerAnchorsPtr()) {
    const auto &peer_node = out_anchor->GetOwnerNode();
    if ((node_filter == nullptr) || node_filter(*peer_node)) {
      in_ctrl_nodes.push_back(peer_node);
    }
  }
  return in_ctrl_nodes;
}

std::vector<NodePtr> NodeUtils::GetInDataNodes(const Node &node, const NodeFilter &node_filter) {
  std::vector<NodePtr> in_data_nodes;
  in_data_nodes.reserve(node.GetInDataNodesSize());
  for (const auto &in_anchor : node.impl_->in_data_anchors_) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto anchor_ptr = in_anchor->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    const auto in_node = anchor_ptr->GetOwnerNode();
    if ((node_filter == nullptr) || node_filter(*in_node)) {
      in_data_nodes.push_back(in_node);
    }
  }
  return in_data_nodes;
}

graphStatus NodeUtils::TryGetWeightByPlaceHolderNode(const NodePtr &node_ptr, ConstGeTensorPtr &ge_tensor) {
  if (ge_tensor != nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "ge_tensor already has value");
    return GRAPH_PARAM_INVALID;
  }
  if (node_ptr->GetType() != PLACEHOLDER) {
    return GRAPH_SUCCESS;
  }
  const auto &op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  // In some case, Placeholder operator may has it's peer const node's weight
  if (ConstantUtils::GetWeight(op_desc, 0U, ge_tensor)) {
    GELOGI("op [%s %s] has direct weight attr", op_desc->GetType().c_str(), op_desc->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  NodePtr parent_node = nullptr;
  parent_node = op_desc->TryGetExtAttr("parentNode", parent_node);
  if (parent_node == nullptr) {
    GELOGI("op [%s %s] get not any ext node attr", op_desc->GetType().c_str(), op_desc->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  const auto &parent_op_desc = parent_node->GetOpDesc();
  GE_CHECK_NOTNULL(parent_op_desc);
  if (ConstantUtils::IsConstant(parent_op_desc)) {
    if (ConstantUtils::GetWeight(parent_op_desc, 0U, ge_tensor)) {
      GELOGI("op [%s %s] has indirect weight attr from other op [%s %s]", op_desc->GetType().c_str(),
             op_desc->GetName().c_str(), parent_op_desc->GetType().c_str(), parent_op_desc->GetName().c_str());
      return GRAPH_SUCCESS;
    }
  }
  if (parent_op_desc->GetType() == DATA) {
    return TryGetWeightByDataNode(parent_node, ge_tensor);
  }
  GELOGI("op [%s %s] get not any weight attr", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus NodeUtils::TryGetWeightByDataNode(const NodePtr &node_ptr, ConstGeTensorPtr &ge_tensor) {
  if (ge_tensor != nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "ge_tensor already has value");
    return GRAPH_PARAM_INVALID;
  }
  if (node_ptr->GetType() != DATA) {
    return GRAPH_SUCCESS;
  }
  const auto &op_desc = node_ptr->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  // the input const data should not be obtained,
  // as the input will change during multiple rounds of infershape for while
  if ((node_ptr->GetOwnerComputeGraphBarePtr() != nullptr) &&
      (node_ptr->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr() != nullptr) &&
      (kWhileOpTypes.count(node_ptr->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr()->GetType()) > 0U)) {
    GELOGI(
        "The value of a const node should not be obtained, when the const node is outside a while node, "
        "while node name: %s",
        node_ptr->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr()->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  NodePtr real_parent_node = nullptr;
  (void)NodeUtils::GetInNodeCrossPartionedCallNode(node_ptr, 0U, real_parent_node);
  if ((real_parent_node != nullptr) && (ConstantUtils::IsConstant(real_parent_node->GetOpDesc()))) {
    GELOGI("Get in real parent node:[%s %s] for node:[%s %s]", real_parent_node->GetName().c_str(),
           real_parent_node->GetType().c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    if (ConstantUtils::IsConstant(real_parent_node)) {
      if (ConstantUtils::GetWeight(real_parent_node->GetOpDesc(), 0U, ge_tensor)) {
        GELOGI("op [%s %s] has indirect weight attr from other op [%s %s]", op_desc->GetType().c_str(),
               op_desc->GetName().c_str(), real_parent_node->GetType().c_str(), real_parent_node->GetName().c_str());
        return GRAPH_SUCCESS;
      }
    }
  }
  GELOGI("op [%s %s] get not any weight attr", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  return GRAPH_SUCCESS;
}
bool NodeUtils::IsNameEqual(const NodePtr &node, const ge::char_t *const name) {
  return strcmp(node->GetNamePtr(), name) == 0;
}
bool NodeUtils::IsTypeEqual(const NodePtr &node, const ge::char_t *const type) {
  return strcmp(node->GetTypePtr(), type) == 0;
}

NodePtr NodeUtils::GetNodeWithMinimalId(const std::vector<NodePtr> &nodes) {
  NodePtr min_id_node = nullptr;
  int64_t min_id = -1;
  for (const auto &node : nodes) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto id = op_desc->GetId();
    if ((min_id == -1) || (id < min_id)) {
      min_id = id;
      min_id_node = node;
    }
  }
  return min_id_node;
}
}  // namespace af
