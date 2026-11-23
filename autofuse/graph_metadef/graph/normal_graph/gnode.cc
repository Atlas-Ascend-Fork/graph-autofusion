/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/gnode_af.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/anchor.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"

#define NODE_ATTR_GET_IMP(ArgType)                                                       \
  graphStatus GNode::GetAttr(const AscendString &name, ArgType &attr_value) const {      \
    const char_t *const ascend_name = name.GetString();                                  \
    if (std::string(ascend_name).empty()) {                                              \
      REPORT_INNER_ERR_MSG("E18888", "ascend std::string error.");                       \
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetAttr: ascend std::string error.");  \
      return GRAPH_PARAM_INVALID;                                                        \
    }                                                                                    \
                                                                                         \
    if (impl_ == nullptr) {                                                              \
      REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");            \
      GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: node impl is nullptr.");             \
      return GRAPH_FAILED;                                                               \
    }                                                                                    \
                                                                                         \
    const std::shared_ptr<Node> node_ptr_share = impl_->node_ptr_.lock();                \
    if (node_ptr_share == nullptr) {                                                     \
      REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");  \
      GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: the node shared ptr is not valid."); \
      return GRAPH_FAILED;                                                               \
    }                                                                                    \
    const Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr_share);             \
    if (op.GetAttr(ascend_name, attr_value) != GRAPH_SUCCESS) {                          \
      GELOGW("[Get][Attr] of node[%s] failed.", node_ptr_share->GetName().c_str());      \
      return GRAPH_FAILED;                                                               \
    }                                                                                    \
                                                                                         \
    return GRAPH_SUCCESS;                                                                \
  }

#define NODE_ATTR_SET_IMP(ArgType)                                                       \
  graphStatus GNode::SetAttr(const AscendString &name, ArgType &attr_value) const {      \
    const char_t *const ascend_name = name.GetString();                                  \
    if (std::string(ascend_name).empty()) {                                              \
      REPORT_INNER_ERR_MSG("E18888", "ascend std::string error.");                       \
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] SetAttr: ascend std::string error.");  \
      return GRAPH_PARAM_INVALID;                                                        \
    }                                                                                    \
                                                                                         \
    if (impl_ == nullptr) {                                                              \
      REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");            \
      GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: node impl is nullptr.");             \
      return GRAPH_FAILED;                                                               \
    }                                                                                    \
                                                                                         \
    const std::shared_ptr<Node> node_ptr_share = impl_->node_ptr_.lock();                \
    if (node_ptr_share == nullptr) {                                                     \
      REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");  \
      GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: the node shared ptr is not valid."); \
      return GRAPH_FAILED;                                                               \
    }                                                                                    \
                                                                                         \
    Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr_share);                   \
    (void)op.SetAttr(ascend_name, attr_value);                                           \
    return GRAPH_SUCCESS;                                                                \
  }

namespace af {
class NodeImpl {
 public:
  NodeImpl() = default;
  ~NodeImpl() = default;

  NodeImpl(NodeImpl &) = delete;
  NodeImpl &operator=(const NodeImpl &) = delete;

 private:
  friend class NodeAdapter;
  friend class GNode;
  std::weak_ptr<Node> node_ptr_;
};

NodePtr NodeAdapter::GNode2Node(const GNode &node) {
  if (node.impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param graph_node.impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GNode2Node: gnode impl is nullptr.");
    return nullptr;
  }

  return node.impl_->node_ptr_.lock();
}

GNode NodeAdapter::Node2GNode(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node2GNode: node is nullptr");
    return GNode();
  }

  const GNode graph_node;
  if (graph_node.impl_ == nullptr) {
    GELOGW("[Check][Param] Gnode impl is nullptr, node:%s", node->GetName().c_str());
    return graph_node;
  }
  graph_node.impl_->node_ptr_ = node;

  return graph_node;
}

GNodePtr NodeAdapter::Node2GNodePtr(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node2GNodePtr: node is nullptr");
    return nullptr;
  }

  const GNodePtr gnode = MakeShared<GNode>();
  if (gnode == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create GNode failed.");
    GELOGE(GRAPH_FAILED, "[Create][GNode] Node2GNodePtr: gnode is nullptr, node[%s].", node->GetName().c_str());
    return nullptr;
  }

  if (gnode->impl_ == nullptr) {
    GELOGW("[Check][Param] Gnode impl is nullptr, node:%s", node->GetName().c_str());
    return nullptr;
  }
  gnode->impl_->node_ptr_ = node;

  return gnode;
}

GNode::GNode() {
  impl_ = ComGraphMakeShared<NodeImpl>();
}

graphStatus GNode::GetType(AscendString &type) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetType: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetType: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  const std::string node_type = node_ptr->GetType();
  const AscendString ascend_type(node_type.c_str());
  type = ascend_type;

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetName(AscendString &name) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetName: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  const std::string node_name = node_ptr->GetName();
  const AscendString ascend_name(node_name.c_str());
  name = ascend_name;
  return GRAPH_SUCCESS;
}

std::pair<GNodePtr, int32_t> GNode::GetInDataNodesAndPortIndexs(const int32_t index) const {
  const std::pair<GNodePtr, int32_t> gnode_idx = {nullptr, 0xFF};
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: node impl is nullptr.");
    return gnode_idx;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node ptr is not valid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: the node shared ptr is not valid.");
    return gnode_idx;
  }

  const auto in_anchor = node_ptr->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Failed to get in data node of index[%d] from node[%s], "
                         "the anchor does not exist",
                         index, node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED,
           "[Get][Anchor] Failed to get in data node of index[%d] from node[%s], "
           "the anchor does not exist",
           index, node_ptr->GetName().c_str());
    return gnode_idx;
  }

  const auto out_anchor = in_anchor->GetPeerOutAnchor();
  if (out_anchor == nullptr) {
    GELOGI("Index[%d] from node[%s] has no peer anchor, maybe it is optional input", index,
           node_ptr->GetName().c_str());
    return gnode_idx;
  }

  const NodePtr peer_node_ptr = out_anchor->GetOwnerNode();
  const GNodePtr gnode = NodeAdapter::Node2GNodePtr(peer_node_ptr);
  if (gnode == nullptr) {
    GELOGE(GRAPH_FAILED, "[Get][GNode] Peer node of node[%s] to gnode failed.", node_ptr->GetName().c_str());
    return gnode_idx;
  }

  return {gnode, out_anchor->GetIdx()};
}

std::vector<GNodePtr> GNode::GetInControlNodes() const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: node impl is nullptr.");
    return {};
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Gnode: node ptr is not valid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: the node shared ptr is not valid.");
    return {};
  }

  std::vector<GNodePtr> gnodes;
  const auto in_control_nodes = node_ptr->GetInControlNodes();
  for (auto &in_control_node : in_control_nodes) {
    GNodePtr gnode = NodeAdapter::Node2GNodePtr(in_control_node);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "[Get][GNode] In control_node of node[%s] to gnode failed.", node_ptr->GetName().c_str());
      return {};
    }
    gnodes.emplace_back(gnode);
  }

  return gnodes;
}

std::vector<std::pair<GNodePtr, int32_t>> GNode::GetOutDataNodesAndPortIndexs(const int32_t index) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: node impl is nullptr.");
    return {};
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Gnode: the node shared ptr is not valid.");
    return {};
  }

  const auto out_anchor = node_ptr->GetOutDataAnchor(index);
  if (out_anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Failed to get out data node of index %d from node %s, "
                         "the anchor does not exist",
                         index, node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED,
           "[Get][Anchor] Failed to get out data node of index %d from node %s, "
           "the anchor does not exist",
           index, node_ptr->GetName().c_str());
    return {};
  }

  std::vector<std::pair<GNodePtr, int32_t>> gnode_index;
  const auto in_data_anchors = out_anchor->GetPeerInDataAnchorsPtr();
  for (auto &in_data_anchor : in_data_anchors) {
    if (in_data_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "In data anchor of node[%s] is nullptr.", node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] In data anchor of node[%s] is nullptr.", node_ptr->GetName().c_str());
      return {};
    }
    const NodePtr peer_node_ptr = in_data_anchor->GetOwnerNode();
    const GNodePtr gnode = NodeAdapter::Node2GNodePtr(peer_node_ptr);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "[Get][GNode] Peer node of node[%s] to gnode failed.", node_ptr->GetName().c_str());
      return {};
    }
    gnode_index.emplace_back(std::pair<GNodePtr, int32_t>(gnode, in_data_anchor->GetIdx()));
  }

  return gnode_index;
}

std::vector<GNodePtr> GNode::GetOutControlNodes() const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutControlNodes: node impl is nullptr.");
    return {};
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutControlNodes: the node shared ptr is not valid.");
    return {};
  }

  std::vector<GNodePtr> gnodes;
  const auto out_control_nodes = node_ptr->GetOutControlNodes();
  for (auto &out_control_node : out_control_nodes) {
    GNodePtr gnode = NodeAdapter::Node2GNodePtr(out_control_node);
    if (gnode == nullptr) {
      GELOGE(GRAPH_FAILED, "[Get][GNode] In control_node of node[%s] to gnode failed.", node_ptr->GetName().c_str());
      return {};
    }
    gnodes.emplace_back(gnode);
  }

  return gnodes;
}

graphStatus GNode::GetInputConstData(const int32_t index, Tensor &data) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputConstData: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputConstData: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const NodePtr input_data_node = NodeUtils::GetInDataNodeByIndex(*node_ptr, index);
  GE_CHECK_NOTNULL(input_data_node);
  const std::string op_type = input_data_node->GetType();
  if ((op_type == CONSTANT) || (op_type == CONSTANTOP)) {
    const Operator const_op = OpDescUtils::CreateOperatorFromNode(input_data_node);
    if (const_op.GetAttr(ATTR_NAME_WEIGHTS.c_str(), data) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Input data node[%s] of node[%s] get data failed.",
                           input_data_node->GetName().c_str(), node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Attr] Input data node[%s] of node[%s] get data failed.",
             input_data_node->GetName().c_str(), node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    return SUCCESS;
  } else if (op_type == DATA) {
    auto parent_node = NodeUtils::GetParentInput(input_data_node);
    while ((parent_node != nullptr) && (parent_node->GetType() == DATA)) {
      parent_node = NodeUtils::GetParentInput(parent_node);
    }
    if ((parent_node != nullptr) && ((parent_node->GetType() == CONSTANT) || (parent_node->GetType() == CONSTANTOP))) {
      const Operator const_op = OpDescUtils::CreateOperatorFromNode(parent_node);
      if (const_op.GetAttr(ATTR_NAME_WEIGHTS.c_str(), data) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "Input data node[%s] of node[%s] get data failed.",
                             parent_node->GetName().c_str(), node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][Attr] Input data node[%s] of node[%s] get data failed.",
               parent_node->GetName().c_str(), node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }
  } else {
    // empty
  }
  REPORT_INNER_ERR_MSG("E18888", "Node[%s] has no const input.", node_ptr->GetName().c_str());
  GELOGE(GRAPH_NODE_WITHOUT_CONST_INPUT, "[Check][Param] Node[%s] has no const input.", node_ptr->GetName().c_str());
  return GRAPH_NODE_WITHOUT_CONST_INPUT;
}

graphStatus GNode::GetInputIndexByName(const AscendString &name, int32_t &index) {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetInputIndexByName: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputIndexByName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputIndexByName: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const auto op_desc = node_ptr->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "get Op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const std::string node_name = ascend_name;
  index = op_desc->GetInputIndexByName(node_name);

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetOutputIndexByName(const AscendString &name, int32_t &index) {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetOutputIndexByName: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputIndexByName: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputIndexByName: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const auto op_desc = node_ptr->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const std::string node_name = ascend_name;
  index = op_desc->GetOutputIndexByName(node_name);

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetDynamicInputIndexesByName(const AscendString &name, std::vector<int32_t> &indexes) {
  const std::string node_name = name.GetString();
  if (node_name.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "input name is empty.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] input name is empty.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr");
    GELOGE(GRAPH_FAILED, "[Check][Param] the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const auto op_desc = node_ptr->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (op_desc->GetDynamicInputIndexesByName(node_name, indexes) != GRAPH_SUCCESS || indexes.empty()) {
    const std::string error_message = indexes.empty()
                                          ? "the dynamic input indexes are empty, input name is " + node_name + "."
                                          : "get dynamic input index by name failed, input name is " + node_name + ".";
    REPORT_INNER_ERR_MSG("E18888", "%s", error_message.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] %s", error_message.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetDynamicOutputIndexesByName(const AscendString &name, std::vector<int32_t> &indexes) {
  const std::string node_name = name.GetString();
  if (node_name.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "output name is empty.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] output name is empty.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] the node shared ptr is nullptr.");
    return GRAPH_FAILED;
  }

  const auto op_desc = node_ptr->GetOpDescBarePtr();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (op_desc->GetDynamicOutputIndexesByName(node_name, indexes) != GRAPH_SUCCESS || indexes.empty()) {
    const std::string error_message =
        indexes.empty() ? "the dynamic output indexes are empty, output name is " + node_name + "."
                        : "get dynamic output index by name failed, output name is " + node_name + ".";
    REPORT_INNER_ERR_MSG("E18888", "%s", error_message.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] %s", error_message.c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

size_t GNode::GetInputsSize() const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputsSize: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputsSize: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return op_desc->GetInputsSize();
}

size_t GNode::GetOutputsSize() const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputsSize: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputsSize: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return op_desc->GetOutputsSize();
}

graphStatus GNode::GetInputDesc(const int32_t index, TensorDesc &tensor_desc) const {
  if (index < 0) {
    REPORT_INNER_ERR_MSG("E18888", "index:%d can not be less than zero, check invalid.", index);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetInputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetInputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  if (static_cast<size_t>(index) >= op_desc->GetAllInputsSize()) {
    REPORT_INNER_ERR_MSG("E18888", "Get tensor desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][TensorDesc] of node[%s] failed, invalid index is %d and valid range should be [0, %d)",
           node_ptr->GetName().c_str(), index, op_desc->GetAllInputsSize());
    return GRAPH_FAILED;
  }
  const ConstGeTensorDescPtr ge_tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(index));
  if (ge_tensor_desc == nullptr) {
    GELOGW("[Get][TensorDesc] of node[%s] failed, input is optional input with no edge and index is [%d]",
           node_ptr->GetName().c_str(), index);
    return GRAPH_FAILED;
  }
  tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc);

  return GRAPH_SUCCESS;
}

graphStatus GNode::UpdateInputDesc(const int32_t index, const TensorDesc &tensor_desc) {
  if (index < 0) {
    REPORT_INNER_ERR_MSG("E18888", "index:%d cannot be less than zero.", index);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] UpdateInputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] UpdateInputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] UpdateInputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const GeTensorDesc ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  if (op_desc->UpdateInputDesc(static_cast<uint32_t>(index), ge_tensor_desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Update input desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Update][InputDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetOutputDesc(const int32_t index, TensorDesc &tensor_desc) const {
  if (index < 0) {
    REPORT_INNER_ERR_MSG("E18888", "index:%d cannot be less than zero.", index);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetOutputDesc: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetOutputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const ConstGeTensorDescPtr ge_tensor_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(index));
  if (ge_tensor_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get tensor desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][TensorDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(*ge_tensor_desc);

  return GRAPH_SUCCESS;
}

graphStatus GNode::UpdateOutputDesc(const int32_t index, const TensorDesc &tensor_desc) {
  if (index < 0) {
    REPORT_INNER_ERR_MSG("E18888", "index:%d cannot be less than zero.", index);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Gnode: index[%d] cannot be less than zero.", index);
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] UpdateOutputDesc: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] UpdateOutputDesc: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const GeTensorDesc ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  if (op_desc->UpdateOutputDesc(static_cast<uint32_t>(index), ge_tensor_desc) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Update input desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Update][InputDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

NODE_ATTR_GET_IMP(int64_t)
NODE_ATTR_GET_IMP(int32_t)
NODE_ATTR_GET_IMP(uint32_t)
NODE_ATTR_GET_IMP(float32_t)
NODE_ATTR_GET_IMP(bool)
NODE_ATTR_GET_IMP(Tensor)
NODE_ATTR_GET_IMP(std::vector<int64_t>)
NODE_ATTR_GET_IMP(std::vector<int32_t>)
NODE_ATTR_GET_IMP(std::vector<uint32_t>)
NODE_ATTR_GET_IMP(std::vector<float32_t>)
NODE_ATTR_GET_IMP(std::vector<bool>)
NODE_ATTR_GET_IMP(std::vector<Tensor>)
NODE_ATTR_GET_IMP(OpBytes)
NODE_ATTR_GET_IMP(std::vector<std::vector<int64_t>>)
NODE_ATTR_GET_IMP(std::vector<DataType>)
NODE_ATTR_GET_IMP(DataType)
NODE_ATTR_GET_IMP(AttrValue)

NODE_ATTR_SET_IMP(int64_t)
NODE_ATTR_SET_IMP(int32_t)
NODE_ATTR_SET_IMP(uint32_t)
NODE_ATTR_SET_IMP(float32_t)
NODE_ATTR_SET_IMP(bool)
NODE_ATTR_SET_IMP(Tensor)
NODE_ATTR_SET_IMP(std::vector<int64_t>)
NODE_ATTR_SET_IMP(std::vector<int32_t>)
NODE_ATTR_SET_IMP(std::vector<uint32_t>)
NODE_ATTR_SET_IMP(std::vector<float32_t>)
NODE_ATTR_SET_IMP(std::vector<bool>)
NODE_ATTR_SET_IMP(std::vector<Tensor>)
NODE_ATTR_SET_IMP(OpBytes)
NODE_ATTR_SET_IMP(std::vector<std::vector<int64_t>>)
NODE_ATTR_SET_IMP(std::vector<DataType>)
NODE_ATTR_SET_IMP(DataType)

graphStatus GNode::SetAttr(const AscendString &name, AttrValue &attr_value) const {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] SetAttr: ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name.c_str(), std::move(attr_value));
  return GRAPH_SUCCESS;
}

graphStatus GNode::SetAttr(const AscendString &name, AscendString &attr_value) const {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "name ascend string error");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] SetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  const std::string node_name = ascend_name;
  const std::string node_attr_value = attr_value.GetString();
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name, node_attr_value);

  return GRAPH_SUCCESS;
}

graphStatus GNode::SetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "name ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] SetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  for (auto &attr_val : attr_values) {
    const char_t *const ascend_attr_value = attr_val.GetString();
    if (std::string(ascend_attr_value).empty()) {
      REPORT_INNER_ERR_MSG("E18888", "param attr values are invalid");
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] SetAttr: attr val error.");
      return GRAPH_PARAM_INVALID;
    }
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetAttr: the shared ptr is not valid.");
    return GRAPH_FAILED;
  }
  std::vector<std::string> node_attr_vals;
  for (const auto &attr_val : attr_values) {
    if (attr_val.GetString() != nullptr) {
      std::string node_attr_val = attr_val.GetString();
      node_attr_vals.emplace_back(node_attr_val);
    }
  }
  const std::string node_name = ascend_name;
  Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  (void)op.SetAttr(node_name, node_attr_vals);

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetAttr(const AscendString &name, AscendString &attr_value) const {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "name ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "GetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const std::string node_name = ascend_name;
  const Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  std::string op_name;
  if (op.GetAttr(node_name, op_name) != GRAPH_SUCCESS) {
    GELOGW("[Check][Param] Get attr of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const AscendString attr_value_get(op_name.c_str());
  attr_value = attr_value_get;

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "name ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GetAttr: name ascend string error.");
    return GRAPH_PARAM_INVALID;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAttr: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const std::string node_name = ascend_name;
  const Operator op = OpDescUtils::CreateOperatorFromNode(node_ptr);
  std::vector<std::string> attr_names;
  if (op.GetAttr(node_name, attr_names) != GRAPH_SUCCESS) {
    GELOGW("[Get][Attr] of node[%s] failed.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (auto &attr_name : attr_names) {
    const AscendString ascend_attr_name(attr_name.c_str());
    attr_values.push_back(ascend_attr_name);
  }

  return GRAPH_SUCCESS;
}

bool GNode::HasAttr(const AscendString &name) {
  const char_t *const ascend_name = name.GetString();
  if (std::string(ascend_name).empty()) {
    REPORT_INNER_ERR_MSG("E18888", "ascend string error.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] HasAttr: ascend string error.");
    return false;
  }

  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] HasAttr: node impl is nullptr.");
    return false;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] HasAttr: the node shared ptr is not valid.");
    return false;
  }

  const OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Get op desc of node[%s] failed.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] of node[%s] failed.", node_ptr->GetName().c_str());
    return false;
  }
  const std::string attr_name = ascend_name;
  if (!op_desc->HasAttr(attr_name)) {
    GELOGW("[Call][HasAttr] Node[%s] has no attr name[%s]", node_ptr->GetName().c_str(), attr_name.c_str());
    return false;
  }

  return true;
}

graphStatus GNode::GetSubgraph(uint32_t index, GraphPtr &graph) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetSubgraph: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetSubgraph: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const ComputeGraphPtr compute_graph_ptr = NodeUtils::GetSubgraph(*node_ptr, index);
  if (compute_graph_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "get subgraph[%u] failed from node[%s].", index, node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][SubGraph] subgraph[%u] from node[%s] is nullptr.", index, node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph_ptr);
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create compute graph failed from %s.", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Create][Graph] failed from %s.", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::GetALLSubgraphs(std::vector<GraphPtr> &graph_list) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetALLSubgraphs: node impl is nullptr.");
    return GRAPH_FAILED;
  }

  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  if (node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "the node shared ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetALLSubgraphs: the node shared ptr is not valid.");
    return GRAPH_FAILED;
  }

  const auto root_graph = GraphUtils::FindRootGraph(node_ptr->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to find root graph from node %s ", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][RootGraph] Failed to find root graph from node %s ", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  const std::vector<ComputeGraphPtr> sub_graphs = root_graph->GetAllSubgraphs();
  if (sub_graphs.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "get all subgraphs failed from node[%s].", node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][ALLSubGraphs] failed from node[%s].", node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  for (auto &sub_graph : sub_graphs) {
    if (sub_graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "get subgraph failed from node[%s].", node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][SubGraph] failed from node[%s].", node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    GraphPtr graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(sub_graph);
    if (graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "create compute graph failed from node[%s].", node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Create][ComputeGraph] failed from node[%s].", node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    graph_list.emplace_back(graph);
  }

  if (graph_list.empty()) {
    GELOGW("[Get][Subgraph] Node %s has no subgraph", node_ptr->GetName().c_str());
  }

  return GRAPH_SUCCESS;
}

graphStatus GNode::SetSubgraph(const AscendString &subgraph_ir_name, const Graph &subgraph) {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");

  auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(subgraph);
  GE_ASSERT_NOTNULL(sub_compute_graph);
  return NodeUtils::AddSubgraph(node_ptr, subgraph_ir_name.GetString(), sub_compute_graph);
}
graphStatus GNode::SetSubgraphs(const AscendString &subgraph_ir_name, const std::vector<Graph> &subgraphs) {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");

  std::vector<ComputeGraphPtr> sub_compute_graphs{};
  for (const auto &subgraph : subgraphs) {
    auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(subgraph);
    GE_ASSERT_NOTNULL(sub_compute_graph);
    sub_compute_graphs.emplace_back(sub_compute_graph);
  }

  return NodeUtils::AddSubgraphs(node_ptr, subgraph_ir_name.GetString(), sub_compute_graphs);
}

namespace {
graphStatus GetAttrValue(const std::shared_ptr<Node> &node_ptr, const AscendString &name, uint32_t index,
                         AttrValue &attr_value, bool is_input) {
  auto tensor_desc =
      is_input ? node_ptr->GetOpDesc()->MutableInputDesc(index) : node_ptr->GetOpDesc()->MutableOutputDesc(index);
  GE_ASSERT_NOTNULL(tensor_desc, "index: %u is invalid for node: %s %s", index, node_ptr->GetNamePtr(),
                    node_ptr->GetTypePtr());
  GE_WARN_ASSERT_GRAPH_SUCCESS(tensor_desc->GetAttr(name.GetString(), attr_value.impl->MutableAnyValue()),
                               "Attr: %s has not been set for node: %s %s", name.GetString(), node_ptr->GetNamePtr(),
                               node_ptr->GetTypePtr());
  return GRAPH_SUCCESS;
}

graphStatus SetAttrValue(const std::shared_ptr<Node> &node_ptr, const AscendString &name, uint32_t index,
                         const AttrValue &attr_value, bool is_input) {
  auto tensor_desc =
      is_input ? node_ptr->GetOpDesc()->MutableInputDesc(index) : node_ptr->GetOpDesc()->MutableOutputDesc(index);
  GE_ASSERT_NOTNULL(tensor_desc, "index: %u is invalid for node: %s %s", index, node_ptr->GetNamePtr(),
                    node_ptr->GetTypePtr());
  return tensor_desc->SetAttr(name.GetString(), attr_value.impl->MutableAnyValue());
}
}  // namespace

// 实现新的AttrValue接口
graphStatus GNode::GetOutputAttr(const AscendString &name, uint32_t output_index, AttrValue &attr_value) const {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");
  return GetAttrValue(node_ptr, name, output_index, attr_value, false);
}

graphStatus GNode::SetOutputAttr(const AscendString &name, uint32_t output_index, const AttrValue &attr_value) {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");
  return SetAttrValue(node_ptr, name, output_index, attr_value, false);
}

graphStatus GNode::GetInputAttr(const AscendString &name, uint32_t input_index, AttrValue &attr_value) const {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");
  return GetAttrValue(node_ptr, name, input_index, attr_value, true);
}

graphStatus GNode::SetInputAttr(const AscendString &name, uint32_t input_index, const AttrValue &attr_value) {
  GE_ASSERT_NOTNULL(impl_);
  const std::shared_ptr<Node> node_ptr = impl_->node_ptr_.lock();
  GE_ASSERT_NOTNULL(node_ptr, "Gnode has invalid node source, try to call `AddNodeByOp` firstly");
  return SetAttrValue(node_ptr, name, input_index, attr_value, true);
}
}  // namespace af
