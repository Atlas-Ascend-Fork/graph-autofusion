/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/node.h"
#include "debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/operator_factory_af.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/operator_factory_impl.h"
#include "graph/shape_refiner.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/trace_manager/trace_manager.h"

namespace af {
Node::NodeImpl::NodeImpl(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
    : op_(op),
      owner_graph_(owner_graph),
      owner_graph_ptr_(owner_graph.get()),
      in_data_anchors_(),
      out_data_anchors_(),
      in_control_anchor_(nullptr),
      out_control_anchor_(nullptr),
      has_init_(false),
      host_node_(false),
      anchor_status_updated_(false) {}

Node::NodeImpl::~NodeImpl() {
  for (const auto &in_data_anchor : in_data_anchors_) {
    if (in_data_anchor != nullptr) {
      in_data_anchor->UnlinkAll();
    }
  }
  for (const auto &out_data_anchor : out_data_anchors_) {
    if (out_data_anchor != nullptr) {
      out_data_anchor->UnlinkAll();
    }
  }
  if (in_control_anchor_ != nullptr) {
    in_control_anchor_->UnlinkAll();
  }
  if (out_control_anchor_ != nullptr) {
    out_control_anchor_->UnlinkAll();
  }
}

graphStatus Node::NodeImpl::Init(const NodePtr &node) {
  if (has_init_) {
    return GRAPH_SUCCESS;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  size_t size = op_->GetAllInputsSize();
  in_data_anchors_.reserve(size);
  for (size_t i = 0UL; i < size; i++) {
    const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Current in_data_anchor is null, malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] Current in_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
  }
  size = op_->GetOutputsSize();
  out_data_anchors_.reserve(size);
  for (size_t i = 0UL; i < size; i++) {
    const std::shared_ptr<OutDataAnchor> anchor = ComGraphMakeShared<OutDataAnchor>(node, i);
    if (anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Current out_data_anchor is null, malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][OutDataAnchor] Current out_data_anchor is null, malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    out_data_anchors_.push_back(anchor);
  }
  in_control_anchor_ = ComGraphMakeShared<InControlAnchor>(node, -1);
  out_control_anchor_ = ComGraphMakeShared<OutControlAnchor>(node, -1);
  if ((in_control_anchor_ == nullptr) || (out_control_anchor_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Current in_control_anchor or out_control_anchor is null, malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED,
           "[Create][ControlAnchor] Current in_control_anchor or out_control_anchor is null, "
           "malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  has_init_ = true;
  return GRAPH_SUCCESS;
}

std::string Node::NodeImpl::GetName() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return op_->GetName();
}

std::string Node::NodeImpl::GetType() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return op_->GetType();
}

const char *Node::NodeImpl::GetNamePtr() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return nullptr, "[Check][Param] original OpDesc is nullptr");
  return op_->GetNamePtr();
}

const char *Node::NodeImpl::GetTypePtr() const {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return nullptr, "[Check][Param] original OpDesc is nullptr");
  return op_->GetTypePtr();
}

bool Node::NodeImpl::NodeMembersAreEqual(const NodeImpl &r_node) const {
  return ((((this->op_ != nullptr) && (r_node.op_ != nullptr) && (IsEqual(*(this->op_), *(r_node.op_), "node.op_"))) ||
           ((this->op_ == nullptr) && (r_node.op_ == nullptr))) &&
          IsEqual(this->has_init_, r_node.has_init_, "node.has_init_") &&
          IsEqual(this->anchor_status_updated_, r_node.anchor_status_updated_, "node.anchor_status_updated_") &&
          IsEqual(this->send_event_id_list_, r_node.send_event_id_list_, "node.send_event_id_list_") &&
          IsEqual(this->recv_event_id_list_, r_node.recv_event_id_list_, "node.recv_event_id_list_"));
}

bool Node::NodeImpl::NodeAnchorIsEqual(const AnchorPtr &left_anchor, const AnchorPtr &right_anchor,
                                       const size_t i) const {
  GE_IF_BOOL_EXEC(left_anchor == nullptr, REPORT_INNER_ERR_MSG("E18888", "left_anchor is nullptr, check invalid.");
                  GELOGE(GRAPH_FAILED, "[Check][Param] left_anchor is null."); return false);
  GE_IF_BOOL_EXEC(right_anchor == nullptr, REPORT_INNER_ERR_MSG("E18888", "right_anchor is nullptr, check invalid.");
                  GELOGE(GRAPH_FAILED, "[Check][Param] right_anchor is null."); return false);
  const auto anchor_peer_size = left_anchor->GetPeerAnchorsSize();
  const auto right_anchor_peer_size = right_anchor->GetPeerAnchorsSize();
  // Firstly, verify anchor's peer anchors size equal or not
  if (anchor_peer_size != right_anchor_peer_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Size of anchor's peer anchors verify failed, node name: %s "
                         "anchor_peer_size [%zu] is different from [%zu] at index [%zu].",
                         this->GetName().c_str(), anchor_peer_size, right_anchor_peer_size, i);
    GELOGE(GRAPH_FAILED,
           "[Check][Param] Size of anchor's peer anchors verify failed, node name: %s "
           "anchor_peer_size [%zu] is different from [%zu] at index [%zu].",
           this->GetName().c_str(), anchor_peer_size, right_anchor_peer_size, i);
    return false;
  }
  // Secondly, verify anchor's peer anchor owner node equal or not
  for (size_t j = 0UL; j < anchor_peer_size; j++) {
    const auto peer_node = left_anchor->GetPeerAnchorsPtr().at(j)->GetOwnerNodeBarePtr();
    const auto r_peer_node = right_anchor->GetPeerAnchorsPtr().at(j)->GetOwnerNodeBarePtr();
    if ((peer_node == nullptr) || (r_peer_node == nullptr)) {
      REPORT_INNER_ERR_MSG("E18888", "anchor's peer node is null, node name: %s index[%zu] peer node index[%zu].",
                           this->GetName().c_str(), i, j);
      GELOGE(GRAPH_FAILED,
             "[Get][OwnerNode] anchor's peer node is null, node name: %s index[%zu] "
             "peer node index[%zu].",
             this->GetName().c_str(), i, j);
      return false;
    }
    // Determine the connection relationship by linking the node's name
    if (peer_node->GetName() != r_peer_node->GetName()) {
      REPORT_INNER_ERR_MSG("E18888",
                           "anchor's peer node name verify failed, node name: %s index[%zu] "
                           "peer node name %s is different from %s at index [%zu].",
                           this->GetName().c_str(), i, peer_node->GetName().c_str(), r_peer_node->GetName().c_str(), j);
      GELOGE(GRAPH_FAILED,
             "[Check][Param] anchor's peer node name verify failed, node name: %s index[%zu] "
             "peer node name %s is different from %s at index [%zu].",
             this->GetName().c_str(), i, peer_node->GetName().c_str(), r_peer_node->GetName().c_str(), j);
      return false;
    }
  }
  return true;
}

graphStatus Node::NodeImpl::AddLinkFrom(const NodePtr &input_node, const NodePtr &owner_node) {
  // This function is deprecated, please use other two overloaded functions
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERR_MSG("E18888", "node:%s out_anchor size is:%zu, only support 1", input_node->GetName().c_str(),
                         out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  const auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(op_desc->GetOutputDesc(0U)) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "add input desc failed, op:%s.", op_->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }
  const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, malloc shared_ptr failed.",
           out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void)out_anchors.at(0U)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFrom(const uint32_t &index, const NodePtr &input_node, const NodePtr &owner_node) {
  return AddLinkFrom(index, input_node, 0U, owner_node);
}

graphStatus Node::NodeImpl::AddLinkFrom(const uint32_t &index, const af::Node::NodePtr &input_node,
                                        const uint32_t &input_node_index, const af::Node::NodePtr &owner_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (input_node_index >= out_anchors.size()) {
    REPORT_INNER_ERR_MSG("E18888", "node:%s out_anchor size is:%zu, but index is %u", input_node->GetName().c_str(),
                         out_anchors.size(), input_node_index);
    GELOGE(GRAPH_FAILED, "[Check][Param] out_anchor size is:%zu, but index is %u", out_anchors.size(),
           input_node_index);
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  const auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (op_->AddInputDesc(index, op_desc->GetOutputDesc(input_node_index)) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "add input desc failed, index:%u.", index);
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }

  if (index < GetAllInDataAnchors(owner_node).size()) {
    (void)out_anchors.at(input_node_index)->LinkTo(in_data_anchors_[static_cast<size_t>(index)]);
  } else if (index == GetAllInDataAnchors(owner_node).size()) {
    const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
    if (anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "out_anchor size is:%zu, malloc shared_ptr failed.", out_anchors.size());
      GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, malloc shared_ptr failed.",
             out_anchors.size());
      return GRAPH_FAILED;
    }
    in_data_anchors_.push_back(anchor);
    (void)out_anchors.at(input_node_index)->LinkTo(in_data_anchors_.back());
  } else {
    REPORT_INNER_ERR_MSG("E18888", "index %u exceeds in data anchors size %zu.", index, in_data_anchors_.size());
    GELOGE(GRAPH_FAILED, "index %u exceeds in data anchors size %zu.", index, in_data_anchors_.size());
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFromForParse(const NodePtr &input_node, const NodePtr &owner_node) {
  //  This function is used for ParseWeights.
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERR_MSG("E18888", "node:%s out_anchor size is:%zu, only support 1", input_node->GetName().c_str(),
                         out_anchors.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "out_anchor size is:%zu, make anchor failed", out_anchors.size());
    GELOGE(GRAPH_FAILED, "[Create][InDataAnchor] out_anchor size is:%zu, make anchor failed", out_anchors.size());
    return GRAPH_FAILED;
  }
  in_data_anchors_.push_back(anchor);
  (void)out_anchors.at(0U)->LinkTo(in_data_anchors_.back());

  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::AddLinkFrom(const std::string &name, const NodePtr &input_node, const NodePtr &owner_node) {
  GE_CHECK_NOTNULL(input_node);
  // Input_node ---> this
  auto out_anchors = input_node->GetAllOutDataAnchors();
  if (out_anchors.size() != 1UL) {
    REPORT_INNER_ERR_MSG("E18888", "node:%s out_anchor size is:%zu, only support 1", input_node->GetName().c_str(),
                         out_anchors.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] out_anchor size is:%zu, only support 1", out_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  GE_CHECK_NOTNULL(op_);
  const auto input_op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(input_op_desc);
  const auto index = op_->GetInputIndexByName(name);
  if (index != -1) {
    if (index >= static_cast<int32_t>(in_data_anchors_.size())) {
      REPORT_INNER_ERR_MSG("E18888",
                           "op %s get input name %s 's index %d is illegal as which >= indataanchors size:%zu.",
                           op_->GetName().c_str(), name.c_str(), index, in_data_anchors_.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] op %s get input name %s 's index %d is illegal.", op_->GetName().c_str(),
             name.c_str(), index);
      return GRAPH_FAILED;
    }
    (void)out_anchors.at(0U)->LinkTo(in_data_anchors_[static_cast<size_t>(index)]);
  } else {
    const std::shared_ptr<InDataAnchor> anchor = ComGraphMakeShared<InDataAnchor>(owner_node, in_data_anchors_.size());
    GE_CHECK_NOTNULL(anchor);
    in_data_anchors_.push_back(anchor);
    (void)out_anchors.at(0U)->LinkTo(in_data_anchors_.back());
  }
  if (op_->AddInputDesc(name, input_op_desc->GetOutputDesc(0U)) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "add input desc failed, name:%s, op:%s", name.c_str(), op_->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputDesc] failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

ComputeGraphPtr Node::NodeImpl::GetOwnerComputeGraph() const {
  return owner_graph_.lock();
}

ComputeGraph *Node::NodeImpl::GetOwnerComputeGraphBarePtr() const {
  return owner_graph_ptr_;
}

graphStatus Node::NodeImpl::SetOwnerComputeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    return GRAPH_PARAM_INVALID;
  }
  owner_graph_ = graph;
  owner_graph_ptr_ = graph.get();

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "owner_graph", "", "", graph->GetName());
  return GRAPH_SUCCESS;
}

graphStatus Node::NodeImpl::ClearOwnerGraph(const ComputeGraphPtr &graph) {
  owner_graph_ = graph;
  owner_graph_ptr_ = graph.get();

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "delete", TraceManager::GetOutGraphName(), this->GetName(),
                   "owner_graph", "", "", ((graph == nullptr) ? std::string("") : graph->GetName()));
  return GRAPH_SUCCESS;
}

Node::Vistor<InDataAnchorPtr> Node::NodeImpl::GetAllInDataAnchors(const ConstNodePtr &node_ptr) const {
  return Node::Vistor<InDataAnchorPtr>(node_ptr, in_data_anchors_);
}

Node::Vistor<OutDataAnchorPtr> Node::NodeImpl::GetAllOutDataAnchors(const ConstNodePtr &node_ptr) const {
  return Node::Vistor<OutDataAnchorPtr>(node_ptr, out_data_anchors_);
}

uint32_t Node::NodeImpl::GetAllInDataAnchorsSize() const {
  return static_cast<uint32_t>(in_data_anchors_.size());
}

uint32_t Node::NodeImpl::GetAllOutDataAnchorsSize() const {
  return static_cast<uint32_t>(out_data_anchors_.size());
}

Node::Vistor<AnchorPtr> Node::NodeImpl::GetAllInAnchors(const ConstNodePtr &owner_node) const {
  std::vector<AnchorPtr> vec;
  // Push back in_data_anchors_
  for (const auto &in_anchor_iter : Node::Vistor<InDataAnchorPtr>(owner_node, in_data_anchors_)) {
    const auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_anchor_iter);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  // Push back in_control_anchor_
  if ((!in_control_anchor_->GetPeerOutControlAnchorsPtr().empty()) ||
      (!in_control_anchor_->GetPeerOutDataAnchors().empty())) {
    const auto in_anchor = Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
    if (in_anchor != nullptr) {
      vec.push_back(in_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(owner_node, vec);
}

std::vector<Anchor *> Node::NodeImpl::GetAllInAnchorsPtr() const {
  std::vector<Anchor *> vec;
  vec.reserve(in_data_anchors_.size() + 1U);  // in_data_anchors_ + in_control_anchor_
  // Push back in_data_anchors_
  for (const auto &in_anchor : in_data_anchors_) {
    if (in_anchor != nullptr) {
      vec.emplace_back(in_anchor.get());
    }
  }
  // Push back in_control_anchor_
  vec.emplace_back(in_control_anchor_.get());
  return vec;
}

Node::Vistor<AnchorPtr> Node::NodeImpl::GetAllOutAnchors(const ConstNodePtr &owner_node) const {
  std::vector<AnchorPtr> vec;
  // Push back out_data_anchors_
  for (const auto &out_anchor_iter : Node::Vistor<OutDataAnchorPtr>(owner_node, out_data_anchors_)) {
    const auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_anchor_iter);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  // Push back out_control_anchor_
  if ((!out_control_anchor_->GetPeerInControlAnchorsPtr().empty()) ||
      (!out_control_anchor_->GetPeerInDataAnchors().empty())) {
    const auto out_anchor = Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
    if (out_anchor != nullptr) {
      vec.push_back(out_anchor);
    }
  }
  return Node::Vistor<AnchorPtr>(owner_node, vec);
}

std::vector<Anchor *> Node::NodeImpl::GetAllOutAnchorsPtr() const {
  std::vector<Anchor *> vec;
  vec.reserve(out_data_anchors_.size() + 1U);  // out_data_anchors_ + out_control_anchor_
  // Push back out_data_anchors_
  for (const auto &out_anchor : out_data_anchors_) {
    if (out_anchor != nullptr) {
      vec.emplace_back(out_anchor.get());
    }
  }
  // Push back out_control_anchor_
  vec.emplace_back(out_control_anchor_.get());
  return vec;
}

InDataAnchorPtr Node::NodeImpl::GetInDataAnchor(const int32_t idx) const {
  if ((idx < 0) || (idx >= static_cast<int32_t>(in_data_anchors_.size()))) {
    GELOGW("[Check][Param] Op %s doesn't have data input %d, type = %s", GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    return in_data_anchors_[static_cast<size_t>(idx)];
  }
}

AnchorPtr Node::NodeImpl::GetInAnchor(const int32_t idx) const {
  // Idx can't be less than -1 or >= in_data_anchors_.size(), -1 means index of control anchor_
  if ((idx < -1) || (idx >= static_cast<int32_t>(in_data_anchors_.size()))) {
    GELOGW("[Check][Param] Op %s doesn't have input %d, type = %s", GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      return Anchor::DynamicAnchorCast<Anchor>(in_control_anchor_);
    }
    // Return data anchor
    return in_data_anchors_[static_cast<size_t>(idx)];
  }
}

AnchorPtr Node::NodeImpl::GetOutAnchor(const int32_t idx) const {
  // Idx can't be less than -1 or >= out_data_anchors_.size(), -1 means index of control anchor_
  if ((idx < -1) || (idx >= static_cast<int32_t>(out_data_anchors_.size()))) {
    REPORT_INNER_ERR_MSG("E18888", "Op:%s(%s) doesn't have index:%d's anchorname", GetName().c_str(), GetType().c_str(),
                         idx);
    GELOGE(GRAPH_FAILED, "[Check][Param] Op[%s] doesn't have index[%d]'s out_anchor which optype is %s.",
           GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    // Return control anchor
    if (idx == -1) {
      return Anchor::DynamicAnchorCast<Anchor>(out_control_anchor_);
    }
    // Return data anchor
    return out_data_anchors_[static_cast<size_t>(idx)];
  }
}

OutDataAnchorPtr Node::NodeImpl::GetOutDataAnchor(const int32_t idx) const {
  if ((idx < 0) || (idx >= static_cast<int32_t>(out_data_anchors_.size()))) {
    REPORT_INNER_ERR_MSG("E18888", "Op:%s(%s) doesn't have index:%d's anchorname", GetName().c_str(), GetType().c_str(),
                         idx);
    GELOGE(GRAPH_FAILED, "[Check][Param] Op[%s] doesn't have index[%d]'s out_data_anchor which optype is %s.",
           GetName().c_str(), idx, GetType().c_str());
    return nullptr;
  } else {
    return out_data_anchors_[static_cast<size_t>(idx)];
  }
}

InControlAnchorPtr Node::NodeImpl::GetInControlAnchor() const {
  return in_control_anchor_;
}

OutControlAnchorPtr Node::NodeImpl::GetOutControlAnchor() const {
  return out_control_anchor_;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInNodes(const af::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  vec.reserve(GetInNodesSize());
  for (const auto &in_anchor : in_data_anchors_) {
    if (in_anchor == nullptr) {
      continue;
    }
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto &node = out_anchor->GetOwnerNode();
    vec.push_back(node);
  }
  if (in_control_anchor_ != nullptr) {
    for (const auto out_control_anchor : in_control_anchor_->GetPeerOutControlAnchorsPtr()) {
      const auto &node = out_control_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

std::vector<Node *> Node::NodeImpl::GetInNodesPtr() const {
  std::vector<Node *> in_nodes;
  in_nodes.reserve(GetInNodesSize());
  for (const auto &in_anchor : in_data_anchors_) {
    if (in_anchor == nullptr) {
      continue;
    }
    const auto &out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    in_nodes.push_back(out_anchor->GetOwnerNodeBarePtr());
  }
  if (in_control_anchor_ != nullptr) {
    for (const auto out_control_anchor : in_control_anchor_->GetPeerOutControlAnchorsPtr()) {
      in_nodes.push_back(out_control_anchor->GetOwnerNodeBarePtr());
    }
  }
  return in_nodes;
}

bool Node::NodeImpl::IsAllInNodesSeen(const std::unordered_set<Node *> &nodes_seen) const {
  for (const auto &in_anchor : in_data_anchors_) {
    GE_CHK_BOOL_EXEC((in_anchor != nullptr), continue, "[Check][Param] in_data_anchor is nullptr, node:%s",
                     GetName().c_str());
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    const auto node = out_anchor->GetOwnerNodeBarePtr();
    if (node == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "peer node is null, node name: %s input index[%d] peer node output index[%d].",
                           GetName().c_str(), in_anchor->GetIdx(), out_anchor->GetIdx());
      GELOGE(GRAPH_FAILED,
             "[Get][OwnerNode] peer node is null, node name: %s input index[%d] peer node output index[%d].",
             GetName().c_str(), in_anchor->GetIdx(), out_anchor->GetIdx());
      return false;
    }
    if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
      continue;
    }
    if (nodes_seen.count(node) == 0U) {
      return false;
    }
  }

  if (in_control_anchor_ != nullptr) {
    if (in_control_anchor_->IsPeerOutAnchorsEmpty()) {
      return true;
    }
    const auto peer_out_control_anchors = in_control_anchor_->GetPeerOutControlAnchors();
    for (const auto &out_control_anchor : peer_out_control_anchors) {
      const auto node = out_control_anchor->GetOwnerNodeBarePtr();
      if (node == nullptr) {
        continue;
      }
      if ((node->GetType() == NEXTITERATION) || (node->GetType() == REFNEXTITERATION)) {
        continue;
      }
      if (nodes_seen.count(node) == 0U) {
        return false;
      }
    }
  }

  return true;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInDataNodes(const af::ConstNodePtr &owner_node) const {
  const auto &vec = NodeUtils::GetInDataNodes(*owner_node, nullptr);
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInControlNodes(const af::ConstNodePtr &owner_node) const {
  const auto &vec = NodeUtils::GetInControlNodes(*owner_node, nullptr);
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutNodes(const af::ConstNodePtr &owner_node) const {
  std::vector<NodePtr> vec;
  vec.reserve(GetOutNodesSize());
  for (const auto &out_anchor : out_data_anchors_) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (const auto peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      const auto &node = peer_in_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  if (out_control_anchor_ != nullptr) {
    for (const auto in_control_anchor : out_control_anchor_->GetPeerInControlAnchorsPtr()) {
      const auto &node = in_control_anchor->GetOwnerNode();
      vec.push_back(node);
    }
  }
  return Node::Vistor<NodePtr>(owner_node, vec);
}

std::vector<Node *> Node::NodeImpl::GetOutNodesPtr() const {
  std::vector<Node *> vec;
  vec.reserve(GetOutNodesSize());
  for (const auto &out_anchor : out_data_anchors_) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (const auto peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      vec.push_back(peer_in_anchor->GetOwnerNodeBarePtr());
    }
  }
  if (out_control_anchor_ != nullptr) {
    for (const auto in_control_anchor : out_control_anchor_->GetPeerInControlAnchorsPtr()) {
      vec.push_back(in_control_anchor->GetOwnerNodeBarePtr());
    }
  }
  return vec;
}

Node::Vistor<NodePtr> Node::NodeImpl::GetInAllNodes(const af::ConstNodePtr &owner_node) const {
  return GetInNodes(owner_node);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutDataNodes(const ConstNodePtr &owner_node) const {
  const auto &vec = NodeUtils::GetOutDataNodes(*owner_node, nullptr);
  return Node::Vistor<NodePtr>(owner_node, vec);
}

std::vector<Node *> Node::NodeImpl::GetOutDataNodesPtr() const {
  std::vector<Node *> vec;
  for (const auto &out_anchor : out_data_anchors_) {
    if (out_anchor != nullptr) {
      for (const auto in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
        if (in_anchor != nullptr) {
          const auto node = in_anchor->GetOwnerNodeBarePtr();
          vec.emplace_back(node);
        }
      }
    }
  }
  return vec;
}

uint32_t Node::NodeImpl::GetOutDataNodesSize() const {
  uint32_t out_nums = 0U;
  for (const auto &out_anchor : out_data_anchors_) {
    GE_CHK_BOOL_EXEC((out_anchor != nullptr), continue, "[Check][Param] out data anchor is nullptr, node:%s",
                     GetName().c_str());
    out_nums += out_anchor->GetPeerInDataNodesSize();
  }
  return out_nums;
}

uint32_t Node::NodeImpl::GetOutControlNodesSize() const {
  uint32_t out_nums = 0U;
  if (out_control_anchor_ != nullptr) {
    out_nums += out_control_anchor_->GetPeerAnchorsSize();
  }
  return out_nums;
}

uint32_t Node::NodeImpl::GetOutNodesSize() const {
  return GetOutDataNodesSize() + GetOutControlNodesSize();
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutControlNodes(const af::ConstNodePtr &owner_node) const {
  const auto &vec = NodeUtils::GetOutControlNodes(*owner_node, nullptr);
  return Node::Vistor<NodePtr>(owner_node, vec);
}

Node::Vistor<NodePtr> Node::NodeImpl::GetOutAllNodes(const af::ConstNodePtr &owner_node) const {
  return GetOutNodes(owner_node);
}

OpDescPtr Node::NodeImpl::GetOpDesc() const {
  return op_;
}

OpDesc *Node::NodeImpl::GetOpDescBarePtr() const {
  return op_.get();
}

graphStatus Node::NodeImpl::UpdateOpDesc(const OpDescPtr &op_desc) {
  GE_CHK_BOOL_EXEC(op_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return GRAPH_FAILED, "[Check][Param] original OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERR_MSG("E18888", "param op_desc is nullptr, check invalid.");
                   return GRAPH_PARAM_INVALID, "[Check][Param] Param OpDesc is nullptr");
  GE_CHK_BOOL_EXEC(op_->GetInputsSize() == op_desc->GetInputsSize(),
                   REPORT_INNER_ERR_MSG("E18888",
                                        "inputs count(%zu) of param op_desc not equal to "
                                        "inputs count(%zu) of original opdesc:%s, check invalid",
                                        op_desc->GetInputsSize(), op_->GetInputsSize(), op_->GetName().c_str());
                   return GRAPH_PARAM_INVALID,
                          "[Check][Param] Inputs count expected to be same, original OpDesc %zu, Param OpDesc %zu",
                          op_->GetInputsSize(), op_desc->GetInputsSize());

  GE_CHK_BOOL_EXEC(op_->GetOutputsSize() == op_desc->GetOutputsSize(),
                   REPORT_INNER_ERR_MSG("E18888",
                                        "outputs count(%zu) of param op_desc not equal to "
                                        "outputs count(%zu) of original opdesc:%s, check invalid",
                                        op_desc->GetOutputsSize(), op_->GetOutputsSize(), op_->GetName().c_str());
                   return GRAPH_PARAM_INVALID,
                          "[Check][Param] Outputs count expected to be same, original OpDesc %zu, Param OpDesc %zu",
                          op_->GetOutputsSize(), op_desc->GetOutputsSize());

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(), this->GetName(),
                   "op_desc", "", "", op_desc->GetName());
  op_ = op_desc;
  return GRAPH_SUCCESS;
}

Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> Node::NodeImpl::GetInDataNodesAndAnchors(
    const ConstNodePtr &owner_node) const {
  std::vector<std::pair<NodePtr, OutDataAnchorPtr>> vec;
  for (const auto &p : in_data_anchors_) {
    if (p == nullptr) {
      GELOGW("[Check][Param] In data anchor is nullptr, node=%s, type=%s", GetName().c_str(), GetType().c_str());
      continue;
    }
    auto anchor_ptr = p->GetPeerOutAnchor();
    if (anchor_ptr == nullptr) {
      continue;
    }
    auto node = anchor_ptr->GetOwnerNode();
    if (node == nullptr) {
      GELOGW("[Check][Param] Src node is nullptr, node=%s, type=%s", GetName().c_str(), GetType().c_str());
      continue;
    }
    vec.emplace_back(node, anchor_ptr);
  }
  return Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>(owner_node, vec);
}

Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>> Node::NodeImpl::GetOutDataNodesAndAnchors(
    const ConstNodePtr &owner_node) const {
  std::vector<std::pair<NodePtr, InDataAnchorPtr>> vec;
  for (const auto &p : out_data_anchors_) {
    if (p == nullptr) {
      GELOGW("[Check][Param] Out data anchor is nullptr, node=%s, type=%s", GetName().c_str(), GetType().c_str());
      continue;
    }
    for (const auto &in_anchor : p->GetPeerInDataAnchors()) {
      if (in_anchor == nullptr) {
        GELOGW("[Check][Param] Dst in data anchor is nullptr, node=%s, type=%s", GetName().c_str(), GetType().c_str());
        continue;
      }
      auto node = in_anchor->GetOwnerNode();
      if (node == nullptr) {
        GELOGW("[Check][Param] Dst node is nullptr, node=%s, type=%s", GetName().c_str(), GetType().c_str());
        continue;
      }
      vec.emplace_back(node, in_anchor);
    }
  }
  return Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>(owner_node, vec);
}

size_t Node::NodeImpl::GetInDataNodesSize() const {
  size_t size = 0U;
  for (const auto &in_anchor : in_data_anchors_) {
    if (in_anchor == nullptr) {
      continue;
    }
    size += in_anchor->GetPeerAnchorsSize();
  }
  return size;
}

size_t Node::NodeImpl::GetInControlNodesSize() const {
  size_t size = 0U;
  if (in_control_anchor_ != nullptr) {
    size = in_control_anchor_->GetPeerAnchorsSize();
  }
  return size;
}

size_t Node::NodeImpl::GetInNodesSize() const {
  return GetInDataNodesSize() + GetInControlNodesSize();
}

std::vector<InDataAnchor *> Node::NodeImpl::GetAllInDataAnchorsPtr() const {
  std::vector<InDataAnchor *> in_data_anchors;
  in_data_anchors.reserve(in_data_anchors_.size());
  for (const auto &in_data_anchor : in_data_anchors_) {
    in_data_anchors.emplace_back(in_data_anchor.get());
  }
  return in_data_anchors;
}

std::vector<OutDataAnchor *> Node::NodeImpl::GetAllOutDataAnchorsPtr() const {
  std::vector<OutDataAnchor *> out_data_anchors;
  out_data_anchors.reserve(out_data_anchors_.size());
  for (const auto &out_data_anchor : out_data_anchors_) {
    out_data_anchors.emplace_back(out_data_anchor.get());
  }
  return out_data_anchors;
}

Node::Node() : enable_shared_from_this(), impl_(ComGraphMakeSharedAndThrow<NodeImpl>()) {}

Node::Node(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
    : enable_shared_from_this(), impl_(ComGraphMakeSharedAndThrow<NodeImpl>(op, owner_graph)) {}

Node::~Node() {}

graphStatus Node::Init() {
  return impl_->Init(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetName() const {
  return impl_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const char *Node::GetNamePtr() const {
  return impl_->GetNamePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetType() const {
  return impl_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const char *Node::GetTypePtr() const {
  return impl_->GetTypePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeMembersAreEqual(const Node &r_node) const {
  return impl_->NodeMembersAreEqual(*(r_node.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeAnchorIsEqual(const AnchorPtr &left_anchor,
                                                                            const AnchorPtr &right_anchor,
                                                                            const size_t i) const {
  return impl_->NodeAnchorIsEqual(left_anchor, right_anchor, i);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeInConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all in data and control anchors size
  const auto in_data_anchor_size = this->GetAllInDataAnchors().size();
  const auto r_in_data_anchor_size = r_node.GetAllInDataAnchors().size();
  if (in_data_anchor_size != r_in_data_anchor_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param node in data anchors count:%zu not equal to "
                         "this in data anchors count:%zu, verify failed, node name: %s.",
                         r_in_data_anchor_size, in_data_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's in data anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto l_in_anchors = this->GetAllInAnchors();
  const auto r_in_anchors = r_node.GetAllInAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto in_control_anchor_size = l_in_anchors.size() - in_data_anchor_size;
  const auto r_in_control_anchor_size = r_in_anchors.size() - r_in_data_anchor_size;
  if (in_control_anchor_size != r_in_control_anchor_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param node in control anchors count:%zu not equal to "
                         "this in control anchors count:%zu, verify failed, node name: %s.",
                         r_in_control_anchor_size, in_control_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's in control anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  // 2.Verify all in data and control anchors connect info
  for (size_t i = 0UL; i < this->GetAllInAnchors().size(); i++) {
    // Verify data anchors
    if (i < in_data_anchor_size) {
      const auto &in_anchor = l_in_anchors.at(i);
      const auto &r_in_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_anchor, r_in_anchor, i))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's in data control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &in_control_anchor = l_in_anchors.at(i);
      const auto &r_in_control_anchor = r_in_anchors.at(i);
      if (!(NodeAnchorIsEqual(in_control_anchor, r_in_control_anchor, i - in_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's in control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::NodeOutConnectsAreEqual(const Node &r_node) const {
  // 1.Verify all out data and control anchors size
  const auto l_out_data_anchors = this->GetAllOutDataAnchors();
  const auto r_out_data_anchors = r_node.GetAllOutDataAnchors();
  const auto out_data_anchor_size = l_out_data_anchors.size();
  const auto r_out_data_anchor_size = r_out_data_anchors.size();
  if (out_data_anchor_size != r_out_data_anchor_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param node out data anchors count:%zu not equal to "
                         "this out data anchors count:%zu, verify failed, node name: %s.",
                         r_out_data_anchor_size, out_data_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's out data anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto l_out_anchors = this->GetAllOutAnchors();
  const auto r_out_anchors = r_node.GetAllOutAnchors();
  // Data anchors size equal, all anchors size not equal, means control anchor size not equal
  const auto out_control_anchor_size = l_out_anchors.size() - out_data_anchor_size;
  const auto r_out_control_anchor_size = r_out_anchors.size() - r_out_data_anchor_size;
  if (out_control_anchor_size != r_out_control_anchor_size) {
    REPORT_INNER_ERR_MSG("E18888",
                         "param node out control anchors count:%zu not equal to "
                         "this out control anchors count:%zu, verify failed, node name: %s.",
                         r_out_control_anchor_size, out_control_anchor_size, this->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of node's out control anchors verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }

  // 2.Verify all out data and control anchors connect info
  for (size_t i = 0UL; i < this->GetAllOutAnchors().size(); i++) {
    // Verify data anchors
    if (i < out_data_anchor_size) {
      const auto &out_anchor = l_out_data_anchors.at(i);
      const auto &r_out_anchor = r_out_data_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_anchor, r_out_anchor, i))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's out data control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    } else {
      // Verify control anchors
      const auto &out_control_anchor = l_out_anchors.at(i);
      const auto &r_out_control_anchor = r_out_anchors.at(i);
      if (!(NodeAnchorIsEqual(out_control_anchor, r_out_control_anchor, i - out_data_anchor_size))) {
        GELOGE(GRAPH_FAILED, "[Call][NodeAnchorIsEqual] Node's out control anchor verify failed, node name: %s.",
               this->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::operator==(const Node &r_node) const {
  return (NodeMembersAreEqual(r_node) && NodeInConnectsAreEqual(r_node) && NodeOutConnectsAreEqual(r_node));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const NodePtr &input_node) {
  return impl_->AddLinkFrom(input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const uint32_t &index,
                                                                             const NodePtr input_node) {
  return impl_->AddLinkFrom(index, input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const uint32_t &index,
                                                                             const af::Node::NodePtr input_node,
                                                                             const uint32_t input_node_index) {
  return impl_->AddLinkFrom(index, input_node, input_node_index, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFromForParse(const NodePtr &input_node) {
  return impl_->AddLinkFromForParse(input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::AddLinkFrom(const std::string &name,
                                                                             const NodePtr input_node) {
  return impl_->AddLinkFrom(name, input_node, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr Node::GetOwnerComputeGraph() const {
  return impl_->GetOwnerComputeGraph();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraph *Node::GetOwnerComputeGraphBarePtr() const {
  return impl_->GetOwnerComputeGraphBarePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::SetOwnerComputeGraph(const ComputeGraphPtr &graph) {
  return impl_->SetOwnerComputeGraph(graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::ClearOwnerGraph(const ComputeGraphPtr &graph) {
  return impl_->ClearOwnerGraph(graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<InDataAnchorPtr> Node::GetAllInDataAnchors() const {
  return impl_->GetAllInDataAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<OutDataAnchorPtr> Node::GetAllOutDataAnchors() const {
  return impl_->GetAllOutDataAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllInDataAnchorsSize() const {
  return impl_->GetAllInDataAnchorsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetAllOutDataAnchorsSize() const {
  return impl_->GetAllOutDataAnchorsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllInAnchors() const {
  return impl_->GetAllInAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Anchor *> Node::GetAllInAnchorsPtr() const {
  return impl_->GetAllInAnchorsPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<AnchorPtr> Node::GetAllOutAnchors() const {
  return impl_->GetAllOutAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Anchor *> Node::GetAllOutAnchorsPtr() const {
  return impl_->GetAllOutAnchorsPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InDataAnchorPtr Node::GetInDataAnchor(const int32_t idx) const {
  return impl_->GetInDataAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetInAnchor(const int32_t idx) const {
  return impl_->GetInAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorPtr Node::GetOutAnchor(const int32_t idx) const {
  return impl_->GetOutAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutDataAnchorPtr Node::GetOutDataAnchor(const int32_t idx) const {
  return impl_->GetOutDataAnchor(idx);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InControlAnchorPtr Node::GetInControlAnchor() const {
  return impl_->GetInControlAnchor();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OutControlAnchorPtr Node::GetOutControlAnchor() const {
  return impl_->GetOutControlAnchor();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInNodes() const {
  return impl_->GetInNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::IsAllInNodesSeen(
    const std::unordered_set<Node *> &nodes_seen) const {
  return impl_->IsAllInNodesSeen(nodes_seen);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInDataNodes() const {
  return impl_->GetInDataNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInControlNodes() const {
  return impl_->GetInControlNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutNodes() const {
  return impl_->GetOutNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetInAllNodes() const {
  return impl_->GetInAllNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutDataNodes() const {
  return impl_->GetOutDataNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> Node::GetOutDataNodesPtr() const {
  return impl_->GetOutDataNodesPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetOutDataNodesSize() const {
  return impl_->GetOutDataNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetOutControlNodesSize() const {
  return impl_->GetOutControlNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t Node::GetOutNodesSize() const {
  return impl_->GetOutNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutControlNodes() const {
  return impl_->GetOutControlNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<NodePtr> Node::GetOutAllNodes() const {
  return impl_->GetOutAllNodes(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr Node::GetOpDesc() const {
  return impl_->GetOpDesc();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc *Node::GetOpDescBarePtr() const {
  return impl_->GetOpDescBarePtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Node::UpdateOpDesc(const OpDescPtr &op_desc) {
  return impl_->UpdateOpDesc(op_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>
Node::GetInDataNodesAndAnchors() const {
  return impl_->GetInDataNodesAndAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Node::Vistor<std::pair<NodePtr, InDataAnchorPtr>>
Node::GetOutDataNodesAndAnchors() const {
  return impl_->GetOutDataNodesAndAnchors(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::AddSendEventId(const uint32_t event_id) {
  impl_->AddSendEventId(event_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::AddRecvEventId(const uint32_t event_id) {
  impl_->AddRecvEventId(event_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<uint32_t> &Node::GetSendEventIdList() const {
  return impl_->GetSendEventIdList();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<uint32_t> &Node::GetRecvEventIdList() const {
  return impl_->GetRecvEventIdList();
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::GetFusionInputFlowList(
    kFusionDataFlowVec_t &fusion_input_list) {
  impl_->GetFusionInputFlowList(fusion_input_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::GetFusionOutputFlowList(
    kFusionDataFlowVec_t &fusion_output_list) {
  impl_->GetFusionOutputFlowList(fusion_output_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetFusionInputFlowList(
    const kFusionDataFlowVec_t &fusion_input_list) {
  impl_->SetFusionInputFlowList(fusion_input_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetFusionOutputFlowList(
    const kFusionDataFlowVec_t &fusion_output_list) {
  impl_->SetFusionOutputFlowList(fusion_output_list);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool Node::GetHostNode() const {
  return impl_->GetHostNode();
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetHostNode(const bool is_host) {
  impl_->SetHostNode(is_host);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void Node::SetOrigNode(const NodePtr &orignode) {
  impl_->SetOrigNode(orignode);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr Node::GetOrigNode() {
  return impl_->GetOrigNode();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t Node::GetInDataNodesSize() const {
  return impl_->GetInDataNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t Node::GetInControlNodesSize() const {
  return impl_->GetInControlNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t Node::GetInNodesSize() const {
  return impl_->GetInNodesSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> Node::GetInNodesPtr() const {
  return impl_->GetInNodesPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> Node::GetOutNodesPtr() const {
  return impl_->GetOutNodesPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<InDataAnchor *> Node::GetAllInDataAnchorsPtr() const {
  return impl_->GetAllInDataAnchorsPtr();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<OutDataAnchor *> Node::GetAllOutDataAnchorsPtr() const {
  return impl_->GetAllOutDataAnchorsPtr();
}

}  // namespace af
