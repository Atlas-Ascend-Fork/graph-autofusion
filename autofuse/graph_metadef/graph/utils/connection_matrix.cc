/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/connection_matrix.h"
#include "connection_matrix_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace af {
ConnectionMatrix::ConnectionMatrix(const ComputeGraphPtr &graph)
    : impl_(ComGraphMakeUnique<ConnectionMatrixImpl>(graph)) {}

bool ConnectionMatrix::IsConnected(const NodePtr &a, const NodePtr &b) const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->IsConnected(a, b);
}

void ConnectionMatrix::SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->SetConnectivity(inputs, node);
}

graphStatus ConnectionMatrix::Generate(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Generate(graph);
}

void ConnectionMatrix::Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->Update(graph, fusion_nodes);
}

void ConnectionMatrix::ExpandAndUpdate(const vector<NodePtr> &fusion_nodes, const std::string &node_name) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->ExpandAndUpdate(fusion_nodes, node_name);
}

ConnectionMatrixImpl::ConnectionMatrixImpl(const ComputeGraphPtr &graph) : graph_(graph) {
  const auto direct_nodes = graph->GetDirectNode();
  size_ = direct_nodes.size();
  bit_maps_.reserve(size_);
  uint64_t index_loop = 0;
  for (const auto &node : direct_nodes) {
    name_to_index_[node->GetName()] = index_loop;
    bit_maps_.emplace_back(size_);
    index_loop++;
  }
  used_ = size_;
};

ConnectionMatrixImpl::~ConnectionMatrixImpl() {
  bit_maps_.clear();
  name_to_index_.clear();
}

uint64_t ConnectionMatrixImpl::AddNode(const std::string &op_name) {
  if (used_ + 1 >= size_) {
    size_t new_size = size_ + expand_step_;
    for (auto &m : bit_maps_) {
      m.ResizeBits(new_size);
    }

    af::LargeBitmap new_bit_vector(new_size);
    bit_maps_.resize(new_size, new_bit_vector);
    for (size_t i = used_; i < new_size; ++i) {
      bit_maps_[i].SetValues(0);
    }

    size_ = new_size;
  }

  uint64_t new_index = used_;
  ++used_;
  name_to_index_[op_name] = new_index;
  return new_index;
}

void ConnectionMatrixImpl::ExpandAndUpdate(const vector<NodePtr> &fusion_nodes, const std::string &node_name) {
  uint64_t new_index = AddNode(node_name);
  af::LargeBitmap &new_bit_vector = GetBitMap(new_index);

  // update
  new_bit_vector.SetBit(new_index);
  std::vector<uint64_t> fusion_indexs(fusion_nodes.size(), 0);
  for (size_t i = 0U; i < fusion_nodes.size(); ++i) {
    auto index = GetIndex(fusion_nodes[i]);
    new_bit_vector.Or(GetBitMap(index));
    fusion_indexs[i] = index;
  }

  for (size_t i = 0; i < used_; ++i) {
    af::LargeBitmap &node_map = bit_maps_[i];
    for (size_t j = 0; j < fusion_nodes.size(); ++j) {
      if (node_map.GetBit(fusion_indexs[j])) {
        node_map.Or(new_bit_vector);
        break;
      }
    }
  }
}

graphStatus ConnectionMatrixImpl::Generate(const ComputeGraphPtr &graph) {
  auto shared_graph = graph_.lock();
  if (shared_graph == nullptr) {
    graph_ = graph;
  }
  for (auto &node : graph->GetDirectNode()) {
    const auto inputs = node->GetInAllNodes();
    SetConnectivity(inputs, node);
  }
  return GRAPH_SUCCESS;
}

void ConnectionMatrixImpl::Update(const ComputeGraphPtr &graph, const vector<NodePtr> &fusion_nodes) {
  auto shared_graph = graph_.lock();
  if (shared_graph == nullptr) {
    return;
  }
  if (graph != shared_graph) {
    GELOGW("Input graph %s is not the same one as %s when contributing to the connection matrix.",
           graph->GetName().c_str(), shared_graph->GetName().c_str());
    return;
  }
  LargeBitmap new_bit_vector(graph->GetDirectNode().size());
  new_bit_vector.SetValues(0U);
  for (size_t i = 0U; i < fusion_nodes.size(); i++) {
    new_bit_vector.Or(GetBitMap(fusion_nodes[i]));
  }
  for (auto &node : graph->GetDirectNode()) {
    bool is_connected_to_fusion = false;
    for (size_t i = 0U; i < fusion_nodes.size(); i++) {
      if (GetBitMap(node).GetBit(static_cast<size_t>(GetIndex(fusion_nodes[i])))) {
        is_connected_to_fusion = true;
        break;
      }
    }
    if (is_connected_to_fusion) {
      GetBitMap(node).Or(new_bit_vector);
    }
  }
}

void ConnectionMatrixImpl::SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node) {
  LargeBitmap &bitmap = GetBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0U);
  }

  bitmap.SetBit(static_cast<size_t>(GetIndex(node)));
  for (const NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetBitMap(input));
    }
  }
}

uint64_t ConnectionMatrixImpl::GetIndex(const std::string &op_name) const {
  const auto iter = name_to_index_.find(op_name);
  if (iter != name_to_index_.end()) {
    return iter->second;
  } else {
    GELOGW("node %s is not found in name_to_index_", op_name.c_str());
    return 0;
  }
}

uint64_t ConnectionMatrixImpl::GetIndex(const NodePtr &node) const {
  return GetIndex(node->GetName());
}

bool ConnectionMatrixImpl::IsConnected(const NodePtr &a, const NodePtr &b) const {
  return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

const LargeBitmap &ConnectionMatrixImpl::GetBitMap(const NodePtr &node) const {
  return bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(const NodePtr &node) {
  return bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(uint64_t index) {
  return bit_maps_[index];
}
}  // namespace af
