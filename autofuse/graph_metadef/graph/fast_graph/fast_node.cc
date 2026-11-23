/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/fast_graph/fast_node.h"
#include <cstddef>
#include <memory>
#include "common/checker.h"
#include "utils/ge_ir_utils.h"
#include "fast_graph_utils.h"
#include "graph/debug/ge_op_types.h"

namespace af {
namespace {
const std::vector<FastEdge *> kEmpty;
}
FastNode::FastNode() {}

FastNode::~FastNode() {}

graphStatus FastNode::Init(const OpDescPtr &op) {
  opdesc_ = op;
  data_in_num_ = op->GetAllInputsSize();
  data_out_num_ = op->GetOutputsSize();
  node_token_ = reinterpret_cast<size_t>(this);

  return Reset();
}

graphStatus FastNode::Reset() {
  if (extend_info_ != nullptr) {
    /* The node in cache. it need to init before using it. */
    in_data_edges_.clear();
    in_control_edges_.clear();
    out_data_edges_.clear();
    out_control_edges_.clear();
    out_data_edges_info_.per_edges_num.clear();

    in_data_edges_count_ = 0UL;
    in_control_edge_count_ = 0UL;
    out_control_edges_count_ = 0UL;
    out_data_edges_info_.total_num = 0UL;
    extend_info_->Clear();
  } else {
    extend_info_ = ComGraphMakeUnique<ExtendInfo>();
    GE_CHK_BOOL_EXEC(extend_info_ != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "Failed to allocate memory for extend information");
                     return GRAPH_FAILED, "[Check][Param] Failed to allocate memory for extend information");
  }

  extend_info_->UpdateInputSymbols(data_in_num_);
  extend_info_->UpdateOutputSymbols(data_out_num_);
  UpdateDataForIoNumChange();
  return GRAPH_SUCCESS;
}

void FastNode::UpdateDataForIoNumChange() {
  if ((out_data_edges_info_.per_edges_num.size() != data_out_num_) || (data_in_num_ != in_data_edges_.size()) ||
      (data_out_num_ != out_data_edges_.size())) {
    out_data_edges_info_.per_edges_num.resize(data_out_num_, 0UL);
    in_data_edges_.resize(data_in_num_, nullptr);
    out_data_edges_.resize(data_out_num_);
  }
}

OpDescPtr FastNode::GetOpDescPtr() const {
  return opdesc_;
}

OpDesc *FastNode::GetOpDescBarePtr() const {
  return opdesc_.get();
}

std::string FastNode::GetType() const {
  GE_CHK_BOOL_EXEC(opdesc_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return opdesc_->GetType();
}

std::string FastNode::GetName() const {
  GE_CHK_BOOL_EXEC(opdesc_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return std::string(), "[Check][Param] original OpDesc is nullptr");
  return opdesc_->GetName();
}

const char *FastNode::GetNamePtr() const {
  GE_CHK_BOOL_EXEC(opdesc_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return nullptr, "[Check][Param] original OpDesc is nullptr");
  return opdesc_->GetNamePtr();
}

const char *FastNode::GetTypePtr() const {
  GE_CHK_BOOL_EXEC(opdesc_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "original OpDesc is nullptr");
                   return nullptr, "[Check][Param] original OpDesc is nullptr");
  return opdesc_->GetTypePtr();
}

bool FastNode::operator==(const FastNode &r_node) const {
  return (IsEqual(name_, r_node.name_, "node.name") && IsEqual(node_token_, r_node.node_token_, "node.token") &&
          IsEqual(opdesc_, r_node.opdesc_, "node.opdesc_") && IsEqual(opdesc_, r_node.opdesc_, "node.opdesc_") &&
          IsEqual(self_ptr_, r_node.self_ptr_, "node.self_ptr_") &&
          IsEqual(data_in_num_, r_node.data_in_num_, "node.data_in_num_") &&
          IsEqual(data_out_num_, r_node.data_out_num_, "node.data_out_num_") &&
          IsEqual(in_data_edges_, r_node.in_data_edges_, "node.in_data_edges_") &&
          IsEqual(out_data_edges_, r_node.out_data_edges_, "node.out_data_edges_") &&
          IsEqual(in_control_edges_, r_node.in_control_edges_, "node.in_control_edges_") &&
          IsEqual(out_control_edges_, r_node.out_control_edges_, "node.out_control_edges_") &&
          IsEqual(in_data_edges_count_, r_node.in_data_edges_count_, "node.in_data_edges_count_") &&
          IsEqual(in_control_edge_count_, in_control_edge_count_, "node.in_control_edge_count_") &&
          IsEqual(*extend_info_, *(r_node.extend_info_), "node.extend_info_") &&
          IsEqual(out_data_edges_info_.total_num, r_node.out_data_edges_info_.total_num,
                  "node.out_data_edges_info_.total_num"));
}

graphStatus FastNode::RecordInControlEdge(FastEdge *const edge) {
  edge->in_edge_index = in_control_edges_.size();
  in_control_edges_.push_back(edge);
  in_control_edge_count_++;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::RecordOutControlEdge(FastEdge *const edge) {
  edge->out_edge_index = out_control_edges_.size();
  out_control_edges_.push_back(edge);
  out_control_edges_count_++;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::RecordInDataEdge(FastEdge *const edge, int32_t index) {
  if (!CheckDataIndexIsValid(index, DirectionType::kDirectionInType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of in edge.", index, data_in_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of in edge.", index, data_in_num_);
    return GRAPH_FAILED;
  }

  if (in_data_edges_[index] != nullptr) {
    // if index > 0 ,it is data edge. the input edges must be empty.
    REPORT_INNER_ERR_MSG("E18888", "Failed to record edge in node [%s] for multiple input.", GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Record][Edge] Failed to record edge in node [%s] for multiple input.", GetName().c_str());
    return GRAPH_FAILED;
  }

  in_data_edges_[index] = edge;
  in_data_edges_count_++;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::RecordOutDataEdge(FastEdge *const edge, int32_t index) {
  if (!CheckDataIndexIsValid(index, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    return GRAPH_FAILED;
  }

  out_data_edges_[index].push_back(edge);
  out_data_edges_info_.total_num++;
  out_data_edges_info_.per_edges_num[index]++;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::RecordEdge(FastEdge *const edge, DirectionType type) {
  if (type == DirectionType::kDirectionInType) {
    int32_t index = edge->dst_input;
    if (index == kControlEdgeIndex) {
      return RecordInControlEdge(edge);
    }
    edge->in_edge_index = 0;
    return RecordInDataEdge(edge, index);
  }

  int32_t index = edge->src_output;
  if (index == kControlEdgeIndex) {
    return RecordOutControlEdge(edge);
  }
  GE_ASSERT_TRUE(static_cast<size_t>(index) < out_data_edges_.size());
  edge->out_edge_index = out_data_edges_[index].size();
  return RecordOutDataEdge(edge, index);
}

graphStatus FastNode::EraseInControlEdge(const FastEdge *const edge) {
  GE_ASSERT_TRUE(static_cast<size_t>(edge->in_edge_index) < in_control_edges_.size());
  GE_ASSERT_TRUE(in_control_edges_[edge->in_edge_index] == edge);
  in_control_edges_[edge->in_edge_index] = nullptr;
  in_control_edge_count_--;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::EraseOutControlEdge(const FastEdge *const edge) {
  GE_ASSERT_TRUE(static_cast<size_t>(edge->out_edge_index) < out_control_edges_.size());
  GE_ASSERT_TRUE(out_control_edges_[edge->out_edge_index] == edge);
  out_control_edges_[edge->out_edge_index] = nullptr;
  out_control_edges_count_--;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::EraseInDataEdge(const FastEdge *const edge) {
  GE_ASSERT_NOTNULL(edge);
  const int32_t index = edge->dst_input;
  if (!CheckDataIndexIsValid(index, DirectionType::kDirectionInType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", index, data_in_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", index, data_in_num_);
    return GRAPH_FAILED;
  }
  GE_ASSERT_TRUE(in_data_edges_[index] == edge);
  in_data_edges_[index] = nullptr;
  in_data_edges_count_--;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::EraseOutDataEdge(const FastEdge *const edge, int32_t index) {
  if (!CheckDataIndexIsValid(index, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    return GRAPH_FAILED;
  }
  GE_ASSERT_TRUE(static_cast<size_t>(edge->out_edge_index) < out_data_edges_[index].size());
  GE_ASSERT_TRUE(out_data_edges_[index][edge->out_edge_index] == edge);
  out_data_edges_[index][edge->out_edge_index] = nullptr;
  out_data_edges_info_.total_num--;
  out_data_edges_info_.per_edges_num[index]--;
  return GRAPH_SUCCESS;
}

graphStatus FastNode::EraseEdge(const FastEdge *const edge, DirectionType type) {
  if (type == DirectionType::kDirectionOutType) {
    int32_t index = edge->src_output;
    if (index == kControlEdgeIndex) {
      return EraseOutControlEdge(edge);
    }

    return EraseOutDataEdge(edge, index);
  }

  int32_t index = edge->dst_input;
  if (index == kControlEdgeIndex) {
    return EraseInControlEdge(edge);
  }

  return EraseInDataEdge(edge);
}

graphStatus FastNode::CheckAllInputParamter(DirectionType type, int32_t io_idx, int32_t cur_array_index,
                                            int32_t replace_array_index) const {
  if (io_idx < -1) {
    REPORT_INNER_ERR_MSG("E18888", "The idx[%d] is out of range of in_edges, idx must be >= -1.", io_idx);
    GELOGE(GRAPH_FAILED, "[Check][Param] The idx[%d] is out of range of in_edges, idx must be >= -1.", io_idx);
    return GRAPH_FAILED;
  }

  size_t io_size = 0UL;
  size_t edge_size = 0UL;

  if (io_idx != kControlEdgeIndex) {
    if (type == DirectionType::kDirectionInType) {
      io_size = data_in_num_;
    } else if (type == DirectionType::kDirectionOutType) {
      io_size = data_out_num_;
    }

    if (io_size <= static_cast<size_t>(io_idx)) {
      REPORT_INNER_ERR_MSG("E18888", "The idx [%d] exceeds the max capacity [%zu] of in_edges.", io_idx, io_size);
      GELOGE(GRAPH_FAILED, "[Check][Param] The idx [%d] exceeds the max capacity [%zu] of in_edges.", io_idx, io_size);
      return GRAPH_FAILED;
    }
  }

  if (io_idx == kControlEdgeIndex) {
    if (type == DirectionType::kDirectionInType) {
      edge_size = in_control_edges_.size();
    } else if (type == DirectionType::kDirectionOutType) {
      edge_size = out_control_edges_.size();
    }
  } else {
    if (type == DirectionType::kDirectionInType) {
      edge_size = 1;
    } else if (type == DirectionType::kDirectionOutType) {
      edge_size = out_data_edges_[io_idx].size();
    }
  }

  if ((edge_size <= static_cast<size_t>(replace_array_index)) || (edge_size <= static_cast<size_t>(cur_array_index))) {
    REPORT_INNER_ERR_MSG("E18888",
                         "The replace index [%d] or current index [%d] exceeds the max capacity [%zu] of in_edges.",
                         replace_array_index, cur_array_index, edge_size);
    GELOGE(GRAPH_FAILED,
           "[Check][Param] The replace index [%d] or current index [%d] exceeds the max capacity [%zu] of in_edges.",
           replace_array_index, cur_array_index, edge_size);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus FastNode::MoveEdge(DirectionType type, int32_t io_idx, int32_t cur_array_index,
                               int32_t replace_array_index) {
  auto ret = CheckAllInputParamter(type, io_idx, cur_array_index, replace_array_index);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  if (type == DirectionType::kDirectionInType) {
    FastEdge *edge = nullptr;
    if (io_idx == kControlEdgeIndex) {
      in_control_edges_[replace_array_index] = in_control_edges_[cur_array_index];
      in_control_edges_[cur_array_index] = nullptr;
      edge = in_control_edges_[replace_array_index];
    } else {
      // don`t need to do something.
    }

    if (edge != nullptr) {
      edge->in_edge_index = replace_array_index;
    }
  } else if (type == DirectionType::kDirectionOutType) {
    FastEdge *edge = nullptr;
    if (io_idx == kControlEdgeIndex) {
      out_control_edges_[replace_array_index] = out_control_edges_[cur_array_index];
      out_control_edges_[cur_array_index] = nullptr;
      edge = out_control_edges_[replace_array_index];
    } else {
      out_data_edges_[io_idx][replace_array_index] = out_data_edges_[io_idx][cur_array_index];
      out_data_edges_[io_idx][cur_array_index] = nullptr;
      edge = out_data_edges_[io_idx][replace_array_index];
    }
    if (edge != nullptr) {
      edge->out_edge_index = replace_array_index;
    }
  }

  return GRAPH_SUCCESS;
}

size_t FastNode::GetAllInEdgeSize() const {
  return in_control_edge_count_ + in_data_edges_count_;
}

const std::vector<Edge<FastNode> *> &FastNode::GetAllInDataEdgesRef() const {
  return in_data_edges_;
}

std::vector<Edge<FastNode> *> &FastNode::MutableAllInDataEdges() {
  return in_data_edges_;
}

const std::vector<Edge<FastNode> *> &FastNode::GetAllInControlEdgesRef() const {
  return in_control_edges_;
}

const std::vector<Edge<FastNode> *> &FastNode::GetAllOutControlEdgesRef() const {
  return out_control_edges_;
}

const std::vector<std::vector<Edge<FastNode> *>> &FastNode::GetAllOutDataEdgesRef() const {
  return out_data_edges_;
}

std::vector<Edge<FastNode> *> FastNode::GetAllInDataEdges() const {
  std::vector<FastEdge *> tmp;
  tmp.reserve(in_data_edges_count_);

  std::for_each(in_data_edges_.begin(), in_data_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge);
    }
  });
  return tmp;
}

std::vector<Edge<FastNode> *> FastNode::GetAllInControlEdges() const {
  std::vector<FastEdge *> tmp;
  tmp.reserve(in_control_edge_count_);

  std::for_each(in_control_edges_.begin(), in_control_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge);
    }
  });
  return tmp;
}

std::vector<Edge<FastNode> *> FastNode::GetAllOutControlEdges() const {
  std::vector<FastEdge *> tmp;
  tmp.reserve(out_control_edges_count_);

  std::for_each(out_control_edges_.begin(), out_control_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge);
    }
  });
  return tmp;
}

std::vector<Edge<FastNode> *> FastNode::GetAllOutDataEdges() const {
  std::vector<FastEdge *> tmp;
  tmp.reserve(out_data_edges_info_.total_num);

  for (size_t i = 0UL; i < out_data_edges_.size(); i++) {
    std::for_each(out_data_edges_[i].begin(), out_data_edges_[i].end(), [&tmp](FastEdge *edge) {
      if (edge != nullptr) {
        tmp.push_back(edge);
      }
    });
  }
  return tmp;
}

inline bool FastNode::CheckDataIndexIsValid(int32_t index, DirectionType type) const {
  if (type == DirectionType::kDirectionOutType) {
    return ((index >= 0) && (index < static_cast<int32_t>(data_out_num_)));
  } else if (type == DirectionType::kDirectionInType) {
    return ((index >= 0) && (index < static_cast<int32_t>(data_in_num_)));
  }
  return false;
}

bool FastNode::OutNodesIsEmpty() const {
  return (out_data_edges_info_.total_num + out_control_edges_count_ == 0);
}

size_t FastNode::GetAllOutEdgesSize() const {
  return out_control_edges_count_ + out_data_edges_info_.total_num;
}

size_t FastNode::GetAllOutDataEdgesSize() const {
  return out_data_edges_info_.total_num;
}

size_t FastNode::GetAllOutControlEdgesSize() const {
  return out_control_edges_count_;
}

size_t FastNode::GetAllInDataEdgesSize() const {
  return in_data_edges_count_;
}

size_t FastNode::GetAllInControlEdgesSize() const {
  return in_control_edge_count_;
}

std::vector<FastNode *> FastNode::GetAllOutNodes() const {
  std::vector<FastNode *> tmp;
  tmp.reserve(out_control_edges_count_ + out_data_edges_info_.total_num);
  for (size_t i = 0UL; i < out_data_edges_.size(); i++) {
    std::for_each(out_data_edges_[i].begin(), out_data_edges_[i].end(), [&tmp](FastEdge *edge) {
      if (edge != nullptr) {
        tmp.push_back(edge->dst);
      }
    });
  }

  std::for_each(out_control_edges_.begin(), out_control_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge->dst);
    }
  });

  return tmp;
}

std::vector<FastNode *> FastNode::GetAllInNodes() const {
  std::vector<FastNode *> tmp;
  tmp.reserve(in_control_edge_count_ + in_data_edges_count_);
  std::for_each(in_data_edges_.begin(), in_data_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge->src);
    }
  });

  std::for_each(in_control_edges_.begin(), in_control_edges_.end(), [&tmp](FastEdge *edge) {
    if (edge != nullptr) {
      tmp.push_back(edge->src);
    }
  });

  return tmp;
}

std::vector<FastNode *> FastNode::GetInDataNodes() const {
  std::vector<FastNode *> in_data_nodes;
  in_data_nodes.reserve(in_data_edges_count_);

  auto &ref = GetAllInDataEdgesRef();
  std::for_each(ref.begin(), ref.end(), [&in_data_nodes](FastEdge *edge) {
    if (edge != nullptr) {
      in_data_nodes.push_back(edge->src);
    }
  });

  return in_data_nodes;
}

std::vector<FastNode *> FastNode::GetOutDataNodesByIndex(int32_t index) const {
  std::vector<FastNode *> out_data_nodes;

  if (!CheckDataIndexIsValid(index, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", index, data_out_num_);
    return out_data_nodes;
  }

  out_data_nodes.reserve(out_data_edges_info_.per_edges_num[index]);

  auto &ref = GetOutEdgesRefByIndex(index);
  std::for_each(ref.begin(), ref.end(), [&out_data_nodes](FastEdge *edge) {
    if (edge != nullptr) {
      out_data_nodes.push_back(edge->dst);
    }
  });

  return out_data_nodes;
}

std::vector<FastNode *> FastNode::GetOutDataNodes() const {
  std::vector<FastNode *> out_nodes;

  out_nodes.reserve(out_data_edges_info_.total_num);
  for (size_t i = 0UL; i < out_data_edges_.size(); i++) {
    std::for_each(out_data_edges_[i].begin(), out_data_edges_[i].end(), [&out_nodes](FastEdge *edge) {
      if (edge != nullptr) {
        out_nodes.push_back(edge->dst);
      }
    });
  }

  return out_nodes;
}

std::vector<FastNode *> FastNode::GetOutControlNodes() const {
  std::vector<FastNode *> out_ctrl_nodes;

  out_ctrl_nodes.reserve(out_control_edges_count_);
  for (const auto &edge : out_control_edges_) {
    if (edge != nullptr) {
      out_ctrl_nodes.push_back(edge->dst);
    }
  }
  return out_ctrl_nodes;
}

std::vector<FastNode *> FastNode::GetInControlNodes() const {
  std::vector<FastNode *> in_ctrl_nodes;

  in_ctrl_nodes.reserve(in_control_edge_count_);
  for (const auto &edge : in_control_edges_) {
    if (edge != nullptr) {
      in_ctrl_nodes.push_back(edge->src);
    }
  }
  return in_ctrl_nodes;
}

size_t FastNode::GetNodeToken() const {
  return node_token_;
}

size_t FastNode::GetInEdgesSizeByIndex(int32_t idx) const {
  if (idx == kControlEdgeIndex) {
    return in_control_edge_count_;
  }

  if (!CheckDataIndexIsValid(idx, DirectionType::kDirectionInType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", idx, data_in_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", idx, data_in_num_);
    return 0UL;
  }

  if (in_data_edges_[idx] != nullptr) {
    return 1UL;
  }

  return 0UL;
}

size_t FastNode::GetOutEdgesSizeByIndex(int32_t idx) const {
  if (idx == kControlEdgeIndex) {
    return out_control_edges_count_;
  }

  if (!CheckDataIndexIsValid(idx, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", idx + 1, data_in_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", idx + 1, data_in_num_);
    return 0UL;
  }

  if (out_data_edges_info_.per_edges_num.size() <= static_cast<size_t>(idx)) {
    return 0UL;
  }

  return out_data_edges_info_.per_edges_num[idx];
}

Edge<FastNode> *FastNode::GetInDataEdgeByIndex(int32_t idx) const {
  if (!CheckDataIndexIsValid(idx, DirectionType::kDirectionInType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of in edge.", idx, data_in_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of in edge.", idx, data_in_num_);
    return nullptr;
  }

  return in_data_edges_[idx];
}

bool FastNode::IsDirectlyControlledByNode(FastNode const *node) const {
  for (const auto in_ctrl_edge : in_control_edges_) {
    if ((in_ctrl_edge != nullptr) && (in_ctrl_edge->src != nullptr) && (in_ctrl_edge->src == node)) {
      return true;
    }
  }
  return false;
}
std::vector<Edge<FastNode> *> FastNode::GetOutEdgesByIndex(int32_t idx) const {
  if (idx == kControlEdgeIndex) {
    return out_control_edges_;
  }

  if (!CheckDataIndexIsValid(idx, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", idx, data_out_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", idx, data_out_num_);
    return std::vector<Edge<FastNode> *>{};
  }

  std::vector<FastEdge *> tmp;
  tmp.reserve(out_data_edges_info_.per_edges_num[idx]);

  for (size_t i = 0UL; i < out_data_edges_[idx].size(); i++) {
    auto edge = out_data_edges_[idx][i];
    if (edge != nullptr) {
      tmp.push_back(edge);
    }
  }

  return tmp;
}

const std::vector<Edge<FastNode> *> &FastNode::GetOutEdgesRefByIndex(int32_t idx) const {
  if (idx == kControlEdgeIndex) {
    return out_control_edges_;
  }

  if (!CheckDataIndexIsValid(idx, DirectionType::kDirectionOutType)) {
    REPORT_INNER_ERR_MSG("E18888", "The index [%d] exceeds the size [%zu] of out edge.", idx, data_out_num_);
    GELOGE(GRAPH_FAILED, "The index [%d] exceeds the size [%zu] of out edge.", idx, data_out_num_);
    return kEmpty;
  }

  return out_data_edges_[idx];
}

graphStatus FastNode::ModifySizeByNodeType(const FastEdge *const fast_edge, size_t &in_edge_size) const {
  if ((fast_edge != nullptr) && (fast_edge->src != nullptr)) {
    auto type = fast_edge->src->GetType();
    if ((strcmp(type.c_str(), NEXTITERATION) == 0) || (strcmp(type.c_str(), REFNEXTITERATION) == 0)) {
      GE_IF_BOOL_EXEC(in_edge_size == 0UL,
                      GELOGE(GRAPH_FAILED, "[Check][Param] If [in_edge_size = 0], the result will be reversed");
                      return GRAPH_FAILED);
      in_edge_size--;
    }
  }

  return GRAPH_SUCCESS;
}

size_t FastNode::GetInEdgeSize() const {
  size_t in_edge_size = GetAllInEdgeSize();
  auto &edges = GetAllInDataEdgesRef();
  for (size_t i = 0UL; i < edges.size(); i++) {
    auto edge = edges[i];
    if (edge == nullptr) {
      continue;
    }
    auto ret = ModifySizeByNodeType(edge, in_edge_size);
    if (ret != GRAPH_SUCCESS) {
      return 0;
    }
  }
  return in_edge_size;
}

void FastNode::RemoveAllEdge(std::function<void(Edge<FastNode> *)> const &remove_edge_func) {
  for (size_t i = 0UL; i < in_data_edges_.size(); ++i) {
    auto edge = in_data_edges_[i];
    if (edge != nullptr) {
      remove_edge_func(edge);
    }
  }

  for (size_t i = 0UL; i < in_control_edges_.size(); ++i) {
    auto edge = in_control_edges_[i];
    if (edge != nullptr) {
      remove_edge_func(edge);
    }
  }

  for (size_t i = 0UL; i < out_data_edges_.size(); ++i) {
    for (size_t j = 0UL; j < out_data_edges_[i].size(); ++j) {
      auto edge = out_data_edges_[i][j];
      if (edge != nullptr) {
        remove_edge_func(edge);
      }
    }
  }

  for (size_t i = 0UL; i < out_control_edges_.size(); ++i) {
    auto edge = out_control_edges_[i];
    if (edge != nullptr) {
      remove_edge_func(edge);
    }
  }

  return;
}

size_t FastNode::GetDataInNum() const {
  return data_in_num_;
}

size_t FastNode::GetDataOutNum() const {
  return data_out_num_;
}

void FastNode::UpdateDataInNum(size_t new_num) {
  data_in_num_ = new_num;
  UpdateDataForIoNumChange();
  extend_info_->UpdateInputSymbols(data_in_num_);
}

void FastNode::UpdateDataOutNum(size_t new_num) {
  data_out_num_ = new_num;
  UpdateDataForIoNumChange();
  extend_info_->UpdateOutputSymbols(data_out_num_);
}

void FastNode::SetNodePtr(const std::shared_ptr<Node> &node) {
  self_ptr_ = node;
  node_bare_ptr_ = node.get();
}

void FastNode::ClearNodePtr() {
  self_ptr_ = nullptr;
}

void FastNode::ClearNodeBarePtr() {
  node_bare_ptr_ = nullptr;
}

std::shared_ptr<Node> FastNode::GetNodePtr() const {
  if (self_ptr_ != nullptr) {
    return self_ptr_;
  }

  if (node_bare_ptr_ != nullptr) {
    return node_bare_ptr_->shared_from_this();
  }

  return nullptr;
}

Node *FastNode::GetNodeBarePtr() const {
  return node_bare_ptr_;
}

void FastNode::UpdateOpDesc(const OpDescPtr &new_opdesc) {
  if (new_opdesc == nullptr) {
    opdesc_.reset();
    return;
  }
  opdesc_ = new_opdesc;
}

ExtendInfo *FastNode::GetExtendInfo() const {
  return extend_info_.get();
}

void ExtendInfo::Clear() {
  execute_graph_ = nullptr;
  output_index_.clear();
  input_index_ = kControlEdgeIndex;
  host_node_ = false;
  input_symbols_.clear();
  output_symbols_.clear();
}

void ExtendInfo::SetInputIndex(int32_t idx) {
  input_index_ = idx;
}

int32_t ExtendInfo::GetInputIndex() const {
  return input_index_;
}

void ExtendInfo::AddOneOutputIndex(int32_t idx) {
  output_index_.push_back(idx);
}

std::vector<int32_t> &ExtendInfo::GetOutputIndex() {
  return output_index_;
}

ExecuteGraph *ExtendInfo::GetOwnerGraphBarePtr() const {
  return execute_graph_;
}

graphStatus ExtendInfo::SetOwnerGraph(ExecuteGraph *const graph, const FastNode *const fast_node) {
  if ((execute_graph_ != nullptr) && (graph != execute_graph_)) {
    auto quick_node = FastGraphUtils::GetListElementAddr(fast_node);
    auto owner = quick_node->owner;
    auto mode = quick_node->mode;
    if ((owner != nullptr) && (mode == ListMode::kFreeMode)) {
      owner->erase(quick_node);
    }
  }
  execute_graph_ = graph;
  return GRAPH_SUCCESS;
}

bool ExtendInfo::operator==(const ExtendInfo &r_info) const {
  return (IsEqual(execute_graph_, r_info.execute_graph_, "node.execute_graph_") &&
          IsEqual(input_index_, r_info.input_index_, "node.input_index_") &&
          IsEqual(output_index_, r_info.output_index_, "node.output_index_"));
}

bool ExtendInfo::GetHostNode() const {
  return host_node_;
}

void ExtendInfo::SetHostNode(const bool is_host) {
  host_node_ = is_host;
}

void ExtendInfo::UpdateInputSymbols(size_t data_in_num) {
  input_symbols_.resize(data_in_num, kInvalidSymbol);
}

void ExtendInfo::UpdateOutputSymbols(size_t data_out_num) {
  output_symbols_.resize(data_out_num, kInvalidSymbol);
}

graphStatus ExtendInfo::SetInputSymbol(size_t idx, uint64_t symbol) {
  if (!IsDataIndexValid(idx, input_symbols_)) {
    return GRAPH_FAILED;
  }
  input_symbols_[idx] = symbol;
  return GRAPH_SUCCESS;
}

graphStatus ExtendInfo::SetOutputSymbol(size_t idx, uint64_t symbol) {
  if (!IsDataIndexValid(idx, output_symbols_)) {
    return GRAPH_FAILED;
  }
  output_symbols_[idx] = symbol;
  return GRAPH_SUCCESS;
}

uint64_t ExtendInfo::GetInputSymbol(size_t idx) {
  if (!IsDataIndexValid(idx, input_symbols_)) {
    return kInvalidSymbol;
  }
  return input_symbols_[idx];
}

uint64_t ExtendInfo::GetOutputSymbol(size_t idx) {
  if (!IsDataIndexValid(idx, output_symbols_)) {
    return kInvalidSymbol;
  }
  return output_symbols_[idx];
}

bool ExtendInfo::IsDataIndexValid(size_t idx, const std::vector<uint64_t> &symbols) const {
  GE_ASSERT_TRUE(idx < symbols.size(), "The index [%zu] exceeds the size [%zu] of symbols.", idx, symbols.size());
  return true;
}
}  // namespace af
