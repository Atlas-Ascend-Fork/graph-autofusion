/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "broadcast_backward_pass.h"
#include "broadcast_backward_shared_split.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "ascgraph_info_complete.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"

using namespace ascir;
using namespace af::ascir_op;
using namespace af::ops;

namespace optimize {
namespace {

using af::AscGraph;
using af::AscNode;
using af::AscNodePtr;
using af::AscTensorAttr;
using af::Expression;
using af::FAILED;
using af::SUCCESS;
using NodePtr = af::AscNodePtr;

constexpr const char *kStoreType = Store::Type;
constexpr const char *kScalarType = Scalar::Type;
constexpr const char *kBroadcastType = Broadcast::Type;
constexpr const char *kCastType = Cast::Type;
constexpr int64_t kMinBroadcastInputCount = 2;

// scalar-like 生产者（Scalar/ScalarData/IndexExpr）的输出视图为空（无 axis/repeats/strides），
// 不满足广播后移要求的前后视图轴数一致前提；后移前必须按 scalar 输入支持性门控。
bool IsScalarLikeProducer(const NodePtr &node) {
  return node->GetType() == Scalar::Type || node->GetType() == ScalarData::Type || node->GetType() == IndexExpr::Type;
}

std::vector<std::string> view_op_type = {Transpose::Type, Broadcast::Type, "Slice", Split::Type, Concat::Type,
                                         Gather::Type,    "Sum",           "Mean",  "Max",       "Min",
                                         "Prod",          "Any",           "All"};

// -------------------- compat shim（替代 GE asc_adapt:: / BackendUtils:: / AutofuseUtils::） --------------------

AscNodePtr ToAscNode(const af::NodePtr &node) {
  return std::dynamic_pointer_cast<AscNode>(node);
}

bool IsEqOne(const Expression &expr) {
  return af::SymbolicUtils::StaticCheckEq(expr, af::sym::kSymbolOne) == af::TriBool::kTrue;
}

bool IsEqZero(const Expression &expr) {
  return af::SymbolicUtils::StaticCheckEq(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

std::string VectorToStr(const std::vector<bool> &vec) {
  std::string str = "[";
  for (size_t i = 0U; i < vec.size(); ++i) {
    if (i > 0U) {
      str += ", ";
    }
    str += vec[i] ? "true" : "false";
  }
  str += "]";
  return str;
}

Status GetPeerInNodes(const NodePtr &node, std::vector<NodePtr> &vec, int32_t idx) {
  auto out_anchor = node->GetOutDataAnchor(idx);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &peer_in : out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in);
    vec.push_back(ToAscNode(peer_in->GetOwnerNode()));
  }
  return SUCCESS;
}

Status GetPeerOutNode(const NodePtr &node, NodePtr &peer, int32_t idx) {
  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  peer = ToAscNode(out_anchor->GetOwnerNode());
  return SUCCESS;
}

Status GetPeerOutNodes(const NodePtr &node, std::vector<NodePtr> &vec) {
  auto in_size = node->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < in_size; ++i) {
    auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    if (in_anchor == nullptr) {
      continue;
    }
    auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }
    vec.push_back(ToAscNode(out_anchor->GetOwnerNode()));
  }
  return SUCCESS;
}

Status GetOutputTensorAttrAt(const NodePtr &node, AscTensorAttr *&attr, uint32_t output_index) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() > output_index);
  attr = &node->outputs[output_index].attr;
  return SUCCESS;
}

Status GetOutputTensorAttr(const NodePtr &node, AscTensorAttr *&attr) {
  return GetOutputTensorAttrAt(node, attr, 0U);
}

bool IsSingleInAndOutNode(const NodePtr &node) {
  return node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() == 1UL;
}

bool IsSingleInNode(const NodePtr &node) {
  return node->GetInDataNodesSize() == 1UL;
}

bool IsSingleOutNode(const NodePtr &node) {
  if (node->GetAllOutDataAnchorsSize() != 1U) {
    return false;
  }
  auto out_anchor = node->GetOutDataAnchor(0);
  if (out_anchor == nullptr) {
    return false;
  }
  return out_anchor->GetPeerInDataAnchors().size() == 1U;
}

bool HasNoControlEdges(const NodePtr &node) {
  return node != nullptr && node->GetInControlNodesSize() == 0U && node->GetOutControlNodesSize() == 0U;
}

bool IsIndirectLoadBoundary(const NodePtr &node) {
  if (IsOps<af::ascir_op::IndirectLoad>(node)) {
    return true;
  }
  auto out_anchor_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t i = 0U; i < out_anchor_size; ++i) {
    std::vector<NodePtr> peer_in_nodes;
    if (GetPeerInNodes(node, peer_in_nodes, static_cast<int32_t>(i)) == SUCCESS) {
      for (const auto &peer_in_node : peer_in_nodes) {
        if (peer_in_node != nullptr && IsOps<af::ascir_op::IndirectLoad>(peer_in_node)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool ChainReachesIndirectLoad(const NodePtr &start_node) {
  std::unordered_set<NodePtr> visited;
  std::vector<NodePtr> pending = {start_node};
  while (!pending.empty()) {
    NodePtr cur_node = pending.back();
    pending.pop_back();
    if (cur_node == nullptr || !visited.insert(cur_node).second) {
      continue;
    }
    if (IsIndirectLoadBoundary(cur_node)) {
      return true;
    }
    const auto out_anchor_size = cur_node->GetAllOutDataAnchorsSize();
    for (uint32_t i = 0U; i < out_anchor_size; ++i) {
      std::vector<NodePtr> peer_in_nodes;
      if (GetPeerInNodes(cur_node, peer_in_nodes, static_cast<int32_t>(i)) == SUCCESS) {
        for (const auto &peer_in_node : peer_in_nodes) {
          if (peer_in_node != nullptr) {
            pending.push_back(peer_in_node);
          }
        }
      }
    }
  }
  return false;
}

bool IsNextViewOp(const NodePtr &next_node);

// 广播后移会把 Broadcast 搬到 compute 链的输出侧；若下游存在 view 算子，跨该 view 的统一 broadcast 展开会
// 破坏其轴语义（Concat 沿轴逐输入拼接、Slice 裁剪、Split 切分、Gather 索引、Reduce 归约、Transpose 重排等），
// 因此该场景必须禁止后移；Broadcast 本身可合并除外。从 start_node 出发（含自身）沿后继 BFS，命中非 Broadcast
// 的 view 算子时返回 true。
bool DownstreamViewOpBlocksBackward(const NodePtr &start_node) {
  if (start_node == nullptr) {
    return false;
  }
  std::unordered_set<NodePtr> visited;
  std::vector<NodePtr> pending = {start_node};
  while (!pending.empty()) {
    NodePtr cur_node = pending.back();
    pending.pop_back();
    if (cur_node == nullptr || !visited.insert(cur_node).second) {
      continue;
    }
    if (cur_node->GetType() != kBroadcastType && IsNextViewOp(cur_node)) {
      GELOGI("Downstream view op[%s](%s) blocks broadcast backward.", cur_node->GetName().c_str(),
             cur_node->GetType().c_str());
      return true;
    }
    const auto out_anchor_size = cur_node->GetAllOutDataAnchorsSize();
    for (uint32_t i = 0U; i < out_anchor_size; ++i) {
      std::vector<NodePtr> peer_in_nodes;
      if (GetPeerInNodes(cur_node, peer_in_nodes, static_cast<int32_t>(i)) == SUCCESS) {
        for (const auto &peer_in_node : peer_in_nodes) {
          if (peer_in_node != nullptr) {
            pending.push_back(peer_in_node);
          }
        }
      }
    }
  }
  return false;
}

void RemoveDuplicates(std::vector<NodePtr> &vec) {
  std::unordered_set<NodePtr> seen;
  seen.reserve(vec.size());
  vec.erase(std::remove_if(vec.begin(), vec.end(), [&seen](const NodePtr &node) { return !seen.insert(node).second; }),
            vec.end());
}

// -------------------- 算法逻辑（移植自 GE broadcast_backward_pass.cpp） --------------------

Status GetSingleNextNode(NodePtr &node, NodePtr &peer_in_node) {
  std::vector<NodePtr> peer_in_nodes;
  GE_ASSERT_SUCCESS(GetPeerInNodes(node, peer_in_nodes, 0));

  if (peer_in_nodes.size() != 1U) {
    GELOGI("node:%s(%s) has %zu peer out nodes", node->GetName().c_str(), node->GetType().c_str(),
           peer_in_nodes.size());
    return FAILED;
  }
  peer_in_node = peer_in_nodes.at(0);
  return SUCCESS;
}

Status GetPeerOutNodeSafe(const NodePtr &node, NodePtr &peer_out_node, int32_t idx) {
  GE_ASSERT_NOTNULL(node);

  if (node->GetAllInDataAnchorsSize() <= static_cast<size_t>(idx)) {
    return FAILED;
  }

  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor);

  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);

  return GetPeerOutNode(node, peer_out_node, idx);
}

// 获取 peer 对应分支上的 Broadcast 节点，支持 Broadcast 直接连接 peer，或经单输入 Cast 连接 peer。
// 多输入场景下，每个 Broadcast 只扩展一个维度；当各分支存在公共 B 轴时，只后移公共 B 轴对应的 Broadcast。
// 若 peer 是 Cast，则沿其唯一输入边穿透 Cast，继续查找前置的 Broadcast(B)：
//   后移前：  x0 -> Broadcast(A) -> Broadcast(B) ------+
//                                                     +-> MultiInputOp
//            x1 -> Broadcast(C) -> Broadcast(B) -> Cast-+
//   后移后：  x0 -> Broadcast(A) ----------------------+
//                                                     +-> MultiInputOp -> Broadcast(B)
//            x1 -> Broadcast(C) -> Cast --------------+
// 返回值表示是否找到符合条件的 Broadcast；retained_cast_peer 用于保留 Broadcast 后面的 Cast 节点。
bool GetBroadcastPeerThroughCast(const NodePtr &peer, NodePtr &broadcast_peer, NodePtr &retained_cast_peer) {
  broadcast_peer = nullptr;
  retained_cast_peer = nullptr;
  if (peer == nullptr) {
    return false;
  }
  if (peer->GetType() == kBroadcastType) {
    broadcast_peer = peer;
    return true;
  }
  // 多消费 Cast 不穿透：公共轴后移会在 UpdateOutputTensor 中塌缩 retained Cast 的
  // 输出视图，多消费时殃及全部消费分支（另一路 shape 不一致）。共享 Cast 由
  // SplitSharedBroadcastConsumers 先整链拆分为单链后再穿透；拆分 fail-closed（如
  // Cast 带控制边）的共享 Cast 到此一并拒绝，fail-closed 保持图原状。
  if (peer->GetType() != kCastType || !IsSingleInNode(peer) || !IsSingleOutNode(peer)) {
    return false;
  }

  NodePtr pre_node;
  if (GetPeerOutNodeSafe(peer, pre_node, 0) != SUCCESS || pre_node == nullptr ||
      pre_node->GetType() != kBroadcastType) {
    return false;
  }
  broadcast_peer = pre_node;
  retained_cast_peer = peer;
  return true;
}

bool IsNextViewOp(const NodePtr &next_node) {
  std::string type = next_node->GetType();
  return std::find(view_op_type.begin(), view_op_type.end(), type) != view_op_type.end();
}

bool IsDtypeNotSupportOp(const NodePtr &next_node, af::DataType &output_dtype) {
  std::vector<af::DataType> input_dtypes;
  std::vector<af::DataType> expect_output_dtypes;
  const auto output_tensor_desc = next_node->GetOpDesc()->MutableOutputDesc(0);
  output_dtype = output_tensor_desc->GetDataType();
  expect_output_dtypes.push_back(output_dtype);
  input_dtypes.push_back(output_dtype);
  return (next_node->GetType() == kCastType) &&
         (ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, expect_output_dtypes) != SUCCESS);
}

Status ReverseCollectBrcNodes(const NodePtr &node, std::vector<NodePtr> &brc_nodes) {
  NodePtr cur_node = node;
  while ((cur_node->GetType() == kBroadcastType) && IsSingleInAndOutNode(cur_node)) {
    brc_nodes.push_back(cur_node);
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, cur_node, 0));
  }
  return SUCCESS;
}

Status GetBrcAxisFromNode(const NodePtr &brc_node, int64_t &brc_axis) {
  brc_axis = -1;
  NodePtr pre_brc_node;
  GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(brc_node, pre_brc_node, 0));
  AscTensorAttr *pre_brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_brc_node, pre_brc_output_attr));
  auto pre_brc_repeats = pre_brc_output_attr->repeats;
  auto pre_brc_strides = pre_brc_output_attr->strides;
  auto pre_brc_axis = pre_brc_output_attr->axis;

  AscTensorAttr *brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(brc_node, brc_output_attr));
  auto brc_repeats = brc_output_attr->repeats;
  auto brc_strides = brc_output_attr->strides;
  auto brc_attr_axis = brc_output_attr->axis;
  GE_ASSERT_TRUE(pre_brc_repeats.size() == pre_brc_strides.size());
  GE_ASSERT_TRUE(brc_repeats.size() == brc_attr_axis.size());
  GE_ASSERT_TRUE(brc_strides.size() == brc_attr_axis.size());
  GE_ASSERT_TRUE(pre_brc_repeats.size() == pre_brc_axis.size());
  const auto all_repeats_one =
      std::all_of(brc_repeats.begin(), brc_repeats.end(), [](const Expression &repeat) { return IsEqOne(repeat); });
  if (all_repeats_one) {
    brc_axis = -1;
    GELOGI("No-op broadcast node %s(%s) with all-one output repeats, treat as no broadcast axis.",
           brc_node->GetName().c_str(), brc_node->GetType().c_str());
    return SUCCESS;
  }
  for (size_t index = 0U; index < brc_attr_axis.size(); index++) {
    if (IsEqOne(brc_repeats[index])) {
      continue;
    }

    const auto pre_axis_iter = std::find(pre_brc_axis.begin(), pre_brc_axis.end(), brc_attr_axis[index]);
    if (pre_axis_iter == pre_brc_axis.end()) {
      // A missing input axis is a scalar/implicit broadcast dimension.
      brc_axis = brc_attr_axis[index];
      return SUCCESS;
    }

    const size_t pre_index = static_cast<size_t>(std::distance(pre_brc_axis.begin(), pre_axis_iter));
    if (IsEqOne(pre_brc_repeats[pre_index]) && IsEqZero(pre_brc_strides[pre_index])) {
      brc_axis = brc_attr_axis[index];
      return SUCCESS;
    }
  }
  GELOGW("Cannot infer broadcast axis: broadcast[%s], source[%s].", brc_node->GetName().c_str(),
         pre_brc_node->GetName().c_str());
  return FAILED;
}

Status GetBrcAxises(const std::vector<NodePtr> &brc_nodes, std::vector<int64_t> &brc_axis_idx) {
  for (const auto &brc_node : brc_nodes) {
    int64_t brc_axis = -1;
    if (GetBrcAxisFromNode(brc_node, brc_axis) != SUCCESS) {
      GELOGI("GetBrcAxisFromNode failed for node %s(%s), skipping.", brc_node->GetName().c_str(),
             brc_node->GetType().c_str());
      continue;
    }
    if (brc_axis >= 0) {
      brc_axis_idx.push_back(brc_axis);
    }
  }
  return SUCCESS;
}

// 收集广播节点的全部扩维轴：输出 repeat!=1 且（生产者缺失该轴，或生产者该轴
// repeat=1/stride=0）。单轴广播结果与 GetBrcAxisFromNode 一致；单节点多轴扩维
// （真实前端形态，如退化轴 Arange [1,1,N] 一步扩到 [B,M,N]）为首轴语义的超集。
// 同链判定与删除判定必须使用全集，否则首轴相同而轴集不同的分支会被误判等价，
// 删除对方广播后丢失非公共轴扩维语义。
Status GetAllBrcAxisesFromNode(const NodePtr &brc_node, std::set<int64_t> &brc_axises) {
  brc_axises.clear();
  NodePtr pre_brc_node;
  GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(brc_node, pre_brc_node, 0));
  AscTensorAttr *pre_brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_brc_node, pre_brc_output_attr));
  AscTensorAttr *brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(brc_node, brc_output_attr));

  // 视图三元组尺寸必须自洽（生产者与广播之间轴数允许不同——前缀扩维形态）；
  // 不自洽的视图属于非法图结构，fail-closed 返回失败，避免越界读取。
  const bool pre_view_valid = pre_brc_output_attr->axis.size() == pre_brc_output_attr->repeats.size() &&
                              pre_brc_output_attr->axis.size() == pre_brc_output_attr->strides.size();
  const bool brc_view_valid = brc_output_attr->axis.size() == brc_output_attr->repeats.size() &&
                              brc_output_attr->axis.size() == brc_output_attr->strides.size();
  if (!pre_view_valid || !brc_view_valid) {
    GELOGW("Cannot infer broadcast axes from inconsistent views: broadcast[%s], source[%s].",
           brc_node->GetName().c_str(), pre_brc_node->GetName().c_str());
    return FAILED;
  }

  for (size_t index = 0U; index < brc_output_attr->axis.size(); index++) {
    if (IsEqOne(brc_output_attr->repeats[index])) {
      continue;
    }
    const auto pre_axis_iter =
        std::find(pre_brc_output_attr->axis.begin(), pre_brc_output_attr->axis.end(), brc_output_attr->axis[index]);
    if (pre_axis_iter == pre_brc_output_attr->axis.end()) {
      brc_axises.insert(brc_output_attr->axis[index]);
      continue;
    }
    const size_t pre_index = static_cast<size_t>(std::distance(pre_brc_output_attr->axis.begin(), pre_axis_iter));
    if (IsEqOne(pre_brc_output_attr->repeats[pre_index]) && IsEqZero(pre_brc_output_attr->strides[pre_index])) {
      brc_axises.insert(brc_output_attr->axis[index]);
    }
  }
  return SUCCESS;
}

Status GetBrcAxisesIndex(std::vector<size_t> &brc_axis_idx, const std::vector<Expression> &pre_brc_repeats,
                         const std::vector<Expression> &pre_brc_strides,
                         const std::vector<Expression> &last_brc_repeats) {
  GE_ASSERT_TRUE(pre_brc_repeats.size() == pre_brc_strides.size());
  GE_ASSERT_TRUE(pre_brc_repeats.size() == last_brc_repeats.size());
  for (size_t index = 0U; index < pre_brc_repeats.size(); index++) {
    if (IsEqOne(pre_brc_repeats[index]) && IsEqZero(pre_brc_strides[index])) {
      if (IsEqOne(last_brc_repeats[index])) {
        continue;
      }
      brc_axis_idx.push_back(index);
    }
  }
  return SUCCESS;
}

// 同轴兄弟链判定。除逐节点全扩维轴集比较外，支持可选公共轴门控 allowed_axises：
// 兄弟链整链删除只有在"链尾按公共轴重建广播"能完全补偿被删扩维时才安全。原始整链后移
// （ProcessOriginalBackwardLogic）把含全部轴的整条链搬到计算链尾，任何同轴兄弟都可整链
// 删除（不传门控）；partial backward（JudgePartBackward）每轮只迁移公共轴，兄弟链中不属于
// 公共轴的扩维无人恢复，会留下消费者两输入视图不一致——此时传入本轮公共轴集，要求兄弟链
// 每个节点的全部扩维轴都落在其中，否则不穿透该消费者、退回公共轴 partial backward。
bool IsSameBrcNodes(const std::vector<NodePtr> &brc_nodes1, const std::vector<NodePtr> &brc_nodes2,
                    const std::set<int64_t> *allowed_axises = nullptr) {
  if (brc_nodes1.size() != brc_nodes2.size()) {
    return false;
  }
  // 逐节点比较全扩维轴集：首轴相同而轴集不同（如 {s0,s1} vs {s0,s2}）的分支
  // 不是同链，删重复广播+整链后移会丢失非公共轴扩维语义，必须改走公共轴
  // partial backward（否则改写后 Add 输入/输出视图不一致）。
  for (size_t index = 0U; index < brc_nodes1.size(); index++) {
    std::set<int64_t> brc_axis_set1;
    std::set<int64_t> brc_axis_set2;
    if (GetAllBrcAxisesFromNode(brc_nodes1[index], brc_axis_set1) != SUCCESS ||
        GetAllBrcAxisesFromNode(brc_nodes2[index], brc_axis_set2) != SUCCESS) {
      return false;
    }
    if (brc_axis_set1 != brc_axis_set2) {
      return false;
    }
    if (allowed_axises != nullptr) {
      for (const auto axis : brc_axis_set1) {
        if (allowed_axises->count(axis) == 0U) {
          return false;
        }
      }
    }
  }
  return true;
}

Status RemoveAndRelinkNodeEdge(af::InDataAnchorPtr &brc_in_anchor, af::OutDataAnchorPtr &brc_out_anchor) {
  GE_ASSERT_NOTNULL(brc_in_anchor);
  GE_ASSERT_NOTNULL(brc_out_anchor);
  GE_ASSERT_TRUE(!brc_out_anchor->GetPeerInDataAnchors().empty());
  auto before_brc_out_anchor = brc_in_anchor->GetPeerOutAnchor();
  auto after_brc_in_anchor = brc_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_brc_out_anchor);
  GE_ASSERT_NOTNULL(after_brc_in_anchor);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(before_brc_out_anchor, brc_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(brc_out_anchor, after_brc_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(before_brc_out_anchor, after_brc_in_anchor));
  return SUCCESS;
}

Status RemoveBroadcastOneByOne(std::vector<NodePtr> &brc_nodes, AscGraph &graph) {
  for (auto &node : brc_nodes) {
    auto brc_in_anchor = node->GetInDataAnchor(0);
    auto brc_out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(brc_in_anchor, brc_out_anchor));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), node));
    af::NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

Status RemoveBroadcasts(std::vector<NodePtr> &brc_nodes, AscGraph &graph) {
  auto brc_in_anchor = brc_nodes.front()->GetInDataAnchor(0);
  auto brc_out_anchor = brc_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_SUCCESS(RemoveAndRelinkNodeEdge(brc_in_anchor, brc_out_anchor));
  for (auto &node : brc_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), node));
    af::NodeUtils::UnlinkAll(*node);
  }
  return SUCCESS;
}

std::set<int64_t> FindSubSet(std::vector<int64_t> &brc_axis_idx1, std::vector<int64_t> &brc_axis_idx2) {
  std::set<int64_t> common_elements;
  if (brc_axis_idx1.empty() || brc_axis_idx2.empty()) {
    return common_elements;
  }
  std::sort(brc_axis_idx1.begin(), brc_axis_idx1.end(), std::greater<int64_t>());
  std::sort(brc_axis_idx2.begin(), brc_axis_idx2.end(), std::greater<int64_t>());
  auto it1 = brc_axis_idx1.begin();
  auto it2 = brc_axis_idx2.begin();
  while (it1 != brc_axis_idx1.end() && it2 != brc_axis_idx2.end()) {
    if (*it1 == *it2) {
      common_elements.insert(*it1);
      ++it1;
      ++it2;
    } else if (*it1 > *it2) {
      ++it1;
    } else {
      ++it2;
    }
  }
  return common_elements;
}

bool CollectSameBrcAxis(NodePtr &cur_node, NodePtr &next_node, std::vector<std::vector<NodePtr>> &brc_nodes_list,
                        std::set<int64_t> &common_axes, std::vector<NodePtr> &origin_brc_nodes,
                        std::vector<NodePtr> *retained_cast_nodes = nullptr) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(next_node, peer_out_nodes));
  for (const auto &node : peer_out_nodes) {
    if ((cur_node != nullptr) && (node == cur_node)) {
      continue;
    }
    NodePtr broadcast_node;
    NodePtr retained_cast_node;
    if (!GetBroadcastPeerThroughCast(node, broadcast_node, retained_cast_node)) {
      brc_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    if (retained_cast_nodes != nullptr && retained_cast_node != nullptr) {
      retained_cast_nodes->push_back(retained_cast_node);
    }
    if (origin_brc_nodes.empty()) {
      ReverseCollectBrcNodes(broadcast_node, origin_brc_nodes);
      std::reverse(origin_brc_nodes.begin(), origin_brc_nodes.end());
      brc_nodes_list.push_back(origin_brc_nodes);
      continue;
    }
    std::vector<NodePtr> temp_brc_nodes;
    ReverseCollectBrcNodes(broadcast_node, temp_brc_nodes);
    std::reverse(temp_brc_nodes.begin(), temp_brc_nodes.end());
    brc_nodes_list.push_back(temp_brc_nodes);

    std::vector<int64_t> brc_axis_idx1;
    std::vector<int64_t> brc_axis_idx2;
    if (common_axes.empty()) {
      GetBrcAxises(origin_brc_nodes, brc_axis_idx1);
    } else {
      brc_axis_idx1.assign(common_axes.begin(), common_axes.end());
    }
    GetBrcAxises(temp_brc_nodes, brc_axis_idx2);
    std::set<int64_t> temp_common_axes = FindSubSet(brc_axis_idx1, brc_axis_idx2);

    if (temp_common_axes.empty()) {
      brc_nodes_list.clear();
      common_axes.clear();
      return false;
    }
    common_axes = temp_common_axes;
    origin_brc_nodes = origin_brc_nodes.size() < temp_brc_nodes.size() ? origin_brc_nodes : temp_brc_nodes;
  }
  return true;
}

Status GetNodeScalarInputList(const af::AscNodePtr &asc_node, std::vector<bool> &is_scalar_list) {
  is_scalar_list.resize(asc_node->GetInDataNodesSize(), false);
  for (size_t i = 0UL; i < is_scalar_list.size(); ++i) {
    const std::vector<Expression> repeats = asc_node->inputs[i].attr.repeats;
    is_scalar_list[i] = ascgen_utils::IsScalarInput(repeats);
  }
  return SUCCESS;
}

Status ProcessOtherInputBranches(const NodePtr &next_comp_op, size_t current_idx, const std::vector<int64_t> &brc_axes,
                                 std::vector<bool> &is_scalar_list);

// 统计 node 的输入中由"非本事务"broadcast 支撑的路数：peer 是 Broadcast 时直接判定，peer 是 compute 时看其
// in-0 生产者（本链广播后移后将停靠在 peer 与 node 之间，属于本链而非外源）。excluded_brc_nodes 为本次
// 后移事务涉及的广播全集：链内广播（brc_nodes，提交期后移到链尾 compute 与 node 之间）与同轴兄弟广播
// （pending_remove_brcs，提交期统一删除，但探测期仍垫在链内多输入节点的输入上，会被 in-0 生产者检查看到）。
// 二者提交后都不会以"peer 背后的外源广播"形态存在，排除后语义变为"≥2 路外源广播才拒绝"，
// 避免本事务自己的广播反过来否决自己的后移。
bool HasMultipleInputsBroadcastBehind(const NodePtr &node, const std::vector<NodePtr> &excluded_brc_nodes) {
  if (node->GetAllInDataAnchorsSize() <= 1U) {
    return false;
  }
  std::vector<NodePtr> peer_out_nodes;
  if (GetPeerOutNodes(node, peer_out_nodes) != SUCCESS) {
    return false;
  }
  int64_t broadcast_input_count = 0;
  for (const auto &peer : peer_out_nodes) {
    if (peer == nullptr) {
      continue;
    }
    bool has_broadcast = false;
    if (peer->GetType() == kBroadcastType) {
      has_broadcast = std::find(excluded_brc_nodes.begin(), excluded_brc_nodes.end(), peer) == excluded_brc_nodes.end();
    } else {
      NodePtr pre_node;
      if (GetPeerOutNodeSafe(peer, pre_node, 0) == SUCCESS && pre_node != nullptr &&
          pre_node->GetType() == kBroadcastType) {
        // peer 背后的广播属于本事务（链内广播将移到 peer 与 node 之间、待删兄弟提交期删除）时不算"别人"。
        has_broadcast =
            std::find(excluded_brc_nodes.begin(), excluded_brc_nodes.end(), pre_node) == excluded_brc_nodes.end();
      }
    }
    if (has_broadcast) {
      broadcast_input_count++;
    }
  }
  return broadcast_input_count >= kMinBroadcastInputCount;
}

bool CheckBackwardCommon(const NodePtr &next_node) {
  if (!HasNoControlEdges(next_node)) {
    return false;
  }
  if (next_node->GetType() == kStoreType) {
    return false;
  }
  // IndirectLoad 的输出轴属于独立的物理视图，Broadcast 后移不能跨过该边界。
  // 直接喂给 IndirectLoad 的 compute 同样不能后移，否则 broadcast 会紧贴 IL。
  if (IsIndirectLoadBoundary(next_node)) {
    return false;
  }
  if (ScheduleUtils::IsRemovePad(next_node)) {
    return false;
  }
  if (IsNextViewOp(next_node)) {
    return false;
  }
  af::DataType output_dtype;
  if (IsDtypeNotSupportOp(next_node, output_dtype)) {
    GELOGI("Node %s(%s) cannot backward with dtype(%s)", next_node->GetName().c_str(), next_node->GetType().c_str(),
           af::TypeUtils::DataTypeToSerialString(output_dtype).c_str());
    return false;
  }
  return true;
}

bool CanBackwardSimplified(const NodePtr &next_node) {
  if (!CheckBackwardCommon(next_node)) {
    return false;
  }
  if (next_node->GetAllOutDataAnchorsSize() > 1U) {
    return false;
  }
  if (!IsSingleInNode(next_node)) {
    return false;
  }
  return true;
}

bool IsScalarInput(const NodePtr &input_node) {
  NodePtr temp_node = input_node;
  while (temp_node != nullptr) {
    if (temp_node->GetType() == kScalarType) {
      return true;
    }
    if (temp_node->GetType() != kBroadcastType) {
      break;
    }
    NodePtr pre_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(temp_node, pre_node, 0));
    temp_node = pre_node;
  }
  return false;
}

Status CheckNodeSupportsScalarInput(const NodePtr &compute_node, int32_t input_idx,
                                    const std::vector<int64_t> &brc_axes, bool &is_support) {
  is_support = false;
  GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<AscNode>(compute_node));
  const auto &asc_node = std::dynamic_pointer_cast<AscNode>(compute_node);

  std::vector<bool> is_scalar_list;
  GE_ASSERT_SUCCESS(GetNodeScalarInputList(asc_node, is_scalar_list));

  if (input_idx >= 0 && static_cast<size_t>(input_idx) < is_scalar_list.size()) {
    is_scalar_list[input_idx] = true;
  }

  GE_ASSERT_SUCCESS(ProcessOtherInputBranches(compute_node, input_idx, brc_axes, is_scalar_list));

  is_support = ascgen_utils::IsNodeSupportsScalarInput(asc_node, is_scalar_list);
  if (!is_support) {
    GELOGD("Compute node %s does not support scalar input, is_scalar_list: %s", compute_node->GetName().c_str(),
           VectorToStr(is_scalar_list).c_str());
  }
  return SUCCESS;
}

bool CheckScalarInputSupport(const NodePtr &next_node, const std::vector<NodePtr> &brc_nodes) {
  GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<AscNode>(next_node));

  auto in_data_anchor_size = next_node->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < in_data_anchor_size; ++i) {
    NodePtr input_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(next_node, input_node, i));

    if (IsScalarInput(input_node)) {
      std::vector<int64_t> brc_axes;
      if (!brc_nodes.empty()) {
        GE_ASSERT_SUCCESS(GetBrcAxises(brc_nodes, brc_axes));
      }
      bool is_support = false;
      GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(next_node, i, brc_axes, is_support));
      if (!is_support) {
        return false;
      }
    }
  }
  return true;
}

bool IsMulInputsCanBackward(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &brc_nodes,
                            std::set<NodePtr> &mul_input_nodes, std::vector<std::vector<NodePtr>> &pending_remove_brcs,
                            const std::set<int64_t> *sibling_axis_gate) {
  auto in_data_anchor_size = next_node->GetAllInDataAnchorsSize();
  if (in_data_anchor_size == 1U) {
    return false;
  }

  // 多输入节点的同轴兄弟 broadcast 删除登记与部分后移注册都发生在本函数内；若该节点下游到达
  // IndirectLoad，任何改写都会破坏索引链视图，必须在此整体拒绝。
  if (ChainReachesIndirectLoad(next_node)) {
    GELOGD("Skip multi-input backward at node[%s]: downstream reaches IndirectLoad.", next_node->GetName().c_str());
    return false;
  }

  // MI-MO：链式收集依赖唯一后继，无法穿过该节点（不登记同轴兄弟删除、图保持原状）；
  // 同链与不同链统一按公共轴探测，分支存在公共轴时登记 partial backward，由
  // JudgePartBackward 将公共轴 Brc 挨个后移到每路输出分支首节点之前。
  if (!IsSingleOutNode(next_node)) {
    std::vector<std::vector<NodePtr>> brc_nodes_list;
    std::set<int64_t> common_axes;
    std::vector<NodePtr> origin_brc_nodes;
    origin_brc_nodes.assign(brc_nodes.begin(), brc_nodes.end());
    brc_nodes_list.push_back(origin_brc_nodes);
    if (CollectSameBrcAxis(cur_node, next_node, brc_nodes_list, common_axes, origin_brc_nodes)) {
      mul_input_nodes.insert(next_node);
    }
    return false;
  }

  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(GetPeerOutNodes(next_node, peer_out_nodes));
  std::vector<std::vector<NodePtr>> remove_brc_nodes_list;
  for (const auto &node : peer_out_nodes) {
    if (node == cur_node) {
      continue;
    }
    NodePtr broadcast_node;
    NodePtr retained_cast_node;
    if (!GetBroadcastPeerThroughCast(node, broadcast_node, retained_cast_node)) {
      return false;
    }

    std::vector<NodePtr> temp_brc_nodes;
    GE_ASSERT_SUCCESS(ReverseCollectBrcNodes(broadcast_node, temp_brc_nodes));
    std::reverse(temp_brc_nodes.begin(), temp_brc_nodes.end());
    if (retained_cast_node != nullptr && temp_brc_nodes.size() <= 1U) {
      return false;
    }
    if (!IsSameBrcNodes(brc_nodes, temp_brc_nodes, sibling_axis_gate)) {
      std::vector<std::vector<NodePtr>> brc_nodes_list;
      std::set<int64_t> common_axes;
      std::vector<NodePtr> origin_brc_nodes;
      origin_brc_nodes.assign(brc_nodes.begin(), brc_nodes.end());
      brc_nodes_list.push_back(origin_brc_nodes);
      if (CollectSameBrcAxis(cur_node, next_node, brc_nodes_list, common_axes, origin_brc_nodes)) {
        mul_input_nodes.insert(next_node);
      }
      return false;
    }
    // Arange 生产者链的同轴重复广播删除与整链后移未适配（退化轴视图代数缺失），
    // fail-closed：不删重复广播、链到此为止，图保持原状，交由 partial backward
    // 的组级守卫决定是否按公共轴后移。
    NodePtr chain_producer;
    if (!brc_nodes.empty() && GetPeerOutNodeSafe(brc_nodes.front(), chain_producer, 0) == SUCCESS &&
        chain_producer != nullptr && chain_producer->GetType() == Arange::Type) {
      GELOGI("Skip multi-input backward at node[%s]: branch producer[%s] is Arange.", next_node->GetName().c_str(),
             chain_producer->GetName().c_str());
      return false;
    }
    remove_brc_nodes_list.push_back(temp_brc_nodes);
  }

  if (!CheckScalarInputSupport(next_node, brc_nodes)) {
    return false;
  }

  for (std::vector<NodePtr> &remove_nodes : remove_brc_nodes_list) {
    // 探测期只登记、提交期统一删除：链收集后的守卫（外源广播/IndirectLoad 下游/视图轴数）拒绝提交时
    // 图保持原状，避免"兄弟已删、链未后移"的半改写状态。
    pending_remove_brcs.push_back(std::move(remove_nodes));
  }
  return true;
}

bool CanBackward(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &brc_nodes,
                 std::set<NodePtr> &mul_input_nodes, std::vector<std::vector<NodePtr>> &pending_remove_brcs,
                 const std::set<int64_t> *sibling_axis_gate) {
  if (!CheckBackwardCommon(next_node)) {
    return false;
  }
  // 单输入多输出仍受限：链式收集依赖唯一后继，且单输入无多分支公共轴可求交；
  // 多输入多输出不再一刀切拒绝，交由 IsMulInputsCanBackward 按公共轴登记 partial backward。
  if (!IsSingleOutNode(next_node) && IsSingleInNode(next_node)) {
    return false;
  }
  if (!IsSingleInNode(next_node) && !IsMulInputsCanBackward(cur_node, next_node, brc_nodes, mul_input_nodes,
                                                            pending_remove_brcs, sibling_axis_gate)) {
    return false;
  }
  return true;
}

Status CollectBrcNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes) {
  if (cur_node->GetType() == kBroadcastType && !HasNoControlEdges(cur_node)) {
    return SUCCESS;
  }
  if (IsSingleInAndOutNode(cur_node) && (cur_node->GetType() == kBroadcastType)) {
    nodes.push_back(cur_node);
  }
  while (IsSingleInAndOutNode(next_node) && (next_node->GetType() == kBroadcastType)) {
    if (!HasNoControlEdges(next_node)) {
      return SUCCESS;
    }
    nodes.push_back(next_node);
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

Status CollectCmpNodes(NodePtr &cur_node, NodePtr &next_node, std::vector<NodePtr> &nodes,
                       std::vector<NodePtr> &brc_nodes, std::set<NodePtr> &mul_input_nodes,
                       std::vector<std::vector<NodePtr>> &pending_remove_brcs,
                       const std::set<int64_t> *sibling_axis_gate) {
  while (CanBackward(cur_node, next_node, brc_nodes, mul_input_nodes, pending_remove_brcs, sibling_axis_gate)) {
    nodes.push_back(next_node);
    cur_node = next_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
  }
  return SUCCESS;
}

Status ReorderBroadcasts(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &brc_nodes) {
  auto brc_in_anchor = brc_nodes.front()->GetInDataAnchor(0);
  auto brc_out_anchor = brc_nodes.back()->GetOutDataAnchor(0);
  auto comp_out_anchor = compute_nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(brc_in_anchor);
  GE_ASSERT_NOTNULL(brc_out_anchor);
  GE_ASSERT_NOTNULL(comp_out_anchor);
  GE_ASSERT_TRUE(!comp_out_anchor->GetPeerInDataAnchors().empty());
  GE_ASSERT_TRUE(!brc_out_anchor->GetPeerInDataAnchors().empty());
  auto before_brc_out_anchor = brc_in_anchor->GetPeerOutAnchor();
  auto after_comp_in_anchor = comp_out_anchor->GetPeerInDataAnchors().at(0);
  auto comp_in_anchor = brc_out_anchor->GetPeerInDataAnchors().at(0);
  GE_ASSERT_NOTNULL(before_brc_out_anchor);
  GE_ASSERT_NOTNULL(after_comp_in_anchor);
  GE_ASSERT_NOTNULL(comp_in_anchor);

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(before_brc_out_anchor, brc_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(brc_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(comp_out_anchor, after_comp_in_anchor));

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(before_brc_out_anchor, comp_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(comp_out_anchor, brc_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(brc_out_anchor, after_comp_in_anchor));
  return SUCCESS;
}

Status UpdateComputeNodesAscTensorAttr(std::vector<NodePtr> &brc_nodes, std::vector<NodePtr> &compute_nodes,
                                       const NodePtr &pre_brc_node) {
  AscTensorAttr *pre_brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_brc_node, pre_brc_output_attr));

  AscTensorAttr *last_brc_output_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(brc_nodes.back(), last_brc_output_attr));

  std::vector<size_t> brc_axis_idx;
  auto pre_brc_axis = pre_brc_output_attr->axis;
  auto pre_brc_repeats = pre_brc_output_attr->repeats;
  auto pre_brc_strides = pre_brc_output_attr->strides;
  auto last_brc_repeats = last_brc_output_attr->repeats;
  GE_ASSERT_SUCCESS(GetBrcAxisesIndex(brc_axis_idx, pre_brc_repeats, pre_brc_strides, last_brc_repeats));
  GELOGD("Broadcast chain contains %zu broadcast axes.", brc_axis_idx.size());
  for (const auto &compute_node : compute_nodes) {
    AscTensorAttr *compute_output_attr = nullptr;
    GE_ASSERT_SUCCESS(GetOutputTensorAttr(compute_node, compute_output_attr));
    compute_output_attr->axis = last_brc_output_attr->axis;
    compute_output_attr->repeats = pre_brc_repeats;
    compute_output_attr->strides = pre_brc_strides;
    GE_ASSERT_TRUE(compute_output_attr->strides.size() > 0U);
    auto it = std::find(brc_axis_idx.begin(), brc_axis_idx.end(), pre_brc_strides.size() - 1U);
    if (it != brc_axis_idx.end()) {
      compute_output_attr->strides[compute_output_attr->strides.size() - 1U] = af::sym::kSymbolZero;
    }
    if (pre_brc_node->GetType() != kScalarType) {
      GE_ASSERT_SUCCESS(
          ScheduleUtils::RecalculateStridesFromRepeats(compute_output_attr->repeats, compute_output_attr->strides));
    }
  }
  return SUCCESS;
}

Status UpdateBroadcastNodesDataType(std::vector<NodePtr> &brc_nodes, const NodePtr &last_comp_node) {
  const auto last_comp_opdesc = last_comp_node->GetOpDesc();
  GE_ASSERT_NOTNULL(last_comp_opdesc);
  const auto comp_output_tensor_desc = last_comp_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(comp_output_tensor_desc);
  auto last_comp_dtype = comp_output_tensor_desc->GetDataType();
  for (const auto &brc_node : brc_nodes) {
    const auto brc_opdesc = brc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(brc_opdesc);
    const auto brc_output_tensor_desc = brc_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(brc_output_tensor_desc);
    brc_output_tensor_desc->SetDataType(last_comp_dtype);
  }
  return SUCCESS;
}

Status BroadcastBackwardReally(std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &brc_nodes,
                               const NodePtr &pre_brc_node) {
  GE_ASSERT_SUCCESS(ReorderBroadcasts(compute_nodes, brc_nodes));
  GE_ASSERT_SUCCESS(UpdateComputeNodesAscTensorAttr(brc_nodes, compute_nodes, pre_brc_node));
  GE_ASSERT_SUCCESS(UpdateBroadcastNodesDataType(brc_nodes, compute_nodes.back()));
  return SUCCESS;
}

Status CollectBranchBroadcastNodes(const NodePtr &input_node, std::vector<NodePtr> &branch_brc_nodes) {
  NodePtr temp_node = input_node;
  while (temp_node->GetType() == kBroadcastType && IsSingleInAndOutNode(temp_node)) {
    branch_brc_nodes.push_back(temp_node);
    NodePtr next_temp_node;
    if (GetPeerOutNodeSafe(temp_node, next_temp_node, 0) != SUCCESS) {
      break;
    }
    temp_node = next_temp_node;
  }
  return SUCCESS;
}

bool HasCommonBroadcastAxis(const std::vector<int64_t> &axes1, const std::vector<int64_t> &axes2) {
  for (const auto &axis : axes1) {
    if (std::find(axes2.begin(), axes2.end(), axis) != axes2.end()) {
      return true;
    }
  }
  return false;
}

NodePtr GetPreBroadcastNode(const NodePtr &branch_brc_node) {
  auto brc_in_anchor = branch_brc_node->GetInDataAnchor(0);
  if (brc_in_anchor == nullptr) {
    return nullptr;
  }
  auto peer_out_anchor = brc_in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return nullptr;
  }
  return ToAscNode(peer_out_anchor->GetOwnerNode());
}

Status ProcessSingleInputBranch(const NodePtr &input_node, const std::vector<int64_t> &brc_axes, bool &is_scalar) {
  std::vector<NodePtr> branch_brc_nodes;
  GE_ASSERT_SUCCESS(CollectBranchBroadcastNodes(input_node, branch_brc_nodes));

  if (!branch_brc_nodes.empty()) {
    std::vector<int64_t> branch_brc_axes;
    GE_ASSERT_SUCCESS(GetBrcAxises(branch_brc_nodes, branch_brc_axes));
    if (HasCommonBroadcastAxis(brc_axes, branch_brc_axes)) {
      NodePtr pre_brc_node = GetPreBroadcastNode(branch_brc_nodes.front());
      if (pre_brc_node != nullptr) {
        AscTensorAttr *pre_brc_attr = nullptr;
        if (GetOutputTensorAttr(pre_brc_node, pre_brc_attr) == SUCCESS) {
          const std::vector<Expression> repeats = pre_brc_attr->repeats;
          is_scalar = ascgen_utils::IsScalarInput(repeats);
        }
      }
    }
  }
  return SUCCESS;
}

Status ProcessOtherInputBranches(const NodePtr &next_comp_op, size_t current_idx, const std::vector<int64_t> &brc_axes,
                                 std::vector<bool> &is_scalar_list) {
  for (size_t i = 0; i < is_scalar_list.size(); ++i) {
    if (i == current_idx) {
      continue;
    }
    NodePtr input_node;
    if (GetPeerOutNodeSafe(next_comp_op, input_node, static_cast<int32_t>(i)) != SUCCESS) {
      continue;
    }
    bool scalar_flag = is_scalar_list[i];
    GE_ASSERT_SUCCESS(ProcessSingleInputBranch(input_node, brc_axes, scalar_flag));
    is_scalar_list[i] = scalar_flag;
  }
  return SUCCESS;
}

Status JudgeNextCompOpSupportsScalarInput(const NodePtr &node, bool &is_next_support_scalar) {
  is_next_support_scalar = false;

  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() == 1U);
  auto out_anchor = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);

  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  GE_ASSERT_TRUE(!peer_in_anchors.empty());

  bool all_branches_support_scalar = true;
  for (const auto &peer_in_anchor : peer_in_anchors) {
    NodePtr branch_start_node = ToAscNode(peer_in_anchor->GetOwnerNode());

    std::vector<NodePtr> brc_nodes;
    NodePtr temp_cur_node = node;
    NodePtr temp_next_node = branch_start_node;
    GE_ASSERT_SUCCESS(CollectBrcNodes(temp_cur_node, temp_next_node, brc_nodes));

    NodePtr compute_node = temp_next_node;
    if (compute_node == nullptr || brc_nodes.empty()) {
      all_branches_support_scalar = false;
      break;
    }

    bool is_support = false;
    std::vector<int64_t> brc_axes;
    GE_ASSERT_SUCCESS(GetBrcAxises(brc_nodes, brc_axes));
    GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(compute_node, peer_in_anchor->GetIdx(), brc_axes, is_support));
    if (!is_support) {
      all_branches_support_scalar = false;
      break;
    }
  }

  is_next_support_scalar = all_branches_support_scalar;
  return SUCCESS;
}

bool ContainsBroadcastNode(const NodePtr &node) {
  // Graph input nodes (for example Data) legitimately have no predecessor.
  // Check the anchor before calling GetPeerOutNodeSafe(), whose assertion is
  // reserved for paths that require a connected input.
  if (node == nullptr || node->GetAllInDataAnchorsSize() == 0U) {
    return false;
  }
  auto in_anchor = node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    return false;
  }

  NodePtr pre_node;
  if (GetPeerOutNodeSafe(node, pre_node, 0) != SUCCESS || pre_node == nullptr) {
    return false;
  }
  return pre_node->GetType() == kBroadcastType;
}

Status CollectCandidateMultiRefNodes(const AscGraph &graph, std::vector<NodePtr> &candidate_nodes) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetAllOutDataAnchorsSize() != 1U) {
      continue;
    }
    auto out_anchor = node->GetOutDataAnchor(0);
    if (out_anchor == nullptr || out_anchor->GetPeerInDataAnchors().size() <= 1) {
      continue;
    }
    if (node->GetType() == kBroadcastType || ContainsBroadcastNode(node)) {
      candidate_nodes.push_back(node);
    }
  }
  return SUCCESS;
}

Status ExtractBroadcastChainFromNode(const NodePtr &node, std::vector<NodePtr> &brc_nodes) {
  NodePtr cur_node = node;

  if (cur_node != nullptr && cur_node->GetType() != kBroadcastType) {
    NodePtr pre_node;
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, pre_node, 0));
    cur_node = pre_node;
  }

  while (cur_node != nullptr && cur_node->GetType() == kBroadcastType) {
    brc_nodes.push_back(cur_node);
    NodePtr pre_node;
    if (GetPeerOutNodeSafe(cur_node, pre_node, 0) != SUCCESS) {
      break;
    }
    cur_node = pre_node;
  }

  std::reverse(brc_nodes.begin(), brc_nodes.end());
  return SUCCESS;
}

Status TraceBranchToMergeNode(const NodePtr &start_node, NodePtr &merge_node, std::vector<NodePtr> &branch_nodes) {
  NodePtr cur_node = start_node;
  std::unordered_set<NodePtr> visited_nodes;

  while (cur_node != nullptr) {
    GE_ASSERT_TRUE(visited_nodes.count(cur_node) == 0, "Found cycle dependency in TraceBranchToMergeNode, node: %s",
                   cur_node->GetName().c_str());
    visited_nodes.insert(cur_node);

    if (!IsSingleInNode(cur_node)) {
      merge_node = cur_node;
      return SUCCESS;
    }

    if (cur_node->GetType() == kStoreType) {
      merge_node = cur_node;
      return SUCCESS;
    }

    if (!IsSingleOutNode(cur_node)) {
      return FAILED;
    }

    branch_nodes.push_back(cur_node);

    NodePtr next_node;
    if (GetSingleNextNode(cur_node, next_node) != SUCCESS) {
      return FAILED;
    }
    cur_node = next_node;
  }
  return FAILED;
}

bool CheckBranchNodesSupportBackward(const std::vector<NodePtr> &branch_nodes) {
  for (const auto &next_node : branch_nodes) {
    if (!CanBackwardSimplified(next_node)) {
      return false;
    }
  }
  return true;
}

bool CheckAllBranchesCanBackward(const NodePtr &multi_ref_node, NodePtr &merge_node,
                                 std::vector<std::vector<NodePtr>> &all_branch_nodes) {
  auto out_anchor = multi_ref_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();

  GE_ASSERT_TRUE(peer_in_anchors.size() > 1U);
  NodePtr first_merge_node = nullptr;

  for (const auto &in_anchor : peer_in_anchors) {
    NodePtr branch_start_node = ToAscNode(in_anchor->GetOwnerNode());
    NodePtr current_merge_node = nullptr;
    std::vector<NodePtr> branch_nodes;

    if (TraceBranchToMergeNode(branch_start_node, current_merge_node, branch_nodes) != SUCCESS) {
      return false;
    }

    if (first_merge_node == nullptr) {
      first_merge_node = current_merge_node;
    } else if (first_merge_node != current_merge_node) {
      return false;
    }
    all_branch_nodes.push_back(branch_nodes);
  }

  merge_node = first_merge_node;
  if (merge_node != nullptr && merge_node->GetInDataNodesSize() > all_branch_nodes.size()) {
    return false;
  }
  return true;
}

bool CheckAllBranchesSupportBackward(const std::vector<std::vector<NodePtr>> &all_branch_nodes,
                                     const NodePtr &candidate_node) {
  if (!HasNoControlEdges(candidate_node)) {
    return false;
  }
  if (candidate_node->GetType() != kBroadcastType) {
    if (!CanBackwardSimplified(candidate_node)) {
      return false;
    }
  }
  for (const auto &branch_nodes : all_branch_nodes) {
    if (!CheckBranchNodesSupportBackward(branch_nodes)) {
      return false;
    }
  }
  return true;
}

Status DisconnectBranchesFromBroadcast(const NodePtr &last_brc_node,
                                       std::vector<af::OutDataAnchorPtr> &branch_out_anchors,
                                       std::vector<af::InDataAnchorPtr> &branch_in_anchors) {
  auto brc_out_anchor = last_brc_node->GetOutDataAnchor(0);
  auto peer_in_anchors = brc_out_anchor->GetPeerInDataAnchors();

  for (const auto &in_anchor : peer_in_anchors) {
    auto branch_node = in_anchor->GetOwnerNode();
    auto branch_in_anchor = branch_node->GetInDataAnchor(in_anchor->GetIdx());
    auto branch_out_anchor = branch_in_anchor->GetPeerOutAnchor();

    branch_out_anchors.push_back(branch_out_anchor);
    branch_in_anchors.push_back(branch_in_anchor);

    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(branch_out_anchor, branch_in_anchor));
  }
  return SUCCESS;
}

Status MoveBroadcastAfterMerge(const NodePtr &merge_node, const NodePtr &first_brc_node, const NodePtr &last_brc_node) {
  auto merge_out_anchor = merge_node->GetOutDataAnchor(0);
  if (merge_out_anchor == nullptr || merge_out_anchor->GetPeerInDataAnchors().empty()) {
    return FAILED;
  }
  auto peer_in_anchors = merge_out_anchor->GetPeerInDataAnchors();
  std::vector<af::InDataAnchorPtr> merge_next_in_anchors(peer_in_anchors.begin(), peer_in_anchors.end());
  for (const auto &in_anchor : merge_next_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(merge_out_anchor, in_anchor));
  }

  auto brc_in_anchor = first_brc_node->GetInDataAnchor(0);
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(merge_out_anchor, brc_in_anchor));

  auto brc_out_anchor = last_brc_node->GetOutDataAnchor(0);
  for (const auto &in_anchor : merge_next_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(brc_out_anchor, in_anchor));
  }
  return SUCCESS;
}

Status BackwardMultiRefBroadcast(const NodePtr &candidate_node, const NodePtr &merge_node,
                                 const std::vector<std::vector<NodePtr>> &all_branch_nodes,
                                 std::vector<NodePtr> &brc_nodes, [[maybe_unused]] AscGraph &graph) {
  auto brc_in_anchor = brc_nodes.front()->GetInDataAnchor(0);
  auto pre_brc_out_anchor = brc_in_anchor->GetPeerOutAnchor();
  NodePtr pre_brc_node = ToAscNode(pre_brc_out_anchor->GetOwnerNode());
  bool is_pre_scalar = IsScalarLikeProducer(pre_brc_node);

  // Arange 生产者的多消费后移沿用整链视图推导（UpdateComputeNodesAscTensorAttr），
  // 退化轴视图代数未适配，与单链路径同等 fail-closed。
  if (pre_brc_node->GetType() == Arange::Type) {
    GELOGI("Skip multi-ref broadcast backward at merge[%s]: branch producer[%s] is Arange.",
           merge_node->GetName().c_str(), pre_brc_node->GetName().c_str());
    return FAILED;
  }

  if (is_pre_scalar) {
    std::vector<int64_t> brc_axes;
    if (!brc_nodes.empty()) {
      GE_ASSERT_SUCCESS(GetBrcAxises(brc_nodes, brc_axes));
    }
    auto brc_out_anchor = brc_nodes.back()->GetOutDataAnchor(0);
    auto peer_in_anchors = brc_out_anchor->GetPeerInDataAnchors();
    for (const auto &in_anchor : peer_in_anchors) {
      NodePtr compute_node = ToAscNode(in_anchor->GetOwnerNode());
      int32_t input_idx = in_anchor->GetIdx();
      bool is_support = false;
      GE_ASSERT_SUCCESS(CheckNodeSupportsScalarInput(compute_node, input_idx, brc_axes, is_support));
      if (!is_support) {
        return FAILED;
      }
    }
  }

  std::vector<af::OutDataAnchorPtr> branch_out_anchors;
  std::vector<af::InDataAnchorPtr> branch_in_anchors;
  GE_ASSERT_SUCCESS(DisconnectBranchesFromBroadcast(brc_nodes.back(), branch_out_anchors, branch_in_anchors));

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(pre_brc_out_anchor, brc_in_anchor));
  GE_ASSERT_SUCCESS(MoveBroadcastAfterMerge(merge_node, brc_nodes.front(), brc_nodes.back()));

  for (size_t i = 0; i < branch_out_anchors.size(); ++i) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(pre_brc_out_anchor, branch_in_anchors[i]));
  }

  std::vector<NodePtr> compute_nodes;
  if (candidate_node->GetType() != kBroadcastType) {
    compute_nodes.push_back(candidate_node);
  }
  for (const auto &branch : all_branch_nodes) {
    compute_nodes.insert(compute_nodes.end(), branch.begin(), branch.end());
  }
  compute_nodes.push_back(merge_node);
  GE_ASSERT_SUCCESS(UpdateComputeNodesAscTensorAttr(brc_nodes, compute_nodes, pre_brc_node));

  if (!compute_nodes.empty()) {
    GE_ASSERT_SUCCESS(UpdateBroadcastNodesDataType(brc_nodes, compute_nodes.back()));
  }
  return SUCCESS;
}

Status ProcessMultiRefBroadcastBackward(AscGraph &graph, bool &is_changed) {
  std::vector<NodePtr> candidate_nodes;
  GE_ASSERT_SUCCESS(CollectCandidateMultiRefNodes(graph, candidate_nodes));

  for (const auto &candidate_node : candidate_nodes) {
    std::vector<NodePtr> brc_nodes;
    GE_ASSERT_SUCCESS(ExtractBroadcastChainFromNode(candidate_node, brc_nodes));

    if (brc_nodes.empty()) {
      continue;
    }

    if (std::any_of(brc_nodes.begin(), brc_nodes.end(), [](const NodePtr &node) { return !HasNoControlEdges(node); })) {
      continue;
    }

    NodePtr brc_node = brc_nodes.front();
    NodePtr merge_node = nullptr;
    std::vector<std::vector<NodePtr>> all_branch_nodes;

    if (!CheckAllBranchesCanBackward(candidate_node, merge_node, all_branch_nodes)) {
      continue;
    }
    if (!CheckAllBranchesSupportBackward(all_branch_nodes, candidate_node)) {
      continue;
    }
    // 守卫④：multi-ref 广播将整体后移到 merge 节点之后；若 merge 节点（含自身）或其下游存在 view 算子，
    // 跨 view 的统一 broadcast 展开会破坏其轴语义，必须 fail-closed 在真正改写（BackwardMultiRefBroadcast）
    // 之前整体拒绝。
    if (DownstreamViewOpBlocksBackward(merge_node)) {
      GELOGI("Skip multi-ref broadcast backward at node[%s]: downstream of merge[%s] has view op blocking backward.",
             candidate_node->GetName().c_str(), merge_node->GetName().c_str());
      continue;
    }
    GELOGI("Move shared broadcast from node[%s] to merge[%s], broadcasts=%zu, branches=%zu.",
           candidate_node->GetName().c_str(), merge_node->GetName().c_str(), brc_nodes.size(), all_branch_nodes.size());

    // 逐候选累积改写标记：不能覆盖赋值，否则前序候选已后移、末位候选被守卫拒绝时
    // 会丢失改写事实，外层迭代将提前收敛。
    if (BackwardMultiRefBroadcast(candidate_node, merge_node, all_branch_nodes, brc_nodes, graph) == SUCCESS) {
      is_changed = true;
    }
  }
  return SUCCESS;
}

Status CollectBackwardStartNodes(const AscGraph &graph, std::vector<NodePtr> &pre_brc_nodes) {
  for (const auto &node : graph.GetAllNodes()) {
    NodePtr cur_node = node;
    while ((cur_node->GetType() == kBroadcastType) && IsSingleOutNode(cur_node)) {
      GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(cur_node, cur_node, 0));
    }
    bool is_next_support_scalar = true;
    if (IsScalarLikeProducer(cur_node)) {
      GE_ASSERT_SUCCESS(JudgeNextCompOpSupportsScalarInput(cur_node, is_next_support_scalar));
    }

    if ((cur_node != node) && is_next_support_scalar) {
      pre_brc_nodes.push_back(cur_node);
    }
  }
  return SUCCESS;
}

Status CollectBackwardSatisfyStartNodes(const NodePtr &node, std::vector<NodePtr> &peer_in_nodes) {
  auto output_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t idx = 0U; idx < output_size; ++idx) {
    std::vector<NodePtr> temp_nodes;
    GE_ASSERT_SUCCESS(GetPeerInNodes(node, temp_nodes, static_cast<int32_t>(idx)));
    if (!temp_nodes.empty() && (temp_nodes.front()->GetType() == kBroadcastType)) {
      peer_in_nodes.insert(peer_in_nodes.end(), temp_nodes.begin(), temp_nodes.end());
    }
  }
  return SUCCESS;
}

Status RemoveBroadcasts(AscGraph &graph, std::vector<std::vector<NodePtr>> &brc_nodes_list,
                        const std::set<int64_t> &common_axises) {
  for (std::vector<NodePtr> &brc_nodes : brc_nodes_list) {
    std::vector<NodePtr> remove_nodes;
    for (auto it = brc_nodes.begin(); it != brc_nodes.end();) {
      // 仅当广播的全部扩维轴都属于公共轴时才可整节点删除；单节点多轴扩维
      // （[1,1,N]→[B,M,N]）含非公共轴时必须保留节点——其输出视图随后由
      // UpdateOutputTensor 塌缩公共轴，节点退化为仅扩非公共轴，非公共轴的
      // 扩维职责保留在计算之前（外积语义不变）。整节点删除会丢失非公共轴
      // 扩维（否则改写后视图不一致）。
      std::set<int64_t> expansion_axes;
      GE_ASSERT_SUCCESS(GetAllBrcAxisesFromNode(*it, expansion_axes));
      const bool all_axes_common = !expansion_axes.empty() && std::all_of(expansion_axes.begin(), expansion_axes.end(),
                                                                          [&common_axises](int64_t axis) {
                                                                            return common_axises.count(axis) != 0;
                                                                          });
      if (all_axes_common) {
        remove_nodes.push_back(*it);
        it = brc_nodes.erase(it);
      } else {
        ++it;
      }
    }
    GE_ASSERT_SUCCESS(RemoveBroadcastOneByOne(remove_nodes, graph));
  }
  return SUCCESS;
}

struct TensorInfo {
  std::vector<int64_t> axis;
  std::vector<Expression> repeats;
  std::vector<Expression> strides;
  af::DataType dtype;
  int64_t sched_axis;
  std::vector<int64_t> broadcast_info;
};

Status GetTensorInfoAtOutput(const NodePtr &node, uint32_t output_index, TensorInfo &tensor_info) {
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttrAt(node, attr, output_index));
  tensor_info.axis = attr->axis;
  tensor_info.repeats = attr->repeats;
  tensor_info.strides = attr->strides;
  tensor_info.dtype = attr->dtype;
  tensor_info.sched_axis = attr->axis.back();
  return SUCCESS;
}

Status GetTensorInfo(const NodePtr &node, TensorInfo &tensor_info) {
  return GetTensorInfoAtOutput(node, 0U, tensor_info);
}

Status UpdateBroadcastNodeAttrs(const NodePtr &b_node, const std::vector<int64_t> &axis,
                                const std::vector<Expression> &repeats, const std::vector<Expression> &strides,
                                int64_t broadcast_axis) {
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(b_node, attr));
  GE_ASSERT_TRUE(axis.size() == repeats.size(), "Broadcast axis/repeats size mismatch: %zu vs %zu", axis.size(),
                 repeats.size());
  GE_ASSERT_TRUE(axis.size() == strides.size(), "Broadcast axis/strides size mismatch: %zu vs %zu", axis.size(),
                 strides.size());
  GE_ASSERT_TRUE(std::find(axis.begin(), axis.end(), broadcast_axis) != axis.end(),
                 "Broadcast axis[%ld] is not present in tensor axes.", broadcast_axis);
  attr->axis = axis;
  attr->repeats = repeats;
  attr->strides = strides;
  // The output of Broadcast keeps the expanded shape. The compact repeat/stride
  // (repeat=1, stride=0) belongs to the input edge and is supplied by the
  // preceding tensor's output attributes.
  GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(attr->repeats, attr->strides));
  return SUCCESS;
}

Status UpdateBroadcastNodeSchedInfo(const NodePtr &b_node, const NodePtr &ref_node) {
  b_node->attr.sched = ref_node->attr.sched;
  return SUCCESS;
}

Status FromDtypeToOtherDtype(const NodePtr &b_node, af::DataType from_dtype, af::DataType to_dtype) {
  std::vector<af::DataType> input_dtypes = {from_dtype};
  std::vector<af::DataType> expect_output_dtypes = {to_dtype};
  GE_ASSERT_SUCCESS(ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, expect_output_dtypes));
  b_node->outputs[0].attr.dtype = to_dtype;
  return SUCCESS;
}

// 插入广播命名：基础名 = "backward_broadcast_<生产者>_<扩维轴>"。多输出场景同轴
// 多路插入（以及迭代式多轮后移时消费侧已挂上一轮广播）会产生同名，按图内现存
// 节点名追加下划线消歧；单输出场景单路无冲突，命名与历史行为完全一致。多路同
// 构插入随后由收尾 MergeSharedBroadcastConsumers 合并回一条共享广播（合并只保留
// 拓扑序在前的链，名字同样取幸存链），这是收尾阶段的预期行为。
std::string MakeUniqueBrcName(const AscGraph &asc_graph, const NodePtr &node, int64_t axis) {
  std::string name = "backward_broadcast_" + node->GetName() + "_" + std::to_string(axis);
  while (asc_graph.FindNode(name.c_str()) != nullptr) {
    name += "_";
  }
  return name;
}

Status CreateAndUpdateBroadcastNode(AscGraph &asc_graph, const NodePtr &node, const NodePtr &first_connect_node,
                                    int32_t first_connect_in_idx, const TensorInfo &tensor_info,
                                    const TensorInfo &expanded_tensor_info) {
  const std::vector<int64_t> &broadcast_info = tensor_info.broadcast_info;
  GE_ASSERT_TRUE(broadcast_info.size() > 0U);
  GE_ASSERT_TRUE(tensor_info.axis == expanded_tensor_info.axis);
  GE_ASSERT_TRUE(tensor_info.repeats.size() == tensor_info.axis.size());
  GE_ASSERT_TRUE(expanded_tensor_info.repeats.size() == expanded_tensor_info.axis.size());

  TensorInfo current_tensor_info = tensor_info;
  NodePtr connect_node = first_connect_node;
  for (size_t index = 0U; index < broadcast_info.size(); index++) {
    const std::string brc_name = MakeUniqueBrcName(asc_graph, node, broadcast_info[index]);
    Broadcast brc_op(brc_name.c_str());
    auto b_node = asc_graph.AddNode(brc_op);
    GE_ASSERT_NOTNULL(b_node);

    brc_op.attr.sched = node->attr.sched;
    brc_op.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    brc_op.attr.api.type = af::ApiType::kAPITypeCompute;

    // 首个广播接管 first_connect_node 的 first_connect_in_idx 输入连接（多输出场景
    // 生产者输出可多路消费，不能从 out-0 唯一消费边推断插入位）；后续广播串联在
    // 前一个广播之前，接管其输入 0。
    const int32_t connect_in_idx = (index == 0U) ? first_connect_in_idx : 0;
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceNodeDataAnchors(b_node, connect_node, {connect_in_idx}, {}));
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), connect_node->GetInDataAnchor(connect_in_idx)));

    auto next_repeats = current_tensor_info.repeats;
    auto axis_iter = std::find(current_tensor_info.axis.begin(), current_tensor_info.axis.end(), broadcast_info[index]);
    auto expanded_axis_iter =
        std::find(expanded_tensor_info.axis.begin(), expanded_tensor_info.axis.end(), broadcast_info[index]);
    GE_ASSERT_TRUE(axis_iter != current_tensor_info.axis.end());
    GE_ASSERT_TRUE(expanded_axis_iter != expanded_tensor_info.axis.end());
    auto axis_index = static_cast<size_t>(std::distance(current_tensor_info.axis.begin(), axis_iter));
    auto expanded_axis_index =
        static_cast<size_t>(std::distance(expanded_tensor_info.axis.begin(), expanded_axis_iter));
    next_repeats[axis_index] = expanded_tensor_info.repeats[expanded_axis_index];
    auto next_strides = std::vector<Expression>();
    GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(next_repeats, next_strides));
    GE_ASSERT_SUCCESS(
        UpdateBroadcastNodeAttrs(b_node, current_tensor_info.axis, next_repeats, next_strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeSchedInfo(b_node, node));
    GE_ASSERT_SUCCESS(FromDtypeToOtherDtype(b_node, tensor_info.dtype, tensor_info.dtype));
    current_tensor_info.repeats = std::move(next_repeats);
    current_tensor_info.strides = std::move(next_strides);
    connect_node = b_node;
  }
  return SUCCESS;
}

Status InsertBroadcastNode(NodePtr &pre_brc_node, AscGraph &graph, std::set<int64_t> &broadcast_axis,
                           const TensorInfo &expanded_tensor_info) {
  std::vector<int64_t> broadcast_info(broadcast_axis.begin(), broadcast_axis.end());
  TensorInfo tensor_info;
  GE_ASSERT_SUCCESS(GetTensorInfo(pre_brc_node, tensor_info));
  tensor_info.broadcast_info = broadcast_info;
  NodePtr connect_node;
  GE_ASSERT_SUCCESS(GetSingleNextNode(pre_brc_node, connect_node));
  auto out_anchor = pre_brc_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  GE_ASSERT_TRUE(out_anchor->GetPeerInDataAnchors().size() == 1U);
  const int32_t connect_in_idx = out_anchor->GetPeerInDataAnchors().at(0)->GetIdx();
  GE_ASSERT_SUCCESS(CreateAndUpdateBroadcastNode(graph, pre_brc_node, connect_node, connect_in_idx, tensor_info,
                                                 expanded_tensor_info));
  return SUCCESS;
}

// MI-MO 提交：将公共轴 Brc 挨个后移到多输出节点的每路输出分支首节点之前（每路
// 独立广播链）。每路以该输出端口的当前（扩维）视图为基线创建广播链：插入时生产
// 者输出尚未塌缩（UpdateOutputTensor 随后执行），公共轴塌缩后广播恢复扩维语义，
// 各路消费者看到的输入视图保持不变。同轴多路插入的命名按图内现存节点消歧。
Status InsertBroadcastNodeForEachOutput(const NodePtr &pre_node, AscGraph &graph,
                                        const std::set<int64_t> &broadcast_axis) {
  std::vector<int64_t> broadcast_info(broadcast_axis.begin(), broadcast_axis.end());
  const auto out_anchor_size = pre_node->GetAllOutDataAnchorsSize();
  for (uint32_t out_idx = 0U; out_idx < out_anchor_size; ++out_idx) {
    auto out_anchor = pre_node->GetOutDataAnchor(static_cast<int32_t>(out_idx));
    if (out_anchor == nullptr) {
      continue;
    }
    // 逐路改写输出边，先拷贝消费者列表避免遍历中边被替换。
    const auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    for (const auto &in_anchor : peer_in_anchors) {
      if (in_anchor == nullptr) {
        continue;
      }
      auto connect_node = ToAscNode(in_anchor->GetOwnerNode());
      if (connect_node == nullptr) {
        continue;
      }
      TensorInfo expanded_tensor_info;
      GE_ASSERT_SUCCESS(GetTensorInfoAtOutput(pre_node, out_idx, expanded_tensor_info));
      TensorInfo tensor_info = expanded_tensor_info;
      tensor_info.broadcast_info = broadcast_info;
      GE_ASSERT_SUCCESS(CreateAndUpdateBroadcastNode(graph, pre_node, connect_node, in_anchor->GetIdx(), tensor_info,
                                                     expanded_tensor_info));
    }
  }
  return SUCCESS;
}

Status UpdateOutputTensor(std::vector<NodePtr> &nodes, std::set<int64_t> &common_axises) {
  for (auto &node : nodes) {
    // 多输出节点（MI-MO partial backward）的每个输出端口视图都要塌缩公共轴；
    // 单输出节点退化为 outputs[0]，与历史行为一致。
    const auto out_anchor_size = node->GetAllOutDataAnchorsSize();
    for (uint32_t out_idx = 0U; out_idx < out_anchor_size; ++out_idx) {
      AscTensorAttr *compute_output_attr = nullptr;
      GE_ASSERT_SUCCESS(GetOutputTensorAttrAt(node, compute_output_attr, out_idx));
      auto &repeats = compute_output_attr->repeats;
      auto &strides = compute_output_attr->strides;
      auto &attr_axis = compute_output_attr->axis;
      for (auto common_axis : common_axises) {
        auto it = std::find(attr_axis.begin(), attr_axis.end(), common_axis);
        GE_ASSERT_TRUE(it != attr_axis.end());
        auto index = std::distance(attr_axis.begin(), it);
        repeats[index] = af::sym::kSymbolOne;
        strides[index] = af::sym::kSymbolZero;
      }
      GE_ASSERT_SUCCESS(ScheduleUtils::RecalculateStridesFromRepeats(repeats, strides));
    }
  }
  return SUCCESS;
}

// 公共轴塌缩要求每个分支的生产者视图都以退化轴（repeat=1）表达全部公共轴：
// 生产者视图缺失公共轴（1D Arange 前缀扩维，如 [cols] → [rows,cols]）时，
// 删除/塌缩分支广播会让生产者直连计算节点并留下两侧轴集不一致的边（tiling
// 不可融合）。改写前 fail-closed。
// 例外：scalar-like 生产者（Scalar/ScalarData/IndexExpr）输出视图恒为空（构图
// 协议），标量语义与轴无关、可广播到任意视图，公共轴塌缩不改变其取值——免轴
// 检查放行，与图级门控合入前的 partial backward 行为一致（标量链经 Cast 穿透
// 参与多输入组的级联后移依赖此路径）。
bool PartialBackwardBranchesCoverCommonAxes(const std::vector<std::vector<NodePtr>> &brc_nodes_list,
                                            const std::set<int64_t> &common_axises) {
  for (const auto &row : brc_nodes_list) {
    if (row.empty()) {
      return false;
    }
    NodePtr producer;
    if (GetPeerOutNodeSafe(row.front(), producer, 0) != SUCCESS || producer == nullptr) {
      return false;
    }
    if (IsScalarLikeProducer(producer)) {
      continue;
    }
    AscTensorAttr *producer_output_attr = nullptr;
    if (GetOutputTensorAttr(producer, producer_output_attr) != SUCCESS) {
      return false;
    }
    for (const auto common_axis : common_axises) {
      const auto axis_iter =
          std::find(producer_output_attr->axis.begin(), producer_output_attr->axis.end(), common_axis);
      if (axis_iter == producer_output_attr->axis.end()) {
        GELOGI("Skip partial broadcast backward: common axis[%ld] is missing in producer[%s] view.",
               static_cast<long>(common_axis), producer->GetName().c_str());
        return false;
      }
      const size_t axis_index = static_cast<size_t>(std::distance(producer_output_attr->axis.begin(), axis_iter));
      if (!IsEqOne(producer_output_attr->repeats[axis_index])) {
        GELOGI("Skip partial broadcast backward: common axis[%ld] varies (repeat != 1) in producer[%s] view.",
               static_cast<long>(common_axis), producer->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

// MI-MO partial backward 的输出侧守卫：公共轴塌缩（UpdateOutputTensor）与每路
// 广播恢复都按"输出视图包含全部公共轴"推导；任何输出端口缺失公共轴，说明该输出
// 与公共轴解耦（非逐元素语义），改写后无法保证视图代数等价，fail-closed 拒绝整
// 个节点的后移，图保持原状。
bool MultiOutViewCoversCommonAxes(const NodePtr &node, const std::set<int64_t> &common_axises) {
  const auto out_anchor_size = node->GetAllOutDataAnchorsSize();
  for (uint32_t out_idx = 0U; out_idx < out_anchor_size; ++out_idx) {
    AscTensorAttr *output_attr = nullptr;
    if (GetOutputTensorAttrAt(node, output_attr, out_idx) != SUCCESS) {
      return false;
    }
    for (const auto common_axis : common_axises) {
      if (std::find(output_attr->axis.begin(), output_attr->axis.end(), common_axis) == output_attr->axis.end()) {
        GELOGI("Skip multi-output partial backward at node[%s]: output[%u] misses common axis[%ld].",
               node->GetName().c_str(), out_idx, static_cast<long>(common_axis));
        return false;
      }
    }
  }
  return true;
}

Status JudgePartBackward(std::set<NodePtr> &mul_input_nodes, bool &is_changed, AscGraph &graph);

// partial backward 收集与提交阶段：门控全部通过后，单输出形态先沿输出收集计算链并登记
// 链内待删的同轴兄弟广播（收集不改写图，失败时直接返回、图保持原状）；随后统一执行图改
// 写——先删登记的链内同轴兄弟广播与分支链公共轴广播，再按多输出/单输出形态重建广播，
// 最后塌缩参与节点的输出视图；多公共轴时就地递归逐轮迁移剩余公共轴。
Status CollectAndCommitPartBackward(const NodePtr &mul_input_node, std::vector<std::vector<NodePtr>> &brc_nodes_list,
                                    const std::vector<NodePtr> &retained_cast_nodes,
                                    std::vector<NodePtr> &compute_nodes, std::vector<NodePtr> &origin_brc_nodes,
                                    std::set<NodePtr> &next_mul_input_nodes, const TensorInfo &expanded_tensor_info,
                                    std::set<int64_t> &common_axises, bool is_multi_out_mul_input,
                                    bool has_remaining_common_axes, bool &is_changed, AscGraph &graph) {
  std::vector<std::vector<NodePtr>> pending_remove_brcs;
  if (!is_multi_out_mul_input) {
    auto cur_node = mul_input_node;
    auto next_node = mul_input_node;
    GE_ASSERT_SUCCESS(GetSingleNextNode(cur_node, next_node));
    GE_ASSERT_SUCCESS(CollectCmpNodes(cur_node, next_node, compute_nodes, origin_brc_nodes, next_mul_input_nodes,
                                      pending_remove_brcs, &common_axises));
  }
  GELOGD("Move partial broadcast at node[%s]: common_axes=%zu, compute_nodes=%zu, broadcasts=%zu, multi_out=%d.",
         mul_input_node->GetName().c_str(), common_axises.size(), compute_nodes.size(), origin_brc_nodes.size(),
         is_multi_out_mul_input ? 1 : 0);
  is_changed = true;
  // 部分后移的分支链删除与链内同轴兄弟删除操作的是不相交节点集，先应用 pending 与原"探测期删除"
  // 行为终态等价，同时保证失败路径上图未被改写。
  for (std::vector<NodePtr> &remove_nodes : pending_remove_brcs) {
    GE_ASSERT_SUCCESS(RemoveBroadcasts(remove_nodes, graph));
  }
  GE_ASSERT_SUCCESS(RemoveBroadcasts(graph, brc_nodes_list, common_axises));
  if (is_multi_out_mul_input) {
    GE_ASSERT_SUCCESS(InsertBroadcastNodeForEachOutput(mul_input_node, graph, common_axises));
  } else {
    GE_ASSERT_SUCCESS(InsertBroadcastNode(compute_nodes.back(), graph, common_axises, expanded_tensor_info));
  }

  std::vector<NodePtr> merged_nodes;
  for (const auto &row : brc_nodes_list) {
    merged_nodes.insert(merged_nodes.end(), row.begin(), row.end());
  }
  merged_nodes.insert(merged_nodes.end(), retained_cast_nodes.begin(), retained_cast_nodes.end());
  merged_nodes.insert(merged_nodes.end(), compute_nodes.begin(), compute_nodes.end());
  RemoveDuplicates(merged_nodes);
  GE_ASSERT_SUCCESS(UpdateOutputTensor(merged_nodes, common_axises));
  if (has_remaining_common_axes) {
    // 本轮仅迁移一个公共轴：就地递归完成本节点剩余公共轴的逐轮迁移，再继续处
    // 理其他节点。链收集登记的下游多输入节点（next_mul_input_nodes）与同帧后
    // 续节点的分支链包含本节点插入的广播，必须等本节点全部轮次完成后处理，否
    // 则会看到"半迁移"状态（公共轴交集为空）而被跳过且不再重试；重入顺序不能
    // 依赖 next_mul_input_nodes 的指针序。
    std::set<NodePtr> reentry_nodes = {mul_input_node};
    GE_ASSERT_SUCCESS(JudgePartBackward(reentry_nodes, is_changed, graph));
  }
  return SUCCESS;
}

Status JudgePartBackward(std::set<NodePtr> &mul_input_nodes, bool &is_changed, AscGraph &graph) {
  std::set<NodePtr> next_mul_input_nodes;
  for (auto mul_input_node : mul_input_nodes) {
    std::vector<std::vector<NodePtr>> brc_nodes_list;
    std::vector<NodePtr> origin_brc_nodes;
    std::vector<NodePtr> compute_nodes;
    std::vector<NodePtr> retained_cast_nodes;
    std::set<int64_t> common_axises;
    NodePtr peer_out_node = nullptr;
    if (!CollectSameBrcAxis(peer_out_node, mul_input_node, brc_nodes_list, common_axises, origin_brc_nodes,
                            &retained_cast_nodes)) {
      continue;
    }
    if (brc_nodes_list.empty()) {
      GELOGD("Skip partial broadcast backward at node[%s]: branches=%zu.", mul_input_node->GetName().c_str(),
             brc_nodes_list.size());
      continue;
    }
    // 多输入多公共轴：迭代式单轴后移。每轮只迁移一个公共轴（先迁最大轴，重建
    // 链沿数据流按轴升序展开），剩余公共轴由本轮改写后就地递归逐轮处理；每轮
    // 均为已验证的单公共轴路径，重入时重新收集分支链并重过全部门控。
    bool has_remaining_common_axes = false;
    if (common_axises.size() > 1U) {
      const int64_t kept_axis = *common_axises.rbegin();
      common_axises.clear();
      common_axises.insert(kept_axis);
      has_remaining_common_axes = true;
    }
    if (!PartialBackwardBranchesCoverCommonAxes(brc_nodes_list, common_axises)) {
      continue;
    }
    // MI-MO：无唯一后继，不沿输出收集计算链，公共轴 Brc 由提交阶段挨个后移到每
    // 路输出分支首节点之前；输出侧守卫先行，缺失公共轴的输出端口 fail-closed。
    const bool is_multi_out_mul_input = !IsSingleOutNode(mul_input_node);
    if (is_multi_out_mul_input) {
      if (!MultiOutViewCoversCommonAxes(mul_input_node, common_axises)) {
        continue;
      }
    }
    TensorInfo expanded_tensor_info;
    GE_ASSERT_SUCCESS(GetTensorInfo(origin_brc_nodes.back(), expanded_tensor_info));

    compute_nodes.push_back(mul_input_node);
    GE_ASSERT_SUCCESS(CollectAndCommitPartBackward(
        mul_input_node, brc_nodes_list, retained_cast_nodes, compute_nodes, origin_brc_nodes, next_mul_input_nodes,
        expanded_tensor_info, common_axises, is_multi_out_mul_input, has_remaining_common_axes, is_changed, graph));
  }
  if (!next_mul_input_nodes.empty()) {
    return JudgePartBackward(next_mul_input_nodes, is_changed, graph);
  }
  return SUCCESS;
}

Status ProcessOriginalBackwardLogic(AscGraph &graph, bool &is_changed, std::set<NodePtr> &mul_input_nodes) {
  std::vector<NodePtr> start_nodes;
  GE_ASSERT_SUCCESS(CollectBackwardStartNodes(graph, start_nodes));
  RemoveDuplicates(start_nodes);
  for (const auto &start_node : start_nodes) {
    std::vector<NodePtr> peer_in_nodes;
    GE_ASSERT_SUCCESS(CollectBackwardSatisfyStartNodes(start_node, peer_in_nodes));
    for (const auto &peer_in_node : peer_in_nodes) {
      NodePtr cur_node = start_node;
      NodePtr next_node = peer_in_node;

      std::vector<NodePtr> brc_nodes;
      std::vector<NodePtr> compute_nodes;
      std::vector<std::vector<NodePtr>> pending_remove_brcs;
      NodePtr pre_brc_node = start_node;
      GE_ASSERT_SUCCESS(CollectBrcNodes(cur_node, next_node, brc_nodes));
      GE_ASSERT_SUCCESS(CollectCmpNodes(cur_node, next_node, compute_nodes, brc_nodes, mul_input_nodes,
                                        pending_remove_brcs, nullptr));

      GELOGI("Move broadcast backward from node[%s]: broadcasts=%zu, computes=%zu.", peer_in_node->GetName().c_str(),
             brc_nodes.size(), compute_nodes.size());

      if (!brc_nodes.empty() && !compute_nodes.empty()) {
        // 后移会把 compute 链输出改写为后移视图；若链尾下游到达 IndirectLoad，
        // 索引链视图被标量化会导致 IndirectLoad 索引解析错误，必须整体拒绝。
        // 守卫只统计外源广播：本事务广播——链内广播（brc_nodes，后移后停靠在链尾 compute 与 next_node
        // 之间）与已登记待删的同轴兄弟广播（pending_remove_brcs，提交期统一删除）——都不构成拒绝理由。
        // 待删兄弟探测期仍垫在链内多输入节点的输入上，若不排除会被守卫经 peer 的 in-0 生产者检查
        // 误计为外源广播、与真正的外源广播凑满 count≥2 而误拒提交；兄弟删除已延迟到提交期，
        // 图未被改写，拒绝时可干净放弃。
        std::vector<NodePtr> transaction_brc_nodes = brc_nodes;
        for (const std::vector<NodePtr> &remove_nodes : pending_remove_brcs) {
          transaction_brc_nodes.insert(transaction_brc_nodes.end(), remove_nodes.begin(), remove_nodes.end());
        }
        if (HasMultipleInputsBroadcastBehind(next_node, transaction_brc_nodes) || ChainReachesIndirectLoad(next_node)) {
          GELOGI(
              "Skip broadcast backward at node[%s]: next node[%s] has peer inputs with broadcast behind or "
              "downstream reaches IndirectLoad.",
              cur_node->GetName().c_str(), next_node->GetName().c_str());
          continue;
        }
        // 广播后移要求前广播视图与广播输出视图轴数一致；scalar-like 生产者（如 IndexExpr）
        // 输出视图为空，不满足该前提，后移会产生前后视图不一致的半改写状态，必须整体拒绝。
        AscTensorAttr *pre_brc_output_attr = nullptr;
        AscTensorAttr *last_brc_output_attr = nullptr;
        GE_ASSERT_SUCCESS(GetOutputTensorAttr(pre_brc_node, pre_brc_output_attr));
        GE_ASSERT_SUCCESS(GetOutputTensorAttr(brc_nodes.back(), last_brc_output_attr));
        if (pre_brc_output_attr->repeats.size() != last_brc_output_attr->repeats.size()) {
          GELOGI(
              "Skip broadcast backward at node[%s]: pre-broadcast node[%s] view has %zu axes but broadcast "
              "output has %zu axes.",
              peer_in_node->GetName().c_str(), pre_brc_node->GetName().c_str(), pre_brc_output_attr->repeats.size(),
              last_brc_output_attr->repeats.size());
          continue;
        }
        // Arange 生产者的整链后移未适配：整链后移把 compute 链输出改写为 Arange
        // 退化视图，退化轴（size=1/stride=0）的扩维职责语义在视图改写中丢失
        // （triu/remainder_cast 类图 codegen 失败）。链级 fail-closed，保持图原状。
        if (pre_brc_node->GetType() == Arange::Type) {
          GELOGI("Skip broadcast backward at node[%s]: pre-broadcast producer[%s] is Arange.",
                 peer_in_node->GetName().c_str(), pre_brc_node->GetName().c_str());
          continue;
        }
        is_changed = true;
        // 提交期统一执行链上多输入节点的同轴兄弟广播删除：守卫与轴数检查全部通过才动手，
        // 保证"删除 + 移动"原子化，杜绝"兄弟已删、链未后移"的半改写状态。
        for (std::vector<NodePtr> &remove_nodes : pending_remove_brcs) {
          GE_ASSERT_SUCCESS(RemoveBroadcasts(remove_nodes, graph));
        }
        GE_ASSERT_SUCCESS(BroadcastBackwardReally(compute_nodes, brc_nodes, pre_brc_node));
      }
    }
  }
  return SUCCESS;
}

Status BroadcastBackward(AscGraph &graph) {
  if (ScheduleUtils::HasComputeType(graph, af::ComputeType::kComputeCube)) {
    GELOGI("graph %s fuse type is cube, don't backward broadcast.", graph.GetName().c_str());
    return SUCCESS;
  }

  // 后移迭代至不动点：is_changed 复用为每轮迭代的改写标记（初值 true 进入循环，
  // 每轮开头清零；三个后移路径只置 true 不清零），任一路径本轮发生后移则继续整轮
  // 重跑，直到一轮内三个路径均无后移为止。has_moved 记录整个过程中是否存在过一次
  // 后移，替代循环标记驱动收尾的拓扑重排（循环正常退出时 is_changed 恒为 false）。
  bool is_changed = true;
  bool has_moved = false;
  while (is_changed) {
    is_changed = false;
    // Arange 生产者的改写防护已下沉为链级/组级判定：单链与多消费路径按生产者
    // 类型 fail-closed；partial backward 按公共轴在生产者视图中的退化性放行
    // （退化轴 Arange 公共轴后移）或拒绝（1D 前缀扩维）。含 Arange 的图中与
    // Arange 无关的正常链不再被整体跳过。

    GE_ASSERT_SUCCESS(broadcast_backward_shared_split::SplitSharedBroadcastBranches(graph));
    GE_ASSERT_SUCCESS(broadcast_backward_shared_split::SplitSharedBroadcastConsumers(graph));

    std::set<NodePtr> mul_input_nodes;
    GE_ASSERT_SUCCESS(ProcessOriginalBackwardLogic(graph, is_changed, mul_input_nodes));
    if (!mul_input_nodes.empty()) {
      GE_ASSERT_SUCCESS(JudgePartBackward(mul_input_nodes, is_changed, graph));
    }
    GE_ASSERT_SUCCESS(ProcessMultiRefBroadcastBackward(graph, is_changed));

    if (is_changed) {
      has_moved = true;
    }
  }
  // 拆分的反向动作（与 SplitSharedBroadcastConsumers 相反）：后移主循环结束后，把
  // 未能后移且同源同签名的广播链合并回一条共享链（链尾恢复多消费），并将幸存克隆的
  // 命名回退为原始链名，保证"拆分无后移收益"时图恢复等价形态。
  GE_ASSERT_SUCCESS(broadcast_backward_shared_split::MergeSharedBroadcastConsumers(graph));

  if (has_moved) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}
}  // namespace

Status BroadcastBackwardPass::RunPass(af::AscGraph &graph) {
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  GE_ASSERT_SUCCESS(BroadcastBackward(graph));
  GELOGI("Graph %s completed BroadcastBackward successfully.", graph.GetName().c_str());
  return SUCCESS;
}
}  // namespace optimize
