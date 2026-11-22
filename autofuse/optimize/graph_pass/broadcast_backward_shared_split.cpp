/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0.
 */
#include "broadcast_backward_shared_split.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"

using namespace ascir;
using namespace af::ascir_op;
using namespace af::ops;

namespace optimize {
namespace broadcast_backward_shared_split {
namespace {

using af::AscGraph;
using af::AscNode;
using af::AscNodePtr;
using af::AscTensorAttr;
using af::FAILED;
using af::SUCCESS;
using NodePtr = AscNodePtr;

constexpr const char *kStoreType = Store::Type;
constexpr const char *kBroadcastType = Broadcast::Type;
constexpr const char *kCastType = Cast::Type;

// 拆分克隆的后缀标记：反向合并阶段的命名回退依赖该后缀识别克隆节点。
constexpr const char *kConsumerSplitSuffix = "_consumer_split_";

NodePtr ToAscNode(const af::NodePtr &node) {
  return std::dynamic_pointer_cast<AscNode>(node);
}

bool IsSingleInAndOutNode(const NodePtr &node) {
  return node != nullptr && node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() == 1UL;
}

Status GetPeerOutNodeSafe(const NodePtr &node, NodePtr &peer, int32_t index) {
  GE_ASSERT_NOTNULL(node);
  if (index < 0 || static_cast<size_t>(index) >= node->GetAllInDataAnchorsSize()) {
    return FAILED;
  }
  auto in_anchor = node->GetInDataAnchor(index);
  GE_ASSERT_NOTNULL(in_anchor);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(out_anchor);
  peer = ToAscNode(out_anchor->GetOwnerNode());
  return SUCCESS;
}

Status GetOutputTensorAttr(const NodePtr &node, AscTensorAttr *&attr) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(node->GetAllOutDataAnchorsSize() > 0U);
  attr = &node->outputs[0].attr;
  return SUCCESS;
}

bool IsViewOp(const NodePtr &node) {
  static const std::vector<std::string> kViewTypes = {
      Transpose::Type, Broadcast::Type, "Slice", Split::Type, Concat::Type, Gather::Type, "Sum",
      "Mean",          "Max",           "Min",   "Prod",      "Any",        "All"};
  return std::find(kViewTypes.begin(), kViewTypes.end(), node->GetType()) != kViewTypes.end();
}

bool IsDtypeNotSupported(const NodePtr &node) {
  if (node->GetType() != kCastType || node->GetOpDesc() == nullptr) {
    return false;
  }
  const auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  if (output_desc == nullptr) {
    return true;
  }
  const auto dtype = output_desc->GetDataType();
  const std::vector<af::DataType> input_dtypes = {dtype};
  std::vector<af::DataType> output_dtypes = {dtype};
  return ScheduleUtils::CallAscirInferDataType<Broadcast>(input_dtypes, output_dtypes) != SUCCESS;
}

bool CanBackward(const NodePtr &node) {
  if (node == nullptr || node->GetType() == kStoreType || IsViewOp(node) || IsDtypeNotSupported(node)) {
    return false;
  }
  return node->GetAllOutDataAnchorsSize() <= 1U && node->GetInDataNodesSize() == 1UL;
}

bool HasNoControlEdges(const NodePtr &node) {
  return node != nullptr && node->GetInControlNodesSize() == 0U && node->GetOutControlNodesSize() == 0U;
}

Status CollectCandidates(const AscGraph &graph, std::vector<NodePtr> &candidates) {
  for (const auto &node : graph.GetAllNodes()) {
    auto candidate = ToAscNode(node);
    if (candidate != nullptr && IsOps<Broadcast>(candidate) && candidate->GetOutDataNodesSize() > 1U) {
      candidates.push_back(candidate);
    }
  }
  return SUCCESS;
}

// 收集"Brc 直连多消费 Cast"候选：单入、单输出锚点、in-0 直连 Broadcast、输出消费
// >1。多消费 Cast 会阻塞主流程的 dtype-aware 单链后移（CanBackward 单出检查），
// 共享时其输出视图又会被 partial backward 的公共轴塌缩殃及全部消费分支。
Status CollectCastSplitCandidates(const AscGraph &graph, std::vector<NodePtr> &candidates) {
  for (const auto &node : graph.GetAllNodes()) {
    auto candidate = ToAscNode(node);
    if (candidate == nullptr || candidate->GetType() != kCastType || candidate->GetInDataNodesSize() != 1UL ||
        candidate->GetAllOutDataAnchorsSize() != 1U) {
      continue;
    }
    auto out = candidate->GetOutDataAnchor(0);
    if (out == nullptr || out->GetPeerInDataAnchors().size() <= 1U) {
      continue;
    }
    NodePtr head_brc = nullptr;
    if (GetPeerOutNodeSafe(candidate, head_brc, 0) == SUCCESS && head_brc != nullptr && IsOps<Broadcast>(head_brc)) {
      candidates.push_back(candidate);
    }
  }
  return SUCCESS;
}

Status CollectBroadcastChain(NodePtr tail, std::vector<NodePtr> &chain, NodePtr &source) {
  NodePtr current = tail;
  while (IsOps<Broadcast>(current) && current->GetInDataNodesSize() == 1UL) {
    chain.push_back(current);
    GE_ASSERT_SUCCESS(GetPeerOutNodeSafe(current, current, 0));
  }
  std::reverse(chain.begin(), chain.end());
  source = current;
  return chain.empty() || source == nullptr || IsOps<Broadcast>(source) ? FAILED : SUCCESS;
}

Status CloneBroadcast(AscGraph &graph, const NodePtr &source, const std::string &name, NodePtr &clone) {
  Broadcast op(name.c_str());
  clone = graph.AddNode(op);
  GE_ASSERT_NOTNULL(clone);
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(source, attr));
  op.attr.sched = source->attr.sched;
  op.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
  op.y.dtype = attr->dtype;
  *op.y.axis = attr->axis;
  *op.y.repeats = attr->repeats;
  *op.y.strides = attr->strides;
  return SUCCESS;
}

// 克隆 Cast：与 CloneBroadcast 对称，复制 sched、api 与输出视图/dtype；输入视图由
// 接边时的生产者输出决定。compute_type 由 Cast 构造函数按算子注册初始化。
Status CloneCast(AscGraph &graph, const NodePtr &source, const std::string &name, NodePtr &clone) {
  Cast op(name.c_str());
  clone = graph.AddNode(op);
  GE_ASSERT_NOTNULL(clone);
  AscTensorAttr *attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(source, attr));
  op.attr.sched = source->attr.sched;
  op.attr.api.type = source->attr.api.type;
  op.attr.api.compute_type = source->attr.api.compute_type;
  op.y.dtype = attr->dtype;
  *op.y.axis = attr->axis;
  *op.y.repeats = attr->repeats;
  *op.y.strides = attr->strides;
  return SUCCESS;
}

bool AllSame(const std::vector<NodePtr> &nodes) {
  return nodes.empty() ||
         std::all_of(nodes.begin() + 1, nodes.end(), [&](const NodePtr &node) { return node == nodes.front(); });
}

struct BranchPlan {
  std::vector<NodePtr> consumers;
  std::vector<NodePtr> successors;
  std::vector<int32_t> successor_inputs;
};

Status BuildBranchPlan(const NodePtr &tail, BranchPlan &plan) {
  auto out = tail->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out);
  const auto &peers = out->GetPeerInDataAnchors();
  if (peers.size() < 2U || peers.size() > 8U) {
    return FAILED;
  }
  for (const auto &peer : peers) {
    if (peer == nullptr) {
      return FAILED;
    }
    auto consumer = ToAscNode(peer->GetOwnerNode());
    if (!IsSingleInAndOutNode(consumer) || !CanBackward(consumer)) {
      return FAILED;
    }
    auto successor = ToAscNode(consumer->GetOutDataNodes().at(0));
    if (successor == nullptr || successor->GetAllInDataAnchorsSize() <= 1U) {
      return FAILED;
    }
    int32_t input = -1;
    for (uint32_t i = 0U; i < successor->GetAllInDataAnchorsSize(); ++i) {
      auto anchor = successor->GetInDataAnchor(static_cast<int32_t>(i));
      if (anchor != nullptr && anchor->GetPeerOutAnchor() != nullptr &&
          ToAscNode(anchor->GetPeerOutAnchor()->GetOwnerNode()) == consumer) {
        input = static_cast<int32_t>(i);
        break;
      }
    }
    if (input < 0) {
      return FAILED;
    }
    plan.consumers.push_back(consumer);
    plan.successors.push_back(successor);
    plan.successor_inputs.push_back(input);
  }
  return SUCCESS;
}

Status BuildBranchChains(AscGraph &graph, const std::vector<NodePtr> &original, const BranchPlan &plan,
                         std::vector<std::vector<NodePtr>> &chains) {
  chains.resize(plan.consumers.size());
  chains[0] = original;
  for (size_t branch = 1U; branch < chains.size(); ++branch) {
    for (const auto &node : original) {
      NodePtr clone;
      GE_ASSERT_SUCCESS(
          CloneBroadcast(graph, node, node->GetName() + "_branch_split_" + std::to_string(branch), clone));
      chains[branch].push_back(clone);
    }
  }
  return SUCCESS;
}

Status ApplyBranchPlan(const NodePtr &source, const std::vector<NodePtr> &original, const BranchPlan &plan,
                       const std::vector<std::vector<NodePtr>> &chains) {
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::RemoveEdge(source->GetOutDataAnchor(0), original.front()->GetInDataAnchor(0)));
  for (size_t i = 0U; i + 1U < original.size(); ++i) {
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveEdge(original[i]->GetOutDataAnchor(0), original[i + 1U]->GetInDataAnchor(0)));
  }
  AscTensorAttr *source_attr = nullptr;
  GE_ASSERT_SUCCESS(GetOutputTensorAttr(source, source_attr));
  for (size_t branch = 0U; branch < plan.consumers.size(); ++branch) {
    auto consumer = plan.consumers[branch];
    auto successor = plan.successors[branch];
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveEdge(original.back()->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(consumer->GetOutDataAnchor(0),
                                                       successor->GetInDataAnchor(plan.successor_inputs[branch])));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(source->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)));
    const auto &chain = chains[branch];
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(consumer->GetOutDataAnchor(0), chain.front()->GetInDataAnchor(0)));
    for (size_t i = 0U; i + 1U < chain.size(); ++i) {
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(chain[i]->GetOutDataAnchor(0), chain[i + 1U]->GetInDataAnchor(0)));
    }
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(chain.back()->GetOutDataAnchor(0),
                                                    successor->GetInDataAnchor(plan.successor_inputs[branch])));
    AscTensorAttr *consumer_attr = nullptr;
    GE_ASSERT_SUCCESS(GetOutputTensorAttr(consumer, consumer_attr));
    consumer_attr->axis = source_attr->axis;
    consumer_attr->repeats = source_attr->repeats;
    consumer_attr->strides = source_attr->strides;
  }
  return SUCCESS;
}

Status SplitOneBranch(AscGraph &graph, const NodePtr &tail, bool &changed) {
  changed = false;
  std::vector<NodePtr> original;
  NodePtr source;
  if (CollectBroadcastChain(tail, original, source) != SUCCESS) {
    return SUCCESS;
  }
  BranchPlan plan;
  if (BuildBranchPlan(tail, plan) != SUCCESS || AllSame(plan.successors)) {
    return SUCCESS;
  }
  std::vector<std::vector<NodePtr>> chains;
  GE_ASSERT_SUCCESS(BuildBranchChains(graph, original, plan, chains));
  GE_ASSERT_SUCCESS(ApplyBranchPlan(source, original, plan, chains));
  changed = true;
  return SUCCESS;
}

struct ConsumerPlan {
  NodePtr source;
  // 广播链 head→tail；广播候选时 tail 即候选节点（单节点广播退化为 size==1），
  // Cast 候选时 tail 为多消费 Cast、其余为 Cast in-0 直连的广播链（整链复制拆分）。
  std::vector<NodePtr> chain;
  std::vector<NodePtr> consumers;
  std::vector<int32_t> inputs;
};

// 无条件拆分门控（仅保留结构性与正确性检查，不再做收益预测）：
// - 链可收集、source 非 Broadcast（CollectBroadcastChain 的结构前提）；
// - 确为多消费（>1 个消费边）；
// - 链上节点无控制边：克隆节点不带控制边，带约束的链被拆分后克隆分支会丢失顺序
//   约束，fail-closed 不拆（对齐主流程/多消费路径对控制边的处理）。
// 拆分后无后移收益的分支由 MergeSharedBroadcastConsumers 在主循环后合并回滚。
Status BuildConsumerPlan(const NodePtr &candidate, ConsumerPlan &plan) {
  if (candidate->GetType() == kCastType) {
    // "Brc 直连多消费 Cast"候选：Cast 入链尾，链 = Cast in-0 广播链 + Cast。
    // 门控对齐主流程 GetBroadcastPeerThroughCast 的穿透条件：单入 Cast、in-0
    // 直连 Broadcast——共享 Cast 的输出视图会被公共轴塌缩一并改写，殃及其余消费
    // 分支，必须与共享广播链一样按消费者整链拆分（只拆 Cast 不拆链会让
    // ReverseCollectBrcNodes 因链头多消费收集为空，公共轴识别失效）。
    NodePtr head_brc = nullptr;
    if (candidate->GetInDataNodesSize() != 1UL || GetPeerOutNodeSafe(candidate, head_brc, 0) != SUCCESS ||
        head_brc == nullptr || !IsOps<Broadcast>(head_brc)) {
      return FAILED;
    }
    if (CollectBroadcastChain(head_brc, plan.chain, plan.source) != SUCCESS) {
      return FAILED;
    }
    plan.chain.push_back(candidate);
  } else if (CollectBroadcastChain(candidate, plan.chain, plan.source) != SUCCESS) {
    // 收集候选所在广播链：单节点广播退化为 size==1；多级链由整链复制拆分。
    return FAILED;
  }
  auto out = candidate->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out);
  const auto &peers = out->GetPeerInDataAnchors();
  if (peers.size() <= 1U) {
    return FAILED;
  }
  for (const auto &node : plan.chain) {
    if (!HasNoControlEdges(node)) {
      return FAILED;
    }
  }
  for (const auto &peer : peers) {
    auto consumer = peer == nullptr ? nullptr : ToAscNode(peer->GetOwnerNode());
    if (consumer == nullptr) {
      return FAILED;
    }
    plan.consumers.push_back(consumer);
    plan.inputs.push_back(peer->GetIdx());
  }
  return SUCCESS;
}

// 克隆节点命名：默认沿用"<链成员名>_consumer_split_<i>"；嵌套多消费拆分（链成员同时
// 是其他候选的链成员）时基名可能已被占用，追加候选名消歧，仍冲突时追加下划线兜底。
std::string MakeUniqueCloneName(const AscGraph &graph, const NodePtr &node, const NodePtr &candidate, size_t index) {
  const auto suffix = kConsumerSplitSuffix + std::to_string(index);
  std::string name = node->GetName() + suffix;
  if (graph.FindNode(name.c_str()) == nullptr) {
    return name;
  }
  name = node->GetName() + "_" + candidate->GetName() + suffix;
  while (graph.FindNode(name.c_str()) != nullptr) {
    name = name + "_";
  }
  return name;
}

Status SplitOneConsumer(AscGraph &graph, const NodePtr &candidate, bool &changed) {
  changed = false;
  ConsumerPlan plan;
  if (BuildConsumerPlan(candidate, plan) != SUCCESS) {
    return SUCCESS;
  }
  // source 输出锚点取链头 in-0 的实际 peer 锚点：source 为多输出节点时克隆链必须
  // 挂在与原链相同的输出锚点上，不能假设 out-0。
  auto source_out = plan.chain.front()->GetInDataAnchor(0)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(source_out);
  for (size_t i = 1U; i < plan.consumers.size(); ++i) {
    // 整链复制：单节点广播退化为克隆 1 个节点，与既有行为一致；Cast 候选的链尾
    // Cast 一并克隆，保证每个消费分支独占"广播链 + Cast"单链。
    std::vector<NodePtr> clone_chain;
    clone_chain.reserve(plan.chain.size());
    for (const auto &node : plan.chain) {
      NodePtr clone;
      const auto clone_name = MakeUniqueCloneName(graph, node, candidate, i);
      if (node->GetType() == kCastType) {
        GE_ASSERT_SUCCESS(CloneCast(graph, node, clone_name, clone));
      } else {
        GE_ASSERT_SUCCESS(CloneBroadcast(graph, node, clone_name, clone));
      }
      clone_chain.push_back(clone);
    }
    auto consumer_input = plan.consumers[i]->GetInDataAnchor(plan.inputs[i]);
    GE_ASSERT_NOTNULL(consumer_input);
    auto peer_out = consumer_input->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out);
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(peer_out, consumer_input));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(source_out, clone_chain.front()->GetInDataAnchor(0)));
    for (size_t j = 0U; j + 1U < clone_chain.size(); ++j) {
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(clone_chain[j]->GetOutDataAnchor(0), clone_chain[j + 1U]->GetInDataAnchor(0)));
    }
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(clone_chain.back()->GetOutDataAnchor(0), consumer_input));
  }
  changed = true;
  return SUCCESS;
}

// -------------------- 反向合并（SplitSharedBroadcastConsumers 的逆操作） --------------------

bool IsExpressionVectorEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t index = 0U; index < lhs.size(); ++index) {
    if (af::SymbolicUtils::StaticCheckEq(lhs[index], rhs[index]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

// sched 一致性只比较有效字段（axis/loop_axis/exec_condition）；exec_order 已废弃，不参与
// 判定，避免兼容字段漂移造成假阴性。
bool IsSchedInfoEquivalent(const af::SchedInfo &lhs, const af::SchedInfo &rhs) {
  return lhs.axis == rhs.axis && lhs.loop_axis == rhs.loop_axis && lhs.exec_condition == rhs.exec_condition;
}

// 合并等价性：同源锚点前提下逐节点比较输出视图（axis/repeats/strides）、dtype 与 sched。
// 同源保证链头输入视图一致，逐位置输出视图一致经归纳保证全链等价，重定向消费边后
// 消费者看到的张量不变。带控制边的节点不参与（删除会丢失顺序约束）。
bool AreBroadcastNodesEquivalent(const NodePtr &lhs, const NodePtr &rhs) {
  if (!HasNoControlEdges(lhs) || !HasNoControlEdges(rhs)) {
    return false;
  }
  AscTensorAttr *lhs_attr = nullptr;
  AscTensorAttr *rhs_attr = nullptr;
  if (GetOutputTensorAttr(lhs, lhs_attr) != SUCCESS || GetOutputTensorAttr(rhs, rhs_attr) != SUCCESS) {
    return false;
  }
  if (lhs_attr->axis != rhs_attr->axis || lhs_attr->dtype != rhs_attr->dtype) {
    return false;
  }
  if (!IsExpressionVectorEqual(lhs_attr->repeats, rhs_attr->repeats) ||
      !IsExpressionVectorEqual(lhs_attr->strides, rhs_attr->strides)) {
    return false;
  }
  return IsSchedInfoEquivalent(lhs->attr.sched, rhs->attr.sched);
}

// 尾 Cast 等价性：与广播节点同口径比较输出视图、dtype 与 sched（Cast 单入单出，
// 输入侧由其所挂的等价广播链保证）。等价 Cast 产出完全相同的张量，可安全去重。
bool AreCastNodesEquivalent(const NodePtr &lhs, const NodePtr &rhs) {
  if (!HasNoControlEdges(lhs) || !HasNoControlEdges(rhs)) {
    return false;
  }
  AscTensorAttr *lhs_attr = nullptr;
  AscTensorAttr *rhs_attr = nullptr;
  if (GetOutputTensorAttr(lhs, lhs_attr) != SUCCESS || GetOutputTensorAttr(rhs, rhs_attr) != SUCCESS) {
    return false;
  }
  if (lhs_attr->axis != rhs_attr->axis || lhs_attr->dtype != rhs_attr->dtype) {
    return false;
  }
  if (!IsExpressionVectorEqual(lhs_attr->repeats, rhs_attr->repeats) ||
      !IsExpressionVectorEqual(lhs_attr->strides, rhs_attr->strides)) {
    return false;
  }
  return IsSchedInfoEquivalent(lhs->attr.sched, rhs->attr.sched);
}

// shorter 是否与 longer 的等价前缀：同源拆分的原链与克隆链链长可能不同（原链在下游
// 延伸更长，如拆分点之后还有连续广播），反向合并必须按前缀位置对齐，否则等长比较
// 会漏合并、把"拆分解除共享"的中间形态泄漏给后续 pass。
bool IsPrefixEquivalent(const std::vector<NodePtr> &longer, const std::vector<NodePtr> &shorter) {
  if (shorter.empty() || shorter.size() > longer.size()) {
    return false;
  }
  for (size_t index = 0U; index < shorter.size(); ++index) {
    if (!AreBroadcastNodesEquivalent(longer[index], shorter[index])) {
      return false;
    }
  }
  return true;
}

// 纯单消费广播链节点：单输入 Broadcast 且输出恰好 1 个消费者。拆分产生的克隆链与
// 去共享后的原链均满足该形态；链尾已多消费的广播本就是合并形态，不参与收集。
bool IsPureChainNode(const NodePtr &node) {
  if (node == nullptr || node->GetType() != kBroadcastType || node->GetInDataNodesSize() != 1UL) {
    return false;
  }
  auto out = node->GetOutDataAnchor(0);
  return out != nullptr && out->GetPeerInDataAnchors().size() == 1U;
}

struct PureChain {
  af::OutDataAnchorPtr source_anchor;  // 链头 in-0 的 peer 输出锚点（同源判定键）
  std::vector<NodePtr> nodes;          // head → tail
  NodePtr tail_cast = nullptr;         // 链尾广播唯一消费者为可合并 Cast 时记录
};

// 链尾广播的唯一消费者若为"单入、单输出锚点、单消费边、无控制边"的 Cast，记为
// tail_cast：等价链合并时若双方尾 Cast 亦等价，则广播链合并后继续合并尾 Cast
// （等价 Cast 去重），使"Brc 直连多消费 Cast"拆分在无后移收益时能整链回滚。
void GetMergeableTailCast(const NodePtr &tail, NodePtr &tail_cast) {
  tail_cast = nullptr;
  auto out = tail->GetOutDataAnchor(0);
  if (out == nullptr || out->GetPeerInDataAnchors().size() != 1U) {
    return;
  }
  auto cast = ToAscNode(out->GetPeerInDataAnchors().at(0)->GetOwnerNode());
  if (cast == nullptr || cast->GetType() != kCastType || cast->GetInDataNodesSize() != 1UL ||
      cast->GetAllOutDataAnchorsSize() != 1U || !HasNoControlEdges(cast)) {
    return;
  }
  auto cast_out = cast->GetOutDataAnchor(0);
  if (cast_out == nullptr || cast_out->GetPeerInDataAnchors().size() != 1U) {
    return;
  }
  tail_cast = cast;
}

// 仅从链头（生产者不是纯链 Broadcast）开始收集，沿唯一消费者向下延伸，保证每个
// 纯链节点恰好被一条极大链收集一次。
Status CollectPureBroadcastChains(const AscGraph &graph, std::vector<PureChain> &chains) {
  for (const auto &node : graph.GetAllNodes()) {
    auto head = ToAscNode(node);
    if (!IsPureChainNode(head)) {
      continue;
    }
    NodePtr producer;
    if (GetPeerOutNodeSafe(head, producer, 0) != SUCCESS || IsPureChainNode(producer)) {
      continue;
    }
    PureChain chain;
    auto head_in = head->GetInDataAnchor(0);
    chain.source_anchor = head_in == nullptr ? nullptr : head_in->GetPeerOutAnchor();
    auto cur = head;
    while (IsPureChainNode(cur)) {
      chain.nodes.push_back(cur);
      auto out = cur->GetOutDataAnchor(0);
      if (out == nullptr || out->GetPeerInDataAnchors().empty()) {
        break;
      }
      cur = ToAscNode(out->GetPeerInDataAnchors().at(0)->GetOwnerNode());
    }
    if (chain.source_anchor != nullptr && !chain.nodes.empty()) {
      GetMergeableTailCast(chain.nodes.back(), chain.tail_cast);
      chains.push_back(std::move(chain));
    }
  }
  return SUCCESS;
}

// dup 链（keep 的等价前缀）尾全部消费者的输入边逐边改接 keep 链同位置节点，拆除 dup
// 链全部数据边后删除节点。重定向目标是 keep.nodes[dup_len-1] 而非 keep 链尾：dup 与
// keep 仅在前 dup_len 个位置等价，链长不同时 keep 链尾的扩维视图与 dup 链尾不同。
// 链尾单消费者只在收集时刻成立：合并循环中作为 keep 吸收过消费者的链之后仍可能作为
// dup 参与合并，必须改接其链尾全部消费边，仅改接首个会把其余消费边随节点删除丢弃。
Status MergeChainInto(AscGraph &graph, const PureChain &keep, const PureChain &dup) {
  const size_t dup_len = dup.nodes.size();
  auto dup_tail_out = dup.nodes.back()->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(dup_tail_out);
  GE_ASSERT_TRUE(!dup_tail_out->GetPeerInDataAnchors().empty());
  const auto peer_ins = dup_tail_out->GetPeerInDataAnchors();
  auto keep_peer_out = keep.nodes[dup_len - 1U]->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(keep_peer_out);
  for (const auto &peer_in : peer_ins) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(dup_tail_out, peer_in));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(keep_peer_out, peer_in));
  }

  auto head_in = dup.nodes.front()->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(head_in);
  auto head_peer_out = head_in->GetPeerOutAnchor();
  if (head_peer_out != nullptr) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(head_peer_out, head_in));
  }
  for (size_t index = 0U; index + 1U < dup.nodes.size(); ++index) {
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveEdge(dup.nodes[index]->GetOutDataAnchor(0), dup.nodes[index + 1U]->GetInDataAnchor(0)));
  }
  for (const auto &node : dup.nodes) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), node));
    af::NodeUtils::UnlinkAll(*node);
  }
  // 尾 Cast 合并：双方链等长且尾 Cast 等价时，dup 尾 Cast 的消费者改接 keep 尾
  // Cast 后删除（等价 Cast 去重，与广播链合并同步回滚）。链长不同（前缀合并）时
  // dup 尾 Cast 挂在 keep 链中间位置，与 keep 链尾 Cast 视图可能不同，保持独立。
  if (keep.tail_cast != nullptr && dup.tail_cast != nullptr && keep.nodes.size() == dup.nodes.size() &&
      AreCastNodesEquivalent(keep.tail_cast, dup.tail_cast)) {
    auto dup_cast_out = dup.tail_cast->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(dup_cast_out);
    GE_ASSERT_TRUE(!dup_cast_out->GetPeerInDataAnchors().empty());
    // 与主链尾同口径：keep 尾 Cast 亦可能在合并循环中累积消费者，改接全部消费边。
    const auto cast_peer_ins = dup_cast_out->GetPeerInDataAnchors();
    auto keep_cast_out = keep.tail_cast->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(keep_cast_out);
    for (const auto &cast_peer_in : cast_peer_ins) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(dup_cast_out, cast_peer_in));
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(keep_cast_out, cast_peer_in));
    }
    auto dup_cast_in = dup.tail_cast->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(dup_cast_in);
    auto dup_cast_peer_out = dup_cast_in->GetPeerOutAnchor();
    if (dup_cast_peer_out != nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(dup_cast_peer_out, dup_cast_in));
    }
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::RemoveNodeWithoutRelink(af::AscGraphUtils::GetComputeGraph(graph), dup.tail_cast));
    af::NodeUtils::UnlinkAll(*dup.tail_cast);
  }
  return SUCCESS;
}

Status MergeEquivalentChains(AscGraph &graph, bool &changed) {
  std::vector<PureChain> chains;
  GE_ASSERT_SUCCESS(CollectPureBroadcastChains(graph, chains));
  std::vector<bool> merged(chains.size(), false);
  for (size_t i = 0U; i < chains.size(); ++i) {
    if (merged[i]) {
      continue;
    }
    for (size_t j = i + 1U; j < chains.size(); ++j) {
      if (merged[j] || chains[j].source_anchor != chains[i].source_anchor) {
        continue;
      }
      // i 严格更短且为 j 的等价前缀：保留更长的 j（额外扩维不可丢失），i 并入后删除，
      // 须立即终止内层循环避免访问悬空节点。等长场景一律走下一分支保留 i（拓扑序在
      // 前的原链），保证原链优先幸存、克隆链回滚。
      if (chains[i].nodes.size() < chains[j].nodes.size() && IsPrefixEquivalent(chains[j].nodes, chains[i].nodes)) {
        GE_ASSERT_SUCCESS(MergeChainInto(graph, chains[j], chains[i]));
        merged[i] = true;
        changed = true;
        break;
      }
      if (IsPrefixEquivalent(chains[i].nodes, chains[j].nodes)) {
        GE_ASSERT_SUCCESS(MergeChainInto(graph, chains[i], chains[j]));
        merged[j] = true;
        changed = true;
      }
    }
  }
  return SUCCESS;
}

// 拆分克隆的命名回退：后移改写可能删除原链而让克隆幸存（如链尾同轴兄弟删除），此时
// 克隆是原始广播的唯一语义继承者，剥掉拆分后缀恢复原名可保持图命名稳定（原名已被
// 其他节点占用时不回退，按图序先到先得，保证确定性）。
Status CanonicalizeSplitCloneNames(AscGraph &graph, bool &changed) {
  for (const auto &node : graph.GetAllNodes()) {
    auto clone = ToAscNode(node);
    if (clone == nullptr || (clone->GetType() != kBroadcastType && clone->GetType() != kCastType)) {
      continue;
    }
    const std::string name = clone->GetName();
    const auto pos = name.rfind(kConsumerSplitSuffix);
    if (pos == std::string::npos || pos == 0U) {
      continue;
    }
    // 后缀后必须全为数字（含消歧兜底追加的下划线），避免误伤含相同子串的普通命名。
    bool digits_only = true;
    for (size_t index = pos + strlen(kConsumerSplitSuffix); index < name.size(); ++index) {
      if (std::isdigit(static_cast<unsigned char>(name[index])) == 0 && name[index] != '_') {
        digits_only = false;
        break;
      }
    }
    if (!digits_only) {
      continue;
    }
    const std::string base = name.substr(0U, pos);
    if (base.empty() || graph.FindNode(base.c_str()) != nullptr) {
      continue;
    }
    auto op_desc = clone->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    op_desc->SetName(base);
    changed = true;
  }
  return SUCCESS;
}

}  // namespace

Status SplitSharedBroadcastBranches(AscGraph &graph) {
  std::vector<NodePtr> candidates;
  GE_ASSERT_SUCCESS(CollectCandidates(graph, candidates));
  bool changed = false;
  for (const auto &candidate : candidates) {
    bool candidate_changed = false;
    GE_ASSERT_SUCCESS(SplitOneBranch(graph, candidate, candidate_changed));
    changed = changed || candidate_changed;
  }
  if (changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}

Status SplitSharedBroadcastConsumers(AscGraph &graph) {
  std::vector<NodePtr> candidates;
  GE_ASSERT_SUCCESS(CollectCandidates(graph, candidates));
  GE_ASSERT_SUCCESS(CollectCastSplitCandidates(graph, candidates));
  bool changed = false;
  for (const auto &candidate : candidates) {
    bool candidate_changed = false;
    GE_ASSERT_SUCCESS(SplitOneConsumer(graph, candidate, candidate_changed));
    changed = changed || candidate_changed;
  }
  if (changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}

Status MergeSharedBroadcastConsumers(AscGraph &graph) {
  bool changed = false;
  GE_ASSERT_SUCCESS(MergeEquivalentChains(graph, changed));
  GE_ASSERT_SUCCESS(CanonicalizeSplitCloneNames(graph, changed));
  if (changed) {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return SUCCESS;
}

}  // namespace broadcast_backward_shared_split
}  // namespace optimize
