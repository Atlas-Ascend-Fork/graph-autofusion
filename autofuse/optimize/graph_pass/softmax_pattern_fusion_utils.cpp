/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.huawei.com
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "softmax_pattern_fusion_utils.h"

#include <functional>
#include <set>
#include <vector>

#include "ascir_ops.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "graph/utils/graph_utils.h"
#include "node_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace softmax_pattern {
namespace {
constexpr const char *kSoftmaxType = "Softmax";

class SoftmaxOp : public af::Operator {
 public:
  explicit SoftmaxOp(const std::string &name) : af::Operator(name.c_str(), kSoftmaxType) {
    InputRegister("x", "T");
    OutputRegister("y", "T");
  }
};

af::AscNodePtr GetInputNode(const af::AscNodePtr &node, const size_t index) {
  if (node == nullptr) {
    return nullptr;
  }
  const auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(in_anchor->GetPeerOutAnchor()->GetOwnerNode());
}

af::OutDataAnchorPtr GetInputSrcAnchor(const af::AscNodePtr &node, const size_t index) {
  if (node == nullptr) {
    return nullptr;
  }
  const auto in_anchor = node->GetInDataAnchor(index);
  if (in_anchor == nullptr) {
    return nullptr;
  }
  return in_anchor->GetPeerOutAnchor();
}

bool HasOnlyConsumers(const af::AscNodePtr &node, const std::set<af::AscNodePtr> &expected_consumers) {
  if (node == nullptr || node->GetOutDataAnchor(0) == nullptr) {
    return false;
  }
  const auto &peer_in_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != expected_consumers.size()) {
    return false;
  }
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr || peer_in_anchor->GetOwnerNode() == nullptr) {
      return false;
    }
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(peer_in_anchor->GetOwnerNode());
    if (expected_consumers.find(consumer) == expected_consumers.end()) {
      return false;
    }
  }
  return true;
}

bool HasSameTensorLayout(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return lhs.axis == rhs.axis && PassUtils::IsExprVectorEqual(lhs.repeats, rhs.repeats) &&
         PassUtils::IsExprVectorEqual(lhs.strides, rhs.strides);
}

template <typename T>
bool IsOpsSafe(const af::AscNodePtr &node) {
  return node != nullptr && af::ops::IsOps<T>(node);
}

bool MatchStableSoftmaxStructure(const af::AscNodePtr &true_div_node, MatchResult &pattern) {
  if (!IsOpsSafe<af::ascir_op::TrueDiv>(true_div_node)) {
    return false;
  }

  const auto exp_node = GetInputNode(true_div_node, 0UL);
  const auto sum_broadcast_node = GetInputNode(true_div_node, 1UL);
  if (!IsOpsSafe<af::ascir_op::Exp>(exp_node) || !IsOpsSafe<af::ascir_op::Broadcast>(sum_broadcast_node)) {
    return false;
  }

  const auto sum_node = GetInputNode(sum_broadcast_node, 0UL);
  if (!IsOpsSafe<af::ascir_op::Sum>(sum_node) || GetInputSrcAnchor(sum_node, 0UL) != exp_node->GetOutDataAnchor(0)) {
    return false;
  }

  const auto sub_node = GetInputNode(exp_node, 0UL);
  if (!IsOpsSafe<af::ascir_op::Sub>(sub_node)) {
    return false;
  }

  const auto max_broadcast_node = GetInputNode(sub_node, 1UL);
  if (!IsOpsSafe<af::ascir_op::Broadcast>(max_broadcast_node)) {
    return false;
  }

  const auto max_node = GetInputNode(max_broadcast_node, 0UL);
  const auto input_anchor = GetInputSrcAnchor(sub_node, 0UL);
  if (!IsOpsSafe<af::ascir_op::Max>(max_node) || input_anchor == nullptr ||
      GetInputSrcAnchor(max_node, 0UL) != input_anchor) {
    return false;
  }

  pattern = {input_anchor, max_node, max_broadcast_node, sub_node,
             exp_node,     sum_node, sum_broadcast_node, true_div_node};
  return true;
}

bool IsSameReduceLayout(const af::AscNodePtr &max_node, const af::AscNodePtr &sum_node) {
  return max_node != nullptr && sum_node != nullptr &&
         HasSameTensorLayout(max_node->outputs[0].attr, sum_node->outputs[0].attr);
}

bool HasStableSoftmaxLayout(const MatchResult &pattern) {
  if (!IsSameReduceLayout(pattern.max_node, pattern.sum_node)) {
    return false;
  }
  return HasSameTensorLayout(pattern.sum_broadcast_node->outputs[0].attr, pattern.true_div_node->outputs[0].attr) &&
         HasSameTensorLayout(pattern.max_broadcast_node->outputs[0].attr, pattern.sub_node->outputs[0].attr) &&
         HasSameTensorLayout(pattern.exp_node->outputs[0].attr, pattern.true_div_node->outputs[0].attr);
}

bool HasStableSoftmaxConsumers(const MatchResult &pattern) {
  return HasOnlyConsumers(pattern.max_node, {pattern.max_broadcast_node}) &&
         HasOnlyConsumers(pattern.max_broadcast_node, {pattern.sub_node}) &&
         HasOnlyConsumers(pattern.sub_node, {pattern.exp_node}) &&
         HasOnlyConsumers(pattern.exp_node, {pattern.sum_node, pattern.true_div_node}) &&
         HasOnlyConsumers(pattern.sum_node, {pattern.sum_broadcast_node}) &&
         HasOnlyConsumers(pattern.sum_broadcast_node, {pattern.true_div_node});
}

bool IsSupportedDtype(const MatchResult &pattern) {
  const auto &all = af::ascir::AscirRegistry::GetInstance().GetAll();
  const auto it = all.find(kSoftmaxType);
  if (it == all.end()) {
    return false;
  }
  const auto &soc_to_store = it->second.GetSocToDataTypeSymbolStore();
  const auto &store = soc_to_store.empty() ? it->second.GetDataTypeSymbolStore() : soc_to_store.begin()->second;
  const auto &named_syms = store.GetNamedSymbols();
  const auto sym_it = named_syms.find("T");
  if (sym_it == named_syms.end() || sym_it->second == nullptr) {
    return false;
  }
  const auto dtype = pattern.sub_node->inputs[0].attr.dtype;
  return sym_it->second->GetTensorType().tensor_type_impl_->IsDataTypeInRange(dtype);
}
}  // namespace

bool MatchStableStructure(const af::AscNodePtr &true_div_node, MatchResult &pattern) {
  return MatchStableSoftmaxStructure(true_div_node, pattern) && IsSupportedDtype(pattern) &&
         HasStableSoftmaxLayout(pattern) && HasStableSoftmaxConsumers(pattern);
}

bool MatchStableDedicated(const af::AscNodePtr &true_div_node, MatchResult &pattern) {
  return MatchStableStructure(true_div_node, pattern) && ScheduleUtils::IsReduceOnTailAxis(pattern.max_node);
}

bool MatchStable(const af::AscNodePtr &true_div_node, MatchResult &pattern) {
  return MatchStableDedicated(true_div_node, pattern);
}

af::Status RemoveMatchedNodes(const MatchResult &pattern) {
  const std::vector<af::AscNodePtr> nodes_to_remove = {
      pattern.true_div_node, pattern.sum_broadcast_node, pattern.sum_node, pattern.exp_node,
      pattern.sub_node,      pattern.max_broadcast_node, pattern.max_node};
  for (const auto &node : nodes_to_remove) {
    GE_CHECK_NOTNULL(node);
    af::NodeUtils::UnlinkAll(*node);
    GE_CHECK_NOTNULL(node->GetOwnerComputeGraph());
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node));
  }
  return af::SUCCESS;
}

af::Status ReplaceWithSoftmax(af::AscGraph &graph, const MatchResult &pattern) {
  GE_CHECK_NOTNULL(pattern.true_div_node);
  GE_CHECK_NOTNULL(pattern.input_anchor);
  SoftmaxOp softmax_op(pattern.true_div_node->GetName() + "_softmax");
  auto softmax_node = graph.AddNode(softmax_op);
  GE_CHECK_NOTNULL(softmax_node);

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(pattern.input_anchor, softmax_node->GetInDataAnchor(0)));

  softmax_node->attr = pattern.true_div_node->attr;
  softmax_node->inputs[0].attr = pattern.sub_node->inputs[0].attr;
  softmax_node->outputs[0].attr = pattern.true_div_node->outputs[0].attr;
  softmax_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  softmax_node->attr.api.type = af::ApiType::kAPITypeCompute;

  const auto true_div_out_anchor = pattern.true_div_node->GetOutDataAnchor(0);
  GE_CHECK_NOTNULL(true_div_out_anchor);
  const auto peer_in_anchors = true_div_out_anchor->GetPeerInDataAnchors();
  for (const auto &peer_in_anchor : peer_in_anchors) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    GE_ASSERT_GRAPH_SUCCESS(
        af::GraphUtils::ReplaceEdgeSrc(true_div_out_anchor, peer_in_anchor, softmax_node->GetOutDataAnchor(0)));
  }
  return RemoveMatchedNodes(pattern);
}

// 仅替换 input_anchor 可回溯到该 IndirectLoad 输出的 Pattern。Pattern 原始输入
// 同时供给 Max 和 Sub（回流点），两者必须同源；回溯沿 Elementwise 生产者链进行，
// 中间节点允许是多输入 Elementwise（如 gather 结果 +bias / +bmm 的 Add 链）：
// 任一输入路径可达 IndirectLoad 输出即视为源自该 IndirectLoad，其余输入是外部
// 数据，替换后保持链原状（其位于 Softmax 之前，语义不变）。其他形态（Transpose、
// Broadcast、Reduce 等）终止该路径，交由通用复合区域路径处理。
bool FindDirectPostSoftmaxPattern(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  MatchResult &pattern) {
  GE_ASSERT_NOTNULL(indirect_load);
  const auto outputs = indirect_load->outputs();
  GE_ASSERT_TRUE(!outputs.empty() && outputs[0] != nullptr, "IndirectLoad output tensor is missing, node[%s].",
                 indirect_load->GetNamePtr());
  const auto output_anchor = indirect_load->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(output_anchor);

  const auto trace_to_indirect_load = [&output_anchor](const af::OutDataAnchorPtr &anchor) -> bool {
    const std::function<bool(const af::OutDataAnchorPtr &, const size_t)> visit =
        [&visit, &output_anchor](const af::OutDataAnchorPtr &current, const size_t depth) -> bool {
      if (current == nullptr || depth > 16UL) {
        return false;
      }
      if (current == output_anchor) {
        return true;
      }
      const auto owner = std::dynamic_pointer_cast<af::AscNode>(current->GetOwnerNode());
      if (owner == nullptr) {
        return false;
      }
      if (depth > 0UL && (owner->attr.api.compute_type != af::ComputeType::kComputeElewise ||
                          owner->GetInControlNodesSize() != 0UL || owner->GetOutControlNodesSize() != 0UL)) {
        return false;
      }
      for (size_t input_idx = 0UL; input_idx < owner->inputs.Size(); ++input_idx) {
        const auto in_anchor = owner->GetInDataAnchor(input_idx);
        if (in_anchor != nullptr && visit(in_anchor->GetPeerOutAnchor(), depth + 1UL)) {
          return true;
        }
      }
      return false;
    };
    return visit(anchor, 0UL);
  };

  for (const auto &node : graph.GetAllNodes()) {
    if (!MatchStableDedicated(node, pattern)) {
      continue;
    }
    if (!trace_to_indirect_load(pattern.input_anchor)) {
      continue;
    }
    return true;
  }
  return false;
}

af::Status NormalizeDirectPostSoftmax(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &changed) {
  changed = false;
  MatchResult pattern;
  if (!FindDirectPostSoftmaxPattern(graph, indirect_load, pattern)) {
    return af::SUCCESS;
  }
  GELOGI("[IndirectLoad] Normalize direct post Softmax[%s] for IndirectLoad[%s] in candidate graph.",
         pattern.true_div_node->GetNamePtr(), indirect_load->GetNamePtr());
  GE_ASSERT_SUCCESS(ReplaceWithSoftmax(graph, pattern));
  changed = true;
  // 替换后图结构已变，重新排序一次即可；同一回流点的第二个 Pattern
  // 不可能存在（消费者闭合校验排除了多 Pattern 共享输入）。
  GE_ASSERT_SUCCESS(PassUtils::PruneGraph(graph));
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return af::SUCCESS;
}

}  // namespace softmax_pattern
}  // namespace optimize
