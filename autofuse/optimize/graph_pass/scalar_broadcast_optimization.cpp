/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scalar_broadcast_optimization.h"
#include "schedule_result.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "common_utils.h"

using namespace ascir;
using namespace af::ascir_op;
using namespace af::ops;

namespace {
constexpr int32_t kFirstInputIndex = 0;
constexpr int32_t kSecondInputIndex = 1;
constexpr int32_t kSupportScalarInputNum = 2;
}  // namespace

namespace optimize {
// 检查VF节点是否不支持Scalar输入，用于决定是否跳过scalar_broadcast优化
static bool IsVfNodeScalarUnsupported(const af::AscNodePtr &next_node, const af::AscNodePtr &first_brc_node) {
  // 判断节点是否支持VF
  if (!ascgen_utils::IsNodeSupportsVectorFunction(next_node)) {
    GELOGD("Node %s(%s) does not support VF, continue with other checks.", next_node->GetTypePtr(),
           next_node->GetNamePtr());
    return false;
  }
  // 从Broadcast链首个节点追溯源节点
  if (first_brc_node->GetInDataNodesSize() == 0UL) {
    GELOGD("First broadcast node %s(%s) has no input, skip optimization.", first_brc_node->GetTypePtr(),
           first_brc_node->GetNamePtr());
    return true;  // 无法追溯源节点，跳过优化（保守处理）
  }
  const auto &source_node = std::dynamic_pointer_cast<af::AscNode>(first_brc_node->GetInDataNodes().at(0UL));
  if (source_node == nullptr) {
    GELOGD("Source node of first broadcast %s(%s) is nullptr, skip optimization.", first_brc_node->GetTypePtr(),
           first_brc_node->GetNamePtr());
    return true;  // 源节点为空，跳过优化
  }
  // 源节点不是Scalar节点，跳过优化
  if (!IsOps<af::ascir_op::Scalar>(source_node)) {
    GELOGD("Source node %s(%s) is not Scalar, skip optimization for VF node %s(%s).", source_node->GetTypePtr(),
           source_node->GetNamePtr(), next_node->GetTypePtr(), next_node->GetNamePtr());
    return true;
  }
  // Scalar场景：检查Micro API是否支持scalar输入
  if (!ScheduleUtils::IsMicroApiSupportsScalarInput(next_node)) {
    GELOGD("VF node %s(%s) micro API does not support scalar input, skip optimization.", next_node->GetTypePtr(),
           next_node->GetNamePtr());
    return true;
  }
  return false;
}
Status ScalarBroadcastOptimizationPass::GetNodeScalarInputList(const af::NodePtr &node,
                                                               std::vector<bool> &is_scalar_list) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<af::AscNode>(node));
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  is_scalar_list.resize(asc_node->GetInDataNodesSize(), false);
  for (size_t i = 0UL; i < is_scalar_list.size(); ++i) {
    is_scalar_list[i] = ascgen_utils::IsScalarInput(asc_node->inputs[i].attr.repeats);
  }
  return af::SUCCESS;
}

Status ScalarBroadcastOptimizationPass::IsNextNodeSupportScalarInput(const NodeView &brc_node, bool &is_supported,
                                                                     const af::AscNodePtr &first_brc_node) {
  is_supported = false;
  if (!IsOps<Broadcast>(brc_node)) {
    return af::SUCCESS;
  }

  for (const auto &out_anchor : brc_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor);
    for (const auto &next_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      GE_CHECK_NOTNULL(next_in_anchor);
      const auto &next_node = std::dynamic_pointer_cast<af::AscNode>(next_in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(next_node);
      // step1: 判断节点是否支持VF，如果支持VF，则需要判断节点的VF接口是否支持scalar
      if (IsVfNodeScalarUnsupported(next_node, first_brc_node)) {
        return af::SUCCESS;
      }
      // step2: 若当前节点支持全部是Scalar，则继续校验其他节点
      if (ascgen_utils::IsNodeSupportsAllScalar(next_node)) {
        continue;
      }

      // step3: 不支持全部是Scalar，则判断当前输入是Scalar时是否支持，若支持，则继续校验其他节点
      std::vector<bool> is_scalar_list;
      GE_ASSERT_SUCCESS(GetNodeScalarInputList(next_node, is_scalar_list));
      const auto idx = static_cast<size_t>(next_in_anchor->GetIdx());
      GE_ASSERT_TRUE(idx < is_scalar_list.size(), "Input index(%zu) of %s(%s) out of range(%zu)", idx,
                     next_node->GetTypePtr(), next_node->GetNamePtr(), is_scalar_list.size());
      is_scalar_list[idx] = true;
      if (ascgen_utils::IsNodeSupportsScalarInput(next_node, is_scalar_list)) {
        GELOGD("Node %s(%s) supports scalar input, index=%zu.", next_node->GetTypePtr(), next_node->GetNamePtr(), idx);
        continue;
      }

      // 因为1个输入和超过2个输入场景比较简单，不需要调换输入，直接结束
      if (is_scalar_list.size() != static_cast<size_t>(kSupportScalarInputNum)) {
        return af::SUCCESS;  // 直接返回， is_supported 为 false
      }

      // step4: 判断调换输入是否可以支持，若不支持，则结束
      if (!ascgen_utils::IsNodeSupportsScalarIfExchangeInputs(next_node, is_scalar_list)) {
        return af::SUCCESS;  // 直接返回， is_supported 为 false
      }

      // step5: 若调换的输入已经是scalar，则不支持，结束
      const int32_t swap_input_index = kSecondInputIndex - static_cast<int32_t>(idx);  // 总共2个输入
      if (ascgen_utils::IsScalarInput(next_node->inputs[swap_input_index].attr.repeats)) {
        GELOGD("The input index 1 of %s[%s] is already scalar, so swapping it with %d is not supported.",
               next_node->GetTypePtr(), next_node->GetNamePtr(), idx);
        return af::SUCCESS;  // 直接返回， is_supported 为 false
      }

      // step6: 可以交换顺序，但又不支持全部是Scalar，则不允许输入是相同节点（即输入都是Scalar），结束。
      if (ScheduleUtils::HasSameInput(next_node)) {
        GELOGD("Node %s(%s) has the same input, so swapping is not supported.", next_node->GetTypePtr(),
               next_node->GetNamePtr());
        return af::SUCCESS;
      }

      // step7: 交换两个输入
      GE_ASSERT_SUCCESS(ScheduleUtils::SwapInputIndex(next_node, kFirstInputIndex, kSecondInputIndex));
    }
  }
  // 所有节点都支持scalar输入，才算支持
  is_supported = true;
  return af::SUCCESS;
}

Status ScalarBroadcastOptimizationPass::RunPass(af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    GELOGD("Check node : %s %s, out size: %d, in size: %d", node->GetTypePtr(), node->GetNamePtr(),
           node->GetOutDataNodesSize(), node->GetInDataNodesSize());
    if (!IsOps<Broadcast>(node) || !ScheduleUtils::IsScalarBroadcastNode(node) ||
        !ascir::GetCvBroadcastShape(*node).empty()) {
      continue;
    }
    GELOGD("Find scalar Broadcast node [%s], start to check", node->GetNamePtr());

    // Find more continuous Broadcast nodes.
    std::vector<af::AscNodePtr> continues_brc_nodes;
    if (!ScheduleUtils::FindContinuesBroadcastNode(node, continues_brc_nodes) || continues_brc_nodes.empty()) {
      continues_brc_nodes.clear();
      continue;
    }
    const auto &last_brc_node = continues_brc_nodes[continues_brc_nodes.size() - 1UL];

    // Determine whether the next node supports scalar input ?
    bool is_next_node_supported_scalar = false;
    const auto &first_brc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    GE_ASSERT_SUCCESS(IsNextNodeSupportScalarInput(last_brc_node, is_next_node_supported_scalar, first_brc_node));
    if (!is_next_node_supported_scalar) {
      continues_brc_nodes.clear();
      continue;
    }

    // Relink first Broadcast input to last Broadcast input
    GE_CHECK_NOTNULL(node->GetInDataAnchor(0));
    auto pre_node_out_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(pre_node_out_anchor);
    for (const auto &out_anchor : last_brc_node->GetAllOutDataAnchors()) {
      GE_CHECK_NOTNULL(out_anchor);
      for (const auto &next_in_anchor : out_anchor->GetPeerInDataAnchors()) {
        GE_CHECK_NOTNULL(next_in_anchor);
        GE_CHECK_NOTNULL(next_in_anchor->GetOwnerNode());
        const auto &next_node = std::dynamic_pointer_cast<af::AscNode>(next_in_anchor->GetOwnerNode());
        next_node->inputs[next_in_anchor->GetIdx()].attr = node->inputs[0].attr;
        GE_ASSERT_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(out_anchor, next_in_anchor, pre_node_out_anchor));
      }
    }

    // Remove all broadcast nodes.
    for (const auto &continues_brc_node : continues_brc_nodes) {
      GELOGD("Scalar Broadcast node [%s] can be optimized, remove it.", continues_brc_node->GetNamePtr());
      af::NodeUtils::UnlinkAll(*continues_brc_node);
      GE_CHECK_NOTNULL(continues_brc_node->GetOwnerComputeGraph());
      af::GraphUtils::RemoveNodeWithoutRelink(continues_brc_node->GetOwnerComputeGraph(), continues_brc_node);
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
