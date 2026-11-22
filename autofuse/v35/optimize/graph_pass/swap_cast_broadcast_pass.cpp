/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swap_cast_broadcast_pass.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/types_af.h"
#include "graph/utils/graph_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace {
bool HasSingleOutputReference(const af::AscNodePtr &node) {
  const auto out_anchor = node->GetOutDataAnchor(0);
  return (out_anchor != nullptr) && (out_anchor->GetPeerInDataAnchors().size() == 1U);
}

bool HasControlEdge(const af::AscNodePtr &node) {
  const auto in_control_anchor = node->GetInControlAnchor();
  const auto out_control_anchor = node->GetOutControlAnchor();
  return ((in_control_anchor != nullptr) && !in_control_anchor->GetPeerOutControlAnchors().empty()) ||
         ((out_control_anchor != nullptr) && !out_control_anchor->GetPeerInControlAnchors().empty());
}

af::AscNodePtr GetInputNode(const af::AscNodePtr &node) {
  const auto in_anchor = node->GetInDataAnchor(0);
  if (in_anchor == nullptr) {
    return nullptr;
  }
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
}

af::AscNodePtr GetOutputNode(const af::AscNodePtr &node) {
  if (!HasSingleOutputReference(node)) {
    return nullptr;
  }
  const auto &peer_in_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  return std::dynamic_pointer_cast<af::AscNode>(peer_in_anchors.at(0)->GetOwnerNode());
}

// 判定 load -> cast(低byte向高byte转换) -> broadcast 模式:
// 仅窄化方向被本pass处理——低->高是NDDMA跨dtype继承stride会产生对齐不足的方向
// (详见头文件与SwapCastBrcAndGenNddma的关系说明); 高->低留在模板期交换无对齐
// 问题, 不在此提前交换。广播需支持cast输入侧dtype。
bool IsLoadCastBroadcastPattern(const af::AscNodePtr &cast_node, const af::AscNodePtr &input_node,
                                const af::AscNodePtr &output_node) {
  if (!af::ops::IsOps<af::ascir_op::Load>(input_node) || !af::ops::IsOps<af::ascir_op::Broadcast>(output_node)) {
    return false;
  }
  const af::DataType src_dtype = cast_node->inputs[0].attr.dtype;
  const af::DataType dst_dtype = cast_node->outputs[0].attr.dtype;
  if (af::GetSizeByDataType(src_dtype) >= af::GetSizeByDataType(dst_dtype)) {
    return false;
  }
  return ScheduleUtils::IsNodeSupportDataType<af::ascir_op::Broadcast>(src_dtype);
}

// 广播结果被IndirectLoad消费时不允许交换: 间接加载模板按load->cast->brc的链形状
// 识别index语义(SIMD/SIMT快速路径的地址计算协议, 见indirect_load_utils), 交换index
// 广播链会破坏其地址计算, 导致取数错误。
bool HasIndirectLoadConsumer(const af::AscNodePtr &brc_node) {
  for (const auto &out_node : brc_node->GetOutDataNodes()) {
    const auto asc_out_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if ((asc_out_node != nullptr) && af::ops::IsOps<af::ascir_op::IndirectLoad>(asc_out_node)) {
      return true;
    }
  }
  return false;
}

// 交换cast与broadcast: cast承接原广播输出的形状/轴/步长视图(dtype保持宽侧),
// 广播输出dtype改为窄侧,广播的其它消费者改接cast输出(值等价)。
af::Status SwapCastAndBroadcast(const af::AscNodePtr &load_node, const af::AscNodePtr &cast_node,
                                const af::AscNodePtr &brc_node) {
  const auto load_out_anchor = load_node->GetOutDataAnchor(0);
  const auto cast_out_anchor = cast_node->GetOutDataAnchor(0);
  const auto cast_in_anchor = cast_node->GetInDataAnchor(0);
  const auto brc_in_anchor = brc_node->GetInDataAnchor(0);
  const auto brc_out_anchor = brc_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(load_out_anchor);
  GE_ASSERT_NOTNULL(cast_out_anchor);
  GE_ASSERT_NOTNULL(cast_in_anchor);
  GE_ASSERT_NOTNULL(brc_in_anchor);
  GE_ASSERT_NOTNULL(brc_out_anchor);

  const af::DataType low_dtype = cast_node->inputs[0].attr.dtype;
  const af::DataType high_dtype = cast_node->outputs[0].attr.dtype;
  const auto brc_out_attr = brc_node->outputs[0].attr;
  cast_node->outputs[0].attr.axis = brc_out_attr.axis;
  cast_node->outputs[0].attr.repeats = brc_out_attr.repeats;
  cast_node->outputs[0].attr.strides = brc_out_attr.strides;
  cast_node->outputs[0].attr.vectorized_axis = brc_out_attr.vectorized_axis;
  cast_node->outputs[0].attr.vectorized_strides = brc_out_attr.vectorized_strides;
  cast_node->outputs[0].attr.dtype = high_dtype;
  brc_node->outputs[0].attr.dtype = low_dtype;
  cast_node->attr.sched = brc_node->attr.sched;

  // 将cast-->brc替换为load-->brc
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(cast_out_anchor, brc_in_anchor, load_out_anchor));
  // 将load-->cast替换为brc-->cast
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(load_out_anchor, cast_in_anchor, brc_out_anchor));
  // 将brc-->其它消费者替换为cast-->其它消费者
  const auto peer_in_anchors = brc_out_anchor->GetPeerInDataAnchors();
  for (const auto &peer_in_anchor : peer_in_anchors) {
    GE_CHECK_NOTNULL(peer_in_anchor);
    if (peer_in_anchor == cast_in_anchor) {
      continue;
    }
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(brc_out_anchor, peer_in_anchor, cast_out_anchor));
  }
  GELOGI("Reordered Load->Cast->Broadcast to Load->Broadcast->Cast, cast node: %s, broadcast node: %s.",
         cast_node->GetNamePtr(), brc_node->GetNamePtr());
  return af::SUCCESS;
}
}  // namespace

Status SwapCastBroadcastPass::RunPass(af::AscGraph &graph) {
  bool is_changed = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Cast>(node) || !HasSingleOutputReference(node) || HasControlEdge(node)) {
      continue;
    }
    const auto input_node = GetInputNode(node);
    const auto output_node = GetOutputNode(node);
    if ((input_node == nullptr) || (output_node == nullptr) || HasControlEdge(output_node)) {
      continue;
    }
    if (!IsLoadCastBroadcastPattern(node, input_node, output_node)) {
      continue;
    }
    if (HasIndirectLoadConsumer(output_node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(SwapCastAndBroadcast(input_node, node, output_node), "Failed to swap cast and broadcast.");
    is_changed = true;
  }
  if (is_changed) {
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return af::SUCCESS;
}
}  // namespace optimize
