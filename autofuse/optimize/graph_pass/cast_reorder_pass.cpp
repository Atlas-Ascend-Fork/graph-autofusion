/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_reorder_pass.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace {
bool IsLowPrecisionFloat(af::DataType dtype) {
  return (dtype == ge::DT_FLOAT16) || (dtype == ge::DT_BF16);
}

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

af::Status UpdateInputDesc(const af::AscNodePtr &node, const af::OutDataAnchorPtr &src_anchor) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(src_anchor);
  const auto src_node = src_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(src_node);
  const auto src_op_desc = src_node->GetOpDesc();
  const auto node_op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(src_op_desc);
  GE_ASSERT_NOTNULL(node_op_desc);
  GE_ASSERT_GRAPH_SUCCESS(
      node_op_desc->UpdateInputDesc(0, src_op_desc->GetOutputDesc(static_cast<uint32_t>(src_anchor->GetIdx()))));
  return af::SUCCESS;
}

af::Status SwapAdjacentNodes(const af::AscNodePtr &first_node, const af::AscNodePtr &second_node) {
  const auto first_in_anchor = first_node->GetInDataAnchor(0);
  const auto first_out_anchor = first_node->GetOutDataAnchor(0);
  const auto second_in_anchor = second_node->GetInDataAnchor(0);
  const auto second_out_anchor = second_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(first_in_anchor);
  GE_ASSERT_NOTNULL(first_out_anchor);
  GE_ASSERT_NOTNULL(second_in_anchor);
  GE_ASSERT_NOTNULL(second_out_anchor);

  const auto src_anchor = first_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  const auto peer_in_anchors = second_out_anchor->GetPeerInDataAnchors();
  GE_ASSERT_TRUE(peer_in_anchors.size() == 1U);
  const auto dst_anchor = peer_in_anchors.at(0);
  GE_ASSERT_NOTNULL(dst_anchor);

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(src_anchor, first_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(first_out_anchor, second_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(second_out_anchor, dst_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(src_anchor, second_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(second_out_anchor, first_in_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(first_out_anchor, dst_anchor));

  GE_ASSERT_SUCCESS(UpdateInputDesc(second_node, src_anchor));
  GE_ASSERT_SUCCESS(UpdateInputDesc(first_node, second_out_anchor));
  return af::SUCCESS;
}

bool IsLoadCastTransposePattern(const af::AscNodePtr &cast_node, const af::AscNodePtr &input_node,
                                const af::AscNodePtr &output_node) {
  if (!af::ops::IsOps<af::ascir_op::Load>(input_node) || !af::ops::IsOps<af::ascir_op::Transpose>(output_node)) {
    return false;
  }
  const af::DataType src_dtype = cast_node->inputs[0].attr.dtype;
  const af::DataType dst_dtype = cast_node->outputs[0].attr.dtype;
  return IsLowPrecisionFloat(src_dtype) && (dst_dtype == ge::DT_FLOAT) &&
         ScheduleUtils::IsNodeSupportDataType<af::ascir_op::Transpose>(src_dtype);
}

bool IsTransposeCastStorePattern(const af::AscNodePtr &cast_node, const af::AscNodePtr &input_node,
                                 const af::AscNodePtr &output_node) {
  if (!af::ops::IsOps<af::ascir_op::Transpose>(input_node) || !af::ops::IsOps<af::ascir_op::Store>(output_node)) {
    return false;
  }
  const af::DataType src_dtype = cast_node->inputs[0].attr.dtype;
  const af::DataType dst_dtype = cast_node->outputs[0].attr.dtype;
  return (src_dtype == ge::DT_FLOAT) && IsLowPrecisionFloat(dst_dtype) &&
         ScheduleUtils::IsNodeSupportDataType<af::ascir_op::Transpose>(dst_dtype);
}

af::Status ReorderLoadCastTranspose(const af::AscNodePtr &cast_node, const af::AscNodePtr &transpose_node) {
  const af::DataType low_dtype = cast_node->inputs[0].attr.dtype;
  const af::DataType high_dtype = cast_node->outputs[0].attr.dtype;

  cast_node->outputs[0].attr = transpose_node->outputs[0].attr;
  cast_node->outputs[0].attr.dtype = high_dtype;
  transpose_node->outputs[0].attr.dtype = low_dtype;
  cast_node->attr.sched = transpose_node->attr.sched;

  GE_ASSERT_SUCCESS(SwapAdjacentNodes(cast_node, transpose_node));
  GELOGI("Reordered Load->Cast->Transpose to Load->Transpose->Cast, cast node: %s, transpose node: %s.",
         cast_node->GetNamePtr(), transpose_node->GetNamePtr());
  return af::SUCCESS;
}

af::Status ReorderTransposeCastStore(const af::AscNodePtr &cast_node, const af::AscNodePtr &transpose_node) {
  const af::DataType low_dtype = cast_node->outputs[0].attr.dtype;
  const auto transpose_in_anchor = transpose_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(transpose_in_anchor);
  const auto src_anchor = transpose_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  const auto src_node = std::dynamic_pointer_cast<af::AscNode>(src_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(src_node);

  cast_node->outputs[0].attr = src_node->outputs[static_cast<uint32_t>(src_anchor->GetIdx())].attr;
  cast_node->outputs[0].attr.dtype = low_dtype;
  transpose_node->outputs[0].attr.dtype = low_dtype;
  cast_node->attr.sched = transpose_node->attr.sched;

  GE_ASSERT_SUCCESS(SwapAdjacentNodes(transpose_node, cast_node));
  GELOGI("Reordered Transpose->Cast->Store to Cast->Transpose->Store, cast node: %s, transpose node: %s.",
         cast_node->GetNamePtr(), transpose_node->GetNamePtr());
  return af::SUCCESS;
}
}  // namespace

af::Status SwapCastWithTranspose(const af::AscGraph &graph, bool &is_changed) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Cast>(node) || !HasSingleOutputReference(node) || HasControlEdge(node)) {
      continue;
    }

    const auto input_node = GetInputNode(node);
    const auto output_node = GetOutputNode(node);
    if ((input_node == nullptr) || (output_node == nullptr)) {
      continue;
    }

    if (IsLoadCastTransposePattern(node, input_node, output_node)) {
      if (!HasSingleOutputReference(output_node) || HasControlEdge(output_node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(ReorderLoadCastTranspose(node, output_node));
      is_changed = true;
      continue;
    }

    if (IsTransposeCastStorePattern(node, input_node, output_node)) {
      if (!HasSingleOutputReference(input_node) || HasControlEdge(input_node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(ReorderTransposeCastStore(node, input_node));
      is_changed = true;
    }
  }
  return af::SUCCESS;
}

af::Status SwapCastWithPrecisionAgnosticOpsPass::Run(af::AscGraph &graph) {
  bool is_changed = false;
  GE_CHK_STATUS_RET(SwapCastWithTranspose(graph, is_changed), "Failed to reorder Cast and Transpose");
  if (is_changed) {
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }
  return af::SUCCESS;
}
}  // namespace optimize
