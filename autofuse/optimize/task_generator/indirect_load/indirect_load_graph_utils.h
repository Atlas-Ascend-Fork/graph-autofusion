/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GRAPH_UTILS_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GRAPH_UTILS_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// IndirectLoad 图遍历/小谓词/轴索引工具（自 indirect_load_schedule_case_generator.cpp 纯搬迁，inline 语义）。
inline bool HasControlEdge(const af::AscNodePtr &node) {
  return node->GetInControlNodesSize() != 0UL || node->GetOutControlNodesSize() != 0UL;
}

inline bool IsSingleConsumerWithoutControlEdge(const af::AscNodePtr &node) {
  return node != nullptr && node->GetOutDataNodesSize() == 1UL && !HasControlEdge(node);
}

inline bool IsBroadcastNode(const af::AscNodePtr &node) {
  return af::ops::IsOps<af::ascir_op::Broadcast>(node);
}

template <typename VisitFn>
void TraverseInputProducers(const NodePath &roots, NodeSet &visited, const VisitFn &visit) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second || !visit(node)) {
      continue;
    }
    for (size_t input_index = 0UL; input_index < node->inputs.Size(); ++input_index) {
      pending.emplace_back(ascgen_utils::indirect_load::GetInputProducer(node, input_index));
    }
  }
}

template <typename VisitFn>
af::Status TraverseOutputConsumers(const NodePath &roots, NodeSet &visited, const VisitFn &visit) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    GE_ASSERT_NOTNULL(node, "IndirectLoad output successor is invalid.");
    if (!visited.emplace(node.get()).second) {
      continue;
    }
    bool stop = false;
    GE_ASSERT_SUCCESS(visit(node, stop));
    if (stop) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  return af::SUCCESS;
}

inline af::AscNodePtr GetLoadTransposeSource(const af::AscNodePtr &node) {
  if (node == nullptr || !af::ops::IsOps<af::ascir_op::Transpose>(node)) {
    return nullptr;
  }
  const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
  return producer != nullptr && af::ops::IsOps<af::ascir_op::Load>(producer) ? producer : nullptr;
}

inline af::AscNodePtr GetFoldableLoadTransposeSource(const af::AscNodePtr &node) {
  const auto load = GetLoadTransposeSource(node);
  return IsSingleConsumerWithoutControlEdge(node) && IsSingleConsumerWithoutControlEdge(load) ? load : nullptr;
}

// Normalize the IndirectLoad axis to a non-negative output axis index. Return the sentinel for invalid metadata.
inline size_t GetIndirectLoadAxisIndex(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto outputs = node->outputs();
  if (outputs.empty() || outputs[0] == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  if (node->attr.ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const auto *ir_attr = node->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  if (ir_attr == nullptr) {
    return kIndirectLoadInvalidAxisIndex;
  }
  int64_t axis = 0L;
  if (ir_attr->GetAxis(axis) != af::SUCCESS) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const size_t rank = outputs[0]->attr.axis.size();
  if (rank > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return kIndirectLoadInvalidAxisIndex;
  }
  const int64_t rank_value = static_cast<int64_t>(rank);
  if (axis < -rank_value || axis >= rank_value) {
    return kIndirectLoadInvalidAxisIndex;
  }
  return static_cast<size_t>(axis < 0L ? axis + rank_value : axis);
}
}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GRAPH_UTILS_H
