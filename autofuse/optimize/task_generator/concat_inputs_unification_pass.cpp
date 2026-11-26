/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize/task_generator/concat_inputs_unification_pass.h"

#include "ascir_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph_utils.h"
#include "schedule_utils.h"
#include "buffer_allocate/tensor_mem_defs.h"

namespace optimize {
Status ConcatInputUnificationPass::Run(std::vector<ascir::ImplGraph> &graphs) {
  for (auto &graph : graphs) {
    GE_ASSERT_SUCCESS(RunOneGraph(graph));
  }
  return af::SUCCESS;
}

Status ConcatInputUnificationPass::RunOneGraph(ascir::ImplGraph &graph) {
  bool changed = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Concat>(node)) {
      std::set<int32_t> input_indices_need_copy;
      const auto need_optimize = NeedOptimize(node, input_indices_need_copy);
      GELOGD("graph: %s, node %s need optimize = %d", graph.GetName().c_str(), node->GetNamePtr(),
             static_cast<int32_t>(need_optimize));
      if (!need_optimize) {
        continue;
      }
      (void)af::AttrUtils::SetBool(node->GetOpDesc(), kAttrNameNoReuseInputs, true);
      if (!input_indices_need_copy.empty()) {
        GE_ASSERT_SUCCESS(DoOptimize(graph, node, input_indices_need_copy));
        changed = true;
      }
    }
  }
  if (changed) {
    ascir::utils::DumpGraph(graph, "AfterConcatInputUnificationPass");
  }
  return af::SUCCESS;
}

bool ConcatInputUnificationPass::CanOptimize(const af::AscNodePtr &concat_node, size_t concat_dim) {
  if (ascir::utils::AreConcatInputShapesEqual(concat_node) == af::TriBool::kFalse) {
    GELOGI("input col sizes of Concat differ, cannot optimize");
    return false;
  }
  std::set<int32_t> unused;
  return CanOptimize(concat_node, concat_dim, unused);
}

bool ConcatInputUnificationPass::CanOptimize(const af::AscNodePtr &concat_node, size_t concat_dim,
                                             std::set<int32_t> &input_indices_need_copy) {
  const auto dtype_size = ge::GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0, "unsupported output data type");
  if (IsSrcColSizeOverLimit(concat_node, concat_dim, dtype_size)) {
    LogSrcColSizeOverLimit(concat_node, concat_dim, dtype_size);
    return false;
  }

  input_indices_need_copy.clear();
  GE_WARN_ASSERT(GetQueInputIndices(concat_node, input_indices_need_copy) == af::SUCCESS);
  const auto load_num = static_cast<uint32_t>(input_indices_need_copy.size());
  if (load_num == concat_node->inputs.Size()) {
    GELOGI("all inputs are of compute type Load, can optimize");
    return true;
  }

  GE_ASSERT_SUCCESS(CollectSharedInputs(concat_node, input_indices_need_copy));
  const auto copy_num = static_cast<uint32_t>(input_indices_need_copy.size());
  constexpr uint32_t kCopyNumLimit = 3U;
  if (copy_num > kCopyNumLimit) {
    GELOGI("ub2ub num needed = %u, over limit = %u, cannot optimize", copy_num, kCopyNumLimit);
    return false;
  }
  return true;
}

bool ConcatInputUnificationPass::NeedOptimize(const af::AscNodePtr &concat_node,
                                              std::set<int32_t> &input_indices_need_copy) {
  GE_WARN_ASSERT(concat_node->inputs.Size() > 0);
  // 1. 输入shape相同
  if (ascir::utils::AreConcatInputShapesEqual(concat_node) == af::TriBool::kFalse) {
    GELOGI("input shapes of Concat differ, no need for optimization");
    return false;
  }

  // 2. 首轴concat不需要
  size_t concat_dim;
  bool is_first_dim = false;
  GE_CHK_STATUS_RET(ScheduleUtils::ResolveDiffDim(concat_node, concat_dim, is_first_dim), "ResolveConcatDim failed");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(is_first_dim, false, "concat on the first dim, no need for optimization");

  // 3. 输入对齐到4B不需要(Scatter在输入未对齐到4B时性能才会劣化)
  const auto dtype_size = ge::GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0, "unsupported output data type");
  GELOGI("input repeat = %s, output repeat = %s, concat_dim = %zu, dtype_size = %d",
         af::ToString(concat_node->inputs[0].attr.repeats).c_str(),
         af::ToString(concat_node->outputs[0].attr.repeats).c_str(), concat_dim, dtype_size);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(IsSrcColSizeAlignedToB4(concat_node, concat_dim, dtype_size), false,
                                 "src col size aligned to B32, no need for optimization");

  // 4. dst_col_size和ub2ub阈值检查
  if (!CanOptimize(concat_node, concat_dim, input_indices_need_copy)) {
    return false;
  }

  // 5. 输入全来自于Load不需要
  if (input_indices_need_copy.size() == concat_node->inputs.Size()) {
    GELOGI("All inputs are of compute type Load, no need for optimization");
    return false;
  }

  return true;
}

Status ConcatInputUnificationPass::DoOptimize(ascir::ImplGraph &graph, const af::AscNodePtr &concat_node,
                                              const std::set<int32_t> &input_indices_need_copy) {
  for (const auto &in_anchor : concat_node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor);
    const auto in_node = out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(in_node);
    const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(in_node);
    GE_ASSERT_NOTNULL(asc_node);
    if (input_indices_need_copy.find(in_anchor->GetIdx()) == input_indices_need_copy.cend()) {
      continue;
    }

    const std::string ub_name = asc_node->GetName() + "_ub_cpy_input_" + std::to_string(in_anchor->GetIdx());
    af::ascir_op::Ub2ub ub2ub(ub_name.c_str());
    af::AscNodePtr ub2ub_node = graph.AddNode(ub2ub);
    GE_ASSERT_NOTNULL(ub2ub_node);
    ub2ub_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
    ub2ub_node->attr.api.type = af::ApiType::kAPITypeCompute;
    ub2ub_node->attr.api.unit = af::ComputeUnit::kUnitVector;
    ub2ub_node->outputs[0].attr = asc_node->outputs[0].attr;
    ub2ub_node->outputs[0].attr.buf = {};
    ub2ub_node->outputs[0].attr.que = {};

    GE_ASSERT_SUCCESS(af::AscGraphUtils::InsertNodeAfter(out_anchor, {in_anchor}, ub2ub_node));
    GELOGD("Ub2ub node: %s added", ub2ub_node->GetNamePtr());
  }
  return af::SUCCESS;
}

af::Expression ConcatInputUnificationPass::GetColSize(const af::AscTensor &tensor, size_t concat_dim) {
  const auto &tensor_repeats = tensor.attr.repeats;
  af::Expression col_size = tensor_repeats[concat_dim];
  for (size_t i = concat_dim + 1; i < tensor_repeats.size(); ++i) {
    col_size = col_size * tensor_repeats[i];
  }
  return col_size;
}

af::Status ConcatInputUnificationPass::GetQueInputIndices(const af::AscNodePtr &concat_node,
                                                          std::set<int32_t> &input_indices_need_copy) {
  for (const auto &in_anchor : concat_node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor);
    const auto in_node = out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(in_node);
    const auto asc_node = dynamic_cast<af::AscNode *>(in_node);
    GE_ASSERT_NOTNULL(asc_node);
    if (asc_node->attr.api.compute_type == af::ComputeType::kComputeLoad) {
      input_indices_need_copy.emplace(in_anchor->GetIdx());
    }
  }
  return af::SUCCESS;
}

bool ConcatInputUnificationPass::IsSrcColSizeAlignedToB4(const af::AscNodePtr &concat_node, size_t concat_dim,
                                                         int32_t dtype_size) {
  const auto src_col_size_expr = GetColSize(concat_node->inputs[0], concat_dim);
  const auto aligned =
      (af::sym::Mod((src_col_size_expr * af::Symbol(dtype_size)), af::Symbol(sizeof(uint32_t))) == af::ops::Zero);
  return aligned;
}

bool ConcatInputUnificationPass::IsSrcColSizeOverLimit(const af::AscNodePtr &concat_node, size_t concat_dim,
                                                       int32_t dtype_size) {
  const auto src_col_size_expr = GetColSize(concat_node->inputs[0], concat_dim);
  if (!src_col_size_expr.IsConstExpr()) {
    return false;
  }
  int64_t src_col_size = -1;
  GE_WARN_ASSERT(src_col_size_expr.GetConstValue(src_col_size));
  constexpr int64_t kSrcColSizeLimit = (256 / 2);
  GELOGI("dst_col_size = %ld", src_col_size);
  return (src_col_size * dtype_size) > kSrcColSizeLimit;
}

void ConcatInputUnificationPass::LogSrcColSizeOverLimit(const af::AscNodePtr &concat_node, const size_t concat_dim,
                                                        const int32_t dtype_size) {
  const auto src_col_size_expr = GetColSize(concat_node->inputs[0], concat_dim);
  int64_t src_col_size = -1;
  if (!src_col_size_expr.IsConstExpr() || !src_col_size_expr.GetConstValue(src_col_size)) {
    GELOGI("dst col size is not a const expression over the limit, cannot optimize");
    return;
  }
  constexpr int64_t kSrcColSizeLimit = (256 / 2);
  GELOGI("dst col size = %ld, dst col bytes = %ld over limit %ld, cannot optimize", src_col_size,
         src_col_size * static_cast<int64_t>(dtype_size), kSrcColSizeLimit);
}

af::Status ConcatInputUnificationPass::CollectSharedInputs(const af::AscNodePtr &concat_node,
                                                           std::set<int32_t> &input_indices_need_copy) {
  std::map<const af::OutDataAnchor *, std::vector<int32_t>> src_anchor_to_input_indices;
  for (const auto &in_anchor : concat_node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor);
    src_anchor_to_input_indices[out_anchor.get()].emplace_back(in_anchor->GetIdx());
  }
  for (auto &[src_anchor, input_indices] : src_anchor_to_input_indices) {
    if (input_indices.size() > 1UL) {
      // 存在多引用
      auto *src_node = dynamic_cast<const af::AscNode *>(src_anchor->GetOwnerNodeBarePtr());
      GE_ASSERT_NOTNULL(src_node);
      // Load类型的会单独计算
      if (src_node->attr.api.compute_type != af::ComputeType::kComputeLoad) {
        GELOGD("src node = %s, output has multiple ref to concat, input indices = %s", src_node->GetName().c_str(),
               af::ToString(input_indices).c_str());
        input_indices_need_copy.insert(input_indices.cbegin() + 1, input_indices.cend());
      }
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
