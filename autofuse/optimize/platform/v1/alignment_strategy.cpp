/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/common/base_alignment_strategy.h"
#include "alignment_strategy.h"
#include "common_utils.h"
#include "ascir_utils.h"
#include "graph/utils/graph_utils.h"
#include "ascir_ops_utils.h"
#include "symbolizer/symbolic_utils.h"

namespace optimize {

AlignmentType AlignmentStrategy::GetDefaultAlignmentType() {
  GELOGD("GetDefaultAlignmentType: kAligned");
  return AlignmentType::kAligned;
}

// brc轴在尾轴,则输入不需要对齐
// 输出为轴为0理论上可以向上一根轴对齐,暂时不考虑这一策略
// 尾轴离散,输出也是对齐的?
af::Status AlignmentStrategy::BroadcastAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  const auto &input_attr = node->inputs[0].attr;
  const auto iter = tensor_to_align_type_.find(&input_attr);
  // 输入是scalar,可以不需要对齐
  if (iter == tensor_to_align_type_.end()) {
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(input_attr.repeats.size() == output_attr.repeats.size());
  GE_ASSERT_TRUE(output_attr.axis.size() == output_attr.repeats.size());
  tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
  bool found_brc_axis = false;
  for (int64_t axis_id : output_attr.vectorized_axis) {
    auto it = std::find(output_attr.axis.begin(), output_attr.axis.end(), axis_id);
    GE_ASSERT_TRUE(it != output_attr.axis.end());
    const size_t distance = std::distance(output_attr.axis.begin(), it);
    if (found_brc_axis) {
      if (af::SymbolicUtils::StaticCheckEq(input_attr.repeats[distance], af::sym::kSymbolOne) != af::TriBool::kTrue &&
          af::SymbolicUtils::StaticCheckEq(output_attr.repeats[distance], af::sym::kSymbolOne) != af::TriBool::kTrue) {
        GELOGD("Brc node[%s]'s input will be set aligned for non_tail axis brc, output align_type [%s].",
               node->GetNamePtr(), AlignmentTypeToString(tensor_to_align_type_[&output_attr].align_type));
        if (iter->second.align_type == AlignmentType::kNotAligned ||
            iter->second.align_type == AlignmentType::kFixedNotAligned) {
          return BackPropagateAlignment(node);
        }
        return af::SUCCESS;
      }
    } else if (af::SymbolicUtils::StaticCheckEq(input_attr.repeats[distance], af::sym::kSymbolOne) ==
                   af::TriBool::kTrue &&
               af::SymbolicUtils::StaticCheckEq(output_attr.repeats[distance], af::sym::kSymbolOne) !=
                   af::TriBool::kTrue) {
      found_brc_axis = true;
    }
  }
  if (!found_brc_axis) {
    tensor_to_align_type_[&output_attr] = iter->second;
  }
  GELOGD("Broadcast node[%s] output align_type set to [%s].", node->GetNamePtr(),
         AlignmentTypeToString(tensor_to_align_type_[&output_attr].align_type));
  return af::SUCCESS;
}

af::Status AlignmentStrategy::ConcatAlignmentInferFunc(const af::AscNodePtr &node) {
  // 小尾轴场景输入不需要对齐
  bool output_need_align = false;
  if (ascir::utils::UseSmallTailConcatApi(*node, &output_need_align)) {
    // TTODO 消除Brc后可能会导致repeat不连续，先用attr规避，后续整改
    (void)af::AttrUtils::SetBool(node->GetOpDesc(), "_concat_small_tail", true);
    const auto &output_attr = node->outputs[0].attr;
    GE_ASSERT_TRUE(!output_attr.strides.empty());
    auto align_type = output_need_align ? AlignmentType::kAligned : AlignmentType::kNotAligned;
    tensor_to_align_type_[&output_attr] = {align_type};
    GELOGD("Concat node[%s] small tail align_type set to [%s].", node->GetNamePtr(), AlignmentTypeToString(align_type));
  } else {
    GE_ASSERT_SUCCESS(DefaultAlignmentInferFunc(node));
  }
  return af::SUCCESS;
}

af::Status AlignmentStrategy::EleWiseAlignmentInferFunc(const af::AscNodePtr &node) {
  if (af::ops::IsOps<af::ascir_op::RemovePad>(node)) {
    tensor_to_align_type_[&node->outputs[0].attr] = {AlignmentType::kFixedNotAligned};
    return af::SUCCESS;
  }

  bool has_aligned_input = false;
  bool has_fix_unaligned = false;
  auto out_type = AlignmentType::kNotAligned;
  for (const auto &input : node->inputs()) {
    // A scalar has no row layout to impose on the tensor operand.
    if (ascgen_utils::IsScalarInput(input->attr.repeats)) {
      GELOGD("Skip scalar input alignment for EleWise node[%s].", node->GetNamePtr());
      continue;
    }
    auto alignment_iter = tensor_to_align_type_.find(&input->attr);
    if (alignment_iter == tensor_to_align_type_.end()) {
      continue;
    }

    const AlignmentType input_alignment = alignment_iter->second.align_type;
    if (input_alignment == AlignmentType::kAligned || input_alignment == AlignmentType::kDiscontinuous) {
      has_aligned_input = true;
      out_type = std::max(out_type, input_alignment);
    } else if (input_alignment == AlignmentType::kFixedNotAligned) {
      has_fix_unaligned = true;
    }
  }

  if (has_aligned_input) {
    for (const auto &output : node->outputs()) {
      tensor_to_align_type_[&output->attr] = {out_type};
    }
    GELOGD("EleWise node[%s] output set to [%s] from aligned input.", node->GetNamePtr(),
           AlignmentTypeToString(out_type));
    return BackPropagateAlignment(node, out_type);
  }

  if (has_fix_unaligned) {
    // 反响传递fix
    out_type = AlignmentType::kFixedNotAligned;
    GE_ASSERT_SUCCESS(BackPropagateFixUnAlignType(node));
  }

  for (const auto &output : node->outputs()) {
    tensor_to_align_type_[&output->attr] = {out_type};
  }

  return af::SUCCESS;
}

af::Status AlignmentStrategy::LoadAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  // 尾轴非连续,引入尾轴离散
  if (ScheduleUtils::IsNeedDiscontinuousAligned(output_attr)) {
    GELOGD("Node[%s] is last axis discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kDiscontinuous};
  } else if (!ScheduleUtils::IsVectorizedAxisContinuousInGM(output_attr) || IsLoadNeedAlignForReduce(node)) {
    // vectorized_axis 在gm上非连续,需要尾轴对齐搬运
    GELOGD("Node[%s] is discontinuous loading, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
  } else {
    GELOGD("Node[%s] is continuous loading, input tensor does not need to be aligned.", node->GetNamePtr());
    // vectorized_axis连续则可以连续搬运
    tensor_to_align_type_[&output_attr] = {AlignmentType::kNotAligned};
  }
  return af::SUCCESS;
}

af::Status AlignmentStrategy::StoreAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto &output_attr = node->outputs[0].attr;
  AlignmentType input_align = tensor_to_align_type_[&node->inputs[0].attr].align_type;
  tensor_to_align_type_[&output_attr] = {input_align};
  if (ScheduleUtils::IsNeedDiscontinuousAligned(output_attr)) {
    GELOGD("Node[%s] is last axis discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kDiscontinuous};
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node, AlignmentType::kDiscontinuous));
  } else if (!ScheduleUtils::IsVectorizedAxisContinuousInGM(output_attr) &&
             (input_align == AlignmentType::kNotAligned || input_align == AlignmentType::kFixedNotAligned)) {
    GELOGD("Node[%s] is discontinuous writing, input tensor needs to be aligned.", node->GetNamePtr());
    tensor_to_align_type_[&output_attr] = {AlignmentType::kAligned};
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node));
  }
  return af::SUCCESS;
}

af::Status AlignmentStrategy::DefaultAlignmentInferFunc(const af::AscNodePtr &node) {
  const auto default_align_type = GetDefaultAlignmentType();
  for (const auto &output : node->outputs()) {
    tensor_to_align_type_[&output->attr] = {default_align_type};
  }
  GELOGD("Default align: node[%s] output set to [%s].", node->GetNamePtr(), AlignmentTypeToString(default_align_type));

  bool has_mismatched_inputs = false;
  for (const auto &input : node->inputs()) {
    auto alignment_iter = tensor_to_align_type_.find(&input->attr);
    if (alignment_iter == tensor_to_align_type_.end()) {
      continue;
    }

    const AlignmentType input_alignment = alignment_iter->second.align_type;
    if (input_alignment != default_align_type) {
      has_mismatched_inputs = true;
      break;
    }
  }
  if (has_mismatched_inputs) {
    GELOGD("Node [%s] will set aligned_type with [%s] for inputs.", node->GetNamePtr(),
           AlignmentTypeToString(default_align_type));
    GE_ASSERT_SUCCESS(BackPropagateAlignment(node));
  }

  return af::SUCCESS;
}

af::Status AlignmentStrategy::SetAlignWidth(const ascir::ImplGraph &impl_graph) {
  // 依据数据类型判断对齐到32B还是64B
  align_width_ = 32U;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    auto dtype = node->outputs[0].attr.dtype;
    if ((dtype == ge::DT_FLOAT) && node->attr.api.compute_type == af::ComputeType::kComputeTranspose) {
      align_width_ = 64U;
      break;
    }
  }
  GELOGD("[%s]'s align width is [%d].", impl_graph.GetName().c_str(), align_width_);
  return af::SUCCESS;
}
}  // namespace optimize
