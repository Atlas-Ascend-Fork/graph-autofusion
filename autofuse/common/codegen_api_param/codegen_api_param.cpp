/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_api_param.h"

#include <utility>
#include <sstream>

using namespace codegen;

namespace {
ge::Expression AlignUp(const ge::Expression &value, uint32_t alignment) {
  return af::sym::Ceiling(value / ge::Symbol(alignment)) * ge::Symbol(alignment);
}

std::vector<bool> BuildStrideZeroFlags(const std::vector<ge::Expression> &strides) {
  std::vector<bool> zero_flags;
  zero_flags.reserve(strides.size());
  for (const auto &stride : strides) {
    zero_flags.emplace_back(stride == af::sym::kSymbolZero);
  }
  return zero_flags;
}

ge::Expression ProductAxes(const std::vector<ge::Expression> &values, const std::vector<size_t> &indices) {
  ge::Expression product = ge::Symbol(1U);
  for (const auto index : indices) {
    product = product * values[index];
  }
  return product;
}

ge::Expression ProductExpressions(const std::vector<ge::Expression> &values) {
  ge::Expression product = ge::Symbol(1U);
  for (const auto &value : values) {
    product = product * value;
  }
  return product;
}

bool IsValidReduceSpecificInput(const ReduceSpecificParamBuildInput &input) {
  return !input.input_repeats.empty() && input.input_repeats.size() == input.input_strides.size() &&
         !input.output_strides.empty() && !input.output_dims.empty() && input.dtype_size != 0U &&
         input.dtype_size <= 32U && input.pattern != ReducePattern::kUnknown &&
         (!input.need_multi_reduce || input.merge_times.IsValid());
}
}  // namespace

bool codegen::IsReduceMergedZeroTail(bool src_stride_zero, bool dst_stride_zero) {
  return src_stride_zero && dst_stride_zero;
}

bool codegen::IsReduceMergedSameSide(bool current_dst_stride_zero, bool last_dst_stride_zero) {
  return !((current_dst_stride_zero && !last_dst_stride_zero) || (!current_dst_stride_zero && last_dst_stride_zero));
}

ReduceMergedAxisAction codegen::UpdateReduceMergedAxisState(bool src_stride_zero, bool dst_stride_zero,
                                                            bool last_not_one_dst_stride_zero, bool is_last_axis,
                                                            size_t axis_index, ReduceMergedAxisState &state) {
  state.is_all_axis_reduce = state.is_all_axis_reduce && dst_stride_zero;
  if (is_last_axis) {
    state.use_zero_stride = !state.is_first && state.last_non_zero_stride_index == static_cast<size_t>(-1);
    if (state.is_first && !state.is_all_axis_reduce) {
      return ReduceMergedAxisAction::kAlignLast;
    }
    if (state.is_first && state.is_all_axis_reduce) {
      return ReduceMergedAxisAction::kAlignFirst;
    }
    return ReduceMergedAxisAction::kLastStride;
  }
  if (IsReduceMergedZeroTail(src_stride_zero, dst_stride_zero)) {
    return ReduceMergedAxisAction::kSkip;
  }
  if (state.is_first && state.last_not_one_axis_index != static_cast<size_t>(-1)) {
    state.is_first = IsReduceMergedSameSide(dst_stride_zero, last_not_one_dst_stride_zero);
  }
  if (!state.is_first) {
    if (!src_stride_zero) {
      state.last_non_zero_stride_index = axis_index;
    }
    return ReduceMergedAxisAction::kLastAxis;
  }
  state.last_not_one_axis_index = axis_index;
  return ReduceMergedAxisAction::kFirstAxis;
}

ReduceMergedAxisPlan codegen::BuildReduceMergedAxisPlan(const std::vector<bool> &src_stride_zero,
                                                        const std::vector<bool> &dst_stride_zero) {
  ReduceMergedAxisPlan plan;
  if (src_stride_zero.empty() || src_stride_zero.size() != dst_stride_zero.size()) {
    return plan;
  }
  size_t num_axes = src_stride_zero.size();
  for (; num_axes > 0U; --num_axes) {
    if (!IsReduceMergedZeroTail(src_stride_zero[num_axes - 1U], dst_stride_zero[num_axes - 1U])) {
      break;
    }
  }
  if (num_axes == 0U) {
    return plan;
  }
  plan.valid = true;
  ReduceMergedAxisState state;
  for (size_t i = 0U; i < num_axes; ++i) {
    const bool last_not_one_dst_stride_zero = state.last_not_one_axis_index == static_cast<size_t>(-1)
                                                  ? false
                                                  : dst_stride_zero[state.last_not_one_axis_index];
    const auto action = UpdateReduceMergedAxisState(src_stride_zero[i], dst_stride_zero[i],
                                                    last_not_one_dst_stride_zero, i == num_axes - 1U, i, state);
    if (action == ReduceMergedAxisAction::kAlignFirst || action == ReduceMergedAxisAction::kAlignLast) {
      plan.align_last_axis = true;
      plan.aligned_axis_index = i;
      break;
    }
    if (action == ReduceMergedAxisAction::kLastStride) {
      plan.use_last_non_zero_stride = true;
      plan.use_zero_stride = state.use_zero_stride;
      plan.last_non_zero_stride_index = state.last_non_zero_stride_index;
      break;
    }
    if (action == ReduceMergedAxisAction::kFirstAxis) {
      plan.first_axis_indices.emplace_back(i);
    } else if (action == ReduceMergedAxisAction::kLastAxis) {
      plan.last_axis_indices.emplace_back(i);
    }
  }
  plan.is_all_axis_reduce = state.is_all_axis_reduce;
  return plan;
}

ReduceMergedShape codegen::BuildReduceMergedShape(const std::vector<ge::Expression> &src_repeats,
                                                  const std::vector<ge::Expression> &src_strides,
                                                  const std::vector<ge::Expression> &dst_strides, uint32_t dtype_size) {
  ReduceMergedShape shape;
  if (src_repeats.empty() || src_repeats.size() != src_strides.size() || src_strides.size() != dst_strides.size() ||
      dtype_size == 0U || dtype_size > 32U) {
    return shape;
  }
  const auto plan = BuildReduceMergedAxisPlan(BuildStrideZeroFlags(src_strides), BuildStrideZeroFlags(dst_strides));
  if (!plan.valid) {
    return shape;
  }
  shape.first = ProductAxes(src_repeats, plan.first_axis_indices);
  shape.last = ProductAxes(src_repeats, plan.last_axis_indices);
  const uint32_t align_ele = 32U / dtype_size;
  if (plan.align_last_axis) {
    auto &target = plan.is_all_axis_reduce ? shape.first : shape.last;
    target = target * AlignUp(src_repeats[plan.aligned_axis_index], align_ele);
  }
  if (plan.use_last_non_zero_stride) {
    if (plan.use_zero_stride) {
      shape.last = af::sym::kSymbolZero;
    } else if (plan.last_non_zero_stride_index < src_strides.size()) {
      shape.last = shape.last * src_strides[plan.last_non_zero_stride_index];
    } else {
      return ReduceMergedShape{};
    }
  }
  if (plan.is_all_axis_reduce) {
    std::swap(shape.first, shape.last);
  }
  shape.valid = true;
  return shape;
}

af::Status codegen::BuildReduceSpecificParams(const ReduceSpecificParamBuildInput &input, ReduceSpecificParams &param) {
  param = ReduceSpecificParams{};
  GE_ASSERT_TRUE(IsValidReduceSpecificInput(input), "Invalid reduce specific param input, node[%s].",
                 input.node_name.c_str());

  param.valid = true;
  param.reduce_type = input.reduce_type;
  param.pattern = input.pattern;
  param.merge_mode = input.need_multi_reduce ? ReduceMergeMode::kCopy : ReduceMergeMode::kNone;
  param.merge_size = ProductExpressions(input.output_dims);
  param.merge_times = input.need_multi_reduce ? input.merge_times : ge::Symbol(1U);
  param.reuse = input.reuse;

  if (input.input_strides.size() == input.output_strides.size()) {
    const auto shape =
        BuildReduceMergedShape(input.input_repeats, input.input_strides, input.output_strides, input.dtype_size);
    param.merged_dims = {shape.valid, shape.first, shape.last};
  }
  return af::SUCCESS;
}

af::Status CodegenApiParam::Register(af::AscNodePtr node, CodegenApiParamPtr api_param) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kCodegenApiParam, api_param), "Graph:%s, Node:%s SetExtAttr failed",
                 node->GetOwnerComputeGraph()->GetName().c_str(), node->GetNamePtr());
  return af::SUCCESS;
}

CodegenApiParamPtr CodegenApiParam::TryGetNodeApiParam(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return nullptr;
  }
  CodegenApiParamPtr api_param = nullptr;
  api_param = node->GetOpDesc()->TryGetExtAttr(kCodegenApiParam, api_param);
  return api_param;
}

CodegenApiParamPtr CodegenApiParam::GetNodeApiParam(af::AscNodePtr node) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  CodegenApiParamPtr api_param = nullptr;
  api_param = op_desc->TryGetExtAttr(kCodegenApiParam, api_param);
  GE_ASSERT_NOTNULL(api_param, "Graph:%s, Node:%s api_param is null", node->GetOwnerComputeGraph()->GetName().c_str(),
                    node->GetNamePtr());
  return api_param;
}
