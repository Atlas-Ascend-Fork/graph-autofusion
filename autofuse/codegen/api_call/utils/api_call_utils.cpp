/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common_utils.h"

using namespace af::ops;

namespace {
constexpr size_t kAxisIndex2 = 2U;
constexpr size_t kAxisIndex3 = 3U;
constexpr uint64_t kDmaMaxLen = 2U;
}  // namespace

namespace codegen {
namespace {
void SaveApiLoopAxisStatus(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                           const Tensor &base_tensor, int64_t vec_cur_idx, VectorizedAixsLoopStatus &axis_info,
                           VectorizedAxisLoopMergeStatus &merge_info, const TPipe &tpipe) {
  std::stringstream ss;
  const auto axis_pos = base_tensor.vectorized_axis_pos[vec_cur_idx];
  GetOneAxisSize(tpipe, base_tensor, vec_cur_idx, ss);
  merge_info.merge_repeats_str.emplace_back(ss.str());
  merge_info.merge_repeats.emplace_back(base_tensor.axis_size[axis_pos]);

  const std::vector<ascir::AxisId> merge_axis = {base_tensor.axis[axis_pos]};
  merge_info.merge_axis_ids.emplace_back(merge_axis);

  for (size_t i = 0; i < inputs.size(); i++) {
    merge_info.inputs_strides[i].emplace_back(inputs[i].vectorized_strides[vec_cur_idx]);
    axis_info.prev_input_axis_stride[i] = inputs[i].vectorized_strides[vec_cur_idx];
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    merge_info.outputs_strides[i].emplace_back(outputs[i].vectorized_strides[vec_cur_idx]);
    axis_info.prev_output_axis_stride[i] = outputs[i].vectorized_strides[vec_cur_idx];
  }
  axis_info.prev_repeat = base_tensor.axis_size[axis_pos];
}

const Tensor &GetBaseTensor(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {
  for (size_t i = 0UL; i < inputs.size(); i++) {
    const bool is_non_one = std::all_of(
        inputs[i].vectorized_axis_pos.begin(), inputs[i].vectorized_axis_pos.end(), [&inputs, &i](const uint32_t pos) {
          return !ascgen_utils::ExpressEq(inputs[i].axis_size[pos], af::sym::kSymbolOne);
        });
    if (is_non_one) {
      return inputs[i];
    }
  }

  for (size_t i = 0UL; i < outputs.size(); i++) {
    const bool is_all_zero = std::all_of(
        outputs[i].vectorized_strides.begin(), outputs[i].vectorized_strides.end(), [](const ascir::SizeExpr &stride) {
          return af::SymbolicUtils::StaticCheckEq(stride.Simplify(), af::sym::kSymbolZero) == af::TriBool::kTrue;
        });
    if (!is_all_zero) {
      return outputs[i];
    }
  }
  return outputs[0];
}
}  // namespace
static void SetDataCopyParams(const MergeInfo &merge_info, DataCopyParams &param, bool multi_axis_copy = false) {
  const auto &merge_repeats = merge_info.merge_repeats;
  const auto &merge_gm_strides = merge_info.merge_gm_strides;
  const auto &merge_ub_strides = merge_info.merge_ub_strides;
  param.repeats.assign(merge_repeats.begin(), merge_repeats.end());
  param.gm_strides.assign(merge_gm_strides.begin(), merge_gm_strides.end());
  param.ub_strides.assign(merge_ub_strides.begin(), merge_ub_strides.end());
  if (multi_axis_copy) {  // nddma场景尾轴stride可以不等于1或者0，这种情况不需要补轴
    return;
  }
  if (!param.repeats.empty() &&
      !ascgen_utils::NeedAppendDataCopyTailAxis(merge_gm_strides.back(), merge_ub_strides.back())) {
    // 对应的场景为: ub和gm的尾轴stride都等于1或者0，这种情况不需要补轴。
    return;
  }
  param.repeats.emplace_back(af::sym::kSymbolOne);
  param.gm_strides.emplace_back(af::sym::kSymbolOne);
  param.ub_strides.emplace_back(af::sym::kSymbolOne);
}

static void UpdateCalculatedDmaStatus(const Tensor &gm_tensor, const Tensor &ub_tensor, int64_t idx,
                                      int64_t vec_axis_pos, AxisInfo &axis_info, MergeInfo &merge_info) {
  merge_info.merge_repeats.emplace_back(gm_tensor.axis_size[idx]);
  merge_info.merge_gm_strides.emplace_back(gm_tensor.axis_strides[idx]);
  merge_info.merge_ub_strides.emplace_back(ub_tensor.vectorized_strides[vec_axis_pos]);
  axis_info.prev_axis_stride = gm_tensor.axis_strides[idx];
  axis_info.prev_vectorized_axis_stride = ub_tensor.vectorized_strides[vec_axis_pos];
  axis_info.prev_repeat = gm_tensor.axis_size[idx];
}

void CreateOuterFor(const TPipe &tpipe, const std::vector<ascir::SizeExpr> &outer_repeats, const std::stringstream &ss1,
                    std::stringstream &ss, size_t cur_idx) {
  if (cur_idx == outer_repeats.size()) {
    return;
  }
  ss << "for(int outer_for_" << cur_idx << " = 0; outer_for_" << cur_idx << " < "
     << tpipe.tiler.ActualSize(outer_repeats[cur_idx]) << "; outer_for_" << cur_idx << "++) {" << std::endl;
  CreateOuterFor(tpipe, outer_repeats, ss1, ss, cur_idx + 1);
  if (cur_idx == outer_repeats.size() - 1) {
    ss << ss1.str() << std::endl;
  }
  ss << "}" << std::endl;
}

bool CalculateDmaParams(const TPipe &tpipe, const Tensor &gm_tensor, const Tensor &ub_tensor, DataCopyParams &param,
                        bool multi_axis_copy) {
  if (gm_tensor.axis.size() < ub_tensor.vectorized_axis.size()) {
    return false;
  }
  AxisInfo axis_info;
  MergeInfo merge_info;
  size_t vec_axis_pos = ub_tensor.vectorized_axis.size() - 1;
  bool has_non_zero_axis = false;
  for (vec_axis_pos = ub_tensor.vectorized_axis.size(); vec_axis_pos-- > 0UL;) {
    const auto pos = std::find(gm_tensor.axis.begin(), gm_tensor.axis.end(), ub_tensor.vectorized_axis[vec_axis_pos]);
    GE_ASSERT_TRUE((pos != gm_tensor.axis.end()), "Codegen vectorized axis[%zu] not found", vec_axis_pos);
    const auto axis_pos = std::distance(gm_tensor.axis.begin(), pos);
    if (af::SymbolicUtils::StaticCheckEq(gm_tensor.axis_size[axis_pos], af::ops::One) == af::TriBool::kTrue) {
      continue;
    }
    // 如果当前轴gm和ub上对应的stride均为0，如果前序轴的stride不为1，则保留当前轴
    const bool ignore_zero_axis =
        ascgen_utils::ShouldIgnoreDataCopyZeroAxis(has_non_zero_axis, vec_axis_pos, ub_tensor.vectorized_strides);
    if (af::SymbolicUtils::StaticCheckEq(gm_tensor.axis_strides[axis_pos], af::ops::Zero) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(ub_tensor.vectorized_strides[vec_axis_pos], af::ops::Zero) ==
            af::TriBool::kTrue &&
        ignore_zero_axis) {
      continue;
    }
    has_non_zero_axis = true;
    ascir::SizeExpr cur_axis_stride = axis_info.prev_axis_stride * axis_info.prev_repeat;
    ascir::SizeExpr cur_vectorized_axis_stride = axis_info.prev_vectorized_axis_stride * axis_info.prev_repeat;
    if (!ascgen_utils::IsDataCopyAxisContinuous(cur_axis_stride, cur_vectorized_axis_stride,
                                                gm_tensor.axis_strides[axis_pos],
                                                ub_tensor.vectorized_strides[vec_axis_pos]) ||
        merge_info.merge_repeats.empty() ||
        (vec_axis_pos < (ub_tensor.vectorized_axis.size() - 1) &&
         tpipe.tiler.GetAxis(ub_tensor.vectorized_axis[vec_axis_pos + 1]).type ==
             ascir::Axis::Type::kAxisTypeTileInner)) {
      UpdateCalculatedDmaStatus(gm_tensor, ub_tensor, axis_pos, vec_axis_pos, axis_info, merge_info);
      continue;
    }

    ascir::SizeExpr product = gm_tensor.axis_size[axis_pos] * merge_info.merge_repeats.back();
    axis_info.prev_repeat = product;
    merge_info.merge_repeats.pop_back();
    merge_info.merge_repeats.push_back(product);
  }
  std::reverse(merge_info.merge_repeats.begin(), merge_info.merge_repeats.end());
  std::reverse(merge_info.merge_gm_strides.begin(), merge_info.merge_gm_strides.end());
  std::reverse(merge_info.merge_ub_strides.begin(), merge_info.merge_ub_strides.end());
  GELOGD(
      "[Codegen NDDMA] effective view: raw_rank=%zu, effective_rank=%zu, repeats=[%s], gm_strides=[%s], "
      "ub_strides=[%s], vectorized_axis=[%s]",
      ub_tensor.vectorized_axis.size(), merge_info.merge_repeats.size(),
      ascgen_utils::VectorToStr(merge_info.merge_repeats).c_str(),
      ascgen_utils::VectorToStr(merge_info.merge_gm_strides).c_str(),
      ascgen_utils::VectorToStr(merge_info.merge_ub_strides).c_str(),
      ascgen_utils::VectorToStr(ub_tensor.vectorized_axis).c_str());
  SetDataCopyParams(merge_info, param, multi_axis_copy);
  return true;
}

void SetDmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, DmaParams &dma_param, bool copy_in,
                  bool need_swap) {
  const size_t total_len = data_copy_param.repeats.size();
  if (total_len <= kDmaMaxLen && need_swap) {
    GELOGI("Can't swap data copy outer_for.");
    return;
  }
  if (total_len == 1) {
    dma_param.block_count = "1";
    dma_param.block_len = tpipe.tiler.ActualSize(data_copy_param.repeats[0]);
    dma_param.src_stride = "0";
    dma_param.dst_stride = "0";
  } else if (total_len >= kDmaMaxLen) {
    int64_t block_count_idx = need_swap ? (total_len - kAxisIndex3) : (total_len - kAxisIndex2);
    // 双切分场景切分在尾轴时，尾块在ub内存在跳写，尾轴上一根轴的stride为尾轴整块向32B对齐的值，因此计算ub内的stride时
    // merge_ub_stride需要调用Size接口，merge_repeats需要调用ActualSize接口
    std::string src_gm_stride =
        tpipe.tiler.ActualSize(data_copy_param.gm_strides[block_count_idx] - data_copy_param.repeats[total_len - 1]);
    std::string src_ub_stride = tpipe.tiler.Size(data_copy_param.ub_strides[block_count_idx]) + " - " +
                                tpipe.tiler.ActualSize(data_copy_param.repeats[total_len - 1]);
    std::string src_stride = copy_in ? src_gm_stride : src_ub_stride;

    std::string dst_ub_stride = tpipe.tiler.Size(data_copy_param.ub_strides[block_count_idx]) + " - " +
                                tpipe.tiler.ActualSize(data_copy_param.repeats[total_len - 1]);
    std::string dst_gm_stride =
        tpipe.tiler.ActualSize(data_copy_param.gm_strides[block_count_idx] - data_copy_param.repeats[total_len - 1]);
    std::string dst_stride = copy_in ? dst_ub_stride : dst_gm_stride;

    dma_param.block_count = tpipe.tiler.ActualSize(data_copy_param.repeats[block_count_idx]);
    dma_param.block_len = tpipe.tiler.ActualSize(data_copy_param.repeats[total_len - 1]);
    dma_param.src_stride = src_stride;
    dma_param.dst_stride = dst_stride;
  }
}

void SetDmaParamsExpr(const TPipe &tpipe, const DataCopyParams &data_copy_param, DmaParamsExpr &dma_param, bool copy_in,
                      bool need_swap) {
  (void)tpipe;
  size_t total_len = data_copy_param.repeats.size();
  if (total_len <= kDmaMaxLen && need_swap) {
    GELOGI("Can't swap data copy outer_for.");
    return;
  }
  if (total_len == 1) {
    dma_param.block_count = CombinedExprFactory::Constant(1);
    dma_param.block_len = CombinedExpression(ExprItemFactory::ActualSize(data_copy_param.repeats[0]));
    dma_param.src_stride = CombinedExprFactory::Constant(0);
    dma_param.dst_stride = CombinedExprFactory::Constant(0);
  } else if (total_len >= kDmaMaxLen) {
    int64_t block_count_idx = need_swap ? (total_len - kAxisIndex3) : (total_len - kAxisIndex2);
    // 双切分场景切分在尾轴时，尾块在ub内存在跳写，尾轴上一根轴的stride为尾轴整块向32B对齐的值，因此计算ub内的stride时
    // merge_ub_stride需要调用Size接口，merge_repeats需要调用ActualSize接口

    // src_gm_stride = ActualSize(gm_strides[block_count_idx] - repeats[total_len - 1])
    CombinedExpression src_gm_stride = CombinedExprFactory::ActualSizeOfDiff(
        data_copy_param.gm_strides[block_count_idx], data_copy_param.repeats[total_len - 1]);
    // src_ub_stride = Size(ub_strides[block_count_idx]) - ActualSize(repeats[total_len - 1])
    CombinedExpression src_ub_stride = CombinedExprFactory::SizeMinusActualSize(
        data_copy_param.ub_strides[block_count_idx], data_copy_param.repeats[total_len - 1]);
    CombinedExpression src_stride = copy_in ? src_gm_stride : src_ub_stride;

    // dst_ub_stride = Size(ub_strides[block_count_idx]) - ActualSize(repeats[total_len - 1])
    CombinedExpression dst_ub_stride = CombinedExprFactory::SizeMinusActualSize(
        data_copy_param.ub_strides[block_count_idx], data_copy_param.repeats[total_len - 1]);
    // dst_gm_stride = ActualSize(gm_strides[block_count_idx] - repeats[total_len - 1])
    CombinedExpression dst_gm_stride = CombinedExprFactory::ActualSizeOfDiff(
        data_copy_param.gm_strides[block_count_idx], data_copy_param.repeats[total_len - 1]);
    CombinedExpression dst_stride = copy_in ? dst_ub_stride : dst_gm_stride;

    dma_param.block_count = CombinedExpression(ExprItemFactory::ActualSize(data_copy_param.repeats[block_count_idx]));
    dma_param.block_len = CombinedExpression(ExprItemFactory::ActualSize(data_copy_param.repeats[total_len - 1]));
    dma_param.src_stride = src_stride;
    dma_param.dst_stride = dst_stride;
  }
}

std::string CalcInnerOffset(const TPipe &tpipe, const std::vector<ascir::SizeExpr> &strides) {
  std::stringstream ss;
  for (size_t i = 0; i < strides.size(); i++) {
    ss << "outer_for_" << i << " * " << tpipe.tiler.Size(strides[i]);
    if (i != strides.size() - 1) {
      ss << " + ";
    }
  }
  return ss.str();
}

// CalcInnerOffset 表达式版本 - 输出到 CombinedExpression
// 输出结果：outer_for_0 * Size(stride0) + outer_for_1 * Size(stride1) + ...
CombinedExpression CalcInnerOffsetExpr(const std::vector<ascir::SizeExpr> &strides) {
  if (strides.empty()) {
    return CombinedExprFactory::Constant(0);
  }

  std::vector<ExpressionItem> items;
  std::vector<std::string> ops;
  for (size_t i = 0; i < strides.size(); i++) {
    items.emplace_back(ExprItemFactory::LoopVarTimesSize(strides[i], static_cast<int64_t>(i)));
    if (i > 0) {
      ops.emplace_back("+");
    }
  }
  return CombinedExpression(items, ops);
}

static void CreateDmaCallInner(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                               const DataCopyParams &param, const ascir::SizeExpr &offset, std::stringstream &ss,
                               bool copy_in, bool need_swap) {
  int64_t total_len = param.repeats.size();
  std::vector<ascir::SizeExpr> gm_stride(param.gm_strides.begin(), param.gm_strides.end() - kAxisIndex3);
  std::vector<ascir::SizeExpr> ub_stride(param.ub_strides.begin(), param.ub_strides.end() - kAxisIndex3);
  std::vector<ascir::SizeExpr> repeats(param.repeats.begin(), param.repeats.end() - kAxisIndex3);
  if (need_swap) {
    gm_stride.emplace_back(param.gm_strides[total_len - kDmaMaxLen]);
    ub_stride.emplace_back(param.ub_strides[total_len - kDmaMaxLen]);
    repeats.emplace_back(param.repeats[total_len - kDmaMaxLen]);
  } else {
    gm_stride.emplace_back(param.gm_strides[total_len - kAxisIndex3]);
    ub_stride.emplace_back(param.ub_strides[total_len - kAxisIndex3]);
    repeats.emplace_back(param.repeats[total_len - kAxisIndex3]);
  }
  std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
  std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
  DmaParams dma_param;
  SetDmaParams(tpipe, param, dma_param, copy_in, need_swap);
  std::stringstream ss1;
  std::stringstream ss2;
  if (copy_in) {
    ss1 << "DataCopyPadExtend(" << output << "[" << ub_inner_offset << "], " << input << "[" << gm_offset << " + "
        << tpipe.tiler.Size(offset) << " + " << gm_inner_offset << "], " << dma_param.block_count << ", "
        << dma_param.block_len << ", " << dma_param.src_stride << ", " << dma_param.dst_stride << ");" << std::endl;
  } else {
    ss1 << "DataCopyPadExtend(" << output << "[" << gm_offset << " + " << tpipe.tiler.Size(offset) << " + "
        << gm_inner_offset << "], " << input << "[" << ub_inner_offset << "], " << dma_param.block_count << ", "
        << dma_param.block_len << ", " << dma_param.src_stride << ", " << dma_param.dst_stride << ");" << std::endl;
  }
  CreateOuterFor(tpipe, repeats, ss1, ss2, 0);
  ascir::SizeExpr last_two_repeat = param.repeats[total_len - kDmaMaxLen];
  ascir::SizeExpr last_three_repeat = param.repeats[total_len - kAxisIndex3];
  if (need_swap) {
    ss << "if (" << tpipe.tiler.ActualSize(last_two_repeat) << " < " << tpipe.tiler.ActualSize(last_three_repeat)
       << " ) { " << std::endl
       << ss2.str() << "} else { " << std::endl;
  } else {
    ss << ss2.str() << "}" << std::endl;
  }
}

void CreateDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                   const DataCopyParams &param, const ascir::SizeExpr &offset, std::stringstream &ss, bool copy_in) {
  CreateDmaCallInner(tpipe, input, output, gm_offset, param, offset, ss, copy_in, true);
  CreateDmaCallInner(tpipe, input, output, gm_offset, param, offset, ss, copy_in, false);
}

void CreateComputeNodeOuterFor(const std::vector<std::string> &outer_repeats, const std::stringstream &ss1,
                               std::stringstream &ss, size_t cur_idx) {
  if (cur_idx == outer_repeats.size()) {
    return;
  }
  ss << "for(int outer_for_" << cur_idx << " = 0; outer_for_" << cur_idx << " < " << outer_repeats[cur_idx]
     << "; outer_for_" << cur_idx << "++) {" << std::endl;
  CreateComputeNodeOuterFor(outer_repeats, ss1, ss, cur_idx + 1);
  if (cur_idx == outer_repeats.size() - 1) {
    ss << ss1.str() << std::endl;
  }
  ss << "}" << std::endl;
}

bool CheckAxisContinuous(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                         VectorizedAixsLoopStatus &axis_info, int64_t index) {
  for (size_t i = 0; i < inputs.size(); i++) {
    ascir::SizeExpr cur_axis_stride = axis_info.prev_input_axis_stride[i] * axis_info.prev_repeat;
    if (af::SymbolicUtils::StaticCheckEq(cur_axis_stride, inputs[i].vectorized_strides[index]) != af::TriBool::kTrue) {
      return false;
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    ascir::SizeExpr cur_axis_stride = axis_info.prev_output_axis_stride[i] * axis_info.prev_repeat;
    if (af::SymbolicUtils::StaticCheckEq(cur_axis_stride, outputs[i].vectorized_strides[index]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

void GetOneAxisSize(const TPipe &tpipe, const Tensor &tensor, const uint32_t idx, std::stringstream &ss) {
  auto axis_pos = tensor.vectorized_axis_pos[idx];
  ascir::AxisId axis_id = tensor.axis[axis_pos];
  if (tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
      tensor.vectorized_axis[0] == axis_id) {
    ss << tpipe.tiler.ActualSize(tensor.axis_size[axis_pos]);
    return;
  }
  ss << tpipe.tiler.Size(tensor.axis_size[axis_pos]);
}

bool IsNeedTailExpansion(const VectorizedAxisLoopMergeStatus &merge_info) {
  for (size_t i = 0; i < merge_info.inputs_strides.size(); i++) {
    if (!(merge_info.inputs_strides[i].empty()) &&
        af::SymbolicUtils::StaticCheckEq(merge_info.inputs_strides[i].back(), af::sym::kSymbolOne) !=
            af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(merge_info.inputs_strides[i].back(), af::sym::kSymbolZero) !=
            af::TriBool::kTrue) {
      return true;
    }
  }
  for (size_t i = 0; i < merge_info.outputs_strides.size(); i++) {
    if (!(merge_info.outputs_strides[i].empty()) &&
        af::SymbolicUtils::StaticCheckEq(merge_info.outputs_strides[i].back(), af::sym::kSymbolOne) !=
            af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(merge_info.outputs_strides[i].back(), af::sym::kSymbolZero) !=
            af::TriBool::kTrue) {
      return true;
    }
  }
  return false;
}

void SaveApiLoopAxisParams(VectorizedAxisLoopMergeStatus &merge_info, ApiLoopParams &param) {
  if (IsNeedTailExpansion(merge_info)) {
    merge_info.merge_repeats_str.push_back("1");
    merge_info.merge_repeats.push_back(af::ops::One);
    for (size_t i = 0; i < merge_info.inputs_strides.size(); i++) {
      merge_info.inputs_strides[i].push_back(af::ops::Zero);
    }
    for (size_t i = 0; i < merge_info.outputs_strides.size(); i++) {
      merge_info.outputs_strides[i].push_back(af::ops::Zero);
    }
  }
  param.inputs_strides.resize(merge_info.inputs_strides.size());
  param.outputs_strides.resize(merge_info.outputs_strides.size());
  if (merge_info.merge_repeats.size() == 1) {
    param.cal_count = merge_info.merge_repeats[0];
  } else if (merge_info.merge_repeats.size() > 1) {
    int64_t total_len = merge_info.merge_repeats.size();
    param.cal_count = merge_info.merge_repeats[total_len - 1];
    if (merge_info.inputs_strides.size() > 0) {
      param.input_second_to_last_stride = merge_info.inputs_strides[0][total_len - 2];
    }
    if (merge_info.outputs_strides.size() > 0) {
      param.output_second_to_last_stride = merge_info.outputs_strides[0][total_len - 2];
    }

    param.outer_repeats.assign(merge_info.merge_repeats_str.begin(), merge_info.merge_repeats_str.end() - 1);
    for (size_t i = 0; i < param.inputs_strides.size(); i++) {
      param.inputs_strides[i].assign(merge_info.inputs_strides[i].begin(), merge_info.inputs_strides[i].end() - 1);
    }

    for (size_t i = 0; i < param.outputs_strides.size(); i++) {
      param.outputs_strides[i].assign(merge_info.outputs_strides[i].begin(), merge_info.outputs_strides[i].end() - 1);
    }
  }
}

bool ShouldIgnoreZeroAxis(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs, int64_t cur_index) {
  if (inputs.empty() && outputs.empty()) {
    return false;
  }

  bool inputs_all_zero = std::all_of(inputs.begin(), inputs.end(), [cur_index](const Tensor &input) {
    int64_t idx = cur_index - 1;
    if (idx < 0 || idx >= static_cast<int64_t>(input.vectorized_strides.size())) {
      return false;
    }
    const auto &stride = input.vectorized_strides[idx];
    return (af::SymbolicUtils::StaticCheckEq(stride, One) == af::TriBool::kTrue) ||
           (af::SymbolicUtils::StaticCheckEq(stride, Zero) == af::TriBool::kTrue);
  });

  bool outputs_all_zero = std::all_of(outputs.begin(), outputs.end(), [cur_index](const Tensor &output) {
    int64_t idx = cur_index - 1;
    if (idx < 0 || idx >= static_cast<int64_t>(output.vectorized_strides.size())) {
      return false;
    }
    const auto &stride = output.vectorized_strides[idx];
    return (af::SymbolicUtils::StaticCheckEq(stride, One) == af::TriBool::kTrue) ||
           (af::SymbolicUtils::StaticCheckEq(stride, Zero) == af::TriBool::kTrue);
  });

  return inputs_all_zero && outputs_all_zero;
}

bool IsInputOutputStrideAllZero(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                                int64_t cur_index) {
  if (inputs.empty() && outputs.empty()) {
    return false;
  }

  bool inputs_all_zero = std::all_of(inputs.begin(), inputs.end(), [cur_index](const Tensor &input) {
    if (cur_index < 0 || cur_index >= static_cast<int64_t>(input.vectorized_strides.size())) {
      return false;
    }
    const auto &stride = input.vectorized_strides[cur_index];
    return af::SymbolicUtils::StaticCheckEq(stride, Zero) == af::TriBool::kTrue;
  });

  bool outputs_all_zero = std::all_of(outputs.begin(), outputs.end(), [cur_index](const Tensor &output) {
    if (cur_index < 0 || cur_index >= static_cast<int64_t>(output.vectorized_strides.size())) {
      return false;
    }
    const auto &stride = output.vectorized_strides[cur_index];
    return af::SymbolicUtils::StaticCheckEq(stride, Zero) == af::TriBool::kTrue;
  });

  return inputs_all_zero && outputs_all_zero;
}

bool GenerateVectorizedAxisMergeStatus(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                                       VectorizedAxisLoopMergeStatus &merge_info, const TPipe &tpipe) {
  VectorizedAixsLoopStatus axis_info;
  for (const auto &input : inputs) {
    if (af::SymbolicUtils::StaticCheckEq(input.vectorized_strides.back(), Zero) == af::TriBool::kTrue) {
      axis_info.prev_input_axis_stride.push_back(af::ops::Zero);
    } else {
      axis_info.prev_input_axis_stride.push_back(af::ops::One);
    }
  }
  for (const auto &output : outputs) {
    if (af::SymbolicUtils::StaticCheckEq(output.vectorized_strides.back(), Zero) == af::TriBool::kTrue) {
      axis_info.prev_output_axis_stride.push_back(af::ops::Zero);
    } else {
      axis_info.prev_output_axis_stride.push_back(af::ops::One);
    }
  }
  merge_info.inputs_strides.resize(inputs.size());
  merge_info.outputs_strides.resize(outputs.size());

  const Tensor &base_tensor = GetBaseTensor(inputs, outputs);
  int64_t vec_cur_idx = base_tensor.vectorized_axis.size() - 1;
  bool has_non_zero_axis = false;
  while (vec_cur_idx >= 0) {
    bool ignore_zero_axis = has_non_zero_axis || vec_cur_idx == 0 || ShouldIgnoreZeroAxis(inputs, outputs, vec_cur_idx);
    if (ignore_zero_axis && IsInputOutputStrideAllZero(inputs, outputs, vec_cur_idx)) {
      vec_cur_idx--;
      continue;
    }
    has_non_zero_axis = true;
    if (!CheckAxisContinuous(inputs, outputs, axis_info, vec_cur_idx)) {
      SaveApiLoopAxisStatus(inputs, outputs, base_tensor, vec_cur_idx, axis_info, merge_info, tpipe);
      vec_cur_idx--;
      continue;
    }

    if (merge_info.merge_repeats.empty()) {
      SaveApiLoopAxisStatus(inputs, outputs, base_tensor, vec_cur_idx, axis_info, merge_info, tpipe);
      vec_cur_idx--;
      continue;
    }

    std::string product_str;
    std::stringstream ss;
    auto axis_pos = base_tensor.vectorized_axis_pos[vec_cur_idx];
    GetOneAxisSize(tpipe, base_tensor, vec_cur_idx, ss);
    product_str = ss.str() + " * " + merge_info.merge_repeats_str.back();
    merge_info.merge_repeats_str.pop_back();
    merge_info.merge_repeats_str.push_back(product_str);

    ascir::SizeExpr product = base_tensor.axis_size[axis_pos] * merge_info.merge_repeats.back();
    axis_info.prev_repeat = product;
    merge_info.merge_repeats.pop_back();
    merge_info.merge_repeats.push_back(product);
    merge_info.merge_axis_ids.back().emplace_back(base_tensor.axis[axis_pos]);

    for (size_t i = 0; i < inputs.size(); i++) {
      axis_info.prev_input_axis_stride[i] = inputs[i].vectorized_strides[vec_cur_idx];
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      axis_info.prev_output_axis_stride[i] = outputs[i].vectorized_strides[vec_cur_idx];
    }
    axis_info.prev_repeat = base_tensor.axis_size[axis_pos];
    vec_cur_idx--;
  }
  std::reverse(merge_info.merge_repeats.begin(), merge_info.merge_repeats.end());
  std::reverse(merge_info.merge_repeats_str.begin(), merge_info.merge_repeats_str.end());
  std::reverse(merge_info.merge_axis_ids.begin(), merge_info.merge_axis_ids.end());
  for (size_t i = 0; i < merge_info.merge_axis_ids.size(); i++) {
    std::reverse(merge_info.merge_axis_ids[i].begin(), merge_info.merge_axis_ids[i].end());
  }
  for (size_t i = 0; i < merge_info.inputs_strides.size(); i++) {
    std::reverse(merge_info.inputs_strides[i].begin(), merge_info.inputs_strides[i].end());
  }

  for (size_t i = 0; i < merge_info.outputs_strides.size(); i++) {
    std::reverse(merge_info.outputs_strides[i].begin(), merge_info.outputs_strides[i].end());
  }
  return true;
}

bool GetMaxDtypeSize(const ge::DataType input_data_type, const ge::DataType out_put_data_type,
                     std::string &dtype_size) {
  const int32_t input_dtype_size = GetSizeByDataType(input_data_type);
  GE_CHK_BOOL_RET_STATUS_NOLOG(input_dtype_size > 0, false);
  const int32_t output_dtype_size = GetSizeByDataType(out_put_data_type);
  GE_CHK_BOOL_RET_STATUS_NOLOG(output_dtype_size > 0, false);
  int32_t max_dtype_size;
  // DT_INT4类型的dtype size会超过kDataTypeSizeBitOffset，当DT_INT4与其他类型之间的转换时，取与它进行转换的dtype size
  if (input_dtype_size >= ge::kDataTypeSizeBitOffset || output_dtype_size >= ge::kDataTypeSizeBitOffset) {
    max_dtype_size = input_dtype_size >= ge::kDataTypeSizeBitOffset ? output_dtype_size : input_dtype_size;
  } else {
    max_dtype_size = std::max(input_dtype_size, output_dtype_size);
  }
  dtype_size = std::to_string(max_dtype_size);
  return true;
}

bool IsCVFusionStage(const ApiCallContext &context) {
  return context.stage != ComputeStage::kDefault;
}

Status GetTensorDtypeSize(const Tensor &tensor, int64_t &dtype_size) {
  const int32_t tensor_dtype_size = GetSizeByDataType(tensor.dtype);
  GE_CHK_BOOL_RET_STATUS(tensor_dtype_size > 0 && tensor_dtype_size < ge::kDataTypeSizeBitOffset, af::FAILED,
                         "get dtype size failed, tensor:%s, dtype:%d", tensor.name.c_str(),
                         static_cast<int32_t>(tensor.dtype));
  dtype_size = tensor_dtype_size;
  return af::SUCCESS;
}

std::string GenBlockAlignNExpr(const Tensor &tensor, const std::string &n_expr) {
  int64_t dtype_size = 0;
  if (GetTensorDtypeSize(tensor, dtype_size) != af::SUCCESS || dtype_size <= 0) {
    return n_expr;
  }
  const int64_t align_value = 32 / dtype_size;
  return "((" + n_expr + " + " + std::to_string(align_value) + " - 1) / " + std::to_string(align_value) + " * " +
         std::to_string(align_value) + ")";
}

std::string GetCVAlignedSize(const ApiCallContext &context, const Tensor &tensor, const std::string &size_expr) {
  return IsCVFusionStage(context) ? GenBlockAlignNExpr(tensor, size_expr) : size_expr;
}

void GenerateLinkStoreEventCode(const Tensor &ub, const std::string &offset_str, std::stringstream &ss) {
  std::hash<std::string> hasher;
  [[maybe_unused]] size_t hasher_value = hasher(offset_str);

  std::stringstream ss_event_id;
  std::stringstream ss_sync_flag_id;
  ss_event_id << ub << "_e_mte3_2_mte2_" << offset_str;
  ss_sync_flag_id << ub << "_s_mte3_2_mte2_" << offset_str;

  ss << "auto " << ss_event_id.str() << " = tpipe.AllocEventID<HardEvent::MTE3_MTE2>();" << std::endl;
  ss << "TQueSync<PIPE_MTE3, PIPE_MTE2> " << ss_sync_flag_id.str() << ";" << std::endl;
  ss << ss_sync_flag_id.str() << ".SetFlag(" << ss_event_id.str() << ");" << std::endl;
  ss << ss_sync_flag_id.str() << ".WaitFlag(" << ss_event_id.str() << ");" << std::endl;
  ss << "tpipe.ReleaseEventID<HardEvent::MTE3_MTE2>(" << ss_event_id.str() << ");" << std::endl;
}

}  // namespace codegen
