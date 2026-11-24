/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "broadcast_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "../utils/api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "codegen/expression_convert_struct.h"

namespace {
constexpr size_t kSingleAxisSize = 1U;
constexpr size_t kDoubleAxisSize = 2U;
constexpr size_t kAxisSizeThree = 3U;
constexpr size_t kAxisSizeFour = 4U;
constexpr size_t kAxisIndex0 = 0U;
constexpr size_t kAxisIndex1 = 1U;
constexpr size_t kAxisIndex2 = 2U;
constexpr size_t kAxisIndex3 = 3U;
}  // namespace
namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status DimensionCollapse(const Tensor &input, const Tensor &output,
                         std::vector<std::pair<bool, std::vector<uint32_t>>> &result, uint32_t &broadcast_num) {
  if (input.vectorized_axis.size() != output.vectorized_axis.size()) {
    GELOGE(af::FAILED, "Codegen broadcast input vec axis size[%zu] not equal output vec axis size[%zu]",
           input.vectorized_axis.size(), output.vectorized_axis.size());
    return af::FAILED;
  }

  std::vector<uint32_t> tmp;
  ascir::SizeExpr prev_input_repeat = Zero;
  ascir::SizeExpr prev_output_repeat = Zero;
  size_t pos = 0;

  for (; pos < input.vectorized_axis.size(); pos++) {
    ascir::SizeExpr input_stride = input.vectorized_strides[pos];
    ascir::SizeExpr output_stride = output.vectorized_strides[pos];
    if (af::SymbolicUtils::StaticCheckEq(input_stride, af::sym::kSymbolZero) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(output_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    prev_input_repeat = input.axis_size[input.vectorized_axis_pos[pos]];
    prev_output_repeat = output.axis_size[output.vectorized_axis_pos[pos]];
    break;
  }

  if (pos >= input.vectorized_axis.size()) {
    return af::FAILED;
  }

  tmp.push_back(pos);
  pos++;
  bool prev_status = af::SymbolicUtils::StaticCheckEq(prev_input_repeat, prev_output_repeat) != af::TriBool::kTrue;
  for (; pos < input.vectorized_axis.size(); pos++) {
    ascir::SizeExpr cur_input_stride = input.vectorized_strides[pos];
    ascir::SizeExpr cur_output_stride = output.vectorized_strides[pos];
    if (af::SymbolicUtils::StaticCheckEq(cur_input_stride, af::sym::kSymbolZero) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(cur_output_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    auto &cur_input_repeat = input.axis_size[input.vectorized_axis_pos[pos]];
    auto &cur_output_repeat = output.axis_size[output.vectorized_axis_pos[pos]];
    bool cur_status = af::SymbolicUtils::StaticCheckEq(cur_input_repeat, cur_output_repeat) != af::TriBool::kTrue;
    if (cur_status != prev_status) {
      broadcast_num = prev_status ? broadcast_num + 1 : broadcast_num;
      result.push_back({prev_status, tmp});
      tmp = {static_cast<uint32_t>(pos)};
      prev_status = cur_status;
    } else {
      tmp.push_back(pos);
    }
  }
  broadcast_num = prev_status ? broadcast_num + 1 : broadcast_num;
  result.push_back({prev_status, tmp});
  return af::SUCCESS;
}

// 对每个分组进行合并
static std::string GetFormerMergedSize(const TPipe &tpipe, const Tensor &tensor,
                                       const std::pair<bool, std::vector<uint32_t>> &merge_group,
                                       const bool &is_input) {
  bool is_brc_group = merge_group.first;
  if (is_brc_group && is_input) {
    return "1";
  }
  std::stringstream ss;
  for (size_t i = 0; i < merge_group.second.size(); i++) {
    GetOneAxisSize(tpipe, tensor, merge_group.second[i], ss);
    if (i != merge_group.second.size() - 1) {
      ss << " * ";
    }
  }
  return ss.str();
}

static std::string GetLatterMergedSize(const TPipe &tpipe, const Tensor &tensor,
                                       const std::vector<std::pair<bool, std::vector<uint32_t>>> &merge_groups,
                                       const bool &is_input) {
  bool is_brc_group = merge_groups.back().first;
  if (is_brc_group && is_input) {
    return "1";
  }
  std::vector<uint32_t> last_group = merge_groups.back().second;
  uint32_t last_group_size = last_group.size();
  uint32_t idx = 0;
  if (last_group_size == static_cast<uint32_t>(1)) {
    // 最后一个分组只有一根轴，此时需要向前一个分组去借轴
    if (merge_groups.size() == 1) {
      return "0";
    }
    idx = merge_groups[merge_groups.size() - kAxisIndex2].second.back();
  } else {
    idx = last_group[last_group_size - kAxisIndex2];
  }
  ascir::SizeExpr last_dim_size = tensor.vectorized_strides[idx];

  std::stringstream ss;
  uint32_t loop_extent = last_group_size - static_cast<uint32_t>(1);
  for (size_t i = 0; i < loop_extent; i++) {
    GetOneAxisSize(tpipe, tensor, last_group[i], ss);
    ss << " * ";
  }
  ss << tpipe.tiler.Size(last_dim_size);
  return ss.str();
}

static std::string BroadcastGetLastDimStride(const TPipe &tpipe, const Tensor &tensor,
                                             const std::vector<std::pair<bool, std::vector<uint32_t>>> &merge_groups) {
  auto &last_merge_group = merge_groups.back();
  bool is_brc_group = last_merge_group.first;
  if (!is_brc_group) {
    return "1";
  }
  if (merge_groups.size() <= 1) {
    return "1";
  }
  auto &last_former_merge_group = merge_groups[merge_groups.size() - kAxisIndex2];
  uint32_t idx = last_former_merge_group.second.back();
  ascir::SizeExpr last_dim_stride = tensor.vectorized_strides[idx];
  return tpipe.tiler.Size(last_dim_stride);
}

static void GetBroadcastSizeParameters(const TPipe &tpipe, const Tensor &tensor,
                                       const std::vector<std::pair<bool, std::vector<uint32_t>>> &merge_groups,
                                       const bool &is_input, std::vector<std::string> &repeat_sizes) {
  for (size_t i = 0; i < merge_groups.size(); i++) {
    if (i != merge_groups.size() - 1) {
      repeat_sizes[i] = GetFormerMergedSize(tpipe, tensor, merge_groups[i], is_input);
    } else {
      repeat_sizes[i] = GetLatterMergedSize(tpipe, tensor, merge_groups, is_input);
    }
  }
}

static void BroadcastAllCommonAxis(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                   const Tensor &input, const Tensor &output, std::string &result) {
  std::stringstream ss;
  std::string dtype_name;
  Tensor::DtypeName(output.dtype, dtype_name);
  ss << "DataCopy(" << output << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, output) << "], " << input
     << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, input) << "], " << KernelUtils::SizeAlign() << "("
     << output.actual_size << ", 32 / sizeof(" << dtype_name << "))"
     << ");" << std::endl;
  result = ss.str();
}

static void BroadcastOneAxis(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                             const Tensor &output, const int64_t tmp_buf_id,
                             const std::vector<std::pair<bool, std::vector<uint32_t>>> &merge_groups,
                             std::string &result) {
  std::vector<std::string> src_size = {"0", "0", "0"};
  std::vector<std::string> dst_size = {"0", "0", "0"};
  GetBroadcastSizeParameters(tpipe, output, merge_groups, true, src_size);
  GetBroadcastSizeParameters(tpipe, output, merge_groups, false, dst_size);
  std::string last_dim_stride = BroadcastGetLastDimStride(tpipe, input, merge_groups);
  std::stringstream ss;
  ss << "Broadcast(" << output << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, output) << "], " << input
     << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, input) << "], " << src_size[kAxisIndex0] << ", "
     << src_size[kAxisIndex1] << ", " << src_size[kAxisIndex2] << ", " << dst_size[kAxisIndex0] << ", "
     << dst_size[kAxisIndex1] << ", " << dst_size[kAxisIndex2] << ", " << tpipe.tmp_buf << "_"
     << std::to_string(tmp_buf_id) << ", " << last_dim_stride << ");" << std::endl;
  result = ss.str();
}

static void BroadcastTwoAxis(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                             const Tensor &output, const int64_t tmp_buf_id,
                             const std::vector<std::pair<bool, std::vector<uint32_t>>> &merge_groups,
                             std::string &result) {
  const auto vectorize_axis_size = merge_groups.size();
  std::vector<std::string> src_size(vectorize_axis_size, "0");
  std::vector<std::string> dst_size(vectorize_axis_size, "0");
  GetBroadcastSizeParameters(tpipe, output, merge_groups, true, src_size);
  GetBroadcastSizeParameters(tpipe, output, merge_groups, false, dst_size);
  std::string last_dim_stride = BroadcastGetLastDimStride(tpipe, input, merge_groups);
  if (vectorize_axis_size == kAxisSizeThree) {
    src_size.insert(src_size.begin(), "1");
    dst_size.insert(dst_size.begin(), "1");
  }
  std::stringstream ss;
  ss << "Broadcast(" << output << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, output) << "], " << input
     << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, input) << "], " << src_size[kAxisIndex0] << ", "
     << src_size[kAxisIndex1] << ", " << src_size[kAxisIndex2] << ", " << src_size[kAxisIndex3] << ", "
     << dst_size[kAxisIndex0] << ", " << dst_size[kAxisIndex1] << ", " << dst_size[kAxisIndex2] << ", "
     << dst_size[kAxisIndex3] << ", " << tpipe.tmp_buf << "_" << std::to_string(tmp_buf_id) << ", " << last_dim_stride
     << ");" << std::endl;
  result = ss.str();
}

bool IsBroadcastConstantTensor(const Tensor &tensor) {
  if (tensor.is_constant) {
    return true;
  }
  bool tensor_constant = true;
  for (size_t i = 0; i < tensor.vectorized_axis.size(); i++) {
    auto &src_repeat = tensor.axis_size[tensor.vectorized_axis_pos[i]];
    if (af::SymbolicUtils::StaticCheckEq(src_repeat, af::sym::kSymbolOne) != af::TriBool::kTrue) {
      tensor_constant = false;
      break;
    }
  }
  return tensor_constant;
}

void BroadcastScalar(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &in,
                     const Tensor &out, const int64_t tmp_buf_id, std::string &result, bool need_tmp_buf) {
  std::stringstream ss;
  std::string int64_tmp_buf;
  if ((in.dtype == ge::DT_INT64 || in.dtype == ge::DT_UINT64) && need_tmp_buf) {
    int64_tmp_buf = ", " + tpipe.tmp_buf.name + "_" + std::to_string(tmp_buf_id);
  }
  if (in.is_constant) {
    ss << "Duplicate(" << out << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, out) << "], " << in.name
       << ", " << out.actual_size << int64_tmp_buf << ");" << std::endl;
  } else if (in.is_ub_scalar) {
    ss << "Duplicate(" << out << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, out) << "], "
       << in.ub_scalar_name << ", " << out.actual_size << int64_tmp_buf << ");" << std::endl;
  } else {
    if (in.position == af::Position::kPositionVecIn) {
      std::string event_id = in.name + "_to_" + out.name + "_event_id";
      ss << "event_t " << event_id << " = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));"
         << std::endl;
      ss << "SetFlag<HardEvent::MTE2_S>(" << event_id << ");" << std::endl;
      ss << "WaitFlag<HardEvent::MTE2_S>(" << event_id << ");" << std::endl;
    } else if (in.position == af::Position::kPositionVecCalc) {
      std::string event_id = in.name + "_to_" + out.name + "_event_id";
      ss << "event_t " << event_id << " = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));"
         << std::endl;
      ss << "SetFlag<HardEvent::V_S>(" << event_id << ");" << std::endl;
      ss << "WaitFlag<HardEvent::V_S>(" << event_id << ");" << std::endl;
    }
    ss << "Duplicate(" << out << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, out) << "], " << in
       << ".GetValue(0), " << out.actual_size << int64_tmp_buf << ");" << std::endl;
  }
  result = ss.str();
}

Status BroadcastApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  const auto &x = inputs[0].get();
  const auto &y = outputs[0].get();
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, {}, tpipe.tmp_buf.name + "_" + std::to_string(id));

  // 处理scalar broadcast场景
  if (IsBroadcastConstantTensor(x)) {
    GE_ASSERT_TRUE(id != -1L, "BroadcastApiCall cannot find tmp buffer id to use.");
    BroadcastScalar(tpipe, current_axis, x, y, id, result);
    return af::SUCCESS;
  }

  std::vector<std::pair<bool, std::vector<uint32_t>>> merge_groups;
  uint32_t broadcast_num = 0;
  Status status = DimensionCollapse(x, y, merge_groups, broadcast_num);
  if (status != af::SUCCESS) {
    GELOGE(af::FAILED, "BroadcastApiCall do dimension collapse failed.");
    return af::FAILED;
  }

  // ub内没有broadcast轴，这种场景下，Schedule会删除无效的broadcast节点
  // 为了防止异常场景，依然保留这种特殊处理逻辑
  if (broadcast_num == static_cast<uint32_t>(0)) {
    BroadcastAllCommonAxis(tpipe, current_axis, x, y, result);
    return af::SUCCESS;
  }

  if (broadcast_num == static_cast<uint32_t>(1)) {
    GE_ASSERT_TRUE(id != -1L, "BroadcastApiCall cannot find tmp buffer id to use.");
    BroadcastOneAxis(tpipe, current_axis, x, y, id, merge_groups, result);
    return af::SUCCESS;
  }

  if (broadcast_num == kDoubleAxisSize &&
      (merge_groups.size() == kAxisSizeThree || merge_groups.size() == kAxisSizeFour)) {
    GE_ASSERT_TRUE(id != -1L, "BroadcastApiCall cannot find tmp buffer id to use.");
    BroadcastTwoAxis(tpipe, current_axis, x, y, id, merge_groups, result);
    return af::SUCCESS;
  }

  GELOGE(af::FAILED, "BroadcastApiCall doesn't support multi discontinuous broadcast axis.");
  GELOGE(af::FAILED, "x_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s",
         x.name.c_str(), VectorToStr(x.axis).c_str(), VectorToStr(x.axis_size).c_str(),
         VectorToStr(x.axis_strides).c_str(), VectorToStr(x.vectorized_axis).c_str(),
         VectorToStr(x.vectorized_axis_pos).c_str(), VectorToStr(x.vectorized_strides).c_str());
  GELOGE(af::FAILED, "y_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s",
         y.name.c_str(), VectorToStr(y.axis).c_str(), VectorToStr(y.axis_size).c_str(),
         VectorToStr(y.axis_strides).c_str(), VectorToStr(y.vectorized_axis).c_str(),
         VectorToStr(y.vectorized_axis_pos).c_str(), VectorToStr(y.vectorized_strides).c_str());
  return af::FAILED;
}
static ApiCallRegister<BroadcastApiCall> register_broadcast_api_call("BroadcastApiCall");
}  // namespace codegen
