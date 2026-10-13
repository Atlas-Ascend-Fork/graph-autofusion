/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_reg_api_call.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

namespace codegen {
Status ConcatRegApiCall::ParseAttr(const ascir::NodeView &node) {
  node_ = node;
  return af::SUCCESS;
}

Status ConcatRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const vector<std::reference_wrapper<const Tensor>> &outputs, string &result) const {
  (void)current_axis;
  GE_CHK_BOOL_RET_STATUS((!inputs.empty()) && (!outputs.empty()), af::FAILED,
                         "Codegen input or output tensor is empty");
  const auto &x0 = inputs[0].get();
  const auto &y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  size_t concat_dim;
  GE_ASSERT_SUCCESS(ParseConcatDim(x0, y, concat_dim), "Failed to parse concat dim");
  std::stringstream ss;
  if (CanConcatOneAxis(inputs, y)) {
    // 单维首轴concat
    GE_ASSERT_SUCCESS(GenerateForOneAxis(inputs, y, ss));
    result = ss.str();
    return af::SUCCESS;
  }

  ConcatTiling concat_tiling;
  GE_ASSERT_SUCCESS(InitializeTiling(concat_dim, inputs, y, concat_tiling));
  if (IsAllAligned(concat_tiling)) {
    // by copy
    GE_ASSERT_SUCCESS(GenerateForAllAligned(inputs, y, concat_tiling, tpipe.tiler, ss));
  } else {
    // 获取tmp_buf复用TBuf的id
    int64_t life_time_axis_id = -1L;
    int64_t id = -1L;
    auto it = this->tmp_buf_id.find(life_time_axis_id);
    GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "ConcatRegApiCall cannot find tmp buffer id to use.");
    id = it->second;
    GE_ASSERT_SUCCESS(CanUseGather(concat_tiling));
    if (concat_tiling.can_use_gather && concat_tiling.dst_col_size_expr.IsConstExpr()) {
      GE_ASSERT_SUCCESS(GenerateForGather(inputs, y, concat_tiling, tpipe, ss, id));
    } else {
      // by scatter
      GE_ASSERT_SUCCESS(GenerateDefault(inputs, y, concat_tiling, tpipe, ss, id));
    }
  }
  result = ss.str();
  return af::SUCCESS;
}

bool ConcatRegApiCall::IsShareInputs() const {
  std::set<int64_t> queue_ids;
  for (uint32_t i = 0; i < node_->inputs.Size(); ++i) {
    queue_ids.emplace(node_->inputs[i].attr.que.id);
  }
  return queue_ids.size() == 1UL;
}

bool ConcatRegApiCall::AreContiguousBufsPreferred() const {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(IsTile(), false, "%s all inputs are from single source");
  const auto is_all_inputs_shape_equal = ascir::utils::AreConcatInputShapesEqual(node_);
  GE_CHK_BOOL_RET_SPECIAL_STATUS((is_all_inputs_shape_equal == af::TriBool::kFalse), false,
                                 "%s cannot use Gather, input shapes differ", node_->GetNamePtr());
  GELOGD("%s may use Gather, contiguous input bufs are preferred", node_->GetNamePtr());
  return true;
}

bool ConcatRegApiCall::CanConcatOneAxis(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const Tensor &y) {
  constexpr int64_t kVecLen = 256;
  auto data_type_size = ge::GetSizeByDataType(y.dtype);
  GE_ASSERT_TRUE(data_type_size > 0);
  const auto max_size = kVecLen / data_type_size;
  bool concat_one_axis = false;
  if ((y.vectorized_strides.size() == 2UL) && (y.vectorized_strides[0] == af::ops::Zero)) {
    // 单维首轴concat
    for (const auto &input : inputs) {
      const auto &x = input.get();
      GE_WARN_ASSERT(x.vectorized_axis.size() == y.vectorized_strides.size());
      auto pos = x.vectorized_axis_pos.back();
      auto axis_size_expr = x.axis_size[pos];
      int64_t axis_size = std::numeric_limits<int64_t>::max();
      if (axis_size_expr.IsConstExpr()) {
        GE_WARN_ASSERT(axis_size_expr.GetConstValue(axis_size));
      }
      GE_CHK_BOOL_RET_STATUS_NOLOG((axis_size <= max_size), false);
    }
    concat_one_axis = true;
  }
  return concat_one_axis;
}

af::Status ConcatRegApiCall::GenerateDefault(const vector<std::reference_wrapper<const Tensor>> &inputs,
                                             const Tensor &y, const ConcatApiCall::ConcatTiling &tiling,
                                             const TPipe &t_pipe, std::stringstream &ss, const int64_t tmp_buf_id) {
  std::string dtype_name;
  (void)Tensor::DtypeName(y.dtype, dtype_name);
  if (tiling.data_type_size == sizeof(uint64_t)) {
    const ConcatTiling tiling_b32 = B64ToB32(tiling);
    DefineConcatTiling(tiling_b32, t_pipe.tiler, ss);
    dtype_name = "uint32_t";
  } else if (NeedB8ToB16(tiling)) {
    GELOGD("can use b16 concat, dtype: %s", dtype_name.c_str());
    const ConcatTiling tiling_b16 = B8ToB16(tiling);
    DefineConcatTiling(tiling_b16, t_pipe.tiler, ss);
    dtype_name = "uint16_t";
  } else {
    DefineConcatTiling(tiling, t_pipe.tiler, ss);
    NormalizeDtype(dtype_name);
  }
  GenSrcAddrs(inputs, dtype_name, ss);
  if (tiling.can_use_gather) {
    GELOGD("use ConcatExtendDyn");
    ss << "concat::ConcatExtendDyn<";
  } else {
    GELOGD("use ConcatExtend");
    ss << "concat::ConcatExtend<";
  }
  ss << dtype_name << ", " << inputs.size();
  if (tiling.can_use_gather && tiling.all_inputs_shape_equal == af::TriBool::kUnknown) {
    ss << ", " << "true";
  }
  ss << ">("
     << "(__ubuf__ " << dtype_name << " *)" << y << ".GetPhyAddr()"
     << ", " << "concat_src_addrs, " << t_pipe.tmp_buf << "_" << std::to_string(tmp_buf_id) << ", concat_tiling);"
     << std::endl;
  return af::SUCCESS;
}

af::Status ConcatRegApiCall::GenerateForGather(const vector<std::reference_wrapper<const Tensor>> &inputs,
                                               const Tensor &y, const ConcatApiCall::ConcatTiling &tiling,
                                               const TPipe &t_pipe, std::stringstream &ss, const int64_t tmp_buf_id) {
  std::string dtype_name;
  (void)Tensor::DtypeName(y.dtype, dtype_name);
  if (tiling.data_type_size == sizeof(uint64_t)) {
    const ConcatTiling tiling_b32 = B64ToB32(tiling);
    DefineConcatTilingGather(tiling_b32, t_pipe.tiler, ss);
    dtype_name = "uint32_t";
  } else if (NeedB8ToB16(tiling)) {
    GELOGD("can use b16 concat, dtype: %s", dtype_name.c_str());
    const ConcatTiling tiling_b16 = B8ToB16(tiling);
    DefineConcatTilingGather(tiling_b16, t_pipe.tiler, ss);
    dtype_name = "uint16_t";
  } else {
    DefineConcatTilingGather(tiling, t_pipe.tiler, ss);
  }
  NormalizeDtype(dtype_name);
  GenSrcAddrs(inputs, dtype_name, ss);
  ss << "concat::ConcatExtendByGather<" << dtype_name << ", " << inputs.size() << ">("
     << "(__ubuf__ " << dtype_name << " *)" << y << ".GetPhyAddr()"
     << ", " << "concat_src_addrs, " << t_pipe.tmp_buf << "_" << std::to_string(tmp_buf_id) << ", concat_tiling);"
     << std::endl;
  GELOGD("use ConcatExtendByGather");
  return af::SUCCESS;
}

bool ConcatRegApiCall::NeedB8ToB16(const ConcatApiCall::ConcatTiling &tiling) {
  if (tiling.data_type_size != sizeof(uint8_t)) {
    return false;
  }
  for (size_t i = 0; i < tiling.is_padded.size(); i++) {
    auto &col_size = tiling.is_padded[i] ? tiling.last_dim_size_exprs[i] : tiling.src_col_size_exprs[i];
    if (af::sym::Mod(col_size, af::Symbol(sizeof(uint16_t))) != af::ops::Zero) {
      return false;
    }
  }
  return true;
}

ConcatApiCall::ConcatTiling ConcatRegApiCall::B64ToB32(const ConcatTiling &tiling) {
  auto kB64ToB32 = af::Symbol(sizeof(uint64_t) / sizeof(uint32_t));
  ConcatTiling tiling_b32 = tiling;
  tiling_b32.total_rows_expr = tiling.total_rows_expr;
  tiling_b32.dst_col_size_expr = tiling.dst_col_size_expr * kB64ToB32;
  for (auto &src_col_size : tiling_b32.src_col_size_exprs) {
    src_col_size = src_col_size * kB64ToB32;
  }
  for (auto &src_row_stride : tiling_b32.src_row_strides) {
    src_row_stride = src_row_stride * kB64ToB32;
  }
  for (auto &src_non_zero_stride : tiling_b32.src_non_zero_strides) {
    src_non_zero_stride = src_non_zero_stride * kB64ToB32;
  }
  for (auto &last_dim_size_expr : tiling_b32.last_dim_size_exprs) {
    last_dim_size_expr = last_dim_size_expr * kB64ToB32;
  }
  return tiling_b32;
}

ConcatApiCall::ConcatTiling ConcatRegApiCall::B8ToB16(const ConcatTiling &tiling) {
  ConcatTiling tiling_b16 = tiling;
  const auto &kB16ToB8 = af::Symbol(2);
  tiling_b16.total_rows_expr = tiling.total_rows_expr;
  tiling_b16.dst_col_size_expr = tiling.dst_col_size_expr / kB16ToB8;
  for (auto &src_col_size : tiling_b16.src_col_size_exprs) {
    src_col_size = src_col_size / kB16ToB8;
  }
  for (auto &src_row_stride : tiling_b16.src_row_strides) {
    src_row_stride = src_row_stride / kB16ToB8;
  }
  for (auto &src_non_zero_stride : tiling_b16.src_non_zero_strides) {
    src_non_zero_stride = src_non_zero_stride / kB16ToB8;
  }
  for (auto &last_dim_size_expr : tiling_b16.last_dim_size_exprs) {
    last_dim_size_expr = last_dim_size_expr / kB16ToB8;
  }
  return tiling_b16;
}

af::Status ConcatRegApiCall::CanUseGather(ConcatTiling &tiling) const {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(tiling.any_padded, af::SUCCESS, "cannot use Gather: input is padded");
  if (IsTile()) {
    tiling.all_inputs_shape_equal = af::TriBool::kTrue;
  } else {
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!is_input_tbuf_contiguous), af::SUCCESS,
                                   "cannot use Gather: input bufs cannot be contiguous");
    tiling.all_inputs_shape_equal = ascir::utils::AreConcatInputShapesEqual(node_);
  }
  if (tiling.src_col_size_exprs[0].IsConstExpr()) {
    uint32_t src_col_size = 0;
    GE_ASSERT_TRUE(tiling.src_col_size_exprs[0].GetConstValue(src_col_size));
    constexpr uint32_t kMaxSrcSize = 256U / 2U;
    if (src_col_size * tiling.data_type_size > kMaxSrcSize) {
      GELOGD("src col size = %u, over %u, cannot use Gather", src_col_size * tiling.data_type_size, kMaxSrcSize);
      return af::SUCCESS;
    }
  }
  tiling.can_use_gather = true;
  return af::SUCCESS;
}

bool ConcatRegApiCall::IsTile() const {
  std::set<const af::OutDataAnchor *> src_anchors;
  for (const auto &node_and_out_anchor : node_->GetInDataNodesAndAnchors()) {
    src_anchors.emplace(node_and_out_anchor.second.get());
  }
  const bool is_tile = (src_anchors.size() == 1UL);
  GELOGI("is tile by concat case = %d", static_cast<int32_t>(is_tile));
  return is_tile;
}

void ConcatRegApiCall::NormalizeDtype(std::string &dtype_name) {
  if ((dtype_name == "int8_t") || (dtype_name == "bool")) {
    // pack不支持int8_t类型，转为uint8_t进行计算
    dtype_name = "uint8_t";
  }
}

std::string ConcatRegApiCall::GetTilingDataType(const ConcatTiling &tiling) {
  std::string tiling_data_type = "concat::ConcatTiling";
  if (tiling.any_padded) {
    tiling_data_type += "Padded";
  }
  return tiling_data_type;
}

void ConcatRegApiCall::DefineConcatTiling(const ConcatTiling &tiling, const Tiler &tiler, std::stringstream &ss) {
  auto tiling_data_type = GetTilingDataType(tiling);
  ss << "const " << tiling_data_type << "<" << tiling.src_col_size_exprs.size() << "> concat_tiling {" << std::endl;
  ss << "  .num_rows = static_cast<uint32_t>(" << tiler.ActualSize(tiling.total_rows_expr) << ")," << std::endl;
  ss << "  .num_dst_cols = " << tiler.Size(tiling.dst_col_size_expr, true) << "," << std::endl;
  ss << "  .num_srcs_cols = {";
  for (const auto &src_col_size : tiling.src_col_size_exprs) {
    ss << tiler.Size(src_col_size, true) << ", ";
  }
  ss << "}," << std::endl;
  if (tiling.any_padded) {
    ss << "  .src_row_strides = {";
    for (const auto &src_row_stride : tiling.src_row_strides) {
      ss << tiler.Size(src_row_stride, true) << ", ";
    }
    ss << "}," << std::endl;
    ss << "  .src_second_last_dim_strides = {";
    for (size_t i = 0UL; i < tiling.src_non_zero_strides.size(); ++i) {
      auto stride = tiling.is_padded[i] ? tiler.Size(tiling.src_non_zero_strides[i]) : "0";
      ss << stride << ", ";
    }
    ss << "}," << std::endl;
    ss << "  .gather_mask_dim_sizes = {";
    for (size_t i = 0UL; i < tiling.last_dim_size_exprs.size(); ++i) {
      auto dim_size = tiling.is_padded[i] ? tiler.Size(tiling.last_dim_size_exprs[i]) : "0";
      ss << dim_size << ", ";
    }
    ss << "}," << std::endl;
  }
  ss << "};" << std::endl;
}

void ConcatRegApiCall::DefineConcatTilingGather(const ConcatTiling &tiling, const Tiler &tiler, std::stringstream &ss) {
  std::string tiling_data_type = "concat::ConcatByGatherTiling";
  ss << "const " << tiling_data_type << " concat_tiling {" << std::endl;
  ss << "  .num_rows = static_cast<uint32_t>(" << tiler.ActualSize(tiling.total_rows_expr) << ")," << std::endl;
  ss << "  .num_dst_cols = " << tiler.Size(tiling.dst_col_size_expr, true) << "," << std::endl;
  ss << "  .num_src_cols = " << tiler.Size(tiling.src_col_size_exprs[0], true) << "," << std::endl;
  ss << "};" << std::endl;
}

void ConcatRegApiCall::GenSrcAddrs(const vector<std::reference_wrapper<const Tensor>> &inputs, const string &dtype_name,
                                   std::stringstream &ss) {
  ss << "__ubuf__ " << dtype_name << " *concat_src_addrs[] { ";
  for (auto &input : inputs) {
    const auto &x = input.get();
    ss << "(__ubuf__ " << dtype_name << " *)" << x << ".GetPhyAddr(), ";
  }
  ss << "};" << std::endl;
}

Status ConcatRegApiCall::GenerateForOneAxis(const vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y,
                                            std::stringstream &ss) {
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  ss << "constexpr concat::ConcatTilingOneAxis<" << inputs.size() << "> concat_tiling {" << std::endl;
  ss << "  .src_col_sizes = { ";
  std::vector<uint32_t> dst_col_offsets;
  uint32_t dst_col_offset = 0U;
  for (const auto &input : inputs) {
    const auto &x = input.get();
    auto pos = x.vectorized_axis_pos[1];
    auto &axis_size = x.axis_size[pos];
    GE_ASSERT_TRUE(axis_size.IsConstExpr());
    uint32_t src_col_size;
    GE_ASSERT_TRUE(axis_size.GetConstValue(src_col_size));
    ss << src_col_size << ", ";
    dst_col_offsets.push_back(dst_col_offset);
    dst_col_offset += src_col_size;
  }
  ss << "}," << std::endl;
  ss << "  .dst_col_offsets = { ";
  for (const auto offset : dst_col_offsets) {
    ss << offset << ", ";
  }
  ss << "}," << std::endl;
  ss << "};" << std::endl;

  GenSrcAddrs(inputs, dtype_name, ss);
  ss << "concat::ConcatOneAxis<" << dtype_name << ", " << inputs.size() << ">("
     << "(__ubuf__ " << dtype_name << " *)" << y << ".GetPhyAddr()"
     << ", " << "concat_src_addrs, concat_tiling);" << std::endl;
  return af::SUCCESS;
}

[[maybe_unused]] static ApiCallRegister<ConcatRegApiCall> register_concat_api_call("ConcatRegApiCall");
}  // namespace codegen
