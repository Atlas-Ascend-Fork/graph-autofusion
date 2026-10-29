/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_api_call.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace ascgen_utils;

namespace {
constexpr uint32_t kDataBlockSize = 32;
constexpr uint32_t kAddrListSize = 16;
constexpr size_t kOffsetSecondLastStride = 2;

Status FindNonZeroStride(const std::vector<ascir::SizeExpr> &vectorized_strides, int32_t index,
                         af::Expression &stride) {
  for (int32_t i = index; i >= 0; --i) {
    stride = vectorized_strides[i];
    if (af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) != af::TriBool::kTrue) {
      break;
    }
  }
  GE_ASSERT_TRUE(stride != af::ops::Zero, "Failed to find non-zero v_stride before index = %d, v_strides = %s", index,
                 af::ToString(vectorized_strides).c_str());
  return af::SUCCESS;
}
}  // namespace

Status ConcatApiCall::ParseAttr(const ascir::NodeView &node) {
  // TTODO 消除Brc后可能会导致repeat不连续，先用attr规避，后续整改
  GE_ASSERT_SUCCESS(ApiCall::ParseAttr(node));
  (void)af::AttrUtils::GetBool(node->GetOpDesc(), "_concat_small_tail", use_concat_small_tail_api_);
  node_ = node;
  return af::SUCCESS;
}

Status ConcatApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
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
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L, id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  // 调用api
  ConcatTiling concat_tiling;
  GE_ASSERT_SUCCESS(InitializeTiling(concat_dim, inputs, y, concat_tiling));
  std::stringstream ss;
  if (use_concat_small_tail_api_) {
    GE_ASSERT_TRUE(id != -1L, "ConcatApiCall cannot find tmp buffer id to use.");
    if (concat_tiling.dst_col_size_expr.IsConstExpr()) {
      GE_ASSERT_SUCCESS(CalcTiling(concat_dim, inputs, concat_tiling));
      DefineInputList(concat_tiling, dtype_name, inputs, ss);
      GE_ASSERT_SUCCESS(DefineConcatContext(concat_tiling, dtype_name, tpipe.tiler, ss));
      DefineConcatTiling(concat_tiling, ss);
      ss << "ConcatExtendV2(concat_context, tiling, " << y << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ");"
         << std::endl;
    } else {
      DefineInputList(concat_tiling, dtype_name, inputs, ss);
      GE_ASSERT_SUCCESS(DefineConcatContext(concat_tiling, dtype_name, tpipe.tiler, ss));
      DefineConcatShape(concat_tiling, tpipe.tiler, ss);
      ss << "ConcatExtendV2Dyn(concat_context, concat_shape, " << y << ", " << tpipe.tmp_buf << "_"
         << std::to_string(id) << ");" << std::endl;
    }
  } else {
    if (IsAllAligned(concat_tiling)) {
      GE_ASSERT_SUCCESS(GenerateForAllAligned(inputs, y, concat_tiling, tpipe.tiler, ss));
    } else {
      GE_ASSERT_TRUE(id != -1L, "ConcatApiCall cannot find tmp buffer id to use.");
      // 生成concat dim
      ss << "uint32_t concat_dim = " << concat_dim << ";" << std::endl;
      GenConcatParams(inputs, y, tpipe.tiler, dtype_name, ss);
      ss << "ConcatExtend<" << dtype_name << ", " << y.vectorized_axis.size() << ", " << std::to_string(inputs.size())
         << ">(dst_params, srcs_params, concat_dim, " << tpipe.tmp_buf << "_" << std::to_string(id) << ");"
         << std::endl;
    }
  }
  result = ss.str();
  return af::SUCCESS;
}

af::Status ConcatApiCall::GenerateForAllAligned(const vector<std::reference_wrapper<const Tensor>> &inputs,
                                                const Tensor &y, const ConcatApiCall::ConcatTiling &tiling,
                                                const Tiler &tiler, std::stringstream &ss) {
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  GenConcatTilingForAllAligned(tiling, tiler, ss);
  GenSrcTensors(inputs, dtype_name, ss);
  ss << "ConcatAllAligned<" << dtype_name << ", " << inputs.size() << ">(" << tiler.ActualSize(tiling.total_rows_expr)
     << ", " << "concat_tiling"
     << ", " << y << ", " << "concat_src_tensors"
     << ");" << std::endl;
  return af::SUCCESS;
}

std::string ConcatApiCall::GenSizeWithUint32Cast(const Tiler &tiler, const ascir::SizeExpr &expr,
                                                 bool using_int_tiling_data) {
  const std::string size_str = tiler.Size(expr, using_int_tiling_data);
  return expr.IsConstExpr() ? size_str : "static_cast<uint32_t>(" + size_str + ")";
}

void ConcatApiCall::GenConcatParams(const std::vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y,
                                    const Tiler &tiler, const string &dtype_name, std::stringstream &ss) {
  ss << "const ConcatParams<" + dtype_name + ", " << y.vectorized_axis.size() << ">"
     << " dst_params = {" << std::endl
     << ".shape  = {";
  for (size_t i = 0; i < y.vectorized_axis.size(); i++) {
    auto pos = y.vectorized_axis_pos[i];
    ss << GenSizeWithUint32Cast(tiler, y.axis_size[pos], true);
    if (i != y.vectorized_axis.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}," << std::endl << ".stride = {";
  for (size_t i = 0; i < y.vectorized_axis.size(); i++) {
    ss << GenSizeWithUint32Cast(tiler, y.vectorized_strides[i], true);
    if (i != y.vectorized_axis.size() - 1) {
      ss << ", ";
    }
  }

  ss << "}," << std::endl << ".tensor = &" << y << "," << std::endl << "};" << std::endl;

  // 生成concat输入参数
  ss << "const ConcatParams<" + dtype_name + ", " << y.vectorized_axis.size() << "> srcs_params["
     << std::to_string(inputs.size()) << "] = {" << std::endl;

  for (auto &input : inputs) {
    const auto &x = input.get();
    ss << "{" << std::endl << ".shape  = {";
    for (size_t i = 0U; i < x.vectorized_axis.size(); i++) {
      auto pos = x.vectorized_axis_pos[i];
      ss << GenSizeWithUint32Cast(tiler, x.axis_size[pos], true);
      if (i != x.vectorized_axis.size()) {
        ss << ", ";
      }
    }
    ss << "}," << std::endl << ".stride = {";
    for (size_t i = 0U; i < x.vectorized_axis.size(); i++) {
      ss << GenSizeWithUint32Cast(tiler, x.vectorized_strides[i], true);
      if (i != x.vectorized_axis.size() - 1) {
        ss << ", ";
      }
    }
    ss << "}," << std::endl << ".tensor = &" << x << "," << std::endl << "}," << std::endl;
  }
  ss << "};" << std::endl;
}

Status ConcatApiCall::ParseConcatDim(const Tensor &x0, const Tensor &y, size_t &concat_dim) {
  GELOGD("x0_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s", x0.name.c_str(),
         VectorToStr(x0.axis).c_str(), VectorToStr(x0.axis_size).c_str(), VectorToStr(x0.axis_strides).c_str(),
         VectorToStr(x0.vectorized_axis).c_str(), VectorToStr(x0.vectorized_axis_pos).c_str(),
         VectorToStr(x0.vectorized_strides).c_str());
  GELOGD("y_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s", y.name.c_str(),
         VectorToStr(y.axis).c_str(), VectorToStr(y.axis_size).c_str(), VectorToStr(y.axis_strides).c_str(),
         VectorToStr(y.vectorized_axis).c_str(), VectorToStr(y.vectorized_axis_pos).c_str(),
         VectorToStr(y.vectorized_strides).c_str());

  GE_CHK_BOOL_RET_STATUS(x0.vectorized_axis.size() == y.vectorized_axis.size(), af::FAILED,
                         "Codegen concat input output vectorized_axis not equal");
  // 遍历向量化轴, 确定concat轴
  bool find_concat_dim = false;
  for (size_t i = 0; i < y.vectorized_axis.size(); i++) {
    auto pos = y.vectorized_axis_pos[i];
    if (af::SymbolicUtils::StaticCheckEq(x0.axis_size[pos], y.axis_size[pos]) != af::TriBool::kTrue) {
      concat_dim = i;
      find_concat_dim = true;
      GELOGI("find concat_dim:%zu, ", concat_dim);
      break;
    }
  }
  GE_ASSERT_TRUE(find_concat_dim, "Failed to find concat dim in vectorized_axis");
  return af::SUCCESS;
}

Status ConcatApiCall::InitializeTiling(size_t concat_dim, const vector<std::reference_wrapper<const Tensor>> &inputs,
                                       const Tensor &y, ConcatApiCall::ConcatTiling &tiling) {
  auto data_type_size = ge::GetSizeByDataType(y.dtype);
  GE_ASSERT_TRUE(data_type_size > 0);
  tiling.data_type_size = static_cast<uint32_t>(data_type_size);
  tiling.total_rows_expr = af::ops::One;
  tiling.dst_col_size_expr = af::ops::One;
  for (size_t i = 0; i < y.vectorized_axis.size(); ++i) {
    auto pos = y.vectorized_axis_pos[i];
    auto axis_size_expr = y.axis_size[pos];
    if (i < concat_dim) {
      tiling.total_rows_expr = tiling.total_rows_expr * axis_size_expr;
    } else {
      tiling.dst_col_size_expr = tiling.dst_col_size_expr * axis_size_expr;
    }
  }
  const auto &concat_dim_stride = y.vectorized_strides[concat_dim];
  GE_ASSERT_SUCCESS(
      FindNonZeroStride(y.vectorized_strides, static_cast<int32_t>(concat_dim) - 1, tiling.dst_row_stride));
  tiling.src_col_size_exprs.resize(inputs.size());
  tiling.src_col_actual_size_exprs.resize(inputs.size());
  tiling.src_non_zero_strides.resize(inputs.size(), af::ops::Zero);
  tiling.src_row_strides.resize(inputs.size(), af::ops::Zero);
  tiling.last_dim_size_exprs.resize(inputs.size());
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    auto &x = inputs[input_index].get();
    tiling.src_col_size_exprs[input_index] = af::ops::One;
    tiling.last_dim_size_exprs[input_index] = x.axis_size[x.vectorized_axis_pos[x.vectorized_axis.size() - 1]];
    ;
    for (size_t i = concat_dim; i < x.vectorized_axis.size(); ++i) {
      auto pos = x.vectorized_axis_pos[i];
      auto axis_size_expr = x.axis_size[pos];
      tiling.src_col_size_exprs[input_index] = tiling.src_col_size_exprs[input_index] * axis_size_expr;
    }
    tiling.src_col_actual_size_exprs[input_index] = x.axis_size[x.vectorized_axis_pos[concat_dim]] * concat_dim_stride;
    // 检查是否存在padding, 当前padding只会在最后一维, RemovePadding与concat dim无关
    GE_ASSERT_SUCCESS(FindNonZeroStride(x.vectorized_strides,
                                        static_cast<int32_t>(x.vectorized_strides.size()) - kOffsetSecondLastStride,
                                        tiling.src_non_zero_strides[input_index]));
    GE_ASSERT_SUCCESS(FindNonZeroStride(x.vectorized_strides, static_cast<int32_t>(concat_dim) - 1,
                                        tiling.src_row_strides[input_index]));
    auto is_padded = af::SymbolicUtils::StaticCheckEq(tiling.src_non_zero_strides[input_index],
                                                      tiling.last_dim_size_exprs[input_index]) != af::TriBool::kTrue;
    GE_CHK_BOOL_ONLY_LOG((!is_padded), "Input[%zu] is_padded = %d, axes = %s, strides = %s", input_index,
                         static_cast<int32_t>(is_padded), VectorToStr(x.vectorized_axis).c_str(),
                         VectorToStr(x.vectorized_strides).c_str());
    tiling.is_padded.emplace_back(is_padded);
    tiling.any_padded = (tiling.any_padded || is_padded);
  }
  return af::SUCCESS;
}

Status ConcatApiCall::CalcTiling([[maybe_unused]] size_t concat_dim,
                                 const vector<std::reference_wrapper<const Tensor>> &inputs,
                                 ConcatApiCall::ConcatTiling &tiling) const {
  const uint32_t kScaleToB16 = tiling.data_type_size / sizeof(uint16_t);
  const uint32_t kEltNumPerBlock = kAddrListSize * kDataBlockSize / tiling.data_type_size;

  GE_ASSERT_TRUE(!node_->attr.tmp_buffers.empty());
  GE_ASSERT_TRUE(node_->attr.tmp_buffers.front().buf_desc.size.GetConstValue(tiling.tmp_buf_size));
  GE_ASSERT_TRUE(tiling.dst_row_stride.GetConstValue(tiling.dst_col_size));
  tiling.gcd = kAddrListSize;
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    int64_t dim_size = -1;
    GE_ASSERT_TRUE(tiling.src_col_size_exprs[input_index].GetConstValue(dim_size));
    GE_ASSERT_TRUE(dim_size < std::numeric_limits<uint32_t>::max());
    tiling.gcd = ascgen_utils::Gcd(tiling.gcd, static_cast<uint32_t>(dim_size));
    tiling.src_col_sizes.emplace_back(dim_size);
  }

  tiling.dst_row_num_unit = tiling.dst_col_size * kScaleToB16;
  tiling.max_repeat_times = (tiling.tmp_buf_size >> 10U) / (tiling.dst_col_size / tiling.gcd);
  GE_ASSERT_TRUE(tiling.max_repeat_times > 0);
  // 非尾块, 每次loop的结果元素个数
  tiling.max_element_num = tiling.max_repeat_times * (tiling.dst_col_size / tiling.gcd) * kEltNumPerBlock;
  // 非尾块, 每次loop的原始行数
  tiling.max_orig_row_num = tiling.max_element_num / tiling.dst_col_size;
  tiling.first_copy_repeat_times = tiling.max_repeat_times * kAddrListSize / kScaleToB16 / tiling.gcd;
  tiling.last_trans_repeat_times = tiling.max_repeat_times * (tiling.dst_col_size / tiling.gcd);
  tiling.per_repeat_size = (tiling.dst_col_size / tiling.gcd) * kEltNumPerBlock;
  CalcTilingForInputs(inputs, kEltNumPerBlock, tiling);
  GELOGI("ConcatTiling: gcd=%u, tmp_buf_size=%u, max_repeat_times=%u, max_element_num=%u, max_orig_row_num=%u",
         tiling.gcd, tiling.tmp_buf_size, tiling.max_repeat_times, tiling.max_element_num, tiling.max_orig_row_num);
  return af::SUCCESS;
}

Status ConcatApiCall::CalcTilingForInputs(const vector<std::reference_wrapper<const Tensor>> &inputs, size_t block_size,
                                          ConcatApiCall::ConcatTiling &concat_tiling) {
  uint32_t buffer_offset = 0;
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    int64_t src_row_stride = -1;
    GE_ASSERT_TRUE(concat_tiling.src_row_strides[input_index].GetConstValue(src_row_stride));
    concat_tiling.src_loop_strides.emplace_back(concat_tiling.max_orig_row_num * src_row_stride);
    concat_tiling.src_buffer_offsets.emplace_back(buffer_offset);
    buffer_offset +=
        (concat_tiling.max_repeat_times * (concat_tiling.src_col_sizes[input_index] / concat_tiling.gcd) * block_size);

    int64_t second_last_stride = -1;
    GE_ASSERT_TRUE(concat_tiling.src_non_zero_strides[input_index].GetConstValue(second_last_stride));
    int64_t last_dim_size = -1;
    GE_ASSERT_TRUE(concat_tiling.last_dim_size_exprs[input_index].GetConstValue(last_dim_size));
    GE_ASSERT_TRUE(second_last_stride > 0, "Failed to get second last stride");
    concat_tiling.second_last_dim_strides.emplace_back(second_last_stride);
    concat_tiling.gather_mask_dim_sizes.emplace_back(last_dim_size);
  }
  return af::SUCCESS;
}

void ConcatApiCall::DefineConcatTiling(const ConcatApiCall::ConcatTiling &tiling, std::stringstream &ss) {
  ss << "constexpr ConcatTiling<" << tiling.src_col_size_exprs.size() << "> tiling {" << std::endl;
  ss << "  .gcd = " << tiling.gcd << ", " << std::endl;
  ss << "  .tmp_buf_size = " << tiling.tmp_buf_size << ", " << std::endl;
  ss << "  .dst_dim_size = " << tiling.dst_col_size << ", " << std::endl;
  ss << "  .dst_row_num_unit = " << tiling.dst_row_num_unit << ", " << std::endl;
  ss << "  .max_repeat_times = " << tiling.max_repeat_times << ", " << std::endl;
  ss << "  .max_element_num = " << tiling.max_element_num << ", " << std::endl;
  ss << "  .max_orig_row_num = " << tiling.max_orig_row_num << ", " << std::endl;
  ss << "  .per_repeat_size = " << tiling.per_repeat_size << ", " << std::endl;
  ss << "  .first_copy_repeat_times = " << tiling.first_copy_repeat_times << ", " << std::endl;
  ss << "  .last_trans_repeat_times = " << tiling.last_trans_repeat_times << ", " << std::endl;
  ss << "  .src_dim_sizes = {";
  for (const auto &src_dim_size : tiling.src_col_size_exprs) {
    ss << src_dim_size << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .src_strides = {";
  for (const auto src_stride : tiling.src_loop_strides) {
    ss << src_stride << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .src_buffer_offsets = {";
  for (const auto src_buffer_offset : tiling.src_buffer_offsets) {
    ss << src_buffer_offset << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .gather_mask_repeat_strides = {";
  for (size_t i = 0; i < tiling.second_last_dim_strides.size(); ++i) {
    auto repeat_stride =
        tiling.is_padded[i] ? tiling.second_last_dim_strides[i] * tiling.data_type_size / kDataBlockSize : 0U;
    ss << repeat_stride << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .gather_mask_dim_sizes = {";
  for (const auto scale : tiling.gather_mask_dim_sizes) {
    ss << scale << ", ";
  }
  ss << "}" << std::endl;
  ss << "};" << std::endl;
}

void ConcatApiCall::DefineConcatShape(const ConcatApiCall::ConcatTiling &tiling, const Tiler &tiler,
                                      std::stringstream &ss) {
  ss << "const ConcatShape<" << tiling.src_col_size_exprs.size() << "> concat_shape {" << std::endl;
  ss << "  .dst_cols = " << GenSizeWithUint32Cast(tiler, tiling.dst_row_stride) << ", " << std::endl;
  ss << "  .src_cols = {";
  for (const auto &src_dim_size : tiling.src_col_size_exprs) {
    ss << GenSizeWithUint32Cast(tiler, src_dim_size) << ", ";
  }
  ss << "}," << std::endl;
  if (tiling.any_padded) {
    ss << "  .src_row_strides = {";
    for (const auto &src_row_stride : tiling.src_row_strides) {
      ss << GenSizeWithUint32Cast(tiler, src_row_stride) << ", ";
    }
    ss << "}," << std::endl;

    ss << "  .src_second_last_dim_strides = {";
    for (const auto &second_last_dim_stride : tiling.src_non_zero_strides) {
      ss << GenSizeWithUint32Cast(tiler, second_last_dim_stride) << ", ";
    }
    ss << "}," << std::endl;

    ss << "  .gather_mask_dim_sizes = {";
    for (const auto &last_dim_size_expr : tiling.last_dim_size_exprs) {
      ss << GenSizeWithUint32Cast(tiler, last_dim_size_expr) << ", ";
    }
    ss << "}," << std::endl;
  }
  ss << "};" << std::endl;
}

af::Status ConcatApiCall::DefineConcatContext(const ConcatTiling &tiling, const std::string &dtype_name,
                                              const Tiler &tiler, std::stringstream &ss) {
  std::string sub_type = "DiffDim";
  bool concat_same_dim = false;
  std::set<int64_t> concat_axis_sizes;
  if (tiling.dst_col_size_expr.IsConstExpr()) {
    int64_t dst_col_size;
    GE_ASSERT_TRUE(tiling.dst_col_size_expr.GetConstValue(dst_col_size));
    int64_t dst_row_stride;
    GE_ASSERT_TRUE(tiling.dst_row_stride.GetConstValue(dst_row_stride));
    concat_axis_sizes.insert(tiling.src_col_sizes.cbegin(), tiling.src_col_sizes.cend());
    concat_same_dim =
        (concat_axis_sizes.size() == 1) && (*concat_axis_sizes.cbegin() == 1) && (dst_col_size == dst_row_stride);
    if (concat_same_dim) {
      sub_type = "SameDim";
    }
  }
  const auto &padding_type = (tiling.any_padded ? "Padded" : "");
  const auto &concat_type = std::string("ConcatContext") + sub_type + padding_type;
  GELOGI("Concat type = %s", concat_type.c_str());
  ss << concat_type << "<" << dtype_name << ", " << tiling.src_col_size_exprs.size();
  if (concat_same_dim) {
    ss << ", " << *concat_axis_sizes.cbegin();
  }
  ss << "> concat_context;" << std::endl;
  ss << "concat_context.total_row_num = " << tiler.ActualSize(tiling.total_rows_expr) << ";" << std::endl;
  ss << "concat_context.input_list = &input_list;" << std::endl;
  return af::SUCCESS;
}

void ConcatApiCall::DefineInputList(const ConcatTiling &tiling, const std::string &dtype_name,
                                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                    std::stringstream &ss) {
  ss << "ConcatInputList<" << dtype_name << ", " << inputs.size() << "> input_list {" << std::endl;
  ss << "  .src_tensor_base_addrs = {";
  for (const auto &input : inputs) {
    ss << "(" << dtype_name << " *)" << input << ".GetPhyAddr(), ";
  }
  ss << "}," << std::endl;
  if (tiling.any_padded) {
    ss << "  .src_tensors = {";
    for (size_t i = 0U; i < inputs.size(); ++i) {
      std::string addr = tiling.is_padded[i] ? std::string("&") + inputs[i].get().Str() : "nullptr";
      ss << addr << ", ";
    }
    ss << "}," << std::endl;
  }
  ss << "};" << std::endl;
}

bool ConcatApiCall::IsAllAligned(ConcatApiCall::ConcatTiling &tiling) {
  const auto &col_size_exprs = tiling.src_col_actual_size_exprs;
  for (size_t index = 0U; index < col_size_exprs.size(); ++index) {
    auto size = af::Symbol(tiling.data_type_size) * col_size_exprs[index];
    if (af::SymbolicUtils::StaticCheckEq(af::sym::Mod(size, af::Symbol(kDataBlockSize)), af::ops::Zero) !=
        af::TriBool::kTrue) {
      GELOGI("input[%zu] size = %s, is not aligned", index, af::SymbolicUtils::ToString(col_size_exprs[index]).c_str());
      return false;
    }
  }
  // 当前Gather->Reshape->Concat融合时, 存在补轴错位问题, 可能会导致src_col_actual_size_exprs的大小不准确,
  // 所以此处改为使用src_row_strides
  tiling.dst_offsets.resize(col_size_exprs.size());
  auto offset = af::ops::Zero;
  for (size_t index = 0U; index < tiling.src_row_strides.size(); ++index) {
    const auto &src_row_stride = tiling.src_row_strides[index];
    if (!src_row_stride.IsConstExpr()) {
      tiling.all_static = false;
    }
    tiling.dst_offsets[index] = offset;
    offset = offset + src_row_stride;
  }
  return true;
}

void ConcatApiCall::GenConcatTilingForAllAligned(const ConcatApiCall::ConcatTiling &tiling, const Tiler &tiler,
                                                 std::stringstream &ss) {
  const auto qualifier = tiling.all_static ? "constexpr " : "const ";
  ss << qualifier;
  ss << "ConcatTilingAllAligned<" << tiling.src_col_size_exprs.size() << "> concat_tiling {" << std::endl;
  ss << "  .dst_col_size = " << GenSizeWithUint32Cast(tiler, tiling.dst_row_stride, true) << "," << std::endl;
  ss << "  .src_col_sizes = { ";
  for (const auto &src_col_size : tiling.src_row_strides) {
    ss << GenSizeWithUint32Cast(tiler, src_col_size, true) << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .dst_offsets = { ";
  for (const auto &src_col_size : tiling.dst_offsets) {
    ss << GenSizeWithUint32Cast(tiler, src_col_size, true) << ", ";
  }
  ss << "}," << std::endl;
  ss << "};" << std::endl;
}

void ConcatApiCall::GenSrcTensors(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::string &dtype_name, std::stringstream &ss) {
  ss << "LocalTensor<" << dtype_name << "> concat_src_tensors[] { ";
  for (auto &input : inputs) {
    const auto &x = input.get();
    ss << x << ", ";
  }
  ss << "};" << std::endl;
}

[[maybe_unused]] static ApiCallRegister<ConcatApiCall> register_concat_api_call("ConcatApiCall");
}  // namespace codegen
