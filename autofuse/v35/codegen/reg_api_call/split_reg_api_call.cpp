/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_reg_api_call.h"
#include "api_call/utils/api_call_factory.h"
#include "common_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops.h"
#include "common/checker.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
namespace semantic = ascgen_utils::template_semantic;
using namespace ascgen_utils;
constexpr uint32_t kDataBlockSize = 32;
Status SplitRegApiCall::ParseAttr(const ascir::NodeView &node) {
  node_ = node;
  return af::SUCCESS;
}

Status SplitRegApiCall::ParseSplitDim(const Tensor &x, const Tensor &y0, size_t &split_dim) {
  GELOGD("x_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s", x.name.c_str(),
         VectorToStr(x.axis).c_str(), VectorToStr(x.axis_size).c_str(), VectorToStr(x.axis_strides).c_str(),
         VectorToStr(x.vectorized_axis).c_str(), VectorToStr(x.vectorized_axis_pos).c_str(),
         VectorToStr(x.vectorized_strides).c_str());
  GELOGD("y_t_name:%s, axis_id:%s, size:%s, strides:%s, v_axis_id:%s, v_axis_pos:%s, v_strides:%s", y0.name.c_str(),
         VectorToStr(y0.axis).c_str(), VectorToStr(y0.axis_size).c_str(), VectorToStr(y0.axis_strides).c_str(),
         VectorToStr(y0.vectorized_axis).c_str(), VectorToStr(y0.vectorized_axis_pos).c_str(),
         VectorToStr(y0.vectorized_strides).c_str());

  GE_CHK_BOOL_RET_STATUS(x.vectorized_axis.size() == y0.vectorized_axis.size(), af::FAILED,
                         "Codegen split input output vectorized_axis not equal");
  // 遍历向量化轴, 确定split轴
  bool find_split_dim = false;
  for (size_t i = 0; i < y0.vectorized_axis.size(); i++) {
    auto pos = y0.vectorized_axis_pos[i];
    if (af::SymbolicUtils::StaticCheckEq(x.axis_size[pos], y0.axis_size[pos]) != af::TriBool::kTrue) {
      split_dim = i;
      find_split_dim = true;
      GELOGI("find split_dim:%zu, ", split_dim);
      break;
    }
  }
  GE_ASSERT_TRUE(find_split_dim, "split dim was not found in vectorized_axis");
  return af::SUCCESS;
}

Status SplitFindNonZeroStride(const std::vector<ascir::SizeExpr> &vectorized_strides, int32_t index,
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

Status SplitRegApiCall::InitializeTiling(size_t split_dim, const vector<std::reference_wrapper<const Tensor>> &outputs,
                                         const Tensor &x, SplitRegApiCall::SplitTiling &tiling) {
  auto data_type_size = ge::GetSizeByDataType(x.dtype);
  GE_ASSERT_TRUE(data_type_size > 0);
  tiling.data_type_size = static_cast<uint32_t>(data_type_size);
  tiling.total_rows_expr = af::ops::One;
  tiling.src_col_size_expr = af::ops::One;
  for (size_t i = 0; i < x.vectorized_axis.size(); ++i) {
    auto pos = x.vectorized_axis_pos[i];
    auto axis_size_expr = x.axis_size[pos];
    if (i < split_dim) {
      tiling.total_rows_expr = tiling.total_rows_expr * axis_size_expr;
    } else {
      tiling.src_col_size_expr = tiling.src_col_size_expr * axis_size_expr;
    }
  }
  auto split_dim_stride = x.vectorized_strides[split_dim];
  tiling.src_col_actual_size_expr = x.axis_size[x.vectorized_axis_pos[split_dim]] * split_dim_stride;
  tiling.dst_col_size_exprs.resize(outputs.size());
  tiling.dst_col_actual_size_exprs.resize(outputs.size());
  for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
    auto &y = outputs[output_index].get();
    tiling.dst_col_size_exprs[output_index] = af::ops::One;
    for (size_t i = split_dim; i < y.vectorized_axis.size(); ++i) {
      auto pos = y.vectorized_axis_pos[i];
      auto axis_size_expr = y.axis_size[pos];
      tiling.dst_col_size_exprs[output_index] = tiling.dst_col_size_exprs[output_index] * axis_size_expr;
    }
    auto split_dim_stride_dst = y.vectorized_strides[split_dim];
    GE_ASSERT_SUCCESS(
        SplitFindNonZeroStride(y.vectorized_strides, static_cast<int32_t>(split_dim), split_dim_stride_dst));
    tiling.dst_col_actual_size_exprs[output_index] =
        y.axis_size[y.vectorized_axis_pos[split_dim]] * split_dim_stride_dst;
  }
  return af::SUCCESS;
}

bool SplitRegApiCall::IsAllAligned(SplitRegApiCall::SplitTiling &tiling) {
  tiling.src_offsets.resize(tiling.dst_col_actual_size_exprs.size());
  auto offset = af::ops::Zero;
  for (size_t index = 0U; index < tiling.dst_col_actual_size_exprs.size(); ++index) {
    auto size = af::Symbol(tiling.data_type_size) * tiling.dst_col_actual_size_exprs[index];
    if (af::SymbolicUtils::StaticCheckEq(af::sym::Mod(size, af::Symbol(kDataBlockSize)), af::ops::Zero) !=
        af::TriBool::kTrue) {
      GELOGI("input[%zu] size = %s, is not aligned", index,
             af::SymbolicUtils::ToString(tiling.dst_col_actual_size_exprs[index]).c_str());
      return false;
    }
    if (!tiling.dst_col_actual_size_exprs[index].IsConstExpr()) {
      tiling.all_static = false;
    }
    tiling.src_offsets[index] = offset;
    offset = offset + tiling.dst_col_actual_size_exprs[index];
  }
  if (!tiling.src_col_actual_size_expr.IsConstExpr()) {
    tiling.all_static = false;
  }
  return true;
}

void SplitRegApiCall::GenSplitTilingForAllAligned(SplitRegApiCall::SplitTiling &tiling, const Tiler &tiler,
                                                  std::stringstream &ss) {
  const auto qualifier = tiling.all_static ? "constexpr " : "const ";
  ss << qualifier;
  ss << "SplitTilingAllAligned<" << tiling.dst_col_size_exprs.size() << "> split_tiling {" << std::endl;
  ss << "  .src_col_size = " << tiler.Size(tiling.src_col_actual_size_expr, true) << "," << std::endl;
  ss << "  .dst_col_sizes = { ";
  for (const auto &dst_col_size : tiling.dst_col_actual_size_exprs) {
    ss << tiler.Size(dst_col_size, true) << ", ";
  }
  ss << "}," << std::endl;
  ss << "  .src_offsets = { ";
  for (const auto &src_col_size : tiling.src_offsets) {
    ss << tiler.Size(src_col_size, true) << ", ";
  }
  ss << "}," << std::endl;
  ss << "};" << std::endl;
}

void SplitRegApiCall::GenSrcTensors(const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                    const std::string &dtype_name, std::stringstream &ss) {
  ss << "LocalTensor<" << dtype_name << "> split_dst_tensors[] { ";
  for (auto &output : outputs) {
    const auto &x = output.get();
    ss << x << ", ";
  }
  ss << "};" << std::endl;
}

af::Status SplitRegApiCall::GenerateForAllAligned(const vector<std::reference_wrapper<const Tensor>> &outputs,
                                                  const Tensor &x, SplitRegApiCall::SplitTiling &tiling,
                                                  const Tiler &tiler, std::stringstream &ss) {
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x.dtype));
  GenSplitTilingForAllAligned(tiling, tiler, ss);
  GenSrcTensors(outputs, dtype_name, ss);
  ss << "SplitAllAligned<" << dtype_name << ", " << outputs.size() << ">(" << tiler.ActualSize(tiling.total_rows_expr)
     << ", " << "split_tiling"
     << ", " << x << ", " << "split_dst_tensors"
     << ");" << std::endl;
  return af::SUCCESS;
}

Status SplitRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                 const vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const vector<std::reference_wrapper<const Tensor>> &outputs, string &result) const {
  (void)current_axis;
  GE_CHK_BOOL_RET_STATUS((!inputs.empty()) && (!outputs.empty()), af::FAILED,
                         "Codegen input or output tensor is empty");
  const auto &x = inputs[0].get();
  const auto &y0 = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  size_t split_dim;
  GE_ASSERT_SUCCESS(ParseSplitDim(x, y0, split_dim), "Failed to parse split dim");
  SplitTiling split_tiling;
  GE_ASSERT_SUCCESS(InitializeTiling(split_dim, outputs, x, split_tiling));
  GE_ASSERT_TRUE(split_tiling.src_col_actual_size_expr.Simplify() == split_tiling.src_col_size_expr.Simplify(),
                 "Padding is not supported by split yet");
  std::stringstream ss;
  if (IsAllAligned(split_tiling)) {
    // by copy
    GE_ASSERT_SUCCESS(SetSemanticFeature(semantic::kUbAllAligned));
    GE_ASSERT_SUCCESS(GenerateForAllAligned(outputs, x, split_tiling, tpipe.tiler, ss));
  } else {
    // 获取tmp_buf复用TBuf的id
    int64_t life_time_axis_id = -1L;
    int64_t id = -1L;
    auto it = this->tmp_buf_id.find(life_time_axis_id);
    GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "SplitRegApiCall cannot find tmp buffer id to use.");
    id = it->second;
    // by scatter
    GE_ASSERT_SUCCESS(SetSemanticFeature(GetSemanticFeature(split_tiling)));
    GE_ASSERT_SUCCESS(GenerateDefault(outputs, x, split_tiling, tpipe, ss, id));
  }
  result = ss.str();
  return af::SUCCESS;
}

std::string SplitRegApiCall::GetSemanticFeature(const SplitTiling &tiling) {
  constexpr int64_t kVectorRegWidth = 256;
  const int64_t gather_max_len = kVectorRegWidth / static_cast<int64_t>(tiling.data_type_size);
  bool has_copy = false;
  bool has_gather = false;
  for (const auto &dst_col_size : tiling.dst_col_size_exprs) {
    if (!dst_col_size.IsConstExpr()) {
      return semantic::kUbExtendAdaptive;
    }
    int64_t col_size = 0;
    GE_WARN_ASSERT(dst_col_size.GetConstValue(col_size));
    if (col_size > gather_max_len) {
      has_copy = true;
    } else {
      has_gather = true;
    }
  }
  if (has_copy && has_gather) {
    return semantic::kUbExtendMixed;
  }
  return has_copy ? semantic::kUbExtendCopy : semantic::kUbExtendGather;
}

af::Status SplitRegApiCall::GenerateDefault(const vector<std::reference_wrapper<const Tensor>> &outputs,
                                            const Tensor &x, SplitRegApiCall::SplitTiling &tiling, const TPipe &t_pipe,
                                            std::stringstream &ss, const int64_t tmp_buf_id) {
  std::string dtype_name;
  (void)Tensor::DtypeName(x.dtype, dtype_name);
  if (tiling.data_type_size == sizeof(uint64_t)) {
    auto kB64ToB32 = af::Symbol(sizeof(uint64_t) / sizeof(uint32_t));
    SplitTiling tiling_b32;
    tiling_b32.total_rows_expr = tiling.total_rows_expr;
    tiling_b32.src_col_size_expr = tiling.src_col_size_expr * kB64ToB32;
    for (auto &dst_col_size : tiling.dst_col_size_exprs) {
      tiling_b32.dst_col_size_exprs.emplace_back(dst_col_size * kB64ToB32);
    }
    DefineSplitTiling(tiling_b32, t_pipe.tiler, ss);
    dtype_name = "uint32_t";
  } else if (NeedB8ToB16(tiling)) {
    GELOGD("can use b16 split, dtype: %s", dtype_name.c_str());
    SplitTiling tiling_b16;
    const auto &kB16ToB8 = af::Symbol(2);
    GE_ASSERT_TRUE(kB16ToB8 != 0);
    tiling_b16.total_rows_expr = tiling.total_rows_expr;
    tiling_b16.src_col_size_expr = tiling.src_col_size_expr / kB16ToB8;
    for (auto &dst_col_size : tiling.dst_col_size_exprs) {
      tiling_b16.dst_col_size_exprs.emplace_back(dst_col_size / kB16ToB8);
    }
    DefineSplitTiling(tiling_b16, t_pipe.tiler, ss);
    dtype_name = "uint16_t";
  } else {
    DefineSplitTiling(tiling, t_pipe.tiler, ss);
  }

  GenDstAddrs(outputs, dtype_name, ss);
  ss << "split::SplitExtend<" << dtype_name << ", " << outputs.size() << ">("
     << "(__ubuf__ " << dtype_name << " *)" << x << ".GetPhyAddr()"
     << ", " << "split_dst_addrs, " << t_pipe.tmp_buf << "_" << std::to_string(tmp_buf_id) << ", split_tiling);"
     << std::endl;
  return af::SUCCESS;
}

bool SplitRegApiCall::NeedB8ToB16(SplitRegApiCall::SplitTiling &tiling) {
  if (tiling.data_type_size != sizeof(uint8_t)) {
    return false;
  }
  return std::all_of(tiling.dst_col_size_exprs.cbegin(), tiling.dst_col_size_exprs.cend(),
                     [](const af::Expression &src_col_size) -> bool {
                       return af::sym::Mod(src_col_size, af::Symbol(sizeof(uint16_t))) == af::ops::Zero;
                     });
}

void SplitRegApiCall::DefineSplitTiling(SplitRegApiCall::SplitTiling &tiling, const Tiler &tiler,
                                        std::stringstream &ss) {
  ss << "const split::SplitTiling<" << tiling.dst_col_size_exprs.size() << "> split_tiling {" << std::endl;
  ss << "  .num_rows = static_cast<uint32_t>(" << tiler.ActualSize(tiling.total_rows_expr) << "), " << std::endl;
  ss << "  .num_src_cols = " << tiler.Size(tiling.src_col_size_expr, true) << ", " << std::endl;
  ss << "  .num_dsts_cols = {";
  for (const auto &dst_col_size : tiling.dst_col_size_exprs) {
    ss << tiler.Size(dst_col_size, true) << ", ";
  }
  ss << "}" << std::endl;
  ss << "};" << std::endl;
}

void SplitRegApiCall::GenDstAddrs(const vector<std::reference_wrapper<const Tensor>> &outputs, const string &dtype_name,
                                  std::stringstream &ss) {
  ss << "__ubuf__ " << dtype_name << " *split_dst_addrs[] { ";
  for (auto &output : outputs) {
    const auto &x = output.get();
    ss << "(__ubuf__ " << dtype_name << " *)" << x << ".GetPhyAddr(), ";
  }
  ss << "};" << std::endl;
}

static ApiCallRegister<SplitRegApiCall> register_split_api_call("SplitRegApiCall");
}  // namespace codegen
