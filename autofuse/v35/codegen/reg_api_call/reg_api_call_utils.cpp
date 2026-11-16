/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "tensor_layout_utils.h"
#include "ascir_ops.h"

namespace {
constexpr size_t kDmaMaxLen = 2U;
constexpr size_t kFourAxisNum = 4U;
constexpr size_t kFiveAxisNum = 5U;
constexpr char kNormalPddingMode[] = "AscendC::PaddingMode::Normal";
constexpr char kCompactPddingMode[] = "AscendC::PaddingMode::Compact";
}  // namespace

namespace codegen {
CvApi2DParams BuildCvApi2DParams(const TPipe &tpipe, const Tensor &input, const Tensor &output) {
  CvApi2DParams params;
  params.first_dim = tpipe.CvRows(output);
  params.last_dim = tpipe.IsCvNarrowTensor(output) ? "1" : "curAivN";
  params.output_stride = GenBlockAlignNExpr(output, params.last_dim);
  const bool input_is_cube_output =
      tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && input.id == tpipe.cube_output_tensor_id;
  const auto input_width = tpipe.cv_n_batch ? tpipe.CvColumns(input) : params.last_dim;
  params.input_stride = input_is_cube_output ? "curAlignN" : GenBlockAlignNExpr(input, input_width);
  params.valid_size = params.first_dim + " * " + params.last_dim;
  params.output_compute_size = params.first_dim + " * " + params.output_stride;
  params.input_compute_size = params.first_dim + " * " + params.input_stride;
  return params;
}

std::string GenCvUint32Dims(const CvApi2DParams &params) {
  return "{ConvertToUint32(" + params.first_dim + "), ConvertToUint32(" + params.last_dim + ")}";
}

std::string GenCvUint32Stride(const std::string &stride) {
  return "{ConvertToUint32(" + stride + "), ConvertToUint32(1)}";
}

std::string GenCvUint16Dims(const CvApi2DParams &params) {
  return "{static_cast<uint16_t>(" + params.first_dim + "), static_cast<uint16_t>(" + params.last_dim + ")}";
}

std::string GenCvUint16Stride(const std::string &stride) {
  return "{static_cast<uint16_t>(" + stride + "), static_cast<uint16_t>(1)}";
}

std::string GetCvInputAlignedSize(const ApiCallContext &context, const Tensor &input, const std::string &size_expr) {
  return IsCVFusionStage(context) ? GenBlockAlignNExpr(input, size_expr) : size_expr;
}

// A5场景，拷贝指令增强，可以通过loop_mode_params消掉另外两层for循环
void SetLoopModeParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, LoopModeParams &loop_mode_param,
                       bool copy_in) {
  int64_t total_len = data_copy_param.repeats.size();
  if (total_len <= static_cast<int64_t>(kDmaMaxLen)) {
    return;
  }

  int64_t loop_idx = 0;
  for (int64_t idx = total_len - 1 - kDmaMaxLen; idx >= 0; idx--) {
    if (loop_idx > 1) {
      break;
    }
    loop_mode_param.loop_size[loop_idx] = tpipe.tiler.ActualSize(data_copy_param.repeats[idx]);
    loop_mode_param.loop_src_stride[loop_idx] =
        copy_in ? tpipe.tiler.Size(data_copy_param.gm_strides[idx]) : tpipe.tiler.Size(data_copy_param.ub_strides[idx]);
    loop_mode_param.loop_dst_stride[loop_idx] =
        copy_in ? tpipe.tiler.Size(data_copy_param.ub_strides[idx]) : tpipe.tiler.Size(data_copy_param.gm_strides[idx]);
    loop_idx++;
  }
}

// SetLoopModeParams 表达式版本 - 输出到 LoopModeParamsExpr
// 功能与 SetLoopModeParams 完全一致，只是输出类型不同
void SetLoopModeParamsExpr(const DataCopyParams &data_copy_param, LoopModeParamsExpr &loop_mode_param, bool copy_in) {
  int64_t total_len = data_copy_param.repeats.size();
  if (total_len <= static_cast<int64_t>(kDmaMaxLen)) {
    return;
  }

  int64_t loop_idx = 0;
  for (int64_t idx = total_len - 1 - kDmaMaxLen; idx >= 0; idx--) {
    if (loop_idx > 1) {
      break;
    }
    // loop_size: ActualSize 转换，使用赋值替换默认值
    loop_mode_param.loop_size[loop_idx] = CombinedExpression(ExprItemFactory::ActualSize(data_copy_param.repeats[idx]));
    // loop_src_stride: Size 转换，根据 copy_in 决定来源
    const auto &src_stride_expr = copy_in ? data_copy_param.gm_strides[idx] : data_copy_param.ub_strides[idx];
    loop_mode_param.loop_src_stride[loop_idx] = CombinedExpression(ExprItemFactory::Size(src_stride_expr));
    // loop_dst_stride: Size 转换，根据 copy_in 决定来源
    const auto &dst_stride_expr = copy_in ? data_copy_param.ub_strides[idx] : data_copy_param.gm_strides[idx];
    loop_mode_param.loop_dst_stride[loop_idx] = CombinedExpression(ExprItemFactory::Size(dst_stride_expr));
    loop_idx++;
  }
}

std::string GetPaddingMode(const Tensor &ub_tensor, const DataCopyParams &data_copy_param, bool has_transpose) {
  af::AscTensorAttr attr;
  attr.axis = ub_tensor.axis;
  attr.repeats = ub_tensor.axis_size;
  attr.strides = ub_tensor.axis_strides;
  attr.vectorized_axis = ub_tensor.vectorized_axis;

  if (data_copy_param.repeats.size() <= 1) {
    return kNormalPddingMode;
  }
  ascir::SizeExpr repeat = data_copy_param.repeats.back();
  ascir::SizeExpr stride = data_copy_param.ub_strides[data_copy_param.ub_strides.size() - kDmaMaxLen];
  bool status = af::SymbolicUtils::StaticCheckEq(repeat, stride) == af::TriBool::kTrue;

  ascgen_utils::DiscontinuityInfo info;
  auto analyze_success = ascgen_utils::TensorLayoutUtils::AnalyzeLoadDiscontinuity(attr, info) == af::SUCCESS;
  auto no_multiple_discontinuities = !info.has_multiple_discontinuities;
  GELOGD("GetPaddingMode conditions: analyze_success=%d, no_multiple_discontinuities=%d", analyze_success,
         no_multiple_discontinuities);
  if (((analyze_success && no_multiple_discontinuities) || has_transpose) && status) {
    return kCompactPddingMode;
  }
  return kNormalPddingMode;
}

std::string GenLoopModeParams(const LoopModeParams &loop_mode_param, int64_t input_dtype_size,
                              int64_t output_dtype_size) {
  std::stringstream ss;
  ss << "{static_cast<uint32_t>(" << loop_mode_param.loop_size[0] << "), static_cast<uint32_t>("
     << loop_mode_param.loop_size[1] << "), static_cast<uint64_t>(" << loop_mode_param.loop_src_stride[0] << " * "
     << input_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_dst_stride[0] << " * "
     << output_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_src_stride[1] << " * "
     << input_dtype_size << "), static_cast<uint64_t>(" << loop_mode_param.loop_dst_stride[1] << " * "
     << output_dtype_size << ")}";
  return ss.str();
}

void CreateBaseDmaCall(const Tensor &input, const Tensor &output, const DmaParams &dma_param, std::string &padding_mode,
                       std::stringstream &ss, bool copy_in) {
  std::string dtype_name;
  Tensor::DtypeName(input.dtype, dtype_name);
  ss << "DataCopyPadExtend<" << dtype_name << ", " << padding_mode << ">(";
  if (copy_in) {
    ss << output << ", " << input << "[" << dma_param.gm_offset << "], ";
  } else {
    ss << output << "[" << dma_param.gm_offset << "], " << input << ", ";
  }
  ss << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
     << dma_param.dst_stride << ");" << std::endl;
}

void CreateBaseEnhanceDmaCall(const Tensor &input, const Tensor &output, const DmaParams &dma_param,
                              std::string padding_mode, const LoopModeParams &loop_mode_param, std::stringstream &ss,
                              bool copy_in) {
  std::string dtype_name;
  Tensor::DtypeName(input.dtype, dtype_name);
  ss << "DataCopyPadExtend<" << dtype_name << ", " << padding_mode << ">(";
  if (copy_in) {
    ss << output << "[" << dma_param.ub_offset << "], " << input << "[" << dma_param.gm_offset << "], ";
  } else {
    ss << output << "[" << dma_param.gm_offset << "], " << input << "[" << dma_param.ub_offset << "], ";
  }
  ss << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
     << dma_param.dst_stride << ", "
     << GenLoopModeParams(loop_mode_param, ge::GetSizeByDataType(input.dtype), ge::GetSizeByDataType(output.dtype))
     << ");" << std::endl;
}

void CreateEnhanceDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                          const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss,
                          bool copy_in, bool has_transpose) {
  size_t total_len = data_copy_param.repeats.size();
  DmaParams dma_param;
  SetDmaParams(tpipe, data_copy_param, dma_param, copy_in);
  dma_param.gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset);
  LoopModeParams loop_mode_param;
  SetLoopModeParams(tpipe, data_copy_param, loop_mode_param, copy_in);
  const Tensor &ub_tensor = copy_in ? output : input;
  std::string padding_mode = GetPaddingMode(ub_tensor, data_copy_param, has_transpose);
  if (total_len <= kDmaMaxLen) {
    CreateBaseDmaCall(input, output, dma_param, padding_mode, ss, copy_in);
    return;
  }

  if (total_len > kDmaMaxLen && total_len <= kFourAxisNum) {
    CreateBaseEnhanceDmaCall(input, output, dma_param, padding_mode, loop_mode_param, ss, copy_in);
    return;
  }

  // 超过四层for循环，需要外抛
  std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                         data_copy_param.gm_strides.end() - kFourAxisNum);
  std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                         data_copy_param.ub_strides.end() - kFourAxisNum);
  std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(), data_copy_param.repeats.end() - kFourAxisNum);
  std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
  std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
  std::stringstream ss1;
  dma_param.gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset) + " + " + gm_inner_offset;
  dma_param.ub_offset = ub_inner_offset;
  CreateBaseEnhanceDmaCall(input, output, dma_param, padding_mode, loop_mode_param, ss1, copy_in);
  CreateOuterFor(tpipe, repeats, ss1, ss, 0);
}

void SetNddmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, NddmaParams &nddma_param,
                    const int64_t &tensor_id, std::string padding_mode, std::stringstream &ss) {
  bool is_compact_mode = (padding_mode == kCompactPddingMode);
  nddma_param.ss_output_dims << "const int64_t output_dims_" << tensor_id << "[5] = {";
  nddma_param.ss_output_stride << "const int64_t output_stride_" << tensor_id << "[5] = {";
  nddma_param.ss_input_stride << "const int64_t input_stride_" << tensor_id << "[5] = {";
  for (size_t i = 0UL;
       data_copy_param.repeats.size() < kFiveAxisNum && i < kFiveAxisNum - data_copy_param.repeats.size(); i++) {
    if (data_copy_param.repeats.empty() && i == kFiveAxisNum - 1) {
      nddma_param.ss_output_dims << "1";
      nddma_param.ss_output_stride << "1";
      nddma_param.ss_input_stride << "1";
      break;
    }
    nddma_param.ss_output_dims << "1, ";
    nddma_param.ss_output_stride << "1, ";
    nddma_param.ss_input_stride << "1, ";
  }
  size_t i = data_copy_param.repeats.size() > kFiveAxisNum ? data_copy_param.repeats.size() - kFiveAxisNum : 0UL;
  size_t j = std::min(data_copy_param.repeats.size(), kFiveAxisNum);
  while (j > 0UL) {
    if (j == 1UL) {
      nddma_param.ss_output_dims << tpipe.tiler.ActualSize(data_copy_param.repeats[i]);
      nddma_param.ss_output_stride << (is_compact_mode ? tpipe.tiler.ActualSize(data_copy_param.ub_strides[i])
                                                       : tpipe.tiler.Size(data_copy_param.ub_strides[i]));
      nddma_param.ss_input_stride << tpipe.tiler.Size(data_copy_param.gm_strides[i]);
      break;
    }
    nddma_param.ss_output_dims << tpipe.tiler.ActualSize(data_copy_param.repeats[i]) << ", ";
    nddma_param.ss_output_stride << (is_compact_mode ? tpipe.tiler.ActualSize(data_copy_param.ub_strides[i])
                                                     : tpipe.tiler.Size(data_copy_param.ub_strides[i]))
                                 << ", ";
    nddma_param.ss_input_stride << tpipe.tiler.Size(data_copy_param.gm_strides[i]) << ", ";
    i++;
    j--;
  }
  nddma_param.ss_output_dims << "};" << std::endl;
  nddma_param.ss_output_stride << "};" << std::endl;
  nddma_param.ss_input_stride << "};" << std::endl;
  ss << nddma_param.ss_output_dims.str() << nddma_param.ss_output_stride.str() << nddma_param.ss_input_stride.str();
}

void CreateNddmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                     const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss) {
  const std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                               data_copy_param.gm_strides.end() - kFiveAxisNum);
  const std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                               data_copy_param.ub_strides.end() - kFiveAxisNum);
  const std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(),
                                             data_copy_param.repeats.end() - kFiveAxisNum);
  const std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
  const std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
  std::stringstream ss1;
  NddmaParams nddma_param;
  std::string padding_mode = GetPaddingMode(output, data_copy_param, false);
  SetNddmaParams(tpipe, data_copy_param, nddma_param, output.id, padding_mode, ss);
  ss1 << "DataCopyNddma(" << output << "[" << ub_inner_offset << "], " << input << "[" << gm_offset << " + "
      << tpipe.tiler.Size(offset) << " + " << gm_inner_offset << "], "
      << "output_dims_" << output.id << ", " << "output_stride_" << output.id << ", " << "input_stride_" << output.id
      << ");" << std::endl;
  CreateOuterFor(tpipe, repeats, ss1, ss, 0UL);
}

void BuildDataCopyApiParamInCVFusion(const TPipe &tpipe, CodegenApiParam &api_param,
                                     DmaSpecificParams &dma_specific_params, const Tensor &gm, const Tensor &ub,
                                     std::string &dtype_name, bool copy_in) {
  api_param.template_params.emplace_back("AscendC::PaddingMode::Normal");
  dma_specific_params.data_copy_params.valid = true;  // 标记数据有效
  dma_specific_params.data_copy_params.block_count = CombinedExprFactory::SymbolVar("curAivM");
  if (copy_in) {
    api_param.input_params.emplace_back(gm.Str(), true, CombinedExprFactory::SymbolVar("offset"));
    api_param.output_params.emplace_back(ub.Str(), true, CombinedExprFactory::Constant(0));
    if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      const auto aligned_n = "KernelUtils::BlkAlign<" + dtype_name + ">(curAivN)";
      dma_specific_params.data_copy_params.block_len = CombinedExprFactory::SymbolVar("curAivN");
      dma_specific_params.data_copy_params.src_stride =
          CombinedExpression(ExprItemFactory::SymbolVar("shapeN"), ExprItemFactory::SymbolVar("curAivN"), "-");
      dma_specific_params.data_copy_params.dst_stride =
          CombinedExpression(ExprItemFactory::SymbolVar(aligned_n), ExprItemFactory::SymbolVar("curAivN"), "-");
    } else {
      dma_specific_params.data_copy_params.block_len = CombinedExprFactory::SymbolVar("load_block_len");
      dma_specific_params.data_copy_params.src_stride = CombinedExprFactory::SymbolVar("load_src_stride");
      dma_specific_params.data_copy_params.dst_stride = CombinedExprFactory::SymbolVar("load_dst_stride");
    }
    int dtype_size = GetSizeByDataType(gm.dtype);
    if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse &&
        (dtype_size == 1 || dtype_size == 2 || dtype_size == 4)) {
      // LoadAlign仅支持字节大小为1、2、4的数据类型，否则GatherMask编译错误。
      // 超过4字节的数据类型，CV融合场景下目前一定是对齐拷入的，不需要RemovePad。
      std::stringstream ss;
      ss << "if (KernelUtils::BlkAlign<" << dtype_name << ">(curAlignN) != curAlignN) {" << std::endl;
      ss << "event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));" << std::endl;
      ss << "SetFlag<HardEvent::MTE2_V>(eventID);" << std::endl;
      ss << "WaitFlag<HardEvent::MTE2_V>(eventID);" << std::endl;
      ss << "uint8_t mask = 7;" << std::endl;
      ss << "uint64_t rsvdCnt = 0;" << std::endl;
      ss << "AscendC::GatherMask(" << ub << ", " << ub << ", mask, true, static_cast<uint32_t>(curAlignN)"
         << ", {1, static_cast<uint16_t>(curAivM), static_cast<uint16_t>(KernelUtils::BlkAlign<" << dtype_name
         << ">(curAlignN) * sizeof(" << dtype_name << ") / ONE_BLK_SIZE), 0}"
         << ", rsvdCnt);" << std::endl;
      ss << "}" << std::endl;
      api_param.api_post_process.emplace_back(ss.str());
    }
  } else {
    api_param.output_params.emplace_back(gm.Str(), true, CombinedExprFactory::SymbolVar("offset"));
    api_param.input_params.emplace_back(ub.Str(), true, CombinedExprFactory::Constant(0));
    const auto aligned_n = "KernelUtils::BlkAlign<" + dtype_name + ">(curAivN)";
    dma_specific_params.data_copy_params.block_len = CombinedExprFactory::SymbolVar("curAivN");
    dma_specific_params.data_copy_params.src_stride =
        CombinedExpression(ExprItemFactory::SymbolVar(aligned_n), ExprItemFactory::SymbolVar("curAivN"), "-");
    // dst_stride = shapeN - curAivN
    dma_specific_params.data_copy_params.dst_stride =
        CombinedExpression(ExprItemFactory::SymbolVar("shapeN"), ExprItemFactory::SymbolVar("curAivN"), "-");
  }
}

void BuildDataCopyBaseParams(const TPipe &tpipe, DataCopyParams &data_copy_param,
                             DmaSpecificParams &dma_specific_params, bool copy_in) {
  DmaParamsExpr dma_param;
  SetDmaParamsExpr(tpipe, data_copy_param, dma_param, copy_in);
  dma_specific_params.data_copy_params.valid = true;  // 标记数据有效
  dma_specific_params.data_copy_params.block_count = dma_param.block_count;
  dma_specific_params.data_copy_params.block_len = dma_param.block_len;
  dma_specific_params.data_copy_params.src_stride = dma_param.src_stride;
  dma_specific_params.data_copy_params.dst_stride = dma_param.dst_stride;
}

void BuildDataCopyLoopModeParams(DataCopyParams &data_copy_param, DmaSpecificParams &dma_specific_params,
                                 int64_t dtype_size, bool copy_in) {
  LoopModeParamsExpr loop_mode_param;
  SetLoopModeParamsExpr(data_copy_param, loop_mode_param, copy_in);

  dma_specific_params.loop_mode_params.valid = true;  // 标记loop mode参数有效

  // loop_sizes: ActualSize + cast
  dma_specific_params.loop_mode_params.loop_sizes.emplace_back(
      CombinedExprFactory::ActualSizeWithCast(loop_mode_param.loop_size[0].items[0].expr));
  dma_specific_params.loop_mode_params.loop_sizes.emplace_back(
      CombinedExprFactory::ActualSizeWithCast(loop_mode_param.loop_size[1].items[0].expr));

  // loop_strides: Size * dtype_size + cast
  dma_specific_params.loop_mode_params.loop_src_strides.emplace_back(
      CombinedExprFactory::SizeWithCastAndMultiplier(loop_mode_param.loop_src_stride[0].items[0].expr, dtype_size));
  dma_specific_params.loop_mode_params.loop_src_strides.emplace_back(
      CombinedExprFactory::SizeWithCastAndMultiplier(loop_mode_param.loop_src_stride[1].items[0].expr, dtype_size));
  dma_specific_params.loop_mode_params.loop_dst_strides.emplace_back(
      CombinedExprFactory::SizeWithCastAndMultiplier(loop_mode_param.loop_dst_stride[0].items[0].expr, dtype_size));
  dma_specific_params.loop_mode_params.loop_dst_strides.emplace_back(
      CombinedExprFactory::SizeWithCastAndMultiplier(loop_mode_param.loop_dst_stride[1].items[0].expr, dtype_size));
}

Status BuildDataCopyApiParamInNormal(const TPipe &tpipe, CodegenApiParam &api_param,
                                     DmaSpecificParams &dma_specific_params, const Tensor &src, const Tensor &dst,
                                     std::string &gm_offset, bool copy_in, bool has_transpose) {
  DataCopyParams data_copy_param;
  GE_ASSERT_TRUE(CalculateDmaParams(tpipe, dst, copy_in ? dst : src, data_copy_param), "CalculateDmaParams failed");
  size_t total_len = data_copy_param.repeats.size();
  const Tensor &ub_tensor = copy_in ? dst : src;
  std::string padding_mode = GetPaddingMode(ub_tensor, data_copy_param, has_transpose);
  api_param.template_params.emplace_back(padding_mode);

  BuildDataCopyBaseParams(tpipe, data_copy_param, dma_specific_params, copy_in);
  if (total_len > kDmaMaxLen) {
    BuildDataCopyLoopModeParams(data_copy_param, dma_specific_params, ge::GetSizeByDataType(src.dtype), copy_in);
  }

  // 初始化 offset 表达式
  CombinedExpression gm_offset_expr = CombinedExprFactory::SymbolVar(gm_offset);
  CombinedExpression ub_offset_expr = CombinedExprFactory::Constant(0);

  if (total_len > kFourAxisNum) {
    // 超过四层for循环，需要外抛
    // 提取外层的 strides 和 repeats
    std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                           data_copy_param.gm_strides.end() - kFourAxisNum);
    std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                           data_copy_param.ub_strides.end() - kFourAxisNum);
    std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(), data_copy_param.repeats.end() - kFourAxisNum);

    // 计算 inner offset 表达式：outer_for_0 * Size(stride0) + outer_for_1 * Size(stride1) + ...
    CombinedExpression gm_inner_offset = CalcInnerOffsetExpr(gm_stride);
    CombinedExpression ub_inner_offset = CalcInnerOffsetExpr(ub_stride);

    // gm_offset = gm_offset + gm_inner_offset
    gm_offset_expr.AddItem(gm_inner_offset.items[0], "+");
    for (size_t i = 1; i < gm_inner_offset.items.size(); i++) {
      gm_offset_expr.AddItem(gm_inner_offset.items[i], gm_inner_offset.operators[i - 1]);
    }

    // ub_offset = ub_inner_offset
    ub_offset_expr = ub_inner_offset;

    // 设置外层循环的 repeats
    for (const auto &repeat : repeats) {
      api_param.outer_loop_axes.emplace_back(CombinedExpression(ExprItemFactory::ActualSize(repeat)));
    }
  }

  // 根据 copy_in 设置 input_params 和 output_params 的 offset
  if (copy_in) {
    api_param.input_params.emplace_back(src.Str(), true, gm_offset_expr);
    api_param.output_params.emplace_back(dst.Str(), true, ub_offset_expr);
  } else {
    api_param.input_params.emplace_back(src.Str(), true, ub_offset_expr);
    api_param.output_params.emplace_back(dst.Str(), true, gm_offset_expr);
  }
  return af::SUCCESS;
}

Status GenDataCopyDimParam(const CodegenApiParam &api_param, const Tiler &tiler, std::string graph_name,
                           std::string node_name, std::stringstream &ss) {
  auto *dma_params = std::get_if<DmaSpecificParams>(&api_param.specific_params);
  GE_ASSERT_NOTNULL(dma_params, "dma_params is null, graph name: %s, node name: %s", graph_name.c_str(),
                    node_name.c_str());

  // 调用 ToStr 方法转换表达式
  ss << dma_params->data_copy_params.block_count.ToStr(tiler) << ", ";
  ss << dma_params->data_copy_params.block_len.ToStr(tiler) << ", ";
  ss << dma_params->data_copy_params.src_stride.ToStr(tiler) << ", ";
  ss << dma_params->data_copy_params.dst_stride.ToStr(tiler);

  // 只有loop mode参数有效时才输出
  if (dma_params->loop_mode_params.valid && !dma_params->loop_mode_params.loop_sizes.empty()) {
    ss << ", {" << dma_params->loop_mode_params.loop_sizes[0].ToStr(tiler) << ", "
       << dma_params->loop_mode_params.loop_sizes[1].ToStr(tiler) << ", "
       << dma_params->loop_mode_params.loop_src_strides[0].ToStr(tiler) << ", "
       << dma_params->loop_mode_params.loop_dst_strides[0].ToStr(tiler) << ", "
       << dma_params->loop_mode_params.loop_src_strides[1].ToStr(tiler) << ", "
       << dma_params->loop_mode_params.loop_dst_strides[1].ToStr(tiler) << "}";
  }
  ss << ");" << std::endl;
  return af::SUCCESS;
}

bool IsGraphHasTransposeNode(const af::AscNodePtr &node) {
  const auto owner_graph = node->GetOwnerComputeGraph();
  for (const auto &graph_node : owner_graph->GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(graph_node)) {
      return true;
    }
  }
  return false;
}

}  // namespace codegen
