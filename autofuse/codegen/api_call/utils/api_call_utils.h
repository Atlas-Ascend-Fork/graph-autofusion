/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_API_CALL_UTILS_H__
#define __AUTOFUSE_API_CALL_UTILS_H__

#include "codegen_kernel.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {

// Load/Store参数数据结构，最内层for循环作为burstlen
struct DataCopyParams {
  std::vector<ascir::SizeExpr> repeats;
  std::vector<ascir::SizeExpr> gm_strides;
  std::vector<ascir::SizeExpr> ub_strides;
};

struct DmaParams {
  std::string block_count = "1";
  std::string block_len = "1";  // 这里的单位是数字个数，在DataCopyExtend中会转换成字节数
  std::string src_stride = "0";
  std::string dst_stride = "0";
  std::string gm_offset = "0";
  std::string ub_offset = "0";
};

// DmaParams 表达式版本 - 使用 CombinedExpression 替代 std::string
struct DmaParamsExpr {
  CombinedExpression block_count = CombinedExprFactory::Constant(1);
  CombinedExpression block_len =
      CombinedExprFactory::Constant(1);  // 这里的单位是数字个数，在DataCopyExtend中会转换成字节数
  CombinedExpression src_stride = CombinedExprFactory::Constant(0);
  CombinedExpression dst_stride = CombinedExprFactory::Constant(0);
  CombinedExpression gm_offset = CombinedExprFactory::Constant(0);
  CombinedExpression ub_offset = CombinedExprFactory::Constant(0);

  // 辅助方法：转换为字符串（需要 tiler）
  std::string ToStr(const Tiler &tiler) const {
    std::stringstream ss;
    ss << block_count.ToStr(tiler) << ", " << block_len.ToStr(tiler) << ", " << src_stride.ToStr(tiler) << ", "
       << dst_stride.ToStr(tiler);
    return ss.str();
  }

  // 辅助方法：生成带 offset 的调用参数
  std::string ToStrWithOffset(const Tiler &tiler, bool copy_in, const std::string &src_name,
                              const std::string &dst_name) const {
    std::stringstream ss;
    if (copy_in) {
      ss << dst_name << "[" << ub_offset.ToStr(tiler) << "], " << src_name << "[" << gm_offset.ToStr(tiler) << "], ";
    } else {
      ss << dst_name << "[" << gm_offset.ToStr(tiler) << "], " << src_name << "[" << ub_offset.ToStr(tiler) << "], ";
    }
    ss << block_count.ToStr(tiler) << ", " << block_len.ToStr(tiler) << ", " << src_stride.ToStr(tiler) << ", "
       << dst_stride.ToStr(tiler);
    return ss.str();
  }
};

struct ApiLoopParams {
  std::vector<std::string> outer_repeats;
  std::vector<std::vector<ascir::SizeExpr>> inputs_strides;
  std::vector<std::vector<ascir::SizeExpr>> outputs_strides;
  ascir::SizeExpr cal_count = af::ops::One;
  ascir::SizeExpr input_second_to_last_stride = af::ops::One;
  ascir::SizeExpr output_second_to_last_stride = af::ops::One;
};

struct MergeInfo {
  std::vector<ascir::SizeExpr> merge_repeats;
  std::vector<ascir::SizeExpr> merge_gm_strides;
  std::vector<ascir::SizeExpr> merge_ub_strides;
};

struct AxisInfo {
  ascir::SizeExpr prev_repeat = af::ops::One;
  ascir::SizeExpr prev_axis_stride = af::ops::One;
  ascir::SizeExpr prev_vectorized_axis_stride = af::ops::One;
};

struct VectorizedAixsLoopStatus {
  ascir::SizeExpr prev_repeat = af::ops::One;
  std::vector<ascir::SizeExpr> prev_input_axis_stride;
  std::vector<ascir::SizeExpr> prev_output_axis_stride;
};

struct VectorizedAxisLoopMergeStatus {
  std::vector<std::string> merge_repeats_str;
  std::vector<ascir::SizeExpr> merge_repeats;
  std::vector<std::vector<ascir::AxisId>> merge_axis_ids;
  std::vector<std::vector<ascir::SizeExpr>> inputs_strides;
  std::vector<std::vector<ascir::SizeExpr>> outputs_strides;
};

bool CalculateDmaParams(const TPipe &tpipe, const Tensor &gm_tensor, const Tensor &ub_tensor, DataCopyParams &param,
                        bool multi_axis_copy = false);
void SetDmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, DmaParams &dma_param, bool copy_in,
                  bool need_swap = false);
void SetDmaParamsExpr(const TPipe &tpipe, const DataCopyParams &data_copy_param, DmaParamsExpr &dma_param, bool copy_in,
                      bool need_swap = false);
void CreateDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                   const DataCopyParams &param, const ascir::SizeExpr &offset, std::stringstream &ss, bool copy_in);
void CreateOuterFor(const TPipe &tpipe, const std::vector<ascir::SizeExpr> &outer_repeats, const std::stringstream &ss1,
                    std::stringstream &ss, size_t cur_idx);
void GetOneAxisSize(const TPipe &tpipe, const Tensor &tensor, const uint32_t idx, std::stringstream &ss);
std::string CalcInnerOffset(const TPipe &tpipe, const std::vector<ascir::SizeExpr> &strides);
CombinedExpression CalcInnerOffsetExpr(const std::vector<ascir::SizeExpr> &strides);
void CreateComputeNodeOuterFor(const std::vector<std::string> &outer_repeats, const std::stringstream &ss1,
                               std::stringstream &ss, size_t cur_idx);
bool GenerateVectorizedAxisMergeStatus(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                                       VectorizedAxisLoopMergeStatus &merge_info, const TPipe &tpipe);

bool CheckAxisContinuous(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                         VectorizedAixsLoopStatus &axis_info, int64_t index);
void SaveApiLoopAxisParams(VectorizedAxisLoopMergeStatus &merge_info, ApiLoopParams &param);
bool GetMaxDtypeSize(const ge::DataType input_data_type, const ge::DataType out_put_data_type, std::string &dtype_size);
bool IsCVFusionStage(const ApiCallContext &context);
Status GetTensorDtypeSize(const Tensor &tensor, int64_t &dtype_size);
std::string GenBlockAlignNExpr(const Tensor &tensor, const std::string &n_expr);
std::string GetCVAlignedSize(const ApiCallContext &context, const Tensor &tensor, const std::string &size_expr);
bool ShouldIgnoreZeroAxis(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs, int64_t cur_index);
bool IsInputOutputStrideAllZero(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                                int64_t cur_index);
void GenerateLinkStoreEventCode(const Tensor &ub, const std::string &offset_str, std::stringstream &ss);
bool IsAllVecAxisContinuous(const af::AscNode &node);
}  // namespace codegen

#endif
