/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "compare_api_call.h"

#include <sstream>
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"

namespace codegen {
using namespace std;

static void CreateComputeNodeOuterForIfRequired(size_t outer_repeats_size, ApiLoopParams param,
                                                const std::stringstream &ss1, std::stringstream &ss) {
  if (outer_repeats_size == 1UL) {
    ss << ss1.str();
  } else {
    CreateComputeNodeOuterFor({param.outer_repeats.begin(), param.outer_repeats.begin() + outer_repeats_size - 1}, ss1,
                              ss, 0);
  }
  return;
}

Status CompareApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                std::string &result) const {
  auto x1 = inputs[0].get();
  auto x2 = inputs[1].get();
  auto y = outputs[0].get();
  stringstream ss;
  ApiLoopParams param;
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "CompareApiCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, {}, tpipe.tmp_buf.name + "_" + std::to_string(id));

  // 如果第2个输入是ub_scalar场景, 初始化x2为ub_scalar对应的变量
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x2.dtype));
  std::string x2_scalar = x2.Str();
  if (x2.is_ub_scalar && x2.need_gen_get_value_of_ub_scalar) {
    x2_scalar = "(" + dtype_name + ")" + x2.ub_scalar_name;
  }

  if (x2.IsAnyScalar()) {
    const std::string actual_size =
        IsCVFusionStage(this->api_call_context) ? GenBlockAlignNExpr(x1, x1.actual_size.Str()) : x1.actual_size.Str();
    ub_inputs.push_back(x1);
    ub_outputs.push_back(y);
    bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
    GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
    SaveApiLoopAxisParams(merge_info, param);
    std::string scalar_local_blk_tensor_name_x2 = x2.IsConstScalar() ? "local_blk_tensor_of_" + x2.name : x2.name;
    size_t outer_repeats_size = param.outer_repeats.size();
    if (outer_repeats_size == 0U) {
      ss << "CompareScalarExtend" << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
         << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2_scalar << ", "
         << "CMPMODE::" << this->api_name_ << ", " << actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
         << ");" << std::endl;
    } else {
      std::stringstream ss1;
      size_t input0_strides_size = param.inputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[0].begin(),
                                                        param.inputs_strides[0].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

      std::string dtype_size = "4";
      ss1 << "CompareExtend<" << dtype_name << ", CMPMODE::" << this->api_name_ << ">(" << y << "["
          << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << scalar_local_blk_tensor_name_x2 << "[0], " << param.outer_repeats[outer_repeats_size - 1] << ", "
          << (IsCVFusionStage(this->api_call_context) ? GenBlockAlignNExpr(x1, tpipe.tiler.ActualSize(param.cal_count))
                                                      : tpipe.tiler.ActualSize(param.cal_count))
          << ", " << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
          << ", ONE_BLK_SIZE);" << std::endl;
      CreateComputeNodeOuterForIfRequired(outer_repeats_size, param, ss1, ss);
    }
  } else {
    const std::string actual_size =
        IsCVFusionStage(this->api_call_context) ? GenBlockAlignNExpr(x1, x1.actual_size.Str()) : x1.actual_size.Str();
    ub_inputs.push_back(x1);
    ub_inputs.push_back(x2);
    ub_outputs.push_back(y);
    bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
    GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
    SaveApiLoopAxisParams(merge_info, param);
    size_t outer_repeats_size = param.outer_repeats.size();
    if (outer_repeats_size == 0U) {
      ss << "CompareExtend" << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1
         << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
         << "CMPMODE::" << this->api_name_ << ", " << actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
         << ");" << std::endl;
    } else {
      size_t input0_strides_size = param.inputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[0].begin(),
                                                        param.inputs_strides[0].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      size_t input1_strides_size = param.inputs_strides[1].size();
      std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[1].begin(),
                                                        param.inputs_strides[1].begin() + input1_strides_size - 1);
      std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);

      std::string dtype_size = "4";
      std::stringstream ss1;
      // Use this tensor's storage requirement, before the queue/buffer takes the
      // maximum size of all tensors sharing its allocation.
      ss1 << "CompareExtend<" << dtype_name << ", CMPMODE::" << this->api_name_ << ">(" << y << "["
          << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], " << x2 << "["
          << input1_inner_offset << "], " << param.outer_repeats[outer_repeats_size - 1] << ", "
          << (IsCVFusionStage(this->api_call_context) ? GenBlockAlignNExpr(x1, tpipe.tiler.ActualSize(param.cal_count))
                                                      : tpipe.tiler.ActualSize(param.cal_count))
          << ", " << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
          << ", " << x2.size.Str() << " * sizeof(" << dtype_name << "));" << std::endl;
      CreateComputeNodeOuterForIfRequired(outer_repeats_size, param, ss1, ss);
    }
  }

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<CompareApiCall> register_compare_api_call("CompareApiCall");

}  // namespace codegen
