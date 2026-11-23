/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transpose_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "transpose_base_type.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

constexpr size_t Dim_Num_2 = 2U;
constexpr size_t Dim_Num_3 = 3U;
constexpr size_t Dim_Num_4 = 4U;
constexpr size_t Axis_0 = 0U;
constexpr size_t Axis_1 = 1U;
constexpr size_t Axis_2 = 2U;
constexpr size_t Axis_3 = 3U;

bool inline IsTranposeType10(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_2) && (output_vectorized_axis[Axis_0] == Axis_1) &&
           (output_vectorized_axis[Axis_1] == Axis_0))
              ? true
              : false);
}

bool inline IsTranposeType102(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_3) && (output_vectorized_axis[Axis_0] == Axis_1) &&
           (output_vectorized_axis[Axis_1] == Axis_0) && (output_vectorized_axis[Axis_2] == Axis_2))
              ? true
              : false);
}
bool inline IsTranposeType021(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_3) && (output_vectorized_axis[Axis_0] == Axis_0) &&
           (output_vectorized_axis[Axis_1] == Axis_2) && (output_vectorized_axis[Axis_2] == Axis_1))
              ? true
              : false);
}

bool inline IsTranposeType210(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_3) && (output_vectorized_axis[Axis_0] == Axis_2) &&
           (output_vectorized_axis[Axis_1] == Axis_1) && (output_vectorized_axis[Axis_2] == Axis_0))
              ? true
              : false);
}
bool inline IsTranposeType0213(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_4) && (output_vectorized_axis[Axis_0] == Axis_0) &&
           (output_vectorized_axis[Axis_1] == Axis_2) && (output_vectorized_axis[Axis_2] == Axis_1) &&
           (output_vectorized_axis[Axis_3] == Axis_3))
              ? true
              : false);
}
bool inline IsTranposeType2103(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_4) && (output_vectorized_axis[Axis_0] == Axis_2) &&
           (output_vectorized_axis[Axis_1] == Axis_1) && (output_vectorized_axis[Axis_2] == Axis_0) &&
           (output_vectorized_axis[Axis_3] == Axis_3))
              ? true
              : false);
}
bool inline IsTranposeType0321(std::vector<uint32_t> &output_vectorized_axis) {
  return (((output_vectorized_axis.size() == Dim_Num_4) && (output_vectorized_axis[Axis_0] == Axis_0) &&
           (output_vectorized_axis[Axis_1] == Axis_3) && (output_vectorized_axis[Axis_2] == Axis_2) &&
           (output_vectorized_axis[Axis_3] == Axis_1))
              ? true
              : false);
}

Status TransposeApiCall::CodeGenGetTransposeType(const Tensor &inputs, const Tensor &outputs,
                                                 AutoFuseTransposeType &transpose_type) const {
  std::vector<uint32_t> output_vectorized_axis;
  /* 在输入tensor的axis中搜索输出的axis索引，生成对应的permute向量 */
  for (auto output_axis : outputs.vectorized_axis) {
    auto pos = find(inputs.vectorized_axis.begin(), inputs.vectorized_axis.end(), output_axis);
    if (pos != inputs.vectorized_axis.end()) {
      output_vectorized_axis.emplace_back(std::distance(inputs.vectorized_axis.begin(), pos));
    }
  }

  /* 将permute Tensor转化为TransposeType */
  PermuteParam permute_param;
  auto it = kPermutationTable.find(output_vectorized_axis);
  if (it != kPermutationTable.end()) {
    permute_param = it->second;
    transpose_type = permute_param.true_transpose_type;
  } else {
    transpose_type = AutoFuseTransposeType::TRANSPOSE_INVALID;
    std::ostringstream oss;
    for (size_t i = 0; i < output_vectorized_axis.size(); ++i) {
      if (i != 0) oss << ", ";
      oss << output_vectorized_axis[i];
    }
    GELOGE(af::FAILED, "Transpose convert permute to transposetype failed, sizes = %zu, permute = [%s]",
           output_vectorized_axis.size(), oss.str().c_str());
    return af::FAILED;
  }
  return af::SUCCESS;
}

Status TransposeApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  GELOGI("Tensor::DtypeName(y.dtype) == %s", dtype_name.c_str());
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "TransposeApiCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, {}, tpipe.tmp_buf.name + "_" + std::to_string(id));

  /* 将permute转化为transposeType */
  stringstream ss;

  /* 计算transposeType */
  AutoFuseTransposeType transpose_type;
  GE_ASSERT_SUCCESS(CodeGenGetTransposeType(x, y, transpose_type));
  std::vector<std::string> transTypeValue = {"TRANSPOSE_ND2ND_ONLY", "TRANSPOSE_ND2ND_102", "TRANSPOSE_ND2ND_0213",
                                             "TRANSPOSE_ND2ND_2103", "TRANSPOSE_ND2ND_021", "TRANSPOSE_ND2ND_210",
                                             "TRANSPOSE_ND2ND_0321", "TRANSPOSE_INVALID"};
  GE_ASSERT_TRUE(static_cast<uint8_t>(transpose_type) < transTypeValue.size());
  ss << "AutoFuseTransposeType transposeType = AutoFuseTransposeType::"
     << transTypeValue[static_cast<uint8_t>(transpose_type)] << ";" << std::endl;

  /* 获取TilingData */
  uint32_t tiling_case_id = tpipe.tiler.GetTilingCaseId();
  std::string apiTilingDataString = "t->" + this->api_tiling_data_field + "_" + std::to_string(tiling_case_id);
  ss << "auto apiTilingData = " << apiTilingDataString << ";" << std::endl;
  ss << "codegen::ConfusionTranspose<" << dtype_name << ">" << "(" << y << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
     << ", "
     << "transposeType, apiTilingData);" << std::endl;
  ss << "AscendC::PipeBarrier<PIPE_ALL>();" << std::endl;

  result = ss.str();
  return af::SUCCESS;
}
Status TransposeApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_ASSERT_SUCCESS(GetApiTilingTypeName(node, this->device_api_tiling_data_type));
  this->host_api_tiling_data_type = "optiling::" + this->device_api_tiling_data_type;

  /* 只有非Transpose节点才异常，当前处理位于Tranpose节点下，无需判断异常 */
  GE_ASSERT_SUCCESS(GetApiTilingFieldName(node, this->api_tiling_data_field));
  GELOGD("TilingData parse success, device_type:%s, host_type:%s, name:%s", this->device_api_tiling_data_type.c_str(),
         this->host_api_tiling_data_type.c_str(), this->api_tiling_data_field.c_str());
  return af::SUCCESS;
}

static ApiCallRegister<TransposeApiCall> register_transpose_api_call("TransposeApiCall");
}  // namespace codegen
