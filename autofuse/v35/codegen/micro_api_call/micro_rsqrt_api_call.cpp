/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"

#include "micro_rsqrt_api_call.h"
#include "ascir_ops.h"
#include "optimize/pre_process/pre_process_config.h"

namespace codegen {
Status MicroRsqrtApiCall::Generate(const TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                   CallParam &param, std::string &result) {
  GE_ASSERT_TRUE(this->inputs_.size() == 1, "Rsqrt micro api call must have one input");
  GE_ASSERT_TRUE(this->outputs_.size() == 1, "Rsqrt micro api call must have one output");

  const auto *input_tensor = tensor_mng.GetTensor(this->inputs_[0].second);
  const auto *output_tensor = tensor_mng.GetTensor(this->outputs_[0].second);
  GE_ASSERT_NOTNULL(input_tensor);
  GE_ASSERT_NOTNULL(output_tensor);

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(input_tensor->dtype_, dtype_name), "Get data type:%d failed",
                    static_cast<int32_t>(input_tensor->dtype_));
  const auto &input_name = input_tensor->name;
  const auto &output_name = output_tensor->name;
  GE_ASSERT_TRUE(input_tensor->dtype_ == output_tensor->dtype_,
                 "Rsqrt input tensor[%s] dtype:%d and output tensor[%s] dtype:%d must match", input_name.c_str(),
                 static_cast<int32_t>(input_tensor->dtype_), output_name.c_str(),
                 static_cast<int32_t>(output_tensor->dtype_));

  const std::string one_name = output_name + "_rsqrt_one";

  std::stringstream ss;
  // 3510/5102 的公开 MicroAPI 没有提供 Rsqrt，使用相同命名空间下的基础指令组合保持 VF 计算。
  ss << "AscendC::MicroAPI::RegTensor<" << dtype_name << "> " << one_name << ";" << std::endl;
  ss << "AscendC::MicroAPI::Duplicate(" << one_name << ", static_cast<" << dtype_name << ">(1.0), " << param.p_reg
     << ");" << std::endl;
  ss << "AscendC::MicroAPI::Sqrt(" << output_name << ", " << input_name << ", " << param.p_reg << ");" << std::endl;
  ss << "AscendC::MicroAPI::Div";
  if (input_tensor->dtype_ == ge::DT_FLOAT) {
    ss << "<" << dtype_name;
    if (!af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist(af::ascir_op::Rsqrt::Type)) {
      ss << ", &high_precision_div_mode";
    }
    ss << ">";
  }
  ss << "(" << output_name << ", " << one_name << ", " << output_name << ", " << param.p_reg << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static MicroApiCallRegister<MicroRsqrtApiCall> register_micro_rsqrt_api_call("MicroRsqrtApiCall");
}  // namespace codegen
