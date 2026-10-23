/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "softmax_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/checker.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
namespace semantic = ascgen_utils::template_semantic;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status SoftmaxApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()));
  GE_ASSERT_SUCCESS(SetSemanticFeature(semantic::kSoftmaxAr));
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, dtype_name), "Codegen(softmax) get data type:%d failed",
                    static_cast<int32_t>(x.dtype));

  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "SoftmaxApiCall cannot find tmp buffer id to use.");
  id = it->second;
  std::string tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);

  std::stringstream a_actual;
  std::stringstream r_actual;
  a_actual << "uint32_t a_actual = 1";
  r_actual << "uint32_t r_actual = 1";
  const size_t num_axes = x.vectorized_axis.size();
  for (size_t i = 0; i < num_axes; ++i) {
    const auto axis = tpipe.tiler.GetAxis(x.vectorized_axis[i]);
    if (i == num_axes - 1) {
      r_actual << " * " << axis.actual_size;
    } else {
      a_actual << " * " << axis.actual_size;
    }
  }

  std::stringstream ss;
  ss << "{" << std::endl;
  ss << a_actual.str() << ";" << std::endl;
  ss << r_actual.str() << ";" << std::endl;
  ss << this->api_name_ << "<" << dtype_name << ">(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y)
     << "], " << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tmp_buf_name << ", "
     << "{a_actual, r_actual});" << std::endl;
  ss << "}" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<SoftmaxApiCall> register_softmax_api_call("SoftmaxApiCall");

}  // namespace codegen
