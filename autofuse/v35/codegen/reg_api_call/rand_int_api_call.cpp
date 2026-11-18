/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rand_int_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
// 生成标量/张量二态输入的调用参数：标量场景直接取标量值，张量场景按向量化偏移寻址
Status GenRandIntParam(const Tensor &tensor, const TPipe &tpipe, const vector<ascir::AxisId> &current_axis,
                       string &param) {
  const bool is_scalar_scene = (tensor.is_constant) || (tensor.is_ub_scalar && tensor.need_gen_get_value_of_ub_scalar);
  if (is_scalar_scene) {
    string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(tensor.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(tensor.dtype));
    param = tensor.need_gen_get_value_of_ub_scalar ? ("(" + dtype_name + ")" + tensor.ub_scalar_name) : tensor.Str();
    return af::SUCCESS;
  }
  param = tensor.Str() + "[" + tpipe.tiler.TensorVectorizedOffset(current_axis, tensor) + "]";
  return af::SUCCESS;
}
}  // namespace

Status RandIntApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                std::string &result) const {
  const auto &x1 = inputs[0].get();
  const auto &x2 = inputs[1].get();
  const auto &x3 = inputs[2].get();
  const auto &x4 = inputs[3].get();
  const auto &y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "RandIntApiCall cannot find tmp buffer id to use.");
  const int64_t id = it->second;

  // size/cal_count 语义为本块输出元素数，取 y 而非 x1（seed 标量化后 actual_size 为 1）
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(y.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  // x2 为 Arange 产出的逐元素位置流，恒按张量寻址
  string x1_param;
  string x3_param;
  string x4_param;
  GE_CHK_STATUS_RET(GenRandIntParam(x1, tpipe, current_axis, x1_param));
  GE_CHK_STATUS_RET(GenRandIntParam(x3, tpipe, current_axis, x3_param));
  GE_CHK_STATUS_RET(GenRandIntParam(x4, tpipe, current_axis, x4_param));
  stringstream ss;
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1_param
     << ", " << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << x3_param << ", "
     << x4_param << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ", " << y.actual_size << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<RandIntApiCall> register_rand_int_api_call("RandIntApiCall");

}  // namespace codegen
