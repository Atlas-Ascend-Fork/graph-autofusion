/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rand_seed_offset_api_call.h"

#include <sstream>

#include "api_call/utils/api_call_factory.h"
#include "ascir_ops.h"
#include "codegen/expression_convert_struct.h"
#include "common/checker.h"
#include "common/ge_common/debug/log.h"
#include "common_utils.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;

namespace {
Status GenRandSeedParam(const Tensor &tensor, const TPipe &tpipe, const vector<ascir::AxisId> &current_axis,
                        string &param) {
  const bool is_scalar_scene = tensor.is_constant || (tensor.is_ub_scalar && tensor.need_gen_get_value_of_ub_scalar);
  if (is_scalar_scene) {
    param = tensor.need_gen_get_value_of_ub_scalar ? "(int64_t)" + tensor.ub_scalar_name : tensor.Str();
    return af::SUCCESS;
  }
  param = tensor.Str() + "[" + tpipe.tiler.TensorVectorizedOffset(current_axis, tensor) + "]";
  return af::SUCCESS;
}

bool UsesLegacySimt(const string &op_type, const ge::DataType output_dtype) {
  if (op_type == "Randn") {
    return output_dtype != ge::DT_UINT32 && output_dtype != ge::DT_INT32;
  }
  return output_dtype == ge::DT_FLOAT16 || output_dtype == ge::DT_BF16;
}
}  // namespace

Status RandSeedOffsetApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                       const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                       std::string &result) const {
  GE_CHK_BOOL_RET_STATUS(outputs.size() == 1U, af::FAILED,
                         "RandSeedOffsetApiCall expects exactly 1 output, but got %zu outputs", outputs.size());
  const auto &y = outputs[0].get();

  stringstream ss;
  if (inputs.empty()) {
    (void)RegisterBasicDumpParam(api_name_, inputs, outputs);
    const std::string legacy_api_name = type == "Randn" ? "Randn" : "Rand";
    if (UsesLegacySimt(type, y.dtype)) {
      GE_ASSERT_SUCCESS(SetSemanticFeature(ascgen_utils::template_semantic::kSimt));
    }
    ss << legacy_api_name << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << y.actual_size << ");" << std::endl;
    result = ss.str();
    return af::SUCCESS;
  }

  GE_CHK_BOOL_RET_STATUS(inputs.size() == 2U, af::FAILED,
                         "RandSeedOffsetApiCall expects no inputs or seed/offset inputs, but got %zu inputs",
                         inputs.size());
  const auto &seed = inputs[0].get();
  const auto &offset = inputs[1].get();

  constexpr int64_t kTmpBufLifeTime = -1L;
  const auto tmp_buf_iter = tmp_buf_id.find(kTmpBufLifeTime);
  GE_ASSERT_TRUE(tmp_buf_iter != tmp_buf_id.end(), "RandSeedOffsetApiCall cannot find tmp buffer id to use.");
  const int64_t tmp_buf_index = tmp_buf_iter->second;

  (void)RegisterBasicDumpParam(api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(y.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(tmp_buf_index));
  GE_ASSERT_SUCCESS(SetSemanticFeature(ascgen_utils::template_semantic::kSimt));

  string seed_param;
  GE_CHK_STATUS_RET(GenRandSeedParam(seed, tpipe, current_axis, seed_param));
  const string api_name = type == "Randn" ? "RandnSeedOffset" : "RandSeedOffset";
  ss << api_name << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << seed_param
     << ", " << offset << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, offset) << "], " << tpipe.tmp_buf.name
     << "_" << std::to_string(tmp_buf_index) << ", " << y.actual_size << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<RandSeedOffsetApiCall> register_rand_seed_offset_api_call("RandSeedOffsetApiCall");

}  // namespace codegen
