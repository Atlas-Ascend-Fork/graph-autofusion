/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unary_api_tmp_call.h"

#include <sstream>
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;

Status UnaryApiTmpCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "UnaryApiTmpCall cannot find tmp buffer id to use.");
  id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  stringstream ss;
  if (IsCVFusionStage(this->api_call_context) || tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    // CV callbacks receive stage dimensions, not the generic kernel's tiling pointer.
    // Match UnaryApiCall's physical row layout while retaining the scratch buffer argument.
    const std::string output_stride = GenBlockAlignNExpr(y, "curAivN");
    const bool input_is_cube_output =
        tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && x.id == tpipe.cube_output_tensor_id;
    const std::string input_stride = input_is_cube_output ? "curAlignN" : GenBlockAlignNExpr(x, "curAivN");
    ss << y.actual_size << " = curAivM * " << output_stride << ";" << std::endl;
    const auto flat_condition = GenCvUbFuseFlatApiCondition(tpipe, inputs, y);
    if (!flat_condition.empty()) {
      ss << "if (" << flat_condition << ") {" << std::endl;
      ss << "  " << this->api_name_ << "(" << y << "[0], " << x << "[0], " << tpipe.tmp_buf << "_" << id
         << ", ConvertToUint32(curAivM * curAivN));" << std::endl;
      ss << "} else {" << std::endl;
    }
    ss << "for (uint32_t cv_row = 0; cv_row < ConvertToUint32(curAivM); cv_row++) {" << std::endl;
    ss << "  " << this->api_name_ << "(" << y << "[cv_row * ConvertToUint32(" << output_stride << ")], " << x
       << "[cv_row * ConvertToUint32(" << input_stride << ")], " << tpipe.tmp_buf << "_" << id
       << ", ConvertToUint32(curAivN));" << std::endl;
    ss << "}" << std::endl;
    if (!flat_condition.empty()) {
      ss << "}" << std::endl;
    }
    result = ss.str();
    return af::SUCCESS;
  }
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << tpipe.tmp_buf << "_" << std::to_string(id)
     << ", " << GetCVAlignedSize(this->api_call_context, y, x.actual_size.Str()) << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryApiTmpCall> register_unary_api_tmp_call("UnaryApiTmpCall");
}  // namespace codegen
