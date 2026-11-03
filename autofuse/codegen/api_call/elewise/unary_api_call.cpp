/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unary_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status UnaryApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                              std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x.actual_size.Str()));

  stringstream ss;
  if (IsCVFusionStage(this->api_call_context) || tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    // CV UBFuse 下 flat 1D 调用的 count 取自 stage 尺寸, 无法覆盖按输出 dtype 对齐的二维布局
    // (如 bool 输入 N=4: 行距32, 总量 M*32, stage 仅 M*8), 尾部行读到未初始化数据。
    // 按 per-dtype 物理行距逐行调用, 输出按输出 dtype 行距落位, 与 stage 写出/下游消费侧行距一致。
    const std::string output_stride = GenBlockAlignNExpr(y, "curAivN");
    const bool input_is_cube_output =
        tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && x.id == tpipe.cube_output_tensor_id;
    const std::string input_stride = input_is_cube_output ? "curAlignN" : GenBlockAlignNExpr(x, "curAivN");
    ss << y.actual_size << " = curAivM * " << output_stride << ";" << std::endl;
    ss << "for (uint32_t cv_row = 0; cv_row < ConvertToUint32(curAivM); cv_row++) {" << std::endl;
    ss << "  " << this->api_name_ << "(" << y << "[cv_row * ConvertToUint32(" << output_stride << ")], " << x
       << "[cv_row * ConvertToUint32(" << input_stride << ")], ConvertToUint32(curAivN));" << std::endl;
    ss << "}" << std::endl;
    result = ss.str();
    return af::SUCCESS;
  }
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << x.actual_size << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<UnaryApiCall> register_unary_api_call("UnaryApiCall");
}  // namespace codegen
