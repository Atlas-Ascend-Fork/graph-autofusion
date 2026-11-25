/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "binary_api_call_v2.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen_api_param/codegen_api_param.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status BinaryApiCallV2::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                      const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  auto x1 = inputs[0].get();
  auto x2 = inputs[1].get();
  auto y = outputs[0].get();
  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);
  api_param->api_name = api_name_;
  api_param->input_params.emplace_back(
      x1.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x1)));
  api_param->input_params.emplace_back(
      x2.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x2)));
  api_param->output_params.emplace_back(
      y.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y)));
  api_param->cal_count = CombinedExprFactory::SymbolVar(x1.actual_size.Str());

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "Register codegen api param for node[%s] failed",
                    this->node_name.c_str());
  return af::SUCCESS;
}

static ApiCallRegister<BinaryApiCallV2> register_binary_v2_api_call("BinaryApiCallV2");
}  // namespace codegen
