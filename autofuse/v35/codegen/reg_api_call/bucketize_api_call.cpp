/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "bucketize_api_call.h"

#include <sstream>
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"
#include "common/checker.h"

namespace codegen {
using namespace std;

Status BucketizeApiCall::ParseAttr(const ascir::NodeView &node) {
  right_ = false;
  if (node->attr.ir_attr != nullptr) {
    (void)node->attr.ir_attr->GetAttrValue("right", right_);
  }
  return af::SUCCESS;
}

Status BucketizeApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  constexpr size_t kValueIndex = 0U;
  constexpr size_t kBoundaryIndex = 1U;
  GE_ASSERT_TRUE(inputs.size() >= 2U, "BucketizeApiCall expects values and boundaries");
  GE_ASSERT_TRUE(!outputs.empty(), "BucketizeApiCall expects an int32 output");
  auto values = inputs[kValueIndex].get();
  auto boundaries = inputs[kBoundaryIndex].get();
  auto y = outputs[0].get();

  int64_t life_time_axis_id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "BucketizeApiCall cannot find tmp buffer id to use.");
  const int64_t id = it->second;

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs,
                               CombinedExprFactory::SymbolVar(values.actual_size.Str()),
                               tpipe.tmp_buf.name + "_" + std::to_string(id));

  stringstream ss;
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << values
     << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, values) << "], " << boundaries << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, boundaries) << "], " << tpipe.tmp_buf << "_"
     << std::to_string(id) << ", " << values.actual_size << ", " << boundaries.actual_size << ", "
     << (right_ ? "true" : "false") << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<BucketizeApiCall> register_bucketize_api_call("BucketizeApiCall");
}  // namespace codegen
