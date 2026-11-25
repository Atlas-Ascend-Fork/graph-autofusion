/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_reg_func_v2.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {
bool GetTensorVectorizedRepeats(const AscTensorAttr &attr, std::vector<Expression> &output_vectorized_repeats) {
  for (size_t i = 0; i < attr.vectorized_axis.size(); i++) {
    auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i]);
    GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID:%ld in vectorized_axis, node tensor valid axis size:%zu",
                   attr.vectorized_axis[i], attr.axis.size());
    auto axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
    output_vectorized_repeats.emplace_back(attr.repeats[axis_id]);
  }
  return true;
}
std::vector<std::unique_ptr<TmpBufDesc>> CalcTransposeTmpSizeV2(const AscNode &node) {
  auto node_outputs = node.outputs;
  auto &attr = node_outputs[0].attr;
  std::vector<Expression> output_vectorized_repeats;
  GE_ASSERT_TRUE(GetTensorVectorizedRepeats(attr, output_vectorized_repeats));
  GE_ASSERT_TRUE(!output_vectorized_repeats.empty(), "output_vectorized_repeats is empty for node %s",
                 node.GetNamePtr());

  Expression input_size = output_vectorized_repeats.back();
  for (int32_t i = attr.vectorized_axis.size() - 2; i >= 0; i--) {
    af::Expression inner_axis_stride = output_vectorized_repeats[i + 1] * attr.vectorized_strides[i + 1];
    if (SymbolicUtils::StaticCheckEq(inner_axis_stride, attr.vectorized_strides[i]) == TriBool::kTrue) {
      input_size = input_size * output_vectorized_repeats[i];
    }
  }
  const auto data_type_size = GetSizeByDataType(attr.dtype);
  const Expression total_size = Symbol(data_type_size) * input_size;
  GELOGD("Node %s[%s] temp buffer size: %s bytes", node.GetTypePtr(), node.GetNamePtr(), total_size.Str().get());
  TmpBufDesc desc = {total_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}
}  // namespace ascir
}  // namespace af
