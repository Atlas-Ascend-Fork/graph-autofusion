/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "reg_func/default_reg_func.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {
std::vector<std::unique_ptr<TmpBufDesc>> CalcExp2TmpSizeV2(const AscNode &node) {
  uint32_t max_live_node_cnt = 0U;
  constexpr uint32_t POW_DOUBLE = 2U;
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0U, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  const auto input_size = GetInputSize(node_inputs);
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    input_id = node_inputs.Size() - 1U;
  }

  const auto data_type = node_inputs[input_id].attr.dtype;
  const auto data_type_size = GetSizeByDataType(data_type);
  GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), input_id,
         data_type_size);
  // f32双倍tmp_buf，f16四倍tmp_buf
  if (data_type == ge::DT_INT32) {
    max_live_node_cnt = 0U;
  } else if (data_type_size == sizeof(float)) {
    max_live_node_cnt = POW_DOUBLE;
  } else {
    max_live_node_cnt = POW_DOUBLE * POW_DOUBLE;
  }

  // 乘以存活节点系数之前，做一次32B对齐
  const Expression total_size = sym::Align(Symbol(data_type_size) * input_size, 32) * Symbol(max_live_node_cnt);

  GELOGD("Node %s[%s] inputs are not all scalar or ub scalar, return size= %s bytes.", node.GetTypePtr(),
         node.GetNamePtr(), total_size.Str().get());
  return GetTmpBuffer(total_size);
}
}  // namespace ascir
}  // namespace af
