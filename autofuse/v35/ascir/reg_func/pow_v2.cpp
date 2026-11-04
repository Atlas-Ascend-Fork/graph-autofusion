/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
std::vector<std::unique_ptr<TmpBufDesc>> CalcPowTmpSizeV2(const AscNode &node) {
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
  if (IsAllScalarOrUbScalar(node_inputs)) {
    // 当两个输入都为scalar，或都为ub scalar，或一个scalar，一个ub scalar时，需要额外分配2个blockSize的tmp buffer；
    // f32/bf16非特判指数走Power DOUBLE_FLOAT_TECH，scratch从tmp_buf前两块（src1_buf/dst_buf复用区）之后偏移
    // 传入（pow.h中left_tmp_buf），scratch需求3段*4B*alignCount（block_cnt(f32=8/bf16=16)对齐后恒384B），需一并分配
    uint32_t pow_scratch_size = 0U;
    if (data_type == ge::DT_FLOAT || data_type == ge::DT_BF16) {
      constexpr uint32_t POW_SCRATCH_SEG = 3U;     // tmpLHI/tmpLLO/tmpLog三段
      constexpr uint32_t POW_SCRATCH_ALIGN = 32U;  // alignCount = (calCount+31)/32*32，block_cnt<=32时恒为32
      constexpr uint32_t FLOAT_SIZE = 4U;          // DOUBLE_FLOAT_TECH中间量为float
      pow_scratch_size = POW_SCRATCH_SEG * FLOAT_SIZE * POW_SCRATCH_ALIGN;
    }
    const Expression total_size = Symbol(data_type_size) * input_size + Symbol(32 * 2) + Symbol(pow_scratch_size);
    GELOGD("Node %s[%s] inputs are all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  } else {
    uint32_t max_live_node_cnt = 0U;
    constexpr uint32_t POW_DOUBLE = 2U;
    constexpr uint32_t POW_TRIPLE = 3U;
    // f32三倍tmp_buf，f16六倍tmp_buf
    if (data_type == ge::DT_INT32) {
      max_live_node_cnt = 0U;
    } else if (data_type_size == sizeof(float)) {
      max_live_node_cnt = POW_TRIPLE;
    } else {
      max_live_node_cnt = POW_DOUBLE * POW_TRIPLE;
    }

    // 乘以存活节点系数之前，做一次32B对齐
    const Expression total_size = sym::Align(Symbol(data_type_size) * input_size, 32) * Symbol(max_live_node_cnt);

    GELOGD("Node %s[%s] inputs are not all scalar or ub scalar, return size= %s.", node.GetTypePtr(), node.GetNamePtr(),
           total_size.Str().get());
    return GetTmpBuffer(total_size);
  }
}
}  // namespace ascir
}  // namespace af
