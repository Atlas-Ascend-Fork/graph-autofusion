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

constexpr uint32_t kAlignWidth = 32U;
constexpr int32_t kSoftmaxExtraMul = 8;

static AscGraphAttr *GetOrCreateSoftmaxGraphAttrsGroup(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
  auto attr = graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
  return attr;
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcSoftmaxTmpSizeV2(const AscNode &node) {
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_desc;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  if (node_outputs[0].attr.vectorized_strides.empty()) {
    return tmp_buf_desc;
  }

  const auto dtype_size = ge::GetSizeByDataType(node_inputs[0].attr.dtype);
  GE_ASSERT_TRUE(dtype_size > 0, "Softmax node %s[%s] invalid dtype size:%d, expect > 0, dtype:%d", node.GetTypePtr(),
                 node.GetNamePtr(), dtype_size, static_cast<int32_t>(node_inputs[0].attr.dtype));
  const uint32_t align_size = kAlignWidth / static_cast<uint32_t>(dtype_size);

  auto attr = GetOrCreateSoftmaxGraphAttrsGroup(node.GetOwnerComputeGraph());

  Expression a_exp = Symbol(1);
  Expression r_exp = Symbol(1);
  const size_t num_axes = node_outputs[0].attr.vectorized_strides.size();
  for (size_t i = 0; i < num_axes; i++) {
    uint64_t vectorized_axis_id = node_outputs[0].attr.vectorized_axis[i];
    Expression axis_size = attr->axis[vectorized_axis_id]->size;
    if (i == num_axes - 1) {
      r_exp = sym::Align(axis_size, align_size);
    } else {
      a_exp = sym::Mul(a_exp, axis_size);
    }
  }

  Expression element_size = sym::Add(sym::Mul(a_exp, r_exp), sym::Mul(a_exp, Symbol(kSoftmaxExtraMul)));
  Expression tmp_size = sym::Mul(element_size, Symbol(dtype_size));
  GELOGD("Softmax node %s[%s] temp buffer size: %s bytes", node.GetTypePtr(), node.GetNamePtr(), tmp_size.Str().get());
  TmpBufDesc desc = {tmp_size, -1};
  tmp_buf_desc.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_desc;
}

}  // namespace ascir
}  // namespace af
