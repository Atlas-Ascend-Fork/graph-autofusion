/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <algorithm>
#include "default_reg_func.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace af {
namespace ascir {
constexpr uint32_t DEFAULT_TEMP_BUFFER_SIZE = 8192;
constexpr int32_t ONE_BLK_SIZE = 32;
constexpr int32_t ONE_REPEAT_BYTE_SIZE = 256;
constexpr int32_t MAX_REPEAT_NUM = 255;

std::vector<std::unique_ptr<TmpBufDesc>> GetTmpBuffer(const Expression &tmp_size) {
  auto valid_tmp_size = sym::Min(tmp_size, Symbol(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE + ONE_BLK_SIZE));
  GELOGD("Get temp buffer size: %s", valid_tmp_size.Str().get());
  TmpBufDesc desc = {valid_tmp_size, -1};
  std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
  tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmp_buf_descs;
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcDefaultTmpSize(const AscNode &node) {
  GELOGD("Node %s[%s] default temp buffer size: %u", node.GetTypePtr(), node.GetNamePtr(), DEFAULT_TEMP_BUFFER_SIZE);
  return GetTmpBuffer(Symbol(DEFAULT_TEMP_BUFFER_SIZE));
}

uint32_t GetNonScalarAxisId(AscNodeInputs &node_inputs) {
  for (uint32_t i = 0U; i < node_inputs.Size(); i++) {
    // 排除掉Scalar节点
    if (node_inputs[i].attr.vectorized_axis.empty()) {
      GELOGD("Index %u of the input is scalar.", i);
      continue;
    }
    // 判断向量化轴Strides不是全0，则认为是正常tensor，将其返回
    if (!std::all_of(node_inputs[i].attr.vectorized_strides.begin(), node_inputs[i].attr.vectorized_strides.end(),
                     [](const Expression &repeat) {
                       return SymbolicUtils::StaticCheckEq(repeat, sym::kSymbolZero) == TriBool::kTrue;
                     })) {
      GELOGD("Index %u of the input is non-one tensor.", i);
      return i;
    }
  }
  // 如果没有tensor，则返回无效值
  GELOGD("Not found non-one input tensor.");
  return UINT32_MAX;
}

Expression GetInputSize(AscNodeInputs &node_inputs) {
  const uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    GELOGD("All input is scalar, return size=1.");
    return Symbol(1);
  }
  auto &attr = node_inputs[input_id].attr;
  uint64_t axis_id = UINT64_MAX;
  uint64_t vectorized_axis_id = UINT64_MAX;
  for (size_t i = 0; i < attr.vectorized_axis.size(); i++) {
    if (SymbolicUtils::StaticCheckEq(attr.vectorized_strides[i], sym::kSymbolZero) != TriBool::kTrue) {
      vectorized_axis_id = static_cast<uint64_t>(i);
      auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i]);
      GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID in vectorized_axis");
      axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
      break;
    }
  }
  GELOGD("[GetInputSize] axis id is: %lu", axis_id);
  GELOGD("[GetInputSize] vectorized_axis id is: %lu", vectorized_axis_id);
  GELOGD("[GetInputSize] inputs[0].repeat is: %s", attr.repeats[axis_id].Str().get());
  GELOGD("[GetInputSize] inputs[0].vectorized_strides is: %s", attr.vectorized_strides[0].Str().get());
  GE_ASSERT_TRUE(vectorized_axis_id < static_cast<uint64_t>(attr.vectorized_strides.size()),
                 "Vectorized_axis id [%lu] is invalid", vectorized_axis_id);
  GE_ASSERT_TRUE(axis_id < static_cast<uint64_t>(attr.repeats.size()), "Axis id [%lu] is invalid", axis_id);
  Expression input_size = attr.repeats[axis_id] * attr.vectorized_strides[vectorized_axis_id];
  return input_size;
}

std::vector<std::unique_ptr<TmpBufDesc>> GetInputDataSizeTmpBuffer(const AscNode &node) {
  auto node_inputs = node.inputs;
  GE_ASSERT_TRUE(node_inputs.Size() > 0U, "Node %s[%s] inputs size is 0.", node.GetTypePtr(), node.GetNamePtr());
  const auto input_size = GetInputSize(node_inputs);
  uint32_t input_id = GetNonScalarAxisId(node_inputs);
  if (input_id == UINT32_MAX) {
    input_id = node_inputs.Size() - 1U;
  }
  const auto data_type_size = GetSizeByDataType(node_inputs[input_id].attr.dtype);
  GELOGD("Node %s[%s] inputs[%u] data type size is: %d", node.GetTypePtr(), node.GetNamePtr(), input_id,
         data_type_size);
  const Expression total_size = Symbol(data_type_size) * input_size;
  return GetTmpBuffer(total_size);
}

std::vector<std::unique_ptr<TmpBufDesc>> CalcBinaryApiTmpSize(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  // second input is scalar
  if (node_inputs[1].attr.repeats.empty()) {
    GELOGD("Node %s[%s] input[%u] repeats is empty", node.GetTypePtr(), node.GetNamePtr(), 1);
    return CalcDefaultTmpSize(node);
  }

  // second input is ub_scalar
  bool is_ub_scalar = true;
  for (uint32_t i = 0; i < node_inputs[1].attr.repeats.size(); i++) {
    GELOGD("Node %s[%s] input[%u] repeat[%u] is : %s", node.GetTypePtr(), node.GetNamePtr(), 1, i,
           node_inputs[1].attr.repeats[i].Str().get());
    if (SymbolicUtils::StaticCheckEq(node_inputs[1].attr.repeats[i], sym::kSymbolOne) != TriBool::kTrue) {
      is_ub_scalar = false;
      break;
    }
  }
  if (is_ub_scalar) {
    GELOGD("Node %s[%s] input[%u] is ub scalar", node.GetTypePtr(), node.GetNamePtr(), 1);
    return CalcDefaultTmpSize(node);
  }
  GELOGD("Node %s[%s] input[%u] is tensor", node.GetTypePtr(), node.GetNamePtr(), 1);
  // second input is tensor
  std::vector<std::unique_ptr<TmpBufDesc>> tmpBufDescs;
  auto tmp_size = Symbol(32);
  TmpBufDesc desc = {tmp_size, -1};
  tmpBufDescs.emplace_back(std::make_unique<TmpBufDesc>(desc));
  return tmpBufDescs;
}

bool IsAllScalarOrUbScalar(AscNodeInputs &node_inputs) {
  for (uint32_t i = 0; i < node_inputs.Size(); i++) {
    for (uint32_t j = 0; j < node_inputs[i].attr.repeats.size(); j++) {
      if (SymbolicUtils::StaticCheckEq(node_inputs[i].attr.repeats[j], Symbol(1)) != TriBool::kTrue) {
        return false;
      }
    }
  }
  return true;
}

bool HasScalarOrUbScalar(AscNodeInputs &node_inputs) {
  for (uint32_t i = 0; i < node_inputs.Size(); i++) {
    for (uint32_t j = 0; j < node_inputs[i].attr.repeats.size(); j++) {
      if (SymbolicUtils::StaticCheckEq(node_inputs[i].attr.repeats[j], Symbol(1)) == TriBool::kTrue) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace ascir
}  // namespace af
