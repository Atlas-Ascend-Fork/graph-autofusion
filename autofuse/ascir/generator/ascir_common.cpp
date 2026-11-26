/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir_common.h"
#include "symbolizer/symbolic_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"

namespace af {
namespace ascir {
namespace {
bool IsScalarInputType(const std::string &node_type) {
  return node_type == "Scalar" || node_type == "ScalarData";
}
}  // namespace

bool OnlySecondInputSupportScalar(const std::vector<bool> &is_scalar_list) {
  GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
  return !is_scalar_list[0] && is_scalar_list[1];
}

[[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionFromDtypeMap(
    const AscNode &node, const std::map<ge::DataType, ge::DataType> &dtype_conversion_map) {
  std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  for (size_t i = 0; i < node_inputs().size(); i++) {
    auto it = dtype_conversion_map.find(node_inputs[i].attr.dtype);
    if (it != dtype_conversion_map.end()) {
      conversion_dtype.first.emplace_back(it->second);  // 使用迭代器访问
    } else {
      conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
    }
  }
  for (size_t i = 0; i < node_outputs().size(); i++) {
    auto it = dtype_conversion_map.find(node_outputs[i].attr.dtype);
    if (it != dtype_conversion_map.end()) {
      conversion_dtype.second.emplace_back(it->second);  // 使用迭代器访问
    } else {
      conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
    }
  }
  return conversion_dtype;
}

bool IsAllVecAxisContinuous(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  for (size_t i = 0; i < node_inputs().size(); i++) {
    if (node_inputs[i].attr.vectorized_axis.size() == 1) {
      continue;
    }
    auto &attr = node_inputs[i].attr;
    for (size_t j = 1; j < attr.vectorized_axis.size(); j++) {
      auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[j]);
      GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID in node: %s input %zu vectorized_axis: %zu",
                     node.GetName(), i, j);
      auto axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
      Expression cur_axis_stride = attr.repeats[axis_id] * attr.vectorized_strides[j];
      if (SymbolicUtils::StaticCheckEq(cur_axis_stride, attr.vectorized_strides[j - 1]) != TriBool::kTrue) {
        return false;
      }
    }
  }
  for (size_t i = 0; i < node_outputs().size(); i++) {
    if (node_outputs[i].attr.vectorized_axis.size() == 1) {
      continue;
    }
    auto &attr = node_outputs[i].attr;
    for (size_t j = 1; j < attr.vectorized_axis.size(); j++) {
      auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[j]);
      GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID in node: %s output %zu vectorized_axis: %zu",
                     node.GetName(), i, j);
      auto axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
      Expression cur_axis_stride = attr.repeats[axis_id] * attr.vectorized_strides[j];
      if (SymbolicUtils::StaticCheckEq(cur_axis_stride, attr.vectorized_strides[j - 1]) != TriBool::kTrue) {
        return false;
      }
    }
  }
  return true;
}

bool IsUBScalarTensor(const AscTensor &tensor) {
  auto &attr = tensor.attr;
  uint64_t axis_id = UINT64_MAX;
  for (size_t i = 0; i < attr.vectorized_axis.size(); i++) {
    auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i]);
    GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID in vectorized_axis");
    axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
    if (SymbolicUtils::StaticCheckEq(attr.repeats[axis_id], Symbol(1)) != TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool IsVectorizedAxisSupportBrc(const AscNode &node, size_t input_id, const BroadcastCapability &broadcast_capability) {
  // 如果是vectorized_axis轴存在不相等，则必须是ub_scalar或node支持brc_inline，且input_id在support_brc_list中才认为合法。
  AscNodeInputs node_inputs = node.inputs;
  if ((IsUBScalarTensor(node_inputs[input_id]) || broadcast_capability.support_inline_brc) &&
      (std::find(broadcast_capability.scalar_inputs.begin(), broadcast_capability.scalar_inputs.end(), input_id) !=
       broadcast_capability.scalar_inputs.end())) {
    return true;
  }
  return false;
}

Status ValidateInputTensorLoopAxis(const AscNode &node, size_t input_id, size_t input_axis_id, size_t output_id) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  auto input_attr = node_inputs[input_id].attr;
  auto output_attr = node_outputs[output_id].attr;

  auto it = std::find(output_attr.axis.begin(), output_attr.axis.end(), input_attr.axis[input_axis_id]);
  if (it == output_attr.axis.end()) {
    // 诊断日志：打印完整 input/output view，便于定位生产者与消费者 view 不一致的来源
    std::string input_axis_str;
    for (const auto &axis_id : input_attr.axis) {
      input_axis_str += std::to_string(axis_id) + ",";
    }
    std::string output_axis_str;
    for (const auto &axis_id : output_attr.axis) {
      output_axis_str += std::to_string(axis_id) + ",";
    }
    GELOGD(
        "ValidateInputTensorLoopAxis:[Diag] Node %s[%s] input tensor[%zu] axis[%s] mismatch with "
        "output tensor[%zu] axis[%s], input vec axis size=%zu.",
        node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_str.c_str(), output_id, output_axis_str.c_str(),
        input_attr.vectorized_axis.size());
  }
  GE_ASSERT_TRUE(it != output_attr.axis.end(),
                 "Node %s[%s]: input tensor %zu loop axis %zu is not in output tensor "
                 "axis",
                 node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id);
  auto output_axis_id = static_cast<uint64_t>(std::distance(output_attr.axis.begin(), it));
  if ((SymbolicUtils::StaticCheckEq(output_attr.repeats[output_axis_id], input_attr.repeats[input_axis_id]) ==
       TriBool::kTrue) ||
      (SymbolicUtils::StaticCheckEq(input_attr.repeats[input_axis_id], Symbol(1)) == TriBool::kTrue)) {
    return af::SUCCESS;
  } else if (SymbolicUtils::StaticCheckEq(output_attr.repeats[output_axis_id], input_attr.repeats[input_axis_id]) ==
             TriBool::kUnknown) {
    GELOGW(
        "Node %s[%s]: input tensor %zu loop axis %zu repeat %s and output tensor %zu loop axis %zu repeat %s may not "
        "be equal or broadcastable(relation cannot be determined)",
        node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id, input_attr.repeats[input_axis_id].Str().get(),
        output_id, output_axis_id, output_attr.repeats[output_axis_id].Str().get());
    return af::SUCCESS;
  }

  GELOGE(af::FAILED,
         "Node %s[%s]: input tensor %zu loop axis %zu repeat %s and output tensor 0 loop axis %zu repeat "
         "%s are not equal or broadcastable",
         node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id, input_attr.repeats[input_axis_id].Str().get(),
         output_axis_id, output_attr.repeats[output_axis_id].Str().get());
  return af::FAILED;
}

Status ValidateInputTensorVectorizedAxis(const AscNode &node, size_t input_id, size_t input_axis_id, size_t output_id,
                                         const BroadcastCapability &broadcast_capability) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  auto input_attr = node_inputs[input_id].attr;
  auto output_attr = node_outputs[output_id].attr;

  auto it = std::find(output_attr.axis.begin(), output_attr.axis.end(), input_attr.axis[input_axis_id]);
  GE_ASSERT_TRUE(it != output_attr.axis.end(),
                 "Node %s[%s]: input tensor %zu vectorized_axis %zu is not in output "
                 "tensor axis",
                 node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id);
  auto output_axis_id = static_cast<uint64_t>(std::distance(output_attr.axis.begin(), it));
  if ((SymbolicUtils::StaticCheckEq(output_attr.repeats[output_axis_id], input_attr.repeats[input_axis_id]) ==
       TriBool::kTrue) ||
      IsVectorizedAxisSupportBrc(node, input_id, broadcast_capability)) {
    return af::SUCCESS;
  } else if (SymbolicUtils::StaticCheckEq(output_attr.repeats[output_axis_id], input_attr.repeats[input_axis_id]) ==
             TriBool::kUnknown) {
    GELOGW(
        "Node %s[%s]: input tensor %zu vectorized_axis %zu repeat: %s and output tensor 0 vectorized_axis %zu "
        "repeat: %s may not be equal or broadcastable(relation cannot be determined)",
        node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id, input_attr.repeats[input_axis_id].Str().get(),
        output_axis_id, output_attr.repeats[output_axis_id].Str().get());
    return af::SUCCESS;
  }

  GELOGE(af::FAILED,
         "Node %s[%s]: input tensor %zu vectorized_axis %zu repeat: %s and output tensor %zu vectorized_axis "
         "%zu repeat: %s are not equal or broadcastable",
         node.GetTypePtr(), node.GetNamePtr(), input_id, input_axis_id, input_attr.repeats[input_axis_id].Str().get(),
         output_id, output_axis_id, output_attr.repeats[output_axis_id].Str().get());
  return af::FAILED;
}

Status ValidateShapeConsistencyWithOutput(const AscNode &node, size_t output_id,
                                          const BroadcastCapability &broadcast_capability) {
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  GE_ASSERT_TRUE(output_id < node_outputs().size(),
                 "Node %s[%s]: output id %zu is out of range, output tensor size is %zu", node.GetTypePtr(),
                 node.GetNamePtr(), output_id, node_outputs().size());
  GE_ASSERT_TRUE(!node_outputs[output_id].attr.vectorized_axis.empty(),
                 "Node %s[%s]: output tensor %zu has empty vectorized_axis", node.GetTypePtr(), node.GetNamePtr(),
                 output_id);

  for (size_t i = 0; i < node_inputs().size(); i++) {
    auto &input = node_inputs[i];
    // 如果输入tensor的向量化轴为空，则不考虑该tensor，在是否支持Scalar输入节点的流程中进行处理。
    // 仅处理能够通过向量化轴获取tensor大小的情况。
    if (input.attr.vectorized_axis.empty()) {
      continue;
    }
    for (size_t j = 0; j < input.attr.repeats.size(); j++) {
      if (std::find(input.attr.vectorized_axis.begin(), input.attr.vectorized_axis.end(), input.attr.axis[j]) !=
          input.attr.vectorized_axis.end()) {
        GE_ASSERT_SUCCESS(
            ValidateInputTensorVectorizedAxis(node, i, j, output_id, broadcast_capability),
            "Node %s[%s]: input tensor %zu axis %zu validate output %zu vectorized_axis consistency failed",
            node.GetTypePtr(), node.GetNamePtr(), i, j, output_id);
      } else {
        GE_ASSERT_SUCCESS(ValidateInputTensorLoopAxis(node, i, j, output_id),
                          "Node %s[%s]: input tensor %zu axis %zu validate output %zu loop axis consistency failed",
                          node.GetTypePtr(), node.GetNamePtr(), i, j, output_id);
      }
    }
  }
  return af::SUCCESS;
}

Status ValidateShapeConsistencyWithSingleOutput(const AscNode &node, const BroadcastCapability &broadcast_capability) {
  AscNodeOutputs node_outputs = node.outputs;
  GE_ASSERT_TRUE(node_outputs().size() == 1U, "Node %s[%s]: output tensor size is not equal to 1", node.GetTypePtr(),
                 node.GetNamePtr());
  return ValidateShapeConsistencyWithOutput(node, 0U, broadcast_capability);
}

bool IsNodeHasScalarInput(const AscNode &node) {
  AscNodeInputs node_inputs = node.inputs;
  for (size_t i = 0; i < node_inputs().size(); i++) {
    if (IsScalarInputType(node.GetInDataNodes().at(i)->GetType())) {
      return true;
    }
  }
  return false;
}

bool IsNodeFirstInputScalar(const AscNode &node) {
  return IsScalarInputType(node.GetInDataNodes().at(0)->GetType());
}

bool IsScalarInput(const std::vector<af::Expression> &repeats) {
  return std::all_of(repeats.begin(), repeats.end(), [](const af::Expression &repeat) {
    return SymbolicUtils::StaticCheckEq(repeat, af::Symbol(1)) == TriBool::kTrue;
  });
}

bool IsAnyInputNodeScalar(const AscNode &node) {
  for (const auto &in_anchor : node.GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor());
    GE_ASSERT_NOTNULL(in_anchor->GetPeerOutAnchor()->GetOwnerNode());
    auto in_node = std::dynamic_pointer_cast<af::AscNode>(in_anchor->GetPeerOutAnchor()->GetOwnerNode());
    int32_t output_idx = in_anchor->GetPeerOutAnchor()->GetIdx();
    auto repeats = in_node->outputs[output_idx].attr.repeats;
    if (IsScalarInputType(in_node->GetType()) || IsScalarInput(repeats)) {
      return true;
    }
  }
  return false;
}
}  // namespace ascir
}  // namespace af
