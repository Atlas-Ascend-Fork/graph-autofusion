/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule_graph_validator.h"

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ascir_ops_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "schedule_utils.h"

namespace optimize {
namespace {
thread_local std::string g_schedule_validation_error;

/**
 * @brief 记录节点级校验失败，并生成统一格式的 Schedule 诊断信息。
 *
 * 同一次请求只保留首个根因，但每次调用都会写入日志，便于定位失败位置。
 */
af::Status Fail(const char *rule, const af::AscNode &node, const std::string &detail) {
  if (g_schedule_validation_error.empty()) {
    std::ostringstream message;
    message << kScheduleGraphValidationErrorPrefix << "[" << rule
            << "] stage=schedule graph=" << node.GetOwnerComputeGraph()->GetName() << " node=" << node.GetName()
            << " op=" << node.GetType() << ": " << detail;
    g_schedule_validation_error = message.str();
  }
  GELOGE(af::FAILED, "%s", g_schedule_validation_error.c_str());
  return af::FAILED;
}

/**
 * @brief 校验 Tensor 的轴、repeat、stride 和向量化元数据。
 *
 * 校验长度一致性、轴唯一性、轴是否属于当前图，以及向量化轴是否存在于 Tensor 轴集合中。
 */
template <typename Tensor>
af::Status ValidateTensor(const af::AscNode &node, const Tensor &tensor, const char *kind, size_t index,
                          const std::unordered_set<af::AxisId> &graph_axes) {
  const auto &axes = tensor.attr.axis;
  const auto &repeats = tensor.attr.repeats;
  const auto &strides = tensor.attr.strides;
  // Hint 图可能只回填了部分 view；只有三者都填写后才能静态判定长度不一致。
  // 任一字段为空都视为待补齐，延后到后续阶段，避免误报合法 Hint 图。
  if (!axes.empty() && !repeats.empty() && !strides.empty() &&
      (axes.size() != repeats.size() || axes.size() != strides.size())) {
    std::ostringstream detail;
    detail << kind << "[" << index << "] metadata lengths differ: axis=" << axes.size() << " repeats=" << repeats.size()
           << " strides=" << strides.size();
    return Fail("TENSOR_METADATA", node, detail.str());
  }

  std::unordered_set<af::AxisId> seen;
  // 轴 ID 是 Tensor 元数据中的唯一标识，同一个 Tensor 内不能重复出现。
  for (size_t position = 0; position < axes.size(); ++position) {
    const auto axis = axes[position];
    if (!seen.emplace(axis).second) {
      std::ostringstream detail;
      detail << kind << "[" << index << "] contains duplicate axis id " << axis << " at position " << position;
      return Fail("AXIS_SET", node, detail.str());
    }
    // 图轴清单尚未建立时不具备静态判定条件，保留未知并延后校验。
    if (!graph_axes.empty() && graph_axes.count(axis) == 0U) {
      std::ostringstream detail;
      detail << kind << "[" << index << "] references unknown axis id " << axis;
      return Fail("AXIS_SET", node, detail.str());
    }
  }

  // 调度前可以只提供向量化轴，stride 由后续对齐阶段推导；已填写的 stride 必须与轴一一对应。
  // 无论 stride 是否已填写，向量化轴都必须来自普通轴列表。
  if (!tensor.attr.vectorized_strides.empty() &&
      tensor.attr.vectorized_axis.size() != tensor.attr.vectorized_strides.size()) {
    std::ostringstream detail;
    detail << kind << "[" << index
           << "] vectorized metadata lengths differ: axis=" << tensor.attr.vectorized_axis.size()
           << " strides=" << tensor.attr.vectorized_strides.size();
    return Fail("TENSOR_METADATA", node, detail.str());
  }
  std::unordered_set<af::AxisId> vectorized_axes;
  for (const auto axis : tensor.attr.vectorized_axis) {
    if (!vectorized_axes.emplace(axis).second) {
      std::ostringstream detail;
      detail << kind << "[" << index << "] contains duplicate vectorized axis id " << axis;
      return Fail("AXIS_SET", node, detail.str());
    }
    if (seen.count(axis) == 0U) {
      std::ostringstream detail;
      detail << kind << "[" << index << "] vectorized axis id " << axis << " is absent from tensor axes";
      return Fail("AXIS_SET", node, detail.str());
    }
  }
  return af::SUCCESS;
}

/**
 * @brief 校验节点调度轴是否属于图轴集合，且同一节点内不重复。
 *
 * 空调度轴允许待回填；已填写的调度轴必须来自当前图。
 */
af::Status ValidateNodeAxisContract(af::AscNode &node, const std::unordered_set<af::AxisId> &graph_axes) {
  // 空调度轴可能仍处于待回填状态。已填写的调度轴必须来自当前图，且同一节点内不能重复。
  if (node.attr.sched.axis.empty()) {
    return af::SUCCESS;
  }
  std::unordered_set<af::AxisId> schedule_axes;
  // 调度轴必须来自当前图，并且同一个节点的调度轴列表不能重复。
  for (const auto axis : node.attr.sched.axis) {
    if (graph_axes.count(axis) == 0U) {
      std::ostringstream detail;
      detail << "schedule axis id " << axis << " is not defined by the graph";
      return Fail("AXIS_SET", node, detail.str());
    }
    if (!schedule_axes.emplace(axis).second) {
      std::ostringstream detail;
      detail << "schedule axis list contains duplicate axis id " << axis;
      return Fail("AXIS_SET", node, detail.str());
    }
  }
  return af::SUCCESS;
}

af::Status ValidateReduceAxis(af::AscNode &node, const af::AscTensorAttr &input, const af::AscTensorAttr &output,
                              size_t input_index, size_t output_index) {
  const auto axis = output.axis[output_index];
  const auto repeat_is_one = af::SymbolicUtils::StaticCheckEq(output.repeats[output_index], af::sym::kSymbolOne);
  const auto repeat_same = af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], output.repeats[output_index]);
  // 长度明确变为 1 时，输出在该轴上只有下标 0，stride 不影响寻址。
  if (repeat_is_one == af::TriBool::kTrue && repeat_same == af::TriBool::kFalse) {
    return af::SUCCESS;
  }
  const auto output_is_zero = af::SymbolicUtils::StaticCheckEq(output.strides[output_index], af::sym::kSymbolZero);
  const auto same_stride = af::SymbolicUtils::StaticCheckEq(input.strides[input_index], output.strides[output_index]);
  const auto input_is_one = af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], af::sym::kSymbolOne);
  const bool output_is_reduced = output_is_zero == af::TriBool::kTrue && same_stride == af::TriBool::kFalse;
  const bool degenerate_reduce = output_is_zero == af::TriBool::kTrue && input_is_one == af::TriBool::kTrue;
  // 既有图也用 output repeat 为 0 描述归约轴，兼容该表示。
  const bool legacy_reduced =
      af::SymbolicUtils::StaticCheckEq(output.repeats[output_index], af::sym::kSymbolZero) == af::TriBool::kTrue;
  if (output_is_reduced || degenerate_reduce || legacy_reduced) {
    // 归约轴的输出长度存在多种合法表示：1、0（旧表示）或保留输入长度。
    // 只有三者都无法静态匹配时才判定为错误，符号值和未知关系一律放行。
    const auto repeat_is_zero = af::SymbolicUtils::StaticCheckEq(output.repeats[output_index], af::sym::kSymbolZero);
    if (repeat_is_one == af::TriBool::kFalse && repeat_is_zero == af::TriBool::kFalse &&
        repeat_same == af::TriBool::kFalse) {
      std::ostringstream detail;
      detail << "reduce output repeat must be 1: axis=" << axis << " position=" << output_index
             << " input[0].repeats=" << input.repeats[input_index].Str().get()
             << " output[0].repeats=" << output.repeats[output_index].Str().get() << " expected=1";
      return Fail("REDUCE_AXIS", node, detail.str());
    }
    return af::SUCCESS;
  }
  // 归约轴无法从 stride 静态判定时保留未知状态，交给后续调度阶段处理。
  const bool may_reduce = output_is_zero != af::TriBool::kFalse &&
                          (same_stride != af::TriBool::kTrue || input_is_one != af::TriBool::kFalse);
  if (may_reduce) {
    return af::SUCCESS;
  }
  if (repeat_same == af::TriBool::kFalse) {
    std::ostringstream detail;
    detail << "non-reduced axis length changed: axis=" << axis << " position=" << output_index
           << " input[0].repeats=" << input.repeats[input_index].Str().get()
           << " output[0].repeats=" << output.repeats[output_index].Str().get();
    return Fail("REDUCE_AXIS", node, detail.str());
  }
  return af::SUCCESS;
}

af::Status ValidateReduceShape(af::AscNode &node, const af::AscTensorAttr &input, const af::AscTensorAttr &output) {
  // Hint 图可能只回填了部分 view；长度不一致时不能按下标读取，统一延后到后续阶段。
  if (input.axis.size() != input.repeats.size() || input.axis.size() != input.strides.size() ||
      output.axis.size() != output.repeats.size() || output.axis.size() != output.strides.size()) {
    return af::SUCCESS;
  }
  std::unordered_map<af::AxisId, size_t> input_positions;
  for (size_t index = 0; index < input.axis.size(); ++index) {
    input_positions.emplace(input.axis[index], index);
  }
  for (size_t output_index = 0; output_index < output.axis.size(); ++output_index) {
    const auto input_iter = input_positions.find(output.axis[output_index]);
    // 归约后新增的轴按未知处理：可能是外部前端补充的 size-1 轴，交后续调度确认。
    if (input_iter == input_positions.end()) {
      continue;
    }
    GE_CHK_STATUS_RET(ValidateReduceAxis(node, input, output, input_iter->second, output_index), "%s",
                      GetScheduleValidationError().c_str());
  }
  return af::SUCCESS;
}

af::Status ValidateReduce(af::AscNode &node) {
  if (node.attr.api.compute_type != af::ComputeType::kComputeReduce) {
    return af::SUCCESS;
  }
  // Softmax 及其变体在调度内部使用归约语义，但输出 view 不遵循普通 Reduce 契约。
  if (node.GetType().find("Softmax") != std::string::npos) {
    return af::SUCCESS;
  }
  if (node.inputs().empty() || node.outputs().empty()) {
    return Fail("REDUCE_AXIS", node, "reduce requires at least one input and one output tensor");
  }
  const auto &input = node.inputs[0].attr;
  const auto &output = node.outputs[0].attr;
  if (input.axis.empty()) {
    return af::SUCCESS;
  }
  return ValidateReduceShape(node, input, output);
}

/**
 * @brief 校验 Transpose 的真实输入输出轴是否构成完整置换，且同一轴的长度保持不变。
 *
 * 通过输入数据锚点获取实际生产者及其输出端口，避免仅按节点邻接关系误取输入轴。
 */
af::Status ValidateTransposeAxisLength(af::AscNode &node, const af::AscTensorAttr &input,
                                       const af::AscTensorAttr &output, size_t input_index, size_t output_index) {
  const auto axis = output.axis[output_index];
  if (af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], output.repeats[output_index]) ==
      af::TriBool::kFalse) {
    std::ostringstream detail;
    detail << "transpose axis length changed: axis=" << axis << " input position=" << input_index
           << " output position=" << output_index << " input[0].repeats=" << input.repeats[input_index].Str().get()
           << " output[0].repeats=" << output.repeats[output_index].Str().get()
           << "; transpose must preserve the length of each axis";
    return Fail("TRANSPOSE_AXIS", node, detail.str());
  }
  return af::SUCCESS;
}

af::Status ValidateTransposeAddedAxis(af::AscNode &node, const af::AscTensorAttr &output, size_t output_index) {
  // Transpose 可以补充 size-1 轴；非 1 长度无法从输入解释，静态可判定为错误。
  const auto repeat_is_one = af::SymbolicUtils::StaticCheckEq(output.repeats[output_index], af::sym::kSymbolOne);
  const auto repeat_is_zero = af::SymbolicUtils::StaticCheckEq(output.repeats[output_index], af::sym::kSymbolZero);
  if (repeat_is_one == af::TriBool::kFalse && repeat_is_zero == af::TriBool::kFalse) {
    std::ostringstream detail;
    detail << "transpose adds axis with length != 1: axis=" << output.axis[output_index] << " position=" << output_index
           << " output[0].repeats=" << output.repeats[output_index].Str().get();
    return Fail("TRANSPOSE_AXIS", node, detail.str());
  }
  return af::SUCCESS;
}

af::Status ValidateTransposeDroppedAxis(af::AscNode &node, const af::AscTensorAttr &input, size_t input_index) {
  // Transpose 可以丢弃 size-1 轴；非 1 长度被丢弃时同样静态可判定为错误。
  const auto repeat_is_one = af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], af::sym::kSymbolOne);
  const auto repeat_is_zero = af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], af::sym::kSymbolZero);
  if (repeat_is_one == af::TriBool::kFalse && repeat_is_zero == af::TriBool::kFalse) {
    std::ostringstream detail;
    detail << "transpose drops axis with length != 1: axis=" << input.axis[input_index] << " position=" << input_index
           << " input[0].repeats=" << input.repeats[input_index].Str().get();
    return Fail("TRANSPOSE_AXIS", node, detail.str());
  }
  return af::SUCCESS;
}

af::Status ValidateTransposeShape(af::AscNode &node, const af::AscTensorAttr &input, const af::AscTensorAttr &output) {
  // Hint 图可能只回填了部分 view；长度不一致时不能按下标读取，统一延后到后续阶段。
  if (input.axis.size() != input.repeats.size() || output.axis.size() != output.repeats.size()) {
    return af::SUCCESS;
  }
  std::unordered_map<af::AxisId, size_t> input_positions;
  for (size_t index = 0; index < input.axis.size(); ++index) {
    input_positions.emplace(input.axis[index], index);
  }
  std::unordered_set<af::AxisId> output_axes;
  for (size_t output_index = 0; output_index < output.axis.size(); ++output_index) {
    output_axes.emplace(output.axis[output_index]);
    const auto input_iter = input_positions.find(output.axis[output_index]);
    if (input_iter == input_positions.end()) {
      GE_CHK_STATUS_RET(ValidateTransposeAddedAxis(node, output, output_index), "%s",
                        GetScheduleValidationError().c_str());
      continue;
    }
    GE_CHK_STATUS_RET(ValidateTransposeAxisLength(node, input, output, input_iter->second, output_index), "%s",
                      GetScheduleValidationError().c_str());
  }
  for (size_t input_index = 0; input_index < input.axis.size(); ++input_index) {
    if (output_axes.count(input.axis[input_index]) != 0U) {
      continue;
    }
    GE_CHK_STATUS_RET(ValidateTransposeDroppedAxis(node, input, input_index), "%s",
                      GetScheduleValidationError().c_str());
  }
  return af::SUCCESS;
}

af::Status ValidateTranspose(af::AscNode &node) {
  if (node.attr.api.compute_type != af::ComputeType::kComputeTranspose) {
    return af::SUCCESS;
  }
  if (node.outputs().empty()) {
    return Fail("TRANSPOSE_AXIS", node, "transpose has no output tensor");
  }
  const auto input_anchor = node.GetInDataAnchor(0);
  if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
    return Fail("TRANSPOSE_AXIS", node, "transpose input[0] has no connected producer output anchor");
  }
  const auto producer_anchor = input_anchor->GetPeerOutAnchor();
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_anchor->GetOwnerNode());
  if (producer == nullptr || producer_anchor->GetIdx() < 0 ||
      static_cast<size_t>(producer_anchor->GetIdx()) >= producer->outputs().size()) {
    return Fail("TRANSPOSE_AXIS", node, "transpose input[0] producer output anchor is out of range");
  }
  const auto &input = producer->outputs[static_cast<size_t>(producer_anchor->GetIdx())].attr;
  const auto &output = node.outputs[0].attr;
  return ValidateTransposeShape(node, input, output);
}
}  // namespace

void ClearScheduleValidationError() {
  g_schedule_validation_error.clear();
}

const std::string &GetScheduleValidationError() {
  return g_schedule_validation_error;
}

void RecordScheduleValidationError(const std::string &message) {
  if (g_schedule_validation_error.empty()) {
    g_schedule_validation_error = message;
  }
  GELOGE(af::FAILED, "%s", g_schedule_validation_error.c_str());
}

af::Status ValidateScheduleGraph(const af::AscGraph &graph) {
  // 每次开始新的图校验时清理上一次请求的诊断，避免错误信息串到下一次调度请求。
  ClearScheduleValidationError();

  // 先建立图级轴集合。后续 Tensor 和节点的轴引用都必须能在这个集合中找到。
  std::unordered_set<af::AxisId> graph_axes;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis == nullptr) {
      RecordScheduleValidationError(std::string(kScheduleGraphValidationErrorPrefix) +
                                    "[AXIS_SET] stage=schedule graph=" + graph.GetName() +
                                    ": graph contains a null axis definition");
      return af::FAILED;
    }
    if (!graph_axes.emplace(axis->id).second) {
      RecordScheduleValidationError(std::string(kScheduleGraphValidationErrorPrefix) +
                                    "[AXIS_SET] stage=schedule graph=" + graph.GetName() +
                                    ": graph contains duplicate axis id " + std::to_string(axis->id));
      return af::FAILED;
    }
  }

  // 按节点逐个检查。空节点和 Buffer 节点不参与计算 Tensor 的调度校验。
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    // Data/Scalar 等 Buffer 的输入视图允许尚未补齐，由 Load/广播处理其计算表示。
    // Reduce、Transpose 依赖输入视图判断形状，仍要求其输入元数据完整。
    const bool requires_input_view = node->attr.api.compute_type == af::ComputeType::kComputeReduce ||
                                     node->attr.api.compute_type == af::ComputeType::kComputeTranspose;
    for (size_t input_index = 0; input_index < node->inputs().size(); ++input_index) {
      const auto producer = std::dynamic_pointer_cast<af::AscNode>(node->inputs[input_index].anchor.GetOwnerNode());
      if (!requires_input_view && producer != nullptr && ScheduleUtils::IsBuffer(producer)) {
        continue;
      }
      GE_CHK_STATUS_RET(ValidateTensor(*node, node->inputs[input_index], "input", input_index, graph_axes), "%s",
                        GetScheduleValidationError().c_str());
    }

    // 再检查所有输出 Tensor，输出也必须具备同样完整且可引用的元数据。
    for (size_t output_index = 0; output_index < node->outputs().size(); ++output_index) {
      GE_CHK_STATUS_RET(ValidateTensor(*node, node->outputs[output_index], "output", output_index, graph_axes), "%s",
                        GetScheduleValidationError().c_str());
    }

    // 基础元数据通过后，检查节点调度轴，再检查 Reduce 和 Transpose 的专用语义。
    GE_CHK_STATUS_RET(ValidateNodeAxisContract(*node, graph_axes), "%s", GetScheduleValidationError().c_str());
    GE_CHK_STATUS_RET(ValidateReduce(*node), "%s", GetScheduleValidationError().c_str());
    GE_CHK_STATUS_RET(ValidateTranspose(*node), "%s", GetScheduleValidationError().c_str());
  }
  return af::SUCCESS;
}

}  // namespace optimize
