/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "schedule_graph_validator.h"

#include <algorithm>
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
  // axis、repeats 和 strides 描述同一组维度，三者必须按位置一一对应。
  if (axes.size() != repeats.size() || axes.size() != strides.size()) {
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
    if (graph_axes.count(axis) == 0U) {
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
  for (const auto axis : tensor.attr.vectorized_axis) {
    if (seen.count(axis) == 0U) {
      std::ostringstream detail;
      detail << kind << "[" << index << "] vectorized axis id " << axis << " is absent from tensor axes";
      return Fail("AXIS_SET", node, detail.str());
    }
  }
  return af::SUCCESS;
}

/**
 * @brief 校验节点调度轴是否属于图轴集合及输出视图。
 *
 * 空调度轴允许待回填；Transpose 的置换由专用校验负责，Cube 的归约调度轴允许不出现在输出中。
 */
af::Status ValidateNodeAxisContract(af::AscNode &node, const std::unordered_set<af::AxisId> &graph_axes) {
  // 空调度轴可能仍处于待回填状态；没有输出时也没有可用于比较的输出轴集合。
  if (node.attr.sched.axis.empty() || node.outputs().empty()) {
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
  // Transpose 的实际输入输出轴集合及置换由专用校验负责。
  // Cube 的调度域包含归约轴，例如 MatMul 调度 [M,N,K]、输出 [M,N]，不能要求输出保留 K。
  if (node.attr.api.compute_type == af::ComputeType::kComputeTranspose ||
      node.attr.api.compute_type == af::ComputeType::kComputeCube) {
    return af::SUCCESS;
  }
  // 普通节点的调度轴必须出现在每个输出 Tensor 中。
  // Transpose 后的 Cast/Store 可以沿用原调度轴顺序，不能要求其与 Tensor view 顺序一致。
  for (size_t output_index = 0; output_index < node.outputs().size(); ++output_index) {
    const auto &output_axes = node.outputs[output_index].attr.axis;
    for (size_t schedule_index = 0; schedule_index < node.attr.sched.axis.size(); ++schedule_index) {
      const auto axis = node.attr.sched.axis[schedule_index];
      const auto iter = std::find(output_axes.begin(), output_axes.end(), axis);
      if (iter == output_axes.end()) {
        std::ostringstream detail;
        detail << "schedule axis id " << axis << " at position " << schedule_index << " is absent from output["
               << output_index << "]";
        return Fail("AXIS_SET", node, detail.str());
      }
    }
  }
  return af::SUCCESS;
}

/**
 * @brief 校验 Reduce 是否能够从输入输出 view 元数据推导出 R 轴。
 *
 * 直接使用 Tensor 轴校验形状，不依赖 sched.axis 回填；归约轴输出 repeat 为 1，非归约轴长度保持不变。
 */
af::Status ValidateReduce(af::AscNode &node) {
  if (node.attr.api.compute_type != af::ComputeType::kComputeReduce) {
    return af::SUCCESS;
  }
  if (node.GetType() == "Softmax") {
    return af::SUCCESS;
  }
  if (node.inputs().empty() || node.outputs().empty()) {
    return Fail("REDUCE_AXIS", node, "reduce requires at least one input and one output tensor");
  }
  const auto &input = node.inputs[0].attr;
  const auto &output = node.outputs[0].attr;
  const auto &axes = input.axis;
  // 比较的是 AF 内部 view 的轴数量，不是每个轴的长度；归约轴长度变成 1 不代表删除该轴。
  if (input.axis.size() != output.axis.size()) {
    std::ostringstream detail;
    detail << "reduce axis count mismatch: input[0] axes=" << input.axis.size()
           << " output[0] axes=" << output.axis.size() << "; keep reduced axes in the output view";
    return Fail("REDUCE_AXIS", node, detail.str());
  }
  // Reduce 不改变轴身份或顺序；只有对应位置是同一个轴，后续长度和 stride 比较才有意义。
  for (size_t index = 0; index < axes.size(); ++index) {
    if (input.axis[index] != output.axis[index]) {
      std::ostringstream detail;
      detail << "reduce input/output axis IDs and order must match: position=" << index
             << " input axis=" << input.axis[index] << " output axis=" << output.axis[index];
      return Fail("REDUCE_AXIS", node, detail.str());
    }
  }
  // Tensor 元数据已由 ValidateTensor 校验；调度轴非空时额外检查其 rank。
  // 调度轴尚未填写不能使下面的形状检查被跳过，也不应因其长度为零误报。
  if (!node.attr.sched.axis.empty() && node.attr.sched.axis.size() != axes.size()) {
    std::ostringstream detail;
    detail << "cannot derive R axis: input rank=" << input.strides.size() << " output rank=" << output.strides.size()
           << " schedule rank=" << node.attr.sched.axis.size();
    return Fail("REDUCE_AXIS", node, detail.str());
  }
  // 无轴的标量没有可比较的维度，由现有 ScalarTo1DTensorPass 完成标量调度表示。
  if (axes.empty()) {
    return af::SUCCESS;
  }
  size_t reduce_count = 0U;
  bool has_unknown_reduce_axis = false;
  // 普通 R 轴通过 stride 变化识别；长度为 1 的轴也必须具有输出零 stride，才能作为退化候选。
  for (size_t index = 0; index < axes.size(); ++index) {
    const auto output_is_zero = af::SymbolicUtils::StaticCheckEq(output.strides[index], af::sym::kSymbolZero);
    const auto same_stride = af::SymbolicUtils::StaticCheckEq(input.strides[index], output.strides[index]);
    const auto input_is_one = af::SymbolicUtils::StaticCheckEq(input.repeats[index], af::sym::kSymbolOne);
    const bool output_is_reduced = output_is_zero == af::TriBool::kTrue && same_stride == af::TriBool::kFalse;
    const bool degenerate_reduce = output_is_zero == af::TriBool::kTrue && input_is_one == af::TriBool::kTrue;
    if (output_is_reduced || degenerate_reduce) {
      // 归约会把该轴上的多个元素合成一个，因此输出 repeat 必须为 1。
      // 例如 [4, 8] 沿第二轴归约应得到 [4, 1]；只把 stride 写成 0 而保留 repeat=8 仍是非法结果。
      // 仅在静态可证明不等于 1 时拒绝；符号表达式无法确定时不误报，也不改写前端元数据。
      if (af::SymbolicUtils::StaticCheckEq(output.repeats[index], af::sym::kSymbolOne) == af::TriBool::kFalse) {
        std::ostringstream detail;
        detail << "reduce output repeat must be 1: axis=" << output.axis[index] << " position=" << index
               << " input[0].repeats=" << input.repeats[index].Str().get()
               << " output[0].repeats=" << output.repeats[index].Str().get() << " expected=1";
        return Fail("REDUCE_AXIS", node, detail.str());
      }
      ++reduce_count;
    } else {
      // 无法证明是否为 R 轴时保留未知状态，不能据此要求轴长度不变或断言图中没有 R 轴。
      const bool may_reduce = output_is_zero != af::TriBool::kFalse &&
                              (same_stride != af::TriBool::kTrue || input_is_one != af::TriBool::kFalse);
      if (may_reduce) {
        has_unknown_reduce_axis = true;
        continue;
      }
      // 例如 [4,8] 沿第二轴归约只能得到 [4,1]，不能把未归约的第一轴改成 5。
      if (af::SymbolicUtils::StaticCheckEq(input.repeats[index], output.repeats[index]) == af::TriBool::kFalse) {
        std::ostringstream detail;
        detail << "non-reduced axis length changed: axis=" << axes[index] << " position=" << index
               << " input[0].repeats=" << input.repeats[index].Str().get()
               << " output[0].repeats=" << output.repeats[index].Str().get();
        return Fail("REDUCE_AXIS", node, detail.str());
      }
    }
  }
  if (reduce_count == 0U && !has_unknown_reduce_axis) {
    return Fail("REDUCE_AXIS", node,
                "cannot find an R axis from input/output strides; check reduce axis and output view metadata");
  }
  return af::SUCCESS;
}

/**
 * @brief 校验 Transpose 的真实输入输出轴是否构成完整置换，且同一轴的长度保持不变。
 *
 * 通过输入数据锚点获取实际生产者及其输出端口，避免仅按节点邻接关系误取输入轴。
 */
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
  const auto &input_axes = input.axis;
  const auto &output_axes = output.axis;
  if (input_axes.size() != output_axes.size()) {
    std::ostringstream detail;
    detail << "transpose rank mismatch: input=" << input_axes.size() << " output=" << output_axes.size();
    return Fail("TRANSPOSE_AXIS", node, detail.str());
  }
  // 长度检查在按位置读取 repeats 前完成，生产者可能是未单独校验的 Buffer 节点。
  if (input.repeats.size() != input_axes.size() || output.repeats.size() != output_axes.size()) {
    return Fail("TENSOR_METADATA", node, "transpose input/output axis and repeats lengths differ");
  }
  // Transpose 只能改变轴顺序，不能新增、删除或重复使用轴。
  std::unordered_map<af::AxisId, size_t> input_positions;
  for (size_t index = 0; index < input_axes.size(); ++index) {
    input_positions.emplace(input_axes[index], index);
  }
  std::unordered_set<af::AxisId> output_set;
  for (const auto axis : output_axes) {
    if (!output_set.emplace(axis).second || input_positions.count(axis) == 0U) {
      std::ostringstream detail;
      detail << "transpose output axis id " << axis << " is not a permutation of the real input axes";
      return Fail("TRANSPOSE_AXIS", node, detail.str());
    }
  }
  if (input_positions.size() != output_set.size()) {
    return Fail("TRANSPOSE_AXIS", node, "transpose input/output axes are not a complete permutation");
  }
  // 按轴 ID 对齐后比较长度，不能直接比较相同下标：例如 [A=4, B=8] -> [B=8, A=4] 是合法转置。
  // 仅比较元素总数也不够；同一轴的长度发生变化，即使总数不变，也不是合法转置。
  for (size_t output_index = 0; output_index < output_axes.size(); ++output_index) {
    const auto axis = output_axes[output_index];
    const auto input_index = input_positions.at(axis);
    // 符号表达式只有在静态可证明不相等时才拒绝，未知关系暂不拒绝。
    if (af::SymbolicUtils::StaticCheckEq(input.repeats[input_index], output.repeats[output_index]) ==
        af::TriBool::kFalse) {
      std::ostringstream detail;
      detail << "transpose axis length changed: axis=" << axis << " input position=" << input_index
             << " output position=" << output_index << " input[0].repeats=" << input.repeats[input_index].Str().get()
             << " output[0].repeats=" << output.repeats[output_index].Str().get()
             << "; transpose must preserve the length of each axis";
      return Fail("TRANSPOSE_AXIS", node, detail.str());
    }
  }
  return af::SUCCESS;
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
