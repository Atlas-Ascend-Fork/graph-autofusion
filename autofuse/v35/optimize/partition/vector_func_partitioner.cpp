/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vector_func_partitioner.h"
#include <sstream>
#include <queue>
#include <climits>
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "schedule_utils.h"
#include "common/util/mem_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "platform/common/base_alignment_strategy.h"

namespace {
constexpr uint32_t kMaxInsNum = 30U;
constexpr size_t kMaxIONum = 8UL;
constexpr int32_t kMaxBitWidthGap = 2;
constexpr int64_t kOutLoopAxisId = -1L;
constexpr size_t kMinVfNodesNum = 2UL;
af::Status ValidateUniqueNodeNames(const af::AscGraph &graph) {
  std::unordered_set<std::string> node_names;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_TRUE(node_names.emplace(node->GetName()).second, "VF input graph [%s] contains duplicate node name [%s].",
                   graph.GetName().c_str(), node->GetName().c_str());
  }
  return af::SUCCESS;
}

// Arange 沿 VF 向量化轴是否恒定（全部 vectorized_strides 为 0，即取值轴与 lane 轴
// 不一致，值沿 lane 不变）。
// 此类 Arange 融合进多节点 VF 后走"常量分量"折叠模板，该模板的块偏移/lane 步长
// 参数推导未覆盖 lane 恒定输入（折叠后 mask 序列错位、结果错误）。取值轴与
// 向量化轴一致的 Arange 融合路径不受影响，保持融合。
bool ArangeConstantAlongVectorizedAxes(const af::AscNodePtr &node) {
  for (const auto &output : node->outputs()) {
    const auto &attr = output->attr;
    if (attr.vectorized_axis.empty() || attr.vectorized_strides.empty() ||
        attr.vectorized_axis.size() != attr.vectorized_strides.size()) {
      continue;
    }
    // 找非退化取值轴：stride≠0 且 repeat 不明确为 1（即产出多个不同值）。
    // 全退化或单元素 Arange（如 [1,1]）不产出多个不同值，无 inter-VF 广播
    // 风险，不视为 lane 恒定。
    bool has_multi_value_axis = false;
    for (size_t i = 0; i < attr.axis.size(); ++i) {
      const bool is_zero_stride =
          af::SymbolicUtils::StaticCheckEq(attr.strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue;
      if (is_zero_stride) {
        continue;
      }
      const bool is_size_one =
          af::SymbolicUtils::StaticCheckEq(attr.repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue;
      if (!is_size_one) {
        has_multi_value_axis = true;
        break;
      }
    }
    if (!has_multi_value_axis) {
      continue;
    }
    // 取值轴存在且沿向量化轴 stride 全为 0 → 值沿 lane 不变。
    bool all_vec_zero = true;
    for (const auto &vec_stride : attr.vectorized_strides) {
      if (af::SymbolicUtils::StaticCheckNe(vec_stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
        all_vec_zero = false;
        break;
      }
    }
    if (all_vec_zero) {
      return true;
    }
  }
  return false;
}

// 簇内是否包含"沿向量化轴恒定"的 Arange（此类 Arange 禁止融合，见上）。
bool ClusterContainsLaneConstantArange(const optimize::Cluster &cluster) {
  for (const auto &node : cluster.Nodes()) {
    if (af::ops::IsOps<af::ascir_op::Arange>(node) && ArangeConstantAlongVectorizedAxes(node)) {
      return true;
    }
  }
  return false;
}

// Arange 广播外轴检测: 逻辑 strides 中最后一个非零 stride 为取值轴(lane 轴),
// 其之前或之后尺寸大于 1 的轴若存在零 stride, 表示 Arange 的取值沿该轴广播(值重复),
// 轴合并/flatten 后无法用线性等差序列表达。尺寸为 1 的退化轴 stride 为 0 时不影响取值语义。
// 使用逻辑 strides 而非 vectorized_strides: 调度器轴合并会重写物理视图,
// 但取值语义始终由逻辑 strides 决定。
// 同时覆盖 Pattern A(前缀广播, lane 轴在最后)和 Pattern B(尾轴广播, lane 轴在前缀)。
bool ArangeHasBroadcastOuterAxis(const af::AscNodePtr &node) {
  for (const auto &output : node->outputs()) {
    const auto &attr = output->attr;
    const auto &strides = attr.strides;
    const auto &repeats = attr.repeats;
    if (strides.empty() || strides.size() != repeats.size()) {
      continue;
    }
    // 找到唯一的非零 stride 轴(lane 轴)。
    // Pattern A: lane 轴在最后(如 strides=[0,...,0,1])。
    // Pattern B: lane 轴在前缀(如 strides=[1,0,...,0])。
    for (size_t i = 0; i < strides.size(); ++i) {
      if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
        // 该轴 stride 为 0, 检查是否非退化(尺寸>1)。
        // 非退化的零 stride 轴表示值沿该轴重复, 是广播轴。
        if (af::SymbolicUtils::StaticCheckNe(repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue) {
          return true;
        }
      }
    }
  }
  return false;
}

// 广播 Arange 保留在根图物化时, 输出按轴尺寸展开为连续物理布局;
// 逻辑取值语义仍由 strides 决定, 由 ArangeApiCall 的逻辑偏移使用。
bool MaterializeArangeViewContiguous(const af::AscNodePtr &node) {
  for (const auto &output : node->outputs()) {
    auto &attr = output->attr;
    if (attr.vectorized_axis.empty() || attr.vectorized_strides.empty() ||
        attr.vectorized_axis.size() != attr.vectorized_strides.size()) {
      continue;
    }
    if (attr.axis.size() != attr.repeats.size()) {
      return false;
    }
    for (const auto &axis_id : attr.vectorized_axis) {
      if (std::find(attr.axis.begin(), attr.axis.end(), axis_id) == attr.axis.end()) {
        return false;
      }
    }
    std::vector<af::Expression> flat_strides(attr.vectorized_strides.size());
    af::Expression stride = af::ops::One;
    for (size_t i = attr.vectorized_strides.size(); i > 0UL; --i) {
      flat_strides[i - 1UL] = stride;
      const auto axis_iter = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i - 1UL]);
      const auto &axis_size = attr.repeats[static_cast<size_t>(std::distance(attr.axis.begin(), axis_iter))];
      stride = (i == attr.vectorized_strides.size()) ? axis_size : af::sym::Mul(stride, axis_size);
    }
    attr.vectorized_strides = std::move(flat_strides);
  }
  return true;
}

// 判断 Arange 输出轴中某个位置的轴是否退化(size==1 且 stride==0)。
bool IsDegenerateAxis(const af::AscTensorAttr &attr, size_t index) {
  return af::SymbolicUtils::StaticCheckEq(attr.repeats[index], af::sym::kSymbolOne) == af::TriBool::kTrue &&
         af::SymbolicUtils::StaticCheckEq(attr.strides[index], af::sym::kSymbolZero) == af::TriBool::kTrue;
}

// 前端显式退化轴 Arange -> Broadcast: 前端用多轴退化视图表达通用系数索引
// (如 c0*p0 + c1*p1 拆成两个退化轴 Arange + Broadcast + Add)。
// Arange 只负责物化一维等差序列: 退化轴(size 1, stride 0)对物理布局和逻辑取值
// 均无贡献, 扩维职责由 Broadcast 通用路径(BroadcastExtend)承担, 与退化轴
// Load -> Broadcast 路径一致。
// 支持两种模式:
// Pattern A: Arange 前缀轴退化(如 size=[1,8], stride=[0,1]), 值沿最后轴递增。
// Pattern B: Arange 尾轴退化(如 size=[4,1], stride=[1,0]), 值沿前缀轴递增。
// 两种模式均保持退化视图原样透传; 仅 1D Arange(轴数少于 Broadcast 输出)在合轴前
// 补齐为同轴数退化视图, 缺失轴按 Broadcast 输出位置补 size 1, stride 0。
af::Status NormalizeArangeBroadcastViewsImpl(af::AscGraph &vf_graph) {
  for (const auto &broadcast : vf_graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Broadcast>(broadcast) || broadcast->inputs().size() != 1UL) {
      continue;
    }
    const auto arange = std::dynamic_pointer_cast<af::AscNode>(broadcast->inputs[0].anchor.GetOwnerNode());
    if (arange == nullptr || !af::ops::IsOps<af::ascir_op::Arange>(arange) || arange->outputs().size() != 1UL ||
        arange->GetOutDataNodes().size() != 1UL) {
      continue;
    }
    auto &input = arange->outputs[0].attr;
    const auto &output = broadcast->outputs[0].attr;
    if (input.axis.empty() || input.axis.size() != input.repeats.size() || input.axis.size() != input.strides.size() ||
        output.axis.empty() || output.axis.size() != output.repeats.size() ||
        output.axis.size() != output.strides.size() || input.axis.size() > output.axis.size() ||
        arange->GetInControlNodesSize() != 0UL || arange->GetOutControlNodesSize() != 0UL ||
        broadcast->GetInControlNodesSize() != 0UL || broadcast->GetOutControlNodesSize() != 0UL) {
      return af::FAILED;
    }
    // Broadcast 输出的尾轴必须与 Arange 的尾轴一致。
    if (output.axis.back() != input.axis.back()) {
      return af::FAILED;
    }
    // Broadcast 输出的尾 stride 必须为 1。
    if (af::SymbolicUtils::StaticCheckEq(output.strides.back(), af::sym::kSymbolOne) != af::TriBool::kTrue) {
      return af::FAILED;
    }

    // Arange 输入必须恰好有一个非退化轴(其余全部退化)。
    size_t non_degenerate_count = 0;
    size_t varying_index = 0;
    for (size_t i = 0; i < input.axis.size(); ++i) {
      if (!IsDegenerateAxis(input, i)) {
        ++non_degenerate_count;
        varying_index = i;
      }
    }
    if (non_degenerate_count != 1UL) {
      return af::FAILED;
    }
    // 保存变化轴 id, 避免后续 input.axis 重写后丢失引用。
    const af::AxisId varying_axis_id = input.axis[varying_index];
    // 非退化轴的 stride 必须为 1。
    if (af::SymbolicUtils::StaticCheckEq(input.strides[varying_index], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      return af::FAILED;
    }
    // Arange 的变化轴必须存在于 Broadcast 输出中。
    if (std::find(output.axis.begin(), output.axis.end(), varying_axis_id) == output.axis.end()) {
      return af::FAILED;
    }

    if (input.axis.size() == output.axis.size()) {
      // 已是同轴数退化视图: 原样透传, Broadcast 走通用扩维路径。
      arange->attr.sched.axis = broadcast->attr.sched.axis;
      continue;
    }

    // 1D Arange: 按 Broadcast 输出位置补齐退化轴(size 1, stride 0),
    // 变化轴保持自身 size 和 stride 1。
    std::vector<af::AxisId> padded_axis;
    std::vector<af::Expression> padded_repeats;
    std::vector<af::Expression> padded_strides;
    const auto &varying_size = input.repeats[varying_index];
    for (size_t i = 0; i < output.axis.size(); ++i) {
      padded_axis.push_back(output.axis[i]);
      if (output.axis[i] == varying_axis_id) {
        padded_repeats.push_back(varying_size);
        padded_strides.push_back(af::sym::kSymbolOne);
      } else {
        padded_repeats.push_back(af::sym::kSymbolOne);
        padded_strides.push_back(af::sym::kSymbolZero);
      }
    }
    input.axis = std::move(padded_axis);
    input.repeats = std::move(padded_repeats);
    input.strides = std::move(padded_strides);
    // vectorized 布局按补齐后 repeats 重建为扁平连续布局。
    input.vectorized_axis = output.vectorized_axis;
    input.vectorized_strides.assign(output.vectorized_axis.size(), af::sym::kSymbolOne);
    af::Expression flat_stride = af::sym::kSymbolOne;
    for (size_t i = output.vectorized_axis.size(); i > 0UL; --i) {
      input.vectorized_strides[i - 1UL] = flat_stride;
      const auto axis_iter = std::find(input.axis.begin(), input.axis.end(), output.vectorized_axis[i - 1UL]);
      GE_ASSERT_TRUE(axis_iter != input.axis.end(), "Arange vectorized axis [%ld] not found in padded view",
                     static_cast<int64_t>(output.vectorized_axis[i - 1UL]));
      const auto &axis_size = input.repeats[static_cast<size_t>(std::distance(input.axis.begin(), axis_iter))];
      flat_stride = af::sym::Mul(flat_stride, axis_size);
    }
    arange->attr.sched.axis = broadcast->attr.sched.axis;
  }
  return af::SUCCESS;
}

using NodeIndegrees = std::unordered_map<af::Node *, size_t>;
using NodeSuccessors = std::unordered_map<af::Node *, std::vector<af::NodePtr>>;

void AddGraphDependencies(const std::vector<af::NodePtr> &nodes, NodeIndegrees &indegrees, NodeSuccessors &out_nodes) {
  for (const auto &node : nodes) {
    std::unordered_set<af::Node *> unique_out_nodes;
    for (const auto &out_node : node->GetOutAllNodes()) {
      if ((indegrees.find(out_node.get()) == indegrees.end()) || !unique_out_nodes.emplace(out_node.get()).second) {
        continue;
      }
      out_nodes[node.get()].push_back(out_node);
      ++indegrees[out_node.get()];
    }
  }
}

std::unordered_map<af::Node *, size_t> BuildDependencyAwareRanks(
    const af::AscGraph &graph, const std::unordered_set<af::Node *> &outer_loop_sequences) {
  std::vector<af::NodePtr> nodes;
  NodeIndegrees indegrees;
  NodeSuccessors out_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    nodes.push_back(node);
    indegrees[node.get()] = 0UL;
  }
  AddGraphDependencies(nodes, indegrees, out_nodes);

  const auto has_higher_priority = [&outer_loop_sequences](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    bool is_node1_in_outer_seq = outer_loop_sequences.find(node1.get()) != outer_loop_sequences.end();
    bool is_node2_in_outer_seq = outer_loop_sequences.find(node2.get()) != outer_loop_sequences.end();
    if (is_node1_in_outer_seq != is_node2_in_outer_seq) {
      return is_node1_in_outer_seq;
    }
    return std::make_pair(node1->GetName(), node1->GetOpDescBarePtr()->GetId()) <
           std::make_pair(node2->GetName(), node2->GetOpDescBarePtr()->GetId());
  };

  std::vector<af::NodePtr> ready_nodes;
  for (const auto &node : nodes) {
    if (indegrees[node.get()] == 0UL) {
      ready_nodes.push_back(node);
    }
  }

  std::unordered_map<af::Node *, size_t> ranks;
  while (!ready_nodes.empty()) {
    auto iter = std::min_element(ready_nodes.begin(), ready_nodes.end(), has_higher_priority);
    auto node = *iter;
    ready_nodes.erase(iter);
    ranks[node.get()] = ranks.size();
    for (const auto &out_node : out_nodes[node.get()]) {
      auto &indegree = indegrees[out_node.get()];
      --indegree;
      if (indegree == 0UL) {
        ready_nodes.push_back(out_node);
      }
    }
  }
  return ranks;
}

namespace cast_helpers {
bool HasHighToLowCastNode(const std::unordered_set<af::AscNodePtr> &nodes) {
  for (const auto &node : nodes) {
    if (!af::ops::IsOps<af::ascir_op::Cast>(node)) {
      continue;
    }
    const auto in_dtype_size = af::GetSizeByDataType(node->inputs[0].attr.dtype);
    const auto out_dtype_size = af::GetSizeByDataType(node->outputs[0].attr.dtype);
    return in_dtype_size > out_dtype_size;
  }
  return false;
}

// 检查 connected_nodes 在 to 中的输出节点是否是低→高 Cast
bool HasLowToHighCastNode(const optimize::Cluster &to, const std::unordered_set<af::AscNodePtr> &connected_nodes) {
  for (const auto &node : connected_nodes) {
    // 找到边界节点的输出节点
    for (const auto &out_node : node->GetOutDataNodes()) {
      auto asc_out_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(asc_out_node);
      if (!to.ContainsNode(asc_out_node)) {
        continue;
      }
      // 检查是否是低→高 Cast
      if (!af::ops::IsOps<af::ascir_op::Cast>(asc_out_node)) {
        continue;
      }
      const auto in_dtype_size = af::GetSizeByDataType(asc_out_node->inputs[0].attr.dtype);
      const auto out_dtype_size = af::GetSizeByDataType(asc_out_node->outputs[0].attr.dtype);
      if (in_dtype_size < out_dtype_size) {
        return true;
      }
    }
  }
  return false;
}

// 检查两个 Cluster 内所有 Cast 节点的位宽差距是否超过阈值
// 检查所有 Cast 涉及的最大位宽和最小位宽，判断整体位宽变换倍数
bool CheckCastBitWidthGap(const optimize::Cluster &from, const optimize::Cluster &to, int32_t max_gap) {
  int32_t global_max_width = 0;
  int32_t global_min_width = std::numeric_limits<int32_t>::max();
  bool has_cast = false;

  // 直接遍历两个 cluster 的节点，避免创建临时集合
  for (const auto &cluster : {std::ref(from), std::ref(to)}) {
    for (const auto &node : cluster.get().nodes_) {
      if (!af::ops::IsOps<af::ascir_op::Cast>(node)) {
        continue;
      }
      has_cast = true;
      const auto in_dtype_size = af::GetSizeByDataType(node->inputs[0].attr.dtype);
      const auto out_dtype_size = af::GetSizeByDataType(node->outputs[0].attr.dtype);

      global_max_width = std::max({global_max_width, in_dtype_size, out_dtype_size});
      global_min_width = std::min({global_min_width, in_dtype_size, out_dtype_size});
    }
  }

  // 如果没有 Cast 节点，直接返回 true
  if (!has_cast) {
    return true;
  }

  // 检查整体位宽变换倍数是否超过阈值
  if (global_max_width > global_min_width * max_gap) {
    GELOGD("Cast nodes global bit width gap [%d vs %d] exceeds threshold [%d].", global_max_width, global_min_width,
           max_gap);
    return false;
  }
  return true;
}
}  // namespace cast_helpers

af::Status UnalignNode(const af::AscNodePtr &node) {
  for (const auto &tensor : node->outputs()) {
    GE_ASSERT_SUCCESS(optimize::BaseAlignmentStrategy::SetVectorizedStridesForTensor(
        node, tensor->attr, optimize::AlignmentType::kNotAligned));
  }
  return af::SUCCESS;
}

bool IsScalarBrc(const af::AscNodePtr &node) {
  if (!optimize::ScheduleUtils::IsBroadcast(node)) {
    return false;
  }
  const auto &vectorized_strides = node->inputs[0].attr.vectorized_strides;
  return std::all_of(vectorized_strides.begin(), vectorized_strides.end(), [](const af::Expression &stride) {
    return ascgen_utils::ExpressEq(stride, af::sym::kSymbolZero);
  });
}

int64_t FindLastNonBrcAxis(const std::vector<int64_t> &vec_axis, const std::vector<af::Expression> &vec_strides) {
  for (int64_t i = static_cast<int64_t>(vec_strides.size()) - 1; i >= 0; i--) {
    if (!ascgen_utils::ExpressEq(vec_strides[i], af::sym::kSymbolZero)) {
      return vec_axis[i];
    }
  }
  return af::kIdNone;
}

std::unordered_set<size_t> IdentifyZeroStrideVectorAxisIndices(const ascir::ImplGraph &owner_graph) {
  std::vector<bool> is_zero_stride_axis;
  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!optimize::ScheduleUtils::IsBuffer(node)) {
      is_zero_stride_axis.resize(node->outputs[0].attr.vectorized_strides.size(), true);
      break;
    }
  }

  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (optimize::ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    // 各节点 vectorized_strides 数量可能不同(如 ScalarData 无向量化轴):
    // 以本节点实际 strides 数量为界判定, 防止对空/短 vector 越界索引触发未定义行为。
    const size_t node_axis_count = node->outputs().empty() ? 0UL : node->outputs[0].attr.vectorized_strides.size();
    for (size_t axis_index = 0UL; axis_index < is_zero_stride_axis.size() && axis_index < node_axis_count;
         ++axis_index) {
      bool has_non_zero_stride = false;
      for (const auto &output : node->outputs()) {
        if (af::SymbolicUtils::StaticCheckEq(output->attr.vectorized_strides[axis_index], af::sym::kSymbolZero) !=
            af::TriBool::kTrue) {
          has_non_zero_stride = true;
          break;
        }
      }

      if (has_non_zero_stride) {
        is_zero_stride_axis[axis_index] = false;
      }
    }
  }

  std::unordered_set<size_t> zero_stride_axis_indices;
  for (size_t i = 0UL; i < is_zero_stride_axis.size(); ++i) {
    if (is_zero_stride_axis[i]) {
      zero_stride_axis_indices.emplace(i);
    }
  }
  // 全0场景,不需要删除
  if (zero_stride_axis_indices.size() == is_zero_stride_axis.size()) {
    return {};
  }

  return zero_stride_axis_indices;
}

Status RemoveAllZeroStrideVectorizedAxis(ascir::ImplGraph &owner_graph) {
  std::unordered_set<size_t> zero_stride_axis_indices = IdentifyZeroStrideVectorAxisIndices(owner_graph);
  if (zero_stride_axis_indices.empty()) {
    return af::SUCCESS;
  }

  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (optimize::ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    for (const auto &output : node->outputs()) {
      std::vector<int64_t> new_axis_ids;
      std::vector<af::Expression> new_strides;

      for (size_t i = 0UL; i < output->attr.vectorized_axis.size(); ++i) {
        if (zero_stride_axis_indices.count(i) == 0UL) {
          new_axis_ids.push_back(output->attr.vectorized_axis[i]);
          new_strides.push_back(output->attr.vectorized_strides[i]);
        }
      }

      output->attr.vectorized_axis = new_axis_ids;
      output->attr.vectorized_strides = new_strides;
    }
  }

  return af::SUCCESS;
}

bool IsVectorizedAxisContinuous(const af::AscGraph &graph, const int64_t pre_id, const int64_t post_id) {
  for (auto node : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(node)) {
      continue;
    }

    for (auto &out_tensor : node->outputs()) {
      GE_ASSERT_TRUE(out_tensor->attr.vectorized_axis.size() == out_tensor->attr.vectorized_strides.size());
      auto pre_iter =
          std::find(out_tensor->attr.vectorized_axis.begin(), out_tensor->attr.vectorized_axis.end(), pre_id);
      auto post_iter =
          std::find(out_tensor->attr.vectorized_axis.begin(), out_tensor->attr.vectorized_axis.end(), post_id);
      if ((pre_iter == out_tensor->attr.vectorized_axis.end()) ||
          (post_iter == out_tensor->attr.vectorized_axis.end())) {
        return false;
      }

      auto pre_idx = std::distance(out_tensor->attr.vectorized_axis.begin(), pre_iter);
      auto post_idx = std::distance(out_tensor->attr.vectorized_axis.begin(), post_iter);
      auto pre_stride = out_tensor->attr.vectorized_strides[pre_idx];
      auto post_axis_iter = std::find(out_tensor->attr.axis.begin(), out_tensor->attr.axis.end(), post_id);
      if (post_axis_iter == out_tensor->attr.axis.end()) {
        return false;
      }

      auto post_axis_idx = std::distance(out_tensor->attr.axis.begin(), post_axis_iter);
      auto post_count = out_tensor->attr.vectorized_strides[post_idx] * out_tensor->attr.repeats[post_axis_idx];
      if ((af::SymbolicUtils::StaticCheckEq(pre_stride, post_count) != af::TriBool::kTrue)) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<int64_t>> MergeContinuousPairs(const std::vector<std::pair<int64_t, int64_t>> &potential_axis) {
  std::vector<std::vector<int64_t>> continuous_ids;
  if (potential_axis.empty()) {
    return continuous_ids;
  }

  std::vector<int64_t> current_chain;
  current_chain.push_back(potential_axis[0].first);
  current_chain.push_back(potential_axis[0].second);

  for (size_t i = 1UL; i < potential_axis.size(); ++i) {
    const auto &cur_pair = potential_axis[i];
    if (current_chain.back() == cur_pair.first) {
      current_chain.push_back(cur_pair.second);
    } else {
      continuous_ids.push_back(current_chain);
      current_chain.clear();
      current_chain.push_back(cur_pair.first);
      current_chain.push_back(cur_pair.second);
    }
  }
  continuous_ids.push_back(current_chain);

  return continuous_ids;
}

af::Status ApplyMerge(const af::AscNodePtr &node, const af::AxisPtr &merged_axis,
                      const std::vector<af::AxisId> &from_ids) {
  // vector axis
  for (const auto output : node->outputs()) {
    // 因为后续不需要更新repeats，而且在有多组axis需要合并的场景，第一组合并之后的新vec_axis会找不到对应的repeats，所以这里仅初始化一个空的repeats
    std::vector<af::Expression> vec_repeats(output->attr.vectorized_axis.size());
    const auto &view = af::AxisUtils::MergeView(
        {output->attr.vectorized_axis, vec_repeats, output->attr.vectorized_strides}, merged_axis->id, from_ids);
    output->attr.vectorized_axis = view.axis_ids;
    output->attr.vectorized_strides = view.strides;
  }

  return af::SUCCESS;
}

void AddAnchorToOrderMap(
    const af::OutDataAnchorPtr &peer_out_anchor, const af::InDataAnchorPtr &in_anchor,
    std::vector<std::pair<af::OutDataAnchorPtr, std::vector<af::InDataAnchorPtr>>> &anchor_vector) {
  bool found = false;
  for (auto &pair : anchor_vector) {
    if (pair.first == peer_out_anchor) {
      pair.second.push_back(in_anchor);
      found = true;
      break;
    }
  }
  if (!found) {
    anchor_vector.emplace_back(peer_out_anchor, std::vector<af::InDataAnchorPtr>{in_anchor});
  }
}

using BoundaryAnchorEntry = std::pair<af::OutDataAnchorPtr, std::vector<af::InDataAnchorPtr>>;

void SortBoundaryAnchors(std::vector<BoundaryAnchorEntry> &anchors) {
  const auto out_anchor_less = [](const BoundaryAnchorEntry &lhs, const BoundaryAnchorEntry &rhs) {
    const auto *lhs_node = lhs.first->GetOwnerNodeBarePtr();
    const auto *rhs_node = rhs.first->GetOwnerNodeBarePtr();
    return std::make_pair(lhs_node->GetName(), lhs.first->GetIdx()) <
           std::make_pair(rhs_node->GetName(), rhs.first->GetIdx());
  };
  const auto in_anchor_less = [](const af::InDataAnchorPtr &lhs, const af::InDataAnchorPtr &rhs) {
    const auto *lhs_node = lhs->GetOwnerNodeBarePtr();
    const auto *rhs_node = rhs->GetOwnerNodeBarePtr();
    return std::make_pair(lhs_node->GetName(), lhs->GetIdx()) < std::make_pair(rhs_node->GetName(), rhs->GetIdx());
  };
  std::sort(anchors.begin(), anchors.end(), out_anchor_less);
  for (auto &entry : anchors) {
    std::sort(entry.second.begin(), entry.second.end(), in_anchor_less);
  }
}

bool NeedRemovePad(const af::AscNodePtr &node) {
  // 如果是非scalar的Broadcast节点，直接插RemovePad，结束循环
  if (optimize::ScheduleUtils::IsBroadcast(node) && !optimize::ScheduleUtils::IsScalarBroadcastNode(node)) {
    return true;
  }
  if (ascgen_utils::IsNodeContainsBrcInline(node)) {
    return true;
  }
  if (optimize::ScheduleUtils::IsLoad(node) && node->GetInDataNodesSize() == 1UL && node->GetOutDataNodesSize() > 0UL) {
    // 判断Load是否是非连续的
    const auto &repeats = node->outputs[0].attr.repeats;
    const auto &strides = node->outputs[0].attr.strides;
    return !optimize::ScheduleUtils::IsContinuesStrides(repeats, strides);
  }
  return false;
}

bool IsPeerNodesContainsVF(const af::OutDataAnchorPtr &anchor) {
  GE_ASSERT_NOTNULL(anchor);
  for (const auto &peer_in_anchor : anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_anchor);
    const auto &peer_in_node = peer_in_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_node);
    if (af::ops::IsOps<af::ascir_op::VectorFunc>(peer_in_node)) {
      return true;
    }
  }
  return false;
}

af::Status ReverseDfsUnAlignNode(af::AscGraph &impl_graph, const af::NodePtr &ge_node,
                                 std::set<af::NodePtr> &visited_nodes) {
  // 这些节点不需要对齐
  if (optimize::ScheduleUtils::IsIOBuffer(ge_node) || optimize::ScheduleUtils::IsRemovePad(ge_node)) {
    return af::SUCCESS;
  }
  const auto &node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
  if (visited_nodes.find(node) != visited_nodes.end()) {
    return af::SUCCESS;
  }
  visited_nodes.insert(node);
  // 判断是否需要插入RemovePad，如果需要则插入RemovePad并结束
  if (NeedRemovePad(node)) {
    af::AscNodePtr remove_pad_node = nullptr;
    for (uint32_t idx = 0U; idx < node->GetAllOutDataAnchorsSize(); idx++) {
      if (IsPeerNodesContainsVF(node->GetOutDataAnchor(static_cast<int32_t>(idx)))) {
        continue;
      }
      GE_ASSERT_SUCCESS(optimize::ScheduleUtils::AddRemovePadAfter(impl_graph, node, remove_pad_node, idx));
      GE_ASSERT_SUCCESS(UnalignNode(remove_pad_node));
      visited_nodes.insert(remove_pad_node);
    }
    return af::SUCCESS;
  }
  // 如果不需要插入RemovePad，则还原不对齐的vector_strides
  GE_ASSERT_SUCCESS(UnalignNode(node));
  for (const auto &in_node : node->GetInDataNodes()) {
    GE_ASSERT_SUCCESS(ReverseDfsUnAlignNode(impl_graph, in_node, visited_nodes));
  }
  return af::SUCCESS;
}
}  // namespace
namespace optimize {
const std::string kNamePrefixLoad = "Load_";
const std::string kNamePrefixStore = "Store_";
const std::string kNamePrefixData = "Data_";

af::Status NormalizeArangeBroadcastViews(af::AscGraph &graph) {
  return NormalizeArangeBroadcastViewsImpl(graph);
}
const std::string kNamePrefixScalar = "Scalar_";
const std::string kNamePrefixOutput = "Output_";

af::Status VectorFuncPartitioner::Partition() {
  GE_ASSERT_SUCCESS(ValidateUniqueNodeNames(impl_graph_));
  ascir::utils::DumpGraph(impl_graph_, "BeforePartition");
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(impl_graph_), "Failed to do topological sorting for graph[%s].",
                    impl_graph_.GetName().c_str());
  root_graph_ = af::AscGraphUtils::GetComputeGraph(impl_graph_);
  GE_ASSERT_NOTNULL(root_graph_);

  // 0.检查图中是否有reduce节点（只检查一次，避免在RefineEnableVFFlag中重复检查）
  graph_has_reduce_node_ = HasReduceNodeInGraph(impl_graph_);
  if (graph_has_reduce_node_) {
    GELOGI("Graph [%s] has reduce node, will disable Cast VF fusion.", impl_graph_.GetName().c_str());
  }

  // 1.InitClusters (Compare nodes' outputs are merged here)
  GE_ASSERT_SUCCESS(InitClusters(), "Failed to do topological sorting for graph[%s].", impl_graph_.GetName().c_str());

  // 2.TryMergeClusters
  GE_ASSERT_GRAPH_SUCCESS(MergeClusters(), "Failed to merge clusters for graph[%s]", impl_graph_.GetName().c_str());

  // 3. lane 恒定 Arange 的 VF 消费者检查：消除单元素 buffer → LoadAlign 垃圾
  //    的 inter-VF 数据通路（详见 DisableLaneConstantArangeVFForVFConsumers 注释）。
  GE_ASSERT_SUCCESS(DisableLaneConstantArangeVFForVFConsumers());

  // 4.SortClusters
  GE_ASSERT_GRAPH_SUCCESS(SortClustersForBuildSubgraph(), "Failed to sort clusters for graph[%s]",
                          impl_graph_.GetName().c_str());
  DebugMergeLog();

  GE_ASSERT_GRAPH_SUCCESS(BuildSubgraphs(), "Failed to build subgraphs for graph[%s].", impl_graph_.GetName().c_str());

  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(impl_graph_), "Failed to do topological sorting for graph[%s].",
                    impl_graph_.GetName().c_str());
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::InitClusterAttr(const std::unique_ptr<af::ascir::AscIrCodegen> &codegen_impl,
                                                  const af::AscNodePtr &node, ClusterPtr &cluster) {
  // 尾轴stride!=1场景，暂不支持生成vf代码
  for (const auto &output : node->outputs()) {
    const auto &vectorized_strides = output->attr.vectorized_strides;
    auto it = std::find_if(vectorized_strides.rbegin(), vectorized_strides.rend(), [](const af::Expression &val) {
      return af::SymbolicUtils::StaticCheckNe(val, af::sym::kSymbolZero) == af::TriBool::kTrue;
    });
    if ((it != vectorized_strides.rend()) &&
        (af::SymbolicUtils::StaticCheckNe(*it, af::sym::kSymbolOne) == af::TriBool::kTrue)) {
      GELOGD("The stride of the node[%s]'s tail axis is not 1, which is not supported in vf.", node->GetNamePtr());
      cluster->meta_data_.enable_vf = false;
      return af::SUCCESS;
    }
    if (node->GetType() == af::ascir_op::Broadcast::Type) {
      const auto &axis = output->attr.axis;
      const auto &vectorized_axis = output->attr.vectorized_axis;
      size_t axis_id = UINT64_MAX;
      for (size_t i = 0; i < vectorized_axis.size(); i++) {
        auto it2 = std::find(axis.begin(), axis.end(), vectorized_axis[i]);
        GE_ASSERT_TRUE(it2 != axis.end(), "vectorized_axis not in axis");
        axis_id = std::distance(axis.begin(), it2);
        cluster->meta_data_.vectorized_repeats.push_back(output->attr.repeats[axis_id]);
      }
    }
  }

  cluster->meta_data_.ins_num = codegen_impl->GetInstNum();
  cluster->meta_data_.loop_axis = node->attr.sched.loop_axis;
  cluster->out_nodes_.insert(node);
  for (const auto &input : node->inputs()) {
    auto in_node = std::dynamic_pointer_cast<af::AscNode>(input->anchor.GetOwnerNode());
    GE_ASSERT_NOTNULL(in_node);
    if (!ScheduleUtils::IsConstantScalar(in_node.get())) {
      cluster->in_nodes_.insert(in_node);
    }
  }

  return af::SUCCESS;
}

void VectorFuncPartitioner::RefineEnableVFFlag(const af::AscNodePtr &node, bool &enable_vf) const {
  if (!enable_vf) {
    return;
  }

  // Arange: 带控制边时 VF 子图无法保留控制依赖; UBFuse 上下文的 VectorFunc codegen 不支持
  // Arange。两类 Arange 均在建簇阶段关闭 VF, 保留在根图走普通 ArangeApiCall。
  if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
    // 广播外轴(stride 为 0)的 Arange 融合进 VF 后, 轴合并/flatten 会把广播轴折叠为
    // 线性索引, Reg::Arange 无法表达逐轴重复取值。此类 Arange 保留在根图, 由
    // ArangeApiCall 按逻辑 stride 物化, 物化布局按轴尺寸展开为连续 stride,
    // 并在调用内部对非固定广播轴展开逐帧写入以覆盖整个分配范围。
    const bool has_broadcast_outer_axis = ArangeHasBroadcastOuterAxis(node);
    if (has_broadcast_outer_axis) {
      if (!MaterializeArangeViewContiguous(node)) {
        enable_vf = false;
        return;
      }
    }
    // 沿向量化轴恒定的 Arange（取值轴与 lane 轴不一致，如 row 型 [1,M,1]）：
    // 单节点 VF 只产出 1 个有效元素（preg_vl1 单 lane 存储），消费侧 VfNode 以
    // LoadAlign 向量加载后 Duplicate 向量拷贝（非标量广播），lane 1+ 读到未初始化
    // 垃圾数据。此类 Arange
    // 保留在根图走 ArangeApiCall（按逻辑 stride 物化全序列 + 广播轴帧展开），
    // 与基线行为一致。
    if (has_broadcast_outer_axis || disable_arange_vf_ || node->GetInControlNodesSize() != 0UL ||
        node->GetOutControlNodesSize() != 0UL) {
      GELOGD("Node [%s] is Arange with broadcast outer axis or control edge or in UBFuse context, disable VF support.",
             node->GetNamePtr());
      enable_vf = false;
      return;
    }
  }

  // 1. 如果当前图中有reduce节点，cast不参与vf融合
  if (af::ops::IsOps<af::ascir_op::Cast>(node)) {
    if (graph_has_reduce_node_) {
      // 当前直接把cast移到外面使用castExtend api有性能问题，待解决后放开。enable_vf = false;
      GELOGD("Node [%s] is Cast and graph has reduce node, disable VF support.", node->GetNamePtr());
      // 当前直接把cast移到外面使用castExtend api有性能问题，待解决后放开。return;
    }
  }

  // 2. ScalarBrc 场景：检查输出节点是否支持 VF
  if (IsScalarBrc(node)) {
    bool is_out_support_vf = false;
    for (const auto &out_node : node->GetOutDataNodes()) {
      auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
      auto out_impl = ascgen_utils::GetAscIrCodegenImpl(out_asc_node->GetType());
      if (out_impl->IsVectorFunctionSupported(*out_asc_node)) {
        is_out_support_vf = true;
        break;
      }
    }
    enable_vf = is_out_support_vf;
    return;
  }

  // 3. 尾轴stride!=1场景，暂不支持生成vf代码
  for (const auto &output : node->outputs()) {
    const auto &vectorized_strides = output->attr.vectorized_strides;
    auto it = std::find_if(vectorized_strides.rbegin(), vectorized_strides.rend(), [](const af::Expression &val) {
      return af::SymbolicUtils::StaticCheckNe(val, af::sym::kSymbolZero) == af::TriBool::kTrue;
    });
    if ((it != vectorized_strides.rend()) &&
        (af::SymbolicUtils::StaticCheckNe(*it, af::sym::kSymbolOne) == af::TriBool::kTrue)) {
      GELOGD("The stride of the node[%s]'s tail axis is not 1, which is not supported in vf.", node->GetNamePtr());
      enable_vf = false;
      return;
    }
  }

  // 4. IndirectLoad template nodes with custom emit paths should not participate in regular VF fusion.
  // [组链无角色设计下的门禁还原] kSimtFanoutBranch 仅由单 gather 扇出路径标注（含 Store）：
  // 其既有契约为"不进 VF + 跳过 API 发射"（输出经 AddSkippedApiEmitProcessCall 注册，
  // 单 gather ST 全量验证），进 VF 反而改变基线行为。组路径的链节点不注解任何角色
  // （behavior 全 false → 本门禁不生效），VF/normal 由分区器自然裁决——无需豁免。
  if (ascgen_utils::indirect_load::GetTemplateBehavior(node).uses_direct_gm_pipeline) {
    enable_vf = false;
    GELOGD("Node [%s] is IndirectLoad custom emit node, disable VF support.", node->GetNamePtr());
    return;
  }

  // 5. Compare/Add/Min/Max/LogicalAnd等微指令支持scalar输入，不需要检查直连scalar
  if (ScheduleUtils::IsMicroApiSupportsScalarInput(node)) {
    return;
  }

  // 6. 非ScalarBrc场景：如果算子的任意输入直连scalar且不支持scalar输入，就把enable_vf标记为false
  for (const auto &in_node : node->GetInDataNodes()) {
    if (ScheduleUtils::IsScalarLikeNode(in_node)) {
      enable_vf = false;
      GELOGD("Node [%s] has direct Scalar input, disable VF support.", node->GetNamePtr());
      break;
    }
  }
}

bool VectorFuncPartitioner::HasReduceNodeInGraph(const af::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
    if (asc_node == nullptr) {
      continue;
    }
    // 使用标准的 ScheduleUtils::IsReduce 方法检查是否为 reduce 节点
    if (ScheduleUtils::IsReduce(asc_node)) {
      GELOGD("Found reduce node [%s] with type [%s] in graph, disable Cast VF fusion.", asc_node->GetNamePtr(),
             asc_node->GetTypePtr());
      return true;
    }
  }
  return false;
}

bool VectorFuncPartitioner::IsCompareOp(const af::AscNodePtr &node) {
  static const std::unordered_set<std::string> compare_types = {
      af::ascir_op::Ge::Type, af::ascir_op::Eq::Type, af::ascir_op::Ne::Type,
      af::ascir_op::Le::Type, af::ascir_op::Lt::Type, af::ascir_op::Gt::Type,
  };
  return compare_types.count(node->GetType()) > 0UL;
}

af::Status VectorFuncPartitioner::InitClusters() {
  size_t rank = 0UL;
  GELOGI("InitClusters enter, graph_name[%s].", impl_graph_.GetName().c_str());

  for (const auto &node : impl_graph_.GetAllNodes()) {
    // 跳过已经加入 cluster 的节点（被 Compare 合并的）
    if (cluster_dict_.GetNodeCluster(node) != nullptr) {
      continue;
    }

    auto cluster = CreateAndInitCluster(node, rank);
    cluster_dict_.AddCluster(cluster);
    cluster_dict_.SetNodeClusterPair(node, cluster);
    EstablishClusterConnections(cluster, node);
  }

  FixAllCompareClusterConnections();
  return af::SUCCESS;
}

ClusterPtr VectorFuncPartitioner::CreateAndInitCluster(const af::AscNodePtr &node, size_t &rank) {
  auto cluster = af::MakeShared<Cluster>(node, ++rank);
  GE_ASSERT_NOTNULL(cluster, "Failed to malloc memory for cluster.");

  auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Cannot find impl for ir type:[%s].", node->GetTypePtr());
  cluster->meta_data_.enable_vf = codegen_impl->IsVectorFunctionSupported(*node);
  RefineEnableVFFlag(node, cluster->meta_data_.enable_vf);
  if (cluster->meta_data_.enable_vf) {
    GE_ASSERT_SUCCESS(InitClusterAttr(codegen_impl, node, cluster));
  }

  return cluster;
}

void VectorFuncPartitioner::EstablishClusterConnections(ClusterPtr &cluster, const af::AscNodePtr &node) {
  for (const auto &in_node : node->GetInAllNodes()) {
    const auto &in_cluster = cluster_dict_.GetNodeCluster(in_node);
    if (in_cluster == nullptr) {
      GELOGD("The in cluster of the node [%s] is nullptr, and the topological sort may be incorrect.",
             in_node->GetNamePtr());
      continue;
    }
    cluster->AddInput(*in_cluster);
  }
}

void VectorFuncPartitioner::FixAllCompareClusterConnections() {
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    af::AscNodePtr compare_node = nullptr;
    for (const auto &node : cluster->nodes_) {
      if (IsCompareOp(node)) {
        compare_node = node;
        break;
      }
    }

    if (compare_node != nullptr) {
      FixCompareClusterConnections(cluster, compare_node);
    }
  }
}

void VectorFuncPartitioner::FixCompareClusterConnections(const ClusterPtr &cluster,
                                                         const af::AscNodePtr &compare_node) {
  std::unordered_set<Cluster *> missing_input_clusters;
  missing_input_clusters.reserve(cluster->in_nodes_.size());
  for (const auto &in_node : cluster->in_nodes_) {
    // 跳过Compare节点自己的输入
    if (in_node == compare_node) {
      continue;
    }
    const auto &in_cluster = cluster_dict_.GetNodeCluster(in_node);
    if (in_cluster == nullptr) {
      continue;
    }
    if (in_cluster->Id() == cluster->Id()) {
      // in_node已经在这个cluster中，跳过
      continue;
    }
    // 检查这个cluster是否已经在inputs_中
    bool found = false;
    for (const auto &existing_input : cluster->inputs_) {
      if (existing_input->Id() == in_cluster->Id()) {
        found = true;
        break;
      }
    }
    if (!found) {
      missing_input_clusters.insert(in_cluster.get());
    }
  }

  // 添加缺失的输入cluster连接
  for (const auto &in_cluster : missing_input_clusters) {
    cluster->AddInput(*in_cluster);
  }
}

void VectorFuncPartitioner::DebugMergeLog() const {
  if (!IsLogEnable(GE, DLOG_DEBUG)) {
    return;
  }
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    std::stringstream ss;
    ss << "[CLUSTER_MERGER][" << cluster->DebugString() << "]";
    size_t debug_string_size = ss.str().size();
    size_t pos = 0UL;
    for (size_t loop = 0UL; loop < (debug_string_size / static_cast<size_t>(MSG_LENGTH)); loop++) {
      GELOGD("%s", ss.str().c_str() + pos);
      pos += static_cast<size_t>(MSG_LENGTH);
    }
    GELOGD("%s", ss.str().c_str() + pos);
  }
}

bool VectorFuncPartitioner::CanMergeClusters(const Cluster &from, const Cluster &to) {
  const auto &from_meta = from.meta_data_;
  const auto &to_meta = to.meta_data_;

  if (!from_meta.enable_vf || !to_meta.enable_vf) {
    return false;
  }
  // 沿向量化轴恒定的 Arange（取值轴与 lane 轴不一致）禁止融入多节点 VF：其折叠
  // 模板参数推导存在缺陷（见 ClusterContainsLaneConstantArange 注释），回退为
  // "单 Arange VfNode + 根图 BroadcastExtend"的安全模式。
  if (ClusterContainsLaneConstantArange(from) || ClusterContainsLaneConstantArange(to)) {
    GELOGD("Cluster contains lane-constant Arange, keep it as single-node VF, skip fuse [%zu] to [%zu].", from.Id(),
           to.Id());
    return false;
  }
  // 需要在同一个循环内
  if (from_meta.loop_axis != to_meta.loop_axis) {
    return false;
  }
  // 最大指令数30条
  if (from_meta.ins_num + to_meta.ins_num > kMaxInsNum) {
    GELOGD("the total ins num after fusion exceeds the threshold, skip to fuse [%zu] to [%zu].", from.Id(), to.Id());
    return false;
  }
  // 两个向量化轴对应的repeats都不空时，尾轴大小必须相同
  if (!from_meta.vectorized_repeats.empty() && !to_meta.vectorized_repeats.empty() &&
      (from_meta.vectorized_repeats != to_meta.vectorized_repeats)) {
    return false;
  }

  auto connected_nodes = Cluster::FindConnectedNodes(from, to);
  // 只针对 Cast 节点进行位宽限制
  // 1. 低位宽→高位宽的 Cast：不允许和输入节点融合
  //    检查 connected_nodes 在 to 中的输出节点是否是低→高 Cast
  if (cast_helpers::HasLowToHighCastNode(to, connected_nodes)) {
    GELOGD("Low-to-high cast in to cluster, skip fuse [%s] to [%s].", from.DebugString().c_str(),
           to.DebugString().c_str());
    return false;
  }

  // 2. 高位宽→低位宽的 Cast：不允许和输出节点融合
  if (cast_helpers::HasHighToLowCastNode(connected_nodes)) {
    GELOGD("High-to-low cast in connected nodes, skip fuse [%s] to [%s].", from.DebugString().c_str(),
           to.DebugString().c_str());
    return false;
  }

  // 3. 防止出现两个cluster上各有一个Cast导致位宽差距超过2倍
  if (!cast_helpers::CheckCastBitWidthGap(from, to, kMaxBitWidthGap)) {
    GELOGD("Cast bit width gap exceeds threshold, skip fuse [%s] to [%s].", from.DebugString().c_str(),
           to.DebugString().c_str());
    return false;
  }

  // 输入+输出节点个数<=8
  auto merged_in = Cluster::CalculateMergedInNodes(from, to, connected_nodes);
  auto merged_out = Cluster::CalculateMergedOutNodes(from, to);
  if (merged_in.size() + merged_out.size() > kMaxIONum) {
    GELOGD("the total io num  after fusion exceeds the threshold, skip to fuse [%s] to [%s].",
           from.DebugString().c_str(), to.DebugString().c_str());
    return false;
  }

  return true;
}

af::Status VectorFuncPartitioner::MergeClusters() {
  // Merge clusters according to the linking relationship
  auto all_clusters = cluster_dict_.GetAllClusters();
  std::unordered_set<const Cluster *> merged_clusters;  // 记录已合并的 cluster

  for (const auto &cluster : all_clusters) {
    // 如果该 cluster 已被合并到其他 cluster，跳过
    if (merged_clusters.count(cluster.get()) > 0UL) {
      continue;
    }

    const auto cluster_inputs = cluster->Inputs();
    for (const auto &in_cluster : cluster_inputs) {
      // 如果输入 cluster 已被合并，跳过
      if (merged_clusters.count(in_cluster) > 0UL) {
        continue;
      }
      if (!CanMergeClusters(*in_cluster, *cluster)) {
        continue;
      }
      if (HasDetectedCycle(in_cluster, cluster.get())) {
        GELOGD("There exists cycle between %zu and %zu, will skip to merge.", in_cluster->Id(), cluster->Id());
        continue;
      }
      // 执行合并
      cluster->MergeFrom(*in_cluster);
      merged_clusters.insert(in_cluster);
      // 批量更新 cluster_dict_ 映射
      for (const auto &node : in_cluster->Nodes()) {
        cluster_dict_.SetNodeClusterPair(node, cluster);
      }
      GELOGD("Merge cluster from %zu to %zu.", in_cluster->Id(), cluster->Id());
    }
  }
  return af::SUCCESS;
}

// lane 恒定 Arange 的 VF 消费者检查（MergeClusters 后、BuildSubgraphs 前执行）。
// CanMergeClusters 已阻止 lane 恒定 Arange 融入多节点 VF（消除折叠 VfNode 参数
// 推导错误）。但拆分后仍可作为单节点 VF 存在——VfNode 用 preg_vl1（单 lane）
// 只写 1 个元素到 buffer，消费侧 VfNode 以 LoadAlign 加载整个向量后 Duplicate
// 向量拷贝（非标量广播），lane 1+ 读到未初始化垃圾（折叠与分离形态均如此）。
// 输出被 VF 簇消费的 lane 恒定 Arange 留在根图走 ArangeApiCall；被根图节点
// （如 Store）消费的保持 VF 不变。
af::Status VectorFuncPartitioner::DisableLaneConstantArangeVFForVFConsumers() {
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    if (!cluster->meta_data_.enable_vf || cluster->Nodes().size() != 1UL) {
      continue;
    }
    const auto &node = cluster->Nodes().front();
    if (!af::ops::IsOps<af::ascir_op::Arange>(node) || !ArangeConstantAlongVectorizedAxes(node)) {
      continue;
    }
    bool has_vf_consumer = false;
    for (const auto &out_node : node->GetOutDataNodes()) {
      // Broadcast 视图融合（BroadcastExtend 路径）内联处理 lane 恒定 Arange 输入（退化前缀/尾轴
      // Arange→Broadcast 的 degenerate 形态），不算 VF 消费者；仅计算类 VF 簇消费才需要留根图。
      if (af::ops::IsOps<af::ascir_op::Broadcast>(out_node)) {
        continue;
      }
      const auto &consumer_cluster = cluster_dict_.GetNodeCluster(out_node);
      if (consumer_cluster != nullptr && consumer_cluster->Id() != cluster->Id() &&
          consumer_cluster->meta_data_.enable_vf) {
        has_vf_consumer = true;
        break;
      }
    }
    if (has_vf_consumer) {
      cluster->meta_data_.enable_vf = false;
      GELOGI("Lane-constant Arange [%s] has VF consumer, keep it in root graph for ArangeApiCall.",
             node->GetName().c_str());
    }
  }
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::SortClustersForBuildSubgraph() {
  // 收集所有唯一的 cluster, 按照id进行合并
  std::unordered_set<ClusterPtr> unique_clusters;
  for (const auto &node : impl_graph_.GetAllNodes()) {
    const auto &cluster = cluster_dict_.GetNodeCluster(node);
    unique_clusters.insert(cluster);
  }

  std::vector<ClusterPtr> sorted_unique_clusters(unique_clusters.begin(), unique_clusters.end());
  std::sort(sorted_unique_clusters.begin(), sorted_unique_clusters.end(),
            [](const ClusterPtr &clu_a, const ClusterPtr &clu_b) -> bool { return clu_a->Id() < clu_b->Id(); });

  cluster_dict_.SwapClusters(sorted_unique_clusters);
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::AddInputDataAnchors(const af::NodePtr &node,
                                                      InsertOrderMap &out_data_to_peer_in_anchors) {
  const auto &dst_graph = node->GetOwnerComputeGraph();
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto in_node = peer_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(in_node);
    const auto &src_graph = in_node->GetOwnerComputeGraph();
    if (src_graph != dst_graph) {
      AddAnchorToOrderMap(peer_out_anchor, in_anchor, out_data_to_peer_in_anchors);
    }
  }

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::AddOutputDataAnchors(const af::NodePtr &node,
                                                       InsertOrderMap &out_data_to_peer_in_anchors) {
  const auto &src_graph = node->GetOwnerComputeGraph();
  for (const auto &anchor : node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(anchor);
    const auto &peer_in_anchors = anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_anchor : peer_in_anchors) {
      GE_ASSERT_NOTNULL(peer_in_anchor);
      auto peer_out_node = peer_in_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_out_node);
      const auto &dst_graph = peer_out_node->GetOwnerComputeGraph();
      if (src_graph != dst_graph) {
        AddAnchorToOrderMap(anchor, peer_in_anchor, out_data_to_peer_in_anchors);
      }
    }
  }
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::InsertDataAndLoadNode(af::AscGraph &asc_graph, const af::OutDataAnchorPtr &out_anchor,
                                                        const std::vector<af::InDataAnchorPtr> &in_anchors,
                                                        int64_t parent_in_index) {
  auto pre_node = std::dynamic_pointer_cast<af::AscNode>(out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(pre_node);
  std::string data_name = kNamePrefixData + pre_node->GetName() + std::to_string(parent_in_index);
  af::ascir_op::Data data(data_name.c_str());
  auto data_node = asc_graph.AddNode(data);
  GE_ASSERT_NOTNULL(data_node);
  data_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  data_node->attr.api.unit = af::ComputeUnit::kUnitNone;
  data_node->outputs[0].attr.dtype = pre_node->outputs[out_anchor->GetIdx()].attr.dtype;
  auto ir_attr = data.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr);
  (void)ir_attr->SetIndex(parent_in_index);

  std::string load_name = kNamePrefixLoad + pre_node->GetName() + std::to_string(parent_in_index);
  af::ascir_op::Load load(load_name.c_str());
  auto load_node = asc_graph.AddNode(load);
  GE_ASSERT_NOTNULL(load_node);
  load_node->attr.sched = pre_node->attr.sched;
  load_node->attr.api = {af::ApiType::kAPITypeCompute, af::ComputeType::kComputeLoad, af::ComputeUnit::kUnitMTE2};
  load_node->outputs[0].attr = pre_node->outputs[out_anchor->GetIdx()].attr;
  load.x = data.y;
  load_node->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load_node->attr.api.type = af::ApiType::kAPITypeCompute;
  load_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  GELOGD("Set Load [%s] attr from node [%s] out_anchor:[%d].", load_name.c_str(), pre_node->GetName().c_str(),
         out_anchor->GetIdx());
  for (const auto &in_anchor : in_anchors) {
    GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(out_anchor, in_anchor));
    GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), in_anchor));
  }
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::InsertScalarNode(af::AscGraph &asc_graph, const af::OutDataAnchorPtr &out_anchor,
                                                   const std::vector<af::InDataAnchorPtr> &in_anchors,
                                                   int64_t parent_in_index) {
  auto pre_node = std::dynamic_pointer_cast<af::AscNode>(out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(pre_node);

  std::string scalar_name = kNamePrefixScalar + pre_node->GetName();
  af::ascir_op::Scalar scalar(scalar_name.c_str());
  auto scalar_node = asc_graph.AddNode(scalar);
  GE_ASSERT_NOTNULL(scalar_node);
  scalar_node->attr.sched = pre_node->attr.sched;
  scalar_node->attr.api = pre_node->attr.api;
  GE_ASSERT_SUCCESS(scalar.ir_attr.SetIndex(parent_in_index));
  scalar_node->outputs[0].attr = pre_node->outputs[0].attr;
  GELOGD("Set Scalar [%s] attr from node [%s] out_anchor:[%d].", scalar_name.c_str(), pre_node->GetName().c_str(),
         out_anchor->GetIdx());
  for (const auto &in_anchor : in_anchors) {
    GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(out_anchor, in_anchor));
    GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(scalar_node->GetOutDataAnchor(0), in_anchor));
  }
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::InsertStoreAndOutputNode(af::AscGraph &asc_graph, af::AscNode &pre_node,
                                                           size_t out_anchor_index, int64_t parent_out_index) {
  std::string store_name = kNamePrefixStore + pre_node.GetName() + std::to_string(parent_out_index);
  af::ascir_op::Store store(store_name.c_str());
  auto store_node = asc_graph.AddNode(store);
  GE_ASSERT_NOTNULL(store_node);
  store_node->attr.sched = pre_node.attr.sched;
  store_node->attr.api = {af::ApiType::kAPITypeCompute, af::ComputeType::kComputeLoad, af::ComputeUnit::kUnitMTE2};
  store_node->outputs[0].attr = pre_node.outputs[out_anchor_index].attr;
  store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
  store_node->attr.api.type = af::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = af::ComputeUnit::kUnitMTE2;

  GE_ASSERT_SUCCESS(
      af::GraphUtils::AddEdge(pre_node.GetOutDataAnchor(out_anchor_index), store_node->GetInDataAnchor(0)));

  std::string output_name = kNamePrefixOutput + pre_node.GetName() + std::to_string(parent_out_index);
  af::ascir_op::Output output(output_name.c_str());
  auto output_node = asc_graph.AddNode(output);
  GE_ASSERT_NOTNULL(output_node);
  output_node->attr.api.compute_type = af::ComputeType::kComputeInvalid;
  output_node->attr.api.type = af::ApiType::kAPITypeBuffer;
  output_node->attr.api.unit = af::ComputeUnit::kUnitNone;
  output_node->outputs[0].attr.dtype = pre_node.outputs[out_anchor_index].attr.dtype;
  output.x = store.y;
  auto ir_attr = output.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  GE_ASSERT_NOTNULL(ir_attr);
  (void)ir_attr->SetIndex(parent_out_index);

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::BuildSubgraph(const ClusterPtr &cluster, af::AscGraph &vf_graph,
                                                af::ascir_op::VectorFunc &vf_op) {
  auto vf_ge_graph = af::AscGraphUtils::GetComputeGraph(vf_graph);
  GE_ASSERT_NOTNULL(vf_ge_graph);
  InsertOrderMap load_to_peer_in_anchors;
  InsertOrderMap store_to_peed_in_anchors;
  // move node to subgraph
  for (const auto &node : cluster->Nodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(vf_ge_graph->AddNode(node), "Failed to add node [%s] to graph [%s].", node->GetNamePtr(),
                      vf_ge_graph->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveJustNode(root_graph_, node),
                            "Failed to remove node [%s] from graph [%s].", node->GetNamePtr(),
                            root_graph_->GetName().c_str());
    GE_ASSERT_GRAPH_SUCCESS(node->SetOwnerComputeGraph(vf_ge_graph));
  }

  for (const auto &node : cluster->Nodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_SUCCESS(AddInputDataAnchors(node, load_to_peer_in_anchors));
    GE_ASSERT_SUCCESS(AddOutputDataAnchors(node, store_to_peed_in_anchors));
  }
  SortBoundaryAnchors(load_to_peer_in_anchors);
  SortBoundaryAnchors(store_to_peed_in_anchors);

  vf_op.InstanceOutputy(store_to_peed_in_anchors.size());
  std::vector<af::AscOpOutput> outputs;
  std::vector<af::Operator> ops;
  outputs.reserve(load_to_peer_in_anchors.size());
  ops.reserve(load_to_peer_in_anchors.size());
  size_t parent_in_idx = 0UL;
  std::vector<std::pair<af::OutDataAnchorPtr, int64_t>> external_boundaries;
  for (auto &iter : load_to_peer_in_anchors) {
    auto out_anchor = iter.first;
    auto pre_node = out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(pre_node);
    if (ScheduleUtils::IsConstantScalar(pre_node) || af::ops::IsOps<af::ascir_op::ScalarData>(pre_node)) {
      // 常量 Scalar 与 ScalarData(运行时标量输入)在子图内都物化为 Scalar 副本, 由
      // codegen 侧 ParseInputOutputInfo 归入 scalar 值参数链传递; 若改走 Data+Load
      // 边界, Load 会对值参数生成 LoadAlign(非法), 且插入的 Load 节点携带
      // ScalarData 的 sched 轴, 污染循环深度推导, 生成未声明的 mask 寄存器引用。
      // 副本与 dynamic input 共享 parent 序号空间, 避免子图内 index 越界。
      GE_ASSERT_SUCCESS(InsertScalarNode(vf_graph, out_anchor, iter.second, parent_in_idx));
      ops.push_back(af::OpDescUtils::CreateOperatorFromNode(out_anchor->GetOwnerNode()));
      af::AscOpOutput op_out(&ops.back(), out_anchor->GetIdx());
      outputs.push_back(std::move(op_out));
    } else {
      external_boundaries.emplace_back(out_anchor, static_cast<int64_t>(parent_in_idx));
      GE_ASSERT_SUCCESS(InsertDataAndLoadNode(vf_graph, out_anchor, iter.second, parent_in_idx));
      ops.push_back(af::OpDescUtils::CreateOperatorFromNode(out_anchor->GetOwnerNode()));
      af::AscOpOutput op_out(&ops.back(), out_anchor->GetIdx());
      outputs.push_back(std::move(op_out));
    }
    ++parent_in_idx;
  }
  // vf_op.x = outputs 经 AssignImpl 一次性完成动态输入实例声明与建边:
  // 此时 vf_op 尚未入图, AddEdgeForNode 会在 impl_graph_ 隐式创建 vf 节点并绑定 operator,
  // 后续不得再次 AddNode(重复绑定失败)。
  // 空 outputs(zero-input cluster)时无任何连接, 显式 AddNode 入图。
  if (!outputs.empty()) {
    vf_op.x = outputs;
  } else {
    GE_ASSERT_NOTNULL(impl_graph_.AddNode(vf_op));
  }
  ge::AscendString str;
  vf_op.GetName(str);
  auto vf_node = impl_graph_.FindNode(str.GetString());
  GE_ASSERT_NOTNULL(vf_node, "Failed to find vf node %s from graph %s.", str.GetString(),
                    impl_graph_.GetName().c_str());

  // Boundary ordinals belong to the subgraph namespace. Resolve them against
  // the actual root input peer anchors after dynamic inputs are connected.
  const auto root_inputs = vf_node->GetAllInDataAnchors();
  for (const auto &boundary : external_boundaries) {
    const auto out_anchor = boundary.first;
    const auto pre_node = out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(pre_node);
    const auto root_input = std::find_if(root_inputs.begin(), root_inputs.end(), [&out_anchor](const auto &in_anchor) {
      return in_anchor->GetPeerOutAnchor() == out_anchor;
    });
    GE_ASSERT_TRUE(root_input != root_inputs.end(), "Root VF input peer anchor is missing for boundary source[%s:%d]",
                   pre_node->GetNamePtr(), out_anchor->GetIdx());
    const auto root_input_index =
        static_cast<int64_t>(std::count_if(root_inputs.begin(), root_input, [](const auto &in_anchor) {
          return in_anchor->GetPeerOutAnchor() != nullptr;
        }));
    const auto data_name = kNamePrefixData + pre_node->GetName() + std::to_string(boundary.second);
    auto data_node = vf_graph.FindNode(data_name.c_str());
    GE_ASSERT_NOTNULL(data_node);
    GE_ASSERT_TRUE(af::AttrUtils::SetInt(data_node->GetOpDesc(), "vf_root_input_index", root_input_index));
  }

  int64_t parent_out_idx = 0;
  bool is_all_input_same_cache = true;
  for (const auto &iter : store_to_peed_in_anchors) {
    const auto &out_anchor = iter.first;
    GE_ASSERT_NOTNULL(out_anchor);
    auto pre_node = std::dynamic_pointer_cast<af::AscNode>(out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(pre_node);
    if (parent_out_idx == 0) {
      vf_node->attr.sched = pre_node->attr.sched;
      vf_node->attr.api = {af::ApiType::kAPITypeCompute, af::ComputeType::kComputeElewise,
                           af::ComputeUnit::kUnitVector};
    } else {
      is_all_input_same_cache =
          is_all_input_same_cache && (vf_node->attr.sched.exec_condition == pre_node->attr.sched.exec_condition);
    }
    vf_node->outputs[parent_out_idx].attr = pre_node->outputs[out_anchor->GetIdx()].attr;
    for (const auto &in_anchor : iter.second) {
      GE_ASSERT_NOTNULL(in_anchor);
      // remove old
      GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(out_anchor, in_anchor));
      // link partition to out
      GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(vf_node->GetOutDataAnchor(parent_out_idx), in_anchor));
    }
    GE_ASSERT_SUCCESS(InsertStoreAndOutputNode(vf_graph, *pre_node, out_anchor->GetIdx(), parent_out_idx));

    ++parent_out_idx;
  }
  if (!is_all_input_same_cache) {
    vf_node->attr.sched.exec_condition = af::ExecuteCondition::kNoCache;
  }

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::MergeContinuousVectorAxis(af::AscGraph &vf_graph) {
  // step 1.生成连续轴组
  std::vector<std::pair<af::AxisId, af::AxisId>> potential_axis_ids;
  for (const auto &node : vf_graph.GetAllNodes()) {
    if (!ScheduleUtils::IsBuffer(node)) {
      GE_ASSERT_TRUE(!node->outputs().empty());
      auto axis_ids = node->outputs[0].attr.vectorized_axis;
      if (axis_ids.size() <= 1UL) {
        return af::SUCCESS;
      }
      potential_axis_ids.reserve(axis_ids.size() - 1);
      for (size_t i = 0UL; i < axis_ids.size() - 1; ++i) {
        potential_axis_ids.emplace_back(axis_ids[i], axis_ids[i + 1]);
      }
      break;
    }
  }
  // step2 进行轴合并
  for (auto it = potential_axis_ids.rbegin(); it != potential_axis_ids.rend();) {
    if (!IsVectorizedAxisContinuous(vf_graph, it->first, it->second)) {
      auto normal_it = it.base();
      ++it;
      potential_axis_ids.erase(normal_it - 1);
    } else {
      ++it;
    }
  }
  std::vector<std::vector<int64_t>> merged_axis_ids = MergeContinuousPairs(potential_axis_ids);
  // step3 剩下的进行合轴
  for (const auto &from_ids : merged_axis_ids) {
    af::AxisPtr node_merge_axis;
    auto iter = from_id_to_merged_axis_.find(from_ids);
    if (iter == from_id_to_merged_axis_.end()) {
      auto merged_axis = impl_graph_.MergeAxis(from_ids);
      node_merge_axis = merged_axis;
      from_id_to_merged_axis_[from_ids] = merged_axis;
    } else {
      node_merge_axis = iter->second;
    }
    // Apply Merge
    for (auto node : vf_graph.GetAllNodes()) {
      if (ScheduleUtils::IsBuffer(node)) {
        continue;
      }
      GELOGD("Apply merged axis id [%ld] to graph [%s] node:[%s].", node_merge_axis->id, vf_graph.GetName().c_str(),
             node->GetNamePtr());
      GE_ASSERT_SUCCESS(ApplyMerge(node, node_merge_axis, from_ids), "Failed to apply axis merge.");
    }
  }

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::SetSubGraphAttrs(af::AscGraph &vf_graph) {
  int64_t tensor_id = 0;
  for (const auto &node : vf_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    // set vectorized axis as sched axis
    GE_ASSERT_TRUE(!node->outputs().empty());
    node->attr.sched.axis = node->outputs[0].attr.vectorized_axis;

    af::Position pos = af::Position::kPositionVecCalc;
    if (ScheduleUtils::IsLoad(node)) {
      pos = af::Position::kPositionVecIn;
    } else if (ScheduleUtils::IsStore(node)) {
      pos = af::Position::kPositionVecOut;
    }

    const bool is_scalar_brc = IsScalarBrc(node);
    for (const auto &tensor : node->outputs()) {
      tensor->attr.mem.tensor_id = tensor_id++;
      tensor->attr.mem.position = pos;

      auto &strides = tensor->attr.vectorized_strides;
      auto &axes = tensor->attr.vectorized_axis;
      // 标量输入(如 ScalarData 经 Data/Load 副本进子图)的 vectorized axis 可为空:
      // 其全零 stride 轴被 RemoveAllZeroStrideVectorizedAxis 清理后不参与向量化循环,
      // 属合法形态, 不构成断言错误。
      if (axes.empty()) {
        continue;
      }
      int64_t loop_axis = kOutLoopAxisId;
      if (is_scalar_brc) {
        // scalar brc在循环外，stride全设置成0
        strides.assign(strides.size(), af::sym::kSymbolZero);
        node->attr.sched.loop_axis = loop_axis;
        continue;
      }

      bool all_zero = std::all_of(strides.rbegin(), strides.rend(), [](const af::Expression &s) {
        return af::SymbolicUtils::StaticCheckEq(s, af::sym::kSymbolZero) == af::TriBool::kTrue;
      });
      if (all_zero) {
        node->attr.sched.loop_axis = loop_axis;
        continue;
      }

      auto iter = std::find_if(strides.rbegin(), strides.rend(), [](const af::Expression &s) {
        return af::SymbolicUtils::StaticCheckEq(s, af::sym::kSymbolOne) == af::TriBool::kTrue;
      });
      if (iter == strides.rend()) {
        node->attr.sched.loop_axis = loop_axis;
        continue;
      }

      size_t idx = strides.size() - 1 - std::distance(strides.rbegin(), iter);
      if (idx < axes.size()) {
        loop_axis = axes[idx];
      } else {
        loop_axis = axes[0];
      }

      node->attr.sched.loop_axis = loop_axis;
    }
  }

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::ModifySubgraphAttrs(af::AscGraph &vf_graph) {
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(vf_graph);
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_GRAPH_SUCCESS(compute_graph->TopologicalSorting(), "TopologicalSorting failed, graph:[%s].",
                          compute_graph->GetName().c_str());
  // step1 remove all zero stride axis
  GE_ASSERT_SUCCESS(RemoveAllZeroStrideVectorizedAxis(vf_graph), "Failed to remove all zero stride vectorized axis");
  // step2 merge_axis
  GE_ASSERT_SUCCESS(MergeContinuousVectorAxis(vf_graph), "Failed to merge continuous vectorized axis");
  // step3 set loop_axis && tensor id tensor attr include position && alloc type
  GE_ASSERT_SUCCESS(SetSubGraphAttrs(vf_graph), "Failed to set tensor attr.");

  // 暂时关闭，后续打开 // step4 reorder axes for brc inline
  // 暂时关闭，后续打开 // GE_ASSERT_SUCCESS(ReorderAxesForBrcInline(vf_graph), "Failed to reorder axes for brc inline,
  // 暂时关闭，后续打开 // Graph[%s].", vf_graph.GetName().c_str());

  // step5 topologic sorting by loop
  GE_ASSERT_SUCCESS(TopologicalSortingForVfGraph(vf_graph), "Failed to do topological sorting for subgraph[%s].",
                    vf_graph.GetName().c_str());
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::BuildSubgraphs() {
  for (const auto &cluster : cluster_dict_.GetAllClusters()) {
    // 不包含隐式广播的单节点不融合
    const auto out_data_nodes = cluster->Nodes().front()->GetOutDataNodes();
    const bool is_arange_single_node =
        cluster->Nodes().size() == 1UL && af::ops::IsOps<af::ascir_op::Arange>(cluster->Nodes().front());
    if (cluster->meta_data_.enable_vf && is_arange_single_node &&
        (cluster->Nodes().front()->GetInControlNodesSize() != 0UL ||
         cluster->Nodes().front()->GetOutControlNodesSize() != 0UL)) {
      GELOGE(af::FAILED, "Arange VF cluster [%zu] has unsupported control edges.", cluster->Id());
      return af::FAILED;
    }
    const bool is_arange_singleton = is_arange_single_node && !out_data_nodes.empty();
    if (!cluster->meta_data_.enable_vf || (cluster->Nodes().size() < kMinVfNodesNum && !is_arange_singleton &&
                                           !ascgen_utils::IsNodeContainsBrcInline(cluster->Nodes().back()))) {
      continue;
    }
    // 1. create vf op && subgraph
    std::string sub_graph_name = impl_graph_.GetName() + "_VfSubgraph_" + std::to_string(subgraph_id_);
    std::string vf_node_name = impl_graph_.GetName() + "_VfNode_" + std::to_string(subgraph_id_);
    ++subgraph_id_;
    af::AscGraph vf_graph(sub_graph_name.c_str());
    af::ascir_op::VectorFunc vf_op(vf_node_name.c_str());
    vf_op.SetAttr("sub_graph_name", sub_graph_name);
    // 2. build subgraph
    GE_ASSERT_SUCCESS(impl_graph_.AddSubGraph(vf_graph));
    GE_ASSERT_SUCCESS(BuildSubgraph(cluster, vf_graph, vf_op));
    (void)ascgen_utils::indirect_load::InheritTemplateRoleIfIL(impl_graph_, vf_node_name, cluster->Nodes().front());
    // step 3. set graph attr
    GE_ASSERT_SUCCESS(ModifySubgraphAttrs(vf_graph));
    sub_graphs_.push_back(vf_graph);
  }

  // reset subgraph axis
  auto all_axis = impl_graph_.GetAllAxis();
  for (auto &sub_graph : sub_graphs_) {
    auto graph_attr = af::AscGraphUtils::GetComputeGraph(sub_graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
    graph_attr->axis = all_axis;
  }

  // 暂时关闭，后续打开 // step 4. add RemovePad for brc inline
  // 暂时关闭，后续打开 // GE_ASSERT_SUCCESS(AddRemovePadForBrcInline(impl_graph_), "Failed to add RemovePad for brc
  // 暂时关闭，后续打开 // inline, Graph[%s].", impl_graph_.GetName().c_str());
  return af::SUCCESS;
}

bool VectorFuncPartitioner::HasDetectedCycle(const Cluster *const src, const Cluster *const dst) {
  if (src->out_nodes_.empty() || dst->inputs_.empty()) {
    return false;
  }

  std::queue<Cluster *> q;
  std::unordered_set<Cluster *> visited;
  for (Cluster *neighbor : src->outputs_) {
    if (neighbor == dst) {
      continue;
    }
    if (visited.find(neighbor) == visited.end()) {
      q.push(neighbor);
      visited.insert(neighbor);
    }
  }

  while (!q.empty()) {
    Cluster *current = q.front();
    q.pop();
    if (current == dst) {
      return true;
    }

    for (Cluster *next : current->outputs_) {
      if (visited.find(next) == visited.end()) {
        visited.insert(next);
        q.push(next);
      }
    }
  }
  return false;
}

af::Status VectorFuncPartitioner::TopologicalSortingForVfGraph(af::AscGraph &graph) {
  std::unordered_set<af::Node *> outer_loop_sequences;
  for (const auto &node : graph.GetAllNodes()) {
    if ((!af::ops::IsOps<af::ascir_op::Output>(node)) && (node->attr.sched.loop_axis == kOutLoopAxisId)) {
      outer_loop_sequences.emplace(node.get());
    }
  }
  const auto ranks = BuildDependencyAwareRanks(graph, outer_loop_sequences);
  const auto func = [&ranks](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    const auto rank1 = ranks.find(node1.get());
    const auto rank2 = ranks.find(node2.get());
    if (rank1 != ranks.end() && rank2 != ranks.end()) {
      return rank1->second < rank2->second;
    }
    return std::make_pair(node1->GetName(), node1->GetOpDescBarePtr()->GetId()) <
           std::make_pair(node2->GetName(), node2->GetOpDescBarePtr()->GetId());
  };

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->TopologicalSorting(func);

  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::ReorderAxesForBrcInline(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetInDataNodes().empty() || node->GetOutDataNodes().empty()) {
      continue;
    }
    const auto &out_attr = node->outputs[0].attr;
    node->attr.sched.loop_axis = FindLastNonBrcAxis(out_attr.vectorized_axis, out_attr.vectorized_strides);
  }
  return af::SUCCESS;
}

af::Status VectorFuncPartitioner::AddRemovePadForBrcInline(af::AscGraph &graph) {
  // step1: 收集所有的Store节点，因为Output只有1个输入且必定是Store
  std::vector<af::NodePtr> store_nodes;
  std::set<af::NodePtr> brc_inline_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsStore(node)) {
      store_nodes.push_back(node);
    }
    if (ascgen_utils::IsNodeContainsBrcInline(node)) {
      brc_inline_nodes.insert(node);
    }
  }
  if (brc_inline_nodes.empty()) {
    GELOGD("Sub graph[%s] not contains brc inline node.", graph.GetName().c_str());
    return af::SUCCESS;
  }

  std::set<af::NodePtr> visited_nodes;
  // step2: 从Store节点倒序遍历，output节点本身不需要取消对齐
  for (const auto &node : store_nodes) {
    const auto &src_nodes = node->GetInDataNodes();
    const auto connect_to_concat = (!src_nodes.empty()) && (src_nodes.at(0U)->GetType() == af::ascir_op::Concat::Type);
    if ((!connect_to_concat) && ScheduleUtils::IsContinuesVecStrides(std::dynamic_pointer_cast<af::AscNode>(node))) {
      GELOGD("Graph[%s] Node[%s] is continuous.", graph.GetName().c_str(), node->GetNamePtr());
      continue;
    }
    GE_ASSERT_SUCCESS(ReverseDfsUnAlignNode(graph, node, visited_nodes));
  }
  visited_nodes.clear();
  return af::SUCCESS;
}
}  // namespace optimize
