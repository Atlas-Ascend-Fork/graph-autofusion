/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "task_generator/indirect_load/indirect_load_graph_utils.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {
af::Status BuildPostReduceLayout(const af::AscNodePtr &reduce, PostReduceLayout &layout, bool &is_legal) {
  is_legal = false;
  GE_ASSERT_TRUE(reduce->inputs().size() == 1UL && reduce->outputs().size() == 1UL,
                 "IndirectLoad post Reduce must be unary, node[%s] input num:%zu, output num:%zu.",
                 reduce->GetNamePtr(), reduce->inputs().size(), reduce->outputs().size());
  const auto input = reduce->inputs()[0];
  const auto output = reduce->outputs()[0];
  GE_ASSERT_NOTNULL(input);
  GE_ASSERT_NOTNULL(output);
  const auto &input_strides = input->attr.strides;
  const auto &output_strides = output->attr.strides;
  layout.axes = output->attr.axis;
  GE_ASSERT_TRUE(input->attr.axis.size() == layout.axes.size() && input->attr.repeats.size() == layout.axes.size() &&
                     output->attr.repeats.size() == layout.axes.size() &&
                     input_strides.size() == output_strides.size() && output_strides.size() == layout.axes.size(),
                 "IndirectLoad post Reduce metadata rank mismatch.");
  const auto reduce_axes = ScheduleUtils::CalcReduceAxes(input_strides, output_strides, layout.axes);
  layout.kinds.clear();
  layout.kinds.reserve(layout.axes.size());
  layout.first_reduce = layout.axes.size();
  for (size_t i = 0UL; i < layout.axes.size(); ++i) {
    const auto input_zero = af::SymbolicUtils::StaticCheckEq(input_strides[i], af::ops::Zero);
    const auto output_zero = af::SymbolicUtils::StaticCheckEq(output_strides[i], af::ops::Zero);
    // 仅拒绝可静态证明矛盾的布局（输入广播 stride=0 但输出非零）。符号 stride 的
    // 零性为 kUnknown 不能作为非法依据：动态 shape 下保留轴 stride 恒为符号表达式，
    // 归约/保留划分由 CalcReduceAxes 依据 dst stride 是否为常量 0 推导。
    if (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kFalse) {
      return af::SUCCESS;
    }
    // 归约轴输出 repeat 恒为 1 且 stride 恒为 0。若某轴输出 repeat=1 但其
    // 输出 stride 的零性无法静态判定，则无法确认它是否为归约轴（可能被改写为
    // 非法布局），保守拒绝；真正保留轴的 repeat 不会退化为常量 1。
    if (af::SymbolicUtils::StaticCheckEq(output->attr.repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue &&
        output_zero == af::TriBool::kUnknown) {
      return af::SUCCESS;
    }
    if (input_zero == af::TriBool::kTrue && output_zero == af::TriBool::kTrue) {
      layout.kinds.emplace_back(ReduceAxisKind::kIgnored);
    } else if (std::find(reduce_axes.begin(), reduce_axes.end(), layout.axes[i]) != reduce_axes.end()) {
      layout.kinds.emplace_back(ReduceAxisKind::kReduced);
      if (layout.first_reduce == layout.axes.size()) {
        layout.first_reduce = i;
      }
    } else {
      layout.kinds.emplace_back(ReduceAxisKind::kRetained);
    }
  }
  is_legal = true;
  return af::SUCCESS;
}

bool HasSupportedReduceSuffix(const PostReduceLayout &layout, size_t begin) {
  bool has_axis = false;
  bool has_reduce_axis = false;
  bool previous_is_reduce = false;
  size_t transitions = 0UL;
  for (size_t i = begin; i < layout.kinds.size(); ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kIgnored) {
      continue;
    }
    const bool is_reduce = layout.kinds[i] == ReduceAxisKind::kReduced;
    if (has_axis && is_reduce != previous_is_reduce) {
      ++transitions;
    }
    has_axis = true;
    has_reduce_axis = has_reduce_axis || is_reduce;
    previous_is_reduce = is_reduce;
  }
  return has_reduce_axis && transitions <= 1UL;
}

af::Status ValidateSimtPostReduceLayout(const af::AscNodePtr &reduce, size_t &boundary, bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  if (layout.first_reduce == layout.axes.size()) {
    is_legal = false;
    return af::SUCCESS;
  }
  boundary = layout.first_reduce;
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

af::Status IsSimtArPostReduce(const af::AscNodePtr &reduce, bool &is_ar) {
  is_ar = false;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  bool is_legal = false;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal || layout.first_reduce == layout.axes.size()) {
    return af::SUCCESS;
  }
  is_ar = std::none_of(layout.kinds.begin() + static_cast<int64_t>(layout.first_reduce), layout.kinds.end(),
                       [](ReduceAxisKind kind) { return kind == ReduceAxisKind::kRetained; });
  return af::SUCCESS;
}

// 形状保持的专用 Norm 节点：归约发生在算子内部、输出沿归约轴仍为满 stride，
// BuildPostReduceLayout 的 stride 推导识别不出其归约轴（first_reduce 恒等于 rank）。
// pattern fusion 仅匹配尾轴归约，ValidateTemplate 的专用分流已据此覆写 boundary
// 为尾轴前一位（outer=保留轴、inner=尾轴 R），各行归约互相独立，与 AR 后置
// Reduce 的行并行语义一致。当前成员为 Softmax（softmax pattern fusion 的产出）；
// 后续新增专用 Norm 算子（如独立的 LayerNorm/RmsNorm 节点）在此处扩展。
bool IsShapePreservingTailAxisNorm(const af::AscNodePtr &reduce) {
  return reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(reduce);
}

af::Status ValidateSimdPostReduceLayout(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce,
                                        bool &is_legal) {
  is_legal = true;
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  PostReduceLayout layout;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, is_legal));
  if (!is_legal) {
    return af::SUCCESS;
  }
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const size_t boundary = axis_index;
  GE_ASSERT_TRUE(boundary <= layout.axes.size(),
                 "IndirectLoad axis %zu is out of range [0, %zu] for post Reduce output.", boundary,
                 layout.axes.size());

  for (size_t i = 0UL; i < boundary; ++i) {
    if (layout.kinds[i] == ReduceAxisKind::kReduced) {
      is_legal = false;
      return af::SUCCESS;
    }
  }
  is_legal = HasSupportedReduceSuffix(layout, boundary);
  return af::SUCCESS;
}

bool HasReduceBeforeGatherAxis(const af::AscNodePtr &indirect_load) {
  const auto reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
  if (reduce == nullptr) {
    return false;
  }
  PostReduceLayout layout;
  bool is_legal = false;
  if (BuildPostReduceLayout(reduce, layout, is_legal) != af::SUCCESS || !is_legal) {
    return false;
  }
  const size_t gather_axis = GetIndirectLoadAxisIndex(indirect_load);
  if (gather_axis == kIndirectLoadInvalidAxisIndex || gather_axis >= layout.kinds.size()) {
    return false;
  }
  for (size_t index = 0UL; index < layout.kinds.size(); ++index) {
    if (layout.kinds[index] == ReduceAxisKind::kReduced) {
      return index < gather_axis;
    }
  }
  return false;
}

// 归约轴推导：复用 Reduce 布局分析的真实 zero 转变规则，不能以任意
// stride 差作为归约依据；保留轴在重排 View 下也可能改变 stride。
std::vector<af::AxisId> CalcReduceAxesFromNode(const af::AscNodePtr &reduce) {
  std::vector<af::AxisId> reduced_axes;
  const auto inputs = reduce->inputs();
  const auto outputs = reduce->outputs();
  if (inputs.empty() || outputs.empty() || inputs.size() != 1UL || outputs.size() != 1UL) {
    return reduced_axes;
  }
  const auto &input_attr = inputs[0]->attr;
  const auto &output_attr = outputs[0]->attr;
  if (input_attr.axis.size() != output_attr.axis.size() || input_attr.strides.size() != output_attr.strides.size() ||
      input_attr.axis.size() != input_attr.strides.size()) {
    return reduced_axes;
  }
  const auto axes = ScheduleUtils::CalcReduceAxes(input_attr.strides, output_attr.strides, output_attr.axis);
  for (const auto axis : axes) {
    reduced_axes.emplace_back(axis);
  }
  return reduced_axes;
}

// 广播恢复轴推导：Broadcast 输入 stride 为 0 的轴即广播恢复轴。
std::vector<af::AxisId> CalcBroadcastAxesFromNode(const af::AscNodePtr &broadcast) {
  std::vector<af::AxisId> broadcast_axes;
  const auto inputs = broadcast->inputs();
  const auto outputs = broadcast->outputs();
  if (inputs.empty() || outputs.empty() || inputs.size() != 1UL || outputs.size() != 1UL) {
    return broadcast_axes;
  }
  const auto &input_attr = inputs[0]->attr;
  const auto &output_attr = outputs[0]->attr;
  if (input_attr.axis.size() != output_attr.axis.size() || input_attr.strides.size() != output_attr.strides.size() ||
      input_attr.axis.size() != input_attr.strides.size()) {
    return broadcast_axes;
  }
  for (size_t i = 0UL; i < input_attr.axis.size(); ++i) {
    const bool zero_stride =
        af::SymbolicUtils::StaticCheckEq(input_attr.strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue;
    if (zero_stride) {
      broadcast_axes.emplace_back(output_attr.axis[i]);
    }
  }
  return broadcast_axes;
}

// Reduce → Broadcast 同轴判定：广播恢复轴集合与归约轴集合完全一致（按轴 ID 集合比较）。
bool IsReduceBroadcastCoaxial(const std::vector<af::AxisId> &reduced_axes,
                              const std::vector<af::AxisId> &broadcast_axes) {
  if (reduced_axes.size() != broadcast_axes.size() || reduced_axes.empty()) {
    return false;
  }
  std::set<af::AxisId> reduced_set(reduced_axes.begin(), reduced_axes.end());
  std::set<af::AxisId> broadcast_set(broadcast_axes.begin(), broadcast_axes.end());
  return reduced_set == broadcast_set;
}

// 从 Reduce 出发沿唯一输出链查找对应 Broadcast：Reduce 输出直接接
// Broadcast，或经 Cast 后接 Broadcast。返回空表示无对应 Broadcast。
af::AscNodePtr FindPairedBroadcast(const af::AscNodePtr &reduce) {
  if (reduce == nullptr) {
    GELOGI("[IndirectLoad] Reduce/Broadcast pairing skipped: Reduce is null.");
    return nullptr;
  }
  NodePath pending{reduce};
  NodeSet visited;
  af::AscNodePtr matched;
  for (size_t cursor = 0UL; cursor < pending.size() && cursor < 64UL; ++cursor) {
    const auto current = pending[cursor];
    if (current == nullptr || !visited.emplace(current.get()).second || current->GetOutDataNodesSize() == 0UL) {
      continue;
    }
    for (const auto &out_node : current->GetOutDataNodes()) {
      const auto successor = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (successor == nullptr || HasControlEdge(successor) || ScheduleUtils::IsReduce(successor)) {
        continue;
      }
      if (IsBroadcastNode(successor)) {
        if (matched != nullptr && matched != successor) {
          GELOGD("[IndirectLoad] Reduce[%s] pairing is ambiguous.", reduce->GetNamePtr());
          return nullptr;
        }
        matched = successor;
        continue;
      }
      // 图优化可能将 Broadcast 融入二元 Elementwise 的输入关系，允许继续
      // 遍历其消费者，但不跨越其他 Reduce 或控制边。
      if (ScheduleUtils::IsElewise(successor)) {
        pending.emplace_back(successor);
      }
    }
  }
  if (matched != nullptr) {
    GELOGD("[IndirectLoad] Reduce[%s] paired with Broadcast[%s].", reduce->GetNamePtr(), matched->GetNamePtr());
  }
  return matched;
}

// 检查 Reduce 的广播配对：每个统计 Reduce 必须有唯一 Broadcast 恢复归约轴且二者同轴。
bool HasPairedBroadcastForReduce(const af::AscNodePtr &reduce) {
  const auto broadcast = FindPairedBroadcast(reduce);
  if (broadcast == nullptr) {
    return false;
  }
  return IsReduceBroadcastCoaxial(CalcReduceAxesFromNode(reduce), CalcBroadcastAxesFromNode(broadcast));
}

bool IsFinalReduce(const af::AscNodePtr &reduce, const af::AscNodePtr &output_store) {
  // 只有 Reduce 直接收敛到最终 Store，或经单个 Cast 收敛到 Store，才允许省略 Broadcast。
  if (reduce == nullptr || output_store == nullptr || reduce->GetOutDataNodesSize() != 1UL) {
    return false;
  }
  const auto successor = std::dynamic_pointer_cast<af::AscNode>(*reduce->GetOutDataNodes().begin());
  return successor == output_store ||
         (af::ops::IsOps<af::ascir_op::Cast>(successor) && successor->GetOutDataNodesSize() == 1UL &&
          *successor->GetOutDataNodes().begin() == output_store);
}

bool HasSameLogicalAxisView(const af::AscTensorAttr &lhs, const af::AscTensorAttr &rhs) {
  return lhs.axis == rhs.axis && PassUtils::IsExprVectorEqual(lhs.repeats, rhs.repeats);
}

// 复合区域视图折叠校验：从入口视图出发沿归约阶段折叠——配对 Broadcast 直接
// 取其输出视图（还原或放大均合法，放大允许超过入口宽度），尾部无配对 Broadcast
// 的 Reduce 将归约轴 repeat 折为 1，Elementwise/Cast 不改变视图。折叠终态须与
// Store 输入视图完全一致。相比逐对视图相等，该口径统一覆盖三类自洽形态：
// 入口==出口（R→B 配对抵消）、入口>出口（尾部归约收缩）、入口<出口（广播放大），
// 仍拒绝区域链路与出口视图对不上的候选。
bool ValidateCompositeRegionOutputView(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis) {
  const auto input_outputs = indirect_load->outputs();
  if (input_outputs.empty() || input_outputs[0] == nullptr) {
    return false;
  }
  const af::AscNodePtr exit_node = analysis.output_store == nullptr ? nullptr : analysis.output_store;
  if (exit_node == nullptr || exit_node->inputs.Size() == 0UL) {
    return false;
  }
  const auto &entry_attr = input_outputs[0]->attr;
  const auto &exit_attr = exit_node->inputs[0].attr;
  if (entry_attr.axis.size() != entry_attr.repeats.size() || exit_attr.axis.size() != exit_attr.repeats.size()) {
    return false;
  }
  if (analysis.composite_reduces.empty()) {
    return HasSameLogicalAxisView(entry_attr, exit_attr);
  }
  std::vector<af::AxisId> folded_axes = entry_attr.axis;
  std::vector<af::Expression> folded_repeats = entry_attr.repeats;
  for (const auto &reduce : analysis.composite_reduces) {
    const auto broadcast = FindPairedBroadcast(reduce);
    if (broadcast != nullptr) {
      const auto broadcast_outputs = broadcast->outputs();
      if (broadcast_outputs.empty() || broadcast_outputs[0] == nullptr ||
          broadcast_outputs[0]->attr.axis.size() != broadcast_outputs[0]->attr.repeats.size()) {
        return false;
      }
      folded_axes = broadcast_outputs[0]->attr.axis;
      folded_repeats = broadcast_outputs[0]->attr.repeats;
      continue;
    }
    for (const auto &axis : CalcReduceAxesFromNode(reduce)) {
      const auto it = std::find(folded_axes.begin(), folded_axes.end(), axis);
      if (it == folded_axes.end()) {
        return false;
      }
      const auto repeat_index = static_cast<size_t>(std::distance(folded_axes.begin(), it));
      if (repeat_index >= folded_repeats.size()) {
        return false;
      }
      folded_repeats[repeat_index] = af::sym::kSymbolOne;
    }
  }
  return folded_axes == exit_node->inputs[0].attr.axis &&
         PassUtils::IsExprVectorEqual(folded_repeats, exit_node->inputs[0].attr.repeats);
}

// Softmax 专用路径的 IO 形态校验：单输入单输出且输入输出 View 与
// IndirectLoad 输出一致，保证 Gather 输出可直接作为 Softmax 的 A/R View。
// 校验通过时通过出参返回输入输出属性，供后续元数据构造使用。
bool TryValidateSoftmaxDedicatedIo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &softmax,
                                   af::AscTensorAttr &input_attr, af::AscTensorAttr &output_attr) {
  const auto softmax_inputs = softmax->inputs();
  const auto softmax_outputs = softmax->outputs();
  const auto indirect_outputs = indirect_load->outputs();
  if (softmax_inputs.empty() || softmax_outputs.empty() || indirect_outputs.empty() || softmax_inputs.size() != 1UL ||
      softmax_outputs.size() != 1UL) {
    GELOGI("[IndirectLoad] Softmax dedicated path rejected: invalid IO, node[%s].", softmax->GetNamePtr());
    return false;
  }
  input_attr = softmax_inputs[0]->attr;
  output_attr = softmax_outputs[0]->attr;
  const auto &indirect_output_attr = indirect_outputs[0]->attr;
  const bool view_consistent = input_attr.axis == indirect_output_attr.axis &&
                               output_attr.axis == indirect_output_attr.axis &&
                               PassUtils::IsExprVectorEqual(input_attr.repeats, indirect_output_attr.repeats) &&
                               PassUtils::IsExprVectorEqual(output_attr.repeats, indirect_output_attr.repeats);
  if (!view_consistent) {
    GELOGI("[IndirectLoad] Softmax dedicated path rejected: view mismatch, node[%s].", softmax->GetNamePtr());
    return false;
  }
  return true;
}

// G ≥ R 布局判定（SIMD 直接融合必要条件）：Softmax 归约轴是输入尾轴，
// 尾轴位置必须在 Gather G 轴位置或内侧；否则由 SIMT/SK 衔接式候选接管。
bool IsSoftmaxReduceInsideOrAtGatherAxis(const af::AscNodePtr &indirect_load, const af::AscTensorAttr &input_attr,
                                         ascir::TemplateId template_id) {
  // 非尾轴 gather（axis < rank-1）时 SIMD 输出 tile 的物理布局带行对齐空洞（行
  // stride 按 2x inner 对齐），SoftmaxARFullLoadExtend 的 {a, r} 隐含 src/dst 稠密
  // 布局假设——读 src 错位、dst 按 a*align32(r) 组织会超出 buffer 分配（生产
  // gather+layernorm 用例的 AIC 341 UB 越界即此组合）。因此仅 SIMD 模板要求 gather
  // 轴为输出尾轴；SIMT 逐元素发射、输出稠密无空洞，非尾轴 gather 同样安全（生产
  // softmax_abs_gather 图 axis=0 的 SIMT 候选不应被拒）。
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  if (axis_index == kIndirectLoadInvalidAxisIndex || input_attr.axis.empty()) {
    return false;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return axis_index == input_attr.axis.size() - 1UL;
  }
  return axis_index <= input_attr.axis.size() - 1UL;
}

// 保存 Softmax 专用路径 NormInfo：归约轴为输入尾轴（尾轴约束已在分流时验证）。
af::Status SaveSoftmaxDedicatedNormInfo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &softmax,
                                        const af::AscTensorAttr &input_attr) {
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  info.entry_node_name = softmax->GetName();
  info.exit_node_name = softmax->GetName();
  info.region_node_names = {softmax->GetName()};
  info.entry_axes = input_attr.axis;
  info.softmax_reduce_axis = input_attr.axis.empty() ? af::kIdNone : input_attr.axis.back();
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  GELOGI("[IndirectLoad] NormInfo saved: Softmax dedicated path, node[%s].", softmax->GetNamePtr());
  return af::SUCCESS;
}

// 保存多 stage 复合区域 NormInfo：每个 Reduce 与配对 Broadcast、归约轴
// 共同写入；出口必须存在且 View 与入口一致，否则淘汰当前候选。
af::Status SaveCompositeNormInfo(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (analysis.output_store == nullptr) {
    GELOGI("[IndirectLoad] Composite Norm rejected: no output store, node[%s].", indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (!ValidateCompositeRegionOutputView(indirect_load, analysis)) {
    GELOGI("[IndirectLoad] Composite Norm rejected: region view folding mismatches the final output view, node[%s].",
           indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  const auto indirect_outputs = indirect_load->outputs();
  if (!indirect_outputs.empty() && indirect_outputs[0] != nullptr) {
    info.entry_axes = indirect_outputs[0]->attr.axis;
  }
  for (const auto &reduce : analysis.composite_reduces) {
    const auto broadcast = FindPairedBroadcast(reduce);
    ascgen_utils::norm::NormStage stage;
    stage.reduce_node_name = reduce->GetName();
    stage.broadcast_node_name = broadcast == nullptr ? "" : broadcast->GetName();
    stage.reduced_axes = CalcReduceAxesFromNode(reduce);
    info.stages.emplace_back(std::move(stage));
    info.region_node_names.emplace_back(reduce->GetName());
    if (broadcast != nullptr) {
      info.region_node_names.emplace_back(broadcast->GetName());
    }
  }
  info.entry_node_name = analysis.post_reduce->GetName();
  info.exit_node_name = analysis.output_store->GetName();
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  GELOGI("[IndirectLoad] NormInfo saved: composite region with %zu stages.", info.stages.size());
  is_candidate_legal = true;
  return af::SUCCESS;
}

// 保存单 Reduce NormInfo：单 Reduce 与复合 Norm 共用区域元数据，完整记录归约后的
// Broadcast/Elementwise 消费链，避免把 Reduce 错误地视为必然的终端节点。
af::Status SaveSingleReduceNormInfo(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce) {
  ascgen_utils::norm::NormInfo info;
  info.kind = ascgen_utils::norm::NormInfo::Kind::kGenericComposite;
  info.entry_node_name = reduce->GetName();
  info.exit_node_name = reduce->GetName();
  info.region_node_names = {reduce->GetName()};
  const auto reduce_inputs = reduce->inputs();
  if (!reduce_inputs.empty() && reduce_inputs[0] != nullptr) {
    info.entry_axes = reduce_inputs[0]->attr.axis;
  }
  PostReduceLayout layout;
  bool layout_legal = false;
  GE_ASSERT_SUCCESS(BuildPostReduceLayout(reduce, layout, layout_legal));
  if (layout_legal) {
    for (size_t i = 0UL; i < layout.kinds.size(); ++i) {
      if (layout.kinds[i] == ReduceAxisKind::kReduced) {
        info.stages.push_back({reduce->GetName(), "", {layout.axes[i]}});
      } else if (layout.kinds[i] == ReduceAxisKind::kRetained) {
        info.preserved_axes.emplace_back(layout.axes[i]);
      }
    }
  }

  NodeSet visited;
  NodePath pending{reduce};
  af::AscNodePtr exit_store;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    if (ScheduleUtils::IsStore(node)) {
      exit_store = node;
      continue;
    }
    if (node != reduce) {
      info.region_node_names.emplace_back(node->GetName());
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(out_node));
    }
  }
  if (exit_store != nullptr) {
    info.exit_node_name = exit_store->GetName();
  }
  GE_ASSERT_SUCCESS(ascgen_utils::norm::SetNormInfo(indirect_load, info));
  GELOGI("[IndirectLoad] NormInfo saved: single Reduce region with %zu nodes.", info.region_node_names.size());
  return af::SUCCESS;
}

// 识别 Gather 后置链中的 Norm 区域并保存元数据：
// - 专用 Softmax：尾轴约束 + IO View + G≥R 校验；
// - 多 Reduce 复合区域：配对同轴校验（收集阶段）+ 出口 View 校验；
// - 单 Reduce：布局推导记录。
af::Status CollectGatherNormInfo(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal, ascir::TemplateId template_id) {
  is_candidate_legal = true;
  if (indirect_load == nullptr || analysis.post_reduce == nullptr) {
    return af::SUCCESS;
  }

  if (af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce)) {
    af::AscTensorAttr softmax_input_attr;
    af::AscTensorAttr softmax_output_attr;
    if (!TryValidateSoftmaxDedicatedIo(indirect_load, analysis.post_reduce, softmax_input_attr, softmax_output_attr) ||
        !IsSoftmaxReduceInsideOrAtGatherAxis(indirect_load, softmax_input_attr, template_id)) {
      is_candidate_legal = false;
      return af::SUCCESS;
    }
    return SaveSoftmaxDedicatedNormInfo(indirect_load, analysis.post_reduce, softmax_input_attr);
  }

  if (!analysis.composite_reduces.empty()) {
    return SaveCompositeNormInfo(indirect_load, analysis, is_candidate_legal);
  }
  return SaveSingleReduceNormInfo(indirect_load, analysis.post_reduce);
}

// Reduce 后置 Norm 消费链只允许保持单路径的 Cast/Broadcast/Elementwise，最终收敛到 Store。
// 不支持的拓扑在候选级淘汰，避免跨分支或跨 Reduce 的依赖进入当前模板。
// allow_multi_consumer：SK 消费子图走通用调度，归约结果可同时供给物化 Store（如被
// 融合图外部的 eq 校验分支消费而保留的输出）与下游 Elementwise 链；此时每个直连
// 分支仍须满足既有 Cast/Broadcast/Elementwise 单链约束并终止于 Store。
af::Status ValidateReduceOutputCandidate(const af::AscNodePtr &reduce, bool &is_valid,
                                         const bool allow_multi_consumer) {
  is_valid = false;
  if (reduce == nullptr || reduce->GetOutDataNodesSize() == 0UL) {
    return af::SUCCESS;
  }
  if (reduce->GetOutDataNodesSize() != 1UL && !allow_multi_consumer) {
    return af::SUCCESS;
  }
  NodePath pending;
  for (const auto &out_node : reduce->GetOutDataNodes()) {
    const auto successor = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if (successor == nullptr) {
      return af::SUCCESS;
    }
    pending.emplace_back(successor);
  }
  NodeSet visited;
  bool reached_store = false;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    if (ScheduleUtils::IsStore(node)) {
      reached_store = true;
      continue;
    }
    if (HasControlEdge(node) || ScheduleUtils::IsReduce(node)) {
      return af::SUCCESS;
    }
    const bool supported = af::ops::IsOps<af::ascir_op::Cast>(node) || af::ops::IsOps<af::ascir_op::Broadcast>(node) ||
                           ScheduleUtils::IsElewise(node);
    if (!supported || node->GetOutDataNodesSize() == 0UL) {
      return af::SUCCESS;
    }
    // A fused normalization may expose both the reduced statistic and the
    // normalized payload.  The common prefix after Reduce can therefore fan
    // out (for example, Add -> Store and Add -> Pow -> Mul -> Store).  Both
    // branches are valid when they remain unary and terminate at Stores; do
    // not reject the candidate merely because an intermediate node has two
    // consumers.
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto successor = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (successor == nullptr) {
        return af::SUCCESS;
      }
      pending.emplace_back(successor);
    }
  }
  is_valid = reached_store;
  return af::SUCCESS;
}

// kUnsupported 同时覆盖动态符号视图与 Reduce/Broadcast 输出的零 stride 约定，二者均为
// 既有合法路径；仅当视图可静态判定且不含零 stride 时才视为异常视图并保守拒绝。
bool IsStaticallyIrregularStoreTarget(const std::vector<af::Expression> &strides, const af::Expression &offset) {
  if (!offset.IsConstExpr()) {
    return false;
  }
  for (const af::Expression &stride : strides) {
    if (!stride.IsConstExpr() || af::SymbolicUtils::StaticCheckEq(stride, af::ops::Zero) == af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

af::Status ClassifyStoreTargetAndGate(const af::AscNodePtr &store, ascir::TemplateId template_id,
                                      bool &is_candidate_legal) {
  const auto &target = store->outputs()[0]->attr;
  GE_ASSERT_TRUE(target.repeats.size() == target.strides.size() && !target.repeats.empty(),
                 "IndirectLoad Store[%s] target view is invalid.", store->GetNamePtr());
  af::Expression offset = af::ops::Zero;
  if (store->attr.ir_attr != nullptr) {
    (void)store->attr.ir_attr->GetAttrValue("offset", offset);
  }
  ascgen_utils::indirect_load::IndirectLoadStoreKind kind =
      ascgen_utils::indirect_load::IndirectLoadStoreKind::kUnsupported;
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::ClassifyIndirectLoadStoreTarget(target.repeats, target.strides, offset, kind));
  bool accepted = false;
  if (kind == ascgen_utils::indirect_load::IndirectLoadStoreKind::kDense) {
    // SIMD/SK 尾部 Store 走通用 Store 路径；SIMT Store 体做 offset/stride 映射。
    accepted = true;
  } else if (kind == ascgen_utils::indirect_load::IndirectLoadStoreKind::kSliceStrided) {
    // SIMD 尾部 Store 走通用 strided MTE3 路径；SIMT Store 体做坐标×stride+offset 映射；SK 恒拒绝。
    accepted = template_id != ascir::TemplateId::kIndirectLoadSK;
  } else if (!IsStaticallyIrregularStoreTarget(target.strides, offset)) {
    accepted = true;
  }
  if (!accepted) {
    is_candidate_legal = false;
    GELOGI("[IndirectLoad] Reject candidate[%d]: Store[%s] target kind[%d] is not supported.",
           static_cast<int32_t>(template_id), store->GetNamePtr(), static_cast<int32_t>(kind));
  }
  return af::SUCCESS;
}

af::Status ValidateOutputStoreTargets(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                      bool &is_candidate_legal) {
  std::vector<af::AscNodePtr> stores;
  NodeSet visited{indirect_load.get()};
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
    roots.emplace_back(consumer);
  }
  const auto visit = [&stores](const af::AscNodePtr &current, bool &stop) -> af::Status {
    if (ScheduleUtils::IsStore(current)) {
      stores.emplace_back(current);
      stop = true;
      return af::SUCCESS;
    }
    // Reduce 后的 Norm 消费链由 ValidateReduceOutputCandidate 统一校验；这里继续遍历，
    // 使 TrueDiv/Broadcast/Elementwise 后的最终 Store 参与输出目标分类。
    return af::SUCCESS;
  };
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(roots, visited, visit));
  for (const af::AscNodePtr &store : stores) {
    GE_ASSERT_SUCCESS(ClassifyStoreTargetAndGate(store, template_id, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  return af::SUCCESS;
}

// Traverse all output branches once, collecting Store/Reduce boundaries and validating each Reduce successor.
af::Status CollectOutputBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis,
                                   bool &is_candidate_legal, const bool allow_multi_consumer_reduce) {
  NodeSet visited{indirect_load.get()};
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto out_asc_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(out_asc_node, "IndirectLoad output successor is invalid.");
    roots.emplace_back(out_asc_node);
  }
  std::vector<af::AscNodePtr> reduce_nodes;
  const auto visit = [&analysis, &reduce_nodes](const af::AscNodePtr &current, bool &stop) -> af::Status {
    if (ScheduleUtils::IsStore(current)) {
      if (analysis.output_store == nullptr) {
        analysis.output_store = current;
      }
      stop = true;
      return af::SUCCESS;
    }
    if (ScheduleUtils::IsReduce(current)) {
      // 收集全部 Reduce，并继续遍历后续链路，以识别复合区域和最终 Store。
      reduce_nodes.emplace_back(current);
      if (analysis.post_reduce == nullptr) {
        analysis.post_reduce = current;
      }
      return af::SUCCESS;
    }
    return af::SUCCESS;
  };
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(roots, visited, visit));

  if (reduce_nodes.size() <= 1UL) {
    // 单 Reduce 作为最小 Norm 区域：允许归约后继续经过归一化 Elementwise 链，
    // 但必须保持单路径并最终收敛到 Store。
    if (analysis.post_reduce != nullptr) {
      bool output_valid = false;
      GE_ASSERT_SUCCESS(ValidateReduceOutputCandidate(analysis.post_reduce, output_valid, allow_multi_consumer_reduce));
      if (!output_valid) {
        is_candidate_legal = false;
      }
    }
    return af::SUCCESS;
  }

  // 中间 Reduce 必须与 Broadcast 同轴；最终直接收敛到 Store 的 Reduce 可无 Broadcast。
  for (size_t index = 0UL; index < reduce_nodes.size(); ++index) {
    const auto &reduce = reduce_nodes[index];
    // 末尾 Reduce 的无 Broadcast 例外仅适用于整个后置区域的最终计算节点。
    const bool final_reduce = index + 1UL == reduce_nodes.size() && IsFinalReduce(reduce, analysis.output_store);
    if (!HasPairedBroadcastForReduce(reduce) && !final_reduce) {
      is_candidate_legal = false;
      GELOGI("[IndirectLoad] Composite region rejected: Reduce[%s] has no coaxial Broadcast pair.",
             reduce->GetNamePtr());
      return af::SUCCESS;
    }
  }
  analysis.composite_reduces = std::move(reduce_nodes);
  GELOGI("[IndirectLoad] Composite region detected with %zu paired Reduce/Broadcast stages.",
         analysis.composite_reduces.size());
  return af::SUCCESS;
}

af::Status CollectRewrittenBoundaries(const af::AscNodePtr &indirect_load, RewrittenGraphAnalysis &analysis,
                                      bool &is_candidate_legal, const bool allow_multi_consumer_reduce) {
  analysis.input_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  analysis.input_boundary = GetLoadTransposeSource(analysis.input_root);
  if (analysis.input_boundary == nullptr) {
    analysis.input_boundary = analysis.input_root;
  }
  analysis.index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_SUCCESS(CollectOutputBoundaries(indirect_load, analysis, is_candidate_legal, allow_multi_consumer_reduce));
  return af::SUCCESS;
}
}  // namespace indirect_load_detail
}  // namespace optimize
