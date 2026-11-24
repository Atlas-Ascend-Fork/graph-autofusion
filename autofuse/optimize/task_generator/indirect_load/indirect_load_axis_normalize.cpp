/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

af::Status BuildSimdInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                              const char *name, ascir::AxisId &input_inner_axis) {
  GE_ASSERT_TRUE(!input_producer->outputs().empty(), "IndirectLoad SIMD input tensor producer has no output.");
  const auto input_axes = input_producer->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(axis_index < input_axes.size(), "IndirectLoad SIMD input axis index %zu is out of range [0, %zu).",
                 axis_index, input_axes.size());
  std::vector<ascir::AxisId> input_inner_axes(input_axes.begin() + static_cast<int64_t>(axis_index), input_axes.end());
  GE_ASSERT_SUCCESS(MergeAxesForTemplate(graph, input_inner_axes, name, input_inner_axis));
  return af::SUCCESS;
}

af::Status BuildSkInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t axis_index,
                                 ascir::AxisId &input_inner_axis) {
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  return BuildSimdInnerAxis(graph, input_boundary, axis_index, kInputInnerAxisName, input_inner_axis);
}

// SK 模板 IL 节点的 tmp 由串行窗口循环复用：单次 IndirectLoadSk 调用仅驻留固定 tile 行数
// （tile_inner 尺寸）的输入与 offsets；tile_outer 轴上 block 切分出的每核串行循环次数
// （<tile_outer 名>b_size）不进入驻留量，否则 UB 上限会与多核切分下限互相矛盾致 tiling
// 无解。填充条件：静态 shape 下全量驻留估算超出平台 UB（SK 会被 StaticUbTemplateFilter
// 误淘汰的形态）；动态 shape 下全量驻留量不可静态判定，同样填充。平台信息不可得 / outer
// 退化（size=1）时不填充，保持全量估算语义。
af::Status FillSkWindowReuseInfo(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                 af::AxisId outer_axis, af::AxisId tile_outer_axis) {
  const auto *outer = graph.FindAxis(outer_axis);
  const auto *tile_outer = graph.FindAxis(tile_outer_axis);
  if (outer == nullptr || tile_outer == nullptr ||
      af::SymbolicUtils::StaticCheckEq(outer->size, af::sym::kSymbolOne) == af::TriBool::kTrue) {
    return af::SUCCESS;
  }
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view));
  if (logical_view.index.sizes.empty() || logical_view.index.sizes.size() != logical_view.input.sizes.size() ||
      logical_view.input.sizes.size() != logical_view.input.strides.size() ||
      boundary > logical_view.index.sizes.size()) {
    return af::SUCCESS;
  }
  const auto inputs = indirect_load->inputs();
  if (inputs.size() <= ascgen_utils::indirect_load::kInputTensorIndex ||
      inputs[ascgen_utils::indirect_load::kInputTensorIndex] == nullptr) {
    return af::SUCCESS;
  }
  int64_t ub_size = 0;
  if (!TryGetIndirectLoadPlatformUbSize(ub_size)) {
    return af::SUCCESS;
  }
  // 全量驻留 tmp 估算，公式与 32B 对齐逐项镜像 CalcTmpBufSize/CalcIndirectLoadSkTmpSize 的全量路径：
  // Align(outer×slice×dtype, 32) + 4×Π(output sizes)，再整体 Align(·, 32)。任一维非正即放弃，回退历史
  // 全量语义；任一维动态则驻留量不可判定，改按窗口语义建模。
  bool has_dynamic_dim = false;
  int64_t offsets_bytes = static_cast<int64_t>(sizeof(uint32_t));
  for (const auto &repeat : logical_view.output.sizes) {
    int64_t value = 0;
    if (!repeat.GetConstValue(value)) {
      has_dynamic_dim = true;
      break;
    }
    if (value <= 0) {
      return af::SUCCESS;
    }
    offsets_bytes *= value;
  }
  const int64_t dtype_size =
      static_cast<int64_t>(af::GetSizeByDataType(inputs[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype));
  int64_t outer_rows = 1;
  if (!has_dynamic_dim) {
    for (size_t dim = 0UL; dim < boundary; ++dim) {
      int64_t value = 0;
      if (!logical_view.index.sizes[dim].GetConstValue(value)) {
        has_dynamic_dim = true;
        break;
      }
      if (value <= 0) {
        return af::SUCCESS;
      }
      outer_rows *= value;
    }
  }
  int64_t slice_elems = 1;
  if (!has_dynamic_dim) {
    for (size_t dim = boundary; dim < logical_view.index.sizes.size(); ++dim) {
      const auto &repeat = dim == boundary ? logical_view.input.sizes[dim] : logical_view.index.sizes[dim];
      int64_t size_value = 0;
      int64_t stride_value = 0;
      if (!repeat.GetConstValue(size_value)) {
        has_dynamic_dim = true;
        break;
      }
      if (size_value <= 0 || !logical_view.input.strides[dim].GetConstValue(stride_value)) {
        return af::SUCCESS;
      }
      slice_elems += (size_value - 1) * stride_value;
    }
  }
  if (!has_dynamic_dim) {
    constexpr int64_t kUbBlockAlignBytes = 32;
    const int64_t input_bytes = outer_rows * slice_elems * dtype_size;
    const int64_t aligned_input = (input_bytes + kUbBlockAlignBytes - 1) / kUbBlockAlignBytes * kUbBlockAlignBytes;
    const int64_t estimate =
        (aligned_input + offsets_bytes + kUbBlockAlignBytes - 1) / kUbBlockAlignBytes * kUbBlockAlignBytes;
    if (estimate <= ub_size) {
      return af::SUCCESS;
    }
  }
  const auto *tile_inner = graph.FindAxis(tile_outer->split_pair_other_id);
  GE_ASSERT_NOTNULL(tile_inner);
  GE_ASSERT_TRUE(tile_inner->type == af::Axis::Type::kAxisTypeTileInner);
  logical_view.window_outer_count = tile_inner->size;
  logical_view.window_iter_axes = {tile_outer_axis};
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, logical_view));
  return af::SUCCESS;
}

// Only a standalone direct-GM SIMT region can change the scheduling unit
// without changing a normal scheduled consumer's UB tile or reduction window.
// Bound the grouped extent to the existing signed block-offset domain.
int64_t GetSimtGroupSize(const af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                         const af::Axis &outer) {
  int64_t total = 0L;
  if (template_id != ascir::TemplateId::kIndirectLoadSimt ||
      ascgen_utils::indirect_load::GetAllIndirectLoadNodes(graph).size() != 1UL ||
      ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load) != nullptr ||
      !outer.size.GetConstValue(total) || total <= 0L || total > std::numeric_limits<int32_t>::max()) {
    return 1L;
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() != "Data" && node->GetType() != "Output" &&
        !ascgen_utils::indirect_load::GetTemplateBehavior(node).uses_direct_gm_pipeline) {
      return 1L;
    }
  }
  constexpr int64_t kWarpElements = 32L;
  return std::min(kWarpElements, total);
}

af::Status NormalizeAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                    ascir::TemplateId template_id, ascir::AxisId input_inner_axis,
                                    ascir::AxisId index_inner_axis, bool simd_index_uses_output_inner_axis,
                                    bool solve_tile_size) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad output axis is empty.");
  ascir::AxisId outer_axis = af::kIdNone;
  ascir::AxisId inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildAxisViewByBoundary(graph, output_axes, boundary, outer_axis, inner_axis));
  if (simd_index_uses_output_inner_axis) {
    index_inner_axis = inner_axis;
  }
  const af::Axis *outer = graph.FindAxis(outer_axis);
  GE_ASSERT_NOTNULL(outer, "IndirectLoad outer axis %ld is not found.", outer_axis);
  const bool synthetic_outer =
      outer->from.empty() && std::find(indirect_load->attr.sched.axis.begin(), indirect_load->attr.sched.axis.end(),
                                       outer_axis) == indirect_load->attr.sched.axis.end();
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  if (solve_tile_size) {
    // tile 行数不在图改写期固定：TemplateAxes 不注解 tile 轴，调度期 TileTiling 发现
    // prebuilt pair 为空后走通用 TileSplit，按 UB 容量求解 tile 内行数。
    GE_ASSERT_TRUE(inner_axis != af::kIdNone, "IndirectLoad solvable tile split requires a non-empty inner view.");
  } else {
    GE_ASSERT_SUCCESS(CreateFixedTileSplit(graph, outer_axis, tile_outer_axis, tile_inner_axis,
                                           GetSimtGroupSize(graph, indirect_load, template_id, *outer)));
  }
  std::vector<af::AxisId> vectorized_axes;
  if (inner_axis != af::kIdNone) {
    const auto *inner = graph.FindAxis(inner_axis);
    GE_ASSERT_NOTNULL(inner, "IndirectLoad inner axis %ld is not found.", inner_axis);
    vectorized_axes =
        inner->type == ascir::Axis::Type::kAxisTypeMerged ? inner->from : std::vector<af::AxisId>{inner_axis};
  } else {
    // A SIMT candidate without post Reduce uses the complete output view as
    // the outer view.  The IndirectLoad/its fused direct-GM path intentionally
    // keeps an empty vectorized view, while ordinary fan-out branches still
    // need the tile-inner axis for alignment and vector-function partitioning.
    GE_ASSERT_TRUE(tile_inner_axis != af::kIdNone, "IndirectLoad tile inner axis is missing.");
    vectorized_axes.emplace_back(tile_inner_axis);
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  axes.outer_axis = outer_axis;
  axes.inner_axis = inner_axis;
  axes.input_inner_axis = input_inner_axis;
  axes.index_inner_axis = index_inner_axis;
  axes.tile_outer_axis = tile_outer_axis;
  axes.tile_inner_axis = tile_inner_axis;
  axes.vectorized_axes = std::move(vectorized_axes);
  axes.synthetic_outer = synthetic_outer;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(indirect_load, axes));
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    GE_ASSERT_SUCCESS(FillSkWindowReuseInfo(graph, indirect_load, boundary, outer_axis, tile_outer_axis));
  }
  return af::SUCCESS;
}

af::Status SeedPostReduceInputVectorizedView(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce) {
  if (reduce == nullptr) {
    return af::SUCCESS;
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes));
  GE_ASSERT_TRUE(!axes.vectorized_axes.empty(), "IndirectLoad post Reduce vectorized axes are empty.");
  GE_ASSERT_TRUE(reduce->inputs.Size() == 1UL, "IndirectLoad post Reduce must have one input.");
  auto &input = reduce->inputs[0].attr;
  GE_ASSERT_TRUE(input.axis.size() == input.strides.size(), "IndirectLoad post Reduce input view is invalid.");
  input.vectorized_axis = axes.vectorized_axes;
  input.vectorized_strides.clear();
  input.vectorized_strides.reserve(input.vectorized_axis.size());
  for (const auto axis : input.vectorized_axis) {
    const auto axis_it = std::find(input.axis.begin(), input.axis.end(), axis);
    GE_ASSERT_TRUE(axis_it != input.axis.end(), "IndirectLoad post Reduce input axis[%ld] is missing.", axis);
    input.vectorized_strides.emplace_back(
        input.strides[static_cast<size_t>(std::distance(input.axis.begin(), axis_it))]);
  }
  return af::SUCCESS;
}

// [组成员 value tensor 视图播种] 单 gather post-Reduce 的 IL 输出经输出重定向与 reduce 输入
// 为同一 tensor，SeedPostReduceInputVectorizedView 一步完成向量化视图播种；组路径无此 id
// 绑定——成员各自持有 value tensor，且 kSimtOp（preserves_vectorized_axis）使调度期
// TileSplit 的全局向量化写入跳过它们（视图保持任务生成期状态）。actual_size 经
// CalcVectorizedElementCount 读该视图，为空则恒为 1（即使逐行 tile，每次调用也
// 只写 1 个元素而非整行）。此处按共享模板的 vectorized 轴（boundary=first_reduce 时为
// inner 归约段的成员轴，仍在组员原轴空间内——组配对约束保证全组输出轴空间一致）为每个
// 成员的 value tensor 播种向量化视图，使 actual_size = 每行元素数；调度期 tensor merge
// 对播种轴的重映射与单 gather 同构（inner 成员轴重映射为 merged inner，不触碰 outer 的
// SelectLoopAxis 判定）。视图缺轴/视图非法是候选形态问题而非程序错误——NormalizeMemberAxes
// 跳过复合轴（from 非空）与分核/分 tile 轴的维（不归一到组共享轴）时，模板 vectorized 轴
// （merged inner 的 from 含成员轴）可能不在某成员视图内，is_candidate_legal=false 拒绝当前
// 候选（候选图为独立副本，丢弃无害），由其余候选/回退承接。
af::Status SeedGroupMemberValueVectorizedViews(const std::vector<af::AscNodePtr> &members, bool &is_candidate_legal) {
  ascgen_utils::indirect_load::TemplateAxes axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(members.front(), axes));
  if (axes.vectorized_axes.empty()) {
    is_candidate_legal = false;
    GELOGI("[IndirectLoad] Reject gather group candidate: shared template vectorized axes are empty.");
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &member : members) {
    GE_ASSERT_TRUE(!member->outputs().empty() && member->outputs()[0] != nullptr,
                   "IndirectLoad gather group member[%s] has no value output.", member->GetNamePtr());
    auto &value = member->outputs()[0]->attr;
    if (value.axis.size() != value.strides.size()) {
      is_candidate_legal = false;
      GELOGI("[IndirectLoad] Reject gather group candidate: member[%s] value view is invalid.", member->GetNamePtr());
      return af::SUCCESS;
    }
    for (const auto axis : axes.vectorized_axes) {
      if (std::find(value.axis.begin(), value.axis.end(), axis) == value.axis.end()) {
        is_candidate_legal = false;
        GELOGI("[IndirectLoad] Reject gather group candidate: member[%s] value view lacks vectorized axis[%ld].",
               member->GetNamePtr(), axis);
        return af::SUCCESS;
      }
    }
    value.vectorized_axis = axes.vectorized_axes;
    value.vectorized_strides.clear();
    value.vectorized_strides.reserve(value.vectorized_axis.size());
    for (const auto axis : value.vectorized_axis) {
      const auto axis_it = std::find(value.axis.begin(), value.axis.end(), axis);
      value.vectorized_strides.emplace_back(
          value.strides[static_cast<size_t>(std::distance(value.axis.begin(), axis_it))]);
    }
  }
  return af::SUCCESS;
}

af::Status RestoreSkTemplateAxes(std::vector<ascir::ImplGraph> &grouped_graphs) {
  for (auto &graph : grouped_graphs) {
    af::AscNodePtr indirect_load;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(graph, indirect_load));
    if (indirect_load == nullptr) {
      continue;
    }
    const auto restored_template = ascir::GetTemplateIdOrDefault(*indirect_load);
    const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
    GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                   indirect_load->GetNamePtr());
    if (restored_template == ascir::TemplateId::kIndirectLoadSimt && ascir::IsSkSegmentedSimt(*indirect_load)) {
      // [SK分段SIMT]
      // 分区后子图轴表仅含节点引用的原始轴，候选图期归一新建的模板轴
      // 不在其中，调度期 BuildIndirectLoadAxisGroup 的 FindAxis 会失败。与原 SK
      // 相同， 在子图内重新执行 SIMT 归一。post 链已被 output workspace
      // 边界隔断为独立子图， 本子图内 IL 输出直连边界 Store，无
      // post-Reduce——与主线无 post-Reduce SIMT 同语义：boundary=输出
      // rank，outer=完整输出视图（SIMT 逐元素枚举全部输出）， 而非 axis
      // 前缀（那会使 outer 退化为合成单轴、发射数=1）。
      const auto simt_output_axes = indirect_load->outputs()[0]->attr.axis;
      GE_ASSERT_SUCCESS(NormalizeSimtAxesForTemplate(graph, indirect_load, simt_output_axes.size(),
                                                     /*solve_tile_size=*/false));
      continue;
    }
    GE_ASSERT_TRUE(restored_template == ascir::TemplateId::kIndirectLoadSK,
                   "IndirectLoad partitioned graph[%s] has unexpected template.", graph.GetName().c_str());
    ascir::AxisId input_inner_axis = af::kIdNone;
    GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
    // 与候选生成保持一致：axis=0 的退化形态改用输出 rank 作 boundary，避免
    // outer 退化为合成单轴导致 IL 全量驻留被 UB filter 淘汰。
    const size_t sk_boundary = axis_index == 0UL ? indirect_load->outputs()[0]->attr.axis.size() : axis_index;
    GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, sk_boundary, ascir::TemplateId::kIndirectLoadSK,
                                               input_inner_axis, af::kIdNone));
  }
  return af::SUCCESS;
}

af::Status ReplaceAxisPrefix(std::vector<af::AxisId> &target_axes, const std::vector<af::AxisId> &output_axes,
                             size_t axis_index) {
  GE_ASSERT_TRUE(target_axes.size() >= axis_index, "IndirectLoad SIMD input region has invalid axis rank.");
  std::copy_n(output_axes.begin(), axis_index, target_axes.begin());
  return af::SUCCESS;
}

af::Status ReplaceRegionAxisPrefix(const NodePath &region, const std::vector<af::AxisId> &output_axes,
                                   size_t axis_index, bool is_input_region) {
  for (const af::AscNodePtr &node : region) {
    if (!is_input_region && IsInputDataSource(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const bool should_replace = is_input_region ? ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)
                                                : role == ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre;
    if (!should_replace) {
      continue;
    }
    GE_ASSERT_SUCCESS(ReplaceAxisPrefix(node->attr.sched.axis, output_axes, axis_index));
    for (const auto &output : node->outputs()) {
      GE_ASSERT_SUCCESS(ReplaceAxisPrefix(output->attr.axis, output_axes, axis_index));
    }
  }
  return af::SUCCESS;
}

af::Status NormalizeSimdAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                        const RewrittenGraphAnalysis &analysis, bool solve_tile_size) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMD output axis is empty.");
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_NOTNULL(analysis.input_root, "IndirectLoad SIMD input producer is missing.");
  GE_ASSERT_SUCCESS(BuildSimdInnerAxis(graph, analysis.input_root, axis_index, kInputInnerAxisName, input_inner_axis));
  // per-IL 输入内轴提示：调度期 ApplyIndirectLoadNodeAxes
  // 优先读取节点提示值，使多 gather 组中各路 input 边界使用各自的输入内轴；单
  // gather 场景提示值与全局模板轴一致，行为不变。
  for (const af::AscNodePtr &node : analysis.input_region) {
    if (ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimdInputInnerAxis(node, input_inner_axis));
    }
  }
  // [per-IL index 内轴] 不再单独
  // BuildSimdInnerAxis(kIndexInnerAxisName)：该轴在 NormalizeAxesForTemplate
  // 内部会被覆写为 inner_axis（simd_index_uses_output_inner_axis=true） 或保持
  // kIdNone（false），单独创建只产生无人消费的孤儿轴。
  const auto index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  GE_ASSERT_NOTNULL(index_root, "IndirectLoad SIMD index producer is missing.");
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.input_region, output_axes, axis_index, true));
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(analysis.index_region, output_axes, axis_index, false));
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, axis_index, ascir::TemplateId::kIndirectLoadSimd,
                                             input_inner_axis, af::kIdNone, analysis.simd_index_uses_output_inner_axis,
                                             solve_tile_size));
  // [per-IL index 内轴 hint] 与 input hint 同点：primary 的 index-pre
  // 节点也写入 hint。 取 TemplateAxes
  // 归一化后的最终值（simd_index_uses_output_inner_axis=true 时
  // NormalizeAxesForTemplate 内部已将 index_inner_axis 覆写为
  // inner_axis），保证 hint
  // 与调度器回退值（indirect_load_info_.axes.index_inner_axis）一致。
  // 仅写有效轴：kIdNone hint ≡ 无 hint（调度器回退全局），绝不写入——共享 index
  // 链 节点会落在多个成员的 region 内，后处理的 flag=false 成员写入 kIdNone
  // 会把先处理 的 flag=true 成员写的有效 hint 覆盖掉（N≥3
  // 或成员顺序不同时触发）。
  ascgen_utils::indirect_load::TemplateAxes final_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, final_axes));
  for (const af::AscNodePtr &node : analysis.index_region) {
    if (ascgen_utils::indirect_load::GetTemplateRole(node) ==
            ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre &&
        final_axes.index_inner_axis != af::kIdNone) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimdIndexInnerAxis(node, final_axes.index_inner_axis));
    }
  }
  return af::SUCCESS;
}

// 多 gather 共享轴归一化：primary 完整建轴（outer/inner/tile/vectorized 与单
// gather 完全一致）， 其余各成员复用共享轴、仅按需构建自己的 input 内轴（各路
// input 的 dim 轴尺寸可以不同）； 各成员区域节点写入其 per-IL
// 内轴提示。组配对约束保证组内各路输出 axis id 完全一致， 因此 primary
// 的切分规则与循环组织形式天然对其余各路生效。
af::Status NormalizeSimdAxesForMember(af::AscGraph &graph, const af::AscNodePtr &primary,
                                      const RewrittenGraphAnalysis &primary_analysis, const af::AscNodePtr &member,
                                      const RewrittenGraphAnalysis &member_analysis) {
  ascgen_utils::indirect_load::TemplateAxes shared_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(primary, shared_axes));
  const size_t axis_index = GetIndirectLoadAxisIndex(member);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 member->GetNamePtr());
  const auto member_output_axes = member->outputs()[0]->attr.axis;
  GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(member_analysis.input_region, member_output_axes, axis_index, true));
  if (member_analysis.simd_index_uses_output_inner_axis) {
    GE_ASSERT_SUCCESS(ReplaceRegionAxisPrefix(member_analysis.index_region, member_output_axes, axis_index, false));
  }
  // 成员的 input 内轴：与 primary
  // 输入轴后缀完全一致（如两路读同一输入）时复用，避免 对相同 from 轴重复
  // MergeAxis；否则按成员自己的 input 构建独立内轴。
  GE_ASSERT_NOTNULL(member_analysis.input_root, "IndirectLoad SIMD input producer is missing.");
  const auto primary_input_axes = primary_analysis.input_root->outputs()[0]->attr.axis;
  const auto member_input_axes = member_analysis.input_root->outputs()[0]->attr.axis;
  const bool same_input_suffix =
      primary_input_axes.size() == member_input_axes.size() &&
      std::equal(primary_input_axes.begin() + static_cast<int64_t>(axis_index), primary_input_axes.end(),
                 member_input_axes.begin() + static_cast<int64_t>(axis_index));
  ascir::AxisId member_input_inner = shared_axes.input_inner_axis;
  if (!same_input_suffix) {
    GE_ASSERT_SUCCESS(
        BuildSimdInnerAxis(graph, member_analysis.input_root, axis_index, kInputInnerAxisName, member_input_inner));
  }
  ascgen_utils::indirect_load::TemplateAxes member_axes = shared_axes;
  member_axes.input_inner_axis = member_input_inner;
  // simd_index_uses_output_inner_axis 为 true 时，index 内轴即共享输出内轴（单
  // gather 流程中 该值同样由输出轴覆写而来）；为 false 时保持空，与单 gather
  // 语义一致。
  member_axes.index_inner_axis =
      member_analysis.simd_index_uses_output_inner_axis ? shared_axes.inner_axis : af::kIdNone;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(member, member_axes));
  for (const af::AscNodePtr &node : member_analysis.input_region) {
    if (ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node)) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimdInputInnerAxis(node, member_input_inner));
    }
  }
  // [per-IL index 内轴 hint] 与 input hint 同点：各成员 index
  // 链形态可不同（Broadcast → 有效轴；计算链 → kIdNone），调度器优先读节点
  // hint、无 hint 回退全局。 kSimdIndexPre 角色仅在
  // simd_index_uses_output_inner_axis=true 时赋角，此时轴必有效。
  // 仅写有效轴（与 primary 侧同因）：共享 index 链节点落在多个成员 region
  // 内，flag=false 成员的 kIdNone 写入会覆盖其他成员写的有效 hint。
  for (const af::AscNodePtr &node : member_analysis.index_region) {
    if (ascgen_utils::indirect_load::GetTemplateRole(node) ==
            ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre &&
        member_axes.index_inner_axis != af::kIdNone) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimdIndexInnerAxis(node, member_axes.index_inner_axis));
    }
  }
  return af::SUCCESS;
}

// members 与 analyses 按序对齐：members[0] 为 primary（analyses[0] 对应），
// 其余为组内成员。primary 完整建轴一次，各成员逐个复用共享轴。
af::Status NormalizeSimdAxesForGroupTemplate(af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                             const std::vector<RewrittenGraphAnalysis> &analyses) {
  GE_ASSERT_TRUE(members.size() == analyses.size() && !members.empty(),
                 "IndirectLoad gather group members[%zu] and analyses[%zu] "
                 "must be aligned and non-empty.",
                 members.size(), analyses.size());
  // 组模板（多 gather 水平融合）保持固定单行 tile：可求解多行 tile 的门控按单
  // gather 候选判定，组内成员共享 primary 的模板轴结构，行数求解语义未定义。
  GE_ASSERT_SUCCESS(NormalizeSimdAxesForTemplate(graph, members.front(), analyses.front(), false));
  for (size_t index = 1UL; index < members.size(); ++index) {
    GE_ASSERT_SUCCESS(
        NormalizeSimdAxesForMember(graph, members.front(), analyses.front(), members[index], analyses[index]));
  }
  return af::SUCCESS;
}

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                        bool solve_tile_size) {
  const auto output_axes = indirect_load->outputs()[0]->attr.axis;
  GE_ASSERT_TRUE(!output_axes.empty(), "IndirectLoad SIMT output axis is empty.");
  GE_ASSERT_TRUE(boundary <= output_axes.size(), "IndirectLoad SIMT boundary %zu is out of range [0, %zu].", boundary,
                 output_axes.size());
  // post-Reduce SIMT（boundary 被 Reduce 布局覆写为 first_reduce，小于输出
  // rank）不预建 固定 tile 轴：tile 行数交由调度期 TileTiling 走通用 TileSplit
  // 按 UB 容量求解，使单个 tile 覆盖多行，消除逐行 SIMT/向量交替（VF_CALL
  // 启动与 PipeBarrier 次数随行数下降）。 无 post-Reduce 的 SIMT boundary
  // 保持输出 rank，维持固定单行 tile 语义不变。 对 AR 后置 Reduce
  // 与形状保持的专用 Norm 节点，solve_tile_size 允许调度期按 UB 容量求解多行
  // tile。 SIMT index evaluator 使用完整 output_index 重建每个 Index Load
  // 的逻辑偏移， 因而 direct index 与 Cast/Add/Where 等复合 index 共用该路径。
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, boundary, ascir::TemplateId::kIndirectLoadSimt,
                                             af::kIdNone, af::kIdNone, false, solve_tile_size));
  return af::SUCCESS;
}

// [跨成员 index 链共享节点豁免集] 收集全组成员的 backward region 并集 + 全部 IL
// 节点。 共享 index Load 的跨成员消费者不视为"外部"——由另一成员的 evaluator
// 逐元素消费 （寄存器级），不要求向量侧物化 tensor。单成员/单 gather 语义下并集
// = 自身 region， 行为与空集等价。
NodeSet BuildGroupIndexExemptSet(const std::vector<af::AscNodePtr> &members, std::vector<NodeSet> *back_regions) {
  NodeSet group_exempt;
  if (back_regions != nullptr) {
    back_regions->resize(members.size());
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    const af::AscNodePtr index_root =
        ascgen_utils::indirect_load::GetInputProducer(members[index], ascgen_utils::indirect_load::kIndexTensorIndex);
    if (index_root == nullptr) {
      GELOGW(
          "[IndirectLoad] BuildGroupIndexExemptSet: index producer of node[%s] "
          "is missing.",
          members[index]->GetNamePtr());
      continue;
    }
    NodeSet back_region;
    if (!CollectSimtBackwardRegion({index_root}, members[index], back_region)) {
      GELOGW(
          "[IndirectLoad] BuildGroupIndexExemptSet: backward region of "
          "node[%s] is unsupported.",
          members[index]->GetNamePtr());
      continue;
    }
    group_exempt.insert(back_region.begin(), back_region.end());
    if (back_regions != nullptr) {
      (*back_regions)[index] = std::move(back_region);
    }
    group_exempt.emplace(members[index].get());
  }
  return group_exempt;
}

// 多 gather 组员（SIMT）：仅收集本路 index 生产链 backward 区域（含 index
// 链外部 store 折叠）。 共享归并/输出链不进入任何组员的 SIMT
// 区域——它们作为普通向量代码在全部组员 invocation
// 之后执行，避免输出链角色冲突与 evaluator 跨依赖。
bool CollectSimtGroupIndexRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load, NodePath &index_region,
                                 const NodeSet &group_exempt, const NodeSet *pre_back_region) {
  index_region.clear();
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  if (index_root == nullptr) {
    return false;
  }
  NodeSet region;
  if (pre_back_region != nullptr) {
    region = *pre_back_region;
  } else if (!CollectSimtBackwardRegion({index_root}, indirect_load, region)) {
    return false;
  }
  // 折叠前的 backward region 快照——FoldSimtIndexChainExternalStores
  // 只增不删（Store 折叠）， 传入快照避免内部第三次遍历。
  const NodeSet index_chain = region;
  if (!FoldSimtIndexChainExternalStores(indirect_load, index_root, region, group_exempt, &index_chain)) {
    return false;
  }
  // 按 GetAllNodes 顺序重建（与 CollectRewrittenRegion 的确定性约定一致）。
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (region.count(node.get()) != 0UL) {
      index_region.emplace_back(node);
    }
  }
  return !index_region.empty();
}

// 多 gather 组员（SIMT）共享轴归一化：primary 完整归一一次，其余成员复用
// primary 的 TemplateAxes——输出轴空间与 boundary 全组相同，轴结构一致；SIMT
// 归一不触碰区域节点轴， 拷贝语义等价，且避免
// BuildAxisViewByBoundary/CreateFixedTileSplit 按成员重复建轴。 boundary
// 语义（与单 gather 同构，按归约形态分档）：
// - 无共享 Reduce 的组（L2 寄存器归并 / 非归约触发的 L3）：boundary=输出
// rank——整空间合并
//   轴，固定 tile inner=1 + codegen 整核调用形态；
// - 有共享 Reduce 的组（归约触发的
// L3）：boundary=first_reduce——outer=行/批（G）、
//   inner=归约段（R），行不被 BlockSplit 劈开。solve_tile 与单 gather
//   同款判定：AR 形态 （归约段后无 Retained 轴，含尾轴整体归约的
//   Sum）与形状保持 Norm 开多行求解（k 行 tile，一次调用覆盖 k 行）；非 AR
//   形态（归约段后有 Retained 后缀）固定单行——多行窗口 会把多组 RA 段并进同一
//   Reduce 窗口，归约语义错误，k=1 是正确性约束。
af::Status NormalizeSimtAxesForGroupTemplate(af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                             const af::AscNodePtr &group_post_reduce, bool &is_candidate_legal,
                                             size_t boundary_override) {
  GE_ASSERT_TRUE(!members.empty(), "IndirectLoad gather group members must be non-empty.");
  size_t boundary = members.front()->outputs()[0]->attr.axis.size();
  // 无共享 Reduce 的组：结构判定已确定该组只能走 UB 归并时，调用方以 boundary_override
  // 指定 template boundary（rank-1：outer=行/批、inner=最内连续轴），使 gather 与输出链
  // 落在同一 tile 循环内；缺省 0 保持原语义（boundary=输出 rank）。
  if (group_post_reduce == nullptr && boundary_override != 0UL && boundary_override < boundary) {
    boundary = boundary_override;
  }
  bool solve_tile_size = false;
  if (group_post_reduce != nullptr) {
    // 形状保持的专用 Norm
    // 节点（Softmax）：归约发生在算子内部、BuildPostReduceLayout
    // 识别不出归约轴（first_reduce == rank），ValidateSimtPostReduceLayout
    // 会判非法。 与单 gather 的 ValidateTemplate Softmax
    // 分流同口径：跳过该断言，boundary 覆写为 softmax 输入
    // rank-1（R=尾轴，SoftmaxAR 多行 tile 的正确语义）。
    if (IsShapePreservingTailAxisNorm(group_post_reduce)) {
      const auto norm_inputs = group_post_reduce->inputs();
      GE_ASSERT_TRUE(!norm_inputs.empty() && norm_inputs[0] != nullptr,
                     "IndirectLoad group Softmax node[%s] has no input.", group_post_reduce->GetNamePtr());
      const size_t softmax_input_rank = norm_inputs[0]->attr.axis.size();
      GE_ASSERT_TRUE(softmax_input_rank > 0UL, "IndirectLoad group Softmax input rank is invalid.");
      boundary = softmax_input_rank - 1UL;
      solve_tile_size = true;
    } else {
      // 布局合法性已在 ApplyGraphPassGroup 调用前校验（同口径拒绝），此处仅取
      // boundary 覆写。 中间仅隔 NormalizeMemberAxes（改 IL 输出轴，理论不动
      // reduce strides），但仍以防御口径
      // 处理"校验后布局变化"——拒绝候选而非硬断言（候选竞标语义：drop 本候选，不
      // abort）。
      bool group_reduce_legal = true;
      GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(group_post_reduce, boundary, group_reduce_legal));
      if (!group_reduce_legal) {
        is_candidate_legal = false;
        GELOGI(
            "[IndirectLoad] Reject gather group SIMT candidate: shared post "
            "Reduce layout became illegal "
            "during normalization.");
        return af::SUCCESS;
      }
      GE_ASSERT_SUCCESS(IsSimtArPostReduce(group_post_reduce, solve_tile_size));
    }
  }
  GE_ASSERT_SUCCESS(NormalizeSimtAxesForTemplate(graph, members.front(), boundary, solve_tile_size));
  ascgen_utils::indirect_load::TemplateAxes shared_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(members.front(), shared_axes));
  for (size_t index = 1UL; index < members.size(); ++index) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(members[index], shared_axes));
  }
  return af::SUCCESS;
}

af::Status NormalizeTemplateAxes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, const RewrittenGraphAnalysis &analysis,
                                 size_t boundary) {
  const bool is_softmax_post =
      analysis.post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce);
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    // 无后置 Reduce 的裸 gather：tile 行数交由调度期 TileTiling 按 UB
    // 容量求解（多行 tile）， 消除逐行 staging/队列周期；带后置 Reduce 的 SIMD
    // 保持固定单行 tile 语义（其区域 视图种子依赖单行 tile
    // 结构）。以下形态同样保持固定单行 tile（多行调用要求 staging 批量产出
    // R×inner，而下列形态的 staging 只能逐行或单份产出，模板会读取仅含单行
    // 的缓冲导致越界/错行）：
    // 1) gather 轴不在第 1 维（axis_index != 1）：批量 DMA 要求 tile 行映射为
    // gather
    //    切片正上方的单一行维；行维跨越多个维度（如 rank4/axis2，行维由 merged
    //    outer 承载）时 Load 视图保留分离行维，staging
    //    按行循环、目的偏移固定（如 index_binary_same_view
    //    用例），多行调用读取单行缓冲；
    // 2) 对齐（padding）视图：staging 按紧凑布局剥离行间
    // padding，模板按逻辑步长寻址； 3) Load 后接 Transpose 的路径：staging
    // 经逐行 TransposeExtend，无法批量化
    //    （其布局被强制标记 kStrided 表达合法置换，NeedsAlignedUbWindow 仅检查
    //    gather 轴内侧维度拦不住行维置换，如 both_transpose 用例）；
    // 4) gather 轴内侧零步长（视图内广播，如 gather 轴广播源）：staging
    // 需逐行展开
    //    广播而非批量 DMA 块（如 broadcast_axis 用例）；
    // 5) 重叠/非法视图（kind==kUnsupported）：非单调布局，不可批量 staging；
    // 6) 输入与索引均为 kDense 的稠密视图：lowering 选
    // kRegisterGather/kGatherApi
    //    稠密 facade（indirect_load_utils.cpp：input/index 任一非 kDense 才选
    //    kStrided），稠密 dispatch 不含行基址项（行由外层 tile
    //    循环承载），多行调用 全部行读 staged 窗口首行（如 rank2_axis1_simd
    //    用例；rank3_axis1_int64_gather 仅因运行期 R=1
    //    偶然通过），必须保持固定单行 tile。kStrided facade 经
    //    input_outer_offset 正确叠加行基址，多行安全。
    // kZeroStrideCompact 及无对齐窗口的 kStrided
    // 均为行主序单调布局，axis_index==1 时可安全批量 staging；行维零步长（如
    // row-invariant 索引 {0,1}，索引路径含 Broadcast 节点但行维零步长）staging
    // 单份、模板按零步长重读，同样安全。
    ascgen_utils::indirect_load::TemplateLogicalView logical_view;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view));
    const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
    GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                   indirect_load->GetNamePtr());
    const auto layout_kind = ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported;
    const auto dense_kind = ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense;
    const bool solve_simd_tile = analysis.post_reduce == nullptr && axis_index == 1UL && !analysis.align_input_path &&
                                 !analysis.align_index_path && !analysis.input_load_transposed &&
                                 !analysis.index_load_transposed && logical_view.input.kind != layout_kind &&
                                 logical_view.index.kind != layout_kind &&
                                 (logical_view.input.kind != dense_kind || logical_view.index.kind != dense_kind) &&
                                 !HasBroadcastInsideWindow(logical_view.input, axis_index) &&
                                 !HasBroadcastInsideWindow(logical_view.index, axis_index);
    GE_ASSERT_SUCCESS(NormalizeSimdAxesForTemplate(graph, indirect_load, analysis, solve_simd_tile));
    if (is_softmax_post) return af::SUCCESS;
    // 多阶段：全部 Reduce 各自种向量化视图；单 Reduce/无 Reduce 保持原行为。
    if (!analysis.composite_reduces.empty()) {
      return SeedCompositeReduceInputViews(indirect_load, analysis.composite_reduces);
    }
  } else {
    // 仅 AR 后置 Reduce 开启可求解 tile：TileInner 代表多个 retained
    // feature 行，reduced suffix 保持完整向量化，避免将 RA 的 retained
    // 后缀误并入同一 Reduce API 窗口。
    bool solve_tile = false;
    GE_ASSERT_SUCCESS(IsSimtArPostReduce(analysis.post_reduce, solve_tile));
    // 形状保持的专用 Norm 节点走不了 stride 推导，其尾轴内部归约、各行独立的
    // 语义与 AR 后置 Reduce 一致，同样开启可求解 tile。
    if (IsShapePreservingTailAxisNorm(analysis.post_reduce)) {
      solve_tile = true;
    }
    // [边界化轴布局] 边界化时输出链整体退出求值器，value 逐 tile 写 UB 局部目标，走
    // post-Reduce 同构的 outer/inner 分离布局：boundary 覆写为 rank-1（outer=行、
    // inner=尾轴向量化段），tile 行数按 UB 容量求解（多行 tile）。否则 boundary 保持
    // 输出 rank 的整空间合并轴，codegen 会误入整核调用形态（loop_size/block_dim_offset）。
    // 判定已限定 post_reduce==nullptr（post-Reduce 优先），boundary 此处必为输出 rank。
    if (ascgen_utils::indirect_load::IsSimtBoundaryFusion(indirect_load) && analysis.post_reduce == nullptr) {
      const size_t output_rank = indirect_load->outputs()[0]->attr.axis.size();
      GE_ASSERT_TRUE(output_rank > 1UL, "IndirectLoad SIMT boundary fusion requires a separable row axis, node[%s].",
                     indirect_load->GetNamePtr());
      boundary = output_rank - 1UL;
      solve_tile = true;
    }
    GE_ASSERT_SUCCESS(NormalizeSimtAxesForTemplate(graph, indirect_load, boundary, solve_tile));
  }
  if (is_softmax_post) return af::SUCCESS;
  return SeedPostReduceInputVectorizedView(indirect_load, analysis.post_reduce);
}
}  // namespace indirect_load_detail
}  // namespace optimize
