/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.hiascend.com/software/licensedistributionexception
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "task_generator/simt_boundary_sync.h"
#include "common/common_utils.h"

#include <algorithm>
#include <set>

#include "utils/axis_utils.h"
#include "ascir_ops.h"

namespace optimize::task_generator {
namespace {

constexpr int32_t kMaxAxisAncestorDepth = 8;

// 判断节点是否为需要 view 对齐的 SIMT 模板角色（跳过主调度 tiling 的角色）
bool IsSimtViewSyncRole(const ascgen_utils::indirect_load::TemplateRole role) {
  return role == ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform ||
         role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch ||
         role == ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary;
}

// 判断 axis_id 是否为 target_id 的祖先轴：沿 from 链向上追溯（含自身），
// 既覆盖 split 链（from.size()==1）也覆盖 merge 关系（target 是 axis 的 from 成员）
bool IsAxisAncestorOf(af::AscGraph &graph, const ascir::AxisId axis_id, const ascir::AxisId target_id) {
  if (axis_id == target_id) {
    return true;
  }
  ascir::AxisId cursor = axis_id;
  for (int32_t depth = 0; depth < kMaxAxisAncestorDepth && cursor != af::kIdNone; ++depth) {
    const auto axis = graph.FindAxis(cursor);
    if (axis == nullptr) {
      break;
    }
    // merge 关系：target 是当前轴的合并源成员（如旧轴 1 被 merge 进 from=[0,1] 的轴 2）
    if (std::find(axis->from.begin(), axis->from.end(), target_id) != axis->from.end()) {
      return true;
    }
    if (axis->from.size() != 1UL) {
      break;
    }
    cursor = axis->from.front();
  }
  return false;
}

// 对单个 output 的 tensor view 执行 split 同步（不改 sched 轴）
void SplitOutputView(const af::AscTensor &output, const af::AxisPtr &outer, const af::AxisPtr &inner) {
  const ascir::AxisId split_original = outer->from[0];
  if (std::find(output.attr.axis.begin(), output.attr.axis.end(), split_original) == output.attr.axis.end()) {
    return;
  }
  const auto view = af::AxisUtils::SplitView({output.attr.axis, output.attr.repeats, output.attr.strides}, inner->size,
                                             outer->id, inner->id, split_original);
  output.attr.axis = view.axis_ids;
  output.attr.repeats = view.repeats;
  output.attr.strides = view.strides;
}

// vectorized_axis 重映射：旧轴被模板 merge 合并、合并轴又被 tiling split 时，
// 支持两级链（旧轴 -> 合并轴 -> inner 轴）映射为 view 中实际存在的新轴
void RemapOutputVectorizedAxes(af::AscGraph &graph, const af::AscNodePtr &node, const af::AscTensor &output,
                               const std::set<ascir::AxisId> &merged_axis_ids,
                               const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  std::vector<ascir::AxisId> remapped;
  remapped.reserve(output.attr.vectorized_axis.size());
  for (const auto vec_axis_id : output.attr.vectorized_axis) {
    if (std::find(output.attr.axis.begin(), output.attr.axis.end(), vec_axis_id) != output.attr.axis.end()) {
      remapped.push_back(vec_axis_id);
      continue;
    }
    // 旧轴不在 view 中：遍历 split 对，若 inner 轴在 view 中且 inner 的祖先链命中
    // 旧轴或旧轴所属的合并轴，则映射到 inner 轴
    bool remapped_ok = false;
    for (const auto &tiled_axes : tiled_axes_list) {
      if (tiled_axes.first == nullptr || tiled_axes.second == nullptr || tiled_axes.first->from.size() != 1UL) {
        continue;
      }
      const ascir::AxisId inner_id = tiled_axes.second->id;
      if (std::find(output.attr.axis.begin(), output.attr.axis.end(), inner_id) == output.attr.axis.end()) {
        continue;
      }
      const bool direct_ancestor = IsAxisAncestorOf(graph, tiled_axes.first->from[0], vec_axis_id);
      bool via_merged = false;
      if (!direct_ancestor) {
        for (const auto merged_axis_id : merged_axis_ids) {
          if (IsAxisAncestorOf(graph, tiled_axes.first->from[0], merged_axis_id) &&
              IsAxisAncestorOf(graph, merged_axis_id, vec_axis_id)) {
            via_merged = true;
            break;
          }
        }
      }
      if (direct_ancestor || via_merged) {
        GELOGD("[IndirectLoad] SIMT role node[%s] remap vectorized axis[%ld] to inner axis[%ld].", node->GetNamePtr(),
               vec_axis_id, inner_id);
        remapped.push_back(inner_id);
        remapped_ok = true;
        break;
      }
    }
    if (!remapped_ok) {
      // 无法映射时保留原值，等待后续校验暴露
      GELOGD("[IndirectLoad] SIMT role node[%s] keep unmapped vectorized axis[%ld].", node->GetNamePtr(), vec_axis_id);
      remapped.push_back(vec_axis_id);
    }
  }
  output.attr.vectorized_axis = remapped;
}

void PrependPostReduceTileAxis(const af::AscNodePtr &indirect_load, const ascir::AxisId tile_inner_axis) {
  const auto reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load);
  if (tile_inner_axis == af::kIdNone || reduce == nullptr) {
    return;
  }
  std::vector<af::AscNodePtr> pending{indirect_load};
  std::set<af::Node *> visited;
  for (size_t index = 0UL; index < pending.size(); ++index) {
    const auto &node = pending[index];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      if (output == nullptr ||
          std::find(output->attr.axis.begin(), output->attr.axis.end(), tile_inner_axis) == output->attr.axis.end() ||
          std::find(output->attr.vectorized_axis.begin(), output->attr.vectorized_axis.end(), tile_inner_axis) !=
              output->attr.vectorized_axis.end()) {
        continue;
      }
      output->attr.vectorized_axis.insert(output->attr.vectorized_axis.begin(), tile_inner_axis);
      output->attr.vectorized_strides.clear();
      output->attr.vectorized_strides.reserve(output->attr.vectorized_axis.size());
      for (const auto vectorized_axis : output->attr.vectorized_axis) {
        const auto axis_it = std::find(output->attr.axis.begin(), output->attr.axis.end(), vectorized_axis);
        if (axis_it == output->attr.axis.end()) {
          return;
        }
        output->attr.vectorized_strides.emplace_back(
            output->attr.strides[static_cast<size_t>(std::distance(output->attr.axis.begin(), axis_it))]);
      }
    }
    if (node.get() == reduce.get()) {
      continue;
    }
    for (const auto &next : node->GetOutDataNodes()) {
      const auto next_node = std::dynamic_pointer_cast<af::AscNode>(next);
      if (next_node == nullptr) {
        continue;
      }
      for (size_t input_index = 0UL; input_index < next_node->inputs().size(); ++input_index) {
        const auto input_anchor = next_node->GetInDataAnchor(static_cast<int32_t>(input_index));
        const auto output_anchor = input_anchor == nullptr ? nullptr : input_anchor->GetPeerOutAnchor();
        if (output_anchor == nullptr || output_anchor->GetOwnerNode().get() != node.get()) {
          continue;
        }
        const size_t output_index = static_cast<size_t>(output_anchor->GetIdx());
        if (output_index >= node->outputs().size()) {
          return;
        }
        const auto &source = node->outputs()[output_index]->attr;
        auto &target = next_node->inputs()[input_index]->attr;
        target.axis = source.axis;
        target.repeats = source.repeats;
        target.strides = source.strides;
        target.vectorized_axis = source.vectorized_axis;
        target.vectorized_strides = source.vectorized_strides;
      }
      pending.emplace_back(next_node);
    }
  }
}

}  // namespace

// 取模板轴表并收集合并轴集合（outer/inner），用于 vectorized_axis 两级重映射；
// 轴表同时输出给调用方，供 post-Reduce tile-inner 传播判定复用（tile_inner_axis/
// outer_axis），避免重复查询。
af::Status GetSyncTemplateAxes(const af::AscNodePtr &indirect_load, std::set<ascir::AxisId> &merged_axis_ids,
                               ascgen_utils::indirect_load::TemplateAxes &template_axes) {
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, template_axes));
  for (const auto axis_id : {template_axes.outer_axis, template_axes.inner_axis}) {
    if (axis_id != af::kIdNone) {
      merged_axis_ids.insert(axis_id);
    }
  }
  return af::SUCCESS;
}

// [Arange 视图同步豁免] Arange 的 SIMT 标量发射（EmitSimtArangeExpr）以
// "原始视图首个非零 stride 轴在模板逻辑视图内"为合约，坐标分解的 stride/size
// 均取自 TemplateLogicalView；物理 tiling split 对 Arange 无消费方（零输入，
// 不参与 GM 寻址/向量发射/UB 驻留）。若把行轴 split 同步进 Arange 视图，其
// 唯一 varying 轴会分裂为 outer/inner 多个非零 stride 轴，破坏
// ArangeAscIrCodegenImplV2::IsSimtScalarSupported 的"单一 varying 轴"前提，
// 调度通过但发射期断言失败（codegen 硬挂）。Arange 保持原始视图，直接跳过 split。
// 其余节点：按调度实际执行的 (outer, inner) 逐层 split，使 view 状态与被
// 完整调度的普通节点等价。
void SyncSimtViewSplitOutputs(const af::AscNodePtr &node,
                              const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
    return;
  }
  for (auto &output : node->outputs()) {
    if (output == nullptr) {
      continue;
    }
    for (const auto &tiled_axes : tiled_axes_list) {
      if (tiled_axes.first == nullptr || tiled_axes.second == nullptr || tiled_axes.first->from.size() != 1UL) {
        continue;
      }
      GELOGD("[IndirectLoad] SIMT role node[%s] sync tensor view split axis[%ld] to [outer:%ld, inner:%ld].",
             node->GetNamePtr(), tiled_axes.first->from[0], tiled_axes.first->id, tiled_axes.second->id);
      SplitOutputView(*output, tiled_axes.first, tiled_axes.second);
    }
  }
}

// vectorized_axis 重映射：旧轴 ->（合并轴 ->）inner 轴。
void RemapSimtViewVectorizedAxes(af::AscGraph &graph, const af::AscNodePtr &node,
                                 const std::set<ascir::AxisId> &merged_axis_ids,
                                 const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  for (auto &output : node->outputs()) {
    if (output == nullptr) {
      continue;
    }
    RemapOutputVectorizedAxes(graph, node, *output, merged_axis_ids, tiled_axes_list);
  }
}

// 只有 SIMT 后接 AR Reduce 的可求解 tile 才需要将动态生成的 tile-inner 作为多行
// vectorized 轴传播（前置校验在调用方）；此处找到 outer 对应的 tile 分裂对，把
// inner 轴前置补入 post-Reduce 链的向量化视图。
void PrependSolvedTileInnerToPostReduce(const af::AscNodePtr &indirect_load,
                                        const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list,
                                        const ascgen_utils::indirect_load::TemplateAxes &template_axes) {
  for (const auto &tiled_axes : tiled_axes_list) {
    if (tiled_axes.first == nullptr || tiled_axes.second == nullptr || tiled_axes.first->from.size() != 1UL ||
        tiled_axes.first->from[0] != template_axes.outer_axis) {
      continue;
    }
    PrependPostReduceTileAxis(indirect_load, tiled_axes.second->id);
    break;
  }
}

af::Status SyncSimtBoundaryViews(af::AscGraph &graph,
                                 const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list) {
  if (tiled_axes_list.empty()) {
    return af::SUCCESS;
  }
  const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }
  // 模板合并轴集合（outer/inner），用于 vectorized_axis 两级重映射
  std::set<ascir::AxisId> merged_axis_ids;
  ascgen_utils::indirect_load::TemplateAxes template_axes;
  GE_ASSERT_SUCCESS(GetSyncTemplateAxes(indirect_load, merged_axis_ids, template_axes));

  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || !IsSimtViewSyncRole(ascgen_utils::indirect_load::GetTemplateRole(node))) {
      continue;
    }
    SyncSimtViewSplitOutputs(node, tiled_axes_list);
    RemapSimtViewVectorizedAxes(graph, node, merged_axis_ids, tiled_axes_list);
  }
  // SIMD 和固定 tile 的 SIMT（tile_inner_axis 已预建）保持原有视图，避免把固定
  // tile-inner 错当作 featuret 插入 Reduce 输入；仅 SIMT 后接 AR Reduce 的可求解
  // tile 需要传播动态生成的 tile-inner。
  if (ascir::GetTemplateIdOrDefault(*indirect_load) != ascir::TemplateId::kIndirectLoadSimt ||
      template_axes.tile_inner_axis != af::kIdNone) {
    return af::SUCCESS;
  }
  PrependSolvedTileInnerToPostReduce(indirect_load, tiled_axes_list, template_axes);
  return af::SUCCESS;
}

// post-Reduce SIMT 输出行数维收口（统一语义）：直接调度路径通过 prepend 把求解的
// TileInner 加入模板向量化轴集合，但多阶段展开（Reduce FirstStage）等路径重建的
// Phase 图仍按预建固定 tile 语义构造，其节点的向量化视图只剩尾轴；而 buffer 分配
// 与数据搬运按求解的多行 tile 生成。此处统一把视图前置的 TileInner 轴补入
// post-Reduce 链上各节点（含 IndirectLoad 输出及其消费者）的向量化视图前部，
// 使 Softmax/Reduce 等消费者按通用 A/R 逻辑自然得到正确行数；vectorized_strides
// 由随后的对齐阶段按补全后的轴重建，无需手工同步。
}  // namespace optimize::task_generator
