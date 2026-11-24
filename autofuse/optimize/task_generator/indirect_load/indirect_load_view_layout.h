/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GRAPH_UTILS_EXT_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GRAPH_UTILS_EXT_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {

// 窗口内是否存在 Broadcast（输入/索引路径布局分析）。
bool HasBroadcastInsideWindow(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_index);

// 广播视图/路径布局/轴视图等图工具扩展（第二批）（自 indirect_load_schedule_case_generator.cpp 纯搬迁，语义不变）。

bool IsInputDataSource(const af::AscNodePtr &node);

bool IsSimtTemplateRole(const ascgen_utils::indirect_load::TemplateRole role);

af::Status ClearSimtTemplateRoles(af::AscGraph &graph);

af::Status ClearNormalScheduleSimtRoles(af::AscGraph &graph, const NodeSet &normal_schedule_nodes,
                                        const NodeSet &exempt_nodes);

// A retained Transpose separates two physical coordinate systems. Do not overwrite its source GM view.
bool IsInputRegionBoundary(const af::AscNodePtr &node);

// 检查已选中的 Broadcast 路径是否可以安全折叠；不支持时只淘汰当前 candidate。
bool IsSupportedBroadcastPath(const NodePath &path, size_t broadcast_index, ascir::TemplateId template_id);

af::Status GetBroadcastPhysicalAttr(const af::AscNodePtr &broadcast, ascir::TemplateId template_id,
                                    af::AscTensorAttr &physical_attr);

af::Status InlineBroadcastPath(NodePath &path, int64_t path_broadcast_index, ascir::TemplateId template_id);

af::Status FoldBroadcastPath(InputViewPlan &plan, ascir::TemplateId template_id, bool &is_candidate_legal);

af::Status BuildBroadcastLogicalView(const af::AscTensorAttr &logical_attr, const af::AscTensorAttr &physical_attr,
                                     ascir::TemplateId template_id,
                                     ascgen_utils::indirect_load::LogicalTensorView &view);

af::Status ApplyPhysicalView(const NodePath &path, const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout);

bool NeedsAlignedUbWindow(const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout, size_t axis_index);

// gather 轴内侧（含轴维）存在尺寸大于 1 的零步长维：该维为视图内广播（如 gather 轴
// 广播源），staging 需逐行展开广播而非批量 DMA 块，多行窗口无法批量搬运；行维
// （axis_index 之前）的零步长不受限——staging 单份、模板按零步长重读（如
// row-invariant 索引;

af::Status AnnotateStridedUbPath(const NodePath &path);

af::Status ApplyIndirectLoadPathLayout(const NodePath &path,
                                       const ascgen_utils::indirect_load::IndirectLoadTensorLayout &layout,
                                       bool needs_alignment);

af::Status MergeAxesForTemplate(af::AscGraph &graph, const std::vector<af::AxisId> &axes, const std::string &name,
                                af::AxisId &merged_axis);

af::Status CreateFixedTileSplit(af::AscGraph &graph, af::AxisId axis_id, af::AxisId &outer_id, af::AxisId &inner_id,
                                int64_t group_size = 1L);

af::Status BuildAxisViewByBoundary(af::AscGraph &graph, const std::vector<af::AxisId> &axes, size_t boundary,
                                   af::AxisId &outer_axis, af::AxisId &inner_axis);

bool CanEmitSimtScalar(const af::AscNodePtr &node);

// [边界化判定] 求值器消费集中首个不可标量发射的节点（nullptr = 全部可发射）。
af::AscNodePtr FindUnemittableSimtEvaluatorNode(const af::AscGraph &graph, const NodeSet &evaluator_nodes);

// [边界化列分块] 把 IL 输出最内层连续轴拆成固定列块，并同步输出链/index 链/输入链
// 闭包的视图与 sched.axis；同时标记 column_tile_inner 供 lowering 使用列块内跨度。
af::Status SplitBoundaryFusionColumnTile(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                         const af::AscNodePtr &output_store,
                                         const std::vector<af::AscNodePtr> &index_region);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
