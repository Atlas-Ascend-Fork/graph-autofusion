/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_SCHEDULE_H
#define AUTOFUSE_SCHEDULE_H

#include <utility>

#include "ascir.h"
#include "indirect_load_utils.h"
#include "optimize.h"
#include "tiling_group.h"
#include "graph_properties_cache.h"

namespace optimize::autoschedule {
constexpr int64_t kDefaultAxisId = -1;

struct TilingCase {
  ascir::AxisId ub_tiling_id_x = kDefaultAxisId;
  ascir::AxisId ub_tiling_id_y = kDefaultAxisId;
  ascir::AxisId ub_tiling_id_r = kDefaultAxisId;
  ascir::AxisId block_tiling_id = kDefaultAxisId;
  ascir::AxisId reduce_outer_id = kDefaultAxisId;
  ascir::AxisId reduce_block_tiling_id = kDefaultAxisId;
  ascir::AxisId merge_reduce_id = kDefaultAxisId;
  ascir::AxisId merge_no_reduce_id = kDefaultAxisId;
  bool reduce_is_block = false;
  std::pair<af::AxisPtr, af::AxisPtr> ub_tiling_x;
  std::pair<af::AxisPtr, af::AxisPtr> ub_tiling_y;
  std::pair<af::AxisPtr, af::AxisPtr> ub_tiling_r;
  std::pair<af::AxisPtr, af::AxisPtr> block_tiling;
  std::pair<af::AxisPtr, af::AxisPtr> reduce_block_tiling;
  af::Expression rm_org_size;  // R轴切多核时，R轴用于分核的轴大小
  af::Expression a_org_size;   // R轴切多核时，A轴大小
};

struct IndirectLoadInfo {
  bool active = false;
  ascgen_utils::indirect_load::TemplateAxes axes;
};

class Scheduler {
 public:
  Scheduler() = delete;
  explicit Scheduler(ascir::ImplGraph &graph, AxisGroup axes_group, TilingCase &tiling_case,
                     bool is_last_axis_reduce = false,
                     optimize::ReduceTemplateType reduce_template = optimize::ReduceTemplateType::kDefault,
                     ascir::CubeTemplateType cube_template = ascir::CubeTemplateType::kDefault)
      : graph_(graph),
        axes_group_(std::move(axes_group)),
        tiling_case_(tiling_case),
        is_last_axis_reduce_(is_last_axis_reduce),
        reduce_template_(reduce_template),
        cube_template_(cube_template),
        graph_cache_(graph) {};

  Status DoScheduler();
  Status ReduceBlockTiling(std::vector<ascir::AxisId> &tile_out_axes,
                           const std::vector<ascir::AxisId> &reduce_outer_axes,
                           const std::vector<ascir::AxisId> &non_reduce_outer_axes);
  Status ApplyBlockSplit(const std::vector<ascir::AxisId> &new_sched_axes);
  void FindVectorizedAxes(std::vector<ascir::AxisId> &vectorized_axes, std::vector<size_t> &vectorized_axes_order);
  static Status RemoveRedundantBroadcastNode(const ascir::ImplGraph &impl_graph);

 private:
  // ub 切分
  Status TileSplit();
  static std::vector<ascir::AxisId> GetSortedNodeVectorizedAxes(Scheduler &scheduler);
  // block 切分
  Status BlockSplit(std::vector<ascir::AxisId> &tile_out_axes);
  void FuseTileOutAxes(const std::vector<ascir::AxisId> &non_reduce_outer_axes,
                       std::vector<ascir::AxisId> &reduce_outer_axes);
  void HandleBlockSplitting(std::vector<ascir::AxisId> &tile_out_axes,
                            const std::vector<ascir::AxisId> &non_reduce_outer_axes,
                            const std::vector<ascir::AxisId> &reduce_outer_axes);
  void RemoveDuplicatedAxisFromGroup();
  Status InitIndirectLoadScheduleCase();
  Status ApplyIndirectLoadNodeAxes(const af::AscNodePtr &node, bool &skip_main_tiling) const;
  // 可求解 tile（通用 TileSplit 产物）返回 TileInner 轴 id；固定 tile 返回 kIdNone。
  ascir::AxisId GetSolvedTileInnerAxis() const;
  // ApplyTiling 后将可求解 TileInner 并入输入侧 pre 节点的向量化窗口（多行 staging）。
  Status FoldSolvedTileInnerIntoPreWindow(const af::AscNodePtr &node) const;
  Status SynchronizeTransposeInputSchedAxis();
  Status ModifyStoreAfterReduce(ascir::NodeView &node, ascir::AxisId reduce_block_id);
  Status ApplyBlockSplitToNode(ascir::NodeView &node, bool is_store_after_reduce);
  void TileTiling(ascir::AxisId tile_id, std::pair<af::AxisPtr, af::AxisPtr> &tiled_axes) const {
    if (tile_id != kDefaultAxisId && tiled_axes.first == nullptr) {
      tiled_axes = graph_.TileSplit(tile_id);
    }
  }

  void ApplyTiling(ascir::NodeView &node, ascir::AxisId tile_id,
                   const std::pair<af::AxisPtr, af::AxisPtr> &tiled_axes) const {
    if (tile_id != kDefaultAxisId) {
      graph_.ApplySplit(node, tiled_axes.first->id, tiled_axes.second->id);
    }
  }

  bool HasXGroup() const {
    return tiling_case_.ub_tiling_id_x != kDefaultAxisId;
  }

  bool HasRGroup() const {
    return tiling_case_.ub_tiling_id_r != kDefaultAxisId;
  }

  void AdjustVectorizedAxesOrderOffsets(std::vector<size_t> &vectorized_axes_order, size_t split_point, size_t end,
                                        size_t offset) const;
  ascir::ImplGraph &graph_;
  AxisGroup axes_group_;
  TilingCase &tiling_case_;
  bool is_last_axis_reduce_;
  optimize::ReduceTemplateType reduce_template_;
  ascir::CubeTemplateType cube_template_;
  IndirectLoadInfo indirect_load_info_;
  GraphPropertiesCache graph_cache_;  // 图属性缓存，避免重复遍历
};
}  // namespace optimize::autoschedule

#endif  // AUTOFUSE_SCHEDULE_H
