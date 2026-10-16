/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "schedule.h"
#include <numeric>
#include <queue>
#include <set>
#include "alignment_handler.h"
#include "indirect_load_utils.h"
#include "platform/common/base_alignment_strategy.h"
#include "schedule_utils.h"
#include "node_cache_marker.h"
#include "task_generator/simt_boundary_sync.h"

namespace {
bool CompareByOrderInTensorAxis(const int64_t &lhs, const int64_t &rhs, const std::vector<int64_t> &tensor_axes) {
  auto iter_lhs = std::find(tensor_axes.begin(), tensor_axes.end(), lhs);
  auto iter_rhs = std::find(tensor_axes.begin(), tensor_axes.end(), rhs);
  return iter_lhs < iter_rhs;
}

bool IsAxisConsumedByReduce(const af::AscNodePtr &node, const int64_t axis_id) {
  const auto &brc_out_attr = node->outputs[0].attr;
  if (brc_out_attr.axis.size() != brc_out_attr.repeats.size()) {
    return false;
  }
  auto brc_axis_iter = std::find(brc_out_attr.axis.begin(), brc_out_attr.axis.end(), axis_id);
  if (brc_axis_iter == brc_out_attr.axis.end()) {
    return false;
  }
  const auto brc_idx = static_cast<size_t>(std::distance(brc_out_attr.axis.begin(), brc_axis_iter));

  for (const auto &out_node : node->GetOutDataNodes()) {
    auto asc_out_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if ((asc_out_node == nullptr) || !optimize::ScheduleUtils::IsReduce(asc_out_node)) {
      continue;
    }
    const auto &reduce_out_attr = asc_out_node->outputs[0].attr;
    if (reduce_out_attr.axis.size() != reduce_out_attr.repeats.size()) {
      continue;
    }

    auto reduce_axis_iter = std::find(reduce_out_attr.axis.begin(), reduce_out_attr.axis.end(), axis_id);
    if (reduce_axis_iter == reduce_out_attr.axis.end()) {
      continue;
    }
    const auto reduce_idx = static_cast<size_t>(std::distance(reduce_out_attr.axis.begin(), reduce_axis_iter));
    const bool is_axis_consumed =
        af::SymbolicUtils::StaticCheckEq(brc_out_attr.repeats[brc_idx], reduce_out_attr.repeats[reduce_idx]) !=
        af::TriBool::kTrue;
    if (is_axis_consumed) {
      return true;
    }
  }
  return false;
}

bool HasBroadcastExpansionOnNonVectorizedAxis(const af::AscNodePtr &brc_node, const af::AscTensorAttr &input_attr,
                                              const af::AscTensorAttr &output_attr) {
  if ((input_attr.axis.size() != input_attr.repeats.size()) ||
      (output_attr.axis.size() != output_attr.repeats.size())) {
    return false;
  }

  for (size_t idx = 0UL; idx < output_attr.axis.size(); ++idx) {
    const auto axis_id = output_attr.axis[idx];
    if (std::find(output_attr.vectorized_axis.begin(), output_attr.vectorized_axis.end(), axis_id) !=
        output_attr.vectorized_axis.end()) {
      continue;
    }

    auto input_axis_iter = std::find(input_attr.axis.begin(), input_attr.axis.end(), axis_id);
    if (input_axis_iter == input_attr.axis.end()) {
      continue;
    }
    const auto input_idx = static_cast<size_t>(std::distance(input_attr.axis.begin(), input_axis_iter));
    const bool is_broadcast_expanded =
        af::SymbolicUtils::StaticCheckEq(output_attr.repeats[idx], input_attr.repeats[input_idx]) != af::TriBool::kTrue;
    if (is_broadcast_expanded && IsAxisConsumedByReduce(brc_node, axis_id)) {
      return true;
    }
  }
  return false;
}

bool IsRedundantBroadcast(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &brc_node,
                          const af::AscNodePtr &pre_node, const uint32_t pre_node_out_index) {
  if (optimize::ScheduleUtils::IsIOBuffer(pre_node)) {
    return false;
  }
  std::vector<af::Expression> in_vec_repeats;
  const auto &in_attr = pre_node->outputs[pre_node_out_index].attr;
  GE_WARN_ASSERT(optimize::ScheduleUtils::GetVectorRepeats(in_attr.repeats, in_attr.axis, in_attr.vectorized_axis,
                                                           in_vec_repeats) == af::SUCCESS,
                 "%s[%s] GetVectorRepeats failed", pre_node->GetTypePtr(), pre_node->GetNamePtr());
  std::vector<af::Expression> out_vec_repeats;
  GE_WARN_ASSERT(optimize::ScheduleUtils::GetNodeOutVectorRepeats(brc_node, out_vec_repeats) == af::SUCCESS);

  if (out_vec_repeats.size() != in_vec_repeats.size()) {
    GELOGD("Graph [%s] Broadcast [%s] output vector strides.size(%zu) != in vector strides.size(%zu), skip it",
           impl_graph.GetName().c_str(), brc_node->GetNamePtr(), out_vec_repeats.size(), in_vec_repeats.size());
    return false;
  }
  for (size_t idx = 0; idx < out_vec_repeats.size(); idx++) {
    if (af::SymbolicUtils::StaticCheckEq(out_vec_repeats[idx], in_vec_repeats[idx]) != af::TriBool::kTrue) {
      return false;
    }
  }

  const auto &out_attr = brc_node->outputs[0].attr;
  if (!optimize::ScheduleUtils::IsBroadcast(pre_node) &&
      HasBroadcastExpansionOnNonVectorizedAxis(brc_node, in_attr, out_attr)) {
    GELOGD("Graph [%s] Broadcast [%s] has valid non-vectorized shape expansion, keep it.", impl_graph.GetName().c_str(),
           brc_node->GetNamePtr());
    return false;
  }

  return true;
}

/**
 *    pre_pre_node
 *        |
 *    pre_node(broadcast)
 *       |
 *    brc_node(broadcast)
 */
bool IsContinuesBroadcast(const ascir::ImplGraph &impl_graph, const af::AscNodePtr &brc_node,
                          const af::AscNodePtr &pre_node) {
  if (!optimize::ScheduleUtils::IsBroadcast(pre_node) || pre_node->GetOutNodesSize() != 1UL ||
      pre_node->GetInNodesSize() != 1UL) {
    return false;
  }
  const auto &in_nodes = pre_node->GetInDataNodes();
  if (optimize::ScheduleUtils::IsScalarLikeNode(in_nodes.at(0UL))) {
    GELOGD("Input of Broadcast[%s] is Scalar[%s], is supported.", brc_node->GetNamePtr(),
           in_nodes.at(0UL)->GetNamePtr());
    return true;
  }
  auto &in_vec_axis = pre_node->inputs[0].attr.vectorized_axis;
  auto &out_vec_axis = brc_node->outputs[0].attr.vectorized_axis;
  if (in_vec_axis.size() != out_vec_axis.size() || in_vec_axis.size() <= 1UL) {
    return false;
  }

  std::vector<af::Expression> in_vec_repeats;
  if (optimize::ScheduleUtils::GetNodeInputVectorRepeats(pre_node, in_vec_repeats) != af::SUCCESS) {
    GELOGD("Graph [%s], get [%s] input vector repeats failed.", impl_graph.GetName().c_str(), pre_node->GetNamePtr());
    return false;
  }
  std::vector<af::Expression> out_vec_repeats;
  if (optimize::ScheduleUtils::GetNodeOutVectorRepeats(brc_node, out_vec_repeats) != af::SUCCESS) {
    GELOGD("Graph [%s], get [%s] output vector repeats failed.", impl_graph.GetName().c_str(), brc_node->GetNamePtr());
    return false;
  }

  if (optimize::ScheduleUtils::IsContinuesBroadcast(in_vec_repeats, out_vec_repeats)) {
    GELOGD("Graph [%s], [%s] and [%s], find continuous broadcast", impl_graph.GetName().c_str(), brc_node->GetNamePtr(),
           pre_node->GetNamePtr());
    return true;
  }

  if (optimize::ScheduleUtils::IsIntervalBroadcast(in_vec_repeats, out_vec_repeats)) {
    GELOGD("Graph [%s], [%s] and [%s], find interval broadcast(ABAB/BABA)", impl_graph.GetName().c_str(),
           brc_node->GetNamePtr(), pre_node->GetNamePtr());
    return true;
  }
  return false;
}

void AppendAxisOrder(const std::vector<size_t> &axes_order, size_t group_axis_size, size_t last_ub_size,
                     size_t current_ub_size, std::vector<size_t> &vectorized_axes_order) {
  const auto axis_count = static_cast<int64_t>(current_ub_size - last_ub_size);
  const auto inner_axis_begin_idx = static_cast<int64_t>(group_axis_size) - axis_count;
  const auto inner_axis_end_idx = inner_axis_begin_idx + axis_count;
  vectorized_axes_order.insert(vectorized_axes_order.end(), axes_order.begin() + inner_axis_begin_idx,
                               axes_order.begin() + inner_axis_end_idx);
}

void GetOuterAxes(const std::vector<ascir::AxisId> &axes_group, const ascir::AxisId &ub_tiling_id,
                  const ascir::Axis &ub_tiling_outer_axis, const std::vector<size_t> &axes_order,
                  std::vector<ascir::AxisId> &outer_axes, std::vector<size_t> &outer_axes_index,
                  size_t axes_order_idx) {
  for (const ascir::AxisId axis_id : axes_group) {
    if (axis_id != ub_tiling_id) {
      outer_axes.push_back(axis_id);
      outer_axes_index.push_back(axes_order[axes_order_idx++]);
    } else if (axis_id == ub_tiling_id) {
      outer_axes.push_back(ub_tiling_outer_axis.id);
      outer_axes_index.push_back(axes_order[axes_order_idx++]);
      break;
    }
  }
}

void FindInnerAxes(vector<ascir::AxisId> &vectorize_axis, const std::vector<ascir::AxisId> &axis_group,
                   ascir::AxisId ub_tiling_id, const std::pair<af::AxisPtr, af::AxisPtr> &ub_tiling) {
  bool find_tile_in = false;
  for (const auto &axis : axis_group) {
    if (axis == ub_tiling_id) {
      find_tile_in = true;
      vectorize_axis.push_back(ub_tiling.second->id);
      continue;
    }

    if (find_tile_in) {
      vectorize_axis.push_back(axis);
    }
  }
}
}  // namespace
namespace optimize::autoschedule {
Status Scheduler::ReduceBlockTiling(std::vector<ascir::AxisId> &tile_out_axes,
                                    const std::vector<ascir::AxisId> &reduce_outer_axes,
                                    const std::vector<ascir::AxisId> &non_reduce_outer_axes) {
  // 计算所有A轴大小
  tiling_case_.a_org_size = af::sym::kSymbolOne;
  for (auto y : axes_group_.y_group) {
    auto axis = graph_.FindAxis(y);
    GE_ASSERT_NOTNULL(axis, "Cannot find axis with id:[%ld].", y);
    tiling_case_.a_org_size = tiling_case_.a_org_size * axis->size;
  }

  if (reduce_outer_axes.size() > 1UL) {
    tiling_case_.reduce_block_tiling_id = graph_.MergeAxis(reduce_outer_axes)->id;
    tiling_case_.merge_reduce_id = tiling_case_.reduce_block_tiling_id;
  } else {
    GE_ASSERT_TRUE((reduce_outer_axes.size() == 1UL), "No reduce outer axis.");
    tiling_case_.reduce_block_tiling_id = reduce_outer_axes[0UL];
  }

  ascir::AxisId non_reduce_axis;
  if (non_reduce_outer_axes.size() > 1UL) {
    non_reduce_axis = graph_.MergeAxis(non_reduce_outer_axes)->id;
    tiling_case_.merge_no_reduce_id = non_reduce_axis;
  } else {
    GE_ASSERT_TRUE((non_reduce_outer_axes.size() == 1UL), "No non_reduce outer axis.");
    non_reduce_axis = non_reduce_outer_axes[0UL];
  }

  TileTiling(tiling_case_.reduce_block_tiling_id, tiling_case_.reduce_block_tiling);
  tiling_case_.rm_org_size = tiling_case_.reduce_block_tiling.first->size;
  auto block_axis = graph_.MergeAxis({non_reduce_axis, tiling_case_.reduce_block_tiling.first->id});
  tiling_case_.block_tiling_id = block_axis->id;
  tile_out_axes.push_back(block_axis->id);
  tile_out_axes.push_back(tiling_case_.reduce_block_tiling.second->id);
  return af::SUCCESS;
}

void Scheduler::FuseTileOutAxes(const std::vector<ascir::AxisId> &non_reduce_outer_axes,
                                std::vector<ascir::AxisId> &reduce_outer_axes) {
  // fuse r group外轴
  if (reduce_outer_axes.size() > 1UL) {
    tiling_case_.reduce_outer_id = graph_.MergeAxis(reduce_outer_axes)->id;
    reduce_outer_axes = {tiling_case_.reduce_outer_id};
  } else if (reduce_outer_axes.size() == 1UL) {
    tiling_case_.reduce_outer_id = reduce_outer_axes.front();
  }

  // fuse非r_group外轴
  if (non_reduce_outer_axes.size() <= 1UL) {
    if (!non_reduce_outer_axes.empty()) {
      tiling_case_.block_tiling_id = non_reduce_outer_axes[0UL];
    }
    return;
  }

  int64_t attr_axis = -1L;
  int64_t params_size = -1L;
  bool has_gather = ScheduleUtils::GetGatherParams(graph_, attr_axis, params_size);
  if (has_gather) {
    // Gather方案：尾轴(非单轴)和非尾轴场景使用外轴切B、内轴切T的方案
    if (!(attr_axis == params_size - 1 && attr_axis == 0)) {
      tiling_case_.block_tiling_id = non_reduce_outer_axes[0];
    }
  } else {
    auto new_axis = graph_.MergeAxis(non_reduce_outer_axes);
    tiling_case_.block_tiling_id = new_axis->id;
  }
}

void Scheduler::HandleBlockSplitting(std::vector<ascir::AxisId> &tile_out_axes,
                                     const std::vector<ascir::AxisId> &non_reduce_outer_axes,
                                     const std::vector<ascir::AxisId> &reduce_outer_axes) {
  if (tiling_case_.block_tiling_id == kDefaultAxisId) {
    return;
  }

  tiling_case_.block_tiling = graph_.BlockSplit(tiling_case_.block_tiling_id);
  tile_out_axes.push_back(tiling_case_.block_tiling.first->id);
  tile_out_axes.push_back(tiling_case_.block_tiling.second->id);

  bool has_gather = graph_cache_.HasComputeType(af::ComputeType::kComputeGather);
  if (has_gather && non_reduce_outer_axes.size() > 1UL) {
    tile_out_axes.insert(tile_out_axes.end(), non_reduce_outer_axes.begin() + 1, non_reduce_outer_axes.end());
  }

  if (HasRGroup()) {
    tile_out_axes.insert(tile_out_axes.end(), reduce_outer_axes.begin(), reduce_outer_axes.end());
  }
}

Status Scheduler::BlockSplit(std::vector<ascir::AxisId> &tile_out_axes) {
  // reorder axes to original order, x group和y group都有值的时候才需要reorder
  std::vector<ascir::AxisId> non_reduce_outer_axes;
  std::vector<ascir::AxisId> reduce_outer_axes;
  std::vector<size_t> non_reduce_outer_axes_index;
  std::vector<size_t> reduce_outer_axes_index;
  size_t axes_order_idx = 0UL;
  if (HasXGroup()) {
    GetOuterAxes(axes_group_.x_group, tiling_case_.ub_tiling_id_x, *(tiling_case_.ub_tiling_x.first),
                 axes_group_.axes_order, non_reduce_outer_axes, non_reduce_outer_axes_index, axes_order_idx);
    axes_order_idx += axes_group_.x_group.size();
  }
  GetOuterAxes(axes_group_.y_group, tiling_case_.ub_tiling_id_y, *(tiling_case_.ub_tiling_y.first),
               axes_group_.axes_order, non_reduce_outer_axes, non_reduce_outer_axes_index, axes_order_idx);
  axes_order_idx += axes_group_.y_group.size();

  if (HasRGroup()) {
    GetOuterAxes(axes_group_.r_group, tiling_case_.ub_tiling_id_r, *(tiling_case_.ub_tiling_r.first),
                 axes_group_.axes_order, reduce_outer_axes, reduce_outer_axes_index, axes_order_idx);
  }

  if (tiling_case_.reduce_is_block) {
    return ReduceBlockTiling(tile_out_axes, reduce_outer_axes, non_reduce_outer_axes);
  }

  // 对所有tile out轴先进行fuse
  FuseTileOutAxes(non_reduce_outer_axes, reduce_outer_axes);
  // 对fuse后的外轴进行block切分
  HandleBlockSplitting(tile_out_axes, non_reduce_outer_axes, reduce_outer_axes);

  return af::SUCCESS;
}

// R 切多核场景，reduce_block_id 是R用于切多核部分
Status Scheduler::ModifyStoreAfterReduce(ascir::NodeView &node, ascir::AxisId reduce_block_id) {
  auto reduce_block_axis = graph_.FindAxis(reduce_block_id);
  GE_ASSERT_NOTNULL(reduce_block_axis, "Cannot find reduce block axis with id:[%ld].", reduce_block_id);
  for (auto &output : node->outputs()) {
    GE_ASSERT_NOTNULL(output);
    auto &output_attr = output->attr;
    ascir::SizeExpr size_product = af::sym::kSymbolOne;
    for (const auto &repeat : output_attr.repeats) {
      size_product = size_product * repeat;
    }
    auto iter = std::find(output_attr.axis.begin(), output_attr.axis.end(), reduce_block_id);
    GE_ASSERT_TRUE(iter != output_attr.axis.end(), "Cannot find axis [%ld] from [%s]'s output tensor.", reduce_block_id,
                   node->GetNamePtr());
    size_t index = std::distance(output_attr.axis.begin(), iter);
    GE_ASSERT_TRUE(index < output_attr.repeats.size(), "Repeats of [%s]'s output tensor not greater than [%lu].",
                   node->GetNamePtr(), index);
    GE_ASSERT_TRUE(index < output_attr.strides.size(), "Strides of [%s]'s output tensor not greater than [%lu].",
                   node->GetNamePtr(), index);
    output_attr.repeats[index] = reduce_block_axis->size;
    output_attr.strides[index] = size_product;
  }
  return af::SUCCESS;
}

Status Scheduler::ApplyBlockSplitToNode(ascir::NodeView &node, bool is_store_after_reduce) {
  if (tiling_case_.merge_reduce_id != kDefaultAxisId) {
    auto merged_axes = graph_.FindAxis(tiling_case_.merge_reduce_id);
    GE_ASSERT_NOTNULL(merged_axes, "Cannot find merged axis with id:[%ld].", tiling_case_.merge_reduce_id);
    graph_.ApplySchedAxisMerge(node, tiling_case_.merge_reduce_id);
    if (is_store_after_reduce) {
      graph_.ApplyTensorAxisMerge(node, tiling_case_.merge_reduce_id);
    }
  }
  if (tiling_case_.merge_no_reduce_id != kDefaultAxisId) {
    auto merged_axes = graph_.FindAxis(tiling_case_.merge_no_reduce_id);
    GE_ASSERT_NOTNULL(merged_axes, "Cannot find merged axis with id:[%ld].", tiling_case_.merge_no_reduce_id);
    graph_.ApplySchedAxisMerge(node, tiling_case_.merge_no_reduce_id);
  }
  ApplyTiling(node, tiling_case_.reduce_block_tiling_id, tiling_case_.reduce_block_tiling);

  auto tile_block_axis = graph_.FindAxis(tiling_case_.block_tiling_id);
  GE_ASSERT_NOTNULL(tile_block_axis, "Cannot find block out axis with id:[%ld].", tiling_case_.block_tiling_id);
  if (tile_block_axis->type == ascir::Axis::Type::kAxisTypeMerged) {
    graph_.ApplySchedAxisMerge(node, tiling_case_.block_tiling_id);
  }

  auto reduce_out_axis = graph_.FindAxis(tiling_case_.reduce_outer_id);
  if (reduce_out_axis != nullptr && reduce_out_axis->type == ascir::Axis::Type::kAxisTypeMerged) {
    graph_.ApplySchedAxisMerge(node, tiling_case_.reduce_outer_id);
  }

  if (tiling_case_.reduce_is_block) {
    // 替换 reduce_block_tiling_id 轴对应的repeat 和 stride
    if (is_store_after_reduce) {
      ModifyStoreAfterReduce(node, tiling_case_.reduce_block_tiling.first->id);
    }
  } else {
    // 多核切R场景，block_tiling_id 是merge出来的, 不需要ApplyTiling
    ApplyTiling(node, tiling_case_.block_tiling_id, tiling_case_.block_tiling);
  }
  return af::SUCCESS;
}

void Scheduler::AdjustVectorizedAxesOrderOffsets(std::vector<size_t> &vectorized_axes_order, size_t split_point,
                                                 size_t end, size_t offset) const {
  const size_t start_idx = is_last_axis_reduce_ ? split_point : 0;
  const size_t end_idx = is_last_axis_reduce_ ? end : split_point;
  for (size_t i = start_idx; i < end_idx; ++i) {
    vectorized_axes_order[i] += offset;
  }
}

void Scheduler::FindVectorizedAxes(std::vector<ascir::AxisId> &vectorized_axes,
                                   std::vector<size_t> &vectorized_axes_order) {
  if (indirect_load_info_.active) {
    vectorized_axes = indirect_load_info_.axes.vectorized_axes;
    vectorized_axes_order.resize(vectorized_axes.size());
    std::iota(vectorized_axes_order.begin(), vectorized_axes_order.end(), 0UL);
    return;
  }

  size_t last_ub_size = 0UL;
  size_t group_axis_size = 0UL;
  const auto &axes_order = axes_group_.axes_order;
  if (HasXGroup()) {
    FindInnerAxes(vectorized_axes, axes_group_.x_group, tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
    const size_t current_ub_size = vectorized_axes.size();
    group_axis_size += axes_group_.x_group.size();
    AppendAxisOrder(axes_order, group_axis_size, last_ub_size, current_ub_size, vectorized_axes_order);
    last_ub_size = current_ub_size;
  }
  // y_group 一定非空
  {
    const size_t prev_ub_size = last_ub_size;
    FindInnerAxes(vectorized_axes, axes_group_.y_group, tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
    const size_t current_ub_size = vectorized_axes.size();
    group_axis_size += axes_group_.y_group.size();
    AppendAxisOrder(axes_order, group_axis_size, prev_ub_size, current_ub_size, vectorized_axes_order);
    last_ub_size = current_ub_size;
  }

  if (HasRGroup()) {
    const size_t prev_ub_size = last_ub_size;
    group_axis_size += axes_group_.r_group.size();
    FindInnerAxes(vectorized_axes, axes_group_.r_group, tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);
    const size_t current_ub_size = vectorized_axes.size();
    AppendAxisOrder(axes_order, group_axis_size, prev_ub_size, current_ub_size, vectorized_axes_order);

    // 带reduce 需要在ub内确保向量化轴是RA或者AR排布
    const size_t offset = axes_order.size() + axes_group_.n_group.size();
    AdjustVectorizedAxesOrderOffsets(vectorized_axes_order, prev_ub_size, current_ub_size, offset);
    last_ub_size = current_ub_size;
  }

  bool has_reduce = graph_cache_.HasComputeType(af::ComputeType::kComputeReduce);
  if (has_reduce && !HasRGroup()) {
    const size_t non_reduce_axis_size = vectorized_axes.size();
    vectorized_axes.insert(vectorized_axes.end(), axes_group_.n_group.begin(), axes_group_.n_group.end());
    for (size_t i = 0UL; i < axes_group_.n_group.size(); ++i) {
      vectorized_axes_order.push_back(i + axes_order.size());
    }

    const size_t offset = axes_order.size() + vectorized_axes.size();
    AdjustVectorizedAxesOrderOffsets(vectorized_axes_order, non_reduce_axis_size, vectorized_axes.size(), offset);
  }
}

Status Scheduler::RemoveRedundantBroadcastNode(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (!ScheduleUtils::IsBroadcast(node) || node->inputs.Size() != 1) {
      continue;
    }
    auto in_data_anchor = node->GetInDataAnchor(0);
    GE_CHECK_NOTNULL(in_data_anchor);
    auto pre_node_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(pre_node_out_anchor);
    auto pre_node = std::dynamic_pointer_cast<af::AscNode>(pre_node_out_anchor->GetOwnerNode());
    GE_CHECK_NOTNULL(pre_node);
    auto pre_node_out_index = static_cast<uint32_t>(pre_node_out_anchor->GetIdx());
    GE_CHK_BOOL_RET_STATUS(pre_node_out_index < pre_node->GetAllOutDataAnchorsSize(), af::FAILED,
                           "Broadcast input node %s[%s] output data anchor size is %u, but out anchor index is %u",
                           pre_node->GetTypePtr(), pre_node->GetNamePtr(), pre_node->GetAllOutDataAnchorsSize(),
                           pre_node_out_index);

    if (IsRedundantBroadcast(impl_graph, node, pre_node, pre_node_out_index)) {
      GELOGD("Graph [%s] Broadcast [%s] is redundant, remove it.", impl_graph.GetName().c_str(), node->GetNamePtr());
      GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(impl_graph, node, pre_node_out_anchor));
    } else if (IsContinuesBroadcast(impl_graph, node, pre_node)) {
      GELOGD("Graph [%s] Broadcast [%s] is continuous, remove it.", impl_graph.GetName().c_str(),
             pre_node->GetNamePtr());
      node->inputs[0].attr = pre_node->inputs[0].attr;
      GE_CHECK_NOTNULL(pre_node->GetInDataAnchor(0));
      auto pre_pre_node_out_anchor = pre_node->GetInDataAnchor(0)->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(pre_pre_node_out_anchor);
      GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(impl_graph, pre_node, pre_pre_node_out_anchor));
    } else {
      GELOGD("Graph [%s] Broadcast [%s] is useful, keep it.", impl_graph.GetName().c_str(), node->GetNamePtr());
    }
  }
  return af::SUCCESS;
}

namespace {
Status AddIndirectLoadSyntheticOuterAxis(const af::AscNodePtr &node, ascir::AxisId outer_axis_id,
                                         bool has_synthetic_outer_axis) {
  if (!has_synthetic_outer_axis) {
    return af::SUCCESS;
  }
  node->attr.sched.axis.insert(node->attr.sched.axis.begin(), outer_axis_id);
  for (const auto &output : node->outputs()) {
    GE_ASSERT_TRUE(!output->attr.axis.empty() && output->attr.axis.size() == output->attr.repeats.size() &&
                       output->attr.axis.size() == output->attr.strides.size(),
                   "Node[%s] has invalid tensor view.", node->GetNamePtr());
    const ascir::SizeExpr stride = af::sym::Mul(output->attr.repeats.front(), output->attr.strides.front());
    output->attr.axis.insert(output->attr.axis.begin(), outer_axis_id);
    output->attr.repeats.insert(output->attr.repeats.begin(), af::sym::kSymbolOne);
    output->attr.strides.insert(output->attr.strides.begin(), stride);
  }
  return af::SUCCESS;
}

Status ApplyIndirectLoadTemplateMerge(ascir::ImplGraph &graph, const af::AscNodePtr &node, ascir::AxisId axis_id,
                                      bool merge_tensor_axis = true) {
  if (axis_id == af::kIdNone) {
    return af::SUCCESS;
  }
  const auto axis = graph.FindAxis(axis_id);
  GE_ASSERT_NOTNULL(axis, "IndirectLoad template axis[%ld] is not found.", axis_id);
  const std::vector<int64_t> merged_from = axis->from;
  if (axis->type == ascir::Axis::Type::kAxisTypeMerged) {
    GELOGD("[IndirectLoad] Graph[%s] apply template merge axis[%ld] for node[%s].", graph.GetName().c_str(), axis_id,
           node->GetNamePtr());
    GE_ASSERT_TRUE(graph.ApplySchedAxisMerge(node, axis_id), "Failed to merge schedule axis[%ld] for node[%s].",
                   axis_id, node->GetNamePtr());
    if (merge_tensor_axis) {
      GE_ASSERT_TRUE(graph.ApplyTensorAxisMerge(node, axis_id), "Failed to merge tensor axis[%ld] for node[%s].",
                     axis_id, node->GetNamePtr());
    }
  } else {
    GELOGD("[IndirectLoad] Graph[%s] apply single-axis template merge axis[%ld] for node[%s].", graph.GetName().c_str(),
           axis_id, node->GetNamePtr());
    GE_ASSERT_TRUE(graph.ApplySchedAxisMerge(node, axis_id, {axis_id}),
                   "Failed to merge schedule axis[%ld] for node[%s].", axis_id, node->GetNamePtr());
    if (merge_tensor_axis) {
      GE_ASSERT_TRUE(graph.ApplyTensorAxisMerge(node, axis_id, {axis_id}),
                     "Failed to merge tensor axis[%ld] for node[%s].", axis_id, node->GetNamePtr());
    }
  }
  if (merge_tensor_axis) {
    // [vectorized_axis 一致性] tensor merge 只折叠 axis/repeats/strides，不处理
    // vectorized_axis——merge 后旧轴失效使 vectorized_axis ⊄ axis，中间窗口
    // （merge → 调度末尾 SyncSimtBoundaryViews 的 remap）内公共检查（如
    // GetVectorRepeats/InitTensorMemInfo）会打出误导性 ERROR（实际为容错路径）。
    // merge 后立即把失效旧轴替换为 merged 轴：向量化身份由 merged 轴承载，后续
    // RemapOutputVectorizedAxes 的 direct_ancestor（IsAxisAncestorOf 自反）路径
    // 将其映射到 split 后的 inner 轴，两级映射无缝衔接。多个旧轴折叠到同一
    // merged 轴时去重。
    for (const auto &output : node->outputs()) {
      if (output == nullptr) {
        continue;
      }
      std::vector<ascir::AxisId> remapped_vec;
      remapped_vec.reserve(output->attr.vectorized_axis.size());
      bool vec_remapped = false;
      for (const auto vec_axis : output->attr.vectorized_axis) {
        const bool stale =
            std::find(output->attr.axis.begin(), output->attr.axis.end(), vec_axis) == output->attr.axis.end() &&
            std::find(merged_from.begin(), merged_from.end(), vec_axis) != merged_from.end();
        const ascir::AxisId mapped = stale ? axis_id : vec_axis;
        if (stale) {
          vec_remapped = true;
        }
        if (std::find(remapped_vec.begin(), remapped_vec.end(), mapped) == remapped_vec.end()) {
          remapped_vec.emplace_back(mapped);
        }
      }
      if (vec_remapped) {
        GELOGI("[IndirectLoad] Graph[%s] node[%s] vectorized axis remapped to merged axis[%ld] after tensor merge.",
               graph.GetName().c_str(), node->GetNamePtr(), axis_id);
      }
      output->attr.vectorized_axis = std::move(remapped_vec);
    }
  }
  return af::SUCCESS;
}

Status SetOuterRepeatsToOne(const af::AscNodePtr &node, const std::vector<ascir::AxisId> &vectorized_axes) {
  if (node->attr.api.compute_type != af::ComputeType::kComputeReduce) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(!vectorized_axes.empty(), "Node[%s] has no IndirectLoad vector axes.", node->GetNamePtr());
  for (const auto &output : node->outputs()) {
    GE_ASSERT_TRUE(output->attr.axis.size() == output->attr.repeats.size());
    GE_ASSERT_TRUE(output->attr.axis.size() >= vectorized_axes.size());
    std::fill_n(output->attr.repeats.begin(), output->attr.axis.size() - vectorized_axes.size(), af::sym::kSymbolOne);
  }
  return af::SUCCESS;
}

Status SetInputInnerVectorizedView(const af::AscNodePtr &node, const ascir::Axis &input_inner_axis,
                                   af::AscTensorAttr &attr) {
  GE_ASSERT_TRUE(attr.axis.size() == attr.repeats.size() && attr.axis.size() == attr.strides.size(),
                 "IndirectLoad input-pre node[%s] has invalid tensor view.", node->GetNamePtr());
  std::vector<ascir::AxisId> vectorized_axes;
  if (std::find(attr.axis.begin(), attr.axis.end(), input_inner_axis.id) != attr.axis.end()) {
    vectorized_axes.emplace_back(input_inner_axis.id);
  } else if (input_inner_axis.type == ascir::Axis::Type::kAxisTypeMerged) {
    for (const ascir::AxisId tensor_axis : attr.axis) {
      if (std::find(input_inner_axis.from.begin(), input_inner_axis.from.end(), tensor_axis) !=
          input_inner_axis.from.end()) {
        vectorized_axes.emplace_back(tensor_axis);
      }
    }
    GE_ASSERT_TRUE(vectorized_axes.size() == input_inner_axis.from.size(),
                   "Cannot map input inner axis[%ld] to tensor axes for node[%s].", input_inner_axis.id,
                   node->GetNamePtr());
  }
  GE_ASSERT_TRUE(!vectorized_axes.empty(), "Input inner axis[%ld] is not in node[%s]'s output tensor.",
                 input_inner_axis.id, node->GetNamePtr());

  attr.vectorized_axis = std::move(vectorized_axes);
  attr.vectorized_strides.clear();
  attr.vectorized_strides.reserve(attr.vectorized_axis.size());
  for (const ascir::AxisId vectorized_axis : attr.vectorized_axis) {
    const auto iter = std::find(attr.axis.begin(), attr.axis.end(), vectorized_axis);
    GE_ASSERT_TRUE(iter != attr.axis.end(), "Vectorized axis[%ld] is not in node[%s]'s output tensor.", vectorized_axis,
                   node->GetNamePtr());
    attr.vectorized_strides.emplace_back(attr.strides[static_cast<size_t>(std::distance(attr.axis.begin(), iter))]);
  }
  return af::SUCCESS;
}

Status ApplyInputInnerVectorizedAxis(ascir::ImplGraph &graph, const af::AscNodePtr &node,
                                     ascir::AxisId input_inner_axis_id) {
  GE_ASSERT_TRUE(input_inner_axis_id != af::kIdNone, "IndirectLoad input inner axis is missing for node[%s].",
                 node->GetNamePtr());
  GE_ASSERT_TRUE(!node->outputs().empty(), "IndirectLoad input-pre node[%s] has no output.", node->GetNamePtr());
  const auto input_inner_axis = graph.FindAxis(input_inner_axis_id);
  GE_ASSERT_NOTNULL(input_inner_axis, "IndirectLoad input inner axis[%ld] is not found.", input_inner_axis_id);
  if (input_inner_axis->type == ascir::Axis::Type::kAxisTypeMerged) {
    GELOGD("[IndirectLoad] Graph[%s] apply input inner vectorized axis[%ld] for node[%s].", graph.GetName().c_str(),
           input_inner_axis_id, node->GetNamePtr());
    GE_ASSERT_TRUE(graph.ApplySchedAxisMerge(node, input_inner_axis_id),
                   "Failed to merge input inner schedule axis[%ld] for node[%s].", input_inner_axis_id,
                   node->GetNamePtr());
    GE_ASSERT_TRUE(graph.ApplyTensorAxisMerge(node, input_inner_axis_id),
                   "Failed to merge input inner tensor axis[%ld] for node[%s].", input_inner_axis_id,
                   node->GetNamePtr());
  }

  for (const auto &output : node->outputs()) {
    GE_ASSERT_SUCCESS(SetInputInnerVectorizedView(node, *input_inner_axis, output->attr));
  }
  return af::SUCCESS;
}

}  // namespace

ascir::AxisId Scheduler::GetSolvedTileInnerAxis() const {
  // 固定 tile（预建 pair）保持既有单行窗口语义；可求解 tile 返回通用 TileSplit 产物
  // TileInner 轴 id（与 TileSplit 中向量化视图前插的判定条件一致）。
  if (indirect_load_info_.axes.tile_inner_axis != af::kIdNone) {
    return af::kIdNone;
  }
  return tiling_case_.ub_tiling_y.second == nullptr ? af::kIdNone : tiling_case_.ub_tiling_y.second->id;
}

Status Scheduler::FoldSolvedTileInnerIntoPreWindow(const af::AscNodePtr &node) const {
  // 可求解多行 tile：TileInner（tile 内行数）并入输入侧 pre 节点的向量化窗口，使输入
  // staging 覆盖 tile 内多行（R×inner 批量搬运）而非按行循环。须在 ApplyTiling 之后
  // 调用（节点视图完成 a0 -> a0T/a0t 分裂后才能看到 TileInner 轴）。
  const ascir::AxisId tile_inner_axis_id = GetSolvedTileInnerAxis();
  if (tile_inner_axis_id == af::kIdNone) {
    return af::SUCCESS;
  }
  const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
  // StridedUbPath（带行间 padding 的对齐路径）保持既有单行窗口语义：其紧凑写回与
  // 视图标注机制假定逐行 staging 结构，多行窗口不在本变更范围内。
  if (role == ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath) {
    return af::SUCCESS;
  }
  const bool is_pre_node = ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node) ||
                           role == ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre;
  if (!is_pre_node) {
    return af::SUCCESS;
  }
  for (const auto &output : node->outputs()) {
    auto &attr = output->attr;
    const auto tile_iter = std::find(attr.axis.begin(), attr.axis.end(), tile_inner_axis_id);
    if (tile_iter == attr.axis.end() || std::find(attr.vectorized_axis.begin(), attr.vectorized_axis.end(),
                                                  tile_inner_axis_id) != attr.vectorized_axis.end()) {
      continue;
    }
    const auto tile_index = static_cast<size_t>(std::distance(attr.axis.begin(), tile_iter));
    const bool zero_stride =
        af::SymbolicUtils::StaticCheckEq(attr.strides[tile_index], af::sym::kSymbolZero) == af::TriBool::kTrue;
    const bool multi_repeat =
        af::SymbolicUtils::StaticCheckEq(attr.repeats[tile_index], af::sym::kSymbolOne) != af::TriBool::kTrue;
    // 零步长且窗口展开（广播流）不并入：并入会使 staging 重复搬运 R 份相同数据；
    // 其余情形并入——多行窗口批量化，或退化轴（repeat==1）仅上移循环层级。
    if (zero_stride && multi_repeat) {
      continue;
    }
    // 插入位置须保持 vectorized_axis 为 attr.axis 的子序列：转置视图下行维度不一定
    // 位于内层轴之前（如 origin [1,5,9,13] 中内层轴 1 先于行轴 9），盲目前插会破坏
    // 视图轴序一致性（CheckContinuous）。按 tile 轴在 attr.axis 中的位置确定插入点。
    size_t insert_pos = 0UL;
    for (size_t axis_pos = 0UL; axis_pos < tile_index; ++axis_pos) {
      if (std::find(attr.vectorized_axis.begin(), attr.vectorized_axis.end(), attr.axis[axis_pos]) !=
          attr.vectorized_axis.end()) {
        ++insert_pos;
      }
    }
    attr.vectorized_axis.insert(attr.vectorized_axis.begin() + static_cast<int64_t>(insert_pos), tile_inner_axis_id);
    attr.vectorized_strides.insert(attr.vectorized_strides.begin() + static_cast<int64_t>(insert_pos),
                                   attr.strides[tile_index]);
  }
  return af::SUCCESS;
}

Status Scheduler::InitIndirectLoadScheduleCase() {
  indirect_load_info_ = {};
  const af::AscNodePtr indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph_);
  if (indirect_load == nullptr) {
    return af::SUCCESS;
  }

  // post-Reduce SIMT 候选的 prebuilt tile pair 为空，延迟到 TileTiling 走通用 TileSplit
  // 按 UB 容量求解 tile 行数；此处仅要求 prebuilt y 轴存在（固定 pair 场景该轴同样有效）。
  GE_ASSERT_TRUE(tiling_case_.ub_tiling_id_y != kDefaultAxisId,
                 "IndirectLoad graph[%s] has no prebuilt outer tiling axis.", graph_.GetName().c_str());
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, indirect_load_info_.axes));
  indirect_load_info_.active = true;
  return af::SUCCESS;
}

Status Scheduler::ApplyIndirectLoadNodeAxes(const af::AscNodePtr &node, bool &skip_main_tiling) const {
  skip_main_tiling = false;
  if (!indirect_load_info_.active) {
    return af::SUCCESS;
  }
  const bool is_input_pre = ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(node);
  if (is_input_pre) {
    // 双 gather 水平融合：优先使用节点上的 per-IL 输入内轴提示（两路 input 的 dim 轴
    // 尺寸可不同）；单 gather 场景无提示，回退全局模板轴，行为与原实现一致。
    const auto hint_axis = ascgen_utils::indirect_load::GetSimdInputInnerAxis(node);
    GE_ASSERT_SUCCESS(ApplyInputInnerVectorizedAxis(
        graph_, node, hint_axis != af::kIdNone ? hint_axis : indirect_load_info_.axes.input_inner_axis));
  }
  const bool is_index_pre =
      ascgen_utils::indirect_load::GetTemplateRole(node) == ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre;
  if (is_index_pre) {
    GE_ASSERT_SUCCESS(ApplyInputInnerVectorizedAxis(graph_, node, indirect_load_info_.axes.index_inner_axis));
  }
  skip_main_tiling = ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_main_schedule_tiling;
  if (skip_main_tiling) {
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const bool is_simt_role = role == ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform ||
                              role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch;
    // SIMT 角色节点虽由 scalar evaluator 发射、跳过主 tiling 切分，但其输出可能同时
    // 被 normal schedule 的 VF 子图消费（如 IndirectLoad 输出直连链汇入 post Reduce）。
    // VF 分区的边界 Load 会拷贝此处 view，必须与消费者经模板 merge 改写后的 view 一致，
    // 否则 ValidateInputTensorLoopAxis 报 input axis not in output axis。
    // 因此 SIMT 角色节点仍需执行 tensor 轴 merge（merge_tensor_axis=true），仅跳过 tiling。
    // 同理，kSimtInputBoundary/kSimtDirectGmBoundary/kSkInputBoundary 等边界节点也跳过主
    // tiling，若不补 tensor 轴 merge，其 view 停留在旧轴空间（如 [0,1]），而模板 merge 已
    // 将调度轴合并改写（[0,1] -> [2] -> [5,6,4]），BufQueAllocator::InitTensorMemInfo 遍历
    // vectorized_axis 时会因 axis 已不存在而失败（Cannot find vectorized axis）。
    const bool is_simt_boundary_role = role == ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary ||
                                       role == ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary ||
                                       role == ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary;
    if (is_simt_role || is_simt_boundary_role) {
      GELOGD("[IndirectLoad] SIMT role node[%s] role[%d] keeps tensor axis merge before skipping main tiling.",
             node->GetNamePtr(), static_cast<int32_t>(role));
      // merge 前置校验：节点的调度轴需包含 outer/inner 全部成员轴才能折叠；index 广播源
      // 等节点可能持有独立副本轴（如 z*_index，仅与输出轴同形不同 id），部分匹配时
      // ApplySchedAxisMerge 的连续性断言会失败。此时保持节点原视图（与主线路径行为
      // 一致：这些节点不参与模板轴折叠，由消费者侧 BroadcastBackward 等统一处理）。
      const auto sched_has_all_members = [&graph = this->graph_](const af::AscNodePtr &n,
                                                                 const ascir::AxisId merged_axis_id) {
        if (merged_axis_id == af::kIdNone) {
          return true;
        }
        const auto merged_axis = graph.FindAxis(merged_axis_id);
        if (merged_axis == nullptr || merged_axis->from.empty()) {
          return true;
        }
        for (const auto member : merged_axis->from) {
          if (std::find(n->attr.sched.axis.begin(), n->attr.sched.axis.end(), member) == n->attr.sched.axis.end()) {
            return false;
          }
        }
        return true;
      };
      if (sched_has_all_members(node, indirect_load_info_.axes.outer_axis)) {
        GE_ASSERT_SUCCESS(ApplyIndirectLoadTemplateMerge(graph_, node, indirect_load_info_.axes.outer_axis, true));
      }
      if (sched_has_all_members(node, indirect_load_info_.axes.inner_axis)) {
        GE_ASSERT_SUCCESS(ApplyIndirectLoadTemplateMerge(graph_, node, indirect_load_info_.axes.inner_axis, false));
      }
      // SIMT 节点跳过主 tiling，其 tensor view 的 split 同步与 vectorized_axis 重映射
      // 统一由 DoScheduler 末尾的 SyncSimtBoundaryViews 完成（收敛在需求自有文件
      // simt_boundary_sync.cpp），此处不再嵌入补丁逻辑。
    }
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddIndirectLoadSyntheticOuterAxis(node, indirect_load_info_.axes.outer_axis,
                                                      indirect_load_info_.axes.synthetic_outer));
  GE_ASSERT_SUCCESS(ApplyIndirectLoadTemplateMerge(graph_, node, indirect_load_info_.axes.outer_axis,
                                                   !is_input_pre && !is_index_pre));
  if (is_input_pre) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(ApplyIndirectLoadTemplateMerge(graph_, node, indirect_load_info_.axes.inner_axis, false));
  GE_ASSERT_SUCCESS(SetOuterRepeatsToOne(node, indirect_load_info_.axes.vectorized_axes));
  return af::SUCCESS;
}

std::vector<ascir::AxisId> Scheduler::GetSortedNodeVectorizedAxes(Scheduler &scheduler) {
  std::vector<ascir::AxisId> vectorized_axes;
  std::vector<size_t> vectorized_axes_order;
  scheduler.FindVectorizedAxes(vectorized_axes, vectorized_axes_order);

  // reorder vectorized axis by original node axis order
  std::vector<size_t> base_order(vectorized_axes.size());
  std::iota(base_order.begin(), base_order.end(), 0UL);
  std::sort(base_order.begin(), base_order.end(), [&vectorized_axes_order](size_t a, size_t b) {
    return vectorized_axes_order[a] < vectorized_axes_order[b];
  });

  std::vector<ascir::AxisId> sorted;
  sorted.reserve(base_order.size());
  for (const size_t index : base_order) {
    sorted.push_back(vectorized_axes[index]);
  }
  return sorted;
}

Status Scheduler::TileSplit() {
  // split ub
  TileTiling(tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
  TileTiling(tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
  TileTiling(tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);

  if (indirect_load_info_.active && indirect_load_info_.axes.tile_inner_axis == af::kIdNone &&
      tiling_case_.ub_tiling_y.second != nullptr) {
    // post-Reduce SIMT 可求解 tile（通用 TileSplit 产物）：TileInner（tile 内行数）加入
    // 向量化视图前部，使其成为 API 一次处理的窗口维度而非内层循环——SIMT VF_CALL、
    // SoftmaxAR（A=行数, R=尾轴）、Reduce 均按 tile 内多行批量执行；同时
    // SetOuterRepeatsToOne 仅折叠外层循环轴，Reduce 输出保留行数维，避免多行
    // 共用首行统计值的数值错误。固定 tile（SIMD/SK/无 post-Reduce SIMT）不经过此处。
    indirect_load_info_.axes.vectorized_axes.insert(indirect_load_info_.axes.vectorized_axes.begin(),
                                                    tiling_case_.ub_tiling_y.second->id);
    GELOGD("[IndirectLoad] Graph[%s] prepend solved tile-inner axis[%ld] to vectorized view.", graph_.GetName().c_str(),
           tiling_case_.ub_tiling_y.second->id);
  }

  auto sorted_node_vectorized_axes = GetSortedNodeVectorizedAxes(*this);

  bool has_reduce = graph_cache_.HasComputeType(af::ComputeType::kComputeReduce);
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    bool skip_main_tiling = false;
    GE_CHK_STATUS_RET(ApplyIndirectLoadNodeAxes(node, skip_main_tiling));
    if (skip_main_tiling) {
      continue;
    }
    ApplyTiling(node, tiling_case_.ub_tiling_id_x, tiling_case_.ub_tiling_x);
    ApplyTiling(node, tiling_case_.ub_tiling_id_y, tiling_case_.ub_tiling_y);
    ApplyTiling(node, tiling_case_.ub_tiling_id_r, tiling_case_.ub_tiling_r);
    if (indirect_load_info_.active) {
      GE_CHK_STATUS_RET(FoldSolvedTileInnerIntoPreWindow(node), "Failed to fold solved tile inner window for node[%s].",
                        node->GetNamePtr());
    }

    auto axes = node->attr.sched.axis;
    const auto &n_group = this->axes_group_.n_group;

    auto node_vectorized_axes = sorted_node_vectorized_axes;
    if (reduce_template_ != optimize::ReduceTemplateType::kAllLoad) {
      for (int64_t axis_id : axes) {
        if (std::find(n_group.begin(), n_group.end(), axis_id) != n_group.end()) {
          node_vectorized_axes.push_back(axis_id);
        }
      }
    }

    // 非reduce场景应该将向量化轴调整为tensor中的相对顺序, 带reduce场景由于tiling策略已经做了特别的reorder,需要跳过
    // tiling策略暂时无法支持具有reduce和transpose融合的场景
    for (auto &output : node->outputs()) {
      if (indirect_load_info_.active &&
          ascgen_utils::indirect_load::GetTemplateBehavior(node).preserves_vectorized_axis) {
        continue;
      }
      output->attr.vectorized_axis = node_vectorized_axes;
      if (!has_reduce) {
        auto tensor_axis = output->attr.axis;
        std::sort(output->attr.vectorized_axis.begin(), output->attr.vectorized_axis.end(),
                  [&tensor_axis](const int64_t &lhs, const int64_t &rhs) {
                    return CompareByOrderInTensorAxis(lhs, rhs, tensor_axis);
                  });
      }
    }
  }
  return af::SUCCESS;
}

Status Scheduler::DoScheduler() {
  if (cube_template_ == ascir::CubeTemplateType::kFixpip) {
    ascir::utils::DumpGraph(graph_, "AfterDoTiling");
    return af::SUCCESS;
  }
  GE_CHK_STATUS_RET(InitIndirectLoadScheduleCase(), "Failed to init IndirectLoad schedule case for graph[%s].",
                    graph_.GetName().c_str());
  RemoveDuplicatedAxisFromGroup();
  // Tile Split
  TileSplit();
  // Block Split
  if (cube_template_ != ascir::CubeTemplateType::kUBFuse) {
    std::vector<ascir::AxisId> new_sched_axes;
    GE_CHK_STATUS_RET(BlockSplit(new_sched_axes), "Failed to gen tile outer axis, graph:[%s].",
                      graph_.GetName().c_str());
    // Apply block split for node
    GE_CHK_STATUS_RET(ApplyBlockSplit(new_sched_axes));
  }
  GE_CHK_STATUS_RET(SynchronizeTransposeInputSchedAxis());
  GE_CHK_STATUS_RET(RemoveRedundantBroadcastNode(graph_));
  // IndirectLoad SIMT 边界适配（收敛在需求自有文件）：调度全部 split 完成后，
  // 一次性将 SIMT 角色节点 tensor view 对齐到与普通节点等价的状态。普通图
  // （无 IndirectLoad）在函数内部直接返回，零影响。
  if (indirect_load_info_.active) {
    const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> tiled_axes_list = {
        tiling_case_.ub_tiling_x, tiling_case_.ub_tiling_y, tiling_case_.ub_tiling_r, tiling_case_.block_tiling,
        tiling_case_.reduce_block_tiling};
    GE_CHK_STATUS_RET(optimize::task_generator::SyncSimtBoundaryViews(graph_, tiled_axes_list),
                      "Failed to sync SIMT boundary views for graph[%s].", graph_.GetName().c_str());
  }
  auto align_ret = AlignmentHandler::AlignVectorizedStrides(graph_);
  if (align_ret != af::SUCCESS) {
    return align_ret;  // 返回 UNSUPPORTED 让上层跳过这个模板
  }
  GE_ASSERT_SUCCESS(NodeCacheMarker(graph_).MarkIfNodeNeedsCache());
  GE_ASSERT_SUCCESS(AlignmentHandler::ModifyVectorizedStrides(graph_));
  ascir::utils::DumpGraph(graph_, "AfterDoTiling");
  return af::SUCCESS;
}

Status Scheduler::ApplyBlockSplit(const std::vector<ascir::AxisId> &new_sched_axes) {
  bool is_reduce_after = false;
  for (auto node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node) ||
        (indirect_load_info_.active &&
         ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_main_schedule_tiling)) {
      continue;
    }
    if ((!is_reduce_after) && ScheduleUtils::IsReduce(node)) {
      is_reduce_after = true;
    }

    std::vector<ascir::AxisId> node_new_sched_axes = new_sched_axes;
    GE_ASSERT_TRUE(!node->outputs.operator()().empty());
    auto &vectorized_axis = node->outputs[0].attr.vectorized_axis;
    if (!indirect_load_info_.active) {
      node_new_sched_axes.insert(node_new_sched_axes.end(), vectorized_axis.begin(), vectorized_axis.end());
    }
    bool is_store_after_reduce = is_reduce_after && ScheduleUtils::IsStore(node);
    GE_ASSERT_SUCCESS(ApplyBlockSplitToNode(node, is_store_after_reduce));
    if (indirect_load_info_.active) {
      continue;
    }
    graph_.ApplySchedAxisReorder(node, node_new_sched_axes);
  }
  return af::SUCCESS;
}

Status Scheduler::SynchronizeTransposeInputSchedAxis() {
  af::AscNodePtr transpose_node = nullptr;
  for (const auto &node : graph_.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      continue;
    }
    if (transpose_node != nullptr) {
      // 多 Transpose 场景应在调度前完成消除，这里不选择任意一个 Transpose 作为同步基准。
      return af::SUCCESS;
    }
    transpose_node = node;
  }
  if (transpose_node == nullptr) {
    return af::SUCCESS;
  }

  std::queue<af::AscNodePtr> path_nodes;
  std::set<af::Node *> visited_nodes;
  for (const auto &input_node : transpose_node->GetInDataNodes()) {
    path_nodes.push(std::dynamic_pointer_cast<af::AscNode>(input_node));
  }
  while (!path_nodes.empty()) {
    const auto current_node = path_nodes.front();
    path_nodes.pop();
    if (current_node == nullptr || !visited_nodes.emplace(current_node.get()).second) {
      continue;
    }
    if (!ScheduleUtils::IsBuffer(current_node)) {
      GE_ASSERT_TRUE(graph_.ApplySchedAxisReorder(current_node, transpose_node->attr.sched.axis),
                     "Failed to synchronize sched.axis for node[%s] with transpose[%s].", current_node->GetNamePtr(),
                     transpose_node->GetNamePtr());
    }
    for (const auto &input_node : current_node->GetInDataNodes()) {
      path_nodes.push(std::dynamic_pointer_cast<af::AscNode>(input_node));
    }
  }
  return af::SUCCESS;
}

void Scheduler::RemoveDuplicatedAxisFromGroup() {
  if (tiling_case_.ub_tiling_id_x != kDefaultAxisId) {
    auto it = std::find(axes_group_.y_group.begin(), axes_group_.y_group.end(), tiling_case_.ub_tiling_id_x);
    if (it != axes_group_.y_group.end()) {
      auto dis = std::distance(axes_group_.y_group.begin(), it);
      axes_group_.y_group.erase(axes_group_.y_group.begin() + dis);
      axes_group_.axes_order.erase(axes_group_.axes_order.begin() + static_cast<int64_t>(axes_group_.x_group.size()) +
                                   dis);
    }
  }

  if (tiling_case_.ub_tiling_id_y != kDefaultAxisId) {
    auto it = std::find(axes_group_.x_group.begin(), axes_group_.x_group.end(), tiling_case_.ub_tiling_id_y);
    if (it != axes_group_.x_group.end()) {
      auto dis = std::distance(axes_group_.x_group.begin(), it);
      axes_group_.x_group.erase(axes_group_.x_group.begin() + dis);
      axes_group_.axes_order.erase(axes_group_.axes_order.begin() + dis);
    }
  }

  // 从x中删除y中存在的id
  std::vector<int64_t> indices_to_remove;
  for (size_t i = 0UL; i < axes_group_.x_group.size(); ++i) {
    auto it = std::find(axes_group_.y_group.begin(), axes_group_.y_group.end(), axes_group_.x_group[i]);
    if (it != axes_group_.y_group.end()) {
      indices_to_remove.push_back(static_cast<int64_t>(i));
    }
  }

  for (auto i = static_cast<int64_t>(indices_to_remove.size() - 1); i >= 0; --i) {
    int64_t index = indices_to_remove[i];
    axes_group_.x_group.erase(axes_group_.x_group.begin() + index);
    // 从order中删除对应位置的值
    axes_group_.axes_order.erase(axes_group_.axes_order.begin() + index);
  }
}
}  // namespace optimize::autoschedule
