/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir/ascendc_ir_check.h"
#include "graph/expression/const_values.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "utils/axis_utils.h"
namespace af {
namespace {
using AstAxisIdToTransInfo = std::map<AxisId, const OneTransInfo *>;
std::vector<AxisId> ToAxisIds(const std::vector<AxisPtr> &axes) {
  std::vector<AxisId> axes_ids;
  axes_ids.reserve(axes.size());
  for (const auto &axis : axes) {
    axes_ids.emplace_back(axis->id);
  }
  return axes_ids;
}

AstAxisIdToTransInfo ToTransInfoMap(const TransInfoRoadOfGraph &trans_info_road_of_graph) {
  AstAxisIdToTransInfo res;
  for (const auto &trans_info : trans_info_road_of_graph) {
    for (const auto &dst_axis : trans_info.dst_axis) {
      res[dst_axis->id] = &trans_info;
    }
  }
  return res;
}
const OneTransInfo *GetTransInfo(const AstAxisIdToTransInfo &trans_infos, const AxisId dst_axis_id) {
  const auto iter = trans_infos.find(dst_axis_id);
  if (iter == trans_infos.cend()) {
    return nullptr;
  }
  return iter->second;
}

std::vector<OneTransInfo> GetTransInfos(const AstAxisIdToTransInfo &trans_infos,
                                        const std::vector<AxisId> &dst_axis_ids, bool revert = false,
                                        std::unordered_set<const OneTransInfo *> uniq_set = {}) {
  std::vector<OneTransInfo> res;
  for (const auto &dst_axis_id : dst_axis_ids) {
    const auto info = GetTransInfo(trans_infos, dst_axis_id);
    if ((info != nullptr) && (uniq_set.insert(info).second)) {
      res.emplace_back(*info);
      // if set revert, need find trans info from src_axis
      auto got_trans_infos = GetTransInfos(
          trans_infos, (revert ? ToAxisIds(info->src_axis) : ToAxisIds(info->dst_axis)), revert, uniq_set);
      res.insert(res.cend(), got_trans_infos.cbegin(), got_trans_infos.cend());
    }
  }
  return res;
}

bool AreAllSrcReady(const View &tensor_view_to_update, const OneTransInfo &trans_info) {
  // check if all src of trans_info(check) is all in axes(source)
  return std::all_of(trans_info.src_axis.begin(), trans_info.src_axis.end(),
                     [&tensor_view_to_update](const AxisPtr &axis) {
                       auto &axis_ids = tensor_view_to_update.axis_ids;
                       return std::find(axis_ids.begin(), axis_ids.end(), axis->id) != axis_ids.end();
                     });
}

std::vector<OneTransInfo> ToRevertTransInfo(const std::vector<OneTransInfo> &trans_infos) {
  std::vector<OneTransInfo> revert_trans_infos;
  for (const auto &trans_info : trans_infos) {
    if (trans_info.trans_type == TransType::kSplit) {
      revert_trans_infos.emplace_back(OneTransInfo{TransType::kMerge, trans_info.dst_axis, trans_info.src_axis});
    } else if (trans_info.trans_type == TransType::kMerge) {
      revert_trans_infos.emplace_back(OneTransInfo{TransType::kSplit, trans_info.dst_axis, trans_info.src_axis});
    }
  }
  return revert_trans_infos;
}

std::pair<std::vector<OneTransInfo>, std::vector<OneTransInfo>> UpdateReadyTransInfos(
    const View &tensor_view_to_update, const std::vector<OneTransInfo> &not_ready_trans_infos) {
  std::vector<OneTransInfo> to_apply_trans_infos;
  std::vector<OneTransInfo> not_read_trans_infos_updated;
  for (auto iter = not_ready_trans_infos.begin(); iter != not_ready_trans_infos.end();) {
    if (AreAllSrcReady(tensor_view_to_update, *iter)) {
      to_apply_trans_infos.emplace_back(*iter);
    } else {
      not_read_trans_infos_updated.emplace_back(*iter);
    }
    ++iter;
  }
  return {to_apply_trans_infos, not_read_trans_infos_updated};
}

bool CheckAxisValid(const OneTransInfo &one_trans_info, const bool revert) {
  const auto trans_type = one_trans_info.trans_type;
  bool need_check_dst_merged =
      (!revert && trans_type == TransType::kMerge) || (revert && (trans_type == TransType::kSplit));
  const auto merged_axis = revert ? one_trans_info.src_axis.front() : one_trans_info.dst_axis.front();
  if (need_check_dst_merged) {
    GE_ASSERT_TRUE(merged_axis->type == Axis::kAxisTypeMerged, "[Check][Axis] failed, axis id[%d], type[%d].",
                   merged_axis->id, merged_axis->type);
  }
  bool need_check_dst_split =
      (!revert && trans_type == TransType::kSplit) || (revert && (trans_type == TransType::kMerge));
  const auto outer_axis = revert ? one_trans_info.src_axis.front() : one_trans_info.dst_axis.front();
  const auto inner_axis = revert ? one_trans_info.src_axis.back() : one_trans_info.dst_axis.back();
  if (need_check_dst_split) {
    GE_ASSERT_TRUE(outer_axis->type == Axis::kAxisTypeBlockOuter || outer_axis->type == Axis::kAxisTypeTileOuter,
                   "[Check][Axis] failed, axis id[%d], type[%d]", outer_axis->id, outer_axis->type);
    GE_ASSERT_TRUE(inner_axis->type == Axis::kAxisTypeBlockInner || inner_axis->type == Axis::kAxisTypeTileInner,
                   "[Check][Axis] failed, axis id[%d], type[%d]", outer_axis->id, outer_axis->type);
  }
  return true;
}

DiffAxesInfo GetDiffAxesInfo(const std::vector<int64_t> &input_api_sched_axes,
                             const std::vector<int64_t> &my_api_sched_axes) {
  DiffAxesInfo diff_axes_info;
  diff_axes_info.add_axes = my_api_sched_axes;
  for (const auto &input_api_sched_axis : input_api_sched_axes) {
    const auto iter = std::find(diff_axes_info.add_axes.cbegin(), diff_axes_info.add_axes.cend(), input_api_sched_axis);
    if (iter != diff_axes_info.add_axes.cend()) {
      diff_axes_info.add_axes.erase(iter);
    } else {
      diff_axes_info.del_axes.emplace_back(input_api_sched_axis);
    }
  }
  return diff_axes_info;
}
// `pair.first == false` means apply failed
std::pair<bool, View> ApplyViewTrans(const TransInfoRoadOfGraph &trans_info_road_of_graph, const bool revert,
                                     View &tensor_view_to_update) {
  for (const auto &one_trans_info : trans_info_road_of_graph) {
    GE_ASSERT_TRUE(CheckAxisValid(one_trans_info, revert));
    switch (one_trans_info.trans_type) {
      case TransType::kSplit:
        GE_ASSERT_TRUE(one_trans_info.src_axis.size() == 1U, "[Check][Axis], size[%zu]",
                       one_trans_info.src_axis.size());
        GE_ASSERT_TRUE(one_trans_info.dst_axis.size() == 2U, "[Check][Axis], size[%zu]",
                       one_trans_info.dst_axis.size());
        GE_ASSERT_NOTNULL(one_trans_info.src_axis.front());
        GE_ASSERT_NOTNULL(one_trans_info.dst_axis.front());
        GE_ASSERT_NOTNULL(one_trans_info.dst_axis.back());
        tensor_view_to_update = AxisUtils::SplitView(
            tensor_view_to_update, one_trans_info.dst_axis.back()->size, one_trans_info.dst_axis.front()->id,
            one_trans_info.dst_axis.back()->id, one_trans_info.src_axis.front()->id);

        break;
      case TransType::kMerge: {
        GE_ASSERT_TRUE(one_trans_info.src_axis.size() >= 2U, "[Check][Axis], size[%zu]",
                       one_trans_info.src_axis.size());
        GE_ASSERT_TRUE(one_trans_info.dst_axis.size() == 1U, "[Check][Axis], size[%zu]",
                       one_trans_info.dst_axis.size());
        GE_ASSERT_NOTNULL(one_trans_info.dst_axis.front());
        std::vector<int64_t> src_axis_ids;
        for (const auto &src_axis : one_trans_info.src_axis) {
          src_axis_ids.push_back(src_axis->id);
        }
        tensor_view_to_update =
            AxisUtils::MergeView(tensor_view_to_update, one_trans_info.dst_axis.back()->id, src_axis_ids);

        break;
      }
      case TransType::kValid:
        break;
      default:
        GELOGW("Unsupported trans type %ld", one_trans_info.trans_type);
        return {false, tensor_view_to_update};
    }
  }
  GELOGD("Update view to [%s].", ToString(tensor_view_to_update.axis_ids).c_str());
  return {true, tensor_view_to_update};
}

std::pair<bool, View> ApplyReadyTransInfos(const std::vector<int64_t> &my_api_sched_axes, View &tensor_view_to_update,
                                           std::vector<OneTransInfo> &not_ready_trans, const bool revert) {
  std::vector<OneTransInfo> to_apply_trans;
  GELOGI("Before apply trans info, view is %s, my api schedule axes is %s, not ready trans size is %zu, revert is %d",
         ViewToString(tensor_view_to_update).c_str(), ToString(my_api_sched_axes).c_str(), not_ready_trans.size(),
         revert);
  std::tie(to_apply_trans, not_ready_trans) = UpdateReadyTransInfos(tensor_view_to_update, not_ready_trans);
  std::pair<bool, View> pair0{true, tensor_view_to_update};
  // break loop condition:
  // current axes can not find any transform info to apply
  while (!to_apply_trans.empty()) {
    pair0 = ApplyViewTrans(to_apply_trans, revert, tensor_view_to_update);
    if (pair0.first) {
      std::tie(tensor_view_to_update) = pair0.second;
    } else {
      return {false, tensor_view_to_update};
    }
    std::tie(to_apply_trans, not_ready_trans) = UpdateReadyTransInfos(tensor_view_to_update, not_ready_trans);
  }
  // 根据当前API的调度轴顺序调整输出View的轴顺序，保证越外侧的调度轴越靠前
  GELOGI("After apply trans info, view is %s", ViewToString(tensor_view_to_update).c_str());
  return {true, AxisUtils::ReorderView(tensor_view_to_update, my_api_sched_axes)};
}
}  // namespace
View AxisUtils::ReduceView(const View &src_view, int64_t reduce_axis) {
  View new_view(src_view);
  auto &axis_ids = new_view.axis_ids;
  auto &repeats = new_view.repeats;
  auto &strides = new_view.strides;
  GELOGI("Before reduce, view is %s", ViewToString(src_view).c_str());
  GE_ASSERT_EQ(axis_ids.size(), repeats.size());
  GE_ASSERT_EQ(axis_ids.size(), strides.size());
  const size_t axis_size = axis_ids.size();
  size_t reduce_index_in_axis = 0U;
  Expression repeat_size;

  for (size_t index = 0; index < axis_size; ++index) {
    if (axis_ids[index] == reduce_axis) {
      repeat_size = repeats[index];
      strides[index] = Symbol(0);
      reduce_index_in_axis = index;
      break;
    }
  }
  // reduce之前的轴的stride应该除去reduce轴的repeat
  for (size_t index = 0; index < reduce_index_in_axis; ++index) {
    strides[index] = strides[index] / repeat_size;
  }
  GELOGI("After reduce, view is %s", ViewToString(new_view).c_str());
  return new_view;
}

std::vector<int64_t> AxisUtils::GetDefaultVectorizedAxis(const std::vector<int64_t> &tensor_axis, int64_t loop_axis) {
  auto iter = std::find(tensor_axis.begin(), tensor_axis.end(), loop_axis);
  if (iter == tensor_axis.end()) {
    return tensor_axis;
  } else {
    return {std::next(iter), tensor_axis.end()};
  }
}

View AxisUtils::SplitView(const View &src_view, const Expression &split_size, const int64_t outter_id,
                          const int64_t inner_id, const int64_t original_id) {
  View new_view;
  auto &new_axes = new_view.axis_ids;
  auto &new_repeat = new_view.repeats;
  auto &new_strides = new_view.strides;
  const auto &axis_ids = src_view.axis_ids;
  const auto &repeats = src_view.repeats;
  const auto &strides = src_view.strides;

  GELOGI("Before split, view is %s", ViewToString(src_view).c_str());
  GE_ASSERT_EQ(axis_ids.size(), repeats.size());
  GE_ASSERT_EQ(axis_ids.size(), strides.size());
  for (uint32_t axis_index = 0U; axis_index < axis_ids.size(); axis_index++) {
    if (axis_ids[axis_index] != original_id) {
      new_axes.push_back(axis_ids[axis_index]);
      new_repeat.push_back(repeats[axis_index]);
      new_strides.push_back(strides[axis_index]);
    } else {
      new_axes.push_back(outter_id);
      new_axes.push_back(inner_id);
      if (repeats[axis_index] == 1) {
        // keep stride when repeat=1
        new_repeat.push_back(sym::kSymbolOne);
        new_strides.push_back(sym::kSymbolZero);
        new_repeat.push_back(sym::kSymbolOne);
        new_strides.push_back(strides[axis_index]);
      } else {
        new_repeat.push_back(repeats[axis_index] / split_size);
        new_strides.push_back(strides[axis_index] * split_size);
        new_repeat.push_back(split_size);
        new_strides.push_back(strides[axis_index]);
      }
    }
  }

  GELOGI("After split, view is %s", ViewToString(new_view).c_str());
  return new_view;
}

View AxisUtils::MergeView(const View &src_view, const int64_t merged_axis_id, const std::vector<int64_t> &original) {
  View new_view;
  std::set<int64_t> original_set(original.begin(), original.end());
  std::set<int64_t> merge_axis_set;
  auto &new_axis_ids = new_view.axis_ids;
  auto &new_repeat = new_view.repeats;
  auto &new_strides = new_view.strides;
  const auto &axis_ids = src_view.axis_ids;
  const auto &repeats = src_view.repeats;
  const auto &strides = src_view.strides;
  GELOGI("Before merge, view is %s", ViewToString(src_view).c_str());
  GE_ASSERT_EQ(axis_ids.size(), repeats.size());
  GE_ASSERT_EQ(axis_ids.size(), strides.size());
  Expression merge_repeat = sym::kSymbolOne;
  // repeats==1 的退化轴（如零步长广播轴）索引恒 0，不贡献物理位置，merged stride 取
  // 组内最后一个非退化轴的 stride：在组内连续（退化轴跳过后相邻有效轴满足
  // stride_i == repeat_{i+1} * stride_{i+1}）时 pos == i_merge * stride_last_effective，
  // 与取原"组内最后轴"在末轴非退化时一致；末轴退化时原实现取 0 会破坏物理位置。
  // 组内全退化时索引恒 0，任意 stride 的位置贡献均为 0，回退取最后轴 stride。
  Expression effective_stride;
  bool has_effective_stride = false;
  for (uint32_t axis_index = 0U; axis_index < axis_ids.size(); axis_index++) {
    if (original_set.find(axis_ids[axis_index]) != original_set.end()) {
      merge_repeat = merge_repeat * repeats[axis_index];
      merge_axis_set.emplace(axis_ids[axis_index]);
      if (SymbolicUtils::StaticCheckEq(repeats[axis_index], sym::kSymbolOne) != TriBool::kTrue) {
        effective_stride = strides[axis_index];
        has_effective_stride = true;
      }
      if (merge_axis_set.size() == original_set.size()) {
        new_axis_ids.push_back(merged_axis_id);
        new_repeat.push_back(merge_repeat);
        if (has_effective_stride) {
          new_strides.push_back(effective_stride);
        } else {
          new_strides.push_back(strides[axis_index]);
        }
      }
    } else {
      new_axis_ids.push_back(axis_ids[axis_index]);
      new_repeat.push_back(repeats[axis_index]);
      new_strides.push_back(strides[axis_index]);
    }
  }
  // 不支持merge不全的情况
  GE_ASSERT_TRUE(merge_axis_set.size() == original_set.size() || merge_axis_set.empty(),
                 "tensor has view %s but origin is %s", ViewToString(src_view).c_str(),
                 ViewMemberToString(original).c_str());
  GELOGI("After merge, view is %s", ViewToString(new_view).c_str());
  return new_view;
}

std::pair<bool, View> AxisUtils::UpdateViewIfCrossLoop(const TransInfoRoadOfGraph &trans_info_road_of_graph,
                                                       const vector<int64_t> &input_api_sched_axes,
                                                       const vector<int64_t> &my_api_sched_axes,
                                                       View &&tensor_view_to_update) {
  // 计算当前API与输入View对应API的差异化轴
  if (my_api_sched_axes == input_api_sched_axes) {
    return {false, tensor_view_to_update};
  }
  // 对输入View应用差异化轴的差异化变换，得到输出View
  // step1 计算调度轴的映射，查找差异化轴的变换信息
  const auto trans_info_map = ToTransInfoMap(trans_info_road_of_graph);
  const auto diff_axes_info = GetDiffAxesInfo(input_api_sched_axes, my_api_sched_axes);
  // step2 应用差异化轴的逆变换
  auto revert_not_ready_trans = ToRevertTransInfo(GetTransInfos(trans_info_map, diff_axes_info.del_axes));
  if (!revert_not_ready_trans.empty()) {
    ApplyReadyTransInfos(my_api_sched_axes, tensor_view_to_update, revert_not_ready_trans, true);
  }
  // step3 应用差异化轴的变换
  auto not_ready_trans = GetTransInfos(trans_info_map, diff_axes_info.add_axes);
  return ApplyReadyTransInfos(my_api_sched_axes, tensor_view_to_update, not_ready_trans, false);
}

View AxisUtils::ReorderView(const View &src_view, const std::vector<int64_t> &my_api_sched_axes) {
  GELOGI("Before reorder, view is %s, my api sched axes is %s", ViewToString(src_view).c_str(),
         ToString(my_api_sched_axes).c_str());
  const auto &src_axes = src_view.axis_ids;
  const auto &repeats = src_view.repeats;
  const auto &strides = src_view.strides;
  GE_ASSERT_EQ(src_axes.size(), repeats.size());
  GE_ASSERT_EQ(src_axes.size(), strides.size());
  using ReorderingView = std::pair<AxisId, std::pair<const Expression *, const Expression *>>;
  std::vector<ReorderingView> reordering_view;
  reordering_view.reserve(src_axes.size());
  for (size_t id = 0UL; id < src_axes.size(); ++id) {
    reordering_view.emplace_back(std::make_pair(src_axes[id], std::make_pair(&repeats[id], &strides[id])));
  }
  std::sort(reordering_view.begin(), reordering_view.end(),
            [&my_api_sched_axes](const ReorderingView &left, const ReorderingView &right) -> bool {
              const auto left_iter = std::find(my_api_sched_axes.cbegin(), my_api_sched_axes.cend(), left.first);
              const auto right_iter = std::find(my_api_sched_axes.cbegin(), my_api_sched_axes.cend(), right.first);
              // the condition for reorder
              if ((left_iter == my_api_sched_axes.cend()) && (right_iter == my_api_sched_axes.cend())) {
                return false;
              }
              if ((left_iter == my_api_sched_axes.cend()) && (right_iter != my_api_sched_axes.cend())) {
                return false;
              }
              if ((left_iter != my_api_sched_axes.cend()) && (right_iter == my_api_sched_axes.cend())) {
                // for example, my schedule = [a,b], axes = [c,a,d], a is in my schedule, should reorder [c,a] in axes
                return true;
              }
              // for example, my schedule = [a,b], axes = [b,a,d], in my schedule b > a, should reorder [b,a] in axes
              return left_iter < right_iter;
            });
  std::vector<int64_t> ordered_axes;
  std::vector<Expression> ordered_repeats;
  std::vector<Expression> ordered_strides;
  for (const auto &reordering_axis : reordering_view) {
    ordered_axes.emplace_back(reordering_axis.first);
    ordered_repeats.emplace_back(*reordering_axis.second.first);
    ordered_strides.emplace_back(*reordering_axis.second.second);
  }
  View output_view{ordered_axes, ordered_repeats, ordered_strides};
  GELOGI("After reorder, view is %s", ViewToString(output_view).c_str());
  return output_view;
}
}  // namespace af
