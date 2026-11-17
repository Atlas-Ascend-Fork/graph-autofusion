/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reuse_group_utils.h"
#include <vector>
#include <memory>
#include <map>
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "equivalent_graph_recognizer.h"
#include "generator/preprocess/args_manager.h"
namespace att {
namespace {
af::Status GetGroupAscGraphsByIdent(
    const ScheduleGroupIdent &group_ident,
    const std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> &all_graphs_lists,
    std::vector<af::AscGraph> &group_asc_graphs) {
  const auto &reuse_asc_graph_id = group_ident.asc_graph_id;
  const auto &reuse_group_id = group_ident.group_id;
  const auto &reuse_result_id = group_ident.impl_graph_id;
  GE_ASSERT_TRUE(all_graphs_lists.size() > reuse_asc_graph_id,
                 "Get group asc graphs failed, asc graph id[%zu] not found, size = %zu.", reuse_asc_graph_id,
                 all_graphs_lists.size());
  GE_ASSERT_TRUE(all_graphs_lists[reuse_asc_graph_id].size() > reuse_result_id,
                 "Get group asc graphs failed, asc graph id[%zu], result id[%zu] not found, size = %zu.",
                 reuse_asc_graph_id, reuse_result_id, all_graphs_lists[reuse_asc_graph_id].size());
  GE_ASSERT_TRUE(
      all_graphs_lists[reuse_asc_graph_id][reuse_result_id].size() > reuse_group_id,
      "Get group asc graphs failed, asc graph id[%zu], group id[%zu] not found, reuse_result_id = %zu size = %zu.",
      reuse_asc_graph_id, reuse_group_id, reuse_result_id,
      all_graphs_lists[reuse_asc_graph_id][reuse_result_id].size());
  group_asc_graphs = all_graphs_lists[reuse_asc_graph_id][reuse_result_id][reuse_group_id];
  return af::SUCCESS;
}

TilingModelInfo *GetGroupModelInfosByIdent(FusedParsedScheduleResult &all_model_infos,
                                           ScheduleGroupIdent &group_ident) {
  const auto &asc_graph_iter = all_model_infos.find(group_ident.asc_graph_id);
  GE_ASSERT_TRUE(asc_graph_iter != all_model_infos.cend(), "Get model info failed, result id[%zu] not found.",
                 group_ident.impl_graph_id);
  const auto &result_iter = asc_graph_iter->second.find(group_ident.impl_graph_id);
  GE_ASSERT_TRUE(result_iter != asc_graph_iter->second.cend(), "Get model info failed, result id[%zu] not found.",
                 group_ident.impl_graph_id);
  const auto &groups_iter = result_iter->second.groups_tiling_model_info.find(group_ident.group_id);
  GE_ASSERT_TRUE(groups_iter != result_iter->second.groups_tiling_model_info.cend(),
                 "Get model info failed, group id[%zu] not found.", group_ident.group_id);
  return &groups_iter->second;
}

af::Status MergeScheduleReuseGroups(ReuseScheduleGroupPtr &merge_from, ReuseScheduleGroupPtr &merge_to) {
  GE_ASSERT_NOTNULL(merge_from, "merge_from is nullptr.");
  GE_ASSERT_NOTNULL(merge_to, "merge_to is nullptr.");
  merge_to->schedule_group_to_info[merge_from->reuse_group_ident].reuse_input_axes = merge_from->info.reuse_input_axes;
  merge_to->schedule_group_to_info[merge_from->reuse_group_ident].reuse_search_axes =
      merge_from->info.reuse_search_axes;
  merge_to->schedule_group_to_info[merge_from->reuse_group_ident].tiling_keys = merge_from->info.tiling_keys;
  merge_from = merge_to;
  return af::SUCCESS;
}

af::Status MergeScheduleReuseGroups(TilingModelInfo *merge_from, TilingModelInfo *merge_to) {
  GE_ASSERT_NOTNULL(merge_from, "merge_from is nullptr.");
  GE_ASSERT_NOTNULL(merge_to, "merge_to is nullptr.");
  GE_ASSERT_TRUE(merge_from->size() == merge_to->size(),
                 "MergeScheduleReuseGroups failed, merge_from size[%zu], merge_to size[%zu]", merge_from->size(),
                 merge_to->size());
  for (size_t i = 0UL; i < merge_from->size(); i++) {
    GELOGD("Merge schedule groups from: %s to: %s", merge_from->at(i).schedule_group_ident.GetGroupPrefix().c_str(),
           merge_to->at(i).schedule_group_ident.GetGroupPrefix().c_str());
    GE_ASSERT_SUCCESS(
        MergeScheduleReuseGroups(merge_from->at(i).reuse_schedule_group, merge_to->at(i).reuse_schedule_group));
  }
  return af::SUCCESS;
}

bool IsGroupEqualSize(const ReuseScheduleGroupInfo &group_info1, const ReuseScheduleGroupInfo &group_info2) {
  return (group_info1.reuse_input_axes.size() == group_info2.reuse_input_axes.size()) &&
         (group_info1.reuse_search_axes.size() == group_info2.reuse_search_axes.size()) &&
         (group_info1.tiling_keys.size() == group_info2.tiling_keys.size());
}

bool IsReuseAxesSame(const ReuseScheduleGroupPtr &group1, const ReuseScheduleGroupPtr &group2) {
  bool is_equivalent = true;
  for (size_t input_id = 0UL; input_id < group1->info.reuse_input_axes.size(); input_id++) {
    if (group1->info.reuse_input_axes[input_id] != group2->info.reuse_input_axes[input_id]) {
      GELOGD("Cannot merge group: %s and %s as they have different reuse input axes[%s vs %s]",
             group1->reuse_group_ident.GetGroupPrefix().c_str(), group2->reuse_group_ident.GetGroupPrefix().c_str(),
             group1->info.reuse_input_axes[input_id].c_str(), group2->info.reuse_input_axes[input_id].c_str());
      is_equivalent = false;
      break;
    }
  }
  return is_equivalent;
}
}  // namespace
// 当前仅考虑graph排序一致的场景
bool ReuseGroupUtils::IsGroupGraphsEquivalent(const std::vector<af::AscGraph> &graphs_to,
                                              const std::vector<af::AscGraph> &graphs_from,
                                              ReuseScheduleGroupInfo &group_info_to,
                                              ReuseScheduleGroupInfo &group_info_from) {
  if (graphs_to.size() != graphs_from.size()) {
    GELOGD("AscGraphs are not equivalent, graphs size[%zu vs %zu]", graphs_to.size(), graphs_from.size());
    return false;
  }
  for (size_t i = 0UL; i < graphs_to.size(); ++i) {
    EquivalentGraphRecognizer equivalent_graph_recognizer(graphs_to[i], graphs_from[i], group_info_to, group_info_from);
    if (!equivalent_graph_recognizer.IsEquivalent()) {
      GELOGD("AscGraphs are not equivalent, graph1[%s], graph2[%s], index[%zu]", graphs_to[i].GetName().c_str(),
             graphs_from[i].GetName().c_str(), i);
      return false;
    }
    if (equivalent_graph_recognizer.GetMappedInputAxesNames() != group_info_to.reuse_input_axes) {
      GELOGD(
          "AscGraphs are not equivalent, graph1[%s], graph2[%s], index[%zu], "
          "mapped_input_axes[%s], reuse_input_axes[%s]",
          graphs_to[i].GetName().c_str(), graphs_from[i].GetName().c_str(), i,
          af::ToString(equivalent_graph_recognizer.GetMappedInputAxesNames()).c_str(),
          af::ToString(group_info_to.reuse_input_axes).c_str());
      return false;
    }
  }
  return true;
}

af::Status ReuseGroupUtils::InitReuseScheduleGroup(const ScheduleGroupIdent &group_ident,
                                                   TilingModelInfo &group_tiling_model_info) {
  auto reuse_schedule_groups = std::make_shared<ReuseScheduleGroup>();
  GE_ASSERT_NOTNULL(reuse_schedule_groups);
  reuse_schedule_groups->reuse_group_ident = group_ident;
  std::set<std::string> input_var_names;
  std::vector<std::string> ordered_var_names;
  // 记录输入轴的顺序，保证输入轴的顺序一致
  for (const auto &model_info : group_tiling_model_info) {
    att::ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    const auto &inputs = args_manager.GetInputVars();
    for (const auto &input : inputs) {
      if (input_var_names.insert(input.Str().get()).second) {
        ordered_var_names.emplace_back(input.Str().get());
      }
    }
  }
  // 按照输入轴的顺序排序
  std::sort(ordered_var_names.begin(), ordered_var_names.end());
  for (auto &ordered_var_name : ordered_var_names) {
    GELOGD("Add input reuse axis: %s for group[%s]", ordered_var_name.c_str(), group_ident.GetItemPrefix().c_str());
    reuse_schedule_groups->info.reuse_input_axes.emplace_back(ordered_var_name);
  }
  for (auto &model_info : group_tiling_model_info) {
    for (const auto &arg : model_info.arg_list) {
      input_var_names.find(arg->size->symbol_expr.Str().get());
      if (arg->axis_pos == AxisPosition::INNER) {
        const auto exp = arg->size->symbol_expr.Str();
        GE_ASSERT_NOTNULL(exp.get());
        reuse_schedule_groups->info.reuse_search_axes.emplace_back(exp.get());
        GELOGD("Add search reuse axis: %s for group[%s] axis_pos[%d]", exp.get(), group_ident.GetItemPrefix().c_str(),
               arg->axis_pos);
      }
    }
    reuse_schedule_groups->info.tiling_keys.emplace_back(model_info.tiling_case_id);
    model_info.reuse_schedule_group = reuse_schedule_groups;
  }
  return af::SUCCESS;
}

af::Status ReuseGroupUtils::MergeEqualReusableGroups(
    const ReuseScheduleGroupPtr &group_to, const ReuseScheduleGroupPtr &group_from,
    const std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> &all_graphs_lists,
    FusedParsedScheduleResult &out_fused_schedule_result) {
  std::vector<af::AscGraph> got_asc_graphs_to;
  GE_ASSERT_SUCCESS(GetGroupAscGraphsByIdent(group_to->reuse_group_ident, all_graphs_lists, got_asc_graphs_to));
  std::vector<af::AscGraph> got_asc_graphs_from;
  GE_ASSERT_SUCCESS(GetGroupAscGraphsByIdent(group_from->reuse_group_ident, all_graphs_lists, got_asc_graphs_from));
  // 检查Group内的AscGraph是否均等价，若均等价， 可以合并所有可以复用的group
  if (IsGroupGraphsEquivalent(got_asc_graphs_to, got_asc_graphs_from, group_to->info, group_from->info)) {
    GE_ASSERT_TRUE(!got_asc_graphs_to.empty());
    GELOGI("AscGraphs are equivalent, graph size[%zu], ident is [%s vs %s]", got_asc_graphs_from.size(),
           group_to->reuse_group_ident.GetGroupPrefix().c_str(),
           group_from->reuse_group_ident.GetGroupPrefix().c_str());
    auto model_group_from = GetGroupModelInfosByIdent(out_fused_schedule_result, group_from->reuse_group_ident);
    auto model_group_to = GetGroupModelInfosByIdent(out_fused_schedule_result, group_to->reuse_group_ident);
    GE_ASSERT_NOTNULL(model_group_from);
    GE_ASSERT_NOTNULL(model_group_to);
    GE_ASSERT_SUCCESS(MergeScheduleReuseGroups(model_group_from, model_group_to));
  }
  return af::SUCCESS;
}

af::Status ReuseGroupUtils::MergeAllReusableGroups(
    const std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> &all_graphs_lists,
    FusedParsedScheduleResult &out_fused_schedule_result) {
  std::vector<ReuseScheduleGroupPtr> reuse_schedule_groups;
  for (const auto &asc_graph_parsed_result : out_fused_schedule_result) {
    for (const auto &parsed_schedule_result : asc_graph_parsed_result.second) {
      for (const auto &group_info : parsed_schedule_result.second.groups_tiling_model_info) {
        if (!group_info.second.empty()) {
          reuse_schedule_groups.emplace_back(group_info.second[0].reuse_schedule_group);
          GELOGD("Insert reuse group: %s", group_info.second[0].schedule_group_ident.GetGroupPrefix().c_str());
        }
      }
    }
  }
  for (size_t i = 0UL; i < reuse_schedule_groups.size(); i++) {
    for (size_t j = i + 1UL; j < reuse_schedule_groups.size(); j++) {
      const auto &group1 = reuse_schedule_groups[i];
      const auto &group2 = reuse_schedule_groups[j];
      if (!IsGroupEqualSize(group1->info, reuse_schedule_groups[j]->info)) {
        GELOGD(
            "Cannot merge group: %s and %s as they have different reuse input axes size[%zu vs %zu]  or reuse search "
            "axes size[%zu vs %zu] or tiling keys size[%zu "
            "vs %zu]",
            group1->reuse_group_ident.GetGroupPrefix().c_str(), group2->reuse_group_ident.GetGroupPrefix().c_str(),
            group1->info.reuse_input_axes.size(), group2->info.reuse_input_axes.size(),
            group1->info.reuse_search_axes.size(), group2->info.reuse_search_axes.size(),
            group1->info.tiling_keys.size(), group2->info.tiling_keys.size());
        break;
      }
      if (IsReuseAxesSame(group1, group2)) {
        GE_ASSERT_SUCCESS(MergeEqualReusableGroups(reuse_schedule_groups[i], reuse_schedule_groups[j], all_graphs_lists,
                                                   out_fused_schedule_result));
      }
    }
  }
  return af::SUCCESS;
}
}  // namespace att
