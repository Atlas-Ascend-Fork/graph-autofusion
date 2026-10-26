/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_code_generator.h"
#include <algorithm>
#include <unordered_set>
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "base/att_const_values.h"
#include "tiling_data_gen/tiling_data_generator.h"

namespace att {
namespace {
std::string EnsureTrailingSlash(const std::string &path) {
  return path.back() == '/' ? path : path + "/";
}
bool IsUniqueGroups(const TilingModelInfo &all_model_infos) {
  std::unordered_set<size_t> asc_graphs;
  std::unordered_set<size_t> groups_ids;
  std::unordered_set<size_t> impl_graphs_ids;
  for (const auto &model_info : all_model_infos) {
    asc_graphs.insert(model_info.schedule_group_ident.asc_graph_id);
    groups_ids.insert(model_info.schedule_group_ident.group_id);
    impl_graphs_ids.insert(model_info.schedule_group_ident.impl_graph_id);
  }
  return (asc_graphs.size() == 1UL) && (groups_ids.size() == 1UL) && (impl_graphs_ids.size() == 1UL);
}

std::map<std::string, int32_t> BuildFinalTilingSubCaseIds(const TilingModelInfo &model_infos) {
  std::set<std::string> sub_case_tags;
  for (const auto &model_info : model_infos) {
    sub_case_tags.insert(model_info.sub_case_tag);
  }
  std::map<std::string, int32_t> sub_case_ids;
  int32_t sub_case_id = 0;
  for (const auto &sub_case_tag : sub_case_tags) {
    sub_case_ids.emplace(sub_case_tag, sub_case_id++);
  }
  return sub_case_ids;
}

std::map<std::string, int32_t> BuildFinalTilingSubCaseIds(const FusedParsedScheduleResult &fused_schedule_result) {
  TilingModelInfo model_infos;
  for (const auto &asc_graph : fused_schedule_result) {
    for (const auto &schedule_result : asc_graph.second) {
      for (const auto &group_graphs : schedule_result.second.groups_tiling_model_info) {
        model_infos.insert(model_infos.end(), group_graphs.second.begin(), group_graphs.second.end());
      }
    }
  }
  return BuildFinalTilingSubCaseIds(model_infos);
}

std::string GetSplitHeaderFileName(const std::string &key) {
  if (key == kTilingStateHeaderIdentify) {
    return kTilingStateHeaderFileName;
  }
  if (key == kTilingLogHeaderIdentify) {
    return kTilingLogHeaderFileName;
  }
  if (key == kTilingPgoHeaderIdentify) {
    return kTilingPgoHeaderFileName;
  }
  if (key == kTilingApiHeaderIdentify) {
    return kTilingApiHeaderFileName;
  }
  if (key == kTilingSolverHeaderIdentify) {
    return kTilingSolverHeaderFileName;
  }
  return "";
}
}  // namespace

af::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                                              const TilingCodeGenConfig &config) {
  std::map<std::string, std::string> tiling_res;
  GE_ASSERT_SUCCESS(GenTilingCode(op_type, model_infos, config, tiling_res), "Gen tiling impl code failed.");
  ge::CodePrinter tiling_dumper;
  if (config.gen_tiling_data) {
    GE_ASSERT_TRUE(tiling_res.find(config.tiling_data_type_name) != tiling_res.end(),
                   "Generate tiling data [%s] failed.", config.tiling_data_type_name.c_str());
    tiling_dumper.AddLine(tiling_res.at(config.tiling_data_type_name));
    if (!config.path.empty()) {
      tiling_dumper.SaveToFile(EnsureTrailingSlash(config.path) + op_type + "_" + kDefaultTilingDataFileName);
    }
  }
  if (!config.path.empty()) {
    for (const auto &[key, value] : tiling_res) {
      tiling_dumper.Reset();
      if (key == kTilingHeadIdentify) {
        tiling_dumper.AddLine(value);
        tiling_dumper.SaveToFile(kDefaultTilingHeadFileName);
      } else if (!GetSplitHeaderFileName(key).empty()) {
        tiling_dumper.AddLine(value);
        tiling_dumper.SaveToFile(EnsureTrailingSlash(config.path) + GetSplitHeaderFileName(key));
      } else if ((key == config.tiling_data_type_name) || (key.find(kDefaultTilingDataTypeName) != std::string::npos)) {
        // doning nothing,在上面做过处理了
      } else {
        tiling_dumper.AddLine(value);
        tiling_dumper.SaveToFile(EnsureTrailingSlash(config.path) + op_type + "_" + key + "_" +
                                 kDefaultTilingFuncFileName);
      }
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type, const TilingModelInfo &model_infos,
                                              const TilingCodeGenConfig &config,
                                              std::map<std::string, std::string> &tiling_res) {
  GELOGI("[DFX] Start to gen tiling code, config[%s].", config.Debug().c_str());
  TilingCodeGenImplPtr impl = CreateTilingCodeGenImpl(op_type, config, model_infos, {}, true);
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(config.type));
  impl->SetFinalTilingObservationMetadata(0UL, 1UL, BuildFinalTilingSubCaseIds(model_infos));
  GE_ASSERT_SUCCESS(impl->GenTilingHead(tiling_res), "Gen tiling head impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTiling(tiling_res), "Gen tiling code impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingTail(tiling_res), "Gen tiling tail impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  GE_ASSERT_TRUE(tiling_res.find(kTilingHeadIdentify) != tiling_res.cend(), "Generate tiling func failed.");
  GE_ASSERT_SUCCESS(impl->FinishGeneratedHeaders(tiling_res), "Finish generated tiling headers failed.");
  return af::SUCCESS;
}

TilingCodeGenImplPtr TilingCodeGenerator::CreateTilingCodeGenImpl(const std::string &op_name,
                                                                  const TilingCodeGenConfig &config,
                                                                  const TilingModelInfo &model_infos,
                                                                  const ScoreFuncs &score_funcs,
                                                                  const bool is_uniq_group) {
  TilingCodeGenImplPtr impl;
  if (config.type == TilingImplType::HIGH_PERF) {
    impl = std::shared_ptr<HighPerfTilingCodeGenImpl>(
        af::MakeShared<HighPerfTilingCodeGenImpl>(op_name, config, model_infos, score_funcs, is_uniq_group));
  } else if (config.type == TilingImplType::AXES_REORDER) {
    impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(
        af::MakeShared<AxesReorderTilingCodeGenImpl>(op_name, config, model_infos, score_funcs, is_uniq_group));
  }
  return impl;
}

inline std::unordered_map<std::string, std::string> GetCacheReuseInfo(
    const FusedParsedScheduleResult &fused_parsed_schedule_result) {
  std::unordered_map<std::string, std::string> cache_reuse_info;
  for (const auto &asc_graph_groups : fused_parsed_schedule_result) {
    for (const auto &schedule_results_groups : asc_graph_groups.second) {
      for (const auto &group_graphs : schedule_results_groups.second.groups_tiling_model_info) {
        const auto &model_infos = group_graphs.second;
        if (model_infos.empty()) {
          continue;
        }
        const auto &cur_ident = model_infos[0].schedule_group_ident;
        const auto &cur_prefix = cur_ident.GetGroupPrefix();
        const auto &reuse_schedule_group = model_infos[0].reuse_schedule_group;
        if (reuse_schedule_group && reuse_schedule_group->IsReuseGroup(cur_ident)) {
          const auto &reuse_ident = reuse_schedule_group->reuse_group_ident;
          const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
          cache_reuse_info[cur_prefix] = reuse_prefix;
        }
      }
    }
  }
  return cache_reuse_info;
}

inline void SaveVarRelationsInfo(
    VarRelations &var_relations, size_t asc_graph_id, size_t impl_graph_id,
    const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &schedule_result_var_relations) {
  for (auto schedule_result_var_relation = schedule_result_var_relations.begin();
       schedule_result_var_relation != schedule_result_var_relations.end(); ++schedule_result_var_relation) {
    size_t dst_schedule_group_id = schedule_result_var_relation->first;
    const auto &dst_var_relations_from_src = schedule_result_var_relation->second;
    for (auto dst_var_relation_from_src = dst_var_relations_from_src.begin();
         dst_var_relation_from_src != dst_var_relations_from_src.end(); ++dst_var_relation_from_src) {
      size_t src_schedule_group_id = dst_var_relation_from_src->first;
      const auto &relations = dst_var_relation_from_src->second;
      if (!relations.empty()) {
        GELOGD("[VAR_RELATIONS] graph_id = [%u], result_id = [%u], dst_group_id = [%u], src_group_id = [%u]:",
               asc_graph_id, impl_graph_id, dst_schedule_group_id, src_schedule_group_id);
      }
      for (auto relation = relations.begin(); relation != relations.end(); ++relation) {
        GELOGD("[VAR_RELATIONS]     dst_var_name is [%s], src_var_expression_string is [%s]", relation->first.c_str(),
               af::SymbolicUtils::ToString(relation->second).c_str());
      }
    }
  }
  var_relations[asc_graph_id][impl_graph_id] = schedule_result_var_relations;
}

inline af::Status GetWorkspaceTensorId(TensorIdSet &workspace_tensor_id_set,
                                       const TilingModelInfo &groups_tiling_model_info, const size_t asc_graph_id,
                                       const size_t impl_graph_id) {
  for (const auto &model_info : groups_tiling_model_info) {
    for (const auto &pair : model_info.workspace_size_map) {
      workspace_tensor_id_set[asc_graph_id][impl_graph_id].insert(pair.first);
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenTilingCode(const std::string &op_type,
                                              const FusedParsedScheduleResult &fused_parsed_schedule_result,
                                              const TilingCodeGenConfig &config,
                                              std::map<std::string, std::string> &tiling_res) {
  TilingModelInfo all_model_infos;
  ScoreFuncs schedule_result_score_func;
  VarRelations var_relations;
  EnableGroupParallels enable_group_parallels;
  TensorIdSet workspace_tensor_id_set;
  size_t group_num = 0UL;

  GE_ASSERT_SUCCESS(
      CollectModelInfosAndMetadata(fused_parsed_schedule_result, all_model_infos, group_num, schedule_result_score_func,
                                   var_relations, enable_group_parallels, workspace_tensor_id_set),
      "Collect model infos and metadata failed.");
  GE_ASSERT_TRUE(group_num != 0UL, "group num is zero of op type = %s.", op_type.c_str());

  const bool is_uniq_group = (group_num == 1UL);
  if (is_uniq_group) {
    return GenTilingCode(op_type, all_model_infos, config, tiling_res);
  }

  generated_headers_.clear();
  GenTilingHead(op_type, all_model_infos, config, tiling_res, enable_group_parallels);
  GELOGD("Got model infos size %zu of op type = %s.", all_model_infos.size(), op_type.c_str());

  std::unordered_map<std::string, std::string> cache_reuse_info = GetCacheReuseInfo(fused_parsed_schedule_result);
  // Group cache cannot be reused when either endpoint depends on workspace.
  // Use the same filtered relations for group bodies and the schedule-group tail
  // so the latter does not emit cache declarations for groups that do not expose
  // a GroupLevelCache type.
  std::unordered_set<std::string> workspace_groups;
  for (const auto &asc_graph : fused_parsed_schedule_result) {
    for (const auto &result : asc_graph.second) {
      for (const auto &group_graphs : result.second.groups_tiling_model_info) {
        if (std::any_of(group_graphs.second.cbegin(), group_graphs.second.cend(),
                        [](const auto &model_info) { return !model_info.workspace_size_map.empty(); })) {
          workspace_groups.insert(group_graphs.second[0].schedule_group_ident.GetGroupPrefix());
        }
      }
    }
  }
  for (auto iter = cache_reuse_info.begin(); iter != cache_reuse_info.end();) {
    if (workspace_groups.count(iter->first) != 0U || workspace_groups.count(iter->second) != 0U) {
      iter = cache_reuse_info.erase(iter);
    } else {
      ++iter;
    }
  }
  uint32_t cache_capacity = static_cast<uint32_t>(all_model_infos.size()) * 2;
  GE_ASSERT_SUCCESS(GenScheduleGroupTilingBodies(op_type, fused_parsed_schedule_result, config, cache_reuse_info,
                                                 cache_capacity, enable_group_parallels, tiling_res),
                    "Generate schedule group tiling bodies failed.");

  GenTilingParams params = {op_type, all_model_infos, config, cache_reuse_info};
  GenTilingTailExtParams ext_params = {schedule_result_score_func, var_relations, enable_group_parallels,
                                       workspace_tensor_id_set, group_num};
  GenTilingTail(params, tiling_res, ext_params);
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::FinishGeneratedHeaders(generated_headers_, config.tiling_data_type_name,
                                                              config.is_autofuse, tiling_res),
                    "Finish generated tiling headers failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenTilingHead(const std::string &op_type, const TilingModelInfo &all_model_infos,
                                              const TilingCodeGenConfig &config,
                                              std::map<std::string, std::string> &tiling_res,
                                              [[maybe_unused]] const EnableGroupParallels &enable_group_parallels) {
  GELOGI("Start to gen tiling head.");
  TilingCodeGenImplPtr impl =
      CreateTilingCodeGenImpl(op_type, config, all_model_infos, {}, IsUniqueGroups(all_model_infos));
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(config.type));
  GE_ASSERT_SUCCESS(impl->GenTilingHead(tiling_res, enable_group_parallels), "Gen tiling head impl failed, type[%d].",
                    static_cast<int32_t>(config.type));
  MergeGeneratedHeaders(*impl);
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenTilingBody(const GenTilingParams &params,
                                              std::map<std::string, std::string> &tiling_res, const bool is_uniq_group,
                                              uint32_t cache_capacity,
                                              [[maybe_unused]] const EnableGroupParallels &enable_group_parallels) {
  GELOGI("Start to gen tiling body.");
  TilingCodeGenImplPtr impl =
      CreateTilingCodeGenImpl(params.op_type, params.config, params.all_model_infos, {}, is_uniq_group);
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(params.config.type));

  GE_ASSERT_SUCCESS(impl->GenTiling(tiling_res, params.cache_reuse_info, cache_capacity, enable_group_parallels),
                    "Gen tiling body impl failed, type[%d].", static_cast<int32_t>(params.config.type));
  MergeGeneratedHeaders(*impl);
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenTilingTail(const GenTilingParams &params,
                                              std::map<std::string, std::string> &tiling_res,
                                              const GenTilingTailExtParams &ext_params) {
  GELOGI("Start to gen tiling tail for %s.", params.op_type.c_str());
  TilingCodeGenImplPtr impl = CreateTilingCodeGenImpl(params.op_type, params.config, params.all_model_infos,
                                                      ext_params.score_funcs, IsUniqueGroups(params.all_model_infos));
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", static_cast<int32_t>(params.config.type));
  impl->SetFinalTilingObservationMetadata(0UL, ext_params.final_tiling_observation_slot_count,
                                          BuildFinalTilingSubCaseIds(params.all_model_infos));
  GenTilingTailImplExtParams impl_ext_params{std::move(params.cache_reuse_info), std::move(ext_params.var_relations),
                                             std::move(ext_params.enable_group_parallels),
                                             std::move(ext_params.workspace_tensor_id_set)};
  GE_ASSERT_SUCCESS(impl->GenTilingTail(tiling_res, impl_ext_params), "Gen tiling tail impl failed, type[%d].",
                    static_cast<int32_t>(params.config.type));
  MergeGeneratedHeaders(*impl);
  return af::SUCCESS;
}

void TilingCodeGenerator::MergeGeneratedHeaders(const TilingCodeGenImpl &impl) {
  for (const auto &[header_id, generated_code] : impl.GetGeneratedHeaders()) {
    autofuse::AppendGeneratedCode(generated_headers_[header_id], generated_code);
  }
}

af::Status TilingCodeGenerator::CollectModelInfosAndMetadata(
    const FusedParsedScheduleResult &fused_parsed_schedule_result, TilingModelInfo &all_model_infos, size_t &group_num,
    ScoreFuncs &schedule_result_score_func, VarRelations &var_relations, EnableGroupParallels &enable_group_parallels,
    TensorIdSet &workspace_tensor_id_set) {
  group_num = 0UL;
  for (const auto &asc_graph_models : fused_parsed_schedule_result) {
    for (const auto &impl_graph_groups : asc_graph_models.second) {
      const auto &parsed_result = impl_graph_groups.second;
      size_t asc_graph_id = parsed_result.asc_graph_id;
      size_t impl_graph_id = parsed_result.impl_graph_id;
      for (const auto &sub_graphs : impl_graph_groups.second.groups_tiling_model_info) {
        group_num++;
        all_model_infos.insert(all_model_infos.end(), sub_graphs.second.begin(), sub_graphs.second.end());
        GE_ASSERT_SUCCESS(
            GetWorkspaceTensorId(workspace_tensor_id_set, sub_graphs.second, asc_graph_id, impl_graph_id));
      }
      schedule_result_score_func[kModelInfoLevel::K_SCHEDULE_RESULT_LEVEL][asc_graph_models.first]
                                [impl_graph_groups.second.impl_graph_id] = impl_graph_groups.second.score_func;
      SaveVarRelationsInfo(var_relations, impl_graph_groups.second.asc_graph_id, impl_graph_groups.second.impl_graph_id,
                           impl_graph_groups.second.var_relations);
      enable_group_parallels[asc_graph_models.first][impl_graph_groups.second.impl_graph_id] =
          impl_graph_groups.second.enable_group_parallel;
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenScheduleGroupTilingBodies(
    const std::string &op_type, const FusedParsedScheduleResult &fused_parsed_schedule_result,
    const TilingCodeGenConfig &config, const std::unordered_map<std::string, std::string> &cache_reuse_info,
    uint32_t cache_capacity, const EnableGroupParallels &enable_group_parallels,
    std::map<std::string, std::string> &tiling_res) {
  GELOGD("[DFX] schedule_results count: %zu, op_type[%s]", fused_parsed_schedule_result.size(), op_type.c_str());
  const std::map<std::string, int32_t> final_tiling_sub_case_ids =
      BuildFinalTilingSubCaseIds(fused_parsed_schedule_result);
  size_t final_tiling_observation_slot = 0UL;
  for (auto &asc_graph : fused_parsed_schedule_result) {
    GELOGD("[DFX] asc_graph_id: %zu, results: %zu, op_type[%s]", asc_graph.first, asc_graph.second.size(),
           op_type.c_str());
    for (auto &result : asc_graph.second) {
      GELOGD("[DFX] got result(impl_graph_id): %zu, op_type[%s]", result.first, op_type.c_str());
      GE_ASSERT_SUCCESS(
          GenScheduleResult(op_type, config, cache_reuse_info, cache_capacity, enable_group_parallels, result.second,
                            final_tiling_sub_case_ids, final_tiling_observation_slot, tiling_res),
          "Generate schedule result failed.");
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenScheduleGroupTilingGroup(
    const std::string &op_type, const TilingCodeGenConfig &config,
    const std::unordered_map<std::string, std::string> &cache_reuse_info, uint32_t cache_capacity,
    const EnableGroupParallels &enable_group_parallels, size_t group_num,
    const std::pair<const size_t, TilingModelInfo> &group_graphs,
    const std::map<std::string, int32_t> &final_tiling_sub_case_ids, size_t final_tiling_observation_slot,
    std::map<std::string, std::string> &tiling_res) {
  TilingCodeGenConfig cur_config = config;
  const auto &model_infos = group_graphs.second;
  const auto group_prefix = model_infos[0].schedule_group_ident.GetGroupPrefix();
  cur_config.tiling_data_type_name = group_prefix + kDefaultTilingDataTypeName;
  std::unordered_map<std::string, std::string> group_cache_reuse_info;
  for (const auto &[source_prefix, reuse_prefix] : cache_reuse_info) {
    if (group_prefix == source_prefix || group_prefix == reuse_prefix) {
      group_cache_reuse_info.emplace(source_prefix, reuse_prefix);
    }
  }
  GenTilingParams params = {op_type, model_infos, cur_config, std::move(group_cache_reuse_info)};
  auto impl = CreateTilingCodeGenImpl(params.op_type, params.config, params.all_model_infos, {}, false);
  GE_ASSERT_NOTNULL(impl, "Create tiling code gen impl failed, type[%d].", params.config.type);
  std::map<std::pair<size_t, size_t>, size_t> group_nums;
  group_nums[{model_infos[0].schedule_group_ident.asc_graph_id, model_infos[0].schedule_group_ident.impl_graph_id}] =
      group_num;
  impl->SetScheduleResultGroupNums(group_nums);
  impl->SetFinalTilingObservationMetadata(final_tiling_observation_slot, group_num, final_tiling_sub_case_ids);
  GE_ASSERT_SUCCESS(impl->GenTiling(tiling_res, params.cache_reuse_info, cache_capacity, enable_group_parallels),
                    "Gen tiling body impl failed, type[%d].", params.config.type);
  MergeGeneratedHeaders(*impl);
  tiling_res[config.tiling_data_type_name] += tiling_res[cur_config.tiling_data_type_name];
  return af::SUCCESS;
}

af::Status TilingCodeGenerator::GenScheduleResult(const std::string &op_type, const TilingCodeGenConfig &config,
                                                  const std::unordered_map<std::string, std::string> &cache_reuse_info,
                                                  uint32_t cache_capacity,
                                                  const EnableGroupParallels &enable_group_parallels,
                                                  const ParsedScheduleResult &result,
                                                  const std::map<std::string, int32_t> &final_tiling_sub_case_ids,
                                                  size_t &final_tiling_observation_slot,
                                                  std::map<std::string, std::string> &tiling_res) {
  const size_t group_num = result.groups_tiling_model_info.size();
  for (const auto &group_graphs : result.groups_tiling_model_info) {
    GE_ASSERT_SUCCESS(GenScheduleGroupTilingGroup(
                          op_type, config, cache_reuse_info, cache_capacity, enable_group_parallels, group_num,
                          group_graphs, final_tiling_sub_case_ids, final_tiling_observation_slot++, tiling_res),
                      "Generate schedule group failed.");
  }
  return af::SUCCESS;
}

}  // namespace att
