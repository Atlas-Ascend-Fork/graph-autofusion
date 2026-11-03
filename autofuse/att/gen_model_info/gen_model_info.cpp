/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gen_model_info.h"
#include <fstream>
#include <set>
#include "common/util/mem_utils.h"
#include "common/ub_expr/asc_graph_ub_expr_builder.h"
#include "common/ub_expr/ub_expr_utils.h"
#include "common/autofuse_config/auto_fuse_config_utils.h"
#include "graph/utils/file_utils.h"
#include "expr_gen/generate_tiling_expr.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "parser/ascend_graph_parser.h"
#include "nlohmann/json.hpp"
#include "util/base_types_printer.h"
#include "util/duration.h"
#include "api_perf_register/soc_perf_config.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
#include "gen_model_info/expr_gen/arg_list_reorder.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "schedule_result.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "common_utils.h"

namespace att {
namespace {
constexpr uint32_t kPathMax = 4096U;
const std::string kFileSeparator = "/";
const std::string kModelInfoFileName = "model_info.json";
constexpr uint32_t kConstType = 1U;
constexpr uint32_t kVarType = 2U;
constexpr uint32_t kDefaultAlignValue = 1U;
constexpr uint32_t kGatherReducePenaltyCacheLineSize = 32U;
const std::string kModelInfoFilePath = "./";

bool HasComputeType(const std::vector<af::AscGraph> &graphs, const af::ComputeType compute_type) {
  for (const auto &graph : graphs) {
    for (const auto &node : graph.GetAllNodes()) {
      if (node->attr.api.compute_type == compute_type) {
        return true;
      }
    }
  }
  return false;
}

bool HasGatherNode(const std::vector<std::vector<af::AscGraph>> &schedule_groups) {
  for (const auto &group : schedule_groups) {
    for (const auto &graph : group) {
      for (const auto &node : graph.GetAllNodes()) {
        if (af::ops::IsOps<af::ascir_op::Gather>(node)) {
          return true;
        }
      }
    }
  }
  return false;
}

void FilterOutputAliasWorkspace(ModelInfo &model_info) {
  for (const auto &output_node : model_info.output_nodes) {
    if (output_node == nullptr || !af::ops::IsOps<af::ascir_op::Output>(output_node)) {
      continue;
    }
    const int64_t tensor_id = output_node->inputs[0].attr.mem.tensor_id;
    const auto workspace_iter = model_info.workspace_size_map.find(tensor_id);
    if (workspace_iter == model_info.workspace_size_map.end()) {
      continue;
    }
    GELOGD("Filter output alias tensor id [%ld] from workspace size map.", tensor_id);
    model_info.workspace_size_map.erase(workspace_iter);
  }
}

bool ShouldEnableGatherReducePenalty(const std::vector<std::vector<af::AscGraph>> &schedule_groups,
                                     const size_t group_id, const bool enable_group_parallel) {
  if (enable_group_parallel) {
    GELOGD(
        "[DFX] GatherReducePenalty decision: group_id=%zu, group_count=%zu, enable_group_parallel=1, enabled=0, "
        "reason=group_parallel",
        group_id, schedule_groups.size());
    return false;
  }
  if (schedule_groups.size() <= 1UL) {
    GELOGD(
        "[DFX] GatherReducePenalty decision: group_id=%zu, group_count=%zu, enable_group_parallel=0, enabled=0, "
        "reason=single_group",
        group_id, schedule_groups.size());
    return false;
  }
  if (!HasComputeType(schedule_groups[group_id], af::ComputeType::kComputeReduce)) {
    GELOGD(
        "[DFX] GatherReducePenalty decision: group_id=%zu, group_count=%zu, enable_group_parallel=0, enabled=0, "
        "reason=group_without_reduce",
        group_id, schedule_groups.size());
    return false;
  }
  const bool has_gather = HasGatherNode(schedule_groups);
  GELOGD(
      "[DFX] GatherReducePenalty decision: group_id=%zu, group_count=%zu, enable_group_parallel=0, enabled=%d, "
      "reason=%s",
      group_id, schedule_groups.size(), has_gather, has_gather ? "gather_reduce" : "gather_not_found");
  return has_gather;
}

std::set<std::string> GetUbContainerNames(const ModelInfo &model_info) {
  std::set<std::string> names;
  const auto ub_iter = model_info.hardware_cons.find(HardwareDef::UB);
  if (ub_iter == model_info.hardware_cons.cend()) {
    return names;
  }
  for (const auto &symbol : ub_iter->second.FreeSymbols()) {
    const auto name_iter = model_info.variable_name_map.find(symbol);
    const auto name = name_iter == model_info.variable_name_map.cend() ? Str(symbol) : name_iter->second;
    if (model_info.container_exprs.find(name) != model_info.container_exprs.cend()) {
      names.insert(name);
    }
  }
  return names;
}

void EraseVariableByName(ExprExprMap &variable_expr_map, std::map<Expr, std::string, ExprCmp> &variable_name_map,
                         const std::string &name) {
  for (auto iter = variable_name_map.begin(); iter != variable_name_map.end();) {
    if (iter->second == name || Str(iter->first) == name) {
      variable_expr_map.erase(iter->first);
      iter = variable_name_map.erase(iter);
    } else {
      ++iter;
    }
  }
  for (auto iter = variable_expr_map.begin(); iter != variable_expr_map.end();) {
    if (Str(iter->first) == name) {
      iter = variable_expr_map.erase(iter);
    } else {
      ++iter;
    }
  }
}

void EraseContainerExprs(const std::set<std::string> &container_names, ModelInfo &model_info) {
  for (const auto &name : container_names) {
    model_info.container_exprs.erase(name);
    EraseVariableByName(model_info.variable_expr_map, model_info.variable_name_map, name);
  }
}

void ApplyCommonUbExprContext(const ascir::UbExprContext &context, ModelInfo &model_info) {
  const auto old_ub_container_names = GetUbContainerNames(model_info);
  const auto ub_expr_result = ascir::UbExprUtils::BuildUbExpr(context);
  if (ub_expr_result.has_ub_expr) {
    model_info.hardware_cons[HardwareDef::UB] = ub_expr_result.ub_expr;
  } else {
    model_info.hardware_cons.erase(HardwareDef::UB);
  }
  EraseContainerExprs(old_ub_container_names, model_info);
  for (const auto &container : context.container_names) {
    const auto expr_iter = context.container_expr.find(container.first);
    if (expr_iter == context.container_expr.cend()) {
      continue;
    }
    model_info.container_exprs[container.second] = expr_iter->second;
    model_info.variable_expr_map[container.first] = expr_iter->second;
    model_info.variable_name_map[container.first] = container.second;
  }
}

af::Status RefreshCommonUbExprContext(const af::AscGraph &graph, ModelInfo &model_info, bool is_cv_ub_fusion) {
  ascir::UbExprContext context;
  GE_ASSERT_SUCCESS(ascir::AscGraphUbExprBuilder().Build(graph, context, is_cv_ub_fusion),
                    "Build common UB expr failed, graph:[%s].", graph.GetName().c_str());
  ApplyCommonUbExprContext(context, model_info);
  return af::SUCCESS;
}
}  // namespace

static af::Status GenerateSingleModelInfoWithContext(const af::AscGraph &graph, ModelInfo &model_info,
                                                     TuningSpacePtr &tuning_space, const uint32_t tiling_case_id,
                                                     bool is_cv_ub_fusion) {
  GE_ASSERT_SUCCESS(ValidateSocPerfEntry(), "Validate SoC performance target failed.");
  GELOGI("[DFX]Begin to generate model info for graph %s of tiling case id %u", graph.GetName().c_str(),
         tiling_case_id);
  DURATION_GUARD(DurationType::DURATION_GEN_MODEL_INFO);
  model_info.tiling_case_id = tiling_case_id;
  // step1: get tuningspace from compute graph
  GE_ASSERT_NOTNULL(tuning_space, "Create tuning space failed.");
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  GE_ASSERT_SUCCESS(ascend_graph_parser.GraphParser(graph), "Get tuning space failed.");
  for (auto &node_info : tuning_space->node_infos) {
    node_info.is_cv_ub_fusion = is_cv_ub_fusion;
  }
  // step2: get basic expr constraint
  att::GenerateTilingExpr tiling_expr(tuning_space);
  GE_ASSERT_SUCCESS(tiling_expr.Generate(model_info), "Get basic expr constraint failed.");
  GE_ASSERT_SUCCESS(RefreshCommonUbExprContext(graph, model_info, is_cv_ub_fusion), "Refresh common UB expr failed.");
  GELOGI("[DFX]End to generate model info for graph %s of tiling case id %u", graph.GetName().c_str(), tiling_case_id);
  return af::SUCCESS;
}

af::Status GenerateModelInfo(const af::AscGraph &graph, ModelInfo &model_info, TuningSpacePtr &tuning_space,
                             const uint32_t tiling_case_id) {
  return GenerateSingleModelInfoWithContext(graph, model_info, tuning_space, tiling_case_id, false);
}

af::Status CheckKeyValid(const std::vector<af::AscGraph> &graph_list) {
  if (graph_list.size() > 0) {
    bool has_set_key = (graph_list[0].GetTilingKey() >= 0);
    for (auto &graph : graph_list) {
      GE_ASSERT_TRUE(has_set_key == (graph.GetTilingKey() >= 0),
                     "If the user has set tiling_case_id for a graph, the tiling_case_id of all graphs must be set.");
    }
  }
  return af::SUCCESS;
}

af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list) {
  GE_ASSERT_SUCCESS(ValidateSocPerfEntry(), "Validate SoC performance target failed.");
  GE_ASSERT_SUCCESS(CheckKeyValid(graph_list));
  uint32_t tiling_key = 0U;
  for (auto &graph : graph_list) {
    tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key;
    ModelInfo model_info;
    TuningSpacePtr tuning_space = af::MakeShared<TuningSpace>();
    GE_ASSERT_NOTNULL(tuning_space, "Make tuning space failed.");
    GE_ASSERT_SUCCESS(GenerateModelInfo(graph, model_info, tuning_space, tiling_key), "General model info failed.");
    model_info_list.emplace_back(model_info);
    tiling_key++;
  }
  GE_ASSERT_TRUE(!model_info_list.empty(), "No graph need convert.");
  return af::SUCCESS;
}

void to_json(nlohmann::json &j, const SymInfoPtr &arg) {
  GE_CHECK_NOTNULL_JUST_RETURN(arg);
  auto type = 0U;
  uint32_t const_value = 0U;
  auto const_arg = std::dynamic_pointer_cast<SymConstInfo>(arg);
  if (const_arg != nullptr) {
    type = kConstType;
    const_value = const_arg->const_value;
  }

  Expr align = af::Symbol(kDefaultAlignValue);
  std::vector<HardwareDef> related_scope;
  Expr max_value = af::sym::kSymbolZero;
  auto var_arg = std::dynamic_pointer_cast<SymVarInfo>(arg);
  std::pair<int64_t, int64_t> value_range;
  std::string symbol_expr;
  if (var_arg != nullptr) {
    type = kVarType;
    max_value = var_arg->max_value;
    related_scope = var_arg->related_scope;
    align = var_arg->align;
    value_range = var_arg->value_range;
    symbol_expr = (!IsValid(arg->symbol_expr)) ? "" : Str(arg->symbol_expr);
  }
  const std::string max_value_str = (!IsValid(max_value)) ? "null" : Str(max_value);
  j = nlohmann::json{
      {"type", (type == kConstType) ? "const" : "var"},
      {"symbol_expr", symbol_expr},
      {"const_value", const_value},
      {"expr_max_value", max_value_str},
      {"related_scope", related_scope},
      {"align", align},
      {"min_value", value_range.first},
      {"max_value", value_range.second},
  };
}

void to_json(nlohmann::json &j, const std::shared_ptr<AttAxis> &arg) {
  GE_CHECK_NOTNULL_JUST_RETURN(arg);
  std::string orig_axis;
  for (auto axis : arg->orig_axis) {
    orig_axis += axis->name;
    orig_axis += ",";
  }
  std::string from_axis;
  for (auto axis : arg->from_axis) {
    from_axis += axis->name;
    from_axis += ",";
  }
  j = nlohmann::json{
      {"name", arg->name},
      {"axis_pos", arg->axis_pos},
      {"bind_multicore", arg->bind_multicore},
      {"is_last", arg->is_last},
      {"is_node_innerest_dim", arg->is_node_innerest_dim},
      {"size", arg->size},
      {"orig_axis", orig_axis},
      {"from_axis", from_axis},
  };
}

void to_json(nlohmann::json &j, const ATTConfig &info) {
  j = nlohmann::json{
      {"config_inputs", info.config_names},
      {"align_config", info.config_value},
  };
}

void to_json(nlohmann::json &j, const HardwareDef &type) {
  j = nlohmann::json{
      {HardwareType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const AxisPosition &type) {
  j = nlohmann::json{
      {AxisType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const PipeType &type) {
  j = nlohmann::json{
      {PipeType2Str.at(type)},
  };
}

void to_json(nlohmann::json &j, const ModelInfo &info) {
  j = nlohmann::json{
      {"tiling_case_id", info.tiling_case_id},
      {"hardware_cons", info.hardware_cons},
      {"eq_exprs", info.eq_exprs},
      {"leq_exprs", info.leq_exprs},
      {"objects", info.objects},
      {"arg_list", info.arg_list},
  };
}

std::string GetRealPath(const std::string &path) {
  if (path.empty() || (path.size() >= kPathMax)) {
    GELOGW("Invalid path: size [%zu].", path.size());
    return "";
  }
  std::string root_path = af::RealPath(path.c_str());
  GE_ASSERT_TRUE(!root_path.empty(), "Invalid path: %s", path.c_str());
  if (!kFileSeparator.empty() && (root_path[root_path.size() - 1U] != kFileSeparator.back())) {
    root_path += kFileSeparator;
  }

  return root_path;
}

void DumpModelInfo(const std::vector<ModelInfo> &model_info_list, std::string dump_dir = "") {
  nlohmann::json j = nlohmann::json{
      {"model_info", model_info_list},
  };
  std::ofstream file_stream;
  std::string model_file = kModelInfoFileName;
  std::string output_path = dump_dir;
  if (output_path.empty()) {
    output_path = kModelInfoFilePath;
  }
  model_file = GetRealPath(output_path) + model_file;
  ge::char_t realpath_file[PATH_MAX] = {0x00};
  auto ret = realpath(model_file.c_str(), realpath_file);
  if (ret == nullptr) {
    GELOGD("Model file path [%s] unfound.", model_file.c_str());
    return;
  }
  file_stream.open(realpath_file, std::ios::out | std::ios::trunc);
  if (file_stream.is_open()) {
    file_stream << j.dump();
    file_stream.close();
  }
}

void ProcessTilingR(std::vector<ModelInfo> &model_info_list, const ModelInfo &model_info,
                    std::vector<AttAxisPtr> &tiling_R_arg_list) {
  if (tiling_R_arg_list.empty()) {
    return;
  }
  ModelInfo model_info_tiling_R = model_info;
  model_info_tiling_R.arg_list = tiling_R_arg_list;
  model_info_tiling_R.sub_case_tag = "R";
  model_info_tiling_R.runtime_reorder_rules.clear();
  model_info_list.emplace_back(model_info_tiling_R);
}

void ProcessGraphOriginalSizeVar(const af::AscGraph &graph, ModelInfo &model_info) {
  optimize::SizeVarSet var_set;
  model_info.sizes.clear();
  optimize::AscGraphInfoComplete::AppendOriginalSizeVar(graph, var_set);
  for (const auto &var : var_set) {
    model_info.sizes.emplace_back(var);
  }
}

inline bool IsAxesReorderAlgorithm() {
  const auto res = AutoFuseConfig::MutableAttStrategyConfig().Init();
  if (res == af::SUCCESS) {
    // 环境变量无配置则默认使用轴排序算法
    const auto &att_config = AutoFuseConfig::GetAttStrategyConfig();
    if (att_config.set_env_tiling_algorithm) {
      return att_config.tiling_algorithm == "AxesReorder";
    }
  }
  return true;
}

struct ModelGenerationContext {
  // 一次 schedule result 内所有 tiling case 共享的内部上下文，避免为 Codegen 路径门禁扩展公开
  // GenerateModelInfo 接口。
  bool enable_gather_reduce_penalty{false};
  // kUBFuse 生成专用的固定 2D NDDMA 描述，不使用默认 DataCopyParams 描述。
  bool is_cv_ub_fusion{false};
};

static af::Status GenerateModelInfoWithContext(const std::vector<af::AscGraph> &graph_list,
                                               std::vector<ModelInfo> &model_info_list,
                                               const std::map<std::string, std::string> &options,
                                               bool enable_group_parallel, const ModelGenerationContext &context,
                                               const std::string &base_semantic_name);

af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options, bool enable_group_parallel) {
  GE_ASSERT_SUCCESS(ValidateSocPerfEntry(), "Validate SoC performance target failed.");
  return GenerateModelInfo(graph_list, model_info_list, options, enable_group_parallel, false);
}

af::Status GenerateModelInfo(const std::vector<af::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options, bool enable_group_parallel,
                             bool enable_gather_reduce_penalty) {
  GE_ASSERT_SUCCESS(ValidateSocPerfEntry(), "Validate SoC performance target failed.");
  return GenerateModelInfoWithContext(graph_list, model_info_list, options, enable_group_parallel,
                                      {enable_gather_reduce_penalty, false}, "");
}

static af::Status GenerateModelInfoWithContext(const std::vector<af::AscGraph> &graph_list,
                                               std::vector<ModelInfo> &model_info_list,
                                               const std::map<std::string, std::string> &options,
                                               bool enable_group_parallel, const ModelGenerationContext &context,
                                               const std::string &base_semantic_name) {
  GE_ASSERT_SUCCESS(CheckKeyValid(graph_list));
  uint32_t tiling_key = 0U;
  for (auto &graph : graph_list) {
    tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key;
    ModelInfo model_info;
    model_info.graph_name = graph.GetName();
    model_info.semantic_name =
        ascgen_utils::NormalizeTemplateSemanticName(ascgen_utils::GetTemplateSemanticName(graph, base_semantic_name));
    model_info.tiling_case_id = tiling_key;
    model_info.enable_group_parallel = enable_group_parallel;
    std::vector<AttAxisPtr> tiling_R_arg_list;
    TuningSpacePtr tuning_space = af::MakeShared<TuningSpace>();
    GE_ASSERT_NOTNULL(tuning_space, "Make tuning space failed.");
    if (context.enable_gather_reduce_penalty) {
      tuning_space->penalty_cache_line_size = kGatherReducePenaltyCacheLineSize;
    }
    GELOGD("[DFX] GatherReducePenalty config: graph=%s, enabled=%d, penalty_cache_line_size=%u",
           graph.GetName().c_str(), context.enable_gather_reduce_penalty, tuning_space->penalty_cache_line_size);
    tuning_space->cache_line_config = &model_info.cache_line_config;
    GetThreadLocalContext().SetOption(options);
    GE_ASSERT_SUCCESS(
        GenerateSingleModelInfoWithContext(graph, model_info, tuning_space, tiling_key, context.is_cv_ub_fusion),
        "General model info failed.");
    if (IsAxesReorderAlgorithm()) {
      ArgListReorder arg_list_reorder(tuning_space);
      GE_ASSERT_SUCCESS(
          arg_list_reorder.SortArgList(model_info.arg_list, tiling_R_arg_list, &model_info.runtime_reorder_rules),
          "Sort arg list failed.");
    }
    const auto iter = options.find(kTilingDataTypeName);
    const std::string tiling_data_type = (iter != options.cend()) ? iter->second : "";
    ApiTilingParams params{graph, tiling_data_type};
    GE_ASSERT_SUCCESS(GetApiTilingInfo(model_info.tiling_case_id, params, model_info.node_name_to_api_code),
                      "Generate api tiling info failed.");
    // 获取所有的原始size_var
    ProcessGraphOriginalSizeVar(graph, model_info);
    ProcessTilingR(model_info_list, model_info, tiling_R_arg_list);
    model_info_list.emplace_back(model_info);
    tiling_key++;
  }
  GE_ASSERT_TRUE(!model_info_list.empty(), "No graph need convert.");
  if (options.find(kDumpDebugInfo) != options.end()) {
    DumpModelInfo(model_info_list, options.at(kDumpDebugInfo).back() == '/' ? options.at(kDumpDebugInfo)
                                                                            : options.at(kDumpDebugInfo) + "/");
  }
  return af::SUCCESS;
}

af::Status MakeJsonIner(const std::vector<ModelInfo> &model_info_list, std::string &json_info) {
  nlohmann::json j = nlohmann::json{
      {"model_info", model_info_list},
  };
  json_info = j.dump();
  DumpModelInfo(model_info_list);
  return af::SUCCESS;
}

af::Status MakeJson(std::vector<ModelInfo> &model_info_list, std::string &json_info) {
  GE_ASSERT_SUCCESS(MakeJsonIner(model_info_list, json_info), "Make json failed.");
  return af::SUCCESS;
}

af::Status GetAllSubImplGraphs(const ascir::FusedScheduledResult &schedule_results,
                               std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> &all_graphs,
                               std::map<std::string, std::string> &all_graph_score_funcs) {
  bool has_none_graph = true;
  for (const auto &asc_graph : schedule_results.node_idx_to_scheduled_results) {
    std::vector<std::vector<std::vector<af::AscGraph>>> cur_asc_graphs;
    for (const auto &schedule_result : asc_graph) {
      std::vector<std::vector<af::AscGraph>> schedule_result_graphs;
      for (const auto &schedule_group : schedule_result.schedule_groups) {
        schedule_result_graphs.emplace_back(schedule_group.impl_graphs);
        all_graph_score_funcs.insert(schedule_group.graph_name_to_score_funcs.begin(),
                                     schedule_group.graph_name_to_score_funcs.end());
        if (!schedule_group.impl_graphs.empty()) {
          has_none_graph = false;
        }
      }
      cur_asc_graphs.emplace_back(schedule_result_graphs);
    }
    all_graphs.emplace_back(cur_asc_graphs);
  }
  GE_ASSERT_TRUE(!has_none_graph);
  return af::SUCCESS;
}

namespace {
af::Status ProcessAndSetScheduleGroupInfo(const std::vector<std::vector<af::AscGraph>> &schedule_groups,
                                          const std::map<std::string, std::string> &all_graph_score_funcs,
                                          const ascir::FusedScheduledResult &schedule_results,
                                          const std::map<std::string, std::string> &options,
                                          att::ParsedScheduleResult &out_schedule_groups, size_t asc_graph_id,
                                          size_t impl_graph_id) {
  // 第三层表示schedule_group_id
  for (size_t schedule_group_id = 0UL; schedule_group_id < schedule_groups.size(); schedule_group_id++) {
    auto &model_info_list = out_schedule_groups.groups_tiling_model_info[schedule_group_id];
    GELOGI(
        "[DFX]Begin to gen model info for asc graph %zu, schedule result %zu, schedule group %zu, tiling_case size "
        "%zu, graph name %s.",
        asc_graph_id, impl_graph_id, schedule_group_id, schedule_groups[schedule_group_id].size(),
        !schedule_groups[schedule_group_id].empty() ? schedule_groups[schedule_group_id][0].GetName().c_str() : "null");
    const bool enable_gather_reduce_penalty =
        ShouldEnableGatherReducePenalty(schedule_groups, schedule_group_id, out_schedule_groups.enable_group_parallel);
    GELOGD(
        "[DFX] GatherReducePenalty context: asc_graph_id=%zu, impl_graph_id=%zu, group_id=%zu, group_count=%zu, "
        "enabled=%d",
        asc_graph_id, impl_graph_id, schedule_group_id, schedule_groups.size(), enable_gather_reduce_penalty);
    const auto &schedule_result = schedule_results.node_idx_to_scheduled_results[asc_graph_id][impl_graph_id];
    // ATT 必须与 Codegen 使用相同的 schedule result 路径，否则 raw 1D 节点会按 1D 模型评分，
    // 但 Codegen 最终生成 kUBFuse 固定 2D 描述。
    const bool is_cv_ub_fusion = schedule_result.cube_type == ascir::CubeTemplateType::kUBFuse;
    const ModelGenerationContext context{enable_gather_reduce_penalty, is_cv_ub_fusion};
    GE_ASSERT_SUCCESS(
        GenerateModelInfoWithContext(schedule_groups[schedule_group_id], model_info_list, options,
                                     out_schedule_groups.enable_group_parallel, context, schedule_result.semantic_name),
        "Get model info failed, impl graph id = %ld, group id = %ld.", impl_graph_id, schedule_group_id);
    for (auto &model_info : model_info_list) {
      model_info.schedule_group_ident.asc_graph_id = asc_graph_id;
      model_info.schedule_group_ident.impl_graph_id = impl_graph_id;
      model_info.schedule_group_ident.group_id = schedule_group_id;
      model_info.input_nodes = schedule_results.input_nodes;
      model_info.output_nodes = schedule_results.output_nodes;
      FilterOutputAliasWorkspace(model_info);
      auto it = all_graph_score_funcs.find(model_info.graph_name);
      if (it != all_graph_score_funcs.end()) {
        model_info.score_func = it->second;
      }
    }
    const auto &ident = model_info_list[0].schedule_group_ident;
    GELOGI(
        "[DFX]End to gen model info for %s tiling_case size %zu, graph name %s.", ident.GetItemPrefix().c_str(),
        schedule_groups[schedule_group_id].size(),
        !schedule_groups[schedule_group_id].empty() ? schedule_groups[schedule_group_id][0].GetName().c_str() : "null");
    GE_ASSERT_SUCCESS(ReuseGroupUtils::InitReuseScheduleGroup(ident, model_info_list),
                      "Init reuse schedule group failed, impl_graph_id[%zu], group_ident[%s].", impl_graph_id,
                      ident.GetItemPrefix().c_str());
  }
  return af::SUCCESS;
}
}  // namespace

af::Status GetModelInfoMap(const ascir::FusedScheduledResult &schedule_results,
                           const std::map<std::string, std::string> &options,
                           FusedParsedScheduleResult &out_all_model_infos) {
  GE_ASSERT_SUCCESS(ValidateSocPerfEntry(), "Validate SoC performance target failed.");
  std::vector<std::vector<std::vector<std::vector<af::AscGraph>>>> all_graphs_lists;
  std::map<std::string, std::string> all_graph_score_funcs;
  GE_ASSERT_SUCCESS(GetAllSubImplGraphs(schedule_results, all_graphs_lists, all_graph_score_funcs));
  // 最外层表示asc_graph_id
  for (size_t asc_graph_id = 0UL; asc_graph_id < all_graphs_lists.size(); asc_graph_id++) {
    auto &out_asc_graph_model_infos = out_all_model_infos[asc_graph_id];
    auto &asc_graph_list = all_graphs_lists[asc_graph_id];
    // 第二层表示impl_graph_id
    for (size_t impl_graph_id = 0UL; impl_graph_id < asc_graph_list.size(); impl_graph_id++) {
      if (asc_graph_list[impl_graph_id].empty() || asc_graph_list[impl_graph_id][0].empty()) {
        continue;
      }
      auto &schedule_groups = asc_graph_list[impl_graph_id];
      auto &out_schedule_groups = out_asc_graph_model_infos[impl_graph_id];
      const auto &schedule_result = schedule_results.node_idx_to_scheduled_results[asc_graph_id][impl_graph_id];
      GELOGD(
          "out_schedule_groups input values: score_func=%s, enable_group_parallel=%d, asc_graph_id=%zu, "
          "impl_graph_id=%zu",
          schedule_result.score_func.GetString(), schedule_result.enable_group_parallel, asc_graph_id, impl_graph_id);
      out_schedule_groups.score_func = schedule_result.score_func.GetString();
      out_schedule_groups.enable_group_parallel = schedule_result.enable_group_parallel;
      out_schedule_groups.asc_graph_id = asc_graph_id;
      out_schedule_groups.impl_graph_id = impl_graph_id;
      out_schedule_groups.var_relations =
          schedule_results.node_idx_to_scheduled_results[asc_graph_id][impl_graph_id].var_relations;
      GE_ASSERT_SUCCESS(ProcessAndSetScheduleGroupInfo(schedule_groups, all_graph_score_funcs, schedule_results,
                                                       options, out_schedule_groups, asc_graph_id, impl_graph_id));
    }
  }
  // 合并所有可以复用的group
  GE_ASSERT_SUCCESS(ReuseGroupUtils::MergeAllReusableGroups(all_graphs_lists, out_all_model_infos));
  return af::SUCCESS;
}
}  // namespace att
