/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_code_gen_impl.h"
#include "final_tiling_log_gen.h"
#include <fstream>
#include <algorithm>
#include <set>
#include <queue>
#include <sstream>
#include <utility>
#include "args_manager.h"
#include "common/checker.h"
#include "util/duration.h"
#include "mmpa/mmpa_api.h"
#include "base_types_printer.h"
#include "base/att_const_values.h"
#include "base/att_group_perf_config.h"
#include "common_utils.h"
#include "generator_utils/tilingdata_gen_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "autofuse_config/auto_fuse_config.h"
#include "symbolizer/symbolic_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"

namespace att {
namespace {
constexpr size_t kLogLength = 200;
constexpr uint32_t kMaxDepth = 20U;
constexpr size_t kPerfAnnotationMaxExprLen = 80U;
constexpr ge::char_t kLogLevelStr[] = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr ge::char_t kEventEnableStr[] = "ASCEND_GLOBAL_EVENT_ENABLE";
constexpr ge::char_t kInlineStr[] = "inline ";

inline int32_t GotLogLevel() {
  ge::char_t env_path[MMPA_MAX_PATH] = {};
  bool has_got = (mmGetEnv(kLogLevelStr, env_path, MMPA_MAX_PATH) == EN_OK);
  int32_t got_log_level = DLOG_ERROR;
  if (has_got) {
    got_log_level = std::atoi(env_path);
  }
  return got_log_level;
}

inline const std::string &AddSlogExtend() {
  const static std::string kGeLogUtils = {
#include "ge_log_utils_src.h"
  };
  if (GotLogLevel() == DLOG_NULL) {
    const static std::string kNullStr;
    return kNullStr;
  }
  return kGeLogUtils;
}

std::set<int64_t> GetWorkspaceTensorIds(const TensorIdSet &workspace_tensor_id_set) {
  std::set<int64_t> workspace_ids;
  for (const auto &[asc_graph_id, impl_graph_ids] : workspace_tensor_id_set) {
    (void)asc_graph_id;
    for (const auto &[impl_graph_id, tensor_ids] : impl_graph_ids) {
      (void)impl_graph_id;
      workspace_ids.insert(tensor_ids.begin(), tensor_ids.end());
    }
  }
  return workspace_ids;
}

std::set<int64_t> GetAscGraphWorkspaceTensorIds(const TensorIdSet &workspace_tensor_id_set, const size_t asc_graph_id) {
  std::set<int64_t> workspace_ids;
  const auto asc_graph_iter = workspace_tensor_id_set.find(asc_graph_id);
  if (asc_graph_iter != workspace_tensor_id_set.end()) {
    for (const auto &[impl_graph_id, tensor_ids] : asc_graph_iter->second) {
      (void)impl_graph_id;
      workspace_ids.insert(tensor_ids.begin(), tensor_ids.end());
    }
  }
  return workspace_ids;
}

template <typename T>
af::Status IsUpperBoundValid(const Expr &min_expr, const Expr &max_expr) {
  T min_value{};
  T max_value{};
  (void)min_expr.GetConstValue(min_value);
  (void)max_expr.GetConstValue(max_value);
  GE_ASSERT_TRUE(min_value <= max_value, "Args manager process failed, min[%u] cannot be greater than max[%u].",
                 min_value, max_value);
  return af::SUCCESS;
}

void GenLogDefine(ge::CodePrinter &print, bool preserve_includes) {
  const auto &slog_extend = AddSlogExtend();
  std::stringstream log_body;
  const std::set<std::string> known_includes = {
      "#include <cinttypes>", "#include <sys/syscall.h>", "#include <unistd.h>",   "#include <cstdint>",
      "#include <stdio.h>",   "#include <stdlib.h>",      "#include <map>",        "#include <securec.h>",
      "#include <vector>",    "#include \"dlog_pub.h\"",  "#include \"err_msg.h\""};
  std::istringstream log_stream(slog_extend);
  std::string line;
  while (std::getline(log_stream, line)) {
    if (known_includes.count(line) == 0U) {
      log_body << line << '\n';
    }
  }
  const auto extend_define = slog_extend.empty() ? "\n" : (preserve_includes ? slog_extend : log_body.str()) + "\n";
  // 根据 ASCEND_GLOBAL_LOG_LEVEL 决定生成的宏内容：
  // DLOG_NULL 时生成空宏，否则生成完整宏
  // 运行时通过 CheckLogLevel 动态判断是否真正输出
  const bool is_null_log = (GotLogLevel() == DLOG_NULL);

  const bool profiling_enabled = IsProfilingEnabled();
  const std::string debug_log_define =
      (is_null_log || profiling_enabled) ? R"(#define OP_LOGD(name, fmt, ...))"
                                         : R"(#define OP_LOGD(name, fmt, ...) GELOGD("[%s]" fmt, name, ##__VA_ARGS__))";
  const std::string info_log_define =
      (is_null_log || profiling_enabled) ? R"(#define OP_LOGI(name, fmt, ...))"
                                         : R"(#define OP_LOGI(name, fmt, ...) GELOGI("[%s]" fmt, name, ##__VA_ARGS__))";
  const std::string final_tiling_info_log_define =
      is_null_log ? R"(#define ATT_FINAL_TILING_LOGI(name, fmt, ...))"
                  : R"(#define ATT_FINAL_TILING_LOGI(name, fmt, ...) GELOGI("[%s]" fmt, name, ##__VA_ARGS__))";
  const std::string warn_log_define =
      (is_null_log || profiling_enabled) ? R"(#define OP_LOGW(name, fmt, ...))"
                                         : R"(#define OP_LOGW(name, fmt, ...) GELOGW("[%s]" fmt, name, ##__VA_ARGS__))";
  const std::string err_log_define =
      (is_null_log || profiling_enabled)
          ? R"(#define OP_LOGE(name, fmt, ...))"
          : R"(#define OP_LOGE(name, fmt, ...) GELOGE(-1, "[%s]" fmt, name, ##__VA_ARGS__))";
  std::string event_log_define = "#define OP_EVENT(name, fmt, ...)";
  const std::string event_append_log =
      (!is_null_log && profiling_enabled) ? R"( GELOGI("[%s]" fmt, name, ##__VA_ARGS__))" : "";
  event_log_define.append(event_append_log);

  print.AddLine(extend_define);
  print.AddLine(debug_log_define);
  print.AddLine(info_log_define);
  print.AddLine(final_tiling_info_log_define);
  print.AddLine(warn_log_define);
  print.AddLine(err_log_define);
  print.AddLine(event_log_define);
}

std::string GenParsePrint(const std::string &log_info, const int32_t log_level) {
  std::string output;
  std::string log_level_str = (log_level == DLOG_ERROR) ? "E" : "I";
  for (size_t i = 0; i < log_info.size(); i += kLogLength) {
    output += "    OP_LOG" + log_level_str + "(OP_NAME, \"" + log_info.substr(i, kLogLength) + "\");\n";
  }
  return output;
}

std::string GenConsExprPrint(const ArgsManager &args_manager, const std::string &group_prefix,
                             const int32_t log_level) {
  std::string output;
  std::string cur_log;
  for (const auto &pair : args_manager.GetTotalHardwareCons()) {
    cur_log = "Set " + BaseTypeUtils::DumpHardware(pair.first) + " for tiling case " +
              std::to_string(args_manager.GetTilingCaseId()) + " of " + group_prefix + " to " + Str(pair.second);
    output += GenParsePrint(cur_log, log_level);
  }
  return output;
}

std::string GenInputParamsPrint(const ArgsManager &args_manager, const std::string &group_prefix,
                                const int32_t log_level) {
  std::string set_code;
  std::string param;
  for (const auto &arg : args_manager.GetInputVars()) {
    set_code.append(" ").append(Str(arg)).append(" = %u.");
    param.append(", tiling_data.get_").append(Str(arg)).append("()");
  }
  std::string output("    OP_LOG");
  std::string log_level_str = (log_level == DLOG_ERROR) ? "E" : "I";
  return output.append(log_level_str)
      .append("(OP_NAME, \"Set input params for tiling case ")
      .append(std::to_string(args_manager.GetTilingCaseId()))
      .append(" of ")
      .append(group_prefix)
      .append(". ")
      .append(set_code)
      .append("\"")
      .append(param)
      .append(");");
}

inline std::string GenScheduleResultFuncsDefine(
    const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map,
    const std::string &pgo = "") {
  std::string schedule_result_funcs_define("const std::array<ScheduleResultFunction" + pgo + ", ");
  schedule_result_funcs_define.append(std::to_string(namespace_map.size()))
      .append("> kScheduleResultFunctions" + pgo + " = {");
  for (size_t id = 0UL; id < namespace_map.size(); id++) {
    schedule_result_funcs_define.append("GetScheduleResult").append(std::to_string(id) + pgo).append(", ");
    if (id == (namespace_map.size() - 1UL)) {
      schedule_result_funcs_define.append("};");
    }
  }
  return schedule_result_funcs_define;
}

inline const std::string GenScheduleResultFuncTypeDefine(const std::string &tiling_data_name) {
  std::string schedule_result_func_define =
      "using ScheduleResultFunction = std::function<bool(const uint32_t ori_block_dim, const int32_t tiling_case_id, ";
  return schedule_result_func_define.append(tiling_data_name)
      .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim)>;");
}

inline const std::string GenPGOScheduleResultFuncTypeDefine(const std::string &tiling_data_name) {
  std::string schedule_result_func_define =
      "using ScheduleResultFunctionPGO = std::function<bool(std::vector<AutofuseTilingDataPerf>& tiling_data_list, "
      "const uint32_t ori_block_dim, const int32_t tiling_case_id, ";
  return schedule_result_func_define.append(tiling_data_name)
      .append(
          " &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim, void* stream, "
          "uint64_t workspaceSize, std::vector<uint32_t*> block_dim_vec, const SearchConfig *search_cfg)>;");
}

inline const std::string GenPGOByCoreNumScheduleResultFuncTypeDefine() {
  std::string schedule_result_func_define =
      "using ScheduleResultFunctionPGOByCoreNum = std::function<bool(std::vector<AutofuseTilingData>& "
      "tiling_data_list, AutofuseTilingData tiling_data)>;";
  return schedule_result_func_define;
}

inline std::string GetScheduleResultPrefix(const size_t asc_graph_id, const size_t result_id) {
  return "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(result_id);
}

inline bool NeedGenScoreFunc(const ScoreFuncs &score_funcs) {
  // asc graph
  for (const auto &single_level_score_funcs : score_funcs) {
    // impl graph
    for (const auto &asc_graph_score_func : single_level_score_funcs.second) {
      for (const auto &impl_graph_score_func : asc_graph_score_func.second) {
        if (!impl_graph_score_func.second.empty()) {
          return true;
        }
      }
    }
  }
  return false;
}

inline std::string GenPGOScheduleGroupDoTiling(const std::string &hardware_param,
                                               const std::string &schedule_result_prefix) {
  return schedule_result_prefix + "::PGOSearchTilingKey(tiling_data_list_tmp, " + hardware_param + "_tiling_data, " +
         "tiling_case_id, &tiling_data, PgoConfig::Instance().tensor_args, stream, workspaceSize, best_perf, "
         "workspace_map_filter_use, multi_group_block_dim_list, search_cfg)";
}
inline std::string GenPGOReuseGroupProfile(const std::string &schedule_result_prefix) {
  return schedule_result_prefix + "::PGOProfileReuseGroup(tiling_data_list_tmp, &tiling_data, " +
         "stream, workspaceSize, best_perf)";
}
inline std::string GenGetScheduleGroupPerf(const std::string &namespace_prefix, const std::string &item_prefix) {
  return namespace_prefix + "::GetPerf(" + item_prefix + "_tiling_data)";
}

inline std::string GenUpdateCurPerfAndBlockByGroup() {
  return ascgen_utils::GenUpdateCurPerfAndBlockByGroupHelper(true, false);
}

inline std::string GenSumAllGroupsPerf(const std::vector<std::string> &groups_perf) {
  std::string sum_all_groups_perf;
  for (const auto &perf : groups_perf) {
    if (sum_all_groups_perf.empty()) {
      sum_all_groups_perf.append("      cur_perf = " + perf + ";\n");
    } else {
      sum_all_groups_perf.append("      cur_perf += " + perf + ";\n");
    }
  }
  return sum_all_groups_perf;
}

inline std::string GenGetCurBlockDim(const std::string &item_prefix) {
  return item_prefix + "_tiling_data.get_block_dim()";
}

inline std::string GenCurMaxBlockDim(const std::string &item_prefix, const std::vector<std::string> &block_num,
                                     std::string &cur_block) {
  cur_block = GenGetCurBlockDim(item_prefix);
  std::string call_max_block_dim = "Max(cur_block_dim, " + cur_block + ")";
  return "      cur_block_dim = " + (!block_num.empty() ? call_max_block_dim : cur_block) + ";";
}

inline std::vector<std::string> GenScheduleResultBlockDimCode(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  std::vector<std::string> block_dim_code;
  if (graph_info.empty()) {
    block_dim_code.emplace_back("      tiling_data.set_block_dim(tiling_data.get_block_dim());");
    return block_dim_code;
  }
  auto group_info = graph_info.cbegin();
  const std::string first_group_block_dim = "tiling_data." + group_info->second.second + "_tiling_data.get_block_dim()";
  block_dim_code.emplace_back("      uint32_t max_block_dim = " + first_group_block_dim + ";");
  ++group_info;
  for (; group_info != graph_info.cend(); ++group_info) {
    const std::string group_block_dim = "tiling_data." + group_info->second.second + "_tiling_data.get_block_dim()";
    block_dim_code.emplace_back("      max_block_dim = Max(max_block_dim, " + group_block_dim + ");");
  }
  block_dim_code.emplace_back("      tiling_data.set_block_dim(max_block_dim);");
  return block_dim_code;
}

inline bool HasSymbol(const Expr &expr) {
  return !expr.FreeSymbols().empty();
}

void GetRelatedInfo(const ArgsManager &args_manager, const Expr &expr, ExprExprMap &param_map, bool &related) {
  related = false;
  param_map.clear();
  ExprExprMap container_map = args_manager.GetContainerMap();
  for (const auto &arg : expr.FreeSymbols()) {
    auto iter = container_map.find(arg);
    if (iter != container_map.end()) {
      GELOGD("Add param map [%s] -> [%s].", Str(arg).c_str(), Str(iter->second).c_str());
      param_map[arg] = iter->second;
    }
  }
  for (const auto &arg : args_manager.GetSearchableVars()) {
    if (expr.ContainVar(arg)) {
      GELOGD("Expr [%s] contain arg [%s].", Str(expr).c_str(), Str(arg).c_str());
      related = true;
    }
    for (const auto &pair : param_map) {
      if (expr.ContainVar(pair.first) && pair.second.ContainVar(arg)) {
        GELOGD("Expr [%s](%s) contain arg [%s].", Str(pair.first).c_str(), Str(pair.second).c_str(), Str(arg).c_str());
        related = true;
      }
    }
  }
}

af::Status UpdateRelatedVars(const Expr &expr, const ExprExprMap &param_map, std::set<std::string> &related_vars,
                             uint32_t depth) {
  GE_ASSERT_TRUE(depth <= kMaxDepth, "Out of max depth!");
  for (const auto &arg : expr.FreeSymbols()) {
    auto iter = param_map.find(arg);
    if (arg.GetExprType() == af::ExprType::kExprVariable) {
      GELOGD("Analysis arg [%s].", Str(arg).c_str());
      if (iter == param_map.end()) {
        GELOGD("Arg [%s] is not a container.", Str(arg).c_str());
        related_vars.insert(Str(arg));
      } else {
        GE_ASSERT_SUCCESS(UpdateRelatedVars(iter->second, param_map, related_vars, depth + 1));
      }
    }
  }
  return af::SUCCESS;
}

// 检查变量名是否以指定后缀结尾
bool CheckPerf(const std::string suffix, const std::string &var_name) {
  if (var_name.length() < suffix.length()) {
    return false;
  }
  return var_name.substr(var_name.length() - suffix.length()) == suffix;
}

std::string GetPerfBreakdownIndent(uint32_t indent) {
  return std::string(4U + indent * 2U, ' ');
}

void AppendPerfBreakdownAnnotations(const ArgsManager &args_manager, const std::string &tiling_id,
                                    std::string &annotations) {
  const auto &perf_breakdowns = args_manager.GetPerfBreakdowns();
  if (perf_breakdowns.empty()) {
    return;
  }
  const auto replace_vars = args_manager.GetTernaryOpReplaceVars();
  annotations += " Reduce perf breakdown used for tiling case " + tiling_id + " is:\n";
  for (const auto &group : perf_breakdowns) {
    annotations += "  " + group.title + ":\n";
    for (const auto &item : group.items) {
      Expr item_expr = item.expr.Replace(replace_vars);
      item_expr.Simplify();
      const std::string indent = GetPerfBreakdownIndent(item.indent);
      const std::string item_desc = item.desc.empty() ? item.name : item.name + ": " + item.desc;
      const std::string expr_str = Str(item_expr);
      if (item_desc.length() + expr_str.length() <= kPerfAnnotationMaxExprLen) {
        annotations += indent + item_desc + " = " + expr_str + "\n";
      } else {
        annotations += indent + item_desc + "\n";
        annotations += indent + "  = " + expr_str + "\n";
      }
    }
  }
}

inline std::string GenCallUpdateBetterTiling(bool is_uniq_group, bool hardware_has_ub) {
  std::string workspace_param;
  if (is_uniq_group) {
    workspace_param = "";
  } else {
    workspace_param = ", workspace_map";
  }
  std::string func_params = std::string("tilingCaseImplPtr, tmp_tiling, tiling_data") + workspace_param +
                            ", tiling_case_id, sub_case_tag, selected_sub_case_tag";
  const std::string ub_ratio_update = hardware_has_ub ? "        ub_ratio = cur_ub_ratio;\n" : "";
  const std::string kUpdateBetterTilingCode = R"(
        if (!UpdateBetterTiling()" + func_params +
                                              R"()) {
          return false;
        }
        sub_case_flag = is_sub_case;
        obj = cur_obj;
  )" + ub_ratio_update;
  return kUpdateBetterTilingCode;
}

inline std::string GenScoreTilingCaseStruct() {
  return R"(
struct ScoreTilingCase {
  const char* sub_case_tag;
  int32_t tiling_case_id;
  TilingCaseImpl *tiling_case_ptr;
  ScoreTilingCase(const char *tag, int32_t case_id, TilingCaseImpl *case_ptr)
      :sub_case_tag(tag), tiling_case_id(case_id), tiling_case_ptr(case_ptr){}
};
)";
}

std::string GenTilingScoreFuncDefineHead(bool is_uniq_group) {
  std::string workspace_param;
  if (!is_uniq_group) {
    workspace_param = ", workspace_map";
  }
  const std::string part1 = R"(  bool ret = false;
  for (const auto &s: score_map) {
    for (const auto &tiling: s.second) {
      OP_LOGD(OP_NAME, "Calculating the tiling data for tiling_case_id %s%d of score[%d]", tiling.sub_case_tag,
              tiling.tiling_case_id, s.first);
      ret |= FindPerfBetterTilingbyCaseId(tiling.tiling_case_ptr, obj, ub_ratio, tmp_tiling, tiling_data)";

  // core_num 始终传递给 FindPerfBetterTilingbyCaseId（用于保持签名一致）
  const std::string part2 =
      std::string(
          ", tiling.tiling_case_id, tiling.sub_case_tag[0] != 0, sub_case_flag, core_num, tiling.sub_case_tag, "
          "selected_sub_case_tag);\n") +
      R"(      OP_LOGD(OP_NAME, "Finish calculating the tiling data for tiling_case_id %s%d", tiling.sub_case_tag,
        tiling.tiling_case_id);
    })";

  return part1 + workspace_param + part2;
}
}  // namespace

namespace {
std::string GetAtomicHeaderKey(autofuse::GeneratedHeaderId header_id) {
  switch (header_id) {
    case autofuse::GeneratedHeaderId::kState:
      return kTilingStateHeaderIdentify;
    case autofuse::GeneratedHeaderId::kLog:
      return kTilingLogHeaderIdentify;
    case autofuse::GeneratedHeaderId::kPgo:
      return kTilingPgoHeaderIdentify;
    case autofuse::GeneratedHeaderId::kSolver:
      return kTilingSolverHeaderIdentify;
    case autofuse::GeneratedHeaderId::kApi:
      return kTilingApiHeaderIdentify;
    default:
      return "";
  }
}

std::string GetAtomicHeaderGuard(autofuse::GeneratedHeaderId header_id) {
  switch (header_id) {
    case autofuse::GeneratedHeaderId::kState:
      return "__AUTOFUSE_TILING_FUNC_STATE_H__";
    case autofuse::GeneratedHeaderId::kLog:
      return "__AUTOFUSE_TILING_FUNC_LOG_H__";
    case autofuse::GeneratedHeaderId::kPgo:
      return "__AUTOFUSE_TILING_FUNC_PGO_H__";
    case autofuse::GeneratedHeaderId::kSolver:
      return "__AUTOFUSE_TILING_FUNC_SOLVER_H__";
    case autofuse::GeneratedHeaderId::kApi:
      return "__AUTOFUSE_TILING_FUNC_API_H__";
    default:
      return "";
  }
}

std::string WrapAtomicHeaderBody(autofuse::GeneratedHeaderId header_id, const std::string &body,
                                 const std::string &tiling_data_type_name, bool is_autofuse) {
  const bool has_global_tiling_data = is_autofuse && tiling_data_type_name == "AutofuseTilingData";
  if (header_id == autofuse::GeneratedHeaderId::kLog) {
    return body;
  }
  if (header_id == autofuse::GeneratedHeaderId::kPgo) {
    if (has_global_tiling_data) {
      return "struct " + tiling_data_type_name + ";\nstruct AutofuseTilingDataPerf;\nnamespace optiling {\n" + body +
             "}  // namespace optiling\n";
    }
    return "struct AutofuseTilingDataPerf;\nnamespace optiling {\nstruct " + tiling_data_type_name + ";\n" + body +
           "}  // namespace optiling\n";
  }
  if (header_id == autofuse::GeneratedHeaderId::kApi) {
    if (has_global_tiling_data) {
      return "struct FinalTilingGroupSelection;\nstruct " + tiling_data_type_name +
             ";\nstruct AutofuseTilingDataPerf;\nbool GetWorkspaceSize(const " + tiling_data_type_name +
             " &tiling_data, uint64_t &workspace_size);\nnamespace optiling {\nstruct PgoTensorArgs;\nstruct "
             "SearchConfig;\n" +
             body + "}  // namespace optiling\n";
    }
    const std::string global_tiling_decl = is_autofuse ? "struct AutofuseTilingData;\n" : "";
    return "struct FinalTilingGroupSelection;\n" + global_tiling_decl +
           "struct AutofuseTilingDataPerf;\nnamespace optiling {\nstruct " + tiling_data_type_name +
           ";\nstruct PgoTensorArgs;\nstruct SearchConfig;\n" + body + "}  // namespace optiling\n";
  }
  std::string prefix;
  if (header_id == autofuse::GeneratedHeaderId::kSolver) {
    prefix = "using std::greater;\nusing std::string;\n";
  }
  return prefix + "namespace optiling {\n" + body + "}  // namespace optiling\n";
}
}  // namespace

void TilingCodeGenImpl::AddAtomicHeaderLine(autofuse::GeneratedHeaderId header_id, const std::string &line) {
  AppendAtomicHeaderBody(header_id, line + "\n");
  tiling_head_.AddLine(line);
}

void TilingCodeGenImpl::AppendAtomicHeaderBody(autofuse::GeneratedHeaderId header_id, const std::string &body) {
  atomic_headers_[header_id].body += body;
}

void TilingCodeGenImpl::ResetTranslationUnit() {
  tiling_func_.Reset();
  translation_unit_ = {};
}

void TilingCodeGenImpl::RequireTranslationUnitSystemHeader(const std::string &header) {
  autofuse::RequireSystemHeader(translation_unit_.dependencies, header);
}

void TilingCodeGenImpl::RequireVarRelationSystemHeaders() {
  RequireTranslationUnitSystemHeader("cmath");
  RequireTranslationUnitSystemHeader("limits");
}

void TilingCodeGenImpl::RequireTranslationUnitExternalHeader(const std::string &header) {
  autofuse::RequireExternalHeader(translation_unit_.dependencies, header);
}

void TilingCodeGenImpl::RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId header_id) {
  autofuse::RequireGeneratedHeader(translation_unit_.dependencies, header_id);
}

void TilingCodeGenImpl::RequireTranslationUnitTilingDataHeader() {
  if (config_.is_autofuse) {
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kTilingData);
    return;
  }
  RequireTranslationUnitExternalHeader(op_name_ + "_tiling_data.h");
}

void TilingCodeGenImpl::AddApiTilingPreambles() {
  const std::string kSystemIncludePrefix = "#include <";
  const std::string kExternalIncludePrefix = "#include \"";
  std::set<std::string> api_tiling_preambles;
  for (const auto &model_info : tiling_model_info_) {
    for (const auto &node_param : model_info.node_name_to_api_code) {
      api_tiling_preambles.insert(node_param.second.head_files);
    }
  }
  for (const auto &preamble : api_tiling_preambles) {
    std::istringstream headers(preamble);
    std::string line;
    while (std::getline(headers, line)) {
      if (line.rfind(kSystemIncludePrefix, 0U) == 0U && line.back() == '>') {
        std::string header = line.substr(kSystemIncludePrefix.size());
        header.pop_back();
        RequireTranslationUnitSystemHeader(header);
      } else if (line.rfind(kExternalIncludePrefix, 0U) == 0U && line.back() == '"') {
        std::string header = line.substr(kExternalIncludePrefix.size());
        header.pop_back();
        RequireTranslationUnitExternalHeader(header);
      } else {
        tiling_func_.AddLine(line);
      }
    }
  }
}

af::Status TilingCodeGenImpl::FinishTranslationUnit(const std::string &key,
                                                    std::map<std::string, std::string> &tiling_res) {
  translation_unit_.body = tiling_func_.GetOutputStr();
  std::string output;
  GE_ASSERT_SUCCESS(autofuse::RenderTranslationUnit(translation_unit_, output));
  tiling_res[key] += output;
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::FinishGeneratedHeaders(std::map<std::string, std::string> &tiling_res) {
  return FinishGeneratedHeaders(atomic_headers_, config_.tiling_data_type_name, config_.is_autofuse, tiling_res);
}

af::Status TilingCodeGenImpl::FinishGeneratedHeaders(
    const std::map<autofuse::GeneratedHeaderId, autofuse::GeneratedCode> &generated_headers,
    const std::string &tiling_data_type_name, bool is_autofuse, std::map<std::string, std::string> &tiling_res) {
  const std::array<autofuse::GeneratedHeaderId, 5U> header_order = {
      autofuse::GeneratedHeaderId::kState, autofuse::GeneratedHeaderId::kLog, autofuse::GeneratedHeaderId::kPgo,
      autofuse::GeneratedHeaderId::kSolver, autofuse::GeneratedHeaderId::kApi};
  for (const auto header_id : header_order) {
    const auto iter = generated_headers.find(header_id);
    if (iter == generated_headers.end() ||
        (header_id == autofuse::GeneratedHeaderId::kPgo && iter->second.body.empty())) {
      continue;
    }
    autofuse::GeneratedCode header = iter->second;
    header.body = WrapAtomicHeaderBody(header_id, header.body, tiling_data_type_name, is_autofuse);
    std::string output;
    GE_ASSERT_SUCCESS(autofuse::RenderGeneratedHeader(header, GetAtomicHeaderGuard(header_id), output));
    tiling_res[GetAtomicHeaderKey(header_id)] = std::move(output);
  }
  return af::SUCCESS;
}

inline void SetTilingDefinition(const std::set<std::string> &var_names, const std::string &param_name,
                                std::set<std::string> &tiling_data_vars,
                                std::map<std::string, std::string> &type_name_to_definition) {
  ge::CodePrinter dumper;
  if (TilingDataGenUtils::NeedWrittenTilingData(var_names, tiling_data_vars)) {
    TilingDataGenUtils::WriteTilingDataElement(dumper, tiling_data_vars, var_names);
    type_name_to_definition[param_name] += dumper.GetOutputStr();
  }
}

inline std::vector<std::string> GetVarsNames(const std::vector<Expr> &vars) {
  std::vector<std::string> var_names;
  for (const auto &var : vars) {
    var_names.emplace_back(Str(var));
  }
  return var_names;
}

inline std::vector<std::string> GetHardwareNames(const std::map<HardwareDef, Expr> &scopes) {
  std::vector<std::string> scope_names;
  for (const auto &scope : scopes) {
    scope_names.emplace_back(BaseTypeUtils::DumpHardware(scope.first));
  }
  return scope_names;
}

inline std::vector<std::string> GetConstVarNames(const ExprUintMap &const_vars) {
  std::vector<std::string> const_var_names;
  for (const auto &const_var : const_vars) {
    const_var_names.emplace_back(GetSymbolName(const_var.first));
  }
  return const_var_names;
}

inline std::string DumpTilingData(const std::map<std::string, std::string> &tiling_data_elements) {
  std::string tiling_data_def;
  for (auto &tiling : tiling_data_elements) {
    if (tiling.second.empty()) {
      continue;
    }
    tiling_data_def += "  // definitions of " + tiling.first + "\n";
    tiling_data_def += tiling.second + "\n";
  }
  return tiling_data_def;
}

inline std::string RemoveSpace(std::string str) {
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  return str;
}

af::Status TilingCodeGenImpl::GenCastReuseTilingDataCode(const ReuseScheduleGroupInfo &reuse_info,
                                                         const ReuseScheduleGroupInfo &info) {
  GE_ASSERT_TRUE(reuse_info.reuse_input_axes.size() == info.reuse_input_axes.size(),
                 "reuse input axes size is not equal: [%zu vs %zu]", reuse_info.reuse_input_axes.size(),
                 info.reuse_input_axes.size());
  GE_ASSERT_TRUE(reuse_info.reuse_search_axes.size() == info.reuse_search_axes.size(),
                 "reuse search axes size is not equal: [%zu vs %zu]", reuse_info.reuse_search_axes.size(),
                 info.reuse_search_axes.size());
  GE_ASSERT_TRUE(reuse_info.tiling_keys.size() == info.tiling_keys.size(),
                 "reuse_keys size is not equal to info, size: [%zu vs %zu]", reuse_info.tiling_keys.size(),
                 info.tiling_keys.size());
  for (size_t i = 0UL; i < reuse_info.reuse_input_axes.size(); i++) {
    // reuse轴是原始轴
    tiling_func_.AddLine("  reuse_tiling_data.set_" + reuse_info.reuse_input_axes[i] + "(tiling_data.get_" +
                         info.reuse_input_axes[i] + "());");
  }
  for (size_t i = 0UL; i < reuse_info.tiling_keys.size(); i++) {
    std::string judge_cond = "if (tiling_data.get_tiling_key() == " + std::to_string(info.tiling_keys[i]) + ") {";
    if (i != 0UL) {
      judge_cond = " else " + judge_cond;
    }
    tiling_func_.AddLine("  " + judge_cond);
    tiling_func_.AddLine("    reuse_tiling_data.set_tiling_key(" + std::to_string(reuse_info.tiling_keys[i]) + ");");
    tiling_func_.AddLine("  }");
  }
  return af::SUCCESS;
}

TilingCodeGenImpl::TilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                     const TilingModelInfo &tiling_model_info, const ScoreFuncs &score_funcs,
                                     const bool is_uniq_group)
    : op_name_(op_name),
      config_(config),
      tiling_data_manager_(tiling_model_info, extra_info_config_),
      extra_info_generator_(extra_info_config_, tiling_model_info, tiling_data_manager_),
      tiling_model_info_(tiling_model_info),
      is_uniq_group_(is_uniq_group),
      score_funcs_(score_funcs),
      operator_level_cache_gen_(std::make_unique<cache::OperatorLevelCacheGen>()),
      group_level_cache_gen_(std::make_unique<cache::GroupLevelCacheGen>()) {
  extra_info_config_.tiling_data_type_name = config_.tiling_data_type_name;
  if (config_.gen_extra_infos) {
    extra_info_config_.do_axes_calc = true;
    extra_info_config_.do_api_tiling = true;
  }

  // 读取编译态缓存配置
  const auto &att_config = AutoFuseConfig::GetAttStrategyConfig();
  const auto &pgo_config = AutoFuseConfig::GetPgoStrategyConfig();
  const bool pgo_enabled = pgo_config.set_env_enable_autofuse_pgo && (pgo_config.enable_autofuse_pgo == "true");
  const bool cache_enabled = config_.is_inductor_scene
                                 ? (!att_config.set_env_enable_tiling_cache || att_config.enable_tiling_cache == "true")
                                 : (att_config.enable_tiling_cache == "true");
  config_.cache_enabled_at_compile_time = (!config_.is_cube) && cache_enabled && !pgo_enabled;

  for (const auto &model_info : tiling_model_info) {
    const auto &hardware_cons = model_info.hardware_cons;
    if (hardware_cons.find(HardwareDef::UB) != hardware_cons.cend()) {
      hardware_has_ub_ = true;
      break;
    }
  }
  if (!config_.force_template_op_name.empty() && (config_.force_template_op_name != op_name_)) {
    config_.force_tiling_case.Clear();
    config_.force_schedule_result = -1L;
  }
  GELOGI("[DFX] Get tiling code gen config(%s)", config.Debug().c_str());
}

af::Status TilingCodeGenImpl::GetRelatedHardware(std::map<std::string, std::string> &hardware_info) {
  std::string cur_code;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
    for (const auto &scope : scope_names) {
      if (hardware_info.find(scope) != hardware_info.end()) {
        continue;
      }
      auto iter = kCoreMemsizeMap.find(scope);
      if (iter != kCoreMemsizeMap.end()) {
        cur_code.clear();
        cur_code += "  uint64_t " + scope + ";\n";
        cur_code +=
            "  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::" + iter->second + ", " + scope + ");";
        hardware_info[scope] = cur_code;
      }
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationCommonCode() {
  const auto duration_head_code = DurationGenHeadCode();
  if (!duration_head_code.empty()) {
    auto &log_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kLog].dependencies;
    for (const auto &header : {"cstdint", "memory", "string"}) {
      autofuse::RequireSystemHeader(log_dependencies, header);
    }
    for (const auto &header : {"chrono", "memory", "new", "string"}) {
      RequireTranslationUnitSystemHeader(header);
    }
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kLog, duration_head_code);
    tiling_func_.AddLine(DurationGenDefineCode());
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationPrintCode(const std::string &indent) {
  const auto duration_print_code = DurationPrintGenCode();
  if (!duration_print_code.empty()) {
    tiling_func_.AddLine(indent + duration_print_code);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationClearCode(const std::string &indent) {
  const auto duration_clear_code = DurationClearGenCode();
  if (!duration_clear_code.empty()) {
    tiling_func_.AddLine(indent + duration_clear_code);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenBaseTilingData(std::map<std::string, std::string> &type_name_to_definition) {
  std::set<std::string> tiling_data_vars;
  std::set<std::string> input_vars_set;
  std::set<std::string> searchable_vars_set;
  std::set<std::string> const_vars_set;
  std::set<std::string> hardware_vars_set;
  std::set<std::string> mem_vars_set;
  std::set<std::string> general_post_var_set;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = GetVarsNames(args_manager.GetInputVars());
    input_vars_set.insert(input_vars.begin(), input_vars.end());
    auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
    hardware_vars_set.insert(scope_names.begin(), scope_names.end());

    // 预留一个变量接口workspacesize，用于记录workspace信息
    std::string workspace_str = "workspaceSize";
    hardware_vars_set.insert(workspace_str);

    // const vars 如果没有出现在searchable vars中，那么归为InputParams。否则归为baseParams
    auto const_vars = GetConstVarNames(args_manager.GetConstVars());
    const_vars_set.insert(const_vars.begin(), const_vars.end());
    auto search_vars = GetVarsNames(args_manager.GetSearchableVars());
    searchable_vars_set.insert(search_vars.begin(), search_vars.end());
    const auto &post_datas = tiling_data_manager_.GetTilingDataWithAnnotation(
        model_info.tiling_case_id, TilingDataGenType::GENERAL_TILING_DATA_GEN);
    for (const auto &data_pair : post_datas) {
      general_post_var_set.insert(data_pair.first);
    }
    for (const auto &mem_pair :
         tiling_data_manager_.GetTilingDataWithAnnotation(TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
      mem_vars_set.insert(mem_pair.first);
    }
  }
  for (auto &const_var : const_vars_set) {
    if (hardware_vars_set.count(const_var) == 0u) {
      input_vars_set.insert(const_var);
    }
  }
  SetTilingDefinition(input_vars_set, "InputParams", tiling_data_vars, type_name_to_definition);
  SetTilingDefinition(hardware_vars_set, "HardWareParams", tiling_data_vars, type_name_to_definition);
  SetTilingDefinition(searchable_vars_set, "BaseParams", tiling_data_vars, type_name_to_definition);
  SetTilingDefinition(general_post_var_set, "GeneralParams", tiling_data_vars, type_name_to_definition);
  SetTilingDefinition(mem_vars_set, "MemoryParams", tiling_data_vars, type_name_to_definition);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHeaderCodesHead() {
  std::string op_spec = RemoveSpace(op_name_);
  std::transform(op_spec.begin(), op_spec.end(), op_spec.begin(), ::toupper);
  tiling_data_.AddLine("#ifndef ATT_TILING_DATA_" + op_spec + "_H_");
  tiling_data_.AddLine("#define ATT_TILING_DATA_" + op_spec + "_H_");
  GE_ASSERT_SUCCESS(GenHeaderInclude(), "Generate tiling data head failed.");
  tiling_data_.AddLine("namespace optiling {");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHeaderCodesTail() {
  tiling_data_.AddLine("REGISTER_TILING_DATA_CLASS(" + op_name_ + ", " + config_.tiling_data_type_name + ")");
  if (!config_.is_autofuse) {
    GenPgoHeaderCodesTail();
  }
  GE_ASSERT_SUCCESS(GenExternFuncDef(), "Generate extern func definition failed.");
  tiling_data_.AddLine("} // namespace optiling");
  if (!config_.is_autofuse) {
    tiling_data_.AddLine("using optiling::AutofuseTilingData;");
    tiling_data_.AddLine("using optiling::PgoTensorArgs;");
    tiling_data_.AddLine(
        "static bool GetWorkspaceSize(const AutofuseTilingData &tiling_data, uint64_t &workspace_size) "
        "{(void)tiling_data; workspace_size = 0; return true;}");
  }
  tiling_data_.AddLine("#endif");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHeaderCodesBody() {
  GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
  GE_ASSERT_SUCCESS(GenHeaderVarsDef(), "Generate vars definition failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHeaderVarsDef() {
  // 生成分的TilingData
  std::string tiling_data_def;
  std::map<std::string, std::string> base_tiling_data;
  GE_ASSERT_SUCCESS(GenBaseTilingData(base_tiling_data), "Generate base tiling data failed.");
  tiling_data_def += DumpTilingData(base_tiling_data);
  tiling_data_def += "\n";
  std::map<std::string, std::string> extra_tiling_data;
  if (config_.gen_extra_infos) {
    GE_ASSERT_SUCCESS(extra_info_generator_.GetExtraTilingDataDef(extra_tiling_data),
                      "Generate extra tiling data failed.");
  }
  tiling_data_def += DumpTilingData(extra_tiling_data);
  // 增加一个tiling_key的param
  ge::CodePrinter tiling_key_dumper;
  TilingDataGenUtils::AddElementDefinition(tiling_key_dumper, "uint32_t", "tiling_key");
  std::map<std::string, std::string> tiling_key_def = {{"TilingKeyParms", tiling_key_dumper.GetOutputStr()}};
  tiling_data_def += DumpTilingData(tiling_key_def) + "\n";
  std::string tiling_data_type_name;
  if (is_uniq_group_) {
    tiling_data_type_name = config_.tiling_data_type_name;
  } else {
    tiling_data_type_name = tiling_model_info_[0].schedule_group_ident.GetGroupPrefix() + "TilingData";
  }
  tiling_data_.AddLine(TilingDataGenUtils::StructElementDefine(tiling_data_type_name, tiling_data_def));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GetReuseVarNames(std::map<std::string, std::string> &var_names_to_reuse_var_name) {
  std::set<ReuseScheduleGroupPtr> reuse_schedule_groups;
  for (const auto &model_info : tiling_model_info_) {
    reuse_schedule_groups.insert(model_info.reuse_schedule_group);
  }
  for (const auto &reuse_schedule_group : reuse_schedule_groups) {
    GE_ASSERT_NOTNULL(reuse_schedule_group);
    // 要求reuse的axes和schedule的axes一一对应
    for (auto &reuse_schedule : reuse_schedule_group->schedule_group_to_info) {
      for (size_t axis_id = 0UL; axis_id < reuse_schedule_group->info.reuse_input_axes.size(); axis_id++) {
        const auto &axis_name = reuse_schedule.second.reuse_input_axes[axis_id];
        const auto &reuse_axis_name = reuse_schedule_group->info.reuse_input_axes[axis_id];
        if (axis_name != reuse_axis_name) {
          var_names_to_reuse_var_name[axis_name] = reuse_axis_name;
        }
      }
    }
    for (auto &reuse_schedule : reuse_schedule_group->schedule_group_to_info) {
      for (size_t axis_id = 0UL; axis_id < reuse_schedule_group->info.reuse_search_axes.size(); axis_id++) {
        const auto &axis_name = reuse_schedule.second.reuse_search_axes[axis_id];
        const auto &reuse_axis_name = reuse_schedule_group->info.reuse_search_axes[axis_id];
        if (axis_name != reuse_axis_name) {
          var_names_to_reuse_var_name[axis_name] = reuse_axis_name;
        }
      }
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::CollectStructCopyVars(std::set<std::string> &tiling_data_vars) {
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = GetVarsNames(args_manager.GetInputVars());
    tiling_data_vars.insert(input_vars.begin(), input_vars.end());
    auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
    tiling_data_vars.insert(scope_names.begin(), scope_names.end());
    auto search_vars = GetVarsNames(args_manager.GetSearchableVars());
    tiling_data_vars.insert(search_vars.begin(), search_vars.end());
    const auto &post_datas = tiling_data_manager_.GetTilingDataWithAnnotation(
        model_info.tiling_case_id, TilingDataGenType::GENERAL_TILING_DATA_GEN);
    for (const auto &data_pair : post_datas) {
      tiling_data_vars.insert(data_pair.first);
    }
    for (const auto &data_pair :
         tiling_data_manager_.GetTilingDataWithAnnotation(TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
      tiling_data_vars.insert(data_pair.first);
    }
    if (config_.gen_extra_infos) {
      std::set<std::string> extra_vars;
      auto extra_tiling_data_ret = extra_info_generator_.GetExtraTilingVars(model_info.tiling_case_id, extra_vars);
      if (extra_tiling_data_ret == af::SUCCESS) {
        tiling_data_vars.insert(extra_vars.begin(), extra_vars.end());
      }
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenStructCopyDef() {
  std::set<std::string> tiling_data_vars;
  // 获取所有的var name和复用var的映射关系
  std::map<std::string, std::string> var_names_to_reuse_var_name;
  GE_ASSERT_SUCCESS(GetReuseVarNames(var_names_to_reuse_var_name));
  GE_ASSERT_SUCCESS(CollectStructCopyVars(tiling_data_vars));
  tiling_data_vars.insert("tiling_key");
  tiling_data_vars.insert(BaseTypeUtils::DumpHardware(HardwareDef::CORENUM));
  if (config_.gen_extra_infos) {
    tiling_data_vars.insert("workspaceSize");
  }
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState, "struct TilingDataCopy {");
  for (const auto &var : tiling_data_vars) {
    // 如果没有复用，则定义该变量
    std::string reuse_var = var;
    const auto &iter = var_names_to_reuse_var_name.find(var);
    if (iter == var_names_to_reuse_var_name.end()) {
      AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState, "  uint32_t " + var + ";");
    } else {
      reuse_var = iter->second;
    }
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState,
                        "  void set_" + var + "(uint32_t val) { " + reuse_var + " = val; }");
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState,
                        "  inline uint32_t get_" + var + "() { return " + reuse_var + "; }");
  }
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState, "};");
  return af::SUCCESS;
}

size_t TilingCodeGenImpl::CollectInputVarsSize() const {
  std::set<std::string> visited_var_names;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = args_manager.GetInputVars();
    for (const auto &var : input_vars) {
      visited_var_names.insert(Str(var));
    }
  }
  return visited_var_names.size();
}

af::Status TilingCodeGenImpl::GenCacheHashMapDef() {
  // HashMap模板和常量可能在Operator缓存或Group缓存中使用
  // Operator缓存：由cache_enabled_at_compile_time控制
  // Group缓存：由with_reuse_info_控制
  GELOGI("Gen cache config=[cache_enabled_at_compile_time=%d, with_reuse_info=%d]",
         config_.cache_enabled_at_compile_time, with_reuse_info_);
  if (!config_.cache_enabled_at_compile_time && !with_reuse_info_) {
    return af::SUCCESS;
  }

  auto &state_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kState].dependencies;
  for (const auto &header : {"array", "cstddef", "cstdint", "memory", "type_traits"}) {
    autofuse::RequireSystemHeader(state_dependencies, header);
  }

  const size_t input_vars_size = CollectInputVarsSize();
  ge::CodePrinter state_header;

  // 生成常量定义（Operator和Group缓存共享）
  cache::OperatorLevelCacheGen::GenConstantDefs(state_header, input_vars_size);

  // 生成FixedSizeHashMap模板定义（Operator和Group缓存共享）
  GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenFixedSizeHashMapDef(state_header),
                    "Generate FixedSizeHashMap definition failed.");

  // 只在Operator缓存开启时生成OperatorCache相关类型和类
  if (config_.cache_enabled_at_compile_time) {
    GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenOperatorCacheTypes(state_header),
                      "Generate Operator cache types failed.");

    GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenTilingCacheContext(state_header),
                      "Generate TilingCacheContext failed.");
  }

  AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, state_header.GetOutputStr());
  tiling_head_.AddLine(state_header.GetOutputStr());

  return af::SUCCESS;
}

void TilingCodeGenImpl::GenPgoCallbackDefs(ge::CodePrinter &pgo_header) {
  pgo_header.AddLine("#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED");
  pgo_header.AddLine("#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED");
  pgo_header.AddLine("struct PgoTensorArgs {");
  pgo_header.AddLine("  void **inputs = nullptr;");
  pgo_header.AddLine("  uint32_t input_num = 0;");
  pgo_header.AddLine("  void **outputs = nullptr;");
  pgo_header.AddLine("  uint32_t output_num = 0;");
  pgo_header.AddLine("};");
  pgo_header.AddLine("#endif");
  pgo_header.AddLine("typedef long int (*ProfilingCallback)(" + GenPgoTensorArgsDef());
  pgo_header.AddLine("void* stream, uint64_t workspaceSize, " + config_.tiling_data_type_name +
                     "* tiling_data, double* cost_time);");
  pgo_header.AddLine("typedef long int (*ProfilingBatchCallback)(" + GenPgoTensorArgsDef());
  pgo_header.AddLine("void* stream, uint64_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);");
}

void TilingCodeGenImpl::GenPgoConfigDefs(ge::CodePrinter &pgo_header) {
  pgo_header.AddLine("#ifndef AUTOFUSE_PGO_CONFIG_DEFINED");
  pgo_header.AddLine("#define AUTOFUSE_PGO_CONFIG_DEFINED");
  pgo_header.AddLine("class PgoConfig {");
  pgo_header.AddLine("public:");
  pgo_header.AddLine("  static PgoConfig& Instance() {");
  pgo_header.AddLine("    static PgoConfig instance;");
  pgo_header.AddLine("    return instance;");
  pgo_header.AddLine("  }");
  pgo_header.AddLine("  void ResetRuntimeOverrides() {");
  pgo_header.AddLine("    need_change_solver_run = false;");
  pgo_header.AddLine("    pgo_threshold_index = 0;");
  pgo_header.AddLine("    pgo_ub_threshold_list = {0.2, 0.1, 0, 0.05, 0.1};");
  pgo_header.AddLine("    pgo_corenum_threshold_list = {0.4, 0.4, 1, 1, 0.8};");
  pgo_header.AddLine("  }");
  pgo_header.AddLine("  ProfilingCallback single_callback;");
  pgo_header.AddLine("  ProfilingBatchCallback batch_callback;");
  pgo_header.AddLine("  PgoTensorArgs *tensor_args = nullptr;");
  pgo_header.AddLine("  std::vector<AutofuseTilingDataPerf> *measured_candidates = nullptr;");
  pgo_header.AddLine("  void *stream = nullptr;");
  pgo_header.AddLine("  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num");
  pgo_header.AddLine("  bool need_change_solver_run = false;");
  pgo_header.AddLine("  size_t pgo_threshold_index = 0;");
  pgo_header.AddLine("  constexpr static size_t pgo_threshold_list_size = 5;");
  pgo_header.AddLine("  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};");
  pgo_header.AddLine("  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};");
  pgo_header.AddLine("private:");
  pgo_header.AddLine("  PgoConfig() = default;");
  pgo_header.AddLine("  ~PgoConfig() = default;");
  pgo_header.AddLine("  PgoConfig(const PgoConfig &) = delete;");
  pgo_header.AddLine("  PgoConfig &operator=(const PgoConfig &) = delete;");
  pgo_header.AddLine("};");
  pgo_header.AddLine("class PgoConfigRuntimeGuard {");
  pgo_header.AddLine("public:");
  pgo_header.AddLine("  PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }");
  pgo_header.AddLine("  ~PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }");
  pgo_header.AddLine("};");
  pgo_header.AddLine("#endif");
}

void TilingCodeGenImpl::GenPgoHeaderCodesTail() {
  if (!config_.is_autofuse) {
    tiling_data_.AddLine("using AutofuseTilingData =  " + config_.tiling_data_type_name + ";\n");
    const std::string pgo_perf_struct = {
        "struct AutofuseTilingDataPerf {\n"
        "  AutofuseTilingData tiling_data;\n"
        "  double best_perf;\n"
        "};\n"};
    tiling_data_.AddLine(pgo_perf_struct);
  }
  ge::CodePrinter pgo_header;
  GenPgoCallbackDefs(pgo_header);
  GenPgoConfigDefs(pgo_header);
  const std::string pgo_header_body = pgo_header.GetOutputStr();
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kPgo, pgo_header_body);
  }
  tiling_head_.AddLine(WrapAtomicHeaderBody(autofuse::GeneratedHeaderId::kPgo, pgo_header_body,
                                            config_.tiling_data_type_name, config_.is_autofuse));
  if (!config_.is_autofuse) {
    tiling_data_.AddLine(pgo_header_body);
  }
}

af::Status TilingCodeGenImpl::GenHeaderInclude() {
  tiling_data_.AddLine("#include <stdint.h>");
  tiling_data_.AddLine("#include <vector>");
  tiling_data_.AddLine("#include <unordered_map>");
  tiling_data_.AddLine("#include <array>");
  tiling_data_.AddLine("#include \"register/tilingdata_base.h\"");
  tiling_data_.AddLine("#include \"tiling/tiling_api.h\"");
  return af::SUCCESS;
}

std::vector<Expr> GetFromAxes(const Expr &hardware_arg, const ArgsManager &args_manager,
                              const HardwareDef &hardware_def) {
  std::vector<Expr> from_axes;
  if (hardware_def == HardwareDef::UB) {
    from_axes = args_manager.GetAncestor(hardware_arg);
  } else if (hardware_def == HardwareDef::CORENUM) {
    from_axes = args_manager.GetParentVars(hardware_arg);
  }
  return from_axes;
}

void TilingCodeGenImpl::InitTilingUpperBound(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager,
                                             const HardwareDef &hardware_def, std::map<std::string, bool> &visited) {
  auto input_vars = GetVarsNames(args_manager.GetInputVars());
  auto const_vars = GetConstVarNames(args_manager.GetConstVars());
  for (const auto &hardware_arg : hardware_args) {
    if ((std::find(input_vars.begin(), input_vars.end(), Str(hardware_arg)) != input_vars.end()) ||
        (std::find(const_vars.begin(), const_vars.end(), Str(hardware_arg)) != const_vars.end())) {
      continue;
    }
    if (visited.find(Str(hardware_arg) + "_upper_bound") != visited.end()) {
      continue;
    }
    tiling_func_.AddLine("    int32_t " + Str(hardware_arg) + "_upper_bound = 1;");
    visited.insert({Str(hardware_arg) + "_upper_bound", true});
    std::vector<Expr> from_axes = GetFromAxes(hardware_arg, args_manager, hardware_def);
    std::string hardware_arg_value = "";

    for (uint32_t i = 0u; i < from_axes.size(); ++i) {
      auto primary_args = from_axes[i].FreeSymbols();
      if (primary_args.empty() && from_axes[i].IsConstExpr()) {
        hardware_arg_value += "      " + Str(hardware_arg) + "_upper_bound *= " + Str(from_axes[i]) + ";\n";
        continue;
      }
      for (uint32_t j = 0u; j < primary_args.size(); ++j) {
        auto pri_arg = primary_args[j];
        if (visited.find(Str(pri_arg)) == visited.end()) {
          hardware_arg_value += "      double " + Str(pri_arg) + " = tiling_data.get_" + Str(pri_arg) + "();\n";
          visited.insert({Str(pri_arg), true});
        } else {
          hardware_arg_value += "    " + Str(pri_arg) + " = tiling_data.get_" + Str(pri_arg) + "();\n";
        }
      }
      hardware_arg_value += "    " + Str(hardware_arg) + "_upper_bound *= " + Str(from_axes[i]) + ";\n";
    }
    tiling_func_.AddLine(hardware_arg_value);
    tiling_func_.AddLine("    tiling_data.set_" + Str(hardware_arg) + "(" + Str(hardware_arg) + "_upper_bound);");
  }
}

std::set<std::string> GetConsRelatedAncestors(HardwareDef hardware_def, const ArgsManager &args_manager,
                                              const std::map<HardwareDef, Expr> &hardware_cons) {
  std::set<std::string> cons_related_ancestor_vars;
  for (const auto &pair : hardware_cons) {
    if (pair.first != hardware_def) {
      continue;
    }
    auto hardware_expr = pair.second;
    auto hardware_args = hardware_expr.FreeSymbols();
    for (const auto &hardware_arg : hardware_args) {
      auto ancestor_vars = args_manager.GetAncestorNames(hardware_arg);
      for (const auto &ancestor_var : ancestor_vars) {
        GELOGD("cons_related_ancestor_vars: %s", ancestor_var.c_str());
        cons_related_ancestor_vars.insert(ancestor_var);
      }
    }
    break;
  }
  return cons_related_ancestor_vars;
}

bool TilingCodeGenImpl::HitSmallShapePattern(ArgsManager &args_manager) const {
  auto hardware_cons = args_manager.GetTotalHardwareCons(false);
  // 仅对vv融合做优化，涉及cube类不做此优化，另外ub或者多核约束缺失的不做优化
  if ((hardware_cons.find(HardwareDef::UB) == hardware_cons.end()) ||
      (hardware_cons.find(HardwareDef::CORENUM) == hardware_cons.end()) ||
      (hardware_cons.find(HardwareDef::L1) != hardware_cons.end()) ||
      (hardware_cons.find(HardwareDef::L0A) != hardware_cons.end()) ||
      (hardware_cons.find(HardwareDef::L0B) != hardware_cons.end()) ||
      (hardware_cons.find(HardwareDef::L0C) != hardware_cons.end())) {
    GELOGD("HitSmallShapePattern: does not support this case");
    return false;
  }
  // 如果ub相关变量的原始轴和多核相关变量的原始轴不一致，那么不做优化
  // 例如： z0z1 is merged by [z0,z1], ub约束只和z1相关，和z0无关，z0z1用来做多核切分
  // 这个case里ub即使能全载，也不能直接返回，因为有一根轴z0和ub无关，但是和多核有关
  // 多核还有切分空间，不能直接返回1个核全载，否则会有严重的性能问题
  std::set<std::string> ub_cons_related_ancestor_vars =
      GetConsRelatedAncestors(HardwareDef::UB, args_manager, hardware_cons);
  std::set<std::string> mc_cons_related_ancestor_vars =
      GetConsRelatedAncestors(HardwareDef::CORENUM, args_manager, hardware_cons);
  if (mc_cons_related_ancestor_vars.empty() && ub_cons_related_ancestor_vars.empty()) {
    GELOGD("HitSmallShapePattern: ub_cons_related_ancestor_vars and mc_cons_related_ancestor_vars is empty");
    return false;
  }
  for (const auto &mc_var : mc_cons_related_ancestor_vars) {
    if (ub_cons_related_ancestor_vars.find(mc_var) == ub_cons_related_ancestor_vars.end()) {
      GELOGD("Cannot find ub_cons_related_ancestor_vars: %s", mc_var.c_str());
      return false;
    }
  }
  return true;
}

std::vector<Expr> TopoHardwareArgs(const Expr &hardware_arg, const ArgsManager &args_manager,
                                   const HardwareDef &hardware_def) {
  std::queue<Expr> expr_queue;
  std::vector<Expr> visited;
  expr_queue.push(hardware_arg);
  visited.emplace_back(hardware_arg);
  auto func = [&expr_queue, &visited](const std::vector<Expr> &primary_args) -> void {
    for (uint32_t j = 0u; j < primary_args.size(); ++j) {
      auto pri_arg = primary_args[j];
      if (std::find(visited.begin(), visited.end(), pri_arg) == visited.end()) {
        visited.emplace_back(pri_arg);
        expr_queue.push(pri_arg);
      }
    }
  };
  while (!expr_queue.empty()) {
    auto expr = expr_queue.front();
    expr_queue.pop();
    std::vector<Expr> from_axes = GetFromAxes(expr, args_manager, hardware_def);
    for (uint32_t i = 0u; i < from_axes.size(); ++i) {
      auto primary_args = from_axes[i].FreeSymbols();
      if (primary_args.empty() && from_axes[i].IsConstExpr()) {
        continue;
      }
      func(primary_args);
    }
  }
  return visited;
}

std::vector<Expr> ReorderHardwareArgs(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager,
                                      const HardwareDef &hardware_def) {
  auto input_vars = GetVarsNames(args_manager.GetInputVars());
  auto const_vars = GetConstVarNames(args_manager.GetConstVars());
  std::vector<Expr> reordered_hardware_args;
  for (const auto &hardware_arg : hardware_args) {
    if ((std::find(input_vars.begin(), input_vars.end(), Str(hardware_arg)) != input_vars.end()) ||
        (std::find(const_vars.begin(), const_vars.end(), Str(hardware_arg)) != const_vars.end())) {
      continue;
    }
    std::vector<Expr> sorted_args = TopoHardwareArgs(hardware_arg, args_manager, hardware_def);
    for (int32_t i = sorted_args.size() - 1; i >= 0; --i) {
      if (std::find(hardware_args.begin(), hardware_args.end(), sorted_args[i]) == hardware_args.end()) {
        continue;
      }
      if (std::find(reordered_hardware_args.begin(), reordered_hardware_args.end(), sorted_args[i]) ==
          reordered_hardware_args.end()) {
        reordered_hardware_args.emplace_back(sorted_args[i]);
      }
    }
  }
  return reordered_hardware_args;
}

// 小shape场景如果不做tiling，原始shape全载ub，ubsize不超过设置的ub阈值大小，那么直接返回不做tiling切分
af::Status TilingCodeGenImpl::GenSmallShapeTiling(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  if (!HitSmallShapePattern(args_manager)) {
    return af::SUCCESS;
  }
  tiling_func_.AddLine("  bool TrySmallShapeTiling(" + config_.tiling_data_type_name + " &tiling_data) {");
  std::map<std::string, bool> visited;
  for (const auto &pair : args_manager.GetTotalHardwareCons()) {
    if (pair.first == HardwareDef::UB) {
      auto hardware_expr = pair.second;
      auto hardware_args = hardware_expr.FreeSymbols();
      auto reordered_hardware_args = ReorderHardwareArgs(hardware_args, args_manager, pair.first);
      InitTilingUpperBound(reordered_hardware_args, args_manager, pair.first, visited);
      // 如果设置了阈值，那么ubsize不超过阈值*原始UB大小，否则ubsize不超过原始UB大小
      if (AutoFuseConfig::GetAttStrategyConfig().enable_multicore_ub_tradeoff != "true") {
        tiling_func_.AddLine(
            "    if ((Getub_size(tiling_data) < 0) || (tiling_data.get_ub_size() < "
            "static_cast<double>(Getub_size(tiling_data)))) {");
      } else {
        tiling_func_.AddLine("    if ((Getub_size(tiling_data) < 0) || (tiling_data.get_ub_size() * " +
                             std::to_string(config_.ub_threshold) +
                             " < static_cast<double>(Getub_size(tiling_data)))) {");
      }
      tiling_func_.AddLine("      return false;");
      tiling_func_.AddLine("    }");
    }
  }
  for (const auto &pair : args_manager.GetTotalHardwareCons()) {
    if (pair.first == HardwareDef::CORENUM) {
      auto hardware_expr = pair.second;
      auto hardware_args = hardware_expr.FreeSymbols();
      auto reordered_hardware_args = ReorderHardwareArgs(hardware_args, args_manager, pair.first);
      InitTilingUpperBound(reordered_hardware_args, args_manager, pair.first, visited);
    }
  }
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"TilingCaseId[" + std::to_string(model_info.tiling_case_id) +
                       "]Match small shape, apply small shape strategy.\");");
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTiling() {
  tiling_func_.AddLine(std::string("  bool GetTiling(") + config_.tiling_data_type_name + " &tiling_data" +
                       (hardware_has_ub_ ? ", double &cur_ub_ratio" : "") + ") {");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Execute DoTiling.\");");
  if (config_.enable_small_shape_strategy) {
    tiling_func_.AddLine("    if (!TrySmallShapeTiling(tiling_data)) {");
    tiling_func_.AddLine(
        "      OP_LOGD(OP_NAME, \"The shape does not match small shape pattern. Turn to main tiling procedure\");");
  }
  tiling_func_.AddLine("    if (!DoTiling(tiling_data)) {");
  tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Failed to do tiling.\");");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  if (config_.enable_small_shape_strategy) {
    tiling_func_.AddLine("    }");
  }
  tiling_func_.AddLine("    if (is_empty_tensor_) {");
  tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Empty tensor, skip DoApiTiling and GeneralTiling.\");");
  tiling_func_.AddLine("      return true;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    DoApiTiling(tiling_data);");
  tiling_func_.AddLine("    GeneralTiling(tiling_data);");
  tiling_func_.AddLine("    if (tiling_data.get_block_dim() > corenum_) {");
  tiling_func_.AddLine(
      "      OP_LOGW(OP_NAME, \"Generated block_dim %u exceeds core budget %u.\", "
      "tiling_data.get_block_dim(), corenum_);");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  if (config_.gen_extra_infos) {
    tiling_func_.AddLine("    GetWorkSpaceSize(tiling_data);");
    tiling_func_.AddLine("    ExtraTilingData(tiling_data);");
  }
  if (hardware_has_ub_) {
    tiling_func_.AddLine("    TilingSummary(tiling_data, cur_ub_ratio);");
  } else {
    tiling_func_.AddLine("    TilingSummary(tiling_data);");
  }
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenProtectedVars() {
  tiling_func_.AddLine("  uint32_t corenum_;");
  tiling_func_.AddLine("  bool is_empty_tensor_{false};");
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    tiling_func_.AddLine("  const SearchConfig *pending_search_cfg_{nullptr};");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingImplBaseClass() {
  std::string data_type = config_.tiling_data_type_name;
  // SearchConfig is defined in tiling_head_ (common header), no need to redefine here
  tiling_func_.AddLine("class TilingCaseImpl {");
  tiling_func_.AddLine(" public:");
  tiling_func_.AddLine("  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}");
  tiling_func_.AddLine("  virtual ~TilingCaseImpl() = default;");
  GE_ASSERT_SUCCESS(GenTilingImplPublicFunc(), "Generate get tiling failed.");
  tiling_func_.AddLine("  virtual void CollectPipeEstimates(" + data_type +
                       " &tiling_data, std::vector<FinalTilingPipeEstimate> &out) { (void)tiling_data; (void)out; }");
  tiling_func_.AddLine("  virtual const char *GetTemplateName() const { return \"\"; }");
  tiling_func_.AddLine("  virtual const char *GetSubCaseTag() const { return \"\"; }");
  tiling_func_.AddLine(" protected:");
  if (config_.enable_small_shape_strategy) {
    tiling_func_.AddLine("  virtual bool TrySmallShapeTiling(" + data_type + " &tiling_data) { return false;}");
  }
  tiling_func_.AddLine("  virtual bool DoTiling(" + data_type + " &tiling_data) = 0;");
  tiling_func_.AddLine("  virtual void DoApiTiling(" + data_type + " &tiling_data) { (void)tiling_data; }");
  tiling_func_.AddLine("  virtual void GeneralTiling(" + data_type + "& tiling_data) { (void)tiling_data; }");
  if (config_.gen_extra_infos) {
    tiling_func_.AddLine("  virtual void GetWorkSpaceSize(" + data_type + "& tiling_data) {}");
    tiling_func_.AddLine("  virtual void ExtraTilingData(" + data_type + " &tiling_data) {}");
  }
  GE_ASSERT_SUCCESS(GenProtectedVars(), "Generate protected vars failed.");
  tiling_func_.AddLine("};");
  tiling_func_.AddLine("using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenCommonStruct() {
  const std::string kPipeType = R"(
enum class PipeType : uint8_t {
  AIC_MTE1 = 0,
  AIC_MTE2,
  AIC_FIXPIPE,
  AIC_MAC,
  AIV_MTE2,
  AIV_MTE3,
  AIV_VEC,
  AICORE_MTE1,
  AICORE_MTE2,
  AICORE_MTE3,
  AICORE_CUBE,
  AICORE_VEC,
  ALL,
};
)";
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState, kPipeType);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenCommonFrameWork() {
  GE_ASSERT_SUCCESS(GenToolFuncs(), "Generate tool funcs failed.");
  GE_ASSERT_SUCCESS(GenCommonStruct());
  GE_ASSERT_TRUE(!tiling_model_info_.empty(), "Tiling model info should not be empty.");
  GE_ASSERT_SUCCESS(GenSolverBaseClass(), "Generate base class failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHardwareCons(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  for (const auto &pair : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
    auto iter = kHardwareNameMap.find(pair.first);
    if (iter == kHardwareNameMap.end()) {
      continue;
    }
    tiling_func_.AddLine("  int Get" + iter->second + "(" + config_.tiling_data_type_name + "& tiling_data) {");
    tiling_func_.AddLine(GenBufRelatedVars(pair.second, args_manager.GetContainerMap()));
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHardwareJudge(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  std::string name;
  std::string judge_code;
  std::set<std::string> related_vars;
  ExprExprMap param_map;
  for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
    bool related = false;
    name = BaseTypeUtils::DumpHardware(hardware.first);
    param_map.clear();
    GetRelatedInfo(args_manager, hardware.second, param_map, related);
    if (!related) {
      GELOGD("Size of param_map [%zu].", param_map.size());
      GELOGD("%s occupy is const, generating if codes.", name.c_str());
      GE_ASSERT_SUCCESS(UpdateRelatedVars(hardware.second, param_map, related_vars, 1U));
      for (const auto &pair : param_map) {
        judge_code += "    double " + Str(pair.first) + " = " + Str(pair.second) + ";\n";
      }
      std::string hardware_orig_expr = Str(hardware.second);
      judge_code.append("// ").append(name).append(" expr = ").append(hardware_orig_expr + "\n");
      Optimizer ast_optimizer;
      Parser parser(hardware_orig_expr);
      ASTPtr ast = parser.Parse();
      GE_ASSERT_NOTNULL(ast, "Parse expr failed: %s", hardware_orig_expr.c_str());
      ast_optimizer.Optimize(ast);
      std::string hardware_expr = ast_optimizer.RebuildExpr(*ast.get(), 1);
      judge_code.append(ast_optimizer.GenerateCode() + "\n");
      int64_t value = 0L;
      if (!hardware.second.IsConstExpr() || (!hardware.second.GetConstValue(value) || (value != 0UL))) {
        hardware_expr = hardware.second.IsConstExpr() ? hardware_expr.append("u") : hardware_expr;
        judge_code.append("    if (")
            .append(hardware_expr)
            .append(" > tiling_data.get_")
            .append(name)
            .append("()) {\n");
        judge_code.append("      OP_LOGW(OP_NAME, \"").append(name).append(" cons unsatisfied!\");\n");
        judge_code.append("      return false;\n");
        judge_code.append("    }\n");
      }
    }
  }
  for (const auto &arg : related_vars) {
    GELOGD("Add related vars [%s].", arg.c_str());
    tiling_func_.AddLine("    uint32_t " + arg + " = tiling_data.get_" + arg + "();");
  }
  tiling_func_.AddLine(judge_code);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHardwareSummary(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  std::string name;
  std::string param;
  std::string set_code;
  for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
    name = BaseTypeUtils::DumpHardware(hardware.first);
    set_code += " " + name + " = %u.";
    if (hardware.first == HardwareDef::CORENUM) {
      param += ", corenum_";
    } else {
      param += ", tiling_data.get_" + name + "()";
    }
  }
  tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"Set hardware params." + set_code + "\"" + param + ");");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenInputSummary(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  tiling_func_.AddLine(GenInputParamsPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), DLOG_INFO));
  tiling_func_.AddLine(GenConsExprPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), DLOG_INFO));
  for (const auto &arg : args_manager.GetSearchableVars()) {
    Expr min_expr = args_manager.GetMinValue(arg);
    Expr max_expr = args_manager.GetMaxValue(arg);
    if (min_expr.IsConstExpr() && max_expr.IsConstExpr()) {
      if (max_expr.GetExprType() == af::ExprType::kExprConstantRation) {
        GE_ASSERT_SUCCESS(IsUpperBoundValid<double>(min_expr, max_expr));
      } else {
        GE_ASSERT_SUCCESS(IsUpperBoundValid<uint64_t>(min_expr, max_expr));
      }
      GELOGD("Check upper bound %s and lower bound %s for %s", min_expr.Str().get(), max_expr.Str().get(),
             arg.Str().get());
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingSummary(const ModelInfo &model_info) {
  std::string codes;
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  std::string case_info_str = " in " + model_info.schedule_group_ident.GetItemPrefix() + "_" + model_info.sub_case_tag +
                              std::to_string(model_info.tiling_case_id);
  if (hardware_has_ub_) {
    tiling_func_.AddLine("  void TilingSummary(" + config_.tiling_data_type_name +
                         " &tiling_data, double& cur_ub_ratio) override {");
  } else {
    tiling_func_.AddLine("  void TilingSummary(" + config_.tiling_data_type_name + " &tiling_data) override {");
  }
  for (const auto &arg : args_manager.GetSearchableVars()) {
    tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + Str(arg) + " is %u" + case_info_str +
                         ".\", tiling_data.get_" + Str(arg) + "());");
  }
  for (const auto &pair : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
    const auto &arg_name = BaseTypeUtils::DumpHardware(pair.first);
    tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + arg_name + " is %d" + case_info_str +
                         ".\", Get" + arg_name + "(tiling_data));");
  }
  for (const auto &var : model_info.container_exprs) {
    tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + var.first + " is %u" + case_info_str +
                         ".\", tiling_data.get_" + var.first + "());");
  }
  GE_ASSERT_SUCCESS(GenExtraSummaryInfo(model_info, args_manager, case_info_str), "Generate summary info failed.");
  if (hardware_has_ub_) {
    tiling_func_.AddLine("    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data) - " +
                         Str(model_info.reserved_ub_size) + ") / tiling_data.get_ub_size();");
    tiling_func_.AddLine("    if (std::isnan(cur_ub_ratio)) {");
    tiling_func_.AddLine("      cur_ub_ratio = 1;");
    tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"The ub ratio is NaN, set it to 1.\");");
    tiling_func_.AddLine("    }");
  }
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPostTiling(const ModelInfo &model_info) {
  GE_ASSERT_SUCCESS(GenDoApiTiling(model_info), "Generate do api tiling failed.");
  GE_ASSERT_SUCCESS(GenGeneralTiling(model_info), "Generate get block num failed.");
  GE_ASSERT_SUCCESS(GenEvalFunc(model_info), "Generate eval funcs failed.");
  GE_ASSERT_SUCCESS(GenMemoryParamCode(model_info), "Gen Mem param code failed.");
  if (config_.gen_extra_infos) {
    GE_ASSERT_SUCCESS(GenExtraTilingData(model_info), "Generate extra tiling data failed.");
  }
  GE_ASSERT_SUCCESS(GenTilingSummary(model_info), "Generate tiling summary failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenImplPtr() {
  int idx = 0;
  tiling_func_.AddLine("TilingCaseImplPtr GetTilingImplPtr(uint32_t tiling_case_id, uint32_t corenum) {");
  tiling_func_.AddLine("  TilingCaseImplPtr tilingCaseImplPtr = nullptr;");
  // group_ident->case_id->sub_case_tags
  std::map<std::string, std::map<uint32_t, std::vector<std::string>>> tiling_case_id_map;
  for (const auto &model_info : tiling_model_info_) {
    tiling_case_id_map[model_info.schedule_group_ident.GetItemPrefix()][model_info.tiling_case_id].push_back(
        model_info.sub_case_tag);
  }
  for (const auto &model_info : tiling_model_info_) {
    const auto case_tags =
        tiling_case_id_map[model_info.schedule_group_ident.GetItemPrefix()][model_info.tiling_case_id];
    const auto force_sub_tag = config_.force_tiling_case.GetTag(model_info.schedule_group_ident.group_id);
    const bool is_exist_tag = std::find(case_tags.cbegin(), case_tags.cend(), force_sub_tag) != case_tags.cend();
    // 若对应tiling case中存在force_sub_tag，并且对应case不是该tag的话，则continue
    // 若不存在该force_case_tag，则继续生成该case
    if (is_exist_tag && (model_info.sub_case_tag != force_sub_tag)) {
      continue;
    }
    std::string tiling_id_str = std::to_string(model_info.tiling_case_id);
    if (idx == 0) {
      tiling_func_.AddLine("  if (tiling_case_id == " + tiling_id_str + "u) {");
    } else {
      tiling_func_.AddLine("  } else if (tiling_case_id == " + tiling_id_str + "u) {");
    }
    idx++;
    tiling_func_.AddLine("    tilingCaseImplPtr = std::make_shared<TilingCase" + model_info.sub_case_tag +
                         tiling_id_str + "Impl>(corenum);");
  }
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  return tilingCaseImplPtr;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  tiling_func_.AddLine("thread_local const char *g_final_selected_sub_case_tag = nullptr;");
  tiling_func_.AddLine("thread_local bool g_final_selected_sub_case_tag_valid = false;");
  tiling_func_.AddLine("");
  tiling_func_.AddLine("int32_t GetFinalTilingCaseId(uint32_t tiling_key) {");
  tiling_func_.AddLine("  switch (tiling_key) {");
  std::set<uint32_t> emitted_final_keys;
  for (const auto &model_info : tiling_model_info_) {
    if (!emitted_final_keys.insert(model_info.tiling_case_id).second) {
      continue;
    }
    tiling_func_.AddLine("    case " + std::to_string(model_info.tiling_case_id) + "u: return " +
                         std::to_string(model_info.tiling_case_id) + ";");
  }
  tiling_func_.AddLine("    default: return -1;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  tiling_func_.AddLine(
      "TilingCaseImplPtr GetSelectedTilingImplPtr(uint32_t tiling_case_id, uint32_t corenum, const char *sub_case_tag) "
      "{");
  tiling_func_.AddLine("  TilingCaseImplPtr tilingCaseImplPtr = nullptr;");
  bool first_selected_case = true;
  std::set<std::pair<uint32_t, std::string>> emitted_selected_cases;
  for (const auto &model_info : tiling_model_info_) {
    const auto selected_case = std::make_pair(model_info.tiling_case_id, model_info.sub_case_tag);
    if (!emitted_selected_cases.insert(selected_case).second) {
      continue;
    }
    const std::string tag = model_info.sub_case_tag;
    const std::string tag_literal = tag.empty() ? "" : tag;
    const std::string condition = "tiling_case_id == " + std::to_string(model_info.tiling_case_id) +
                                  "u && (sub_case_tag == nullptr || sub_case_tag[0] == '\\0' || "
                                  "std::string(sub_case_tag) == \"" +
                                  tag_literal + "\")";
    tiling_func_.AddLine(std::string("  ") + (first_selected_case ? "if (" : "else if (") + condition + ") {");
    tiling_func_.AddLine("    tilingCaseImplPtr = std::make_shared<TilingCase" + model_info.sub_case_tag +
                         std::to_string(model_info.tiling_case_id) + "Impl>(corenum);");
    tiling_func_.AddLine("  }");
    first_selected_case = false;
  }
  tiling_func_.AddLine("  return tilingCaseImplPtr;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  tiling_func_.AddLine("bool GetFinalTilingSelection(" + config_.tiling_data_type_name +
                       " &tiling_data, ::FinalTilingGroupSelection &selection) {");
  tiling_func_.AddLine("  const uint32_t tiling_key = tiling_data.get_tiling_key();");
  tiling_func_.AddLine("  if (!g_final_selected_sub_case_tag_valid) { return false; }");
  tiling_func_.AddLine("  const int32_t case_id = GetFinalTilingCaseId(tiling_key);");
  tiling_func_.AddLine("  if (case_id < 0) { return false; }");
  tiling_func_.AddLine(
      "  auto impl = GetSelectedTilingImplPtr(static_cast<uint32_t>(case_id), tiling_data.get_block_dim(), "
      "g_final_selected_sub_case_tag);");
  tiling_func_.AddLine("  if (impl == nullptr) { return false; }");
  tiling_func_.AddLine("  selection = ::FinalTilingGroupSelection{" +
                       std::to_string(tiling_model_info_[0].schedule_group_ident.group_id) +
                       "U, case_id, tiling_key, impl->CalcScore(tiling_data), impl->GetSubCaseTag(), "
                       "impl->GetTemplateName()};");
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  AddAtomicHeaderLine(
      autofuse::GeneratedHeaderId::kApi,
      "bool GetFinalTilingSelection(" + config_.tiling_data_type_name + " &, ::FinalTilingGroupSelection &);");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::CheckImplPtr(const std::string &indent) {
  tiling_func_.AddLine(indent + "if (tilingCaseImplPtr == nullptr) {");
  GE_ASSERT_SUCCESS(GenOpLog(indent + "  ", "Pointer for tiling_case_id is null."));
  tiling_func_.AddLine(indent + "  return false;");
  tiling_func_.AddLine(indent + "}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenOpLog(const std::string &indent, const std::string &log) {
  if (is_uniq_group_ && !config_.is_cube && !config_.enable_autofuse_pgo && !config_.is_inductor_scene) {
    tiling_func_.AddLine(indent + "OP_LOGE(OP_NAME, \"" + log + "\");");
  } else {
    tiling_func_.AddLine(indent + "OP_LOGW(OP_NAME, \"" + log + "\");");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenOpLog(const std::string &indent, const std::string &uniq_log,
                                       const std::string &sched_log) {
  if (is_uniq_group_) {
    tiling_func_.AddLine(indent + "OP_LOGI(OP_NAME, \"" + uniq_log + "\");");
  } else {
    tiling_func_.AddLine(indent + "OP_LOGI(OP_NAME, \"" + sched_log + "\");");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenIsStaticShape() {
  bool is_static_graph{true};
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    const auto &input_vars = args_manager.GetInputVars();
    for (const auto &input_var : input_vars) {
      if (HasSymbol(input_var)) {
        GELOGD("Got dynamic shape model as input var: %s has symbol.", af::SymbolicUtils::ToString(input_var).c_str());
        is_static_graph = false;
        break;
      }
    }
  }
  tiling_func_.AddLine(R"(extern "C" bool IsStaticShape() {)");
  std::string return_str("   return ");
  tiling_func_.AddLine(return_str + (is_static_graph ? "true" : "false") + ";");
  tiling_func_.AddLine("}");
  GELOGD("Gen IsStaticShape function success, is_static: %d", is_static_graph);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingImpl() {
  GE_ASSERT_SUCCESS(GenGetTilingWithCaseId());
  if (is_uniq_group_) {
    GE_ASSERT_SUCCESS(GenIsStaticShape());
    GE_ASSERT_SUCCESS(GenSingleGroupGetTilingCoreAlias());
  }
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingFuncCallEntrance() {
  GE_ASSERT_SUCCESS(GenGetTilingImpl(), "Generate context tiling impl failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationBeginCode(const TilingFuncDurationType type, const std::string &indent) {
  const auto duration_begin_code = DurationBeginGenCode(type);
  if (!duration_begin_code.empty()) {
    tiling_func_.AddLine(indent + duration_begin_code);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationEndCode(const TilingFuncDurationType type, const std::string &indent) {
  const auto duration_end_code = DurationEndGenCode(type);
  if (!duration_end_code.empty()) {
    tiling_func_.AddLine(indent + duration_end_code);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExternFuncDef() {
  tiling_data_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                       " &tiling_data, int32_t tiling_case_id = -1);");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExpressionMacro() {
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver,
                      "template <typename T, typename U> inline auto Max(T a, U b) { return static_cast<double>(a) > "
                      "static_cast<double>(b) ? a : b; }");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver,
                      "template <typename T, typename U> inline auto Min(T a, U b) { return static_cast<double>(a) < "
                      "static_cast<double>(b) ? a : b; }");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver,
                      "template <typename T> inline auto Abs(T a) { return static_cast<double>(a) >= 0 ? a : -a; }");
  const std::array<std::string, 11U> macros = {"#define Log(a) (log((double)(a)))",
                                               "#define Pow(a, b) pow(a, b)",
                                               "#define Rational(a, b) ((double)(a) / (double)(b))",
                                               "#define ExpectEq(a, b) ((a) == (b))",
                                               "#define ExpectNe(a, b) ((a) != (b))",
                                               "#define ExpectLe(a, b) ((a) <= (b))",
                                               "#define ExpectLt(a, b) ((a) < (b))",
                                               "#define LogicAnd(a, b) ((a) && (b))",
                                               "#define LogicOr(a, b) ((a) || (b))",
                                               "#define True true",
                                               "#define False false"};
  for (const auto &macro : macros) {
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver, macro);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenMacroInclude() {
  tiling_head_.AddLine("#include <cstdint>");
  tiling_head_.AddLine("#include <memory>");
  tiling_head_.AddLine("#include <cmath>");
  tiling_head_.AddLine("#include <cstdlib>");
  tiling_head_.AddLine("#include <memory.h>");
  tiling_head_.AddLine("#include <iostream>");
  tiling_head_.AddLine("#include <fstream>");
  tiling_head_.AddLine("#include <sstream>");
  tiling_head_.AddLine("#include <cstddef>");
  tiling_head_.AddLine("#include <iomanip>");
  tiling_head_.AddLine("#include <cfloat>");
  tiling_head_.AddLine("#include <algorithm>");
  tiling_head_.AddLine("#include <set>");
  tiling_head_.AddLine("#include <unordered_map>");
  tiling_head_.AddLine("#include <array>");
  tiling_head_.AddLine("#include <iterator>");
  tiling_head_.AddLine("#include <functional>");
  tiling_head_.AddLine("#include <chrono>");
  tiling_head_.AddLine("#include <cstdint>");
  tiling_head_.AddLine("#include <limits>");
  tiling_head_.AddLine("#include <string>");
  tiling_head_.AddLine("#include <vector>");
  std::set<std::string> uniq_head_files;
  for (const auto &model_info : tiling_model_info_) {
    for (const auto &node_param : model_info.node_name_to_api_code) {
      uniq_head_files.insert(node_param.second.head_files);
    }
  }
  for (const auto &head_file : uniq_head_files) {
    tiling_head_.AddLine(head_file);
  }
  GenLogDefine(tiling_head_, true);
  ge::CodePrinter log_header;
  GenLogDefine(log_header, false);
  AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kLog, log_header.GetOutputStr());
  if (config_.gen_tiling_data) {
    // 如果是自己生成的tilingdata定义，需要在实现里面include 该头文件
    tiling_head_.AddLine("#include \"" + op_name_ + "_tiling_data.h\"");
  }
  GenExpressionMacro();
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kSolver, "#define MAX_SOLUTION 50");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kLog, "#define OP_NAME \"" + op_name_ + "\"");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kLog,
                      final_tiling::GenFinalTilingLogHelpers(GotLogLevel() == DLOG_NULL));
  tiling_head_.AddLine("");
  GE_ASSERT_SUCCESS(GenDurationCommonCode(), "Generate duration common code failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenToolFuncs() {
  auto &solver_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kSolver].dependencies;
  for (const auto &header : {"cmath", "cstdint", "type_traits"}) {
    autofuse::RequireSystemHeader(solver_dependencies, header);
  }
  ge::CodePrinter solver_header;
  solver_header.AddLine("inline bool IsEqual(double a, double b)");
  solver_header.AddLine("{");
  solver_header.AddLine("    const double epsilon = 1e-8;");
  solver_header.AddLine("    double abs = (a > b) ? (a - b) : (b - a);");
  solver_header.AddLine("    return abs < epsilon;");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename T1, typename T2>");
  solver_header.AddLine("inline double TernaryOp(bool cond, T1 a, T2 b)");
  solver_header.AddLine("{");
  solver_header.AddLine("    return static_cast<double>(cond ? a : b);");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename T>");
  solver_header.AddLine("inline T Ceiling(T a)");
  solver_header.AddLine("{");
  solver_header.AddLine("    T value = static_cast<T>(static_cast<int64_t>(a));");
  solver_header.AddLine("    return (IsEqual(value, a)) ? value : (value + 1);");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename T>");
  solver_header.AddLine("inline T Floor(T a)");
  solver_header.AddLine("{");
  solver_header.AddLine("    return static_cast<T>(static_cast<int64_t>(a));");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename T1, typename T2>");
  solver_header.AddLine("inline auto Mod(T1 a, T2 b)->decltype(a % b)");
  solver_header.AddLine("{");
  solver_header.AddLine("    return a % b;");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename T1, typename T2>");
  solver_header.AddLine(
      "inline auto Mod(T1 a, T2 b)->typename std::enable_if<std::is_floating_point<T1>::value || "
      "std::is_floating_point<T2>::value, decltype(std::fmod(a, b))>::type");
  solver_header.AddLine("{");
  solver_header.AddLine("    return std::fmod(a, b);");
  solver_header.AddLine("}");
  solver_header.AddLine("template<typename TI, typename TO>");
  solver_header.AddLine("inline TO &RefToRef(TI &ptr) {");
  solver_header.AddLine("  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&ptr)));");
  solver_header.AddLine("}");
  solver_header.AddLine("");
  AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kSolver, solver_header.GetOutputStr());
  tiling_head_.AddLine(solver_header.GetOutputStr());
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingImplPublicFunc() {
  std::string data_type = config_.tiling_data_type_name;
  GE_ASSERT_SUCCESS(GenGetTiling(), "Generate get tiling failed.");
  tiling_func_.AddLine("  virtual double GetPerf(" + data_type + " &tiling_data) { (void)tiling_data; return 0.0; }");
  if (!is_uniq_group_) {
    tiling_func_.AddLine("  virtual const char* GetScheduleName() { return \"\"; }");
  }
  if (hardware_has_ub_) {
    tiling_func_.AddLine("  virtual void TilingSummary(" + data_type + " &tiling_data, double &cur_ub_ratio) = 0;");
  } else {
    tiling_func_.AddLine("  virtual void TilingSummary(" + data_type + " &tiling_data) = 0;");
  }
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    tiling_func_.AddLine("  virtual bool ExecutePGOSolver(" + data_type +
                         " &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* "
                         "autofuse_tiling_data, void* stream, " +
                         "std::unordered_map<int64_t, uint64_t> &workspace_map, " +
                         "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {");
    // Suppress unused-parameter warnings for all parameters in the base-class stub.
    {
      std::string void_casts = "    (void)tiling_data; (void)tiling_data_list; (void)autofuse_tiling_data; ";
      void_casts += "(void)stream; (void)workspace_map; (void)block_dim_vec; (void)search_cfg;";
      tiling_func_.AddLine(void_casts);
    }
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
  }
  tiling_func_.AddLine("  virtual int32_t CalcScore(const " + data_type +
                       " &tiling_data) { (void)tiling_data; return 0;}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenVirtualDataTransferFuncs() {
  const std::string &data_type = config_.tiling_data_type_name;
  tiling_func_.AddLine("  virtual void GetTilingData(TilingDataCopy &from_tiling, " + data_type +
                       " &to_tiling) { (void)from_tiling; (void)to_tiling; }");
  tiling_func_.AddLine("  virtual void SetTilingData(" + data_type +
                       " &from_tiling, TilingDataCopy &to_tiling) { (void)from_tiling; (void)to_tiling; }");
  tiling_func_.AddLine("  virtual bool SetWorkspaceSize(" + data_type +
                       " &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map)" +
                       " { (void)tiling_data; (void)workspace_map; return true; }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenVariableAnnotation(const ArgsManager &args_manager) {
  const std::string tiling_id = std::to_string(args_manager.GetTilingCaseId());
  std::string annotations;
  const auto variable_names = args_manager.GetContainerNames();
  if (config_.do_variable_replace && !variable_names.empty()) {
    annotations += " Tensor used for tiling case " + tiling_id + " is:\n";
    for (const auto &pair : variable_names) {
      annotations += "  " + Str(pair.first) + ":" + pair.second + "\n";
    }
  }
  // 输出性能相关变量（包括 _perf、_exe_time 和 _contrib 结尾的变量）
  if (const auto &ternary_ops = args_manager.GetTernaryOps(); !ternary_ops.empty()) {
    annotations += " Exe time & Perf time used for tiling case " + tiling_id + " is:\n";
    for (const auto &[fst, snd] : ternary_ops) {
      std::string variable_name = Str(fst);
      const bool is_perf = CheckPerf("_perf", variable_name);
      const bool is_exe_time = CheckPerf("_exe_time", variable_name);
      const bool is_contrib = CheckPerf("_contrib", variable_name);
      if (!is_perf && !is_exe_time && !is_contrib) {
        continue;
      }
      std::string display_name = variable_name;
      if (is_perf || is_contrib) {
        if (std::string desc = snd.GetDescription(); !desc.empty()) {
          display_name = desc;
        }
      }
      std::string var_preamble;
      std::string ternary_expr;
      snd.DecomposeNamedVars(variable_name, var_preamble, ternary_expr);
      if (std::string full_expr = var_preamble + ternary_expr; full_expr.length() <= kPerfAnnotationMaxExprLen) {
        annotations += "  " + display_name + ":" + ternary_expr + "\n";
      } else {
        // 对 preamble 每行增加 2 空格，与变量名标题行缩进区分
        std::string indented_preamble;
        std::istringstream ss(var_preamble);
        std::string line;
        while (std::getline(ss, line)) {
          indented_preamble += "  " + line + "\n";
        }
        annotations += "  " + display_name + ":\n" + indented_preamble;
        annotations += "    " + variable_name + " = " + ternary_expr + "\n";
      }
    }
  }
  AppendPerfBreakdownAnnotations(args_manager, tiling_id, annotations);
  if (!annotations.empty()) {
    tiling_func_.AddLine("/*");
    tiling_func_.AddLine(annotations);
    tiling_func_.AddLine("*/");
  }
  return af::SUCCESS;
}

std::string TilingCodeGenImpl::GenLaunchLikeInputOutputDef(bool is_define) {
  std::stringstream ss;
  std::string void_str = "";
  if (is_define) {
    void_str = "void* ";
  }
  int index = 0;
  for (auto input : tiling_model_info_[0].input_nodes) {
    ss << void_str << "input" << index++ << ", ";
  }
  index = 0;
  for (auto node : tiling_model_info_[0].output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << void_str << "output" << index++ << ", ";
    }
  }
  return ss.str();
}

std::string TilingCodeGenImpl::GenPgoTensorArgsDef(bool is_define) const {
  return is_define ? "PgoTensorArgs *tensor_args, " : "tensor_args, ";
}

void TilingCodeGenImpl::GenPGOMultiGroupBlockDimList(const FusedGraphNamespaceMap &namespace_map,
                                                     std::string &block_dim_list_arg) {
  if (config_.is_inductor_scene && !config_.is_cube) {
    block_dim_list_arg = "{}";
    return;
  }
  tiling_func_.AddLine("  std::vector<uint32_t*> multi_group_block_dim_list;");
  int block_dim_idx = 0;
  for (const auto &asc_graph_namespace_map : namespace_map) {
    for (auto &asc_graph_map_iter : asc_graph_namespace_map.second) {
      auto &asc_graph_map = asc_graph_map_iter.second;
      for (const auto &graph_info : asc_graph_map) {
        auto &graph_info_map = graph_info.second;
        std::string tiling_item_name = graph_info_map.second + "_tiling_data";
        std::string var_name = "block_dim_" + std::to_string(block_dim_idx++);
        tiling_func_.AddLine("  uint32_t " + var_name + " = tilingTmp." + tiling_item_name + ".get_block_dim();");
        tiling_func_.AddLine("  multi_group_block_dim_list.push_back(&" + var_name + ");");
      }
    }
  }
  block_dim_list_arg = "multi_group_block_dim_list";
}

af::Status TilingCodeGenImpl::GenTilingCaseImpl(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  std::string tiling_id = std::to_string(args_manager.GetTilingCaseId());
  GE_ASSERT_SUCCESS(GenVariableAnnotation(args_manager));
  tiling_func_.AddLine("class TilingCase" + model_info.sub_case_tag + tiling_id + "Impl : public TilingCaseImpl {");
  tiling_func_.AddLine(" public:");
  tiling_func_.AddLine("  TilingCase" + model_info.sub_case_tag + tiling_id +
                       "Impl(uint32_t corenum) : TilingCaseImpl(corenum) {\n");
  tiling_func_.AddLine("  }");
  const std::string template_name =
      model_info.schedule_group_ident.GetItemPrefix() + "_" + model_info.sub_case_tag + tiling_id;
  tiling_func_.AddLine("  const char *GetTemplateName() const override { return \"" + template_name + "\"; }");
  tiling_func_.AddLine("  const char *GetSubCaseTag() const override { return \"" + model_info.sub_case_tag + "\"; }");
  std::string tiling_data_key_word = "AutofuseTilingData";
  if (!is_uniq_group_) {
    std::string schedule_name_value = "\"" + model_info.schedule_group_ident.GetGroupPrefix() + "\"";
    tiling_func_.AddLine("  const char *GetScheduleName() override { return " + schedule_name_value + "; }");
    tiling_data_key_word = model_info.schedule_group_ident.GetGroupPrefix() + "TilingData";
  }
  tiling_func_.AddLine(" protected:");
  tiling_func_.AddLine(" std::unordered_map<std::string, std::vector<" + tiling_data_key_word + ">> filter_map{};");
  GE_ASSERT_SUCCESS(GenPreTiling(model_info), "Generate pretiling failed.");
  if (config_.enable_small_shape_strategy) {
    GE_ASSERT_SUCCESS(GenSmallShapeTiling(model_info), "Generate small shape tiling failed.");
  }
  GE_ASSERT_SUCCESS(GenDoTiling(model_info), "Generate dotiling failed.");
  GE_ASSERT_SUCCESS(GenPostTiling(model_info), "Generate posttiling failed.");
  if (config_.is_inductor_scene) {
    GenInductorExecutePGOSolver(model_info);
  }
  tiling_func_.AddLine("};");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenInductorExecutePGOSolver(const ModelInfo &model_info) {
  tiling_func_.AddLine("  bool ExecutePGOSolver(" + config_.tiling_data_type_name +
                       " &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* "
                       "autofuse_tiling_data, void* stream, "
                       "std::unordered_map<int64_t, uint64_t> &workspace_map, "
                       "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) override {");
  tiling_func_.AddLine("    (void)workspace_map;");
  tiling_func_.AddLine("    (void)block_dim_vec;");
  tiling_func_.AddLine("    (void)stream;");
  tiling_func_.AddLine("    pending_search_cfg_ = search_cfg;");
  tiling_func_.AddLine("    double cur_ub_ratio = -1.0;");
  tiling_func_.AddLine("    if (!GetTiling(tiling_data, cur_ub_ratio)) {");
  tiling_func_.AddLine("      pending_search_cfg_ = nullptr;");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  if (!is_uniq_group_) {
    std::string sub_field = model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data";
    tiling_func_.AddLine("    autofuse_tiling_data->" + sub_field + " = tiling_data;");
    tiling_func_.AddLine("    AutofuseTilingDataPerf tiling_perf;");
    tiling_func_.AddLine("    tiling_perf.tiling_data = *autofuse_tiling_data;");
  } else {
    tiling_func_.AddLine("    (void)autofuse_tiling_data;");
    tiling_func_.AddLine("    AutofuseTilingDataPerf tiling_perf;");
    tiling_func_.AddLine("    tiling_perf.tiling_data = tiling_data;");
  }
  tiling_func_.AddLine("    tiling_perf.best_perf = GetPerf(tiling_data);");
  tiling_func_.AddLine("    tiling_data_list.push_back(tiling_perf);");
  tiling_func_.AddLine("    pending_search_cfg_ = nullptr;");
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
}
af::Status TilingCodeGenImpl::GenPreTiling(const ModelInfo &model_info) {
  (void)model_info;
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDoApiTiling(const ModelInfo &model_info) {
  for (const auto &tiling_api_code : model_info.node_name_to_api_code) {
    tiling_func_.AddLine(tiling_api_code.second.function_impl);
    tiling_func_.AddLine("");
  }
  tiling_func_.AddLine("void DoApiTiling(" + config_.tiling_data_type_name + " &tiling_data) override {");
  if (model_info.node_name_to_api_code.empty()) {
    tiling_func_.AddLine("  (void)tiling_data;");
  }
  for (const auto &tiling_api_code : model_info.node_name_to_api_code) {
    tiling_func_.AddLine(tiling_api_code.second.function_invoke);
  }
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenMemoryParamCode(const ModelInfo &model_info) {
  std::string func_call_code;
  std::string func_define_code;
  std::set<std::string> var_names;
  for (const auto &line :
       tiling_data_manager_.GetTilingFuncImpl(model_info.tiling_case_id, TilingDataGenType::MEMORY_TILING_DATA_GEN)) {
    tiling_func_.AddLine(line);
  }
  tiling_func_.AddLine("  void ComputeMemoryParam(" + config_.tiling_data_type_name + " &tiling_data) {");
  tiling_func_.AddLine(
      tiling_data_manager_.GetTilingFuncInvoke(model_info.tiling_case_id, TilingDataGenType::MEMORY_TILING_DATA_GEN));
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraTilingFuncInvoke(const ModelInfo &model_info) {
  tiling_func_.AddLine(
      tiling_data_manager_.GetTilingFuncInvoke(model_info.tiling_case_id, TilingDataGenType::AXES_TILING_DATA_GEN));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGeneralTiling(const ModelInfo &model_info) {
  std::string impl_code;
  std::set<std::string> used_vars;
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  tiling_func_.AddLine("  void GeneralTiling(" + config_.tiling_data_type_name + " &tiling_data) override {");
  auto all_cons = args_manager.GetTotalHardwareCons(config_.do_variable_replace);
  if (all_cons.find(HardwareDef::CORENUM) != all_cons.end()) {
    auto expr = all_cons.at(HardwareDef::CORENUM);
    for (const auto &var : expr.FreeSymbols()) {
      if (!var.IsConstExpr()) {
        used_vars.insert(Str(var));
      }
    }
    impl_code += "    tiling_data.set_block_dim(Max(1, " + Str(expr) + "));";
    for (const auto &var : used_vars) {
      tiling_func_.AddLine("    double " + var + " = static_cast<double>(tiling_data.get_" + var + "());");
    }
    tiling_func_.AddLine(impl_code);
  } else {
    GELOGW("Did not apply block split.");
    tiling_func_.AddLine("    tiling_data.set_block_dim(1);");
  }
  tiling_func_.AddLine("    ComputeMemoryParam(tiling_data);");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraTilingData(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  GE_ASSERT_SUCCESS(GenExtraTilingFuncImpl(model_info), "Gen extra tiling func failed.");

  std::string param = config_.tiling_data_type_name + " &tiling_data";
  tiling_func_.AddLine("  void ExtraTilingData(" + param + ") {");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Start executing extra tiling for tiling_case_id " +
                       std::to_string(model_info.tiling_case_id) + ".\");");
  GE_ASSERT_SUCCESS(GenExtraTilingFuncInvoke(model_info), "Gen extra tiling invoke failed.");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Execute extra tiling for tiling_case_id " +
                       std::to_string(model_info.tiling_case_id) + " successfully.\");");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraEvalFunc(const ModelInfo &model_info) {
  GE_ASSERT_SUCCESS(GenPipeTypeObj(model_info), "Generate PipeTypeObj failed.");
  GE_ASSERT_SUCCESS(GenGetObj(model_info), "Generate GetObj failed.");
  GE_ASSERT_SUCCESS(GenCalcScore(model_info), "Generate CalcScore failed, graph name %s, tiling case %u",
                    model_info.graph_name.c_str(), model_info.tiling_case_id);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenEvalFunc(const ModelInfo &model_info) {
  GE_ASSERT_SUCCESS(GenHardwareCons(model_info), "Generate HardwareCons failed.");
  GE_ASSERT_SUCCESS(GenExtraEvalFunc(model_info), "Generate ExtraEvalFunc failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraTilingFuncImpl(const ModelInfo &model_info) {
  for (auto &axes_tiling_data_impl :
       tiling_data_manager_.GetTilingFuncImpl(model_info.tiling_case_id, TilingDataGenType::AXES_TILING_DATA_GEN)) {
    tiling_func_.AddLine(axes_tiling_data_impl);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPipeTypeObj(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  auto tiling_id_str = std::to_string(args_manager.GetTilingCaseId());
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  std::set<std::string> pipe_names;
  for (const auto &pair : args_manager.GetObjectFunc()) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter == kPipetypeNameMap.end()) {
      continue;
    }
    pipe_names.insert(iter->second);
    tiling_func_.AddLine("  double Get" + iter->second + "(" + config_.tiling_data_type_name + "& tiling_data) {");
    tiling_func_.AddLine(
        GenRelatedVars({pair.second}, args_manager.GetContainerMap(), args_manager.GetTernaryOpRelatedVars()));
    tiling_func_.AddLine("    return " + Str(pair.second.Replace(args_manager.GetTernaryOpReplaceVars())) + ";");
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("");
  }
  tiling_func_.AddLine("  void CollectPipeEstimates(" + config_.tiling_data_type_name +
                       " &tiling_data, std::vector<FinalTilingPipeEstimate> &out) override {");
  for (const auto &pipe_name : pipe_names) {
    tiling_func_.AddLine("    out.push_back(FinalTilingPipeEstimate{\"" + pipe_name + "\", Get" + pipe_name +
                         "(tiling_data), true});");
  }
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraSummaryInfo(const ModelInfo &model_info, const ArgsManager &args_manager,
                                                  std::string &case_info_str) {
  (void)model_info;
  for (const auto &pair : args_manager.GetObjectFunc()) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter != kPipetypeNameMap.end()) {
      tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The value of " + iter->second + " is %f" + case_info_str +
                           ".\", Get" + iter->second + "(tiling_data));");
    }
  }
  tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"[PROF]The objective value of the tiling data is %f" + case_info_str +
                       ".\", GetPerf(tiling_data));");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenScheduleGroupTilingHead() {
  if (config_.gen_tiling_data) {
    GE_ASSERT_SUCCESS(GenHeaderCodesBody(), "Generate tiling data head failed.");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenExtraParamCode(const ModelInfo &model_info, std::string &pass_code) {
  std::set<std::string> tiling_vars;
  auto extra_tiling_data_ret = extra_info_generator_.GetExtraTilingVars(model_info.tiling_case_id, tiling_vars);
  if (extra_tiling_data_ret == af::SUCCESS) {
    for (const auto &var : tiling_vars) {
      pass_code += "    to_tiling.set_" + var + "(from_tiling.get_" + var + "());\n";
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetObj(const ModelInfo &model_info) {
  Expr expression;
  std::vector<Expr> funcs;
  Expr expr;
  std::string codes;
  ArgsManager args_manager(model_info);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  Expr head_cost = args_manager.GetHeadCost();
  tiling_func_.AddLine("  double GetPerf(" + config_.tiling_data_type_name + "& tiling_data) override {");
  for (const auto &pair : args_manager.GetObjectFunc()) {
    auto iter = kPipetypeNameMap.find(pair.first);
    if (iter != kPipetypeNameMap.end()) {
      funcs.emplace_back(pair.second);
      expression = CreateExpr(iter->second.c_str());
      codes += "    double " + Str(expression) + " = " +
               Str(pair.second.Replace(args_manager.GetTernaryOpReplaceVars())) + ";\n";
      expr = (!IsValid(expr)) ? expression : af::sym::Max(expr, expression);
    }
  }
  funcs.emplace_back(head_cost);
  tiling_func_.AddLine(GenRelatedVars(funcs, args_manager.GetContainerMap(), args_manager.GetTernaryOpRelatedVars()));
  tiling_func_.AddLine(codes);
  if (!IsValid(expr)) {
    tiling_func_.AddLine("    return 0;");
  } else {
    expr = af::sym::Add(expr, head_cost);
    tiling_func_.AddLine("    return " + Str(expr) + ";");
  }
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenCalcScore(const ModelInfo &model_info) {
  if (!model_info.score_func.empty()) {
    tiling_func_.AddLine(model_info.score_func);
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetSetTilingImpl(const ModelInfo &model_info) {
  ArgsManager args_manager(model_info);
  args_manager.Process(false);
  std::string set_codes;
  std::string data_type = config_.tiling_data_type_name;
  for (const auto &arg : args_manager.GetInputVars()) {
    set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
  }
  for (const auto &arg : args_manager.GetSearchableVars()) {
    set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
  }
  for (const auto &var : model_info.container_exprs) {
    set_codes += "    to_tiling.set_" + var.first + "(from_tiling.get_" + var.first + "());\n";
  }
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  set_codes += "    to_tiling.set_" + core_num + "(from_tiling.get_" + core_num + "());\n";
  if (config_.gen_extra_infos) {
    std::string additional_code;
    GE_ASSERT_SUCCESS(GenExtraParamCode(model_info, additional_code), "Gen ExtraParamCode failed.");
    set_codes += additional_code;
  }
  set_codes += "    to_tiling.set_tiling_key(from_tiling.get_tiling_key());\n";

  tiling_func_.AddLine("  void GetTilingData(TilingDataCopy &from_tiling, " + data_type + " &to_tiling) override {");
  tiling_func_.AddLine(set_codes);
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  void SetTilingData(" + data_type + " &from_tiling, TilingDataCopy &to_tiling) override {");
  tiling_func_.AddLine(set_codes);
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  bool SetWorkspaceSize(" + data_type +
                       " &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map) override {");
  auto ws_vars = GenWorkspaceRelatedVars(model_info.workspace_size_map, args_manager.GetContainerMap());
  if (ws_vars.empty()) {
    tiling_func_.AddLine("    (void)tiling_data; (void)workspace_map;");
  } else {
    tiling_func_.AddLine(ws_vars);
  }
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenerateInputParamsAndTiling() {
  tiling_func_.AddLine("  } else {");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Calculating the tiling data for tiling_case_id %u.\", tiling_case_id);");
  tiling_func_.AddLine("    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_case_id, corenum);");
  GE_ASSERT_SUCCESS(CheckImplPtr("    "), "Generate implptr check failed!");
  if (is_uniq_group_) {
    GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    "),
                      "Generate begin code!");
  }
  tiling_func_.AddLine(std::string("    ret = tilingCaseImplPtr->GetTiling(tiling_data") +
                       (hardware_has_ub_ ? ", ub_ratio" : "") + ");");
  tiling_func_.AddLine("    tiling_data.set_tiling_key(tiling_case_id);");
  tiling_func_.AddLine(
      "    OP_LOGD(OP_NAME, \"Finish calculating the tiling data for tiling_case_id %u.\", tiling_case_id);");
  if (is_uniq_group_) {
    GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    "),
                      "Generate end code!");
  }
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDoTilingCommon(const ModelInfo &model_info,
                                                const std::pair<std::string, std::string> &codes) {
  tiling_func_.AddLine(codes.first);
  tiling_func_.AddLine("  bool DoTiling(" + config_.tiling_data_type_name + " &tiling_data) override {");
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenInputSummary(model_info),
                    "Generate input summary failed, group[%s], case[%u,%s].",
                    model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                    model_info.sub_case_tag.c_str());
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareSummary(model_info),
                    "Generate hardware summary failed, group[%s], case[%u,%s].",
                    model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                    model_info.sub_case_tag.c_str());
  GE_ASSERT_SUCCESS(TilingCodeGenImpl::GenHardwareJudge(model_info),
                    "Generate hardware judge failed, group[%s], case[%u,%s].",
                    model_info.schedule_group_ident.GetItemPrefix().c_str(), model_info.tiling_case_id,
                    model_info.sub_case_tag.c_str());
  tiling_func_.AddLine(codes.second);
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingDataFromCopy() {
  std::string set_codes;
  bool first_model_info = true;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    if (first_model_info) {
      for (const auto &arg : args_manager.GetInputVars()) {
        set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
      }
      first_model_info = false;
    }
    for (const auto &arg : args_manager.GetSearchableVars()) {
      set_codes += "    to_tiling.set_" + Str(arg) + "(from_tiling.get_" + Str(arg) + "());\n";
    }
    for (const auto &arg : model_info.container_exprs) {
      set_codes += "    to_tiling.set_" + arg.first + "(from_tiling.get_" + arg.first + "());\n";
    }
  }
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  set_codes += "    to_tiling.set_" + core_num + "(from_tiling.get_" + core_num + "());\n";
  set_codes += "    to_tiling.set_tiling_key(from_tiling.get_tiling_key());";
  std::string data_type = config_.tiling_data_type_name;
  tiling_func_.AddLine("void GetScheduleGroupTilingData(TilingDataCopy &from_tiling, " + data_type + " &to_tiling) {");
  tiling_func_.AddLine(set_codes);
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFindCacheAndSaveCache() {
  ge::CodePrinter state_header;
  auto &state_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kState].dependencies;
  for (const auto &header : {"array", "cstddef", "cstdint", "type_traits"}) {
    autofuse::RequireSystemHeader(state_dependencies, header);
  }
  // 当OperatorCache关闭但GroupCache开启时，需要在这里补齐GroupCache依赖的共享定义
  if (!config_.cache_enabled_at_compile_time) {
    cache::OperatorLevelCacheGen::GenConstantDefs(state_header, CollectInputVarsSize());
    GE_ASSERT_SUCCESS(operator_level_cache_gen_->GenFixedSizeHashMapDef(state_header),
                      "Generate FixedSizeHashMap definition for Group cache failed.");
  }

  GE_ASSERT_SUCCESS(group_level_cache_gen_->GenGroupCacheTypes(state_header, cache_capacity_),
                    "Generate Group cache types failed.");
  AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, state_header.GetOutputStr());
  tiling_head_.AddLine(state_header.GetOutputStr());

  GE_ASSERT_SUCCESS(group_level_cache_gen_->GenGroupCacheFunctions(tiling_func_, config_.tiling_data_type_name),
                    "Generate Group cache functions failed.");

  return af::SUCCESS;
}

void TilingCodeGenImpl::GenCalcScoreVarsDefine() {
  tiling_func_.AddLine(GenScoreTilingCaseStruct());
  // 根据 is_uniq_group_ 决定是否包含 workspace_map 参数，确保与 FindPerfBetterTilingbyCaseId 定义一致
  // core_num 始终作为函数参数（用于保持签名一致）
  std::string function_signature =
      std::string(
          "bool GetTilingCaseScoreFunc(const std::map<int32_t, std::vector<ScoreTilingCase>, greater<int32_t>> ") +
      "&score_map, double &obj, double &ub_ratio, TilingDataCopy &tmp_tiling, bool &sub_case_flag, " +
      "const char *&selected_sub_case_tag, " + config_.tiling_data_type_name + " &tiling_data" +
      (is_uniq_group_ ? "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map") + ", uint32_t core_num";
  const char_t *cache_args =
      with_reuse_info_ ? ", std::array<uint32_t, kInputShapeSize> &input_shapes, GroupLevelCache *cache = nullptr" : "";
  tiling_func_.AddLine(function_signature.append(cache_args).append(") {"));
  tiling_func_.AddLine(GenTilingScoreFuncDefineHead(is_uniq_group_));
  tiling_func_.AddLine("    if (ret) {");
  if (with_reuse_info_) {
    tiling_func_.AddLine("      if (cache != nullptr) {");
    tiling_func_.AddLine("        SaveGroupCache(input_shapes, tmp_tiling, *cache);");
    tiling_func_.AddLine("      }");
  }
  tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"[PROF]The score_map[%d] has been processed, tiling case %s%u of " +
                       tiling_model_info_[0].schedule_group_ident.GetItemPrefix() + " is the best choice.\",");
  tiling_func_.AddLine(R"(        s.first, sub_case_flag ? "R" : "", tiling_data.get_tiling_key());
      break;
    }
  }
)");
  GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    ");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
}

af::Status TilingCodeGenImpl::GenAllSameScoreTilingCases(
    std::map<std::string, std::vector<const ModelInfo *>> &same_args_name_to_graphs,
    const std::vector<std::string> &ordered_assemble_args_name) {
  bool is_first_same_args_graph = true;
  for (const auto &assemble_args_name : ordered_assemble_args_name) {
    auto &same_args_graphs = same_args_name_to_graphs[assemble_args_name];
    std::sort(same_args_graphs.begin(), same_args_graphs.end(), [](const ModelInfo *a, const ModelInfo *b) -> bool {
      return (a->tiling_case_id < b->tiling_case_id) || (a->sub_case_tag < b->sub_case_tag);
    });
    if (is_first_same_args_graph) {
      std::string kInitVar = R"(    TilingCaseImpl *tilingCaseImplPtr;
    TilingDataCopy tmp_tiling;
    int32_t score = 0;
    std::map<int32_t, std::vector<ScoreTilingCase>, greater<int32_t>> score_map;)";
      tiling_func_.AddLine(kInitVar);
      is_first_same_args_graph = false;
    } else {
      tiling_func_.AddLine("    score = 0;");
      tiling_func_.AddLine("    score_map.clear();");
    }
    for (const auto &models : same_args_graphs) {
      auto model_info = models;
      std::string case_id_str = model_info->sub_case_tag + std::to_string(model_info->tiling_case_id);
      tiling_func_.AddLine(std::string("    TilingCase")
                               .append(case_id_str)
                               .append("Impl case")
                               .append(case_id_str)
                               .append("(corenum);"));
      tiling_func_.AddLine("    tilingCaseImplPtr = &case" + case_id_str + ";");
      tiling_func_.AddLine("    score = tilingCaseImplPtr->CalcScore(tiling_data);");
      tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"tiling case" + case_id_str + " of " +
                           model_info->schedule_group_ident.GetGroupPrefixSnakeCase() + " score is %d\", score);");
      tiling_func_.AddLine("    score_map[score].emplace_back(\"" + model_info->sub_case_tag + "\", " +
                           std::to_string(model_info->tiling_case_id) + ", tilingCaseImplPtr);");
    }
    std::string call_str =
        "    ret |= GetTilingCaseScoreFunc(score_map, obj, ub_ratio, tmp_tiling, sub_case_flag, "
        "selected_sub_case_tag, tiling_data";
    std::string workspace_str = is_uniq_group_ ? "" : ", workspace_map";
    // 始终传递 corenum 参数给 GetTilingCaseScoreFunc（用于保持签名一致）
    std::string cache_str = with_reuse_info_ ? ", input_shapes, cache" : "";
    tiling_func_.AddLine(call_str.append(workspace_str).append(", corenum").append(cache_str).append(");"));
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGroupCacheLookupCode() {
  ArgsManager args_manager(tiling_model_info_[0]);
  GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
  auto input_vars = args_manager.GetInputVars();

  std::string input_shapes_init = "  std::array<uint32_t, kInputShapeSize> input_shapes = {";
  for (size_t i = 0; i < input_vars.size(); ++i) {
    if (i > 0) input_shapes_init += ", ";
    input_shapes_init += "tiling_data.get_" + Str(input_vars[i]) + "()";
  }
  input_shapes_init += "};";
  tiling_func_.AddLine(input_shapes_init);

  tiling_func_.AddLine("  if (cache != nullptr) {");
  tiling_func_.AddLine("    if (FindGroupCache(input_shapes, tiling_data, *cache)) {");
  tiling_func_.AddLine("      OP_LOGD(OP_NAME, \"" + config_.tiling_data_type_name + " find cache for this shape.\");");
  tiling_func_.AddLine("      if (::g_final_tiling_observe_enabled) { g_final_selected_sub_case_tag_valid = false; }");
  tiling_func_.AddLine("      return true;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"" + config_.tiling_data_type_name +
                       " find no cache, turn to main tiling procedure.\");");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTemplateIterationLogic() {
  tiling_func_.AddLine("    OP_LOGI(OP_NAME, \"The user didn't specify tiling_case_id, iterate all templates.\");");
  // 1.对所有model info内的切分轴按照字符顺序进行排序
  std::map<std::string, std::vector<std::string>> graph_name_to_arg_list;
  for (const auto &i : tiling_model_info_) {
    std::vector<AttAxisPtr> copy_args = i.arg_list;
    std::sort(copy_args.begin(), copy_args.end(),
              [](const AttAxisPtr &a, const AttAxisPtr &b) { return a->name < b->name; });
    for (const auto &arg : copy_args) {
      graph_name_to_arg_list[i.graph_name].emplace_back(arg->name);
    }
  }
  // 2.取所有model info判断切分轴名字是否一致
  std::map<std::string, std::vector<const ModelInfo *>> same_args_name_to_graphs;
  std::vector<std::string> ordered_assemble_args_name;
  for (const auto &i : tiling_model_info_) {
    auto args_name = graph_name_to_arg_list[i.graph_name];
    std::string assemble_args_name;
    if (!args_name.empty()) {
      for (const auto &arg : args_name) {
        assemble_args_name.append(arg).append(",");
      }
    }
    auto &same_args_name_to_graph = same_args_name_to_graphs[assemble_args_name];
    if (same_args_name_to_graph.empty()) {
      ordered_assemble_args_name.emplace_back(assemble_args_name);
    }
    same_args_name_to_graph.emplace_back(&i);
  }
  // 3.若切分轴一致，则将所有model info的切分轴加入同一个score_map，保证顺序为model info的顺序为tiling case id的顺序
  GE_ASSERT_SUCCESS(GenAllSameScoreTilingCases(same_args_name_to_graphs, ordered_assemble_args_name));
  tiling_func_.AddLine("    if (ret) {");
  tiling_func_.AddLine("      OP_LOGI(OP_NAME, \"[PROF]Among the templates, tiling case %s%u of " +
                       tiling_model_info_[0].schedule_group_ident.GetItemPrefix() +
                       R"( is the best choice.", sub_case_flag ? "R" : "", tiling_data.get_tiling_key());)");
  tiling_func_.AddLine("    }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingbyCaseId() {
  tiling_func_.AddLine("  if (tiling_case_id == -1) {");
  // Group级缓存查询
  if (with_reuse_info_) {
    GE_ASSERT_SUCCESS(GenGroupCacheLookupCode(), "Gen group cache lookup failed.");
  }
  GE_ASSERT_SUCCESS(GenTemplateIterationLogic(), "Gen template iteration failed.");
  GE_ASSERT_SUCCESS(GenerateInputParamsAndTiling(), "Gen GenerateInputParamsAndTiling failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGODefaultTiling() {
  tiling_func_.AddLine("    TilingDataCopy tmp_tiling;");
  tiling_func_.AddLine("    size_t malloc_size = 0;");
  for (const auto &model_info : tiling_model_info_) {
    tiling_func_.AddLine("    malloc_size = Max(malloc_size, sizeof(TilingCase" + model_info.sub_case_tag +
                         std::to_string(model_info.tiling_case_id) + "Impl));");
  }

  tiling_func_.AddLine("    void* memory = malloc(malloc_size);");
  tiling_func_.AddLine("    if (memory == nullptr) {");
  tiling_func_.AddLine(
      "      OP_LOGE(OP_NAME, \"Failed to allocate memory for tiling case, malloc_size = %zu.\", malloc_size);");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    TilingCaseImpl *tilingCaseImplPtr;");
  tiling_func_.AddLine("    double best_perf = DBL_MAX;");
  tiling_func_.AddLine("    double cur_perf = DBL_MAX;");
  tiling_func_.AddLine("    AutofuseTilingData autofuse_tiling_data_tmp;");
  tiling_func_.AddLine("    AutofuseTilingData autofuse_tiling_data_best = *output_tiling_data;");

  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOTilingCase(const ModelInfo &model_info) {
  std::string tiling_id_str = std::to_string(model_info.tiling_case_id);

  tiling_func_.AddLine("    tilingCaseImplPtr = new (memory) TilingCase" + model_info.sub_case_tag + tiling_id_str +
                       "Impl(corenum);");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Calculating the tiling data for tiling_case_id " +
                       model_info.sub_case_tag + tiling_id_str + ".\");");
  tiling_func_.AddLine("    autofuse_tiling_data_tmp = *output_tiling_data;");

  if (!is_uniq_group_) {
    tiling_func_.AddLine("    autofuse_tiling_data_tmp." + model_info.schedule_group_ident.GetItemPrefix() +
                         "_tiling_data.set_tiling_key(" + tiling_id_str + ");");
  } else {
    tiling_func_.AddLine("    autofuse_tiling_data_tmp.set_tiling_key(" + tiling_id_str + ");");
  }

  tiling_func_.AddLine("    ret = (SearchAllTilingbyCaseId(tilingCaseImplPtr, tiling_data, tiling_data_list, " +
                       tiling_id_str + "u, &autofuse_tiling_data_tmp, " +
                       "stream, workspace_map, block_dim_vec, effective_cfg) || ret);");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Finish calculating the tiling data for tiling_case_id " +
                       model_info.sub_case_tag + tiling_id_str + ".\");");
  tiling_func_.AddLine("    tilingCaseImplPtr->~TilingCaseImpl();");

  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenEnableGroupParallelPgoInvoke(const std::string &tiling_name, bool is_pointer,
                                                              const std::string &indent, std::string &invoke_code) {
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  std::string access;
  std::string obj_arg;
  if (is_pointer) {
    access = "->";
    obj_arg = std::to_string('*') + tiling_name;
  } else {
    access = ".";
    obj_arg = tiling_name;
  }

  std::stringstream ss;
  for (const auto &asc_graph_map_iter : namespace_map) {
    const auto &asc_graph_id = asc_graph_map_iter.first;
    const auto &asc_graph_namespace_map = asc_graph_map_iter.second;
    for (const auto &result_id_and_groups : asc_graph_namespace_map) {
      const auto &result_id = result_id_and_groups.first;
      if (enable_group_parallels_[asc_graph_id][result_id]) {
        ss << indent << "if (" << tiling_name << access << "get_graph" << asc_graph_id
           << "_tiling_key() == " << result_id << ") {" << std::endl;
        ss << indent << "  ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id << "(" << obj_arg
           << ", " << tiling_name << access << "get_block_dim());" << std::endl;
        ss << indent << "}" << std::endl;
      }
    }
  }
  invoke_code = ss.str();
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOGetTilingbyCaseId() {
  GE_ASSERT_SUCCESS(GenPGODefaultTiling(), "Gen default tiling failed.");

  for (const auto &model_info : tiling_model_info_) {
    std::string tiling_id_str = std::to_string(model_info.tiling_case_id);
    GE_ASSERT_SUCCESS(GenPGOTilingCase(model_info), "Gen tiling case %s failed.", tiling_id_str.c_str());
  }

  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenUpdateBetterTiling() {
  tiling_func_.AddLine("bool UpdateBetterTiling(TilingCaseImpl *tilingCaseImplPtr, TilingDataCopy &tmp_tiling, " +
                       config_.tiling_data_type_name + " &tiling_data" +
                       (is_uniq_group_ ? "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map") +
                       ", uint32_t tiling_case_id, const char *sub_case_tag, const char *&selected_sub_case_tag) {");
  tiling_func_.AddLine(
      "  OP_LOGD(OP_NAME, \"The solution for tiling_case_id %u is better, updating the tiling data.\", "
      "tiling_case_id);");
  tiling_func_.AddLine("  tiling_data.set_tiling_key(tiling_case_id);");
  tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
  if (!is_uniq_group_) {
    tiling_func_.AddLine("  if (!tilingCaseImplPtr->SetWorkspaceSize(tiling_data, workspace_map)) {");
    tiling_func_.AddLine("    OP_LOGW(OP_NAME, \"Failed to set workspace size.\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
  }
  tiling_func_.AddLine("  selected_sub_case_tag = sub_case_tag;");
  tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"Set the output tiling data.\");");
  tiling_func_.AddLine("  OP_LOGD(OP_NAME, \"Updated the best tiling_case_id to %u.\", tiling_case_id);");
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

// 获取当前 ScheduleResult 的 Group 数量，用于并发性能优化
uint32_t TilingCodeGenImpl::GetGroupNumForCurrentScheduleResult(
    const std::pair<size_t, size_t> &schedule_result_key) const {
  auto it = schedule_result_group_nums_.find(schedule_result_key);
  if (it != schedule_result_group_nums_.end()) {
    return static_cast<uint32_t>(it->second);
  }
  return 1;  // 默认单 Group
}
af::Status TilingCodeGenImpl::GenSelectBetterTilingBasedOnObjAndUbRatio() {
  double ub_threshold_perf_val_effect = 0.0;
  double perf_effect_val = 0.0;
  GE_ASSERT_TRUE(!tiling_model_info_.empty());
  if (tiling_model_info_[0].tiling_schedule_config_table != nullptr) {
    ub_threshold_perf_val_effect = tiling_model_info_[0].tiling_schedule_config_table->GetUbThresholdPerfValEffect();
    perf_effect_val = tiling_model_info_[0].tiling_schedule_config_table->GetPerfEffectVal();
    std::string perf_effect_val_str = std::to_string(perf_effect_val);
    tiling_func_.AddLine("    if (obj < 0) {");
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("      return true;");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine(
        "    double ub_ratio_diff = cur_ub_ratio > ub_ratio ? (cur_ub_ratio - ub_ratio) : (ub_ratio - cur_ub_ratio);");
    tiling_func_.AddLine("    if ((cur_obj - obj > " + perf_effect_val_str + ")) {\n");
    tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
    tiling_func_.AddLine("    } else if ((obj - cur_obj > " + perf_effect_val_str + ")) {");
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("    } else if (cur_ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                         " && ub_ratio >= " + std::to_string(ub_threshold_perf_val_effect) + ") {");
    tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
    tiling_func_.AddLine("    } else if (cur_ub_ratio >= " + std::to_string(ub_threshold_perf_val_effect) +
                         " && ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) + ") {");
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("    } else if (cur_ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                         " && ub_ratio < " + std::to_string(ub_threshold_perf_val_effect) +
                         " && !IsEqual(cur_ub_ratio, ub_ratio)) {");
    tiling_func_.AddLine("        if (cur_ub_ratio > ub_ratio) {");
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("        } else {");
    tiling_func_.AddLine("          tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
    tiling_func_.AddLine("        }");
    tiling_func_.AddLine("    } else {");
    tiling_func_.AddLine("      if (cur_obj < obj) {");
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("      } else {");
    tiling_func_.AddLine("        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
    tiling_func_.AddLine("      }");
    tiling_func_.AddLine("    }");
  } else {
    tiling_func_.AddLine(GenCallUpdateBetterTiling(is_uniq_group_, hardware_has_ub_));
    tiling_func_.AddLine("    sub_case_flag = is_sub_case;");
    tiling_func_.AddLine("    obj = cur_obj;");
    tiling_func_.AddLine("    return true;");
  }
  return af::SUCCESS;
}

// 辅助函数：生成性能调整代码
std::string TilingCodeGenImpl::GenPerformanceAdjustmentCode(bool enable_group_parallel_optimize,
                                                            bool add_core_num_param, uint32_t group_num,
                                                            bool is_uniq_group) {
  if (!enable_group_parallel_optimize) {
    return "";
  }

  std::string code;
  code += "    const auto org_cur_obj = cur_obj;\n";
  code += "    constexpr uint32_t group_num = " + std::to_string(group_num) + ";  // 编译时生成\n";

  if (add_core_num_param) {
    code += "    const double core_ratio = (double)tiling_data.get_block_dim() / (double)core_num;\n";
    code += "    cur_obj = cur_obj / 100.0 * group_num * core_ratio;\n";
    code +=
        "    OP_LOGD(OP_NAME, \"The optimal objective for tiling_case_id %u of %s is %lf(original obj is %lf), "
        "group_num is %u, limited core num is %u, used core num is %u.\",\n";
    if (!is_uniq_group) {
      code +=
          "      tiling_case_id, schedule_name, cur_obj, org_cur_obj, group_num, core_num, "
          "tiling_data.get_block_dim());";
    } else {
      code += "      tiling_case_id, \"\", cur_obj, org_cur_obj, group_num, core_num, tiling_data.get_block_dim());";
    }
  }
  // 当 add_core_num_param 为 false 时，不进行性能调整

  return code;
}

// 辅助函数：生成有UB情况下的日志代码
std::string TilingCodeGenImpl::GenLogOutputCodeWithUb(const bool is_uniq_group) {
  if (!is_uniq_group) {
    return "    OP_LOGD(OP_NAME, \"The ub ratio for tiling_case_id %u of %s is %f.\", tiling_case_id, schedule_name, "
           "cur_ub_ratio);\n"
           "    OP_LOGD(OP_NAME, \"The optimal objective for tiling_case_id %u of %s is %f.\", tiling_case_id, "
           "schedule_name, cur_obj);";
  } else {
    return "    OP_LOGD(OP_NAME, \"The ub ratio for tiling_case_id %u is %f.\", tiling_case_id, cur_ub_ratio);\n"
           "    OP_LOGD(OP_NAME, \"The optimal objective for tiling_case_id %u is %f.\", tiling_case_id, cur_obj);";
  }
}

// 辅助函数：生成有UB情况下的主体代码
af::Status TilingCodeGenImpl::GenFindPerfBetterTilingbyCaseIdWithUb(bool enable_group_parallel_optimize,
                                                                    bool add_core_num_param, uint32_t group_num,
                                                                    bool is_uniq_group) {
  GE_ASSERT_SUCCESS(CheckImplPtr("  "), "Generate implptr check failed!");
  tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
  tiling_func_.AddLine(std::string("  if (tilingCaseImplPtr->GetTiling(tiling_data, cur_ub_ratio)) {"));
  tiling_func_.AddLine("    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);");

  if (!is_uniq_group) {
    tiling_func_.AddLine("    const char *schedule_name = tilingCaseImplPtr->GetScheduleName();");
  }

  std::string perf_code =
      GenPerformanceAdjustmentCode(enable_group_parallel_optimize, add_core_num_param, group_num, is_uniq_group);
  if (!perf_code.empty()) {
    tiling_func_.AddLine(perf_code);
  }

  tiling_func_.AddLine(GenLogOutputCodeWithUb(is_uniq_group));
  GenSelectBetterTilingBasedOnObjAndUbRatio();

  return af::SUCCESS;
}

// 辅助函数：生成无UB情况下的主体代码
af::Status TilingCodeGenImpl::GenFindPerfBetterTilingbyCaseIdWithoutUb(bool enable_group_parallel_optimize,
                                                                       bool add_core_num_param, uint32_t group_num) {
  GE_ASSERT_SUCCESS(CheckImplPtr("  "), "Generate implptr check failed!");
  tiling_func_.AddLine("  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);");
  tiling_func_.AddLine(std::string("  if (tilingCaseImplPtr->GetTiling(tiling_data)) {"));
  tiling_func_.AddLine("    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);");

  if (enable_group_parallel_optimize) {
    tiling_func_.AddLine("    const auto org_cur_obj = cur_obj;");
    tiling_func_.AddLine("    constexpr uint32_t group_num = " + std::to_string(group_num) + ";  // 编译时生成");
    if (add_core_num_param) {
      tiling_func_.AddLine("    const double core_ratio = (double)tiling_data.block_dim_ / (double)core_num;");
      tiling_func_.AddLine("    cur_obj = cur_obj / 100.0 * group_num * core_ratio;");
      tiling_func_.AddLine(
          "    OP_LOGD(OP_NAME, \"The optimal objective for tiling_case_id %u is %lf(original obj is %lf), "
          "group_num is %u, limited core num is %u, used core num is %u.\",");
      tiling_func_.AddLine("      tiling_case_id, cur_obj, org_cur_obj, group_num, core_num, tiling_data.block_dim_);");
    }
    // 当 add_core_num_param 为 false 时，不进行性能调整
  }

  tiling_func_.AddLine(
      "    OP_LOGD(OP_NAME, \"The optimal objective for tiling_case_id %u is %f.\", tiling_case_id, cur_obj);");
  tiling_func_.AddLine("    if (obj < 0 || cur_obj < obj) {");
  const std::string workspace_arg = is_uniq_group_ ? "" : ", workspace_map";
  tiling_func_.AddLine("      if (!UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data" + workspace_arg +
                       ", tiling_case_id, sub_case_tag, selected_sub_case_tag)) {");
  tiling_func_.AddLine("        return false;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("      sub_case_flag = is_sub_case;");
  tiling_func_.AddLine("      obj = cur_obj;");
  tiling_func_.AddLine("    } else {");
  tiling_func_.AddLine("      tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
  tiling_func_.AddLine("    }");

  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFindPerfBetterTilingbyCaseId(bool enable_group_parallel_optimize,
                                                              bool add_core_num_param, uint32_t group_num) {
  // core_num 始终作为函数参数（用于保持签名一致），add_core_num_param 只控制是否使用它进行性能调整
  std::string core_num_param = ", uint32_t core_num";
  // 根据 is_uniq_group_ 决定是否包含 workspace_map 参数，确保与 GetTilingCaseScoreFunc 和调用代码一致
  tiling_func_.AddLine(
      "bool FindPerfBetterTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, double &obj, double &ub_ratio, "
      "TilingDataCopy &tmp_tiling, " +
      config_.tiling_data_type_name + " &tiling_data, " +
      (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
      "uint32_t tiling_case_id, bool is_sub_case, bool &sub_case_flag" + core_num_param +
      ", const char *sub_case_tag, const char *&selected_sub_case_tag) {");
  if (!add_core_num_param) {
    tiling_func_.AddLine("  (void)core_num;");
  }
  tiling_func_.AddLine("  double cur_obj;");
  if (hardware_has_ub_) {
    tiling_func_.AddLine("  double cur_ub_ratio;");
    GE_ASSERT_SUCCESS(GenFindPerfBetterTilingbyCaseIdWithUb(enable_group_parallel_optimize, add_core_num_param,
                                                            group_num, is_uniq_group_),
                      "Gen FindPerfBetterTilingbyCaseId with ub failed.");
  } else {
    GE_ASSERT_SUCCESS(
        GenFindPerfBetterTilingbyCaseIdWithoutUb(enable_group_parallel_optimize, add_core_num_param, group_num),
        "Gen FindPerfBetterTilingbyCaseId without ub failed.");
  }
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  } else {");
  tiling_func_.AddLine("    tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  return false;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenSearchAllTilingbyCaseId() {
  tiling_func_.AddLine("bool SearchAllTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, " +
                       config_.tiling_data_type_name + " &tiling_data" +
                       ", std::vector<AutofuseTilingDataPerf>& tiling_data_list" +
                       ", uint32_t tiling_case_id, AutofuseTilingData* output_tiling_data, void* stream, " +
                       "std::unordered_map<int64_t, uint64_t> &workspace_map, " +
                       "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {");

  tiling_func_.AddLine("    tiling_data.set_tiling_key(tiling_case_id);");
  tiling_func_.AddLine(
      "    if (!tilingCaseImplPtr->ExecutePGOSolver(tiling_data, tiling_data_list, output_tiling_data, "
      "stream, workspace_map, block_dim_vec, search_cfg)) {");
  tiling_func_.AddLine(
      "      OP_LOGW(OP_NAME, \"Failed to execute PGO solver for tiling_case_id %d .\", tiling_case_id);");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine(
      "    OP_LOGD(OP_NAME, \"Execute PGO solver for tiling_case_id %d successfully.\", tiling_case_id);");
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::ValidateSingleResultAndGroup(bool need_free_memory) {
  tiling_func_.AddLine("  if (!ret) {");
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    int32_t log_level = (is_uniq_group_ && !config_.is_cube) ? DLOG_ERROR : DLOG_INFO;
    tiling_func_.AddLine(
        GenInputParamsPrint(args_manager, model_info.schedule_group_ident.GetGroupPrefix(), log_level));
  }
  GE_ASSERT_SUCCESS(GenOpLog("    ", "Failed to execute tiling func."));
  tiling_func_.AddLine("  }");
  // Issue #274: release memory allocated in GenPGODefaultTiling only in PGO/inductor function scope
  // 仅在 PGO 路径（GenPGOSearchTilingKey）内调用时传入 true，因为 memory 变量仅在该函数体内声明
  if (need_free_memory) {
    tiling_func_.AddLine("  free(memory);");
  }
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingKey() {
  if (with_reuse_info_) {
    GE_ASSERT_SUCCESS(GenGetTilingDataFromCopy(), "Gen GetTilingDataFromCopy failed.");
    GE_ASSERT_SUCCESS(GenFindCacheAndSaveCache(), "Gen FindCacheAndSaveCache failed.");
  }
  GE_ASSERT_SUCCESS(GenUpdateBetterTiling(), "Gen UpdateBetterTiling failed.");

  // 获取当前 Group 数量，判断是否启用并发性能优化
  auto schedule_result_key = std::make_pair(tiling_model_info_[0].schedule_group_ident.asc_graph_id,
                                            tiling_model_info_[0].schedule_group_ident.impl_graph_id);
  const uint32_t group_num = GetGroupNumForCurrentScheduleResult(schedule_result_key);
  const bool enable_group_parallel_optimize =
      enable_group_parallels_[schedule_result_key.first][schedule_result_key.second] && (group_num > 1);
  GELOGD("Enable group parallel optimize[%d] group num[%u] of asc graph[%zu], result[%zu]",
         enable_group_parallel_optimize, group_num, schedule_result_key.first, schedule_result_key.second);
  GE_ASSERT_SUCCESS(
      GenFindPerfBetterTilingbyCaseId(enable_group_parallel_optimize, enable_group_parallel_optimize, group_num),
      "Gen FindPerfBetterTilingbyCaseId failed.");
  // 根据 is_uniq_group_ 决定是否包含 workspace_map 参数，确保与 FindPerfBetterTilingbyCaseId 定义一致
  std::string params = config_.tiling_data_type_name + " &tiling_data" +
                       (is_uniq_group_ ? "" : ", std::unordered_map<int64_t, uint64_t> &workspace_map") +
                       ", int32_t tiling_case_id = -1";
  const ge::char_t *cache_str = (with_reuse_info_) ? ", GroupLevelCache *cache = nullptr" : "";
  GenCalcScoreVarsDefine();
  if (config_.is_inductor_scene && is_uniq_group_) {
    tiling_func_.AddLine("#if defined(__clang__)");
    tiling_func_.AddLine("#pragma GCC diagnostic push");
    tiling_func_.AddLine("#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"");
    tiling_func_.AddLine("#endif");
    tiling_func_.AddLine("extern \"C\" std::string GetTilingDataRepr(const " + config_.tiling_data_type_name +
                         " *) __attribute__((weak));");
    tiling_func_.AddLine("#if defined(__clang__)");
    tiling_func_.AddLine("#pragma GCC diagnostic pop");
    tiling_func_.AddLine("#endif");
  }
  tiling_func_.AddLine("bool GetTilingKey(" + params + cache_str + ") {");
  GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_DOTILING, "    "),
                    "Generate begin code!");
  tiling_func_.AddLine("  if (::g_final_tiling_observe_enabled) { g_final_selected_sub_case_tag_valid = false; }");
  tiling_func_.AddLine("  bool ret = false;");
  tiling_func_.AddLine("  bool sub_case_flag = false;");
  tiling_func_.AddLine("  double obj = -1;");
  tiling_func_.AddLine("  double ub_ratio = -1;");
  const std::string force_sub_case_tag =
      config_.force_tiling_case.GetTag(tiling_model_info_[0].schedule_group_ident.group_id);
  tiling_func_.AddLine("  const char *selected_sub_case_tag = \"" + force_sub_case_tag + "\";");
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  tiling_func_.AddLine("  uint32_t corenum = tiling_data.get_" + core_num + "();");
  GE_ASSERT_SUCCESS(GenGetTilingbyCaseId(), "Gen GetTilingbyCaseId failed.");
  tiling_func_.AddLine("  if (::g_final_tiling_observe_enabled) {");
  tiling_func_.AddLine("    g_final_selected_sub_case_tag = selected_sub_case_tag;");
  tiling_func_.AddLine("    g_final_selected_sub_case_tag_valid = true;");
  tiling_func_.AddLine("  }");
  GE_ASSERT_SUCCESS(ValidateSingleResultAndGroup(), "Gen ValidateSingleResultAndGroup failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFinalTilingRepresentation() {
  if (config_.is_inductor_scene && is_uniq_group_) {
    tiling_func_.AddLine("  std::string final_tiling_repr;");
    tiling_func_.AddLine(
        "  if (GetTilingDataRepr != nullptr) { final_tiling_repr = GetTilingDataRepr(&tiling_data); }");
    tiling_func_.AddLine(
        "  const std::string final_repr_kind = final_tiling_repr.empty() ? \"unavailable\" : \"full_json\";");
  } else {
    tiling_func_.AddLine("  const std::string final_tiling_repr = GetTilingDataRepr(&tiling_data);");
    tiling_func_.AddLine(
        "  const std::string final_repr_kind = final_tiling_repr.empty() ? \"unavailable\" : \"full_json\";");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFinalTilingByCaseEmitter() {
  const auto &ident = tiling_model_info_[0].schedule_group_ident;
  const std::string data_type = config_.tiling_data_type_name;
  tiling_func_.AddLine("void EmitFinalTilingByCase(" + data_type +
                       " &tiling_data, uint32_t corenum, const char *source, const char *selection_mode, "
                       "uint32_t group_id, int32_t requested_case_id) {");
  AddAtomicHeaderLine(
      autofuse::GeneratedHeaderId::kApi,
      "void EmitFinalTilingByCase(" + data_type + " &, uint32_t, const char *, const char *, uint32_t, int32_t);");
  tiling_func_.AddLine("  if (!ShouldEmitFinalTiling()) { return; }");
  tiling_func_.AddLine("  const uint32_t final_tiling_key = tiling_data.get_tiling_key();");
  tiling_func_.AddLine(
      "  const int32_t final_case_id = requested_case_id >= 0 ? requested_case_id : "
      "GetFinalTilingCaseId(final_tiling_key);");
  tiling_func_.AddLine("  if (final_case_id < 0) { return; }");
  tiling_func_.AddLine(
      "  TilingCaseImplPtr final_case_impl = GetSelectedTilingImplPtr(static_cast<uint32_t>(final_case_id), corenum, "
      "nullptr);");
  tiling_func_.AddLine("  if (final_case_impl == nullptr) { return; }");
  tiling_func_.AddLine("  std::vector<FinalTilingPipeEstimate> final_pipe_estimates;");
  tiling_func_.AddLine("  final_case_impl->CollectPipeEstimates(tiling_data, final_pipe_estimates);");
  tiling_func_.AddLine("  const int32_t final_score = final_case_impl->CalcScore(tiling_data);");
  GE_ASSERT_SUCCESS(GenFinalTilingRepresentation(), "Generate final tiling representation failed.");
  tiling_func_.AddLine("  FinalTilingContext final_context{");
  tiling_func_.AddLine("      1U, source, selection_mode, OP_NAME,");
  tiling_func_.AddLine("      " + std::to_string(ident.asc_graph_id) + "U, " + std::to_string(ident.impl_graph_id) +
                       "U, group_id, final_case_id, final_tiling_key, final_score, final_case_impl->GetSubCaseTag(),");
  tiling_func_.AddLine("      final_case_impl->GetTemplateName(), final_pipe_estimates, final_tiling_repr,");
  tiling_func_.AddLine("      final_repr_kind};");
  tiling_func_.AddLine("  EmitFinalTilingRecord(final_context);");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFinalTilingEmitter() {
  GE_ASSERT_SUCCESS(GenFinalTilingByCaseEmitter(), "Generate final tiling case emitter failed.");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOSearchTilingKey() {
  GE_ASSERT_SUCCESS(GenSearchAllTilingbyCaseId(), "Gen SearchAllTilingbyCaseId failed.");
  std::string params = config_.tiling_data_type_name + " &tiling_data" + ", int32_t tiling_case_id";
  const std::string output_tiling_data_type =
      config_.is_autofuse ? "AutofuseTilingData" : config_.tiling_data_type_name;
  AddAtomicHeaderLine(
      autofuse::GeneratedHeaderId::kApi,
      "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params + ", " +
          output_tiling_data_type + "* output_tiling_data," + GenPgoTensorArgsDef() +
          "void* stream, uint64_t workspaceSize, double& out_best_perf, std::unordered_map<int64_t, uint64_t> "
          "&workspace_map, std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr);");
  tiling_func_.AddLine(
      "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params +
      ", AutofuseTilingData* output_tiling_data," + GenPgoTensorArgsDef() +
      "void* stream, uint64_t workspaceSize, double& out_best_perf, std::unordered_map<int64_t, uint64_t> "
      "&workspace_map, std::vector<uint32_t*> block_dim_vec, const SearchConfig *search_cfg) {");
  // Suppress unused-parameter warnings for parameters not used in every code path
  {
    std::string void_casts = "  (void)out_best_perf; (void)tiling_case_id; (void)output_tiling_data; ";
    void_casts += "(void)stream; (void)workspaceSize; (void)workspace_map; (void)block_dim_vec;";
    tiling_func_.AddLine(void_casts);
  }
  tiling_func_.AddLine("  PgoConfig::Instance().tensor_args = tensor_args;");
  tiling_func_.AddLine("  SearchConfig local_search_cfg;");
  tiling_func_.AddLine("  const SearchConfig *effective_cfg = search_cfg;");
  tiling_func_.AddLine("  if (effective_cfg == nullptr && PgoConfig::Instance().need_change_solver_run == 1) {");
  tiling_func_.AddLine("    local_search_cfg.ub_threshold_enabled = true;");
  tiling_func_.AddLine(
      "    local_search_cfg.ub_threshold = "
      "PgoConfig::Instance().pgo_ub_threshold_list[PgoConfig::Instance().pgo_threshold_index];");
  tiling_func_.AddLine("    local_search_cfg.corenum_threshold_enabled = true;");
  tiling_func_.AddLine(
      "    local_search_cfg.corenum_threshold = "
      "PgoConfig::Instance().pgo_corenum_threshold_list[PgoConfig::Instance().pgo_threshold_index];");
  tiling_func_.AddLine("    effective_cfg = &local_search_cfg;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  bool ret = false;");
  tiling_func_.AddLine("  double obj = -1;");
  tiling_func_.AddLine("  double ub_ratio = -1;");
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  tiling_func_.AddLine("  uint32_t corenum = tiling_data.get_" + core_num + "();");
  GE_ASSERT_SUCCESS(GenPGOGetTilingbyCaseId(), "Gen GetTilingbyCaseId failed.");
  if (is_uniq_group_) {
    GenPGOSearchTilingKeyUniqGroupBatch();
  }
  GE_ASSERT_SUCCESS(ValidateSingleResultAndGroup(true), "Gen ValidateSingleResultAndGroup failed.");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenPGOSearchTilingKeyUniqGroupBatch() {
  tiling_func_.AddLine("  workspaceSize = 0;");
  tiling_func_.AddLine("  for (const auto &tiling_data_perf : tiling_data_list) {");
  tiling_func_.AddLine("    uint64_t workspaceSizeTmp = 0U;");
  tiling_func_.AddLine("    if (!GetWorkspaceSize(tiling_data_perf.tiling_data, workspaceSizeTmp)) {");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    if (workspaceSizeTmp > workspaceSize) {");
  tiling_func_.AddLine("      workspaceSize = workspaceSizeTmp;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  if (__builtin_add_overflow(workspaceSize, 16ULL * 1024ULL * 1024ULL, &workspaceSize)) {");
  tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"Workspace size overflow in PGO batch profiling.\");");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  if (PgoConfig::Instance().batch_callback != nullptr) {");
  tiling_func_.AddLine(
      "    if (PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, "
      "workspaceSize, &tiling_data_list) != 0) {");
  // Issue #274: release memory allocated in GenPGODefaultTiling before early return
  tiling_func_.AddLine("      free(memory);");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  for (const auto &tiling_data_perf : tiling_data_list) {");
  tiling_func_.AddLine("    if (best_perf > tiling_data_perf.best_perf) {");
  tiling_func_.AddLine("      best_perf = tiling_data_perf.best_perf;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
}

af::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKeyCollectTilingData(FusedGraphNamespaceMap namespace_map) {
  for (auto &asc_graph_map_iter : namespace_map) {
    size_t asc_graph_id = asc_graph_map_iter.first;
    tiling_func_.AddLine("    for (auto ascgraph_tiling_data_" + std::to_string(asc_graph_id) + " : vec" +
                         std::to_string(asc_graph_id) + ") {");
  }

  tiling_func_.AddLine("      AutofuseTilingData tiling_data_tmp;");
  tiling_func_.AddLine("      tiling_data_tmp = *tiling_data; // 用于初始化部分常量参数");

  for (auto &asc_graph_map_iter : namespace_map) {
    auto &asc_graph_map = asc_graph_map_iter.second;
    size_t asc_graph_id = asc_graph_map_iter.first;
    tiling_func_.AddLine("      tiling_data_tmp.set_graph" + std::to_string(asc_graph_id) +
                         "_tiling_key(ascgraph_tiling_data_" + std::to_string(asc_graph_id) + ".get_graph" +
                         std::to_string(asc_graph_id) + "_tiling_key());");
    for (auto &graph_info_map : asc_graph_map) {
      auto graph_info = graph_info_map.second;
      for (auto &group_info : graph_info) {
        auto schedule_result_prefix = group_info.second.second;
        tiling_func_.AddLine("      tiling_data_tmp." + schedule_result_prefix +
                             "_tiling_data = ascgraph_tiling_data_" + std::to_string(asc_graph_id) + "." +
                             schedule_result_prefix + "_tiling_data" + ";");
      }
    }
  }

  std::set<int64_t> workspace_ids;
  for (const auto &asc_graph_workspace_ids : workspace_tensor_id_set_) {
    for (const auto &impl_graph_workspace_ids : asc_graph_workspace_ids.second) {
      workspace_ids.insert(impl_graph_workspace_ids.second.begin(), impl_graph_workspace_ids.second.end());
    }
  }
  for (const auto &tensor_id : workspace_ids) {
    const auto tensor_id_str = std::to_string(tensor_id);
    for (const auto &asc_graph_map_iter : namespace_map) {
      const auto asc_graph_id = asc_graph_map_iter.first;
      tiling_func_.AddLine("      tiling_data_tmp.set_workspace" + tensor_id_str +
                           "(std::max(tiling_data_tmp.get_workspace" + tensor_id_str + "(), ascgraph_tiling_data_" +
                           std::to_string(asc_graph_id) + ".get_workspace" + tensor_id_str + "()));");
    }
  }
  GenWorkspaceOffsetFinalize("tiling_data_tmp");

  // The root search starts from the current core-count probe.  After composing
  // graph candidates, normalize the outer block_dim to the shared block range
  // instead of carrying that probe value into the candidate identity.
  tiling_func_.AddLine("      uint32_t total_block_dim = 0U;");
  for (const auto &asc_graph_map_iter : namespace_map) {
    const auto asc_graph_id = asc_graph_map_iter.first;
    for (const auto &graph_info_map : asc_graph_map_iter.second) {
      const auto result_id = graph_info_map.first;
      tiling_func_.AddLine("      if (tiling_data_tmp.get_graph" + std::to_string(asc_graph_id) +
                           "_tiling_key() == " + std::to_string(result_id) + ") {");
      tiling_func_.AddLine("        uint32_t result_block_dim = 0U;");
      for (const auto &group_info : graph_info_map.second) {
        const auto schedule_result_prefix = group_info.second.second;
        const auto group_block_dim = "tiling_data_tmp." + schedule_result_prefix + "_tiling_data.get_block_dim()";
        if (enable_group_parallels_[asc_graph_id][result_id]) {
          tiling_func_.AddLine("        result_block_dim += " + group_block_dim + ";");
        } else {
          tiling_func_.AddLine("        result_block_dim = std::max(result_block_dim, " + group_block_dim + ");");
        }
      }
      if (enable_group_parallels_[asc_graph_id][result_id]) {
        tiling_func_.AddLine("        result_block_dim = std::min(result_block_dim, block_dim_i);");
      }
      tiling_func_.AddLine("        total_block_dim = std::max(total_block_dim, result_block_dim);");
      tiling_func_.AddLine("      }");
    }
  }
  tiling_func_.AddLine("      tiling_data_tmp.set_block_dim(total_block_dim);");

  tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_tmp);");
  for ([[maybe_unused]] size_t i = 0; i < namespace_map.size(); ++i) {
    tiling_func_.AddLine("    }");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKeySingleGroup() {
  for (auto model_info : tiling_model_info_) {
    tiling_func_.AddLine("    tiling_case = " + std::to_string(model_info.tiling_case_id) + ";");
    tiling_func_.AddLine("    tiling_data->set_block_dim(block_dim_i);");
    tiling_func_.AddLine("    tiling_data->set_tiling_key(tiling_case);");
    tiling_func_.AddLine("    if (GetTilingKey(*tiling_data, tiling_case)) {");
    tiling_func_.AddLine("      tiling_data_tmp = *tiling_data;");
    tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_tmp);");
    tiling_func_.AddLine("    }");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOByCoreNumSearchTilingKey() {
  tiling_func_.AddLine(
      "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* "
      "tiling_data, uint32_t max_block_dim) {");
  tiling_func_.AddLine("  (void)tiling_data_list; (void)tiling_data; (void)max_block_dim;");
  tiling_func_.AddLine("  bool ret = true;");
  GenWorkspaceOffsetReset("*tiling_data");
  tiling_func_.AddLine("  for (uint32_t block_dim_i=1; block_dim_i <= max_block_dim; block_dim_i++) {");
  tiling_func_.AddLine("    int32_t tiling_case;");
  tiling_func_.AddLine("    AutofuseTilingData tiling_data_tmp;");
  if (is_uniq_group_) {
    GenPGOByCoreNumSearchTilingKeySingleGroup();
  }
  tiling_func_.AddLine("   }");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenHeaderCodesSummaryBody() {
  // add fixed tiling data
  ge::CodePrinter dumper;
  std::set<std::string> fixed_var_name;
  std::set<size_t> tiling_keys;
  std::set<uint32_t> case_id_set;
  for (const auto &model_info : tiling_model_info_) {
    if (!case_id_set.insert(model_info.tiling_case_id).second) {
      continue;
    }
    tiling_keys.insert(model_info.schedule_group_ident.asc_graph_id);
  }
  for (const size_t tiling_key : tiling_keys) {
    fixed_var_name.insert("graph" + std::to_string(tiling_key) + "_tiling_key");
  }
  std::set<std::string> keep_uniq;
  for (const auto &model_info : tiling_model_info_) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto scope_names = GetHardwareNames(args_manager.GetTotalHardwareCons(config_.do_variable_replace));
    fixed_var_name.insert(scope_names.begin(), scope_names.end());
  }
  TilingDataGenUtils::WriteTilingDataElement(dumper, keep_uniq, fixed_var_name);
  // add struct tiling data
  std::map<std::string, std::string> struct_set;
  for (const auto &model_info : tiling_model_info_) {
    struct_set[model_info.schedule_group_ident.GetGroupPrefix() + "TilingData"] =
        model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data";
  }
  for (const auto &pair : struct_set) {
    TilingDataGenUtils::WriteTilingDataStruct(dumper, keep_uniq, pair.first, pair.second);
  }
  tiling_data_.AddLine(TilingDataGenUtils::StructElementDefine(config_.tiling_data_type_name, dumper.GetOutputStr()));
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenTilingHeadMultiGroup() {
  std::string params = config_.tiling_data_type_name + " &tiling_data, int32_t tiling_case_id";
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    AddAtomicHeaderLine(
        autofuse::GeneratedHeaderId::kApi,
        "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " + params + ", " +
            config_.tiling_data_type_name + "* output_tiling_data," + GenPgoTensorArgsDef() +
            "void* stream, uint64_t workspaceSize, double& best_perf, const SearchConfig *search_cfg=nullptr);");
  }
}

void TilingCodeGenImpl::InitTilingHeadCodeGeneration() {
  tiling_head_.Reset();
  ResetTranslationUnit();
  tiling_data_.Reset();
  atomic_headers_.clear();
  auto &state_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kState].dependencies;
  autofuse::RequireSystemHeader(state_dependencies, "cstdint");
  auto &solver_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kSolver].dependencies;
  for (const auto &header : {"functional", "string"}) {
    autofuse::RequireSystemHeader(solver_dependencies, header);
  }
  auto &api_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kApi].dependencies;
  autofuse::RequireSystemHeader(api_dependencies, "cstdint");
  if (!is_uniq_group_ || config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    autofuse::RequireSystemHeader(api_dependencies, "unordered_map");
  }
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    autofuse::RequireSystemHeader(api_dependencies, "vector");
  }
  auto &log_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kLog].dependencies;
  for (const auto &header : {"array", "algorithm", "cmath", "cstdint", "cstddef", "iomanip", "iterator", "limits",
                             "sstream", "string", "vector"}) {
    autofuse::RequireSystemHeader(log_dependencies, header);
  }
  if (!AddSlogExtend().empty()) {
    for (const auto &header : {"cinttypes", "cstdarg", "cstdint", "cstring", "securec.h", "stdlib.h", "sys/syscall.h",
                               "unistd.h", "vector"}) {
      autofuse::RequireSystemHeader(log_dependencies, header);
    }
    autofuse::RequireExternalHeader(log_dependencies, "dlog_pub.h");
    autofuse::RequireExternalHeader(log_dependencies, "err_msg.h");
  }
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kLog);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kSolver);
}

void TilingCodeGenImpl::GenPgoSearchConfigDef() {
  auto &pgo_dependencies = atomic_headers_[autofuse::GeneratedHeaderId::kPgo].dependencies;
  for (const auto &header : {"array", "cstddef", "cstdint", "vector"}) {
    autofuse::RequireSystemHeader(pgo_dependencies, header);
  }
  if (config_.is_autofuse) {
    GenPgoHeaderCodesTail();
  }
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "struct SearchConfig {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "  bool ub_threshold_enabled = true;");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "  double ub_threshold = 0.0;");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "  bool corenum_threshold_enabled = true;");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "  double corenum_threshold = 1.0;");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "  bool enable_multicore_ub_tradeoff = true;");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kPgo, "};");
}

af::Status TilingCodeGenImpl::GenTilingHead(std::map<std::string, std::string> &tiling_res,
                                            const EnableGroupParallels &enable_group_parallels) {
  enable_group_parallels_ = enable_group_parallels;
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  // 1、生成总TilingData
  InitTilingHeadCodeGeneration();
  GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
  if (config_.gen_tiling_data) {
    GE_ASSERT_SUCCESS(GenHeaderCodesHead(), "Generate tiling data head failed.");
  }
  // 2、生成公共的TilingFunc代码
  GE_ASSERT_SUCCESS(GenMacroInclude(), "Generate macro include failed.");
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    GenPgoSearchConfigDef();
  }
  tiling_head_.AddLine("namespace optiling{};");
  tiling_head_.AddLine("using namespace optiling;");
  tiling_head_.AddLine("bool GetWorkspaceSize(const AutofuseTilingData &tiling_data, uint64_t &workspace_size);");
  tiling_head_.AddLine("namespace optiling {");
  tiling_func_.AddLine("namespace optiling {");
  // 支持二次Tiling：在第一个namespace optiling中extern声明全局变量（定义在GenTilingTail中）
  // PGO场景不需要此全局变量声明
  if (!config_.enable_autofuse_pgo) {
    tiling_func_.AddLine("// 支持二次Tiling：全局变量，用于传递调整后的核数比例");
    tiling_func_.AddLine("extern thread_local double g_secondary_tiling_ratio;");
  }
  if (!is_uniq_group_) {
    GenTilingHeadMultiGroup();
  }
  if (config_.enable_autofuse_pgo) {
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi, "bool PGOByCoreNumSearchTilingKey(std::vector<" +
                                                               config_.tiling_data_type_name + ">& tiling_data_list, " +
                                                               config_.tiling_data_type_name +
                                                               "* tiling_data, uint32_t max_block_dim);");
  }
  tiling_head_.AddLine("using namespace std;");
  // 生成ArrangeBlockOffsets函数声明
  GenArrangeBlockOffsetsDeclarations(namespace_map);
  GE_ASSERT_SUCCESS(GenCommonFrameWork(), "Generate common framework failed.");
  tiling_func_.AddLine("} // namespace optiling");
  for (const auto &header : {"algorithm", "cmath", "cstddef", "cstdint", "functional", "utility", "vector"}) {
    RequireTranslationUnitSystemHeader(header);
  }
  if (config_.gen_tiling_data) {
    tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
  }
  tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
  GE_ASSERT_SUCCESS(FinishTranslationUnit(kTilingSolverIdentify, tiling_res));
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenArrangeBlockOffsetsDeclarations(const FusedGraphNamespaceMap &namespace_map) {
  for (const auto &asc_graph_map_iter : namespace_map) {
    const auto &asc_graph_id = asc_graph_map_iter.first;
    const auto &asc_graph_namespace_map = asc_graph_map_iter.second;
    for (const auto &result_id_and_groups : asc_graph_namespace_map) {
      const auto &result_id = result_id_and_groups.first;
      if (enable_group_parallels_[asc_graph_id][result_id]) {
        AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                            "void ArrangeBlockOffsetsAscGraph" + std::to_string(asc_graph_id) + "Result" +
                                std::to_string(result_id) + "(" + config_.tiling_data_type_name +
                                " &t, uint32_t aiv_num);");
      }
    }
  }
}

af::Status TilingCodeGenImpl::ObtainInnerParams(std::map<std::string, std::set<std::string>> &hardware_map,
                                                FusedGraphNamespaceMap &namespace_map) {
  std::string obj_name;
  for (const auto &model_info : tiling_model_info_) {
    obj_name = model_info.schedule_group_ident.GetItemPrefix();
    auto &asc_graph_namespace_map = namespace_map[model_info.schedule_group_ident.asc_graph_id];
    auto &schedule_result_namespace_map = asc_graph_namespace_map[model_info.schedule_group_ident.impl_graph_id];
    auto &schedule_group_namespace_map = schedule_result_namespace_map[model_info.schedule_group_ident.group_id];
    schedule_group_namespace_map = std::make_pair(model_info.schedule_group_ident.GetGroupPrefix(), obj_name);
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    for (const auto &hardware : args_manager.GetTotalHardwareCons(config_.do_variable_replace)) {
      hardware_map[obj_name].insert(BaseTypeUtils::DumpHardware(hardware.first));
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetResultSummary(const size_t asc_graph_id) {
  tiling_func_.AddLine("bool GetResultSummary(const double best_perf, " + config_.tiling_data_type_name +
                       " &tiling_data) {");
  tiling_func_.AddLine("  if (IsEqual(best_perf, -1)) {");
  const std::string failed_log_level =
      (config_.enable_autofuse_pgo || config_.is_inductor_scene) ? "OP_LOGW" : "OP_LOGE";
  tiling_func_.AddLine("    " + failed_log_level + "(OP_NAME, \"GetTiling Failed.\");");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"[PROF]Among all schedule results, " + tiling_key_prefix +
                       "result%u is the best choice.\", "
                       "tiling_data.get_" +
                       tiling_key_prefix + "tiling_key());");
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingForAllInitLines(bool pgo) {
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start GetTiling.\");");
  tiling_func_.AddLine("  double cur_perf;");
  if (!pgo) {
    tiling_func_.AddLine("  double best_perf = -1;");
  }
  tiling_func_.AddLine("  uint32_t cur_block_dim = 1;");
  tiling_func_.AddLine("  uint32_t ori_block_dim = tiling_data.get_block_dim();");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenCacheInit() {
  if (with_reuse_info_) {
    std::unordered_set<std::string> declared_cache_types_;  // 防止重复声明
    for (const auto &pair : cache_reuse_info_) {
      if (declared_cache_types_.find(pair.second) == declared_cache_types_.end()) {
        tiling_func_.AddLine("  " + pair.second + "::GroupLevelCache " + pair.second + "_Cache;");
        declared_cache_types_.insert(pair.second);
      }
    }
  }
}

inline af::Expression GetInputVarFromSrcVarExpr(const af::Expression &src_var_expr,
                                                const std::string &src_tiling_data_name) {
  std::unordered_set<std::string> contain_vars;
  for (const auto &arg : src_var_expr.FreeSymbols()) {
    if (arg.GetExprType() == af::ExprType::kExprVariable) {
      contain_vars.insert(Str(arg));
    }
  }
  std::vector<std::pair<Expr, Expr>> var_replacement;
  for (auto &var : contain_vars) {
    var_replacement.emplace_back(std::make_pair(
        CreateExpr(var.c_str()),
        CreateExpr(("static_cast<double>(" + src_tiling_data_name + "_tiling_data.get_" + var + "())").c_str())));
  }
  return src_var_expr.Replace(var_replacement);
}

inline af::Expression GetInputVarFromSrcVarExprWithPrefix(const af::Expression &src_var_expr,
                                                          const std::string &src_tiling_data_name) {
  std::unordered_set<std::string> contain_vars;
  for (const auto &arg : src_var_expr.FreeSymbols()) {
    if (arg.GetExprType() == af::ExprType::kExprVariable) {
      contain_vars.insert(Str(arg));
    }
  }
  std::vector<std::pair<Expr, Expr>> var_replacement;
  for (auto &var : contain_vars) {
    var_replacement.emplace_back(std::make_pair(
        CreateExpr(var.c_str()),
        CreateExpr(
            ("static_cast<double>(tiling_data." + src_tiling_data_name + "_tiling_data.get_" + var + "())").c_str())));
  }
  return src_var_expr.Replace(var_replacement);
}

inline std::vector<std::string> GetSortedFreeVarNames(const af::Expression &src_var_expr) {
  std::vector<std::string> vars;
  for (const auto &arg : src_var_expr.FreeSymbols()) {
    if (arg.GetExprType() == af::ExprType::kExprVariable) {
      vars.emplace_back(Str(arg));
    }
  }
  std::sort(vars.begin(), vars.end());
  return vars;
}

inline std::string GetLeadingSpaces(const std::string &prefix) {
  const auto pos = prefix.find_first_not_of(' ');
  return pos == std::string::npos ? prefix : prefix.substr(0UL, pos);
}

inline std::string GenVarRelationSourceZeroCond(const std::vector<std::string> &vars,
                                                const std::string &src_tiling_data_name,
                                                const std::string &tiling_data_prefix) {
  std::string cond;
  for (const auto &var : vars) {
    if (!cond.empty()) {
      cond += " || ";
    }
    cond += tiling_data_prefix + src_tiling_data_name + "_tiling_data.get_" + var + "() == 0U";
  }
  return cond;
}

inline std::string GenVarRelationInvalidCheck(const std::string &value_name) {
  return "!std::isfinite(" + value_name + ") || " + value_name + " <= 0.0 || " + value_name +
         " > static_cast<double>(std::numeric_limits<uint32_t>::max())";
}

inline std::string GenInvalidVarRelationActions(const std::string &indent,
                                                const std::vector<std::string> &invalid_actions) {
  std::string code;
  for (const auto &action : invalid_actions) {
    code += indent + "  " + action + "\n";
  }
  return code;
}

inline std::string GenVarRelationSetStatement(const std::string &target_prefix, const std::string &dst_tiling_data_name,
                                              const std::string &var_name, const std::string &src_tiling_data_name,
                                              const af::Expression &src_var_expr,
                                              const std::vector<std::string> &invalid_actions) {
  const std::string indent = GetLeadingSpaces(target_prefix);
  const auto src_vars = GetSortedFreeVarNames(src_var_expr);
  const std::string zero_cond = GenVarRelationSourceZeroCond(src_vars, src_tiling_data_name, "tiling_data.");
  const std::string value_name = dst_tiling_data_name + "_" + var_name + "_var_relation_value";
  const auto dst_expr = GetInputVarFromSrcVarExprWithPrefix(src_var_expr, src_tiling_data_name);
  std::string code;
  if (!zero_cond.empty()) {
    code += indent + "if (" + zero_cond + ") {\n";
    code += indent + "  OP_LOGW(OP_NAME, \"Skip invalid var relation " + var_name + ": source value is zero.\");\n";
    code += GenInvalidVarRelationActions(indent, invalid_actions);
    code += indent + "}\n";
  }
  code += indent + "double " + value_name + " = " + Str(dst_expr) + ";\n";
  code += indent + "if (" + GenVarRelationInvalidCheck(value_name) + ") {\n";
  code += indent + "  OP_LOGW(OP_NAME, \"Skip invalid var relation " + var_name + ": invalid computed value.\");\n";
  code += GenInvalidVarRelationActions(indent, invalid_actions);
  code += indent + "}\n";
  code += target_prefix + dst_tiling_data_name + "_tiling_data.set_" + var_name + "(static_cast<uint32_t>(" +
          value_name + "));\n";
  return code;
}

inline std::string GenVarRelationSetCondition(const std::string &dst_tiling_data_name, const std::string &var_name,
                                              const std::string &src_tiling_data_name,
                                              const af::Expression &src_var_expr) {
  const auto src_vars = GetSortedFreeVarNames(src_var_expr);
  const std::string zero_cond = GenVarRelationSourceZeroCond(src_vars, src_tiling_data_name, "");
  const std::string value_name = dst_tiling_data_name + "_" + var_name + "_var_relation_value";
  const auto dst_expr = GetInputVarFromSrcVarExpr(src_var_expr, src_tiling_data_name);
  std::string condition = "([&]() -> bool { ";
  if (!zero_cond.empty()) {
    condition += "if (" + zero_cond + ") { OP_LOGW(OP_NAME, \"Skip invalid var relation " + var_name +
                 ": source value is zero.\"); return false; } ";
  }
  condition += "double " + value_name + " = " + Str(dst_expr) + "; ";
  condition += "if (" + GenVarRelationInvalidCheck(value_name) + ") { OP_LOGW(OP_NAME, \"Skip invalid var relation " +
               var_name + ": invalid computed value.\"); return false; } ";
  condition += dst_tiling_data_name + "_tiling_data.set_" + var_name + "(static_cast<uint32_t>(" + value_name +
               ")); return true; }())";
  return condition;
}

inline std::pair<std::string, bool> ProcessVarRelations(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &var_relation, size_t group_id) {
  std::string input_vars_set_code;
  bool need_update = false;
  auto it = var_relation.find(group_id);
  if (it != var_relation.end()) {
    for (const auto &var_expr_pair : it->second) {
      size_t src_id = var_expr_pair.first;
      for (const auto &pair : var_expr_pair.second) {
        need_update = true;
        auto src_it = graph_info.find(src_id);
        auto dst_it = graph_info.find(group_id);
        if (src_it != graph_info.end() && dst_it != graph_info.end()) {
          input_vars_set_code +=
              (input_vars_set_code.empty() ? "" : "&&") +
              GenVarRelationSetCondition(dst_it->second.second, pair.first, src_it->second.second, pair.second);
        }
      }
    }
  }
  return {input_vars_set_code, need_update};
}

inline std::pair<std::string, bool> ProcessVarRelationsStatement(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &var_relation, size_t group_id,
    const std::string &prefix, const std::vector<std::string> &invalid_actions) {
  std::string input_vars_set_code;
  bool need_update = false;
  auto it = var_relation.find(group_id);
  if (it != var_relation.end()) {
    for (const auto &var_expr_pair : it->second) {
      size_t src_id = var_expr_pair.first;
      for (const auto &pair : var_expr_pair.second) {
        auto src_it = graph_info.find(src_id);
        auto dst_it = graph_info.find(group_id);
        if (src_it != graph_info.end() && dst_it != graph_info.end()) {
          input_vars_set_code += GenVarRelationSetStatement(prefix, dst_it->second.second, pair.first,
                                                            src_it->second.second, pair.second, invalid_actions);
          need_update = true;
        }
      }
    }
  }
  return {input_vars_set_code, need_update};
}

void TilingCodeGenImpl::GenSetHardwareCodes(const std::string &group_prefix,
                                            const std::set<std::string> &hardware_names) {
  for (const auto &hardware_name : hardware_names) {
    std::string set_code("  ");
    set_code.append(group_prefix).append("_tiling_data.set_").append(hardware_name);
    bool is_block_dim = (hardware_name == "block_dim");
    std::string hardware_val = is_block_dim ? "(ori_block_dim);" : "(tiling_data.get_" + hardware_name + "());";
    tiling_func_.AddLine(set_code.append(hardware_val));
  }
}

void TilingCodeGenImpl::GenGetScheduleResultTail(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  for (const auto &group_info : graph_info) {
    tiling_func_.AddLine("  " + group_info.second.second + "_tiling_data.set_block_dim(0);");
  }
  tiling_func_.AddLine("  return false;");
  tiling_func_.AddLine("}");
}

void TilingCodeGenImpl::GenWorkspaceOffsetHelpers() {
  const auto workspace_ids = GetWorkspaceTensorIds(workspace_tensor_id_set_);
  if (workspace_ids.empty()) {
    return;
  }

  const auto &tiling_data_type = config_.tiling_data_type_name;
  // ATT 内部仍使用 workspace 字段暂存全局大小，所有子图完成后再统一转换为 offset。
  tiling_func_.AddLine("inline void ResetWorkspaceSizes(" + tiling_data_type + " &tiling_data) {");
  for (const auto &tensor_id : workspace_ids) {
    tiling_func_.AddLine("  tiling_data.set_workspace" + std::to_string(tensor_id) + "(0);");
  }
  tiling_func_.AddLine("}");

  tiling_func_.AddLine("inline bool FinalizeWorkspaceOffsets(" + tiling_data_type + " &tiling_data) {");
  tiling_func_.AddLine("  uint64_t workspace_offset = 0U;");
  for (const auto &tensor_id : workspace_ids) {
    const auto tensor_id_str = std::to_string(tensor_id);
    tiling_func_.AddLine("  const uint64_t workspace_size_" + tensor_id_str + " = tiling_data.get_workspace" +
                         tensor_id_str + "();");
    tiling_func_.AddLine("  tiling_data.set_workspace" + tensor_id_str + "(workspace_offset);");
    tiling_func_.AddLine("  if (__builtin_add_overflow(workspace_offset, workspace_size_" + tensor_id_str +
                         ", &workspace_offset)) {");
    tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"Workspace offset overflow, workspace id is " + tensor_id_str +
                         ".\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
  }
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
}

void TilingCodeGenImpl::GenWorkspaceOffsetReset(const std::string &tiling_data_name) {
  if (GetWorkspaceTensorIds(workspace_tensor_id_set_).empty()) {
    return;
  }
  tiling_func_.AddLine("  ResetWorkspaceSizes(" + tiling_data_name + ");");
}

void TilingCodeGenImpl::GenWorkspaceOffsetFinalize(const std::string &tiling_data_name) {
  if (GetWorkspaceTensorIds(workspace_tensor_id_set_).empty()) {
    return;
  }
  tiling_func_.AddLine("  if (!FinalizeWorkspaceOffsets(" + tiling_data_name + ")) {");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
}

void TilingCodeGenImpl::GenUpdateWorkspace(const size_t asc_graph_id, const size_t impl_graph_id) {
  // 选中新 best 候选时，先清空本子图全部 workspace id 的占用，再写入当前候选的 size。
  // workspace 当前不存在复用，每个 id 仅由一个候选持有，直接赋值即可：
  // 落选候选独有的 id 会被清零，字段值始终等于选中候选各 id 的 size，
  // FinalizeWorkspaceOffsets 计算出的偏移与 GetWorkspaceSize 按选中分支计算的总大小严格一致。
  for (const auto &tensor_id : GetAscGraphWorkspaceTensorIds(workspace_tensor_id_set_, asc_graph_id)) {
    tiling_func_.AddLine("      tiling_data.set_workspace" + std::to_string(tensor_id) + "(0U);");
  }
  for (const auto &tensor_id : workspace_tensor_id_set_[asc_graph_id][impl_graph_id]) {
    auto tensor_id_str = to_string(tensor_id);
    tiling_func_.AddLine("      auto it" + tensor_id_str + " = workspace_map.find(" + tensor_id_str + ");");
    tiling_func_.AddLine("      if (it" + tensor_id_str + " != workspace_map.end()) {");
    tiling_func_.AddLine("        tiling_data.set_workspace" + tensor_id_str + "(it" + tensor_id_str + "->second);");
    tiling_func_.AddLine("      }");
  }
}

std::string TilingCodeGenImpl::GenPerfUpdateCode(const std::vector<std::string> &groups_perf,
                                                 const std::vector<std::string> &groups_block_num,
                                                 const std::string &indent) {
  if (groups_perf.size() == 1UL) {
    return indent + "cur_perf = " + groups_perf[0] + ";\n";
  }
  std::string update_code(indent + "cur_perf = 0.0;\n");
  (void)update_code.append(indent + "bool has_update = false;\n")
      .append(indent + "auto cur_tmp_perf = ")
      .append(groups_perf[0])
      .append(";\n")
      .append(indent + "auto cur_block = ")
      .append(groups_block_num[0])
      .append(";\n");
  for (size_t id = 1UL; id < groups_perf.size(); ++id) {
    (void)update_code.append(indent + "has_update = UpdateCurPerfAndBlockByGroup({")
        .append(groups_block_num[id])
        .append(", ")
        .append(groups_perf[id])
        .append("}, ori_block_dim, cur_block, cur_perf, cur_tmp_perf);\n");
  }
  (void)update_code.append(indent + "OP_LOGD(OP_NAME, \"Begin to add group perf %lf\", cur_tmp_perf);\n")
      .append(indent + "cur_perf += cur_tmp_perf;\n");
  return update_code;
}

std::vector<std::pair<size_t, size_t>> TilingCodeGenImpl::GetSerializedGroupEdges(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &var_relation,
    const bool enable_group_parallel) const {
  if (enable_group_parallel) {
    return {};
  }

  std::set<std::pair<size_t, size_t>> edge_set;
  for (const auto &dst_relation : var_relation) {
    const size_t dst_id = dst_relation.first;
    if (graph_info.find(dst_id) == graph_info.end()) {
      continue;
    }
    for (const auto &src_relation : dst_relation.second) {
      const size_t src_id = src_relation.first;
      if (!src_relation.second.empty() && src_id != dst_id && graph_info.find(src_id) != graph_info.end()) {
        edge_set.emplace(src_id, dst_id);
      }
    }
  }
  return {edge_set.begin(), edge_set.end()};
}

std::vector<std::pair<size_t, size_t>> TilingCodeGenImpl::GetWorkspaceSharedGroupEdges(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) const {
  std::map<size_t, std::set<int64_t>> group_workspace_ids;
  for (const auto &model_info : tiling_model_info_) {
    const auto &ident = model_info.schedule_group_ident;
    if (ident.asc_graph_id != asc_graph_id || ident.impl_graph_id != impl_graph_id ||
        graph_info.find(ident.group_id) == graph_info.end()) {
      continue;
    }
    auto &tensor_ids = group_workspace_ids[ident.group_id];
    for (const auto &workspace : model_info.workspace_size_map) {
      tensor_ids.insert(workspace.first);
    }
  }
  std::vector<std::pair<size_t, size_t>> edges;
  for (auto lhs = group_workspace_ids.begin(); lhs != group_workspace_ids.end(); ++lhs) {
    auto rhs = lhs;
    for (++rhs; rhs != group_workspace_ids.end(); ++rhs) {
      const auto &smaller = (lhs->second.size() <= rhs->second.size()) ? lhs->second : rhs->second;
      const auto &larger = (lhs->second.size() <= rhs->second.size()) ? rhs->second : lhs->second;
      const bool shared = std::any_of(smaller.cbegin(), smaller.cend(),
                                      [&larger](const int64_t tensor_id) { return larger.count(tensor_id) != 0U; });
      if (shared) {
        edges.emplace_back(lhs->first, rhs->first);
      }
    }
  }
  return edges;
}

std::string TilingCodeGenImpl::GenSerializedEdgePenalty(const std::vector<std::string> &groups_perf,
                                                        const std::vector<std::pair<size_t, size_t>> &serialized_edges,
                                                        const std::string &indent) const {
  std::vector<std::pair<size_t, size_t>> valid_edges;
  valid_edges.reserve(serialized_edges.size());
  for (const auto &edge : serialized_edges) {
    if (edge.first >= groups_perf.size() || edge.second >= groups_perf.size()) {
      return {};
    }
    valid_edges.emplace_back(edge);
  }
  if (valid_edges.empty()) {
    return {};
  }

  const std::string ratio = std::to_string(SerializedEdgePenaltyConfig::kPenaltyRatio);
  const std::string cap = std::to_string(SerializedEdgePenaltyConfig::kPenaltyCapCycles);
  std::string code = indent + "double serialized_edge_base = 0.0;\n";
  for (const auto &edge : valid_edges) {
    code += indent + "serialized_edge_base += std::max(" + groups_perf[edge.first] + ", " + groups_perf[edge.second] +
            ");\n";
  }
  code += indent + "double serialized_edge_penalty = std::min(" + cap + ", " + ratio + " * serialized_edge_base);\n";
  code += indent + "cur_perf += serialized_edge_penalty;\n";
  code += indent +
          "OP_LOGD(OP_NAME, \"serialized_edge_model=" + std::string(SerializedEdgePenaltyConfig::kModelVersion) +
          " serialized_edge_penalty=min(" + cap + ", " + ratio +
          " * serialized_edge_base)=%lf (base=%lf)\", serialized_edge_penalty, serialized_edge_base);\n";
  return code;
}

void TilingCodeGenImpl::GenBestPerfUpdateCode(const size_t asc_graph_id, const size_t impl_graph_id,
                                              const std::vector<std::string> &assign_max_block_num,
                                              const std::string &indent, const std::string &objective_log) {
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
  if (objective_log.empty()) {
    tiling_func_.AddLine(indent + "OP_LOGI(OP_NAME, \"The value of graph" + std::to_string(asc_graph_id) + "_result" +
                         std::to_string(impl_graph_id) + " is %lf\", cur_perf);");
  } else {
    tiling_func_.AddLine(objective_log);
  }
  tiling_func_.AddLine(indent + "if (IsEqual(best_perf, -1) || cur_perf < best_perf) {");
  tiling_func_.AddLine(indent + "  best_perf = cur_perf;");
  for (const auto &code : assign_max_block_num) {
    tiling_func_.AddLine(code);
  }
  tiling_func_.AddLine(indent + "  tiling_data.set_block_dim(cur_block_dim);");
  tiling_func_.AddLine(indent + "  tiling_data.set_" + tiling_key_prefix + "tiling_key(" +
                       std::to_string(impl_graph_id) + ");");
  tiling_func_.AddLine(indent +
                       "  OP_LOGI(OP_NAME, \"Update best perf to %lf, tiling key = %u, block dim = %u\", best_perf, "
                       "tiling_data.get_" +
                       tiling_key_prefix + "tiling_key(), cur_block_dim);");
  GenUpdateWorkspace(asc_graph_id, impl_graph_id);
  tiling_func_.AddLine(indent + "  return true;");
  tiling_func_.AddLine(indent + "}");
  tiling_func_.AddLine(indent + "return true;");
}

af::Status TilingCodeGenImpl::GenUpdatePerf(const size_t asc_graph_id, const size_t impl_graph_id,
                                            const std::vector<std::string> &groups_perf,
                                            const std::vector<std::pair<size_t, size_t>> &serialized_edges,
                                            const std::vector<std::string> &groups_block_num,
                                            const std::vector<std::string> &assign_max_block_num) {
  std::string objective_log;
  if (!IsScheduleResultEnableParallel(asc_graph_id, impl_graph_id)) {
    tiling_func_.AddLine(GenSumAllGroupsPerf(groups_perf));
    tiling_func_.AddLine("      double group_sum = cur_perf;");
    bool edge_model_enabled = false;
    if (!serialized_edges.empty()) {
      const auto penalty_code = GenSerializedEdgePenalty(groups_perf, serialized_edges, "      ");
      if (!penalty_code.empty()) {
        edge_model_enabled = true;
        tiling_func_.AddLine(penalty_code);
        tiling_func_.AddLine(
            "      OP_LOGD(OP_NAME, \"objective=group_sum(%lf)+edge_penalty(%lf)=%lf\", group_sum, "
            "serialized_edge_penalty, cur_perf);");
      }
    }

    std::string group_formula;
    for (size_t i = 0U; i < groups_perf.size(); ++i) {
      if (i != 0U) {
        group_formula += " + ";
      }
      group_formula += "group" + std::to_string(i);
    }
    std::string edge_base_formula = "0";
    if (edge_model_enabled) {
      edge_base_formula.clear();
      for (size_t i = 0U; i < serialized_edges.size(); ++i) {
        if (i != 0U) {
          edge_base_formula += " + ";
        }
        edge_base_formula += "max(group" + std::to_string(serialized_edges[i].first) + ", group" +
                             std::to_string(serialized_edges[i].second) + ")";
      }
    }
    objective_log = "    OP_LOGI(OP_NAME, \"The value of graph" + std::to_string(asc_graph_id) + "_result" +
                    std::to_string(impl_graph_id) + " is %lf objective(group_sum(" + group_formula +
                    ")=%lf + edge_model(" + std::string(SerializedEdgePenaltyConfig::kModelVersion) +
                    ", base=" + edge_base_formula + "=%lf, penalty=min(" +
                    std::to_string(SerializedEdgePenaltyConfig::kPenaltyCapCycles) + ", " +
                    std::to_string(SerializedEdgePenaltyConfig::kPenaltyRatio) + "*base)=%lf)=%lf\", cur_perf, ";
    if (edge_model_enabled) {
      objective_log += "group_sum, serialized_edge_base, serialized_edge_penalty, cur_perf);";
    } else {
      objective_log += "group_sum, 0.0, 0.0, cur_perf);";
    }
  } else {
    tiling_func_.AddLine(GenPerfUpdateCode(groups_perf, groups_block_num, "  "));
  }
  GenBestPerfUpdateCode(asc_graph_id, impl_graph_id, assign_max_block_num, "    ", objective_log);
  tiling_func_.AddLine("  }");  // 闭合 GetScheduleResult 函数
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDoGroupTilingFunction(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  // 检查是否需要生成二次Tiling支持（多Group且enable_group_parallel）
  if (!IsScheduleResultEnableParallel(asc_graph_id, impl_graph_id) || graph_info.size() <= 1) {
    return af::SUCCESS;  // 单Group或未使能Group并行，不需要生成DoGroupTiling
  }

  // 生成函数声明
  std::string func_decl = std::string(kInlineStr) + "bool DoGroupTiling" + std::to_string(impl_graph_id) + "(";
  func_decl += config_.tiling_data_type_name + " &tiling_data, ";
  func_decl += "const uint32_t ori_block_dim, ";
  func_decl += "const std::vector<int32_t> &case_ids_or_keys, ";
  func_decl += "std::unordered_map<int64_t, uint64_t> &workspace_map, ";
  func_decl += "std::vector<uint32_t> &output_tiling_keys, ";
  func_decl += "double secondary_ratio = 0.0) {";
  tiling_func_.AddLine(func_decl);
  tiling_func_.AddLine("  // 设置每个Group的block_dim和ub_size");

  // 声明每个Group的tiling_data引用并设置block_dim和ub_size
  for (const auto &group_info : graph_info) {
    std::string group_var = group_info.second.second + "_tiling_data";
    tiling_func_.AddLine("  auto &" + group_var + " = tiling_data." + group_var + ";");
    tiling_func_.AddLine("  " + group_var + ".set_block_dim(ori_block_dim);");
    tiling_func_.AddLine("  " + group_var + ".set_ub_size(tiling_data.get_ub_size());");
  }

  // 二次Tiling比例设置
  tiling_func_.AddLine("  // 如果是二次Tiling，设置调整后的比例");
  tiling_func_.AddLine("  if (secondary_ratio > 0.0) {");
  tiling_func_.AddLine("    g_secondary_tiling_ratio = secondary_ratio;");
  tiling_func_.AddLine("  }");

  // 生成调用所有Group的GetTiling部分
  GenDoGroupTilingGetTilingCalls(graph_info);

  // 失败处理
  GenDoGroupTilingFailureHandler(graph_info);

  return af::SUCCESS;
}

void TilingCodeGenImpl::GenDoGroupTilingGetTilingCalls(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  tiling_func_.AddLine("  // 调用所有Group的GetTiling");
  std::string get_tiling_condition = "  if (";
  size_t group_index = 0;
  for (const auto &group_info : graph_info) {
    std::string group_var = group_info.second.second + "_tiling_data";
    std::string group_class = group_info.second.first;
    get_tiling_condition += "(" + group_class + "::GetTiling(" + group_var + ", workspace_map, case_ids_or_keys[" +
                            std::to_string(group_index) + "], nullptr))";
    if (group_index < graph_info.size() - 1) {
      get_tiling_condition += " && ";
    }
    group_index++;
  }
  get_tiling_condition += ") {";
  tiling_func_.AddLine(get_tiling_condition);

  // 保存tiling_keys（仅首次Tiling）
  tiling_func_.AddLine("    // 仅首次Tiling时保存tiling_keys");
  tiling_func_.AddLine("    if (secondary_ratio <= 0.0) {");
  group_index = 0;
  for (const auto &group_info : graph_info) {
    std::string group_var = group_info.second.second + "_tiling_data";
    tiling_func_.AddLine("      output_tiling_keys[" + std::to_string(group_index) + "] = " + group_var +
                         ".get_tiling_key();");
    group_index++;
  }
  tiling_func_.AddLine("    }");

  // 重置全局变量
  tiling_func_.AddLine("    // 重置全局变量");
  tiling_func_.AddLine("    g_secondary_tiling_ratio = 0.0;");
  tiling_func_.AddLine("    return true;");
  tiling_func_.AddLine("  }");
}

void TilingCodeGenImpl::GenDoGroupTilingFailureHandler(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  // 失败处理
  tiling_func_.AddLine("  // 失败时清零");
  tiling_func_.AddLine("  g_secondary_tiling_ratio = 0.0;");
  for (const auto &group_info : graph_info) {
    std::string group_var = group_info.second.second + "_tiling_data";
    tiling_func_.AddLine("  " + group_var + ".set_block_dim(0);");
  }
  tiling_func_.AddLine("  return false;");
  tiling_func_.AddLine("}");
}

void TilingCodeGenImpl::GenGroupParallelFirstTiling(const size_t impl_graph_id) {
  tiling_func_.AddLine("  // ========== 首次Tiling ==========");
  tiling_func_.AddLine("  std::vector<int32_t> tiling_case_ids(group_num, tiling_case_id);");
  tiling_func_.AddLine("  std::vector<uint32_t> first_tiling_keys(group_num);");
  tiling_func_.AddLine("  if (!DoGroupTiling" + std::to_string(impl_graph_id) +
                       "(tiling_data, ori_block_dim, tiling_case_ids, workspace_map, first_tiling_keys)) {");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
}

void TilingCodeGenImpl::GenGroupParallelSecondTiling(
    const size_t impl_graph_id, const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  tiling_func_.AddLine("    // ========== 二次Tiling ==========");
  tiling_func_.AddLine("    double original_ratio = 1.0 / group_num;");
  tiling_func_.AddLine("    double adjusted_ratio = original_ratio * ori_block_dim / first_total_block_dim;");
  tiling_func_.AddLine("    adjusted_ratio = std::min(1.0, adjusted_ratio);");
  tiling_func_.AddLine("    std::vector<int32_t> first_keys_as_int(group_num);");
  size_t group_index = 0;
  for ([[maybe_unused]] const auto &group_info : graph_info) {
    tiling_func_.AddLine("    first_keys_as_int[" + std::to_string(group_index) +
                         "] = static_cast<int32_t>(first_tiling_keys[" + std::to_string(group_index) + "]);");
    group_index++;
  }
  tiling_func_.AddLine("    std::vector<uint32_t> dummy_keys(group_num);");
  tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"Begin to second tiling for result" + std::to_string(impl_graph_id) +
                       " core_usage_ratio=%lf, adjusted_ratio=%lf.\", original_ratio, adjusted_ratio);");
  tiling_func_.AddLine("    if (!DoGroupTiling" + std::to_string(impl_graph_id) +
                       "(tiling_data, ori_block_dim, first_keys_as_int, workspace_map, dummy_keys, adjusted_ratio)) {");
  for (const auto &group_info : graph_info) {
    tiling_func_.AddLine("      " + group_info.second.second + "_tiling_data.set_block_dim(0);");
  }
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
}

af::Status TilingCodeGenImpl::GenScheduleGroupDoTiling(std::string &check_cond, const std::string &hardware_param,
                                                       const std::string &schedule_result_prefix) {
  std::string tiling_hyphens = "&&";
  if (check_cond.empty()) {
    tiling_hyphens = "";
  }
  std::string first_param = hardware_param + "_tiling_data, ";
  std::string cache_param;
  const auto &key = schedule_result_prefix;
  for (const auto &pair : cache_reuse_info_) {
    if (pair.first == key || pair.second == key) {
      // 看是否为复用方/被复用方
      cache_param = ", &" + pair.second + "_Cache";
      break;
    }
  }
  std::string workspace_param;
  if (!is_uniq_group_) {
    workspace_param = "workspace_map, ";
  }
  std::string tiling = "(" + schedule_result_prefix + "::GetTiling(" + first_param + workspace_param +
                       "tiling_case_id" + cache_param + "))";
  check_cond += (tiling_hyphens + tiling);
  return af::SUCCESS;
}

// 辅助函数：生成Group并行首次Tiling后的声明和计算
void TilingCodeGenImpl::GenGroupParallelFirstTilingDecls(
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  // 声明每个Group的tiling_data引用
  for (const auto &group_info : graph_info) {
    tiling_func_.AddLine("  auto &" + group_info.second.second + "_tiling_data = tiling_data." +
                         group_info.second.second + "_tiling_data;");
  }

  // 计算首次Tiling的总block_dim
  tiling_func_.AddLine("  uint32_t first_total_block_dim = ");
  for (auto it = graph_info.begin(); it != graph_info.end(); ++it) {
    tiling_func_.AddLine("    " + it->second.second + "_tiling_data.get_block_dim()" +
                         (std::next(it) != graph_info.end() ? " + " : ";"));
  }

  // 判断是否需要二次Tiling
  tiling_func_.AddLine("  // 判断是否需要二次Tiling");
  tiling_func_.AddLine("  const double kSecondaryTilingThreshold = 0.8;");
  tiling_func_.AddLine("  double core_usage_ratio = static_cast<double>(first_total_block_dim) / ori_block_dim;");
  tiling_func_.AddLine("  if (core_usage_ratio < kSecondaryTilingThreshold) {");
}

// 辅助函数：生成单Group场景的tiling处理
af::Status TilingCodeGenImpl::GenSingleGroupScheduleResult(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<std::string, std::set<std::string>> &hardware_map) {
  const auto &var_relation = var_relations_[asc_graph_id][impl_graph_id];
  std::string check_cond;
  std::vector<std::string> assign_max_block_num;
  std::vector<std::string> groups_perf;
  std::vector<std::string> groups_block_num;
  std::map<size_t, size_t> group_perf_positions;
  for (const auto &group_info : graph_info) {
    auto [input_vars_set_code, need_update_second_group_input_vars] =
        ProcessVarRelations(graph_info, var_relation, group_info.first);
    std::string cur_block;
    tiling_func_.AddLine("  auto &" + group_info.second.second + "_tiling_data = tiling_data." +
                         group_info.second.second + "_tiling_data;");
    const auto &hardware_iter = hardware_map.find(group_info.second.second);
    if (hardware_iter != hardware_map.cend()) {
      GenSetHardwareCodes(group_info.second.second, hardware_iter->second);
      if (need_update_second_group_input_vars) {
        RequireVarRelationSystemHeaders();
        std::string tiling_hyphens = check_cond.empty() ? "" : "&&";
        check_cond += (tiling_hyphens + "(" + input_vars_set_code + ")");
      }
      GE_ASSERT_SUCCESS(GenScheduleGroupDoTiling(check_cond, group_info.second.second, group_info.second.first),
                        "Gen schedule group do tiling failed, graph id[%zu], impl id[%zu]", asc_graph_id,
                        impl_graph_id);
      groups_perf.emplace_back(GenGetScheduleGroupPerf(group_info.second.first, group_info.second.second));
      group_perf_positions[group_info.first] = groups_perf.size() - 1U;
      assign_max_block_num.emplace_back(GenCurMaxBlockDim(group_info.second.second, groups_block_num, cur_block));
      groups_block_num.emplace_back(GenGetCurBlockDim(group_info.second.second));
    }
  }
  GE_ASSERT_TRUE(!groups_perf.empty(), "groups_perf size of asc_graph_id %zu impl_graph_id %zu is 0", asc_graph_id,
                 impl_graph_id);
  GE_ASSERT_EQ(groups_block_num.size(), groups_perf.size());
  tiling_func_.AddLine("  if (" + (check_cond.empty() ? "true" : check_cond) + ") {");
  const bool enable_group_parallel = IsScheduleResultEnableParallel(asc_graph_id, impl_graph_id);
  std::vector<std::pair<size_t, size_t>> group_edges =
      GetSerializedGroupEdges(graph_info, var_relation, enable_group_parallel);
  if (!enable_group_parallel) {
    const auto workspace_edges = GetWorkspaceSharedGroupEdges(asc_graph_id, impl_graph_id, graph_info);
    std::set<std::pair<size_t, size_t>> merged_edges(group_edges.cbegin(), group_edges.cend());
    merged_edges.insert(workspace_edges.cbegin(), workspace_edges.cend());
    group_edges.assign(merged_edges.cbegin(), merged_edges.cend());
  }
  std::vector<std::pair<size_t, size_t>> serialized_edges;
  serialized_edges.reserve(group_edges.size());
  for (const auto &edge : group_edges) {
    const auto src_pos = group_perf_positions.find(edge.first);
    const auto dst_pos = group_perf_positions.find(edge.second);
    if (src_pos != group_perf_positions.end() && dst_pos != group_perf_positions.end()) {
      serialized_edges.emplace_back(src_pos->second, dst_pos->second);
    }
  }
  GE_ASSERT_SUCCESS(
      GenUpdatePerf(asc_graph_id, impl_graph_id, groups_perf, serialized_edges, groups_block_num, assign_max_block_num),
      "Gen update perf failed, asc_graph_id %zu impl_graph_id %zu", asc_graph_id, impl_graph_id);
  GenGetScheduleResultTail(graph_info);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetScheduleResult(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<std::string, std::set<std::string>> &hardware_map) {
  // 判断是否为Group并行场景（多Group且enable_group_parallel）
  const bool is_group_parallel = IsScheduleResultEnableParallel(asc_graph_id, impl_graph_id) && graph_info.size() > 1;

  std::string func_define(kInlineStr);
  func_define.append("bool GetScheduleResult")
      .append(std::to_string(impl_graph_id))
      .append("(const uint32_t ori_block_dim, const int32_t tiling_case_id,")
      .append(config_.tiling_data_type_name)
      .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim) {");
  tiling_func_.AddLine(func_define);
  tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map{};");
  GenCacheInit();

  if (is_group_parallel) {
    // ========== Group并行场景：使用DoGroupTiling + 二次Tiling ==========
    const size_t group_num = graph_info.size();
    tiling_func_.AddLine("  constexpr size_t group_num = " + std::to_string(group_num) + ";");

    // ========== 首次Tiling ==========
    GenGroupParallelFirstTiling(impl_graph_id);
    GenGroupParallelFirstTilingDecls(graph_info);

    // ========== 二次Tiling ==========
    GenGroupParallelSecondTiling(impl_graph_id, graph_info);

    // ========== cache line conflict helpers ==========
    GE_ASSERT_SUCCESS(GenConflictGroupHelpers(asc_graph_id, impl_graph_id, graph_info));

    // ========== perf计算和更新 ==========
    tiling_func_.AddLine("  // ========== perf计算和更新 ==========");
    GenGetScheduleResultPerfAndTail(asc_graph_id, impl_graph_id, graph_info);
  } else {
    // ========== 单Group场景：原有逻辑 ==========
    GE_ASSERT_SUCCESS(GenSingleGroupScheduleResult(asc_graph_id, impl_graph_id, graph_info, hardware_map),
                      "Gen single group schedule result failed, asc_graph_id %zu impl_graph_id %zu", asc_graph_id,
                      impl_graph_id);
  }
  return af::SUCCESS;
}

// 辅助函数：生成perf计算和更新部分
af::Status TilingCodeGenImpl::GenGetScheduleResultPerfAndTail(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  std::vector<std::string> groups_perf;
  std::vector<std::string> groups_block_num;
  std::vector<std::string> assign_max_block_num;
  std::string cur_block;

  for (const auto &group_info : graph_info) {
    groups_perf.emplace_back(GenGetScheduleGroupPerf(group_info.second.first, group_info.second.second));
    assign_max_block_num.emplace_back(GenCurMaxBlockDim(group_info.second.second, groups_block_num, cur_block));
    groups_block_num.emplace_back(GenGetCurBlockDim(group_info.second.second));
  }

  // 收集每个group的冲突标志
  std::vector<std::string> groups_conflict_flags;
  for (const auto &group_info : graph_info) {
    groups_conflict_flags.emplace_back(
        GenConflictGroupInvoke(asc_graph_id, impl_graph_id, group_info.first, group_info.second.second));
  }
  // 使用混合聚合生成perf更新代码（冲突组求和 + 普通组并行合并）
  tiling_func_.AddLine(GenMixedPerfUpdateCode(groups_perf, groups_block_num, groups_conflict_flags, "  "));
  // 使用公共函数生成best perf更新代码
  GenBestPerfUpdateCode(asc_graph_id, impl_graph_id, assign_max_block_num, "  ");
  tiling_func_.AddLine("}");  // 闭合 GetScheduleResult 函数

  return af::SUCCESS;
}

void TilingCodeGenImpl::GenPGOByCoreNumDoTiling(
    const std::pair<size_t, std::pair<std::string, std::string>> &group_info, const uint32_t group_index,
    const size_t asc_graph_id, const size_t impl_graph_id) {
  auto hard_ware_param = group_info.second.second;
  auto schedule_result_prefex = group_info.second.first;
  auto group_id = group_info.first;
  uint32_t index = 0U;

  for (auto &model_info : tiling_model_info_) {
    if (model_info.schedule_group_ident.asc_graph_id != asc_graph_id ||
        model_info.schedule_group_ident.impl_graph_id != impl_graph_id ||
        model_info.schedule_group_ident.group_id != group_id) {
      continue;
    }

    tiling_func_.AddLine("    " + config_.tiling_data_type_name + " tiling_data_tmp" + std::to_string(index) +
                         "= tiling_data;");
    tiling_func_.AddLine("    auto sub_tiling_data_tmp" + std::to_string(index) + "= tiling_data_tmp" +
                         std::to_string(index) + "." + hard_ware_param + "_tiling_data" + ";");
    tiling_func_.AddLine("    sub_tiling_data_tmp" + std::to_string(index) + ".set_tiling_key(" +
                         std::to_string(model_info.tiling_case_id) + ");");
    tiling_func_.AddLine("    if (" + schedule_result_prefex + "::GetTiling(sub_tiling_data_tmp" +
                         std::to_string(index) + ", workspace_map, " + std::to_string(model_info.tiling_case_id) +
                         ", nullptr)) { ");
    std::string tiling_data_add("      ");
    tiling_data_add.append("tiling_data_tmp" + std::to_string(index) + ".")
        .append(hard_ware_param)
        .append("_tiling_data=sub_tiling_data_tmp" + std::to_string(index) + ";");
    tiling_func_.AddLine(tiling_data_add);
    tiling_func_.AddLine("      tiling_data_list_tmp" + std::to_string(group_index) + ".push_back(tiling_data_tmp" +
                         std::to_string(index) + ");");
    tiling_func_.AddLine("    }");
    index++;
  }
}

void TilingCodeGenImpl::GenPGOByCoreNumFunctionHead(size_t impl_graph_id) {
  std::string func_define(kInlineStr);
  func_define.append("bool GetScheduleResult")
      .append(std::to_string(impl_graph_id) + "PGOByCoreNum")
      .append("(std::vector<" + config_.tiling_data_type_name + ">& tiling_data_list, ")
      .append(config_.tiling_data_type_name + " tiling_data")
      .append(") {");
  tiling_func_.AddLine(func_define);
  tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map{};");
  tiling_func_.AddLine("  " + config_.tiling_data_type_name + " tiling_data_tmp = tiling_data;");
  tiling_func_.AddLine("  std::vector<" + config_.tiling_data_type_name +
                       "> tiling_data_list_tmp0 = {tiling_data_tmp};");
}

bool TilingCodeGenImpl::TryGenPGOByCoreNumReuseTiling(size_t asc_graph_id, size_t impl_graph_id, size_t group_id,
                                                      uint32_t group_index) {
  const auto iter = std::find_if(tiling_model_info_.cbegin(), tiling_model_info_.cend(),
                                 [asc_graph_id, impl_graph_id, group_id](const auto &model_info) {
                                   const auto &ident = model_info.schedule_group_ident;
                                   return ident.asc_graph_id == asc_graph_id && ident.impl_graph_id == impl_graph_id &&
                                          ident.group_id == group_id && model_info.reuse_schedule_group != nullptr &&
                                          model_info.reuse_schedule_group->IsReuseGroup(ident);
                                 });
  if (iter == tiling_model_info_.cend()) {
    return false;
  }
  const auto &reuse_ident = iter->reuse_schedule_group->reuse_group_ident;
  const auto &current_ident = iter->schedule_group_ident;
  const auto source = "tiling_data." + reuse_ident.GetItemPrefix() + "_tiling_data";
  const auto target = current_ident.GetItemPrefix() + "_tiling_data";
  tiling_func_.AddLine("    auto reuse_tiling_data = RefToRef<" + reuse_ident.GetGroupPrefix() + "TilingData, " +
                       current_ident.GetGroupPrefix() + "TilingData>(" + source + ");");
  tiling_func_.AddLine("    tiling_data." + target + " = reuse_tiling_data;");
  tiling_func_.AddLine("    tiling_data_list_tmp" + std::to_string(group_index) + ".push_back(tiling_data);");
  return true;
}

void TilingCodeGenImpl::GenPGOByCoreNumGetScheduleResult(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<std::string, std::set<std::string>> &hardware_map,
    const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &var_relation) {
  GenPGOByCoreNumFunctionHead(impl_graph_id);
  uint32_t group_index = 0U;

  for (const auto &group_info : graph_info) {
    group_index++;
    tiling_func_.AddLine("  std::vector<" + config_.tiling_data_type_name + "> tiling_data_list_tmp" +
                         std::to_string(group_index) + ";");
    tiling_func_.AddLine("  for (auto &tiling_data : tiling_data_list_tmp" + std::to_string(group_index - 1) + ") {");
    if (TryGenPGOByCoreNumReuseTiling(asc_graph_id, impl_graph_id, group_info.first, group_index)) {
      tiling_func_.AddLine("  }");
      tiling_func_.AddLine("");
      continue;
    }
    auto [input_vars_set_code, need_update_second_group_input_vars] =
        ProcessVarRelationsStatement(graph_info, var_relation, group_info.first, "    ", {"continue;"});
    std::string tiling_item_name = group_info.second.second + "_tiling_data";
    tiling_func_.AddLine("    auto &" + tiling_item_name + " = tiling_data." + tiling_item_name + ";");
    const auto &hardware_iter = hardware_map.find(group_info.second.second);
    if (hardware_iter != hardware_map.cend()) {
      for (const auto &hardware_name : hardware_iter->second) {
        std::string set_hardware_code("    ");
        set_hardware_code.append(tiling_item_name).append(".set_").append(hardware_name);
        std::string hardware_val = "(tiling_data.get_" + hardware_name + "());";
        tiling_func_.AddLine(set_hardware_code.append(hardware_val));
      }
      if (need_update_second_group_input_vars) {
        RequireVarRelationSystemHeaders();
        tiling_func_.AddLine(input_vars_set_code);
      }
      GenPGOByCoreNumDoTiling(group_info, group_index, asc_graph_id, impl_graph_id);
    }
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("");
  }
  tiling_func_.AddLine("  for (auto &tiling_data : tiling_data_list_tmp" + std::to_string(group_index) + ") {");
  if (enable_group_parallels_[asc_graph_id][impl_graph_id]) {
    tiling_func_.AddLine("    const uint32_t ori_block_dim = tiling_data.get_block_dim();");
  }
  GenPGOUpdateTilingInfo(asc_graph_id, impl_graph_id, graph_info);
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  tiling_data_list.insert(tiling_data_list.end(), tiling_data_list_tmp" +
                       std::to_string(group_index) + ".begin(), tiling_data_list_tmp" + std::to_string(group_index) +
                       ".end());");
  tiling_func_.AddLine("  return !tiling_data_list_tmp" + std::to_string(group_index) + ".empty();");
  tiling_func_.AddLine("}");
}

void TilingCodeGenImpl::GenPGOUpdateTilingInfo(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  // Serial groups reuse the same launch block range; the outer block_dim is therefore the group maximum.
  for (const auto &line : GenScheduleResultBlockDimCode(graph_info)) {
    tiling_func_.AddLine(line);
  }
  GenUpdateWorkspace(asc_graph_id, impl_graph_id);
  if (enable_group_parallels_[asc_graph_id][impl_graph_id]) {
    tiling_func_.AddLine("      ArrangeBlockOffsetsAscGraph" + std::to_string(asc_graph_id) + "Result" +
                         std::to_string(impl_graph_id) + "(tiling_data, ori_block_dim);");
    return;
  }
}

void TilingCodeGenImpl::GenFillOtherGroupsGetTiling(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
    const std::map<std::string, std::set<std::string>> &hardware_map) {
  auto current_group_iter = graph_info.find(group_info.first);
  if (current_group_iter == graph_info.end()) {
    GELOGE(ge::GRAPH_FAILED, "Current graph id not found in graph info.");
    return;
  }
  const std::string candidate_begin_name = "candidate_begin_index" + std::to_string(group_info.first);
  auto emit_get_tiling = [&](const auto &group_iter) {
    const auto &hw_iter = hardware_map.find(group_iter.second.second);
    if (hw_iter != hardware_map.cend()) {
      GenSetHardwareCodes(std::string("    tiling_data.") + group_iter.second.second, hw_iter->second);
    } else {
      GELOGW("Hardware info not found for group %s.", group_iter.second.second.c_str());
    }
    auto [input_vars_set_code, need_update] = ProcessVarRelationsStatement(
        graph_info, var_relations_[asc_graph_id][impl_graph_id], group_iter.first, "      tiling_data.",
        {"valid_candidates[candidate_index - " + candidate_begin_name + "] = false;", "continue;"});
    if (need_update) {
      RequireVarRelationSystemHeaders();
      tiling_func_.AddLine(input_vars_set_code);
    }
    tiling_func_.AddLine("      has_solution = " + group_iter.second.first + "::GetTiling(tiling_data." +
                         group_iter.second.second + "_tiling_data, workspace_map, -1);");
    tiling_func_.AddLine("      if (!has_solution) {");
    tiling_func_.AddLine("        OP_LOGI(OP_NAME, \"No solution for " + group_info.second.second + " at " +
                         group_iter.second.second + "\");");
    tiling_func_.AddLine("        valid_candidates[candidate_index - " + candidate_begin_name + "] = false;");
    tiling_func_.AddLine("        continue;");
    tiling_func_.AddLine("      }");
  };
  for (auto group_iter = std::next(current_group_iter); group_iter != graph_info.end(); ++group_iter) {
    emit_get_tiling(*group_iter);
  }
}

af::Status TilingCodeGenImpl::GenPGOGetScheduleResultPerGroup(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
    const std::map<std::string, std::set<std::string>> &hardware_map) {
  const std::string candidate_begin_name = "candidate_begin_index" + std::to_string(group_info.first);
  tiling_func_.AddLine("    bool has_solution = true;");
  tiling_func_.AddLine("    std::vector<bool> valid_candidates(tiling_data_list_tmp.size() - " + candidate_begin_name +
                       ", true);");
  tiling_func_.AddLine("    for (size_t candidate_index = " + candidate_begin_name +
                       "; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {");
  tiling_func_.AddLine("      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];");
  tiling_func_.AddLine("      auto &tiling_data = tiling_data_perf.tiling_data;");
  tiling_func_.AddLine("      std::unordered_map<int64_t, uint64_t> workspace_map;");
  tiling_func_.AddLine("      workspace_map.reserve(workspace_map_filter_use.size());");
  tiling_func_.AddLine("      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());");
  GenFillOtherGroupsGetTiling(asc_graph_id, impl_graph_id, graph_info, group_info, hardware_map);
  GenPGOUpdateTilingInfo(asc_graph_id, impl_graph_id, graph_info);
  tiling_func_.AddLine("      uint64_t workspaceSizeTmp = 0U;");
  tiling_func_.AddLine("      if (!GetWorkspaceSize(tiling_data, workspaceSizeTmp)) {");
  tiling_func_.AddLine("        return false;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("      if (workspaceSizeTmp > workspaceSize) {");
  tiling_func_.AddLine("        workspaceSize = workspaceSizeTmp;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    std::vector<AutofuseTilingDataPerf> valid_tiling_data_list;");
  tiling_func_.AddLine("    valid_tiling_data_list.reserve(valid_candidates.size());");
  tiling_func_.AddLine("    for (size_t candidate_index = " + candidate_begin_name +
                       "; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {");
  tiling_func_.AddLine("      const size_t candidate_offset = candidate_index - " + candidate_begin_name + ";");
  tiling_func_.AddLine("      if (candidate_offset < valid_candidates.size() && valid_candidates[candidate_offset]) {");
  tiling_func_.AddLine("        valid_tiling_data_list.push_back(tiling_data_list_tmp[candidate_index]);");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    if (__builtin_add_overflow(workspaceSize, 16ULL * 1024ULL * 1024ULL, &workspaceSize)) {");
  tiling_func_.AddLine("      OP_LOGE(OP_NAME, \"Workspace size overflow in PGO group profiling.\");");
  tiling_func_.AddLine("      return false;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    if (PgoConfig::Instance().batch_callback && !valid_tiling_data_list.empty()) {");
  tiling_func_.AddLine(
      "      if (PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, "
      "workspaceSize, &valid_tiling_data_list) != 0) {");
  tiling_func_.AddLine("        return false;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("    for (auto &tiling_data_perf : valid_tiling_data_list) {");
  tiling_func_.AddLine("      tiling_data_list.push_back(tiling_data_perf);");
  tiling_func_.AddLine("      if (tiling_data_perf.best_perf < best_perf) {");
  tiling_func_.AddLine("        tiling_data = tiling_data_perf.tiling_data;");
  tiling_func_.AddLine("        best_perf = tiling_data_perf.best_perf;");
  tiling_func_.AddLine("      }");
  tiling_func_.AddLine("    }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOScheduleGroupSearchEntry(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<std::string, std::set<std::string>> &hardware_map,
    const std::pair<size_t, std::pair<std::string, std::string>> &group_info, const std::string &result_name) {
  std::string tiling_item_name = group_info.second.second + "_tiling_data";
  tiling_func_.AddLine("  auto &" + tiling_item_name + " = tiling_data." + tiling_item_name + ";");
  const auto &hardware_iter = hardware_map.find(group_info.second.second);
  if (hardware_iter == hardware_map.cend()) {
    return af::SUCCESS;
  }
  GenSetHardwareCodes(group_info.second.second, hardware_iter->second);
  auto [input_vars_set_code, need_update] =
      ProcessVarRelationsStatement(graph_info, var_relations_[asc_graph_id][impl_graph_id], group_info.first,
                                   "  tiling_data.", {"return tiling_data_list.size() > candidate_count_before;"});
  if (need_update) {
    RequireVarRelationSystemHeaders();
    tiling_func_.AddLine(input_vars_set_code);
  }
  const bool is_reuse =
      std::any_of(tiling_model_info_.cbegin(), tiling_model_info_.cend(), [&group_info](const att::ModelInfo &mi) {
        return mi.reuse_schedule_group != nullptr && mi.reuse_schedule_group->IsReuseGroup(mi.schedule_group_ident) &&
               mi.schedule_group_ident.GetGroupPrefix() == group_info.second.first;
      });
  if (config_.is_inductor_scene && is_reuse) {
    return af::SUCCESS;
  }
  const std::string candidate_begin_name = "candidate_begin_index" + std::to_string(group_info.first);
  tiling_func_.AddLine("  size_t " + candidate_begin_name + " = tiling_data_list_tmp.size();");
  if (is_reuse) {
    tiling_func_.AddLine("  auto " + result_name + " = " + GenPGOReuseGroupProfile(group_info.second.first) + ";");
  } else {
    tiling_func_.AddLine("  auto " + result_name + " = " +
                         GenPGOScheduleGroupDoTiling(group_info.second.second, group_info.second.first) + ";");
  }
  tiling_func_.AddLine("  if (" + result_name + ") {");
  GE_ASSERT_SUCCESS(GenPGOGetScheduleResultPerGroup(asc_graph_id, impl_graph_id, graph_info, group_info, hardware_map));
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOGetScheduleResult(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
    const std::map<std::string, std::set<std::string>> &hardware_map) {
  std::string func_define(kInlineStr);
  func_define.append("bool GetScheduleResult")
      .append(std::to_string(impl_graph_id) + "PGO")
      .append(
          "(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t "
          "tiling_case_id,")
      .append(config_.tiling_data_type_name)
      .append(" &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim,")
      .append(
          "void* stream, uint64_t workspaceSize, std::vector<uint32_t*> multi_group_block_dim_list = {}, const "
          "SearchConfig *search_cfg=nullptr) {");
  tiling_func_.AddLine(func_define);
  tiling_func_.AddLine("  (void)cur_perf; (void)cur_block_dim;");
  tiling_func_.AddLine("  const size_t candidate_count_before = tiling_data_list.size();");
  uint32_t group_index = 0U;
  tiling_func_.AddLine("  std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{};");
  tiling_func_.AddLine("  workspaceSize = 0;");
  tiling_func_.AddLine("  std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{};");
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
  tiling_func_.AddLine("  tiling_data.set_" + tiling_key_prefix + "tiling_key(" + std::to_string(impl_graph_id) + ");");
  for (const auto &group_info : graph_info) {
    std::string result_name = "result" + std::to_string(group_index);
    GE_ASSERT_SUCCESS(
        GenPGOScheduleGroupSearchEntry(asc_graph_id, impl_graph_id, graph_info, hardware_map, group_info, result_name));
    group_index++;
  }
  tiling_func_.AddLine("  return tiling_data_list.size() > candidate_count_before;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenGetScoreFuncs(
    const size_t asc_graph_id,
    const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map) {
  auto &schedule_results_score_func = score_funcs_[kModelInfoLevel::K_SCHEDULE_RESULT_LEVEL][asc_graph_id];
  for (size_t i = 0UL; i < namespace_map.size(); i++) {
    tiling_func_.AddLine("namespace " + GetScheduleResultPrefix(asc_graph_id, i) + " {");
    auto &score_func = schedule_results_score_func[i];
    if (score_func.empty()) {
      score_func =
          "int32_t CalcScore(" + config_.tiling_data_type_name + " &tiling_data) { (void)tiling_data; return 0;}";
    }
    // Inject (void)tiling_data; for externally-provided score funcs that don't use the parameter
    if (!score_func.empty() && score_func.find("(void)tiling_data") == std::string::npos) {
      auto brace_pos = score_func.find('{');
      if (brace_pos != std::string::npos) {
        score_func.insert(brace_pos + 1, " (void)tiling_data;");
      }
    }
    tiling_func_.AddLine(schedule_results_score_func[i]);
    tiling_func_.AddLine("}");
  }
}

void TilingCodeGenImpl::GenGetScoreFuncsCalling(
    const size_t asc_graph_id,
    const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map) {
  tiling_func_.AddLine("  int32_t scores[" + std::to_string(namespace_map.size()) + "]{};");
  for (size_t i = 0UL; i < namespace_map.size(); i++) {
    tiling_func_.AddLine("  scores[" + std::to_string(i) + "] = " + GetScheduleResultPrefix(asc_graph_id, i) +
                         "::CalcScore(tiling_data);");
  }
}

void TilingCodeGenImpl::GenGetMaxScoreIndex(const AscGraphNamepspaceMap &namespace_map) {
  tiling_func_.AddLine("  int32_t max_index = 0L;");
  if (namespace_map.size() > 1) {
    tiling_func_.AddLine("   for (int32_t index = 1; index < " + std::to_string(namespace_map.size()) + "; index++) {");
    tiling_func_.AddLine("    if (scores[index] > scores[max_index]) {");
    tiling_func_.AddLine("      max_index = index;");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine("  }");
  }
}

void TilingCodeGenImpl::GenScheduleResultGetTilingCalling(const std::string &index, const std::string &ident) {
  tiling_func_.AddLine(ident + "  if (kScheduleResultFunctions[" + index +
                       "](ori_block_dim, tiling_case_id, tiling_data, cur_perf, best_perf, "
                       "cur_block_dim)) {");
  tiling_func_.AddLine(ident + "    auto res = GetResultSummary(best_perf, tiling_data);");
  GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, ident + "    ");
  GenDurationPrintCode(ident + "    ");
  GenDurationClearCode(ident + "    ");
  tiling_func_.AddLine(ident + "    return res;");
  tiling_func_.AddLine(ident + "  }");
}

af::Status TilingCodeGenImpl::GenGetAllSchedulesResults(const AscGraphNamepspaceMap &namespace_map) {
  std::string chosen_index =
      (config_.force_schedule_result < 0) ? "max_index" : std::to_string(config_.force_schedule_result);
  if (NeedGenScoreFunc(score_funcs_)) {
    GenScheduleResultGetTilingCalling(chosen_index);
  }
  if (config_.force_schedule_result >= 0) {
    GELOGI("Force schedule result %ld for op %s", config_.force_schedule_result, config_.tiling_data_type_name.c_str());
    GE_ASSERT_TRUE(config_.force_schedule_result < static_cast<int32_t>(namespace_map.size()),
                   "Force schedule "
                   "result[%ld] should be less than result size[%zu]",
                   config_.force_schedule_result, namespace_map.size());
    tiling_func_.AddLine("  auto got_result = kScheduleResultFunctions[" + chosen_index +
                         "](ori_block_dim, tiling_case_id, tiling_data, cur_perf, "
                         "best_perf, cur_block_dim);");
    tiling_func_.AddLine("  if (!got_result) {");
    tiling_func_.AddLine("    OP_LOGW(OP_NAME, \"Schedule result" + std::to_string(config_.force_schedule_result) +
                         " cannot be found for op\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
    return af::SUCCESS;
  }
  tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
  if (NeedGenScoreFunc(score_funcs_)) {
    tiling_func_.AddLine("    if (max_index == index) {");
    tiling_func_.AddLine("      continue;");
    tiling_func_.AddLine("    }");
  }
  tiling_func_.AddLine(
      "    (void)kScheduleResultFunctions[index](ori_block_dim, tiling_case_id, tiling_data, cur_perf, "
      "best_perf, cur_block_dim);");
  tiling_func_.AddLine("  }");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenEnableGroupParallelFunctions(const FusedGraphNamespaceMap &namespace_map) {
  size_t asc_graph_id = 0UL;
  for (const auto &asc_graph_namespace_map : namespace_map) {
    std::stringstream ss;
    for (const auto &result_id_and_groups : asc_graph_namespace_map.second) {
      const auto &groups = result_id_and_groups.second;
      if (enable_group_parallels_[asc_graph_id][result_id_and_groups.first]) {
        // 先简单按顺序遍历，非最优
        ss << "void ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id_and_groups.first
           << "(AutofuseTilingData &t, uint32_t aiv_num) {" << std::endl;
        ss << "  uint32_t block_offset = 0U;" << std::endl;
        ss << "  uint32_t block_dim = 0U;" << std::endl;
        ss << "  uint32_t max_block_dim = aiv_num;" << std::endl;
        // The outer block range is derived from the group offsets.  Do not
        // seed it from t.block_dim, which may still contain a PGO probe value.
        ss << "  uint32_t actual_max_block_dim = 0U;" << std::endl;
        for (const auto &group_id_and_names : groups) {
          const auto group_id = group_id_and_names.first;
          const auto &sub_tiling_data = "t." + group_id_and_names.second.second + "_tiling_data";
          const auto var_name = std::string("sub_tiling_data_") + std::to_string(group_id);
          ss << "  block_dim = " << sub_tiling_data << ".get_block_dim();" << std::endl;
          ss << "  " << sub_tiling_data << ".set_ub_size(block_offset); // reuse ub_size as block_offset" << std::endl;
          ss << "  block_offset += block_dim;" << std::endl;
          // block不够则回绕
          ss << "  if (block_offset > max_block_dim) {" << std::endl;
          ss << "    block_offset = block_offset - max_block_dim;" << std::endl;
          ss << "    actual_max_block_dim = max_block_dim;" << std::endl;
          ss << "  }" << std::endl;
          ss << "  actual_max_block_dim = std::max(actual_max_block_dim, block_offset);" << std::endl;
        }
        ss << "  t.set_block_dim(actual_max_block_dim);" << std::endl;
        ss << "}" << std::endl;
      }
    }
    tiling_func_.AddLine(ss.str());
    asc_graph_id++;
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenEnableGroupParallelInvoke(size_t asc_graph_id,
                                                           const AscGraphNamepspaceMap &asc_graph_namespace_map) {
  for (const auto &result_id_and_groups : asc_graph_namespace_map) {
    const auto result_id = result_id_and_groups.first;
    if (enable_group_parallels_[asc_graph_id][result_id]) {
      std::stringstream ss;
      ss << "  if (tiling_data.get_graph" << asc_graph_id << "_tiling_key() == " << result_id << ") {" << std::endl;
      ss << "    ArrangeBlockOffsetsAscGraph" << asc_graph_id << "Result" << result_id
         << "(tiling_data, org_block_dim);" << std::endl;
      ss << "  }" << std::endl;
      tiling_func_.AddLine(ss.str());
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map) {
  GE_ASSERT_SUCCESS(GenFusedScheduleResultsGetTilingCoreDefine(namespace_map));
  GE_ASSERT_SUCCESS(GenFusedScheduleResultsGetTilingObserveWrapper(namespace_map));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFusedScheduleResultsCoreSelection(const FusedGraphNamespaceMap &namespace_map) {
  GenWorkspaceOffsetReset("tiling_data");
  size_t asc_graph_id = 0UL;
  const std::string failed_log_level =
      (config_.enable_autofuse_pgo || config_.is_inductor_scene) ? "OP_LOGW" : "OP_LOGE";
  for (const auto &asc_graph_namespace_map : namespace_map) {
    if (asc_graph_id == 0UL) {
      tiling_func_.AddLine("  uint32_t max_block_dim = 0U;");
      tiling_func_.AddLine("  uint32_t org_block_dim = tiling_data.get_block_dim();");
    }
    const std::string asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
    tiling_func_.AddLine("  if (!" + asc_graph_namespace + "::GetTiling(tiling_data, tiling_case_id, perf)) {");
    tiling_func_.AddLine("    " + failed_log_level + "(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace +
                         ".\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
    GenEnableGroupParallelInvoke(asc_graph_id, asc_graph_namespace_map.second);
    tiling_func_.AddLine(
        "  max_block_dim = (tiling_data.get_block_dim() > max_block_dim) ? tiling_data.get_block_dim() : "
        "max_block_dim;");
    ++asc_graph_id;
  }
  tiling_func_.AddLine("  if (org_block_dim > 0U && max_block_dim > org_block_dim) {");
  tiling_func_.AddLine("    " + failed_log_level +
                       "(OP_NAME, \"Aggregated block_dim %u exceeds core budget %u.\", max_block_dim, org_block_dim);");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
  GenWorkspaceOffsetFinalize("tiling_data");
  tiling_func_.AddLine("  tiling_data.set_block_dim(max_block_dim);");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenFusedScheduleResultsGetTilingCoreDefine(const FusedGraphNamespaceMap &namespace_map) {
  tiling_func_.AddLine("bool GetTilingCore(" + config_.tiling_data_type_name +
                       " &tiling_data, int32_t tiling_case_id, double *perf) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "bool GetTilingCore(" + config_.tiling_data_type_name +
                          " &tiling_data, int32_t tiling_case_id, double *perf = nullptr);");
  const bool has_forced_tiling_case =
      std::any_of(tiling_model_info_.cbegin(), tiling_model_info_.cend(), [this](const auto &model_info) {
        return config_.force_tiling_case.GetCase(model_info.schedule_group_ident.group_id).first >= 0;
      });
  if (config_.cache_enabled_at_compile_time && !has_forced_tiling_case) {
    GE_ASSERT_SUCCESS(cache::OperatorLevelCacheGen::GenInitAndQueryCacheCode(tiling_func_, tiling_model_info_, config_,
                                                                             true, "tiling_case_id == -1"),
                      "Generate init and query cache code failed.");
  }
  tiling_func_.AddLine("  bool ret = true;");
  GE_ASSERT_SUCCESS(GenFusedScheduleResultsCoreSelection(namespace_map));
  if (config_.cache_enabled_at_compile_time && !has_forced_tiling_case) {
    tiling_func_.AddLine("  if (tiling_case_id == -1) {");
    GE_ASSERT_SUCCESS(cache::OperatorLevelCacheGen::GenSaveCacheCalls(tiling_func_, tiling_model_info_, config_),
                      "Generate save cache calls failed.");
    tiling_func_.AddLine("  }");
  }
  tiling_func_.AddLine("  (void)perf;");
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End GetTiling.\");");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenFinalTilingSummaryForResult(
    size_t asc_graph_id, size_t result_id, const std::map<size_t, std::pair<std::string, std::string>> &groups) {
  tiling_func_.AddLine("  if (tiling_data.get_graph" + std::to_string(asc_graph_id) +
                       "_tiling_key() == " + std::to_string(result_id) + "U) {");
  tiling_func_.AddLine(
      "    FinalTilingSummaryContext final_summary{1U, \"runtime\", "
      "tiling_case_id == -1 ? \"default\" : \"explicit\", OP_NAME, " +
      std::to_string(asc_graph_id) + "U, " + std::to_string(result_id) + "U, {}};");
  for (const auto &group_info : groups) {
    const std::string group_suffix = std::to_string(group_info.first);
    const std::string final_group_tiling_data = "final_group_tiling_data_" + group_suffix;
    const std::string final_group_selection = "final_group_selection_" + group_suffix;
    tiling_func_.AddLine("    auto &" + final_group_tiling_data + " = tiling_data." + group_info.second.second +
                         "_tiling_data;");
    tiling_func_.AddLine("    FinalTilingGroupSelection " + final_group_selection + "{" + group_suffix +
                         "U, -1, 0U, 0, \"\", \"\"};");
    tiling_func_.AddLine("    if (" + group_info.second.first + "::GetFinalTilingSelection(" + final_group_tiling_data +
                         ", " + final_group_selection + ")) {");
    tiling_func_.AddLine("      " + group_info.second.first + "::EmitFinalTilingByCase(" + final_group_tiling_data +
                         ", org_block_dim, \"runtime\", tiling_case_id == -1 ? \"default\" : \"explicit\", " +
                         group_suffix + "U, " + final_group_selection + ".case_id);");
    tiling_func_.AddLine("      final_summary.groups.push_back(" + final_group_selection + ");");
    tiling_func_.AddLine("    }");
  }
  tiling_func_.AddLine("    if (!final_summary.groups.empty()) { EmitFinalTilingSummary(final_summary); }");
  tiling_func_.AddLine("  }");
}

af::Status TilingCodeGenImpl::GenFusedScheduleResultsGetTilingObserveWrapper(
    const FusedGraphNamespaceMap &namespace_map) {
  tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name +
                       " &tiling_data, int32_t tiling_case_id, double *perf) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "bool GetTiling(" + config_.tiling_data_type_name + " &, int32_t, double *perf = nullptr);");
  tiling_func_.AddLine("  const bool emit_final_tiling = ShouldEmitFinalTiling();");
  tiling_func_.AddLine("  const uint32_t org_block_dim = tiling_data.get_block_dim();");
  tiling_func_.AddLine("  ::g_final_tiling_observe_enabled = emit_final_tiling;");
  tiling_func_.AddLine("  const bool ret = GetTilingCore(tiling_data, tiling_case_id, perf);");
  tiling_func_.AddLine("  ::g_final_tiling_observe_enabled = false;");
  tiling_func_.AddLine("  if (!ret || !emit_final_tiling) { return ret; }");
  for (const auto &asc_graph_namespace_map : namespace_map) {
    for (const auto &result_id_and_groups : asc_graph_namespace_map.second) {
      GenFinalTilingSummaryForResult(asc_graph_namespace_map.first, result_id_and_groups.first,
                                     result_id_and_groups.second);
    }
  }
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");

  // PGO-loaded data bypasses GetTilingCore, so expose a side-effect-free emitter
  // for the already selected multi-group result.
  tiling_func_.AddLine("void EmitFinalTilingForPgo(" + config_.tiling_data_type_name +
                       " &tiling_data, uint32_t corenum) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "void EmitFinalTilingForPgo(" + config_.tiling_data_type_name + " &, uint32_t);");
  tiling_func_.AddLine("  if (!ShouldEmitFinalTiling()) { return; }");
  for (const auto &asc_graph_namespace_map : namespace_map) {
    for (const auto &result_id_and_groups : asc_graph_namespace_map.second) {
      tiling_func_.AddLine("  if (tiling_data.get_graph" + std::to_string(asc_graph_namespace_map.first) +
                           "_tiling_key() == " + std::to_string(result_id_and_groups.first) + "U) {");
      for (const auto &group_info : result_id_and_groups.second) {
        const std::string group_suffix = std::to_string(group_info.first);
        const std::string group_tiling_data = "tiling_data." + group_info.second.second + "_tiling_data";
        tiling_func_.AddLine("    " + group_info.second.first + "::EmitFinalTilingByCase(" + group_tiling_data +
                             ", corenum, \"pgo\", \"pgo\", " + group_suffix + "U, -1);");
      }
      tiling_func_.AddLine("  }");
    }
  }
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(
    const FusedGraphNamespaceMap &namespace_map) {
  tiling_func_.AddLine(
      "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* "
      "tiling_data, uint32_t max_block_dim) {");
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start PGOSearchTilingKey root.\");");
  tiling_func_.AddLine("  bool ret = true;");
  tiling_func_.AddLine("  for (uint32_t block_dim_i=1; block_dim_i <= max_block_dim; block_dim_i++) {");
  for (const auto &asc_graph_namespace_map : namespace_map) {
    tiling_func_.AddLine("    std::vector<AutofuseTilingData> vec" + std::to_string(asc_graph_namespace_map.first) +
                         ";");
  }
  auto core_num = BaseTypeUtils::DumpHardware(HardwareDef::CORENUM);
  tiling_func_.AddLine("    tiling_data->set_block_dim(block_dim_i);");
  tiling_func_.AddLine("    tiling_data->set_" + core_num + "(block_dim_i);");
  GenWorkspaceOffsetReset("*tiling_data");
  for (const auto &asc_graph_namespace_map : namespace_map) {
    const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
    tiling_func_.AddLine("    if (!" + asc_graph_namespace + "::PGOByCoreNumSearchTilingKey(vec" +
                         std::to_string(asc_graph_namespace_map.first) + ", *tiling_data)) {");
    tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace + ".\");");
    tiling_func_.AddLine("      continue;");
    tiling_func_.AddLine("    }");
  }
  GenPGOByCoreNumSearchTilingKeyCollectTilingData(namespace_map);
  tiling_func_.AddLine("  }");

  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey root.\");");

  tiling_func_.AddLine("  ret = !tiling_data_list.empty();");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map) {
  tiling_func_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " +
                       config_.tiling_data_type_name + " &tiling_data, " +
                       " int32_t tiling_case_id, AutofuseTilingData* tilingData," + GenPgoTensorArgsDef() +
                       "void* stream, uint64_t workspaceSize, double& best_perf, const SearchConfig *search_cfg) {");
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"Start PGOSearchTilingKey root.\");");
  tiling_func_.AddLine("  (void)tilingData;");
  tiling_func_.AddLine("  PgoConfig::Instance().tensor_args = tensor_args;");
  tiling_func_.AddLine("  double cur_perf = DBL_MAX;");
  tiling_func_.AddLine("  uint32_t cur_block_dim = 1;");
  tiling_func_.AddLine("  uint32_t ori_block_dim = tiling_data.get_block_dim();");
  GenWorkspaceOffsetReset("tiling_data");
  tiling_func_.AddLine("  AutofuseTilingData tilingTmp;");
  tiling_func_.AddLine("  tilingTmp = tiling_data;");

  // Rebuild size aggregation base by calling each AscGraph::GetTiling to populate workspace values
  // This is the critical fix to ensure each subgraph's workspace values are included in tilingTmp
  for (const auto &asc_graph_namespace_map : namespace_map) {
    const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
    tiling_func_.AddLine("  if (!" + asc_graph_namespace + "::GetTiling(tilingTmp, -1, nullptr)) {");
    tiling_func_.AddLine("    OP_LOGW(OP_NAME, \"Failed to rebuild size base for " + asc_graph_namespace + ".\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
  }

  std::string block_dim_list_arg = "multi_group_block_dim_list";
  GenPGOMultiGroupBlockDimList(namespace_map, block_dim_list_arg);

  // Record baseline before search loop to identify new candidates
  tiling_func_.AddLine("  const size_t ws_baseline = tiling_data_list.size();");

  for (const auto &asc_graph_namespace_map : namespace_map) {
    const std::string &asc_graph_namespace = "AscGraph" + std::to_string(asc_graph_namespace_map.first);
    tiling_func_.AddLine("  if (!" + asc_graph_namespace +
                         "::PGOSearchTilingKey(tiling_data_list, tilingTmp, tiling_case_id, &tilingTmp, " +
                         "stream, workspaceSize, cur_perf, " + block_dim_list_arg + ", search_cfg)) {");
    tiling_func_.AddLine("    OP_LOGW(OP_NAME, \"Failed to get tiling of " + asc_graph_namespace + ".\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("  if (best_perf > cur_perf) {");
    tiling_func_.AddLine("    tiling_data = tilingTmp;");
    tiling_func_.AddLine("    best_perf = cur_perf;");
    tiling_func_.AddLine("  }");
  }

  // Finalize only the new candidates from this search iteration (skip baseline candidates)
  // This prevents double-finalization of existing offset-semantic candidates
  // 仅在存在 workspace tensor 时外抛：helper 定义由 GenWorkspaceOffsetHelpers 条件生成，
  // 无 workspace 时调用点必须同步跳过，否则生成的 tiling func 编译失败。
  if (!GetWorkspaceTensorIds(workspace_tensor_id_set_).empty()) {
    tiling_func_.AddLine("  for (size_t i = ws_baseline; i < tiling_data_list.size(); ++i) {");
    tiling_func_.AddLine("    if (!FinalizeWorkspaceOffsets(tiling_data_list[i].tiling_data)) {");
    tiling_func_.AddLine("      return false;");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine("  }");
  }

  GenWorkspaceOffsetFinalize("tiling_data");
  tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey root.\");");
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenPGOByCoreNumGetAllSchedulesResults(const size_t asc_graph_id,
                                                              const AscGraphNamepspaceMap &namespace_map) {
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";
  tiling_func_.AddLine("  bool has_valid_tiling = false;");
  tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
  tiling_func_.AddLine("    tiling_data.set_" + tiling_key_prefix + "tiling_key(index);");
  for (const auto &result_id_and_groups : namespace_map) {
    for (const auto &group_info : result_id_and_groups.second) {
      tiling_func_.AddLine("    tiling_data." + group_info.second.second + "_tiling_data = {};");
    }
  }
  tiling_func_.AddLine("    if (kScheduleResultFunctionsPGOByCoreNum[index](tiling_data_list, tiling_data)) {");
  tiling_func_.AddLine("      has_valid_tiling = true;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
}

void TilingCodeGenImpl::GenPGOGetAllSchedulesResults(const size_t asc_graph_id,
                                                     const AscGraphNamepspaceMap &namespace_map) {
  std::string tiling_key_prefix = "graph" + std::to_string(asc_graph_id) + "_";

  tiling_func_.AddLine("  AutofuseTilingData tilingTmp;");
  tiling_func_.AddLine("  bool has_valid_tiling = false;");
  tiling_func_.AddLine("  for (int32_t index = 0; index < " + std::to_string(namespace_map.size()) + "; index++) {");
  tiling_func_.AddLine("    tilingTmp = tiling_data;");
  tiling_func_.AddLine("    tilingTmp.set_" + tiling_key_prefix + "tiling_key(index);");
  tiling_func_.AddLine("    cur_perf = DBL_MAX;");
  tiling_func_.AddLine("    if (!AscGraph" + std::to_string(asc_graph_id) +
                       "::GetTiling(tilingTmp, index, nullptr)) {");
  tiling_func_.AddLine("      OP_LOGW(OP_NAME, \"Skip invalid PGO schedule result candidate.\");");
  tiling_func_.AddLine("      continue;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine(
      "    if (kScheduleResultFunctionsPGO[index](tiling_data_list, ori_block_dim, "
      "tiling_case_id, tilingTmp, cur_perf, best_perf, cur_block_dim, "
      "stream, workspaceSize, block_dim_vec, search_cfg)) {");
  tiling_func_.AddLine("      has_valid_tiling = true;");
  tiling_func_.AddLine("    }");
  tiling_func_.AddLine("  }");
}

af::Status TilingCodeGenImpl::GenGetTilingForAllSchedulesResults(const uint32_t asc_graph_id,
                                                                 const AscGraphNamepspaceMap &asc_graph_map) {
  tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                       "int32_t tiling_case_id, double *perf) {");
  AddAtomicHeaderLine(
      autofuse::GeneratedHeaderId::kApi,
      "bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " + "int32_t tiling_case_id, double *perf);");
  tiling_func_.AddLine("  (void)perf;");
  GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                    "Generate begin code!");
  GE_ASSERT_SUCCESS(GenGetTilingForAllInitLines());
  if (NeedGenScoreFunc(score_funcs_)) {
    GenGetScoreFuncsCalling(asc_graph_id, asc_graph_map);
    GenGetMaxScoreIndex(asc_graph_map);
  }
  GE_ASSERT_SUCCESS(GenGetAllSchedulesResults(asc_graph_map));
  tiling_func_.AddLine("  const bool result = GetResultSummary(best_perf, tiling_data);");
  GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "), "Generate end code!");
  GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
  GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
  tiling_func_.AddLine("  return result;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id) + " {");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingForScheduleResult() {
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  for (auto &asc_graph_map_iter : namespace_map) {
    auto &asc_graph_map = asc_graph_map_iter.second;
    size_t asc_graph_id = asc_graph_map_iter.first;
    tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");

    if (NeedGenScoreFunc(score_funcs_)) {
      GenGetScoreFuncs(asc_graph_id, asc_graph_map);
    }
    GE_ASSERT_SUCCESS(GenGetResultSummary(asc_graph_id),
                      "Gen GetResultSummary failed, asc_graph_id = %zu, tiling data name = %s.", asc_graph_id,
                      config_.tiling_data_type_name.c_str());
    const bool enable_groups_parallel = GenUpdateCurPerfAndBlockByGroupIfNeeded(asc_graph_id, asc_graph_map);
    if (enable_groups_parallel) {
      tiling_func_.AddLine(GenUpdateCurPerfAndBlockByGroup());
    }
    // 生成DoGroupTiling公共函数（在GenGetScheduleResult之前）
    for (const auto &graph_info : asc_graph_map) {
      GE_ASSERT_SUCCESS(GenDoGroupTilingFunction(asc_graph_id, graph_info.first, graph_info.second),
                        "GenDoGroupTilingFunction failed, asc_graph_id=%zu, impl_graph_id=%zu", asc_graph_id,
                        graph_info.first);
    }
    for (const auto &graph_info : asc_graph_map) {
      GenGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map);
    }
    tiling_func_.AddLine(GenScheduleResultFuncTypeDefine(config_.tiling_data_type_name));
    tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map));
    GE_ASSERT_SUCCESS(GenGetTilingForAllSchedulesResults(asc_graph_id, asc_graph_map),
                      "Generate GetTiling for all schedules results failed, asc_graph_id = %zu.", asc_graph_id);
  }
  GenWorkspaceOffsetHelpers();
  GE_ASSERT_SUCCESS(GenEnableGroupParallelFunctions(namespace_map));
  GE_ASSERT_SUCCESS(GenFusedScheduleResultsGetTilingDefine(namespace_map));
  GE_ASSERT_SUCCESS(GenIsStaticShape());
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOGetTilingForAll() {
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  for (auto &asc_graph_map_iter : namespace_map) {
    auto &asc_graph_map = asc_graph_map_iter.second;
    size_t asc_graph_id = asc_graph_map_iter.first;
    tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");
    for (const auto &graph_info : asc_graph_map) {
      GE_ASSERT_SUCCESS(GenPGOGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map));
    }
    tiling_func_.AddLine(GenPGOScheduleResultFuncTypeDefine(config_.tiling_data_type_name));
    tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map, "PGO"));
    tiling_func_.AddLine("bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, " +
                         config_.tiling_data_type_name + " &tiling_data, " +
                         " int32_t tiling_case_id, AutofuseTilingData* tilingData, "
                         "void* stream, uint64_t workspaceSize, double& best_perf, "
                         "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {");
    GE_ASSERT_SUCCESS(GenGetTilingForAllInitLines(true));
    tiling_func_.AddLine("  (void)tilingData;");
    GenPGOGetAllSchedulesResults(asc_graph_id, asc_graph_map);
    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey in AscGraph.\");");
    GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
    GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
    tiling_func_.AddLine("  return has_valid_tiling;");
    tiling_func_.AddLine("}");
    tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id) + " {");
  }
  GE_ASSERT_SUCCESS(GenPGOFusedScheduleResultsGetTilingDefine(namespace_map));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOByCoreNumTilingForAll() {
  std::map<std::string, std::set<std::string>> hardware_map;
  FusedGraphNamespaceMap namespace_map;
  GE_ASSERT_SUCCESS(ObtainInnerParams(hardware_map, namespace_map));
  for (auto &asc_graph_map_iter : namespace_map) {
    auto &asc_graph_map = asc_graph_map_iter.second;
    size_t asc_graph_id = asc_graph_map_iter.first;
    tiling_func_.AddLine("namespace AscGraph" + std::to_string(asc_graph_id) + " {");
    for (const auto &graph_info : asc_graph_map) {
      GenPGOByCoreNumGetScheduleResult(asc_graph_id, graph_info.first, graph_info.second, hardware_map,
                                       var_relations_[asc_graph_id][graph_info.first]);
    }
    tiling_func_.AddLine(GenPGOByCoreNumScheduleResultFuncTypeDefine());
    tiling_func_.AddLine(GenScheduleResultFuncsDefine(asc_graph_map, "PGOByCoreNum"));
    tiling_func_.AddLine(
        "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData "
        "tiling_data) {");
    GenPGOByCoreNumGetAllSchedulesResults(asc_graph_id, asc_graph_map);
    tiling_func_.AddLine("  OP_LOGI(OP_NAME, \"End PGOSearchTilingKey in AscGraph.\");");
    GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
    GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
    tiling_func_.AddLine("  return has_valid_tiling;");
    tiling_func_.AddLine("}");
    tiling_func_.AddLine("} // namespace AscGraph" + std::to_string(asc_graph_id));
  }
  GE_ASSERT_SUCCESS(GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingWithCaseId(bool is_tail) {
  bool use_cache = (!is_tail && with_reuse_info_);
  bool use_workspace = !(is_uniq_group_ || is_tail);

  // 收集并验证tiling case信息
  int32_t min_tiling_case_size = INT32_MAX;
  std::map<string, int32_t> group_tiling_case_ids;
  for (auto &model : tiling_model_info_) {
    group_tiling_case_ids[model.schedule_group_ident.GetItemPrefix()]++;
  }
  for (auto &group_tiling_case : group_tiling_case_ids) {
    min_tiling_case_size = std::min(group_tiling_case.second, min_tiling_case_size);
  }
  GE_ASSERT_SUCCESS(ValidateForceTilingCase(group_tiling_case_ids, min_tiling_case_size));

  // 获取参数定义字符串
  std::string cache_define_head, cache_define_func, cache_used;
  std::string workspace_define =
      GetGetTilingParamDefines(use_cache, use_workspace, cache_define_head, cache_define_func, cache_used);

  // 生成GetTiling函数签名
  GenGetTilingFunctionSignature(workspace_define, cache_define_func, cache_define_head);

  // 生成GetTiling函数体
  GE_ASSERT_SUCCESS(GenGetTilingFunctionBody(use_cache, is_tail, cache_used), "Generate function body failed.");

  return af::SUCCESS;
}

std::string TilingCodeGenImpl::GetGetTilingParamDefines(bool use_cache, bool use_workspace,
                                                        std::string &cache_define_head, std::string &cache_define_func,
                                                        std::string &cache_used) const {
  cache_define_head = use_cache ? (", GroupLevelCache *cache = nullptr") : "";
  cache_define_func = use_cache ? (", GroupLevelCache *cache") : "";
  cache_used = use_cache ? (", cache") : "";
  return use_workspace ? (", std::unordered_map<int64_t, uint64_t> &workspace_map") : "";
}

void TilingCodeGenImpl::GenGetTilingFunctionSignature(const std::string &workspace_define,
                                                      const std::string &cache_define_func,
                                                      const std::string &cache_define_head) {
  if (tiling_model_info_.empty()) {
    GELOGE(ge::GRAPH_FAILED, "[GenGetTilingFunctionSignature] tiling_model_info_ is empty.");
    return;
  }
  tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data" + workspace_define +
                       ", int32_t tiling_case_id" + cache_define_func + ", double *perf) {");
  const auto header_id =
      cache_define_head.empty() ? autofuse::GeneratedHeaderId::kApi : autofuse::GeneratedHeaderId::kState;
  AddAtomicHeaderLine(header_id, "bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data" +
                                     workspace_define + ", int32_t tiling_case_id" + cache_define_head +
                                     ", double *perf = nullptr);");
}

af::Status TilingCodeGenImpl::GenGetTilingFunctionBody(bool use_cache, bool is_tail, const std::string &cache_used) {
  (void)use_cache;  // 保留参数以备将来使用
  const bool has_forced_tiling_case =
      std::any_of(tiling_model_info_.cbegin(), tiling_model_info_.cend(), [this](const auto &model_info) {
        return config_.force_tiling_case.GetCase(model_info.schedule_group_ident.group_id).first >= 0;
      });
  const bool need_operator_cache =
      config_.cache_enabled_at_compile_time && !has_forced_tiling_case && (is_tail || (!is_tail && is_uniq_group_));

  // 初始化返回值
  tiling_func_.AddLine("  bool ret = true;");
  if (need_operator_cache) {
    tiling_func_.AddLine("  bool cache_hit = false;");
  }

  // 生成duration begin代码
  GE_ASSERT_SUCCESS(GenDurationCode(true), "Generate duration begin code failed.");

  // 第一级：算子级缓存查询（在GetTilingKey之前）
  if (need_operator_cache) {
    GE_ASSERT_SUCCESS(cache::OperatorLevelCacheGen::GenInitAndQueryCacheCode(tiling_func_, tiling_model_info_, config_,
                                                                             false, "tiling_case_id == -1"),
                      "Generate init and query cache code failed.");
  }

  // Enable sub-case tracking before GetTilingKey so the single-group emitter can report the selected template.
  GenSingleGroupFinalTilingObserveBegin();

  // 缓存未命中时生成日志和GetTilingKey调用
  if (need_operator_cache) {
    tiling_func_.AddLine("  if (!cache_hit) {");
  }
  GE_ASSERT_SUCCESS(GenGetTilingKeyCall(cache_used), "Generate GetTilingKey call failed.");

  if (need_operator_cache) {
    tiling_func_.AddLine("  }");
    tiling_func_.AddLine("  if (cache_hit) {");
    tiling_func_.AddLine("    if (::g_final_tiling_observe_enabled) { g_final_selected_sub_case_tag_valid = false; }");
    tiling_func_.AddLine("  }");
    // 保存算子级缓存，仅在实际完成tiling且未命中缓存时执行。
    tiling_func_.AddLine("  if (ret) {");
    tiling_func_.AddLine("    if (tiling_case_id == -1 && !cache_hit) {");
    GE_ASSERT_SUCCESS(GenOperatorCacheSaveCode(true), "Generate save cache calls failed.");
    tiling_func_.AddLine("    }");
    tiling_func_.AddLine("  }");
  }

  // Clear the observation state only after cache-hit cleanup has completed.
  GenSingleGroupFinalTilingObserveReset();

  // 生成duration end代码
  GE_ASSERT_SUCCESS(GenDurationCode(false), "Generate duration end code failed.");

  // 如果 perf 不为空，计算并赋值性能
  tiling_func_.AddLine("  if (ret && perf != nullptr) {");
  tiling_func_.AddLine("    *perf = GetPerf(tiling_data);");
  tiling_func_.AddLine("  }");

  // Emit after duration/perf bookkeeping so diagnostics do not affect measured tiling time.
  GenSingleGroupFinalTilingObserveEmit(need_operator_cache);

  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetTilingKeyCall(const std::string &cache_used) {
  GE_ASSERT_SUCCESS(
      GenOpLog("  ", "Start GetTiling.",
               "Start tiling for sched group " + tiling_model_info_[0].schedule_group_ident.GetGroupPrefix() + "."));

  int32_t force_case_id = config_.force_tiling_case.GetCase(tiling_model_info_[0].schedule_group_ident.group_id).first;
  std::string tiling_case = (force_case_id < 0) ? "tiling_case_id" : std::to_string(force_case_id);
  tiling_func_.AddLine(std::string("  if (!GetTilingKey(tiling_data, ") + (is_uniq_group_ ? "" : "workspace_map, ") +
                       tiling_case + cache_used + ")) {");
  GE_ASSERT_SUCCESS(GenOpLog("    ", "GetTiling Failed."));
  tiling_func_.AddLine("    ret = false;");
  tiling_func_.AddLine("  }");

  GE_ASSERT_SUCCESS(GenOpLog("  ", "End GetTiling.", "End tiling for sched group."));
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenDurationCode(bool is_begin) {
  if (!is_uniq_group_) {
    return af::SUCCESS;
  }
  if (is_begin) {
    GE_ASSERT_SUCCESS(GenDurationBeginCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                      "Generate duration begin code failed.");
  } else {
    GE_ASSERT_SUCCESS(GenDurationEndCode(TilingFuncDurationType::TILING_FUNC_DURATION_TOTAL, "  "),
                      "Generate duration end code failed.");
    GE_ASSERT_SUCCESS(GenDurationPrintCode("  "), "Generate print code failed.");
    GE_ASSERT_SUCCESS(GenDurationClearCode("  "), "Generate clear code failed.");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenOperatorCacheSaveCode(bool need_operator_cache) {
  if (need_operator_cache) {
    GE_ASSERT_SUCCESS(cache::OperatorLevelCacheGen::GenSaveCacheCalls(tiling_func_, tiling_model_info_, config_),
                      "Generate save cache calls failed.");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::ValidateForceTilingCase(const std::map<string, int32_t> &group_tiling_case_ids,
                                                      int32_t min_tiling_case_size) const {
  // 如果配置了 force_schedule_result，但当前 tiling_model_info_ 不包含该 result，
  // 说明当前是 GenTilingBody 在处理单个 result，跳过该校验（留给 GenTilingTail 处理）
  if (config_.force_schedule_result >= 0) {
    bool has_force_result = false;
    for (const auto &model : tiling_model_info_) {
      if (static_cast<int64_t>(model.schedule_group_ident.impl_graph_id) == config_.force_schedule_result) {
        has_force_result = true;
        break;
      }
    }
    if (!has_force_result) {
      GELOGD("Skip force tiling case validation: current model info does not contain result[%ld]",
             config_.force_schedule_result);
      return af::SUCCESS;
    }
  }

  GE_ASSERT_SUCCESS(ValidateSingleModeForceTilingCase(min_tiling_case_size));
  if (!config_.force_tiling_case.is_single_mode) {
    GE_ASSERT_SUCCESS(ValidateGroupModeForceTilingCase(group_tiling_case_ids));
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::ValidateSingleModeForceTilingCase(int32_t min_tiling_case_size) const {
  int32_t force_case = config_.force_tiling_case.single_case;

  // 如果 force_case 为默认值 -1，跳过校验
  if (force_case == -1) {
    GELOGD("Force tiling case is not set, skip validation");
    return af::SUCCESS;
  }

  // 如果 force_case < min_tiling_case_size，按 case_id 校验
  if (force_case < min_tiling_case_size) {
    GE_ASSERT_TRUE(force_case >= 0, "Force tiling case[%d] should be non-negative", force_case);
    return af::SUCCESS;
  }

  // 否则，检查是否为有效的 tiling_key
  bool tiling_key_found = false;
  for (const auto &model : tiling_model_info_) {
    if (static_cast<int32_t>(model.tiling_case_id) == force_case) {
      tiling_key_found = true;
      break;
    }
  }

  GE_ASSERT_TRUE(tiling_key_found, "Force tiling case[%d] not found as tiling_key in model info", force_case);
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::ValidateGroupModeForceTilingCase(
    const std::map<string, int32_t> &group_tiling_case_ids) const {
  // 遍历当前传入的 tiling_model_info_，只校验这些 group
  // 因为 GenTilingBody 每次只传入单个 group，而不是所有 groups
  for (const auto &model : tiling_model_info_) {
    size_t cur_group_id = model.schedule_group_ident.group_id;
    int64_t cur_result_id = static_cast<int64_t>(model.schedule_group_ident.impl_graph_id);

    // 如果配置了 force_schedule_result，跳过非指定 result 的 group
    if (config_.force_schedule_result >= 0 && cur_result_id != config_.force_schedule_result) {
      continue;
    }

    // 查找该 group 是否有强制指定的 case
    auto it = config_.force_tiling_case.group_cases.find(cur_group_id);
    if (it == config_.force_tiling_case.group_cases.end()) {
      continue;  // 该 group 没有强制指定，跳过
    }

    int32_t force_case_id = it->second.first;

    // 获取该 group 的 case 数量
    auto case_it = group_tiling_case_ids.find(model.schedule_group_ident.GetItemPrefix());
    GE_ASSERT_TRUE(case_it != group_tiling_case_ids.end(),
                   "Group[%zu] in result[%ld] not found in "
                   "group_tiling_case_ids",
                   cur_group_id, cur_result_id);
    size_t group_case_size = case_it->second;

    GELOGD("Validate force tiling case: result[%ld] group[%zu] case[%d] < size[%zu]", cur_result_id, cur_group_id,
           force_case_id, group_case_size);

    // 校验 case_id 或 tiling_key
    if (force_case_id < static_cast<int32_t>(group_case_size)) {
      // case_id 在范围内，校验通过
      GELOGD("Validate force tiling case by case_id: result[%ld] group[%zu] case[%d]", cur_result_id, cur_group_id,
             force_case_id);
    } else {
      // 检查是否为有效的 tiling_key
      bool tiling_key_found = false;
      for (const auto &m : tiling_model_info_) {
        if (m.schedule_group_ident.group_id == cur_group_id &&
            static_cast<int64_t>(m.schedule_group_ident.impl_graph_id) == cur_result_id &&
            static_cast<int32_t>(m.tiling_case_id) == force_case_id) {
          tiling_key_found = true;
          break;
        }
      }
      GE_ASSERT_TRUE(tiling_key_found, "Force tiling case[%d] for group[%zu] result[%ld] not found as tiling_key",
                     force_case_id, cur_group_id, cur_result_id);
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenScheduleGroupTilingTail() {
  if (config_.gen_tiling_data) {
    if (!is_uniq_group_) {
      GE_ASSERT_SUCCESS(GenHeaderCodesSummaryBody(), "Generate tiling data summary body failed.");
    }
    GE_ASSERT_SUCCESS(GenHeaderCodesTail(), "Generate tiling data tail failed.");
  }
  if (!is_uniq_group_) {
    GE_ASSERT_SUCCESS(GenGetTilingForScheduleResult());
    if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
      GE_ASSERT_SUCCESS(GenPGOGetTilingForAll());
      if (config_.enable_autofuse_pgo) {
        GE_ASSERT_SUCCESS(GenPGOByCoreNumTilingForAll());
      }
    }
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingTail(std::map<std::string, std::string> &tiling_res,
                                            GenTilingTailImplExtParams ext_params) {
  var_relations_ = std::move(ext_params.var_relations);
  enable_group_parallels_ = std::move(ext_params.enable_group_parallels);
  workspace_tensor_id_set_ = std::move(ext_params.workspace_tensor_id_set);
  if (!ext_params.cache_reuse_info.empty()) {
    cache_reuse_info_ = std::move(ext_params.cache_reuse_info);
    with_reuse_info_ = true;
  }
  ResetTranslationUnit();
  tiling_head_.Reset();
  tiling_data_.Reset();
  if (!is_uniq_group_) {
    for (const auto &header :
         {"algorithm", "array", "cfloat", "cstddef", "cstdint", "functional", "unordered_map", "utility", "vector"}) {
      RequireTranslationUnitSystemHeader(header);
    }
    RequireTranslationUnitTilingDataHeader();
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kState);
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kLog);
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kSolver);
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kApi);
    if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
      RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kPgo);
    }
  }
  tiling_func_.AddLine("namespace optiling {");

  // 支持二次Tiling：定义全局变量（多Group场景）
  tiling_func_.AddLine("// 支持二次Tiling：全局变量，用于传递调整后的核数比例");
  tiling_func_.AddLine("thread_local double g_secondary_tiling_ratio = 0.0;");

  GE_ASSERT_SUCCESS(GenScheduleGroupTilingTail(), "Generate tiling data tail inner failed.");

  tiling_head_.AddLine("} // namespace optiling");
  tiling_func_.AddLine("} // namespace optiling");
  GE_ASSERT_SUCCESS(FinishTranslationUnit(kTilingScheduleGroupTailIdentify, tiling_res));
  tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
  if (config_.gen_tiling_data) {
    tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetPerf() {
  tiling_func_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "double GetPerf(" + config_.tiling_data_type_name + " &tiling_data);");
  tiling_func_.AddLine(
      "  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_data.get_tiling_key(), "
      "tiling_data.get_block_dim());");
  tiling_func_.AddLine("  return tilingCaseImplPtr->GetPerf(tiling_data);");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenGetSummary() {
  tiling_func_.AddLine("void GetSummary(" + config_.tiling_data_type_name + " &tiling_data) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "void GetSummary(" + config_.tiling_data_type_name + " &tiling_data);");
  tiling_func_.AddLine(
      "  TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tiling_data.get_tiling_key(), "
      "tiling_data.get_block_dim());");
  tiling_func_.AddLine("  if (tilingCaseImplPtr == nullptr) {");
  tiling_func_.AddLine("    return;");
  tiling_func_.AddLine("  }");
  if (hardware_has_ub_) {
    tiling_func_.AddLine("  double ub_radio;");
    tiling_func_.AddLine("  tilingCaseImplPtr->TilingSummary(tiling_data, ub_radio);");
  } else {
    tiling_func_.AddLine("  tilingCaseImplPtr->TilingSummary(tiling_data);");
  }
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTilingKeyFunc() {
  GE_ASSERT_SUCCESS(GenTilingImplBaseClass(), "Generate base class failed.");
  for (const auto &model_info : tiling_model_info_) {
    GE_ASSERT_SUCCESS(GenSolverTiling(model_info), "Generate do op tiling failed.");
    GE_ASSERT_SUCCESS(GenTilingCaseImpl(model_info), "Generate solver definition failed.");
  }
  GE_ASSERT_SUCCESS(GenImplPtr(), "Generate func call entrance failed.");
  GE_ASSERT_SUCCESS(GenFinalTilingReprFunction(), "Generate final tiling representation function failed.");
  GE_ASSERT_SUCCESS(GenGetTilingKey(), "Generate func call entrance failed.");
  GE_ASSERT_SUCCESS(GenFinalTilingEmitter(), "Generate final tiling emitter failed.");
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    GE_ASSERT_SUCCESS(GenPGOSearchTilingKey(), "Generate func call entrance failed.");
  }
  GE_ASSERT_SUCCESS(GenTilingFuncCallEntrance(), "Generate func call entrance failed.");
  if (config_.enable_autofuse_pgo) {
    GE_ASSERT_SUCCESS(GenPGOByCoreNumSearchTilingKey(), "Generate pgo by core num func call entrance failed.");
  }
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::InitTilingGeneration(const std::unordered_map<std::string, std::string> &cache_reuse_info,
                                                   uint32_t cache_capacity) {
  cache_capacity_ = cache_capacity;
  if (!cache_reuse_info.empty()) {
    cache_reuse_info_ = cache_reuse_info;
    with_reuse_info_ = true;
  }
  tiling_head_.Reset();
  ResetTranslationUnit();
  tiling_data_.Reset();
  GE_ASSERT_TRUE(!tiling_model_info_.empty());
  GE_ASSERT_SUCCESS(tiling_data_manager_.Init());
  GE_ASSERT_SUCCESS(GenScheduleGroupTilingHead());
  return af::SUCCESS;
}

void TilingCodeGenImpl::GenGroupNamespaceHead(const ScheduleGroupIdent &cur_ident) {
  tiling_func_.AddLine("namespace optiling{");
  // 支持二次Tiling：extern声明全局变量（定义在第一个namespace optiling中）
  tiling_func_.AddLine("// 支持二次Tiling：全局变量，用于传递调整后的核数比例");
  tiling_func_.AddLine("extern thread_local double g_secondary_tiling_ratio;");
  if (!is_uniq_group_) {
    const std::string tiling_data_forward_decl = "struct " + config_.tiling_data_type_name + ";\n";
    if (config_.is_autofuse) {
      const std::string global_forward_decl =
          "}  // namespace optiling\n" + tiling_data_forward_decl + "namespace optiling {\n";
      AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, global_forward_decl);
      AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kApi, global_forward_decl);
    } else {
      AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, tiling_data_forward_decl);
      AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kApi, tiling_data_forward_decl);
    }
    autofuse::RequireSystemHeader(atomic_headers_[autofuse::GeneratedHeaderId::kState].dependencies, "unordered_map");
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi, "namespace " + cur_ident.GetGroupPrefix() + " {");
    AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, "namespace " + cur_ident.GetGroupPrefix() + " {\n");
    tiling_func_.AddLine("namespace " + cur_ident.GetGroupPrefix() + " {");
  }
}

void TilingCodeGenImpl::RequireReuseGroupTranslationUnitHeaders() {
  for (const auto &header : {"cstdint", "unordered_map"}) {
    RequireTranslationUnitSystemHeader(header);
  }
  RequireTranslationUnitTilingDataHeader();
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kState);
  // The reuse wrapper assigns FinalTilingGroupSelection members, whose complete definition lives in the
  // log header, so split translation units must include it before compiling the final-selection query.
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kLog);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kSolver);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kApi);
}

void TilingCodeGenImpl::RequireGroupTranslationUnitHeaders() {
  for (const auto &header : {"algorithm", "cstdint", "map", "memory", "string", "unordered_map", "vector"}) {
    RequireTranslationUnitSystemHeader(header);
  }
  if (hardware_has_ub_) {
    RequireTranslationUnitSystemHeader("cmath");
  }
  if (with_reuse_info_ || config_.cache_enabled_at_compile_time) {
    RequireTranslationUnitSystemHeader("array");
  }
  if (config_.cache_enabled_at_compile_time) {
    RequireTranslationUnitSystemHeader("cstring");
  }
  RequireTranslationUnitTilingDataHeader();
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kState);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kLog);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kSolver);
  RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kApi);
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    for (const auto &header : {"cfloat", "cstddef", "cstdlib", "new"}) {
      RequireTranslationUnitSystemHeader(header);
    }
    RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kPgo);
  }
}

af::Status TilingCodeGenImpl::FinishGroupTiling(const ScheduleGroupIdent &cur_ident,
                                                std::map<std::string, std::string> &tiling_res) {
  tiling_func_.AddLine("} // namespace optiling");
  if (config_.gen_tiling_data) {
    tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
  }
  tiling_res[cur_ident.GetGroupPrefixSnakeCase()].clear();
  GE_ASSERT_SUCCESS(FinishTranslationUnit(cur_ident.GetGroupPrefixSnakeCase(), tiling_res));
  tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenTiling(std::map<std::string, std::string> &tiling_res,
                                        std::unordered_map<std::string, std::string> cache_reuse_info,
                                        uint32_t cache_capacity, const EnableGroupParallels &enable_group_parallels) {
  enable_group_parallels_ = enable_group_parallels;
  if (config_.enable_autofuse_pgo || config_.is_inductor_scene) {
    GE_ASSERT_SUCCESS(GenEnableGroupParallelPgoInvoke("autofuse_tiling_data", true, "      ", arrange_code_));
  }
  GE_ASSERT_SUCCESS(InitTilingGeneration(cache_reuse_info, cache_capacity));
  const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
  GenGroupNamespaceHead(cur_ident);
  auto &reuse_group = tiling_model_info_[0].reuse_schedule_group;
  GE_ASSERT_NOTNULL(reuse_group, "reuse_schedule_group is nullptr for op %s.", op_name_.c_str());
  GELOGD("Generate tiling code for %s of %s reuse_ident is %s.", cur_ident.GetGroupPrefix().c_str(), op_name_.c_str(),
         reuse_group->reuse_group_ident.GetGroupPrefix().c_str());
  if (reuse_group->IsReuseGroup(cur_ident)) {
    RequireReuseGroupTranslationUnitHeaders();
    if (config_.enable_autofuse_pgo) {
      RequireTranslationUnitSystemHeader("cfloat");
      RequireTranslationUnitSystemHeader("vector");
      RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId::kPgo);
      GE_ASSERT_SUCCESS(GenPGOReuseGroupTilingWrapper(), "Generate func call entrance failed.");
    }
    return GenReuseGroupTilingWrapper(tiling_res);
  }
  RequireGroupTranslationUnitHeaders();
  AddApiTilingPreambles();
  GE_ASSERT_SUCCESS(GenTilingKeyFunc());
  GE_ASSERT_SUCCESS(GenGetPerf(), "Generate getperf failed.");
  if (!is_uniq_group_) {
    GE_ASSERT_SUCCESS(GenGetSummary(), "Generate getsummary failed.");
    AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, "} // namespace " + cur_ident.GetGroupPrefix() + "\n");
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi, "} // namespace " + cur_ident.GetGroupPrefix());
    tiling_func_.AddLine("} // namespace " + cur_ident.GetGroupPrefix());
  }
  return FinishGroupTiling(cur_ident, tiling_res);
}

af::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetTiling(
    const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
    std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
  if (with_reuse_info_) {
    tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                         (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                         "int32_t tiling_case_id, " + reuse_prefix + "::GroupLevelCache* cache) {");
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kState,
                        "bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                            (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                            "int32_t tiling_case_id, " + reuse_prefix + "::GroupLevelCache* cache = nullptr);");
  } else {
    tiling_func_.AddLine("bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                         (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                         "int32_t tiling_case_id) {");
    AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                        "bool GetTiling(" + config_.tiling_data_type_name + " &tiling_data, " +
                            (is_uniq_group_ ? "" : "std::unordered_map<int64_t, uint64_t> &workspace_map, ") +
                            "int32_t tiling_case_id);");
  }
  auto reuse_tiling_data =
      "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix + "TilingData>(tiling_data);";
  tiling_func_.AddLine(reuse_tiling_data);
  GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
  tiling_func_.AddLine("  auto ret = " + reuse_prefix + "::GetTiling(reuse_tiling_data, " +
                       (is_uniq_group_ ? "" : "workspace_map, ") + "tiling_case_id" +
                       (with_reuse_info_ ? ", cache" : "") + ");");
  tiling_func_.AddLine("  tiling_data = RefToRef<" + reuse_prefix + "TilingData, " + cur_prefix +
                       "TilingData>(reuse_tiling_data);");
  tiling_func_.AddLine("  return ret;");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetPerf(
    const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
    std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
  tiling_func_.AddLine("double GetPerf(" + config_.tiling_data_type_name + " &tiling_data) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "double GetPerf(" + config_.tiling_data_type_name + " &tiling_data);");
  auto reuse_tiling_data =
      "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix + "TilingData>(tiling_data);";
  tiling_func_.AddLine(reuse_tiling_data);
  GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
  tiling_func_.AddLine("  return " + reuse_prefix + "::GetPerf(reuse_tiling_data);");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenReuseGroupTilingWrapperGetSummary(
    const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
    std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter) {
  tiling_func_.AddLine("void GetSummary(" + config_.tiling_data_type_name + " &tiling_data) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "void GetSummary(" + config_.tiling_data_type_name + " &tiling_data);");
  auto reuse_tiling_data =
      "  auto reuse_tiling_data = RefToRef<" + cur_prefix + "TilingData, " + reuse_prefix + "TilingData>(tiling_data);";
  tiling_func_.AddLine(reuse_tiling_data);
  GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
  tiling_func_.AddLine(reuse_prefix + "::GetSummary(reuse_tiling_data);");
  tiling_func_.AddLine("}");

  // Reused groups share the primary group's tiling implementation.  Expose the
  // same final-selection query after converting the reused data so the fused
  // summary can use one interface for both ordinary and reused groups.
  tiling_func_.AddLine("bool GetFinalTilingSelection(" + config_.tiling_data_type_name +
                       " &tiling_data, ::FinalTilingGroupSelection &selection) {");
  AddAtomicHeaderLine(
      autofuse::GeneratedHeaderId::kApi,
      "bool GetFinalTilingSelection(" + config_.tiling_data_type_name + " &, ::FinalTilingGroupSelection &);");
  tiling_func_.AddLine(reuse_tiling_data);
  GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
  tiling_func_.AddLine("  if (!" + reuse_prefix + "::GetFinalTilingSelection(reuse_tiling_data, selection)) {");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  selection.group = " + std::to_string(tiling_model_info_[0].schedule_group_ident.group_id) +
                       "U;");
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  tiling_func_.AddLine("void EmitFinalTilingByCase(" + config_.tiling_data_type_name +
                       " &tiling_data, uint32_t corenum, const char *source, const char *selection_mode, "
                       "uint32_t group_id, int32_t requested_case_id) {");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi,
                      "void EmitFinalTilingByCase(" + config_.tiling_data_type_name +
                          " &, uint32_t, const char *, const char *, uint32_t, int32_t);");
  tiling_func_.AddLine(reuse_tiling_data);
  GE_ASSERT_SUCCESS(GenCastReuseTilingDataCode(reuse_info, iter->second));
  tiling_func_.AddLine("  " + reuse_prefix +
                       "::EmitFinalTilingByCase(reuse_tiling_data, corenum, source, selection_mode, group_id, "
                       "requested_case_id);");
  tiling_func_.AddLine("}");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenReuseGroupTilingWrapper(std::map<std::string, std::string> &tiling_res) {
  const auto &reuse_ident = tiling_model_info_[0].reuse_schedule_group->reuse_group_ident;
  const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
  const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
  const auto &cur_prefix = cur_ident.GetGroupPrefix();
  GELOGD("Cast reuse group %s to %s of %s.", reuse_prefix.c_str(), cur_prefix.c_str(), op_name_.c_str());
  const auto iter = tiling_model_info_[0].reuse_schedule_group->schedule_group_to_info.find(cur_ident);
  GE_ASSERT_TRUE(iter != tiling_model_info_[0].reuse_schedule_group->schedule_group_to_info.cend(),
                 "Find reuse group %s failed.", cur_prefix.c_str());
  const auto &reuse_info = tiling_model_info_[0].reuse_schedule_group->info;
  const auto &reuse_input_axes = reuse_info.reuse_input_axes;
  GE_ASSERT_TRUE(iter->second.reuse_input_axes.size() == reuse_input_axes.size(),
                 "Reuse group %s input axes size %zu not equal to current axes size %zu.", cur_prefix.c_str(),
                 iter->second.reuse_input_axes.size(), reuse_input_axes.size());
  const auto &reuse_search_axes = reuse_info.reuse_search_axes;
  GE_ASSERT_TRUE(iter->second.reuse_search_axes.size() == reuse_search_axes.size(),
                 "Reuse group %s search axes size %zu not equal to current axes size %zu.", cur_prefix.c_str(),
                 iter->second.reuse_search_axes.size(), reuse_search_axes.size());
  // Gen GetTiling
  GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetTiling(cur_prefix, reuse_prefix, reuse_info, iter));
  // Gen GetPerf
  GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetPerf(cur_prefix, reuse_prefix, reuse_info, iter));
  // Gen GetSummary
  GE_ASSERT_SUCCESS(GenReuseGroupTilingWrapperGetSummary(cur_prefix, reuse_prefix, reuse_info, iter));
  AppendAtomicHeaderBody(autofuse::GeneratedHeaderId::kState, "} // namespace " + cur_prefix + "\n");
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi, "} // namespace " + cur_prefix);
  tiling_func_.AddLine("} // namespace " + cur_prefix);
  tiling_func_.AddLine("} // namespace optiling");
  if (config_.gen_tiling_data) {
    tiling_res[config_.tiling_data_type_name] += tiling_data_.GetOutputStr();
  }
  tiling_res[cur_ident.GetGroupPrefixSnakeCase()].clear();
  GE_ASSERT_SUCCESS(FinishTranslationUnit(cur_ident.GetGroupPrefixSnakeCase(), tiling_res));
  tiling_res[kTilingHeadIdentify] += tiling_head_.GetOutputStr();
  GELOGD("Generate reuse group tiling wrapper for %s of %s success.", cur_prefix.c_str(), op_name_.c_str());
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenPGOReuseGroupTilingWrapper() {
  const auto &reuse_ident = tiling_model_info_[0].reuse_schedule_group->reuse_group_ident;
  const auto &cur_ident = tiling_model_info_[0].schedule_group_ident;
  const auto &reuse_prefix = reuse_ident.GetGroupPrefix();
  const auto &cur_prefix = cur_ident.GetGroupPrefix();
  const auto &reuse_item_prefix = reuse_ident.GetItemPrefix();
  const auto &cur_item_prefix = cur_ident.GetItemPrefix();

  GELOGD("Cast pgo reuse group %s to %s of %s.", reuse_prefix.c_str(), cur_prefix.c_str(), op_name_.c_str());

  // Gen PGOProfileReuseGroup: reuse group does not search, only profiles by copying from primary group
  std::string pgo_profile_sig =
      std::string("bool PGOProfileReuseGroup(std::vector<AutofuseTilingDataPerf>& tiling_data_list, ") +
      "AutofuseTilingData* output_tiling_data, void* stream, uint64_t workspaceSize, double& best_perf)";
  AddAtomicHeaderLine(autofuse::GeneratedHeaderId::kApi, pgo_profile_sig + ";");
  tiling_func_.AddLine(pgo_profile_sig + " {");
  tiling_func_.AddLine("  double cur_perf = DBL_MAX;");
  tiling_func_.AddLine("  AutofuseTilingData autofuse_tiling_data_tmp = *output_tiling_data;");
  auto reuse_tiling_data = "  auto reuse_tiling_data = RefToRef<" + reuse_prefix + "TilingData, " + cur_prefix +
                           "TilingData>(autofuse_tiling_data_tmp." + reuse_item_prefix + "_tiling_data);";
  tiling_func_.AddLine(reuse_tiling_data);
  tiling_func_.AddLine("  autofuse_tiling_data_tmp." + cur_item_prefix + "_tiling_data = reuse_tiling_data;");
  tiling_func_.AddLine("  if (!GetWorkspaceSize(autofuse_tiling_data_tmp, workspaceSize)) {");
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  }");
  if (!config_.is_inductor_scene) {
    tiling_func_.AddLine("  if (__builtin_add_overflow(workspaceSize, 16ULL * 1024ULL * 1024ULL, &workspaceSize)) {");
    tiling_func_.AddLine("    OP_LOGE(OP_NAME, \"Workspace size overflow in PGO reuse group profiling.\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  }");
  }
  std::string invoke_code;
  GE_ASSERT_SUCCESS(GenEnableGroupParallelPgoInvoke("autofuse_tiling_data_tmp", false, "      ", invoke_code));
  tiling_func_.AddLine("  if (PgoConfig::Instance().single_callback) {");
  tiling_func_.AddLine(
      "    PgoConfig::Instance().single_callback(PgoConfig::Instance().tensor_args, stream, "
      "workspaceSize, &autofuse_tiling_data_tmp, &cur_perf);");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  AutofuseTilingDataPerf tiling_perf;");
  tiling_func_.AddLine("  tiling_perf.tiling_data = autofuse_tiling_data_tmp;");
  tiling_func_.AddLine("  tiling_perf.best_perf = cur_perf;");
  tiling_func_.AddLine("  tiling_data_list.push_back(tiling_perf);");
  tiling_func_.AddLine("  if (best_perf > cur_perf) {");
  tiling_func_.AddLine("    *output_tiling_data = autofuse_tiling_data_tmp;");
  tiling_func_.AddLine("    best_perf = cur_perf;");
  tiling_func_.AddLine("  }");
  tiling_func_.AddLine("  return true;");
  tiling_func_.AddLine("}");
  GELOGD("Generate pgo reuse group tiling wrapper for %s of %s success.", cur_prefix.c_str(), op_name_.c_str());
  return af::SUCCESS;
}

bool TilingCodeGenImpl::IsScheduleResultEnableParallel(const size_t asc_graph_id, const size_t impl_graph_id) const {
  bool enable_group_parallel = false;
  for (const auto &info : tiling_model_info_) {
    if (info.schedule_group_ident.asc_graph_id == asc_graph_id &&
        info.schedule_group_ident.impl_graph_id == impl_graph_id) {
      enable_group_parallel = info.enable_group_parallel;
      break;
    }
  }
  GELOGD("Enable parallel flag of graph%d_result%d is: %d", asc_graph_id, impl_graph_id, enable_group_parallel);
  return enable_group_parallel;
}

bool TilingCodeGenImpl::GenUpdateCurPerfAndBlockByGroupIfNeeded(const size_t asc_graph_id,
                                                                const AscGraphNamepspaceMap &asc_graph_map) const {
  for (const auto &graph_info : asc_graph_map) {
    if (IsScheduleResultEnableParallel(asc_graph_id, graph_info.first)) {
      GenUpdateCurPerfAndBlockByGroup();
      return true;
    }
  }
  return false;
}
// ======================== Cache line conflict helper functions ========================

bool TilingCodeGenImpl::IsConflictCacheLineConfig(const CacheLineConfig &cfg) {
  return cfg.IsCacheLineConflictCandidate();
}

std::pair<std::string, bool> TilingCodeGenImpl::GenConflictExprContextCode(
    const ModelInfo &model_info, const ge::Expression &expr, std::set<std::string> &declared_symbols) const {
  std::string code;
  std::set<std::string> input_var_names;
  ArgsManager args_manager(model_info);
  if (args_manager.Process(false)) {
    auto input_vars = GetVarsNames(args_manager.GetInputVars());
    input_var_names.insert(input_vars.begin(), input_vars.end());
  }
  auto emit_decl = [&code, &declared_symbols](const std::string &name, const std::string &src) {
    code += "    auto " + name + " = " + src + ".get_" + name + "();\n";
    declared_symbols.insert(name);
  };
  for (const auto &symbol : expr.FreeSymbols()) {
    const std::string name = Str(symbol);
    if (declared_symbols.count(name) != 0U) {
      continue;
    }
    if (name == "block_dim") {
      emit_decl(name, "tiling_data");
      continue;
    }
    bool is_hw = std::any_of(model_info.hardware_cons.begin(), model_info.hardware_cons.end(),
                             [&](const auto &p) { return BaseTypeUtils::DumpHardware(p.first) == name; });
    if (is_hw) {
      emit_decl(name, "tiling_data");
      continue;
    }
    if (model_info.container_exprs.count(name) != 0U || model_info.tensor_exprs.count(name) != 0U) {
      emit_decl(name, "group_tiling_data");
      continue;
    }
    if (input_var_names.count(name) != 0U) {
      emit_decl(name, "group_tiling_data");
      continue;
    }
    bool is_arg = std::any_of(model_info.arg_list.begin(), model_info.arg_list.end(),
                              [&](const auto &a) { return a->name == name; });
    if (is_arg) {
      emit_decl(name, "group_tiling_data");
      continue;
    }
    return {"", false};
  }
  return {code, true};
}

af::Status TilingCodeGenImpl::GenConflictGroupHelper(const ModelInfo &model_info,
                                                     const std::string &group_item_prefix) {
  const auto &ident = model_info.schedule_group_ident;
  const std::string helper_name = "IsConflictGroup_" + std::to_string(ident.asc_graph_id) + "_" +
                                  std::to_string(ident.impl_graph_id) + "_" + std::to_string(ident.group_id) + "_" +
                                  std::to_string(model_info.tiling_case_id);
  tiling_func_.AddLine("  auto " + helper_name + " = [&]() -> bool {");
  if (model_info.tiling_schedule_config_table == nullptr ||
      !model_info.tiling_schedule_config_table->IsEnableCacheLineCheck()) {
    tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"cache line size is unavailable, fallback to normal group\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  };");
    tiling_func_.AddLine("");
    return af::SUCCESS;
  }
  const uint32_t cache_line_size = model_info.tiling_schedule_config_table->GetCacheLineSize();
  tiling_func_.AddLine("    auto &group_tiling_data = tiling_data." + group_item_prefix + "_tiling_data;");
  std::set<std::string> declared_symbols;
  bool has_valid_expr = false;
  for (const auto &cfg : model_info.cache_line_config) {
    if (!IsConflictCacheLineConfig(cfg)) {
      continue;
    }
    const auto [context_code, ok] = GenConflictExprContextCode(model_info, cfg.cache_line_expr, declared_symbols);
    if (!ok) {
      tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"cache line expr is not codegenable, fallback to normal group\");");
      tiling_func_.AddLine("    return false;");
      tiling_func_.AddLine("  };");
      tiling_func_.AddLine("");
      return af::SUCCESS;
    }
    has_valid_expr = true;
    tiling_func_.AddLine(context_code);
    const uint32_t cfg_cache_line_size = cfg.cache_line_size > 0 ? cfg.cache_line_size : cache_line_size;
    tiling_func_.AddLine("    if (" + Str(cfg.cache_line_expr) + " < " + std::to_string(cfg_cache_line_size) + ") {");
    tiling_func_.AddLine("      return true;");
    tiling_func_.AddLine("    }");
  }
  if (!has_valid_expr) {
    tiling_func_.AddLine("    OP_LOGD(OP_NAME, \"no valid gm<->ub cache line expr, fallback to normal group\");");
    tiling_func_.AddLine("    return false;");
    tiling_func_.AddLine("  };");
    tiling_func_.AddLine("");
    return af::SUCCESS;
  }
  tiling_func_.AddLine("    return false;");
  tiling_func_.AddLine("  };");
  tiling_func_.AddLine("");
  return af::SUCCESS;
}

af::Status TilingCodeGenImpl::GenConflictGroupHelpers(
    const size_t asc_graph_id, const size_t impl_graph_id,
    const std::map<size_t, std::pair<std::string, std::string>> &graph_info) {
  for (const auto &[group_id, group_info] : graph_info) {
    std::set<uint32_t> generated_tiling_keys;
    for (const auto &model_info : tiling_model_info_) {
      const auto &ident = model_info.schedule_group_ident;
      if (ident.asc_graph_id == asc_graph_id && ident.impl_graph_id == impl_graph_id && ident.group_id == group_id) {
        const uint32_t final_tiling_key = static_cast<uint32_t>(model_info.tiling_case_id);
        if (generated_tiling_keys.find(final_tiling_key) != generated_tiling_keys.end()) {
          GELOGD("Duplicate final tiling key %u for group %zu, skip conflict helper generation.", final_tiling_key,
                 group_id);
          continue;
        }
        generated_tiling_keys.insert(final_tiling_key);
        GE_ASSERT_SUCCESS(GenConflictGroupHelper(model_info, group_info.second));
      }
    }
  }
  return af::SUCCESS;
}

std::string TilingCodeGenImpl::GenConflictGroupInvoke(const size_t asc_graph_id, const size_t impl_graph_id,
                                                      size_t group_id, const std::string &group_item_prefix) const {
  std::map<uint32_t, std::string> key_to_helper;
  for (const auto &model_info : tiling_model_info_) {
    const auto &ident = model_info.schedule_group_ident;
    if (ident.asc_graph_id == asc_graph_id && ident.impl_graph_id == impl_graph_id && ident.group_id == group_id) {
      const uint32_t final_tiling_key = static_cast<uint32_t>(model_info.tiling_case_id);
      const std::string helper_name = "IsConflictGroup_" + std::to_string(asc_graph_id) + "_" +
                                      std::to_string(impl_graph_id) + "_" + std::to_string(group_id) + "_" +
                                      std::to_string(model_info.tiling_case_id);
      if (key_to_helper.count(final_tiling_key) != 0U) {
        return "([&]() -> bool { OP_LOGD(OP_NAME, \"duplicate final tiling key mapping, fallback to normal group\"); "
               "return false; })()";
      }
      key_to_helper.emplace(final_tiling_key, helper_name);
    }
  }

  std::string code;
  code += "([&]() -> bool {\n";
  code += "  const uint32_t final_tiling_key = tiling_data." + group_item_prefix + "_tiling_data.get_tiling_key();\n";
  code += "  switch (final_tiling_key) {\n";
  for (const auto &[final_tiling_key, helper_name] : key_to_helper) {
    code += "    case " + std::to_string(final_tiling_key) + ": return " + helper_name + "();\n";
  }
  code +=
      "    default: OP_LOGD(OP_NAME, \"no conflict helper matched final tiling key, fallback to normal group\"); "
      "return false;\n";
  code += "  }\n";
  code += "})()";
  return code;
}

std::string TilingCodeGenImpl::GenMixedPerfUpdateCode(const std::vector<std::string> &groups_perf,
                                                      const std::vector<std::string> &groups_block_num,
                                                      const std::vector<std::string> &groups_conflict_flags,
                                                      const std::string &indent) {
  if (groups_perf.size() == 1UL) {
    return indent + "cur_perf = " + groups_perf[0] + ";\n";
  }
  std::string code;
  code += indent + "double conflict_perf_sum = 0.0;\n";
  code += indent + "double normal_perf_merged = 0.0;\n";
  code += indent + "bool has_normal_group = false;\n";
  code += indent + "double cur_tmp_perf = 0.0;\n";
  code += indent + "uint32_t cur_block = 0U;\n";

  for (size_t id = 0UL; id < groups_perf.size(); ++id) {
    code += indent + "if (" + groups_conflict_flags[id] + ") {\n";
    // conflict group: sum the perf directly
    code += indent + "  OP_LOGD(OP_NAME, \"Conflict group perf %lf\", " + groups_perf[id] + ");\n";
    code += indent + "  conflict_perf_sum += " + groups_perf[id] + ";\n";
    code += indent + "} else if (!has_normal_group) {\n";
    // first normal group: initialize cur_tmp_perf and cur_block
    code += indent + "  cur_tmp_perf = " + groups_perf[id] + ";\n";
    code += indent + "  cur_block = " + groups_block_num[id] + ";\n";
    code += indent + "  has_normal_group = true;\n";
    code += indent + "} else {\n";
    // subsequent normal group: reuse existing parallel merge logic
    code += indent + "  (void)UpdateCurPerfAndBlockByGroup({" + groups_block_num[id] + ", " + groups_perf[id] +
            "}, ori_block_dim, cur_block, normal_perf_merged, cur_tmp_perf);\n";
    code += indent + "}\n";
  }

  // flush remaining normal perf
  code += indent + "if (has_normal_group) {\n";
  code += indent + "  OP_LOGD(OP_NAME, \"Final normal perf %lf\", cur_tmp_perf);\n";
  code += indent + "  normal_perf_merged += cur_tmp_perf;\n";
  code += indent + "}\n";

  code +=
      indent + "OP_LOGD(OP_NAME, \"Mixed perf: conflict=%lf, normal=%lf\", conflict_perf_sum, normal_perf_merged);\n";
  code += indent + "cur_perf = conflict_perf_sum + normal_perf_merged;\n";
  return code;
}
}  // namespace att
