/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen.h"

#include <set>
#include <sstream>
#include <fstream>

#include "ascir_node_param/ascir_param_builder.h"
#include "codegen_kernel.h"
#include "codegen_tiling_data.h"
#include "ascir_utils.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "codegen_infershape.h"
#include "common/platform_context.h"
#include "gen_model_info/api_perf_register/soc_perf_config.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"

using namespace codegen;
using namespace ascgen_utils;

namespace {
constexpr uint32_t ELEMENTS_PER_LINE = 20;
constexpr size_t kMaxUnfoldedIoNum = 64U;
constexpr size_t kKernelMaxIoNum = 190U;
constexpr const char *kSplitFileBegin = "// AUTOFUSE_SPLIT_FILE_BEGIN: ";
constexpr const char *kSplitFileEnd = "// AUTOFUSE_SPLIT_FILE_END: ";

// Include path prefixes to be removed
const std::string kBasicApiInclude = "#include \"basic_api/";
const std::string kAdvApiInclude = "#include \"adv_api/";
const std::string kMicroApiInclude = "#include \"micro_api/";
const std::string kSimtApiInclude = "#include \"simt_api/";
const std::string kUtilsStdInclude = "#include \"utils/std/";

std::string GetKernelTaskType(const ascir::FusedScheduledResult &schedule_results) {
  if (ascgen_utils::IsJustCubeFixpip(schedule_results)) {
    return kKernelTaskTypeAICOnly;
  } else if (ascgen_utils::IsCubeFusedScheduled(schedule_results)) {
    if (ascgen_utils::IsConv2DFusedScheduled(schedule_results)) {
      std::string npu_arch;
      GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
      if (npu_arch == "5102") {
        return kKernelTaskTypeMixAICOneOne;
      } else {
        return kKernelTaskTypeMixAICOneTwo;
      }
    } else {
      return kKernelTaskTypeMixAICOneTwo;
    }
  }
  return schedule_results.workspace_nodes.size() != 0 ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly;
}

std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(kTilingHeadGuard) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

const void *GetAscirParamGraphKey(const ascir::ImplGraph &impl_graph) {
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  return compute_graph == nullptr ? static_cast<const void *>(&impl_graph)
                                  : static_cast<const void *>(compute_graph.get());
}

Status EnrichScheduleGroupAscirParams(const ascir::ScheduleGroup &schedule_group,
                                      std::set<const void *> &enriched_graphs) {
  for (const auto &impl_graph : schedule_group.impl_graphs) {
    const auto graph_key = GetAscirParamGraphKey(impl_graph);
    if (!enriched_graphs.insert(graph_key).second) {
      continue;
    }
    GE_ASSERT_SUCCESS(ascir_param::EnrichAscirGraphNodeParams(impl_graph), "Enrich ascir node params failed.");
  }
  return af::SUCCESS;
}

Status EnrichScheduledResultAscirParams(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::set<const void *> enriched_graphs;
  for (const auto &asc_graph_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : asc_graph_results) {
      for (const auto &schedule_group : scheduled_result.schedule_groups) {
        GE_ASSERT_SUCCESS(EnrichScheduleGroupAscirParams(schedule_group, enriched_graphs),
                          "Enrich ascir node params failed.");
      }
    }
  }
  return af::SUCCESS;
}

bool IsSplitTilingHeaderKey(const std::string &key) {
  return key == kTilingStateHeaderIdentify || key == kTilingLogHeaderIdentify || key == kTilingPgoHeaderIdentify ||
         key == kTilingSolverHeaderIdentify || key == kTilingApiHeaderIdentify || key == kTilingBaseHeaderIdentify ||
         key == kTilingEntryHeaderIdentify || key == kTilingTailHeaderIdentify;
}

std::string RemoveSplitCppIncludes(const std::string &content);

Status CombineTilings(const std::map<std::string, std::string> &tiling_file_name_to_content, std::string &result) {
  GE_CHK_BOOL_RET_STATUS(tiling_file_name_to_content.find(kTilingHeadIdentify) != tiling_file_name_to_content.end(),
                         af::FAILED, "tiling_file_name_to_content has no tiling head");
  result += RemoveAutoFuseTilingHeadGuards(
      tiling_file_name_to_content.at(kTilingHeadIdentify));  // 删除头文件的宏保护，cpp文件不需要

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tiling_file_name_to_content) {
    if (key == kTilingHeadIdentify || IsSplitTilingHeaderKey(key) ||
        key.find(kTilingDataIdentify) != std::string::npos) {
      continue;
    }

    result += RemoveSplitCppIncludes(value);

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }

  return af::SUCCESS;
}

void AppendSplitBegin(const std::string &key, std::string &result) {
  result += kSplitFileBegin;
  result += key;
  result += "\n";
}

void AppendSplitEnd(const std::string &key, std::string &result) {
  if (!result.empty() && result.back() != '\n') {
    result += '\n';
  }
  result += kSplitFileEnd;
  result += key;
  result += "\n";
}

std::string RemoveSplitCppIncludes(const std::string &content) {
  const std::set<std::string> split_includes = {
      kTilingHeadInclude,       kTilingStateHeaderInclude,      kTilingLogHeaderInclude, kTilingPgoHeaderInclude,
      kTilingBaseHeaderInclude, kTilingSolverHeaderInclude,     kTilingApiHeaderInclude, kTilingEntryHeaderInclude,
      kTilingTailHeaderInclude, kCubeKernelTilingWrapperInclude};
  std::istringstream input(content);
  std::stringstream output;
  std::string line;
  bool in_include_prefix = true;
  while (std::getline(input, line)) {
    if (in_include_prefix && !line.empty() && line.rfind("#include ", 0U) != 0U) {
      in_include_prefix = false;
    }
    if (!in_include_prefix || split_includes.count(line) == 0U) {
      output << line << '\n';
    }
  }
  return output.str();
}

void AppendSplitSource(const std::string &key, const std::string &content, std::string &result) {
  AppendSplitBegin(key, result);
  result += content;
  AppendSplitEnd(key, result);
}

std::string BuildSplitHeaderContent(const std::map<std::string, std::string> &tiling_file_name_to_content) {
  return RemoveAutoFuseTilingHeadGuards(tiling_file_name_to_content.at(kTilingHeadIdentify));
}

bool ShouldSkipSplitCppSource(const std::string &key) {
  return key == kTilingHeadIdentify || IsSplitTilingHeaderKey(key) || key == kCubeKernelTilingWrapperHpp ||
         key.find(kTilingDataIdentify) != std::string::npos;
}

bool HasNewSplitTilingHeaders(const std::map<std::string, std::string> &tiling_file_name_to_content) {
  return tiling_file_name_to_content.find(kTilingStateHeaderIdentify) != tiling_file_name_to_content.end();
}

void AppendSplitHeaderSources(const std::map<std::string, std::string> &tiling_file_name_to_content, bool is_new_format,
                              std::string &result) {
  std::string tiling_head = BuildSplitHeaderContent(tiling_file_name_to_content);
  if (!is_new_format) {
    auto wrapper_header = tiling_file_name_to_content.find(kCubeKernelTilingWrapperHpp);
    if (wrapper_header != tiling_file_name_to_content.end()) {
      if (!tiling_head.empty() && tiling_head.back() != '\n') {
        tiling_head += '\n';
      }
      tiling_head += wrapper_header->second;
    }
    AppendSplitSource(kTilingHeadIdentify, tiling_head, result);
    return;
  }
  for (const auto &key : {kTilingStateHeaderIdentify, kTilingLogHeaderIdentify, kTilingPgoHeaderIdentify,
                          kTilingSolverHeaderIdentify, kTilingApiHeaderIdentify, kCubeKernelTilingWrapperHpp}) {
    auto iter = tiling_file_name_to_content.find(key);
    if (iter != tiling_file_name_to_content.end()) {
      AppendSplitSource(key, iter->second, result);
    }
  }
}

Status CombineTilingsWithSplitMarkers(const std::map<std::string, std::string> &tiling_file_name_to_content,
                                      std::string &result) {
  GE_CHK_BOOL_RET_STATUS(tiling_file_name_to_content.find(kTilingHeadIdentify) != tiling_file_name_to_content.end(),
                         af::FAILED, "tiling_file_name_to_content has no tiling head");
  const bool is_new_format = HasNewSplitTilingHeaders(tiling_file_name_to_content);
  AppendSplitHeaderSources(tiling_file_name_to_content, is_new_format, result);
  for (const auto &[key, value] : tiling_file_name_to_content) {
    if (ShouldSkipSplitCppSource(key)) {
      continue;
    }
    AppendSplitSource(key, is_new_format ? value : RemoveSplitCppIncludes(value), result);
  }
  return af::SUCCESS;
}

struct ScanResult {
  int open_braces = 0;               // 当前行的开大括号{总数
  int close_braces = 0;              // 当前行的闭大括号}总数
  int line_starts_close_braces = 0;  // 当前行首连续闭大括号数量
  bool has_preprocessor = false;     // 当前行是否预处理宏
};

// 处理字符串状态
void StringState(char c, size_t i, const std::string &line, bool &in_string, char &string_char) {
  if (!in_string && (c == '"' || c == '\'')) {
    in_string = true;
    string_char = c;
  } else if (in_string && c == string_char && (i == 0 || line[i - 1] != '\\')) {
    in_string = false;
  }
}

// 处理单行/多行注释状态
bool CommentState(char c, char next_c, size_t &i, bool in_string, bool &in_block_comment, bool &in_line_comment) {
  if (!in_string && !in_block_comment && !in_line_comment) {
    if (c == '/' && next_c == '/') {
      in_line_comment = true;
      return true;
    } else if (c == '/' && next_c == '*') {
      in_block_comment = true;
      i++;
      return true;
    }
  } else if (in_block_comment && c == '*' && next_c == '/') {
    in_block_comment = false;
    i++;
    return true;
  }
  return false;
}

// 大括号统计
void BraceCount(char c, bool &in_line_start_close_brace, ScanResult &result) {
  if (in_line_start_close_brace && !std::isspace(static_cast<unsigned char>(c)) &&
      (c != '}')) {  // 判断是否仍在闭大括号行首状态
    in_line_start_close_brace = false;
  }
  if (c == '{') {
    result.open_braces++;
  } else if (c == '}') {
    result.close_braces++;
    if (in_line_start_close_brace) {
      result.line_starts_close_braces++;
    }
  }
}

ScanResult ScanLine(const std::string &line, bool &in_block_comment, bool &in_string, char &string_char) {
  ScanResult result;
  bool in_line_comment = false;           // 是否在//单行注释内
  bool in_line_start_close_brace = true;  // 闭大括号是否在行首
  for (size_t i = 0; i < line.length(); ++i) {
    char c = line[i];
    char next_c = (i + 1 < line.length()) ? line[i + 1] : '\0';
    // 1.1 字符串状态判断
    if (!in_block_comment && !in_line_comment) {
      StringState(c, i, line, in_string, string_char);
    }
    // 1.2 单行/多行注释状态判断
    bool should_continue = CommentState(c, next_c, i, in_string, in_block_comment, in_line_comment);
    if (should_continue) {
      if (in_line_comment) break;
      continue;
    }
    // 1.3 不在字符串或注释状态，统计大括号数量
    if (!in_string && !in_block_comment && !in_line_comment) {
      BraceCount(c, in_line_start_close_brace, result);
    }
    // 1.4 是否是宏判断
    if (!in_string && !in_block_comment && !in_line_comment && i == line.find_first_not_of(" \t") && c == '#') {
      result.has_preprocessor = true;
    }
  }
  return result;
}

std::string FormatIndentation(const std::string &code) {
  std::istringstream iss(code);
  std::ostringstream oss;
  std::string line;
  int indent_level = 0;           // 当前缩进层级
  bool in_block_comment = false;  // 是否在/* */多行注释内
  bool in_string = false;         // 是否在字符串常量内
  char string_char = '\0';        // 字符串的引号类型

  // 逐行遍历处理
  while (std::getline(iss, line)) {
    if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
      oss << "\n";
      continue;
    }

    // 阶段1 单行扫描
    ScanResult result = ScanLine(line, in_block_comment, in_string, string_char);

    // 阶段2 格式化缩进
    std::string trimmed = line;
    // 2.1 去掉全部前导空格
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    // 2.2 宏不做缩进处理
    if (result.has_preprocessor) {
      oss << trimmed << "\n";
      continue;
    }
    // 2.3 按照缩进等级增加缩进
    int current_indent = indent_level;          // 当前缩进等级
    if (result.line_starts_close_braces > 0) {  // 如果行首有闭大括号，先减少缩进（用于当前行）
      current_indent = std::max(0, indent_level - result.line_starts_close_braces);
    }

    // 阶段3 输出当前行
    oss << std::string(current_indent * 2, ' ') << trimmed << "\n";

    // 更新下一行的缩进等级
    indent_level = indent_level - result.close_braces + result.open_braces;
    indent_level = std::max(0, indent_level);
  }
  return oss.str();
}

}  // namespace

std::string RemoveSubDirInclude(const std::string &kernel_str) {
  std::string result = R"(
#include "kernel_operator.h"
)";
  std::stringstream ss(kernel_str);
  std::string line;
  while (std::getline(ss, line)) {
    auto shouldRemove = [&line]() {
      return line.compare(0, kBasicApiInclude.size(), kBasicApiInclude) == 0 ||
             line.compare(0, kAdvApiInclude.size(), kAdvApiInclude) == 0 ||
             line.compare(0, kMicroApiInclude.size(), kMicroApiInclude) == 0 ||
             line.compare(0, kSimtApiInclude.size(), kSimtApiInclude) == 0 ||
             line.compare(0, kUtilsStdInclude.size(), kUtilsStdInclude) == 0;
    };
    if (!shouldRemove()) {
      result += line + "\n";
    }
  }
  return result;
}

Codegen::Codegen(const CodegenOptions &options)
    : tiling_lib_(options.tiling_lib_path, options.tiling_lib_codegen_symbol),
      using_att_calc_qbt_size_(options.using_att_calc_qbt_size) {}

Status Codegen::Generate(const ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const {
  // for UT/ST shape info stub
  std::map<std::string, std::string> shape_info;
  return this->Generate(shape_info, fused_schedule_result, result);
}

// inductor路径仍返回单个host_tiling字符串，通过注释marker保留拆分边界
Status Codegen::GenerateForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                    CodegenResult &result) const {
  GE_CHK_STATUS_RET(att::ValidateSocPerfEntry(), "Validate SoC performance target failed");
  if (tiling_lib_.ShouldFallbackPgo(fused_schedule_result)) {
    GELOGW("Tiling key count exceeds 10000, fallback to non-PGO Inductor codegen");
    Codegen fallback(*this);
    fallback.tiling_lib_.DisableInductorPgo();
    return fallback.GenerateForInductor(fused_schedule_result, result);
  }
  const auto generate_tiling_without_pgo = [this, &fused_schedule_result, &result]() {
    if (!tiling_lib_.IsInductorPgoEnabled() || ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) ||
        !ascgen_utils::IsStaticSchedResult(fused_schedule_result)) {
      return af::FAILED;
    }
    GELOGW("Inductor PGO codegen failed, fallback to non-PGO codegen for static non-CV kernel");
    Codegen fallback(*this);
    fallback.tiling_lib_.DisableInductorPgo();
    return fallback.GenerateForInductor(fused_schedule_result, result);
  };
  GE_CHK_STATUS_RET(GenerateKernel(fused_schedule_result, result.kernel, true), "Codegen generate kernel failed");
  result.tiling_data = GenerateTilingData(fused_schedule_result, true);
  std::map<std::string, std::string> tiling_file_name_to_content;
  const auto tiling_ret = GenerateTilingForInductor(fused_schedule_result, tiling_file_name_to_content);
  if (tiling_ret != af::SUCCESS) {
    if (generate_tiling_without_pgo() == af::SUCCESS) {
      return af::SUCCESS;
    }
    GE_CHK_STATUS_RET(tiling_ret, "Codegen generate tiling failed");
  }
  const auto pgo_runner = tiling_file_name_to_content.find(kPgoRunnerIdentify);
  if (pgo_runner != tiling_file_name_to_content.end()) {
    GE_CHK_BOOL_RET_STATUS(!pgo_runner->second.empty() && !result.kernel.empty(), af::FAILED,
                           "Inductor PGO runner or device source is empty");
    tiling_file_name_to_content[kPgoDeviceSourceIdentify] = result.kernel;
  }
  GE_CHK_STATUS_RET(CombineTilingsWithSplitMarkers(tiling_file_name_to_content, result.tiling));
  return af::SUCCESS;
}

Status Codegen::Generate(const std::map<std::string, std::string> &shape_info,
                         const ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const {
  GE_CHK_STATUS_RET(att::ValidateSocPerfEntry(), "Validate SoC performance target failed");
  GE_CHK_STATUS_RET(GenerateKernel(fused_schedule_result, result.kernel, false), "Codegen generate kernel failed");
  result.tiling_data = GenerateTilingData(fused_schedule_result, false);
  std::map<std::string, std::string> tiling_file_name_to_content;
  GE_CHK_STATUS_RET(GenerateTiling(fused_schedule_result, shape_info, "", "0", tiling_file_name_to_content));
  GE_CHK_STATUS_RET(CombineTilings(tiling_file_name_to_content, result.tiling));

  return af::SUCCESS;
}

std::string Codegen::GenerateTilingData(const ascir::FusedScheduledResult &fused_schedule_result,
                                        bool is_inductor) const {
  std::stringstream ss;
  ss << TilingData("Autofuse").Generate(fused_schedule_result, is_inductor);
  return ss.str();
}

Status Codegen::GenerateTilingForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                          std::map<std::string, std::string> &tiling_file_name_to_content) const {
  GE_CHK_STATUS_RET(att::ValidateSocPerfEntry(), "Validate SoC performance target failed");
  tiling_file_name_to_content = this->tiling_lib_.GenerateForInductor(fused_schedule_result);
  for (const auto &pair : tiling_file_name_to_content) {
    GE_CHK_BOOL_RET_STATUS(pair.second != ascgen_utils::INVALID_TILING, af::FAILED, "tilings(%s) is invalid",
                           pair.second.c_str());
  }
  return af::SUCCESS;
}

std::map<std::string, std::string> Codegen::GenerateTilingForInductor(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::map<std::string, std::string> tiling_file_name_to_content;
  (void)GenerateTilingForInductor(fused_schedule_result, tiling_file_name_to_content);
  return tiling_file_name_to_content;
}

Status Codegen::GenerateTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                               const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
                               const std::string &core_num,
                               std::map<std::string, std::string> &tiling_file_name_to_content) const {
  GE_CHK_STATUS_RET(att::ValidateSocPerfEntry(), "Validate SoC performance target failed");
  tiling_file_name_to_content = this->tiling_lib_.Generate(fused_schedule_result, shape_info, pgo_dir, core_num);
  for (const auto &pair : tiling_file_name_to_content) {
    GE_CHK_BOOL_RET_STATUS(pair.second != ascgen_utils::INVALID_TILING, af::FAILED, "tilings(%s) is invalid",
                           pair.second.c_str());
  }
  return af::SUCCESS;
}

std::map<std::string, std::string> Codegen::GenerateTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                                           const std::map<std::string, std::string> &shape_info,
                                                           const std::string &pgo_dir,
                                                           const std::string &core_num) const {
  std::map<std::string, std::string> tiling_file_name_to_content;
  (void)GenerateTiling(fused_schedule_result, shape_info, pgo_dir, core_num, tiling_file_name_to_content);
  return tiling_file_name_to_content;
}

std::string Codegen::GenerateInferShape(const std::vector<std::vector<std::string>> &symbol_shape_str,
                                        const std::map<std::string, string> &shape_info) const {
  InfershapeGen gen;
  return gen.GenInferShapeFunc(symbol_shape_str, shape_info);
}

std::string Codegen::GeneratorPgo(const ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::string &pgo_dir) const {
  return this->tiling_lib_.GenerateForPgo(fused_schedule_result, pgo_dir);
}

Status Codegen::GenerateKernel(const ascir::FusedScheduledResult &fused_schedule_result, std::string &result,
                               bool is_inductor) const {
  ascir::utils::FusedGraphNameGuard guard(fused_schedule_result.fused_graph_name.GetString());
  GE_CHK_STATUS_RET(EnrichScheduledResultAscirParams(fused_schedule_result), "Enrich ascir node params failed.");
  const auto io_num = fused_schedule_result.input_nodes.size() + fused_schedule_result.output_nodes.size();
  bool use_list_tensor = io_num >= kMaxUnfoldedIoNum;
  CodegenConfig config = {is_inductor, using_att_calc_qbt_size_};
  if (is_inductor) {
    use_list_tensor = false;
    GE_LOGW_IF(io_num >= kKernelMaxIoNum, "Too many io, io num is %zu", io_num);
  }
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  GELOGI("kernel_name = %s, num_inputs = %zu, num_outputs = %zu, use list tensor desc = %d", graph_name.c_str(),
         fused_schedule_result.input_nodes.size(), fused_schedule_result.output_nodes.size(),
         static_cast<int32_t>(use_list_tensor));
  std::stringstream ss;
  std::string kernel_task_type = GetKernelTaskType(fused_schedule_result);
  ss << Kernel::IncludeAndDefines(fused_schedule_result, kernel_task_type, use_list_tensor, is_inductor);
  GE_CHK_STATUS_RET(
      Kernel::GenKernelFuncByTilingKey(fused_schedule_result, ss, use_list_tensor, config, kernel_task_type),
      "Generate kernel func by tiling_key failed.");
  if (is_inductor) {
    ss << Kernel::GenKernelFuncCallForInductor(fused_schedule_result);
  }
  result = FormatIndentation(ss.str());
  return af::SUCCESS;
}

std::string Codegen::GenGetKernelAndJson(const std::string &kernel_path, const std::string &json_path) const {
  // 当前只获取kernel，后续计划json也会打包起来
  (void)json_path;
  std::stringstream ss;
  std::string real_kernel_path;
  if (!ascgen_utils::GetRealPath(kernel_path, real_kernel_path)) {
    GELOGE(af::FAILED, "kernel_path::%s realpath failed", kernel_path.c_str());
    return "";
  }
  std::ifstream kernel_file(real_kernel_path, std::ios::binary | std::ios::ate);
  if (!kernel_file.is_open()) {
    GELOGE(af::FAILED, "kernel_path::%s open failed", kernel_path.c_str());
    return "";
  }

  std::streamsize kernel_file_size = kernel_file.tellg();
  kernel_file.seekg(0, std::ios::beg);
  std::vector<uint8_t> kernel_data(kernel_file_size);
  kernel_file.read(reinterpret_cast<char *>(kernel_data.data()), kernel_file_size);
  kernel_file.close();

  ss << "#include <cstdint>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <vector>" << std::endl;
  ss << "extern \"C\" void GetKernelBin(std::vector<char> &kernel_bin) {" << std::endl;
  if (kernel_file_size == 0) {
    ss << "  std::vector<uint8_t> temp_kernel = {};" << std::endl;
    ss << "  return;" << std::endl;
  } else {
    ss << "  static const uint8_t temp_kernel[] = {";
    for (uint32_t i = 0; i < kernel_file_size; i++) {
      if (i % ELEMENTS_PER_LINE == 0) {
        ss << std::endl << "    ";
      }
      ss << std::to_string(kernel_data[i]) << ", ";
    }
    ss << "};" << std::endl;

    ss << "  kernel_bin.resize(sizeof(temp_kernel));" << std::endl;
    ss << "  std::memcpy(kernel_bin.data(), temp_kernel, sizeof(temp_kernel));" << std::endl;
  }
  ss << "}";
  return ss.str();
}
