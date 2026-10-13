/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"
#include "codegen_tiling_data.h"
#include "codegen_tiling_utils.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <string>
#include <cstdlib>
#include <set>
#include <fstream>
#include <securec.h>

#include "dlfcn.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"

#include "common_utils.h"
#include "gen_tiling_impl.h"
#include "common/ge_common/debug/log.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "graph/ge_context.h"
#include "backend/backend_spec.h"
#include "common/ascgraph_info_complete.h"
#include "codegen_tiling_cube_wrapper.h"
#include "common/tiling_source_dependencies.h"

namespace codegen {
using optimize::AscGraphInfoComplete;
using optimize::SizeVarSet;
using namespace af::ascir_op;
using namespace ascir;
using namespace codegen;
using namespace af::ops;
using namespace ascgen_utils;
namespace {
constexpr uint64_t kMaxPgoTilingKeyCount = 10000U;
constexpr uint64_t kInt64TilingKeyCapacity = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1U;
// fp32且K轴不小于该阈值且未启用hf32时, CV融合UB模板存在精度问题, 需走common兜底模板
constexpr int64_t kFp32LargeKThreshold = 2048;

std::string GenWorkspaceCheckedHelpers() {
  std::stringstream ss;
  ss << "inline bool CheckedWorkspaceAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {\n"
     << "  return !__builtin_add_overflow(lhs, rhs, &result);\n"
     << "}\n"
     << "inline bool CheckedWorkspaceAlign(uint64_t size, uint64_t alignment, uint64_t &result) {\n"
     << "  if (alignment == 0U) {\n"
     << "    return false;\n"
     << "  }\n"
     << "  uint64_t aligned = 0U;\n"
     << "  if (__builtin_add_overflow(size, alignment - 1U, &aligned)) {\n"
     << "    return false;\n"
     << "  }\n"
     << "  result = aligned / alignment * alignment;\n"
     << "  return true;\n"
     << "}\n";
  return ss.str();
}

std::string GenUint64Literal(uint64_t value) {
  return std::to_string(value) + (value >= kInt64TilingKeyCapacity ? "ULL" : "");
}

void GenInductorCvSafetyFallback(std::stringstream &ss, uint64_t count, const std::string &indent) {
  ss << indent << "set_g_basen_basem_align(1);" << std::endl;
  ss << indent << "uint32_t vec_core_num = limit->aiv_num;" << std::endl;
  ss << indent << "tiling->tiling_data.set_block_dim(vec_core_num);" << std::endl;
  ss << indent << "tiling->tiling_data.set_ub_size(limit->ub_size - 256);" << std::endl;
  ss << indent << "double min_perf = DBL_MAX;" << std::endl;
  ss << indent << "size_t choice_case_id = 2U;" << std::endl;
  ss << indent << "for (size_t i = 2U; i < " << GenUint64Literal(count) << "; i++) {" << std::endl;
  ss << indent << "  double cur_perf;" << std::endl;
  ss << indent << "  if (!optiling::GetTiling(tiling->tiling_data, i, &cur_perf)) {" << std::endl;
  ss << indent << "    return -1;" << std::endl;
  ss << indent << "  }" << std::endl;
  ss << indent << "  if (cur_perf < min_perf) {" << std::endl;
  ss << indent << "    min_perf = cur_perf;" << std::endl;
  ss << indent << "    choice_case_id = i;" << std::endl;
  ss << indent << "  }" << std::endl;
  ss << indent << "}" << std::endl;
  ss << indent << "if (!optiling::GetTiling(tiling->tiling_data, choice_case_id)) {" << std::endl;
  ss << indent << "  return -1;" << std::endl;
  ss << indent << "}" << std::endl;
  ss << indent << "tiling->stage_size_name = tiling->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << indent << "tiling->tiling_data.set_tiling_key(tiling->tiling_data.get_tiling_key() - 2);" << std::endl;
  ss << indent << "// Subtract 2 from tiling_key because case 0/1 are reserved for CV UB normal/fallback tiling."
     << std::endl;
  ss << indent << "const bool is_cv_safety_aiv_only = is_cv_safety_aiv_only_mode(cube_tiling_key);" << std::endl;
  ss << indent << "const bool is_cv_safety_mix = is_cv_safety_mix_mode(cube_tiling_key);" << std::endl;
  ss << indent << "const bool use_launch_aic_num = is_cv_safety_blockidx_scheduled_mode(cube_tiling_key);" << std::endl;
  ss << indent << "uint32_t vec_block_dim = tiling->tiling_data.get_block_dim();" << std::endl;
  ss << indent << "uint64_t vec_wss = 0U;" << std::endl;
  ss << indent << "if (!GetWorkspaceSize(tiling->tiling_data, vec_wss)) {" << std::endl;
  ss << indent << "  return -1;" << std::endl;
  ss << indent << "}" << std::endl;
  ss << indent
     << "*blockDim = is_cv_safety_aiv_only ? vec_block_dim : "
        "((cube_block_dim * 2 < vec_block_dim) ? (vec_block_dim + 1) / 2 : cube_block_dim);"
     << std::endl;
  ss << indent << "if (!CheckedWorkspaceAdd(vec_wss, static_cast<uint64_t>(ws_size), *workspaceSize)) {" << std::endl;
  ss << indent << "  return -1;" << std::endl;
  ss << indent << "}" << std::endl;
  ss << indent << "tiling->cv_tiling_data.fusion_mode = 1;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.mix_mode = is_cv_safety_aiv_only ? 2 : (is_cv_safety_mix ? 1 : 0);"
     << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_aic_num = use_launch_aic_num ? *blockDim : cube_block_dim;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_aiv_num = vec_block_dim;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_vec_wss = vec_wss;" << std::endl;
  ss << indent << "return 0;" << std::endl;
}

// 生成cv_tiling_data字段重置代码, ub_mode区分UB模板(1)与common模板(0)
// 仅在求解器成功选出STAGE_SIZE_NAME(a0a1t)后调用: 此时行数按预算覆盖为
// STAGE_SIZE_NAME / basen_align, kernel侧队列申请与vector计算长度均以此为准
void GenCvTilingDataReset(std::stringstream &ss, const std::string &indent, uint32_t ub_mode) {
  ss << indent << "tiling->stage_size_name = tiling->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << indent
     << "tiling->cube_m_stage_size = (uint32_t)std::max(1, static_cast<int32_t>(tiling->stage_size_name / "
        "basen_align));"
     << std::endl;
  ss << indent << "tiling->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.ub_mode = " << ub_mode << ";" << std::endl;
  ss << indent << "tiling->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << indent << "tiling->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
}

// 生成tiling函数尾部的GetTiling选模板分支: case 0(UB)失败则尝试case 1, 仍失败走common兜底
void GenInductorGetTilingBranch(std::stringstream &ss, uint64_t count, uint32_t type_size) {
  ss << "  if (!optiling::GetTiling(tiling->tiling_data, 0)) {" << std::endl;
  ss << "    const uint32_t basen_basem_align_tmp = (uint32_t)basen_basem_align;" << std::endl;
  ss << "    set_g_basen_basem_align(basen_align);" << std::endl;
  ss << "    tiling->tiling_data.set_ub_size(limit->ub_size - 256 - basen_basem_align_tmp * " << type_size << ");"
     << std::endl;
  ss << "    if (!optiling::GetTiling(tiling->tiling_data, 1)) {" << std::endl;
  GenInductorCvSafetyFallback(ss, count, "      ");
  ss << "    } else {" << std::endl;
  GenCvTilingDataReset(ss, "      ", 1U);
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  GenCvTilingDataReset(ss, "    ", 0U);
  ss << "  }" << std::endl;
}

bool TryCalcTilingKeyCount(const ascir::FusedScheduledResult &result, uint64_t limit, uint64_t &count) {
  count = 0U;
  for (const auto &scheduled_results : result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      const auto &schedule_groups = scheduled_result.schedule_groups;
      const bool has_empty_group = std::any_of(schedule_groups.begin(), schedule_groups.end(),
                                               [](const auto &group) { return group.impl_graphs.empty(); });
      if (has_empty_group) {
        continue;
      }
      uint64_t per_result_count = 1U;
      for (const auto &schedule_group : schedule_groups) {
        const uint64_t impl_count = schedule_group.impl_graphs.size();
        if (impl_count == 0U || per_result_count > (limit - count) / impl_count) {
          return false;
        }
        per_result_count *= impl_count;
      }
      if (per_result_count > limit - count) {
        return false;
      }
      count += per_result_count;
    }
  }
  return true;
}

bool CheckTilingHeadersValid(const std::map<std::string, std::string> &tiling_file_name_to_content) {
  for (const auto &pair : tiling_file_name_to_content) {
    if (pair.second == INVALID_TILING) {
      GELOGE(af::FAILED, "tilings(%s) is invalid", pair.first.c_str());
      return false;
    }
  }
  return true;
}

void RequireSystemHeaders(autofuse::SourceDependencies &dependencies, std::initializer_list<const char *> headers) {
  for (const auto *header : headers) {
    autofuse::RequireSystemHeader(dependencies, header);
  }
}

void RequireEntrySystemHeaders(autofuse::SourceDependencies &dependencies, bool is_inductor, bool is_cv,
                               bool is_multi_group) {
  if (is_inductor && is_cv) {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cstddef", "cstdint", "cstring", "ostream", "sstream",
                                        "iomanip", "string", "vector"});
  } else if (is_inductor) {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "map", "ostream",
                                        "sstream", "string", "unordered_map", "vector"});
  } else {
    RequireSystemHeaders(dependencies, {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "cstdlib", "map",
                                        "ostream", "sstream", "string", "unordered_map", "vector"});
  }
  if (is_multi_group) {
    autofuse::RequireSystemHeader(dependencies, "utility");
  }
}

bool ImplGraphHasWorkspace(const ascir::ImplGraph &graph) {
  auto nodes = graph.GetAllNodes();
  for (const auto &node : nodes) {
    if (IsOps<Workspace>(node)) {
      return true;
    }
  }
  return false;
}

bool ScheduleGroupHasWorkspace(const ascir::ScheduleGroup &group) {
  return std::any_of(group.impl_graphs.begin(), group.impl_graphs.end(), ImplGraphHasWorkspace);
}

bool ScheduledResultHasWorkspace(const ascir::ScheduledResult &result) {
  return std::any_of(result.schedule_groups.begin(), result.schedule_groups.end(), ScheduleGroupHasWorkspace);
}

bool EntryWorkspaceUsesSolver(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    if (std::any_of(scheduled_results.begin(), scheduled_results.end(), ScheduledResultHasWorkspace)) {
      return true;
    }
  }
  return false;
}

struct EntryTranslationUnitOptions {
  bool is_inductor;
  bool is_cv;
  bool include_pgo;
  bool enable_pgo_runtime;
  bool include_solver;
  bool is_multi_group;
  bool include_cube_wrapper = false;
};

EntryTranslationUnitOptions GetInductorEntryTranslationUnitOptions(
    const ascir::FusedScheduledResult &fused_schedule_result, bool is_cv, bool enable_pgo_runtime) {
  return {true,
          is_cv,
          true,
          enable_pgo_runtime,
          enable_pgo_runtime || EntryWorkspaceUsesSolver(fused_schedule_result),
          !ascgen_utils::IsSingleGroup(fused_schedule_result),
          is_cv};
}

std::string RenderEntryTranslationUnit(const std::string &body, const EntryTranslationUnitOptions &options) {
  autofuse::GeneratedCode code;
  code.body = "using namespace optiling;\n\n" + body;
  RequireEntrySystemHeaders(code.dependencies, options.is_inductor, options.is_cv, options.is_multi_group);
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kTilingData);
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kLog);
  if (options.include_pgo) {
    autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kPgo);
  }
  if (options.enable_pgo_runtime) {
    RequireSystemHeaders(code.dependencies, {"fstream", "securec.h", "unordered_set", "utility"});
  }
  if (options.include_solver) {
    autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kSolver);
  }
  autofuse::RequireGeneratedHeader(code.dependencies, autofuse::GeneratedHeaderId::kApi);
  if (!options.is_inductor) {
    autofuse::RequireExternalHeaderUnlessCceKtTest(code.dependencies, "exe_graph/runtime/tiling_context.h");
    autofuse::RequireExternalHeaderUnlessCceKtTest(code.dependencies, "tiling/platform/platform_ascendc.h");
  }
  if (options.is_cv && !options.is_inductor) {
    autofuse::RequireExternalHeader(code.dependencies, "autofuse_cube_tiling_data.h");
  }
  if (options.include_cube_wrapper) {
    autofuse::RequireExternalHeader(code.dependencies, "cube_kernel_tiling_wrapper.h");
  }
  std::string output;
  GE_ASSERT_SUCCESS(autofuse::RenderTranslationUnit(code, output));
  return output;
}

void AddFallbackHeader(std::map<std::string, std::string> &headers, const std::string &key, const std::string &guard,
                       autofuse::GeneratedCode code) {
  std::string output;
  if (autofuse::RenderGeneratedHeader(code, guard, output) != af::SUCCESS) {
    return;
  }
  headers.emplace(key, std::move(output));
}

std::string GetFallbackSolverMacros() {
  return "#define Log(a) (log((double)(a)))\n"
         "#define Pow(a, b) pow(a, b)\n"
         "#define Rational(a, b) ((double)(a) / (double)(b))\n"
         "#define ExpectEq(a, b) ((a) == (b))\n"
         "#define ExpectNe(a, b) ((a) != (b))\n"
         "#define ExpectLe(a, b) ((a) <= (b))\n"
         "#define ExpectLt(a, b) ((a) < (b))\n"
         "#define LogicAnd(a, b) ((a) && (b))\n"
         "#define LogicOr(a, b) ((a) || (b))\n"
         "#define True true\n#define False false\n#define MAX_SOLUTION 50\n";
}

std::string GetFallbackSolverFunctions() {
  return R"(namespace optiling {
template <typename T, typename U>
inline auto Max(T a, U b) {
  return static_cast<double>(a) > static_cast<double>(b) ? a : b;
}
template <typename T, typename U>
inline auto Min(T a, U b) {
  return static_cast<double>(a) < static_cast<double>(b) ? a : b;
}
template <typename T>
inline auto Abs(T a) {
  return static_cast<double>(a) >= 0 ? a : -a;
}
inline bool IsEqual(double a, double b) {
  constexpr double kEpsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < kEpsilon;
}
template <typename T1, typename T2>
inline double TernaryOp(bool cond, T1 a, T2 b) {
  return static_cast<double>(cond ? a : b);
}
template <typename T>
inline T Ceiling(T a) {
  T value = static_cast<T>(static_cast<int64_t>(a));
  return IsEqual(value, a) ? value : (value + 1);
}
template <typename T>
inline T Floor(T a) {
  return static_cast<T>(static_cast<int64_t>(a));
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) -> decltype(a % b) {
  return a % b;
}
template <typename T1, typename T2>
inline auto Mod(T1 a, T2 b) -> typename std::enable_if<std::is_floating_point<T1>::value ||
                                                       std::is_floating_point<T2>::value,
                                                   decltype(std::fmod(a, b))>::type {
  return std::fmod(a, b);
}
template <typename TI, typename TO>
inline TO &RefToRef(TI &value) {
  return *(reinterpret_cast<TO *>(reinterpret_cast<void *>(&value)));
}
}  // namespace optiling
)";
}

void EnsureFallbackAtomicHeaders(std::map<std::string, std::string> &headers, const std::string &pgo_body,
                                 const std::string &api_body) {
  if (headers.find(kTilingStateHeaderIdentify) != headers.end()) {
    return;
  }
  autofuse::GeneratedCode state;
  state.body = "namespace optiling {}\n";
  AddFallbackHeader(headers, kTilingStateHeaderIdentify, "__AUTOFUSE_TILING_FUNC_STATE_H__", std::move(state));

  autofuse::GeneratedCode log;
  log.body =
      "#define OP_LOGD(name, fmt, ...)\n#define OP_LOGI(name, fmt, ...)\n"
      "#define OP_LOGW(name, fmt, ...)\n#define OP_LOGE(name, fmt, ...)\n#define OP_NAME \"Autofuse\"\n";
  AddFallbackHeader(headers, kTilingLogHeaderIdentify, "__AUTOFUSE_TILING_FUNC_LOG_H__", std::move(log));

  autofuse::GeneratedCode solver;
  for (const auto &header : {"cmath", "cstdint", "type_traits"}) {
    autofuse::RequireSystemHeader(solver.dependencies, header);
  }
  solver.body = GetFallbackSolverMacros() + GetFallbackSolverFunctions();
  AddFallbackHeader(headers, kTilingSolverHeaderIdentify, "__AUTOFUSE_TILING_FUNC_SOLVER_H__", std::move(solver));

  autofuse::GeneratedCode api;
  autofuse::RequireSystemHeader(api.dependencies, "cstdint");
  if (!pgo_body.empty()) {
    autofuse::RequireSystemHeader(api.dependencies, "unordered_map");
    autofuse::RequireSystemHeader(api.dependencies, "vector");
  }
  api.body =
      "struct AutofuseTilingData;\nstruct AutofuseTilingDataPerf;\nstruct PgoTensorArgs;\n"
      "namespace optiling {\nstruct SearchConfig;\n" +
      api_body + "}  // namespace optiling\n";
  AddFallbackHeader(headers, kTilingApiHeaderIdentify, "__AUTOFUSE_TILING_FUNC_API_H__", std::move(api));
  if (!pgo_body.empty()) {
    autofuse::GeneratedCode pgo;
    for (const auto &header : {"array", "cstddef", "cstdint", "vector"}) {
      autofuse::RequireSystemHeader(pgo.dependencies, header);
    }
    pgo.body = "struct AutofuseTilingData;\nstruct AutofuseTilingDataPerf;\n" + pgo_body;
    AddFallbackHeader(headers, kTilingPgoHeaderIdentify, "__AUTOFUSE_TILING_FUNC_PGO_H__", std::move(pgo));
  }
}

void AddCvDeclarationsToApiHeader(std::map<std::string, std::string> &headers) {
  auto iter = headers.find(kTilingApiHeaderIdentify);
  if (iter == headers.end()) {
    return;
  }
  const auto guard_end = iter->second.rfind("#endif");
  if (guard_end == std::string::npos) {
    return;
  }
  iter->second.insert(guard_end,
                      "int32_t get_g_basen_basem_align();\nvoid set_g_basen_basem_align(int32_t value);\n\n");
}

void GenMulGroupFindBestTilingKey(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) {
  uint64_t tiling_key_offset = 0U;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    for (size_t i = 0; i < scheduled_results.size(); i++) {
      const auto &schedule_groups = scheduled_results[i].schedule_groups;
      ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
         << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
      const bool has_empty_group = std::any_of(schedule_groups.begin(), schedule_groups.end(),
                                               [](const auto &group) { return group.impl_graphs.empty(); });
      uint64_t result_tiling_key_count = 0U;
      if (has_empty_group) {
        ss << "    return -1;" << std::endl;
      } else {
        ss << "    int64_t local_tiling_key = 0;" << std::endl;
        result_tiling_key_count = 1U;
        for (size_t j = 0; j < schedule_groups.size(); j++) {
          const size_t impl_count = schedule_groups[j].impl_graphs.size();
          const std::string field_name =
              CamelToLowerSneak("t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                std::to_string(j) + "_tiling_data");
          ss << "    if (" << field_name << ".tiling_key >= " << impl_count << ") {" << std::endl;
          ss << "      return -1;" << std::endl;
          ss << "    }" << std::endl;
          ss << "    local_tiling_key = local_tiling_key * " << impl_count << " + " << field_name << ".tiling_key;"
             << std::endl;
          result_tiling_key_count *= impl_count;
        }
        ss << "    return " << tiling_key_offset << " + local_tiling_key;" << std::endl;
      }
      tiling_key_offset += result_tiling_key_count;
      ss << "  }";
    }
  }
  ss << std::endl;
}

uint64_t CalcTilingKeyCount(const ascir::FusedScheduledResult &result) {
  if (!ascgen_utils::CanUseTilingKey(result)) {
    return 1ULL;
  }
  uint64_t count = 0U;
  return TryCalcTilingKeyCount(result, std::numeric_limits<uint64_t>::max(), count)
             ? count
             : std::numeric_limits<uint64_t>::max();
}

bool HasWorkSpaceNode(const af::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->GetType() == "Workspace") {
      return true;
    }
  }
  return false;
}

void CodegenTilingKeyKerneType(std::stringstream &ss, const std::vector<std::vector<bool>> &per_group_conditions,
                               std::vector<bool> &current, uint32_t depth, uint32_t &tiling_key) {
  if (per_group_conditions.size() == depth) {
    bool has_workspace_node = false;
    for (const auto &workspace_node : current) {
      if (workspace_node) {
        has_workspace_node = true;
        break;
      }
    }
    std::string kernel_type = (has_workspace_node ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly);
    ss << "    {" << std::to_string(tiling_key) << ",\"" << kernel_type << "\"}," << std::endl;
    tiling_key++;
    return;
  }
  for (const auto &condition : per_group_conditions[depth]) {
    current.push_back(condition);
    CodegenTilingKeyKerneType(ss, per_group_conditions, current, depth + 1, tiling_key);
    current.pop_back();
  }
}

void CollectWorkspaceDenominatorSymbols(const af::Expression &expr, std::set<std::string> &denominator_symbols) {
  if (!expr.IsValid() || expr.IsConstExpr()) {
    return;
  }
  af::Expression numer;
  af::Expression denom;
  expr.AsNumerDenom(numer, denom);
  if (denom.IsValid() && !denom.IsConstExpr()) {
    for (const auto &symbol : denom.FreeSymbols()) {
      if (symbol.GetExprType() == af::ExprType::kExprVariable) {
        denominator_symbols.insert(af::SymbolicUtils::ToString(symbol));
      }
    }
  }
  af::Expression expr_copy = expr;
  for (const auto &arg : expr_copy.GetArgs()) {
    CollectWorkspaceDenominatorSymbols(arg, denominator_symbols);
  }
}

void GenWorkspaceDenominatorGuards(std::stringstream &ss, const af::Expression &expr, const std::string &indent) {
  std::set<std::string> denominator_symbols;
  CollectWorkspaceDenominatorSymbols(expr, denominator_symbols);
  for (const auto &symbol : denominator_symbols) {
    ss << indent << "if (" << symbol << " <= 0) {" << std::endl;
    ss << indent << "  OP_LOGW(OP_NAME, \"Invalid workspace denominator " << symbol << "=%lf.\", static_cast<double>("
       << symbol << "));" << std::endl;
    ss << indent << "  return false;" << std::endl;
    ss << indent << "}" << std::endl;
  }
}
}  // namespace

TilingLib::TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name) {
  af::GetContext().Init();
  auto ret = att::AutoFuseConfig::MutablePgoStrategyConfig().Init();
  if (ret == af::SUCCESS || ret == af::NOT_CHANGED) {
    if (att::AutoFuseConfig::GetPgoStrategyConfig().set_env_enable_autofuse_pgo) {
      enable_autofuse_pgo_ = (att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo == "true");
    }
  } else {
    GELOGE(af::FAILED, "TilingLib function ENV init failed");
    return;
  }
  GELOGI("TilingLib lib_path:%s, symbol_name:%s", lib_path.c_str(), codegen_symbol_name.c_str());
  if (lib_path.empty() || codegen_symbol_name.empty()) {
    GELOGI("TilingLib using default att api: GenTilingImplAutoFuseV3");
    this->codegen_func_ = att::GenTilingImplAutoFuseV3;
    return;
  }

  this->codegen_func_ = nullptr;
  std::string real_lib_path;
  if (!ascgen_utils::GetRealPath(lib_path, real_lib_path)) {
    GELOGE(af::FAILED, "lib_path::%s realpath failed", lib_path.c_str());
    return;
  }
  auto handle = dlopen(real_lib_path.c_str(), RTLD_LAZY);
  GE_CHK_BOOL_EXEC(handle != nullptr, return, "TilingLib lib dlopen fail lib_path:%s", real_lib_path.c_str());

  auto func = dlsym(handle, codegen_symbol_name.c_str());
  if (func == nullptr) {
    GELOGE(af::FAILED, "TilingLib function dlsym fail symbol_name:%s", codegen_symbol_name.c_str());
    dlclose(handle);
    return;
  }

  this->codegen_func_ = reinterpret_cast<TilingLibCodegenFunc>(func);
}

bool TilingLib::ShouldFallbackPgo(const ascir::FusedScheduledResult &fused_schedule_result) const {
  uint64_t count = 0U;
  return enable_autofuse_pgo_ && !TryCalcTilingKeyCount(fused_schedule_result, kMaxPgoTilingKeyCount, count);
}

std::map<std::string, std::string> TilingLib::GenerateForInductor(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  ascir::FusedScheduledResult elemwise_schedule_result = fused_schedule_result;
  const bool is_cube_fused_scheduled = ascgen_utils::IsCubeFusedScheduled(fused_schedule_result);
  if (enable_autofuse_pgo_ && !IsSupportedInductorPgoScene(fused_schedule_result)) {
    GELOGE(af::FAILED, "Inductor MSPTI PGO only supports static, non-CV kernels");
    return {{kTilingDefAndConstIdentify, ascgen_utils::INVALID_TILING}};
  }
  if (is_cube_fused_scheduled) {
    GE_ASSERT_SUCCESS(ascgen_utils::ProcessCubeFusionResultDynamic(elemwise_schedule_result));
  }
  std::map<std::string, std::string> tiling_file_name_to_content =
      GetTilingHeaders(elemwise_schedule_result, true, is_cube_fused_scheduled);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);
  std::stringstream ss;

  ss << "#pragma GCC diagnostic push\n" << "#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"\n";
  ss << "extern \"C\" std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data);\n";
  ss << "#pragma GCC diagnostic pop\n";
  ss << TilingFuncDefForInductor(fused_schedule_result, elemwise_schedule_result) << std::endl;
  if (!is_cube_fused_scheduled) {
    GenInductorTopnSources(elemwise_schedule_result, ss, tiling_file_name_to_content);
  }
  // 生成GenConstTilingData方法（对所有场景生成，包括CV fusion静态shape）
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result) << std::endl;
  if (is_cube_fused_scheduled) {
    tiling_file_name_to_content[kCubeKernelTilingWrapperHpp] = kCubeKernelTilingWrapperHppValue;
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] = kCubeKernelTilingWrapperInclude;
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] += kCubeKernelTilingWrapperCppValue;
  }
  const std::string entry_body = tiling_file_name_to_content[kTilingDefAndConstIdentify] + ss.str();
  const auto entry_options =
      GetInductorEntryTranslationUnitOptions(elemwise_schedule_result, is_cube_fused_scheduled, enable_autofuse_pgo_);
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);

  return tiling_file_name_to_content;
}

bool TilingLib::IsSupportedInductorPgoScene(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return !ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && IsStaticSchedResult(fused_schedule_result);
}

void TilingLib::GenPgoMixTilingTable(const ascir::FusedScheduledResult &fused_schedule_result,
                                     std::stringstream &ss) const {
  for (size_t graph_id = 0U; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    const auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    ss << "std::vector<uint32_t> g_mix_graph" << graph_id << "_tiling_keys = {" << std::endl;
    for (size_t result_id = 0U; result_id < scheduled_results.size(); result_id++) {
      const auto &schedule_groups = scheduled_results[result_id].schedule_groups;
      bool has_workspace_node = false;
      for (size_t group_id = 0U; group_id < schedule_groups.size() - 1U; group_id++) {
        const auto &impl_graphs = schedule_groups[group_id].impl_graphs;
        has_workspace_node = std::any_of(impl_graphs.begin(), impl_graphs.end(),
                                         [](const auto &graph) { return HasWorkSpaceNode(graph); });
      }
      if (has_workspace_node) {
        ss << "    " << result_id << "," << std::endl;
      }
    }
    ss << "};" << std::endl;
  }
}

std::map<std::string, std::string> TilingLib::GenerateCVFusion(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::map<std::string, std::string> &shape_info,
                                                               const std::string &pgo_dir,
                                                               const std::string &core_num) const {
  std::map<std::string, std::string> tiling_file_name_to_content;
  ascir::FusedScheduledResult elemwise_schedule_result = fused_schedule_result;
  bool is_static = IsStaticSchedResult(elemwise_schedule_result);
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !is_static) {
    GE_ASSERT_SUCCESS(ascgen_utils::ProcessCubeFusionResultDynamic(elemwise_schedule_result));
  } else {
    if (ascgen_utils::IsCubeUBFusedScheduled(elemwise_schedule_result)) {
      GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionResult(elemwise_schedule_result));
    } else if (ascgen_utils::IsCubeCommonFusedScheduled(elemwise_schedule_result)) {
      GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionCommonResult(elemwise_schedule_result));
    }
  }
  tiling_file_name_to_content = GetTilingHeaders(elemwise_schedule_result, false, true);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);

  std::map<std::string, std::string> result;
  if (is_static) {
    result = GenerateCVFusionStatic(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num);
  } else {
    result = GenerateCVFusionDynamic(fused_schedule_result, elemwise_schedule_result, shape_info, pgo_dir, core_num);
  }

  const std::string entry_body =
      tiling_file_name_to_content[kTilingDefAndConstIdentify] + result[kTilingDefAndConstIdentify];
  const EntryTranslationUnitOptions entry_options = {
      false,
      true,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_ || EntryWorkspaceUsesSolver(elemwise_schedule_result),
      !ascgen_utils::IsSingleGroup(elemwise_schedule_result),
      !is_static};
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);
  if (!is_static) {
    tiling_file_name_to_content[kCubeKernelTilingWrapperHpp] = result[kCubeKernelTilingWrapperHpp];
    tiling_file_name_to_content[kCubeKernelTilingWrapperCpp] = result[kCubeKernelTilingWrapperCpp];
  }
  return tiling_file_name_to_content;
}

std::map<std::string, std::string> TilingLib::Generate(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const std::map<std::string, std::string> &shape_info,
                                                       const std::string &pgo_dir, const std::string &core_num) const {
  if (ShouldFallbackPgo(fused_schedule_result)) {
    GELOGW("Tiling key count exceeds 10000, fallback to non-PGO codegen");
    TilingLib fallback(*this);
    fallback.DisableInductorPgo();
    return fallback.Generate(fused_schedule_result, shape_info, pgo_dir, core_num);
  }
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) &&
      !ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    return GenerateCVFusion(fused_schedule_result, shape_info, pgo_dir, core_num);
  }

  std::map<std::string, std::string> tiling_file_name_to_content = GetTilingHeaders(fused_schedule_result, false);
  GE_CHK_BOOL_RET_STATUS_NOLOG(CheckTilingHeadersValid(tiling_file_name_to_content), tiling_file_name_to_content);
  std::stringstream ss;
  ss << TilingFuncDef(fused_schedule_result, fused_schedule_result, shape_info, pgo_dir, core_num) << std::endl;
  // 生成GenConstTilingData方法
  ss << TilingData("Autofuse").GenerateConst(fused_schedule_result, false) << std::endl;

  ss << kTilingHeadCceKtTestGuard << std::endl;
  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result) && CanUseTilingKey(fused_schedule_result) &&
      IsStaticSchedResult(fused_schedule_result)) {
    ss << GenGetTilingKeyForStatic();
    ss << GenGetTilingKeyKernelTypeForStatic(fused_schedule_result);
  }
  ss << "#endif" << std::endl;
  const std::string entry_body = tiling_file_name_to_content[kTilingDefAndConstIdentify] + ss.str();
  const EntryTranslationUnitOptions entry_options = {
      false,
      false,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_,
      enable_autofuse_pgo_ || EntryWorkspaceUsesSolver(fused_schedule_result),
      !ascgen_utils::IsSingleGroup(fused_schedule_result)};
  tiling_file_name_to_content[kTilingDefAndConstIdentify] = RenderEntryTranslationUnit(entry_body, entry_options);

  return tiling_file_name_to_content;
}

std::string TilingLib::StubHeadersWithoutCodegenFunc() const {
  std::stringstream ss;
  ss << "#include <iostream>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  ss << "#include <cinttypes>" << std::endl;
  ss << "#include <sys/syscall.h>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <securec.h>" << std::endl;
  ss << "#include \"dlog_pub.h\"" << std::endl;
  ss << "#define OP_LOGD(name, fmt, ...)" << std::endl;
  ss << "#define OP_LOGI(name, fmt, ...)" << std::endl;
  ss << "#define GE_MODULE_NAME static_cast<int32_t>(45)" << std::endl;
  ss << "inline uint64_t GetTid() {" << std::endl;
  ss << "     return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
  ss << "}" << std::endl;

  ss << "#define GELOGE(ERROR_CODE, fmt, ...)" << std::endl;

  ss << "#define OP_LOGE(name, fmt, ...)" << std::endl;
  ss << "#define OP_NAME \"asc0000_autofused_abs\"" << std::endl;
  ss << "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Log(a) (log((double)(a)))" << std::endl;
  ss << "#define Pow(a, b) pow(a, b)" << std::endl;
  ss << "#define Rational(a, b) ((double)(a) / (double)(b))" << std::endl;
  ss << "" << std::endl;

  return ss.str();
}

std::string TilingLib::GetStubTilingHeaders(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << StubHeadersWithoutCodegenFunc();
  ss << "namespace optiling {" << std::endl;
  ss << "extern \"C\" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1, double *perf=nullptr) {"
     << std::endl;
  ss << "  (void)perf;" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  if (enable_autofuse_pgo_) {
    ss << "struct SearchConfig;" << std::endl;
    ss << "bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, "
       << "AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* output_tiling_data, "
       << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
       << "void* stream, uint64_t workspaceSize, double& out_best_perf, "
       << "std::unordered_map<int64_t, uint64_t> &workspace_map, "
       << "std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {" << std::endl;
    ss << "  return true;" << std::endl;
    ss << "}" << std::endl;
    ss << "bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, "
       << "AutofuseTilingData* tiling_data, uint32_t max_block_dim=48) {" << std::endl;
    ss << "  return true;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;
  ss << std::endl;
  return ss.str();
}

std::string TilingLib::GetStubTilingApi(const ascir::FusedScheduledResult &fused_schedule_result,
                                        bool include_pgo) const {
  std::stringstream ss;
  ss << "extern \"C\" inline bool GetTiling(AutofuseTilingData &tiling_data, int32_t tiling_case_id = -1, "
        "double *perf = nullptr) {\n";
  ss << "  (void)tiling_data; (void)tiling_case_id; (void)perf; return true;\n}\n";
  if (!include_pgo) {
    return ss.str();
  }
  const std::string common_params =
      "std::vector<AutofuseTilingDataPerf> &tiling_data_list, AutofuseTilingData &tiling_data, "
      "int32_t tiling_case_id, AutofuseTilingData *output_tiling_data, " +
      PGOSearchFuncInputOutputCallBackDef(fused_schedule_result) +
      "void *stream, uint64_t workspace_size, double &out_best_perf";
  ss << "inline bool PGOSearchTilingKey(" << common_params
     << ", std::unordered_map<int64_t, uint64_t> &workspace_map, "
        "std::vector<uint32_t *> block_dim_vec = {}, const SearchConfig *search_cfg = nullptr) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)tiling_case_id; (void)output_tiling_data;\n"
        "  (void)tensor_args; (void)stream; (void)workspace_size; (void)out_best_perf; (void)workspace_map;\n"
        "  (void)block_dim_vec; (void)search_cfg; return true;\n}\n";
  ss << "inline bool PGOSearchTilingKey(" << common_params << ", const SearchConfig *search_cfg = nullptr) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)tiling_case_id; (void)output_tiling_data;\n"
        "  (void)tensor_args; (void)stream; (void)workspace_size; (void)out_best_perf; (void)search_cfg; return true;\n"
        "}\n";
  ss << "inline bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData> &tiling_data_list, "
        "AutofuseTilingData *tiling_data, uint32_t max_block_dim = 48) {\n";
  ss << "  (void)tiling_data_list; (void)tiling_data; (void)max_block_dim; return true;\n}\n";
  return ss.str();
}

std::string TilingLib::GetTilingIncludeHead(bool is_cv) const {
  std::stringstream ss;
  ss << "#ifndef __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#define __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  ss << "#include <stdexcept>" << std::endl;
  ss << "#include <sstream>" << std::endl;
  ss << "#include <cmath>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include \"autofuse_tiling_data.h\"" << std::endl;
  if (is_cv) {
    ss << "int32_t get_g_basen_basem_align();" << std::endl;
    ss << "void set_g_basen_basem_align(int32_t value);" << std::endl;
  }
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << "#include \"exe_graph/runtime/infer_shape_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/kernel_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/continuous_vector.h\"" << std::endl;
  ss << "#include \"platform/platform_infos_def.h\"" << std::endl;
  ss << "#include \"platform_ascendc.h\"" << std::endl;
  ss << "#include \"acl/acl.h\"" << std::endl;

  return ss.str();
}

void TilingLib::PopulateFallbackAtomicHeaders(std::map<std::string, std::string> &tiling_file_name_to_content,
                                              const ascir::FusedScheduledResult &fused_schedule_result,
                                              bool use_att_codegen, bool include_pgo) const {
  std::string fallback_pgo_body;
  std::string fallback_api_body;
  if (!use_att_codegen) {
    fallback_api_body = GetStubTilingApi(fused_schedule_result, include_pgo);
    if (include_pgo) {
      fallback_pgo_body = PGOProfilingCallbackDef(fused_schedule_result, "AutofuseTilingData", false);
      fallback_pgo_body +=
          "namespace optiling {\nstruct SearchConfig {\n"
          "  bool ub_threshold_enabled = true;\n  double ub_threshold = 0.0;\n"
          "  bool corenum_threshold_enabled = true;\n  double corenum_threshold = 1.0;\n"
          "  bool enable_multicore_ub_tradeoff = true;\n};\n}  // namespace optiling\n";
    }
  }
  EnsureFallbackAtomicHeaders(tiling_file_name_to_content, fallback_pgo_body, fallback_api_body);
}

std::map<std::string, std::string> TilingLib::GetTilingHeaders(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               bool is_inductor_scene, bool is_cv) const {
  std::stringstream ss;
  std::string graph_name = GenValidName(fused_schedule_result.fused_graph_name.GetString());
  ss << GetTilingIncludeHead(is_cv);
  ss << "#endif" << std::endl;
  ss << std::endl;

  std::map<std::string, std::string> tiling_file_name_to_content;
  std::string tiling_name = "AutofuseTilingData";

  // just cube kernel skip GetTiling
  if (ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    EnsureFallbackAtomicHeaders(tiling_file_name_to_content, "", GetStubTilingApi(fused_schedule_result, false));
    if (is_cv) {
      AddCvDeclarationsToApiHeader(tiling_file_name_to_content);
    }
    return tiling_file_name_to_content;
  }

  const bool use_att_codegen = this->codegen_func_ != nullptr && !IsEmptyTensorSence(fused_schedule_result);
  if ((enable_autofuse_pgo_ || is_inductor_scene) && !use_att_codegen) {
    ss << PGOProfilingCallbackDef(fused_schedule_result, tiling_name);
  }
  if (use_att_codegen) {
    std::map<std::string, std::string> options;
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
    options.emplace("tiling_data_type_name", tiling_name);
    options.emplace("solver_type", "AxesReorder");
    if (is_inductor_scene) {
      options.emplace(att::kInternalEnableAutofusePgo, enable_autofuse_pgo_ ? "true" : "false");
    }
    GE_CHK_BOOL_EXEC(
        this->codegen_func_(fused_schedule_result.fused_graph_name.GetString(), fused_schedule_result, options,
                            tiling_file_name_to_content, is_inductor_scene),
        GELOGE(af::FAILED, "Codegen Gen tiling func failed, graph:%s", graph_name.c_str());
        tiling_file_name_to_content[kTilingHeadIdentify] += "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__\n";
        tiling_file_name_to_content[kTilingDefAndConstIdentify] = INVALID_TILING; return tiling_file_name_to_content);
  } else {
    GELOGI("TilingLib generate stub GetTiling func start");
    ss << GetStubTilingHeaders(fused_schedule_result);
    tiling_file_name_to_content[kTilingHeadIdentify] += ss.str();
  }
  std::stringstream ss_end;
  ss_end << "#endif // __AUTOFUSE_TILING_FUNC_COMMON_H__" << std::endl;
  tiling_file_name_to_content[kTilingHeadIdentify] += ss_end.str();
  const bool include_pgo = enable_autofuse_pgo_ || is_inductor_scene;
  PopulateFallbackAtomicHeaders(tiling_file_name_to_content, fused_schedule_result, use_att_codegen, include_pgo);
  if (is_cv) {
    AddCvDeclarationsToApiHeader(tiling_file_name_to_content);
  }
  return tiling_file_name_to_content;
}

std::string TilingLib::TilingFuncDefForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const ::ascir::FusedScheduledResult &elemwise_schedule_result) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(elemwise_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, "CVAutofuseTilingData", true) << std::endl;
  } else {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, tiling_data_name, true) << std::endl;
  }
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, elemwise_schedule_result) << std::endl;
  ss << this->GenTilingFuncForInductor(fused_schedule_result, elemwise_schedule_result, tiling_func_name,
                                       tiling_data_name)
     << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  ss << this->ExternFunctionDeclare(elemwise_schedule_result, tiling_data_name) << std::endl;
  ss << "#endif" << std::endl;

  return ss.str();
}

std::string TilingLib::TilingFuncDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                     const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                     const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
                                     const std::string &core_num) const {
  std::stringstream ss;
  std::string graph_name = ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString());
  std::string tiling_func_name = "AutofuseTiling";
  std::string tiling_data_name = "AutofuseTilingData";

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !IsStaticSchedResult(fused_schedule_result)) {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, "CVAutofuseTilingData") << std::endl;
  } else {
    ss << this->GenGetTilingSizeFunc(fused_schedule_result, graph_name, tiling_data_name) << std::endl;
  }
  ss << this->GenGetWorkspaceSizeFunc(tiling_data_name, elemwise_schedule_result) << std::endl;
  ss << this->GenTilingFunc(shape_info, elemwise_schedule_result, tiling_func_name, tiling_data_name, core_num)
     << std::endl;
  ss << kTilingHeadCceKtTestGuard << std::endl;
  // 生成判断是否为静态shape的接口
  bool is_static = IsFrontendStaticSchedResult(elemwise_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  if (ascgen_utils::CanUseTilingKey(elemwise_schedule_result)) {
    ss << this->GenFindBestTilingKeyFunc(elemwise_schedule_result, tiling_data_name);
  }
  if (enable_autofuse_pgo_) {
    ss << GenGetTilingKeyCount(elemwise_schedule_result);
  }
  ss << this->GenExternTilingFunc(elemwise_schedule_result, shape_info, tiling_data_name, pgo_dir, core_num)
     << std::endl;
  ss << this->GenTilingCacheFunc(elemwise_schedule_result, shape_info);
  ss << this->GenDfxInputSymbolInfo(elemwise_schedule_result, shape_info);
  ss << "#endif" << std::endl;

  return ss.str();
}

void TilingLib::TilingProcessSymbolToTiling(const ascir::ImplGraph &graph, size_t graph_num, size_t res_num,
                                            size_t group_num,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (auto size : graph.GetAllSizeVar()) {
    if (size->expr.IsConstExpr()) {
      continue;
    }
    std::string ori_sym = af::SymbolicUtils::ToString(size->expr);
    std::string tiling_var = "t.graph" + std::to_string(graph_num) + "_result" + std::to_string(res_num) + "_g" +
                             std::to_string(group_num) + "_tiling_data";
    ori_sym_tiling_map[ori_sym] = tiling_var;
    GELOGD("TilingProcessSymbolToTiling make tiling var set [%s:%s]", ori_sym.c_str(), tiling_var.c_str());
  }
}

void TilingLib::TilingMappingSymbolToTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((scheduled_results.size() == 0) ||
        ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1))) {
      ori_sym_tiling_map.clear();
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          for (auto graph : scheduled_results[j].schedule_groups[k].impl_graphs) {
            TilingProcessSymbolToTiling(graph, i, j, k, ori_sym_tiling_map);
          }
        }
      }
    }
  }
}

std::string TilingLib::GenImplGraphWorkspaceSize(const ascir::ImplGraph &graph, const std::string &tiling_data,
                                                 uint32_t index) const {
  std::stringstream ss;
  std::vector<af::AscNodePtr> ws_nodes;
  af::Expression ws_size = af::Symbol(0);

  for (const auto &node : graph.GetAllNodes()) {
    if (IsOps<Workspace>(node)) {
      ws_nodes.push_back(node);
    }
  }

  ss << (index == 0U ? "    if (" : " else if(") << tiling_data << ".tiling_key == " << std::to_string(index) << ") {"
     << std::endl;
  ws_size = ascgen_utils::CalculateWorkspaceSize(ws_nodes);
  std::vector<af::Expression> ori_symbols = ws_size.FreeSymbols();
  std::vector<std::pair<af::Expression, std::string>> symbol_tiling_vars;
  for (auto &ori : ori_symbols) {
    if (!(ori.IsConstExpr())) {
      const std::string tiling_var = tiling_data + "." + af::SymbolicUtils::ToString(ori);
      GELOGD("GenImplGraphWorkspaceSize make tiling var set[%s:%s]", af::SymbolicUtils::ToString(ori).c_str(),
             tiling_var.c_str());
      symbol_tiling_vars.emplace_back(ori, tiling_var);
    }
  }

  std::vector<std::pair<af::Expression, af::Expression>> sizes;
  for (size_t i = 0UL; i < symbol_tiling_vars.size(); i++) {
    const auto &symbol_tiling_var = symbol_tiling_vars[i];
    const std::string local_var = "workspace_size_symbol_" + std::to_string(i);
    // Workspace formulas multiply uint32_t tiling fields. Promote them before Max and arithmetic to avoid
    // intermediate uint32_t overflow.
    ss << "      const uint64_t " << local_var << " = static_cast<uint64_t>(" << symbol_tiling_var.second << ");"
       << std::endl;
    sizes.emplace_back(symbol_tiling_var.first, af::Symbol(local_var.c_str()));
  }

  af::Expression replaced_ws_size = ws_size.Replace(sizes);
  std::string ws_size_str = af::SymbolicUtils::ToString(replaced_ws_size);

  GenWorkspaceDenominatorGuards(ss, replaced_ws_size, "      ");
  ss << "      if (!CheckedWorkspaceAdd(ws_size, " << ws_size_str << ", ws_size)) {" << std::endl;
  ss << "        OP_LOGE(OP_NAME, \"Workspace size add overflow.\");" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetWorkspaceSizeFunc(const std::string &tiling,
                                               const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;

  std::unordered_map<std::string, std::string> ori_sym_tiling_map;
  TilingMappingSymbolToTiling(fused_schedule_result, ori_sym_tiling_map);

  ss << GenWorkspaceCheckedHelpers();
  ss << "bool GetWorkspaceSize(const " << tiling << " &t, uint64_t &workspace_size) {" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  uint64_t ws_size = 0;" << std::endl;
  for (size_t graph_id = 0; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[graph_id];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) && (scheduled_results.size() == 1) &&
        (scheduled_results[0].schedule_groups.size() == 1)) {
      auto schedule_graphs = scheduled_results[0].schedule_groups[0].impl_graphs;
      for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
        ss << GenImplGraphWorkspaceSize(schedule_graphs[i], "t", i);
      }
    } else {
      for (uint32_t i = 0; i < scheduled_results.size(); i++) {
        auto schedule_groups = scheduled_results[i].schedule_groups;
        ss << (i == 0 ? "  if " : "  else if ") << "(t." << "graph" << std::to_string(graph_id)
           << "_tiling_key == " << std::to_string(i) << ") {" << std::endl;
        for (uint32_t j = 0; j < schedule_groups.size(); j++) {
          auto schedule_graphs = schedule_groups[j].impl_graphs;
          for (uint32_t k = 0; k < schedule_graphs.size(); k++) {
            std::string filed_name = "t.graph" + std::to_string(graph_id) + "_result" + std::to_string(i) + "_g" +
                                     std::to_string(j) + "_tiling_data";
            ss << GenImplGraphWorkspaceSize(schedule_graphs[k], filed_name, k);
          }
        }
        ss << "  }";
      }
    }
  }

  ss << std::endl;
  ss << "  if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Workspace size align overflow.\");" << std::endl;
  ss << "    return false;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  workspace_size = ws_size;" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

bool TilingLib::IsVarUsedInScheduleGroup(const std::string &var_define,
                                         const ::ascir::ScheduleGroup &schedule_group) const {
  SizeVarSet used_vars;
  for (const auto &impl_graph : schedule_group.impl_graphs) {
    AscGraphInfoComplete::AppendOriginalSizeVar(impl_graph, used_vars);
  }

  // 检查 var_define 是否在 used_vars 中
  for (const auto &var : used_vars) {
    if (auto var_str = var.Str()) {
      if (std::string(var_str.get()) == var_define) {
        return true;
      }
    }
  }
  return false;
}

void TilingLib::TilingSetShapeDim(std::stringstream &tiling_set_shape_dim, const std::string &var_define,
                                  const ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::string &tiling_expr) const {
  for (size_t i = 0; i < fused_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[i];
    if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1) &&
        (scheduled_results.empty() ||
         ((scheduled_results.size() == 1) && (scheduled_results[0].schedule_groups.size() == 1)))) {
      // 检查变量是否被此 schedule_group 使用
      if (!IsVarUsedInScheduleGroup(var_define, scheduled_results[0].schedule_groups[0])) {
        continue;
      }
      // 简单情况：直接设置（保持原逻辑）
      tiling_set_shape_dim << "  " << tiling_expr << "set_" << var_define << "(" << var_define << ");" << std::endl;
    } else {
      for (size_t j = 0; j < scheduled_results.size(); j++) {
        for (size_t k = 0; k < scheduled_results[j].schedule_groups.size(); k++) {
          // 新增：检查变量是否被此 schedule_group 使用
          if (!IsVarUsedInScheduleGroup(var_define, scheduled_results[j].schedule_groups[k])) {
            continue;
          }
          // 原有的 var_relations 检查
          if (scheduled_results[j].var_relations.find(k) != scheduled_results[j].var_relations.end()) {
            continue;
          }
          tiling_set_shape_dim << "  " << tiling_expr << "graph" << i << "_result" << j << "_g" << k
                               << "_tiling_data.set_" << var_define << "(" << var_define << ");" << std::endl;
        }
      }
    }
  }
}

std::string TilingLib::GenFp32LargeKCondition(const MatMulCubeInfo &cube_info) const {
  // 非fp32类型(type_size!=4)不触发common兜底
  if (cube_info.type_size != 4U) {
    return "false";
  }
  // 启用hf32时不触发common兜底
  if (cube_info.enable_hf32) {
    return "false";
  }
  // 从matmul节点输入0获取K轴表达式
  std::vector<TensorInfo> inputs;
  if (ExtractInputsFromMatMulNode(cube_info.matmul_node, inputs) != af::SUCCESS || inputs.empty() ||
      inputs[0].shape.size() < 2U) {
    return "false";
  }
  // 非transpose_x1: K轴是shape[-1]; transpose_x1: K轴是shape[-2]
  const auto &shape = inputs[0].shape;
  const af::Expression &k_expr = cube_info.transpose_x1 ? shape[shape.size() - 2U] : shape[shape.size() - 1U];
  if (!k_expr.IsValid()) {
    return "false";
  }
  // K轴为常量: 编译期判断
  if (k_expr.IsConstExpr()) {
    std::string k_str = std::string(k_expr.Str().get());
    try {
      int64_t k_value = std::stoll(k_str);
      return k_value >= kFp32LargeKThreshold ? "true" : "false";
    } catch (...) {
      return "false";
    }
  }
  // K轴为动态变量: 生成运行时判断
  std::string k_var = std::string(k_expr.Str().get());
  return "(static_cast<int64_t>(" + k_var + ") >= " + std::to_string(kFp32LargeKThreshold) + ")";
}

std::string TilingLib::GenCubeFusionTilingBodyInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                       const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                                       const std::string &shape_dim_param) const {
  std::stringstream ss;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  uint64_t count = CalcTilingKeyCount(elemwise_schedule_result);
  ss << "  int64_t ws_size = 0;" << std::endl;
  ss << "  int64_t cube_tiling_key = 0;" << std::endl;
  ss << "  uint32_t cube_block_dim = 0;" << std::endl;
  ss << "  uint32_t basem = 0;" << std::endl;
  ss << "  uint32_t basen = 0;" << std::endl;
  ss << "  CallCubeTiling(" << shape_dim_param << "ws_size, cube_block_dim, cube_tiling_key, basem, basen, tiling);"
     << std::endl;
  ss << "  tiling->cube_tiling_key = cube_tiling_key;" << std::endl;
  ss << "  int64_t cube_tiling_key_ub = cube_tiling_key & ~0xF0;" << std::endl;
  // 32元素对齐: 最坏dtype(bool)的32B行对齐行宽也是basen_align, 保证按行宽申请的
  // queue tensor在任何dtype下都不超过求解器按STAGE_SIZE_NAME给出的UB元素预算
  ss << "  const int32_t ub_align_value = 32;" << std::endl;
  ss << "  const int32_t basen_align = (basen + ub_align_value - 1) / ub_align_value * ub_align_value;" << std::endl;
  ss << "  const int32_t basen_basem_align = (basem * basen_align) / 2 + basen_align;" << std::endl;
  ss << "  set_g_basen_basem_align(basen_basem_align);" << std::endl;
  ss << "  tiling->cube_ub_stage_size = (uint32_t)basen_basem_align;" << std::endl;
  // 兜底默认值(容量行数); 求解成功后在GenCvTilingDataReset中按预算行数覆盖
  ss << "  tiling->cube_m_stage_size = (uint32_t)(basen_basem_align / basen_align);" << std::endl;
  ss << "  tiling->tiling_data.set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->tiling_data.set_ub_size(limit->ub_size - 256);" << std::endl;

  // fp32且K轴大于阈值且未启用hf32时, CV融合UB模板有精度问题, 强制走common兜底
  std::string fp32_large_k_cond = GenFp32LargeKCondition(cube_info);
  ss << "  if (cube_tiling_key_ub != 1 || (" << fp32_large_k_cond << ")) {" << std::endl;
  GenInductorCvSafetyFallback(ss, count, "    ");
  ss << "  }" << std::endl;

  GenInductorGetTilingBranch(ss, count, cube_info.type_size);
  ss << "  *blockDim = cube_block_dim;" << std::endl;
  ss << "  uint64_t vector_workspace_size = 0U;" << std::endl;
  ss << "  if (!GetWorkspaceSize(tiling->tiling_data, vector_workspace_size) ||" << std::endl;
  ss << "      !CheckedWorkspaceAdd(vector_workspace_size, static_cast<uint64_t>(ws_size), *workspaceSize)) {"
     << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

void TilingLib::GenInductorShapeDim(const ascir::FusedScheduledResult &elemwise_schedule_result,
                                    codegen::PgoShapeStringStream &pgo_shape_dim,
                                    std::vector<std::string> &dynamic_shape_vars, const std::string &tiling_var) const {
  for (auto vars : GetFrontendShapeVars(elemwise_schedule_result)) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      dynamic_shape_vars.push_back(var_define);
      pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
      pgo_shape_dim.shape_dim_use << var_define << ", ";
      TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, elemwise_schedule_result, tiling_var);
    }
  }
}

std::string TilingLib::GenCallCubeTilingForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                    const std::vector<std::string> &dynamic_shape_vars,
                                                    const codegen::PgoShapeStringStream &pgo_shape_dim) const {
  std::stringstream ss;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  ss << "using namespace ge::autofuse;" << std::endl;
  AppendCvBaseAlignHelperDefs(ss);
  AppendCvSafetyMixModeHelperDefs(ss, cube_info.is_batch);

  // 在CallCubeTiling函数之前定义全局变量（用于静态shape常量生成）
  ss << "// Global variable to store tiling bytes for const generation in static shape\n";
  ss << "std::vector<uint8_t> g_matmul_tiling_bytes;\n\n";
  ss << "extern \"C\" void CallCubeTiling(" << pgo_shape_dim.shape_dim_def.str()
     << "int64_t &ws_size, uint32_t &cube_block_dim, int64_t &tiling_key, uint32_t &basem, uint32_t "
        "&basen, CVAutofuseTilingData *tiling_data) {"
     << std::endl;
  GenCallCubeTilingCacheRead(ss, dynamic_shape_vars);
  ss << ProcessCubeKernelTilingFromFusedResult(fused_schedule_result) << std::endl;
  GenCallCubeTilingCacheWrite(ss, dynamic_shape_vars);
  ss << "}" << std::endl;
  return ss.str();
}

void TilingLib::GenCallCubeTilingCacheRead(std::stringstream &ss,
                                           const std::vector<std::string> &dynamic_shape_vars) const {
  ss << "static bool g_cube_tiling_cache_valid = false;\n";
  for (const auto &var_name : dynamic_shape_vars) {
    ss << "static uint32_t g_cube_tiling_cache_" << var_name << " = 0;\n";
  }
  ss << "static int64_t g_cube_tiling_cache_ws_size = 0;\n";
  ss << "static uint32_t g_cube_tiling_cache_block_dim = 0;\n";
  ss << "static int64_t g_cube_tiling_cache_tiling_key = 0;\n";
  ss << "static uint32_t g_cube_tiling_cache_basem = 0;\n";
  ss << "static uint32_t g_cube_tiling_cache_basen = 0;\n";
  ss << "static uint8_t g_cube_tiling_cache_bytes[sizeof(tiling_data->matmul_tiling_data)] = {};\n";
  ss << "static size_t g_cube_tiling_cache_bytes_size = 0;\n";
  ss << "if (g_cube_tiling_cache_valid";
  for (const auto &var_name : dynamic_shape_vars) {
    ss << " && g_cube_tiling_cache_" << var_name << " == " << var_name;
  }
  ss << ") {\n";
  ss << "  ws_size = g_cube_tiling_cache_ws_size;\n";
  ss << "  cube_block_dim = g_cube_tiling_cache_block_dim;\n";
  ss << "  tiling_key = g_cube_tiling_cache_tiling_key;\n";
  ss << "  basem = g_cube_tiling_cache_basem;\n";
  ss << "  basen = g_cube_tiling_cache_basen;\n";
  ss << "  std::memcpy(tiling_data->matmul_tiling_data, g_cube_tiling_cache_bytes, "
        "g_cube_tiling_cache_bytes_size);\n";
  ss << "  return;\n";
  ss << "}\n";
}

void TilingLib::GenCallCubeTilingCacheWrite(std::stringstream &ss,
                                            const std::vector<std::string> &dynamic_shape_vars) const {
  ss << "g_cube_tiling_cache_valid = true;\n";
  for (const auto &var_name : dynamic_shape_vars) {
    ss << "g_cube_tiling_cache_" << var_name << " = " << var_name << ";\n";
  }
  ss << "g_cube_tiling_cache_ws_size = ws_size;\n";
  ss << "g_cube_tiling_cache_block_dim = cube_block_dim;\n";
  ss << "g_cube_tiling_cache_tiling_key = tiling_key;\n";
  ss << "g_cube_tiling_cache_basem = basem;\n";
  ss << "g_cube_tiling_cache_basen = basen;\n";
  ss << "std::memcpy(g_cube_tiling_cache_bytes, tiling_data->matmul_tiling_data, copy_size);\n";
  ss << "g_cube_tiling_cache_bytes_size = copy_size;\n";
}

std::string TilingLib::GenPlainInductorTilingTail(const ascir::FusedScheduledResult &elemwise_schedule_result,
                                                  codegen::PgoShapeStringStream &pgo_shape_dim,
                                                  const std::string &tiling) const {
  std::stringstream ss;
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(limit->ub_size - 256);" << std::endl;
  ss << "  if (!optiling::GetTiling(*tiling, -1, nullptr)) {return -1;}" << std::endl;
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  using namespace optiling;" << std::endl;
  ss << "  if (!GetWorkspaceSize(*tiling, *workspaceSize)) {" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  if (enable_autofuse_pgo_) {
    // PGOGetTilingKey
    ss << GenPGOGetTilingKey(tiling);
    // AutofuseTilingWithConfig
    ss << GenPgoTilingFunc(elemwise_schedule_result, tiling, pgo_shape_dim, true);
  } else {
    // 生成 AutofuseTilingWithConfig 函数
    ss << GenPgoAutofuseTiling(elemwise_schedule_result, pgo_shape_dim, tiling, true);
  }
  return ss.str();
}

std::string TilingLib::GenTilingFuncForInductor(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                                const std::string func, const std::string tiling) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;
  std::vector<std::string> dynamic_shape_vars;
  std::string tiling_var = "tiling->";
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    tiling_var = "tiling->tiling_data.";
  }
  GenInductorShapeDim(elemwise_schedule_result, pgo_shape_dim, dynamic_shape_vars, tiling_var);

  ss << GenGetResLimitStru();
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << GenCallCubeTilingForInductor(fused_schedule_result, dynamic_shape_vars, pgo_shape_dim);
  }

  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ss << "CVAutofuseTilingData* tiling, uint64_t* workspaceSize, uint32_t *blockDim,";
  } else {
    ss << tiling << "* tiling, uint64_t* workspaceSize, uint32_t *blockDim,";
  }
  ss << " ResLimit *res_limit = nullptr)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit effective_res_limit = GetResLimit(res_limit);" << std::endl;
  ss << " const ResLimit *limit = &effective_res_limit;" << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    return ss.str() + GenCubeFusionTilingBodyInductor(fused_schedule_result, elemwise_schedule_result,
                                                      pgo_shape_dim.shape_dim_use.str());
  }
  ss << GenPlainInductorTilingTail(elemwise_schedule_result, pgo_shape_dim, tiling);
  return ss.str();
}

std::string TilingLib::GenTilingFunc(const std::map<std::string, std::string> &shape_info,
                                     const ascir::FusedScheduledResult &fused_schedule_result, const std::string func,
                                     const std::string tiling, const std::string &core_num) const {
  std::stringstream ss;
  codegen::PgoShapeStringStream pgo_shape_dim;
  std::string tiling_var = "tiling->";
  for (auto vars : GetFrontendShapeVars(fused_schedule_result)) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim参数和tiling set shape dim匹配
        pgo_shape_dim.shape_dim_def << "uint32_t " << var_define << ", ";
        pgo_shape_dim.shape_dim_use << var_define << ", ";
        TilingSetShapeDim(pgo_shape_dim.tiling_set_shape_dim, var_define, fused_schedule_result, tiling_var);
      }
    }
  }
  ss << GenGetResLimitStru();
  // AutofuseTiling
  ss << "extern \"C\" int64_t " << func << "(";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << "* tiling, uint64_t* workspaceSize, uint32_t *blockDim,";
  ss << " uint32_t aiv_num, uint32_t ub_size)" << std::endl;
  ss << "{" << std::endl;

  // Use first input shape pass all size variable value
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(aiv_num);" << std::endl;
  ss << "  tiling->set_ub_size(ub_size);" << std::endl;

  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    ss << "  if (!optiling::GetTiling(*tiling, -1, nullptr)) {" << std::endl;
    ss << "      return -1;" << std::endl;
    ss << "  }" << std::endl;
  }
  ss << "  *blockDim = tiling->get_block_dim();" << std::endl;  // Only consider 48 for now
  ss << "  if (!GetWorkspaceSize(*tiling, *workspaceSize) ||" << std::endl;
  ss << "      !CheckedWorkspaceAdd(*workspaceSize, 16ULL * 1024ULL * 1024ULL, *workspaceSize)) {" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  if (enable_autofuse_pgo_) {
    // PGOGetTilingKey
    ss << GenPGOGetTilingKey(tiling);
    // AutofuseTilingWithConfig
    ss << GenPgoTilingFunc(fused_schedule_result, tiling, pgo_shape_dim, false, core_num);
  } else {
    // 生成 AutofuseTilingWithConfig 函数
    ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, false);
  }
  return ss.str();
}

static void GetTilingParse(std::string &tiling_parse, int &vector_core_num) {
  std::stringstream ss;
  ss << "bool version_is_ASCEND950 = false;" << std::endl;
  ss << "struct AfTilingParseData{" << std::endl;
  ss << " uint32_t aiv_num;" << std::endl;
  ss << " uint64_t ub_size;" << std::endl;
  ss << "};" << std::endl;

  ss << "extern \"C\" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {" << std::endl;
  ss << " auto platform = context->GetPlatFormInfos();" << std::endl;
  ss << " if (platform == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);" << std::endl;
  ss << " uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();" << std::endl;

  ss << " uint32_t aiv_num = 0;" << std::endl;
  ss << " uint64_t ub_size = (184 * 1024);" << std::endl;
  if (vector_core_num == 0) {
    ss << " aiv_num = platform_core_num;" << std::endl;
  } else {
    ss << " aiv_num = std::min(platform_core_num, static_cast<uint32_t>(" << vector_core_num << "));" << std::endl;
  }

  ss << " ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);" << std::endl;

  ss << " auto extend_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << " auto tiling_parse_data_av = extend_context->GetOutput(0);" << std::endl;
  ss << " if (tiling_parse_data_av == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];" << std::endl;
  ss << " if (tiling_parse_data_ptr == nullptr) {" << std::endl;
  ss << " return ge::GRAPH_FAILED;" << std::endl;
  ss << " }" << std::endl;
  ss << " tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);" << std::endl;

  ss << " auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);" << std::endl;
  ss << " (*tiling_parse_data)->aiv_num = aiv_num;" << std::endl;
  // 当前A5获取ubsize没减256，和静态编译获取的不一致，临时规避
  ss << " if (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {" << std::endl;
  ss << " version_is_ASCEND950 = true;" << std::endl;
  ss << " }" << std::endl;
  ss << " ub_size -= (ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910 && "
        "ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B && ub_size % 1024 == 0) ? "
        "256 : 0;"
     << std::endl;
  ss << " (*tiling_parse_data)->ub_size = ub_size;" << std::endl;
  ss << " return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  tiling_parse = ss.str();
}

static void FillShapeDimInfo(const ascir::FusedScheduledResult &fused_schedule_result,
                             const std::map<std::string, std::string> &shape_info, std::stringstream &shape_dim_def,
                             std::stringstream &shape_dim_param) {
  for (const auto &vars : GetFrontendShapeVars(fused_schedule_result)) {
    if (!vars.IsConstExpr()) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        shape_dim_def << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        shape_dim_param << it->first << ", ";
      }
    }
  }
}

static bool HasWorkspaceInNonLastGroup(const ascir::ScheduledResult &schedule_result) {
  const auto &schedule_groups = schedule_result.schedule_groups;
  for (size_t j = 0; j < schedule_groups.size() - 1; j++) {
    for (const auto &impl_graph : schedule_groups[j].impl_graphs) {
      for (const auto &node : impl_graph.GetAllNodes()) {
        if (IsOps<Workspace>(node)) {
          return true;
        }
      }
    }
  }
  return false;
}

static std::set<size_t> GetWorkspaceNodeResultSet(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::set<size_t> result;
  for (const auto &schedule_result_list : fused_schedule_result.node_idx_to_scheduled_results) {
    for (size_t i = 0; i < schedule_result_list.size(); i++) {
      if (HasWorkspaceInNonLastGroup(schedule_result_list[i])) {
        result.insert(i);
      }
    }
  }
  return result;
}

static std::string GenWorkspaceNodeCheckCode(const ascir::FusedScheduledResult &fused_schedule_result) {
  std::stringstream ss;
  std::set<size_t> schedule_result_has_workspace_node = GetWorkspaceNodeResultSet(fused_schedule_result);

  if (schedule_result_has_workspace_node.empty()) {
    return ss.str();
  }

  ss << "  std::set<size_t> schedule_result_has_workspace_node = {";
  bool first = true;
  for (const auto &result_idx : schedule_result_has_workspace_node) {
    if (!first) {
      ss << ", ";
    }
    ss << result_idx;
    first = false;
  }
  ss << "};" << std::endl;

  ss << "  if (version_is_ASCEND950 && ";
  ss << "schedule_result_has_workspace_node.count(tiling_data->graph0_tiling_key) > 0) {" << std::endl;
  ss << "    context->SetScheduleMode(1);" << std::endl;
  ss << "  }" << std::endl;

  return ss.str();
}

static std::string GenLocalMemorySizeCode() {
  std::stringstream ss;
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  if (backend_spec->set_local_memory_size > 0) {
    ss << "  #ifdef CV_RELU_FIXPIP_MODE" << std::endl;
    ss << "  context->SetLocalMemorySize(0);" << std::endl;
    ss << "  #else" << std::endl;
    ss << "  context->SetLocalMemorySize(" << backend_spec->set_local_memory_size << ");" << std::endl;
    ss << "  #endif" << std::endl;
  }
  return ss.str();
}

static void AppendCubeFusionInitCode(std::stringstream &ss, const std::string &shape_dim_param,
                                     const MatMulCubeInfo &cube_info) {
  (void)cube_info;
  ss << "  auto tiling_data =  context->GetTilingData<CVAutofuseTilingData>();" << std::endl;
  ss << "  int64_t ws_size = 0;" << std::endl;
  ss << "  int64_t cube_tiling_key = 0;" << std::endl;
  ss << "  uint32_t cube_block_dim = 0;" << std::endl;
  ss << "  uint32_t basem = 0;" << std::endl;
  ss << "  uint32_t basen = 0;" << std::endl;
  ss << "  CallCubeTiling(" << shape_dim_param
     << "ws_size, cube_block_dim, cube_tiling_key, basem, basen, tiling_data);" << std::endl;
  ss << "  int64_t cube_tiling_key_ub = cube_tiling_key & ~0xF0;" << std::endl;
  ss << "  ResLimit limit;" << std::endl << "  limit.aiv_num = parse->aiv_num;" << std::endl;
  ss << "  limit.ub_size = (uint32_t)parse->ub_size;" << std::endl;
  ss << "  auto ret = ge::GRAPH_SUCCESS;" << std::endl;
  ss << "  const int32_t ub_align_value = 32;" << std::endl;
  ss << "  const int32_t basen_align = (basen + ub_align_value - 1) / ub_align_value * ub_align_value;" << std::endl;
  ss << "  const int32_t basen_basem_align = (basem * basen_align) / 2 + basen_align;" << std::endl;
  ss << "  tiling_data->cube_ub_stage_size = (uint32_t)basen_basem_align;" << std::endl;
  ss << "  tiling_data->cube_m_stage_size = (uint32_t)(basen_basem_align / basen_align);" << std::endl;
}

static void AppendCubeFusionUbModeCode(std::stringstream &ss) {
  ss << "  if (cube_tiling_key_ub != 1) {" << std::endl;
  ss << "    set_g_basen_basem_align(1);" << std::endl;
  ss << "    uint32_t vec_core_num = limit.aiv_num;" << std::endl;
  ss << "    tiling_data->tiling_data.set_block_dim(vec_core_num);" << std::endl;
  ss << "    tiling_data->tiling_data.set_ub_size(limit.ub_size - 256);" << std::endl;
  ss << "    if (!optiling::GetTiling(tiling_data->tiling_data, 2)) {return ge::GRAPH_FAILED;}" << std::endl;
  ss << "    tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "    tiling_data->tiling_data.set_tiling_key(tiling_data->tiling_data.get_tiling_key() - 2);" << std::endl;
  ss << "    // Subtract 2 from tiling_key because case 0/1 are reserved for CV UB normal/fallback tiling."
     << std::endl;
  ss << "    uint32_t vec_block_dim = tiling_data->tiling_data.get_block_dim();" << std::endl;
  ss << "    uint64_t vec_wss = 0U;" << std::endl;
  ss << "    if (!GetWorkspaceSize(tiling_data->tiling_data, vec_wss)) {return ge::GRAPH_FAILED;}" << std::endl;
  ss << "    uint64_t total_workspace_size = 0U;" << std::endl;
  ss << "    if (!CheckedWorkspaceAdd(vec_wss, static_cast<uint64_t>(ws_size), total_workspace_size)) {return "
        "ge::GRAPH_FAILED;}"
     << std::endl;
  ss << "    uint32_t new_block_dim = (cube_block_dim * 2 < vec_block_dim) ? (vec_block_dim + 1) / 2 : cube_block_dim;"
     << std::endl;
  ss << "    const bool is_cv_safety_mix = is_cv_safety_mix_mode(cube_tiling_key);" << std::endl;
  ss << "    const bool use_launch_aic_num = is_cv_safety_blockidx_scheduled_mode(cube_tiling_key);" << std::endl;
  ss << "    context->SetBlockDim(new_block_dim);" << std::endl;
  ss << "    *context->GetWorkspaceSizes(1) = static_cast<int64_t>(total_workspace_size);" << std::endl;
  ss << "    tiling_data->cv_tiling_data.fusion_mode = 1;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.mix_mode = is_cv_safety_mix ? 1 : 0;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_aic_num = use_launch_aic_num ? new_block_dim : cube_block_dim;"
     << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_aiv_num = vec_block_dim;" << std::endl;
  ss << "    tiling_data->cv_tiling_data.cv_vec_wss = vec_wss;" << std::endl;
  ss << "  } else {" << std::endl;
}

static void AppendCubeFusionFallbackCode(std::stringstream &ss, const std::string &shape_dim_param) {
  ss << "  set_g_basen_basem_align(basen_basem_align);" << std::endl;
  ss << "  ret = AutofuseTilingWithConfig(config_file, ";
  ss << shape_dim_param;
  ss << "&(tiling_data->tiling_data), &workspace_size, &block_dim, &limit);" << std::endl;
  ss << "  if (ret == 0) {" << std::endl;
  ss << "  tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "  tiling_data->cube_m_stage_size = (uint32_t)std::max(1, static_cast<int32_t>(tiling_data->"
        "stage_size_name / basen_align));"
     << std::endl;
  ss << "  context->SetBlockDim(cube_block_dim);" << std::endl;
  ss << "  *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024 + ws_size;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.ub_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "  tiling_data->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      ret = ge::GRAPH_SUCCESS;" << std::endl;
  ss << "      set_g_basen_basem_align(1);" << std::endl;
  ss << "      tiling_data->tiling_data.set_block_dim(limit.aiv_num);" << std::endl;
  ss << "      tiling_data->tiling_data.set_ub_size(limit.ub_size - 256);" << std::endl;
  ss << "      if (!optiling::GetTiling(tiling_data->tiling_data, 1)) {return ge::GRAPH_FAILED;}" << std::endl;
  ss << "      tiling_data->stage_size_name = tiling_data->tiling_data.STAGE_SIZE_NAME;" << std::endl;
  ss << "      tiling_data->cube_m_stage_size = (uint32_t)std::max(1, static_cast<int32_t>(tiling_data->"
        "stage_size_name / basen_align));"
     << std::endl;
  ss << "      context->SetBlockDim(cube_block_dim);" << std::endl;
  ss << "      *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024 + ws_size;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.fusion_mode = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.ub_mode = 1;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.mix_mode = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_aic_num = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_aiv_num = 0;" << std::endl;
  ss << "      tiling_data->cv_tiling_data.cv_vec_wss = 0;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  }" << std::endl;
}

static void AppendCubeFusionTilingKeyCode(std::stringstream &ss,
                                          const ascir::FusedScheduledResult &fused_schedule_result) {
  if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
    ss << R"(
  auto tiling_key = FindBestTilingKey(tiling_data->tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(cube_tiling_key));
)";
  }
}

std::string TilingLib::GenCubeFusionTilingBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string &shape_dim_param) const {
  std::stringstream ss;
  MatMulCubeInfo cube_info;
  GE_ASSERT_SUCCESS(ExtractMatMulCubeInfoFromFusedResult(fused_schedule_result, cube_info),
                    "[Extract][MatMulCubeInfo]Failed to extract MatMul cube info from FusedScheduledResult");
  AppendCubeFusionInitCode(ss, shape_dim_param, cube_info);
  AppendCubeFusionUbModeCode(ss);
  AppendCubeFusionFallbackCode(ss, shape_dim_param);
  ss << GenLocalMemorySizeCode();
  ss << GenWorkspaceNodeCheckCode(fused_schedule_result);
  AppendCubeFusionTilingKeyCode(ss, fused_schedule_result);
  return ss.str();
}

std::string TilingLib::GenNonCubeFusionTilingBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                                  const std::string &tiling, const std::string &shape_dim_param) const {
  std::stringstream ss;
  ss << "  auto tiling_data =  context->GetTilingData<" << tiling << ">();" << std::endl;
  ss << "  ResLimit limit;" << std::endl << "  limit.aiv_num = parse->aiv_num;" << std::endl;
  ss << "  limit.ub_size = (uint32_t)parse->ub_size;" << std::endl;
  ss << "  auto ret = AutofuseTilingWithConfig(config_file, ";
  ss << shape_dim_param;
  ss << "tiling_data, &workspace_size, &block_dim, &limit);" << std::endl;
  ss << "  if (ret != 0) {" << std::endl;
  ss << "    return ge::GRAPH_FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  context->SetBlockDim(block_dim);" << std::endl;

  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) &&
      !ascgen_utils::IsJustCubeFixpip(fused_schedule_result) &&
      !ascgen_utils::IsCubeCommonFusedScheduled(fused_schedule_result)) {
    ss << "  *context->GetWorkspaceSizes(1) = 16 * 1024 * 1024;" << std::endl;
  } else {
    ss << "  *context->GetWorkspaceSizes(1) = workspace_size;" << std::endl;
  }
  ss << GenLocalMemorySizeCode();
  ss << GenWorkspaceNodeCheckCode(fused_schedule_result);

  if (ascgen_utils::CanUseTilingKey(fused_schedule_result)) {
    ss << R"(
  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
)";
  }
  return ss.str();
}

std::string TilingLib::GenExternTilingFuncBody(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::map<std::string, std::string> &shape_info,
                                               const std::string &tiling, const std::string &pgo_dir) const {
  std::stringstream ss;
  std::stringstream shape_dim_def;
  std::stringstream shape_dim_param;

  FillShapeDimInfo(fused_schedule_result, shape_info, shape_dim_def, shape_dim_param);
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);" << std::endl;
  ss << "  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);" << std::endl;
  ss << "  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);" << std::endl;
  ss << shape_dim_def.str();
  ss << "  uint64_t workspace_size;" << std::endl << "  uint32_t block_dim;" << std::endl;
  if (enable_autofuse_pgo_) {
    ss << "  static const char* config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt\";" << std::endl;
  } else {
    ss << "  static const char* config_file = nullptr;" << std::endl;
  }
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && !IsStaticSchedResult(fused_schedule_result)) {
    return ss.str() + GenCubeFusionTilingBody(fused_schedule_result, shape_dim_param.str());
  }
  return ss.str() + GenNonCubeFusionTilingBody(fused_schedule_result, tiling, shape_dim_param.str());
}

std::string TilingLib::GenExternTilingFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                           const std::map<std::string, std::string> &shape_info,
                                           const std::string tiling, const std::string &pgo_dir,
                                           const std::string &core_num) const {
  std::stringstream ss;
  std::string extern_c = "extern \"C\"";
  std::string tiling_context = R"(
namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };

  class SymbolTilingParseContext : public KernelContext {
    public:
      fe::PlatFormInfos *GetPlatFormInfos() const {
        auto platform = GetInputValue<fe::PlatFormInfos *>(0);
        if (platform == nullptr) {
          return nullptr;
        }
        return platform;
      }
  };
})";

  ss << tiling_context << std::endl;
  std::string tiling_parse_def;
  int vector_core_num = std::atoi(core_num.c_str());
  GetTilingParse(tiling_parse_def, vector_core_num);
  ss << tiling_parse_def << std::endl;
  const std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result) && IsStaticSchedResult(fused_schedule_result)) {
    ss << extern_c << " ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)" << std::endl;
    ss << "{" << std::endl;
    ss << GenExternTilingFuncBody(fused_schedule_result, shape_info, tiling, pgo_dir);
    ss << "  context->SetBlockDim(CUBE_BLOCK_DIM);" << std::endl;
    ss << "  context->SetTilingKey(static_cast<uint64_t>(CUBE_TILING_KEY));" << std::endl;
    ss << "  return ret;" << std::endl;
    ss << "}" << std::endl;
    ss << extern_c << " ge::graphStatus TilingFuncVec(gert::TilingSymbolEvalContext *context)" << std::endl;
  } else {
    ss << extern_c << " ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)" << std::endl;
  }
  ss << "{" << std::endl;
  if (!IsEmptyTensorSence(fused_schedule_result)) {
    ss << GenExternTilingFuncBody(fused_schedule_result, shape_info, tiling, pgo_dir);
    ss << "  return ret;" << std::endl;
  } else {
    ss << "  context->SetBlockDim(1);" << std::endl;
    ss << "  *context->GetWorkspaceSizes(1) = 0;" << std::endl;
    ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  }
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetTilingSizeFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                            const std::string graph_name, const std::string tiling,
                                            bool is_inductor) const {
  std::stringstream ss;
  GELOGI("start %s Gen GetTilingDataSize function", graph_name.c_str());
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    bool is_batch = false;
    bool is_conv = false;
    std::string input_type;
    std::string output_type;
    GE_ASSERT_SUCCESS(ascgen_utils::GetCubeInfo(fused_schedule_result, is_batch, is_conv, input_type, output_type),
                      "Failed to get cube info from FusedScheduledResult");
    std::string struct_name = is_batch ? "BatchMatMulV3BasicTilingData" : "MatMulV3BasicTilingData";
    if (is_conv) {
      struct_name = "Conv2DTilingData";
    }
    ss << "extern \"C\" size_t GetTilingDataSize()" << std::endl;
    ss << "{" << std::endl;
    if (is_inductor) {
      ss << "  return sizeof(" << tiling << ");" << std::endl;
    } else {
      ss << "  return sizeof(" << struct_name << ");" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "extern \"C\" size_t GetTilingDataSizeVec()" << std::endl;
    ss << "{" << std::endl;
    ss << "  return sizeof(" << tiling << ");" << std::endl;
    ss << "}" << std::endl;
    return ss.str();
  }

  ss << "extern \"C\" size_t GetTilingDataSize()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return sizeof(" << tiling << ");" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::InferShapeDef(const ascir::HintGraph &graph) const {
  (void)graph;
  std::stringstream ss;

  ss << "namespace ge {" << std::endl;
  ss << "static ge::graphStatus InferShape(gert::InferShapeContext* context)" << std::endl;
  ss << "{" << std::endl;
  ss << "    return GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenCheckStaticShapeFunc(bool is_static) const {
  std::stringstream ss;
  ss << "extern \"C\" bool AutofuseIsStaticShape() {" << std::endl;
  ss << "  return " << (is_static ? "true" : "false") << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

// 生成tiling缓存需要的接口
std::string TilingLib::GenTilingCacheFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                          const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  std::string extern_c = "extern \"C\"";
  ss << extern_c << " ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)" << std::endl;
  ss << "{" << std::endl;
  ss << "  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);" << std::endl;
  ss << "  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);"
     << std::endl;
  ss << "  if (symbol_src_vec == nullptr) {" << std::endl;
  ss << "    return ge::GRAPH_FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;

  uint32_t index = 0U;
  std::stringstream ss_tmp;

  for (const auto &vars : GetFrontendShapeVars(fused_schedule_result)) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        // shape dim 定义赋值和传参匹配
        ss_tmp << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss_tmp << "  symbol_src_vec->MutableData()[" << std::to_string(index) << "] = " << it->first << ";"
               << std::endl;
        ss_tmp << std::endl;
        index++;
      }
    }
  }

  std::stringstream ss_size_chk;
  ss_size_chk << "  if (symbol_src_vec->GetCapacity() < " << std::to_string(index) << ") {" << std::endl;
  ss_size_chk << "    return ge::GRAPH_FAILED;" << std::endl;
  ss_size_chk << "  }" << std::endl;
  ss_size_chk << std::endl;
  ss << ((index != 0U) ? ss_size_chk.str() : "");

  ss << ss_tmp.str();
  ss << "  symbol_src_vec->SetSize(" << std::to_string(index) << ");" << std::endl;
  ss << "  return ge::GRAPH_SUCCESS;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenDfxInputSymbolInfo(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::map<std::string, std::string> &shape_info) const {
  std::stringstream ss;
  ss << R"(extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;)"
     << std::endl;

  bool first_sym = true;
  for (const auto &vars : GetFrontendShapeVars(fused_schedule_result)) {
    if (!(vars.IsConstExpr())) {
      std::string var_define = std::string(vars.Str().get());
      auto it = shape_info.find(var_define);
      if (it != shape_info.end()) {
        ss << "  auto " << it->first << " = " << it->second << ";" << std::endl;
        ss << "  symbol_info += (\"";
        if (first_sym) {
          first_sym = false;
        } else {
          ss << ", ";
        }
        ss << it->first << ": \" + std::to_string(" << it->first << "));" << std::endl;
        ss << std::endl;
      }
    }
  }
  ss << R"(
  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
})" << std::endl;
  return ss.str();
}

std::string TilingLib::GenFindBestTilingKeyFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                                const std::string &tiling_data_name) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t FindBestTilingKey(" << tiling_data_name << " &t)" << std::endl;
  ss << "{" << std::endl;
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    auto schedule_graphs = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
    for (uint32_t i = 0; i < schedule_graphs.size(); i++) {
      auto tiling_key = std::to_string(i);
      ss << (i == 0U ? "  if (" : "  } else if (") << ("t.tiling_key == " + tiling_key + ") {") << std::endl;
      ss << "    return " + tiling_key + ";" << std::endl;
    }
    ss << "  }" << std::endl;
  } else {
    uint64_t tiling_key_count = 0U;
    if (TryCalcTilingKeyCount(fused_schedule_result, kInt64TilingKeyCapacity, tiling_key_count)) {
      GenMulGroupFindBestTilingKey(fused_schedule_result, ss);
    }
  }
  ss << "  return -1;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  uint64_t count = CalcTilingKeyCount(fused_schedule_result);
  ss << "extern \"C\" uint64_t GetTilingKeyCount()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return " << GenUint64Literal(count) << ";" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyForStatic() const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t GetTilingKeyForStatic()" << std::endl;
  ss << "{" << std::endl;
  ss << "  return FindBestTilingKey(TilingDataValue);" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenGetTilingKeyKernelTypeForStatic(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  ss << "std::string kernel_type;" << std::endl;
  ss << "extern \"C\" const char* GetTilingKeyKernelTypeForStatic()" << std::endl;
  ss << "{" << std::endl;
  ss << "  const std::map<int64_t, std::string> kernel_type_map = {" << std::endl;
  uint32_t tiling_key = 0U;
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      auto schedule_groups = scheduled_result.schedule_groups;
      std::vector<std::vector<bool>> per_group_conditions;
      for (const auto &schedule_group : schedule_groups) {
        auto schedule_graphs = schedule_group.impl_graphs;
        std::vector<bool> conditions;
        for (const auto &schedule_graph : schedule_graphs) {
          bool has_workspace_node = HasWorkSpaceNode(schedule_graph);
          conditions.emplace_back(has_workspace_node);
        }
        per_group_conditions.emplace_back(std::move(conditions));
      }
      std::vector<bool> current;
      CodegenTilingKeyKerneType(ss, per_group_conditions, current, 0, tiling_key);
    }
  }
  ss << "  };" << std::endl;
  ss << R"(
  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
})" << std::endl;
  return ss.str();
}

}  // namespace codegen
