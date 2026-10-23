/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

#include "autofuse_config/auto_fuse_config.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/platform_context.h"

namespace codegen {
using namespace ascgen_utils;
using namespace ascir;

namespace {
std::string GenPgoMeasuredSearchModel() {
  return R"(
inline std::string PgoMeasuredCandidateKey(const AutofuseTilingDataPerf &candidate) {
  const char *ptr = reinterpret_cast<const char *>(&candidate.tiling_data);
  return std::string(ptr, ptr + sizeof(AutofuseTilingData));
}

inline std::vector<AutofuseTilingDataPerf> NormalizePgoMeasuredCandidates(
    std::vector<AutofuseTilingDataPerf> raw_candidates) {
  std::vector<AutofuseTilingDataPerf> candidates;
  candidates.reserve(raw_candidates.size());
  std::unordered_set<std::string> seen;
  for (auto &candidate : raw_candidates) {
    const std::string key = PgoMeasuredCandidateKey(candidate);
    if (!seen.insert(key).second) {
      continue;
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}
)";
}
}  // namespace

std::string TilingLib::GenPgoTilingFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                        const std::string &tiling, codegen::PgoShapeStringStream &pgo_shape_dim,
                                        bool is_inductor_scene, const std::string &core_num) const {
  std::stringstream ss;
  ss << GenPgoMeasuredSearchModel();
  // 生成 AutofuseTilingWithConfig 函数
  ss << GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene);
  // 生成 PgoSaveTilingKey 函数
  GenPgoSaveTilingKey(ss);
  // 生成 SavePGOSearchTilingDataFunc 函数
  ss << GenSavePGOSearchTilingDataFunc(tiling);
  // 生成 SavePGOConfigTilingDataFunc 函数
  ss << GenSavePGOConfigTilingDataFunc();

  // 生成 PgoByCoreNumTilingSearch函数
  ss << GenPgoTilingSearchByCoreNum(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene, core_num);

  // 生成 PgoTilingSearch 函数
  ss << GenPgoTilingSearchPGO(fused_schedule_result, pgo_shape_dim, tiling, is_inductor_scene, core_num);

  ss << GenPgoTilingSearch(fused_schedule_result, pgo_shape_dim, tiling);

  return ss.str();
}

std::string TilingLib::GenPgoAutofuseTiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                            codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                            bool is_inductor_scene) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t AutofuseTilingWithConfig(const char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint64_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, int32_t tiling_case_id = -1)" << std::endl;
  ss << "{" << std::endl;

  ss << " const ResLimit effective_res_limit = GetResLimit(res_limit);" << std::endl;
  ss << " const ResLimit *limit = &effective_res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  tiling->set_block_dim(limit->aiv_num);" << std::endl;
  if (is_inductor_scene) {
    ss << "  tiling->set_ub_size(limit->ub_size - 256);" << std::endl;
  } else {
    ss << "  tiling->set_ub_size(limit->ub_size);" << std::endl;
  }
  if (!ascgen_utils::IsJustCubeFixpip(fused_schedule_result)) {
    if (enable_autofuse_pgo_) {
      ss << "  auto pgo_tiling = *tiling;" << std::endl;
      ss << "  bool use_pgo_tiling = PGOGetTilingKey(config_file, pgo_tiling);" << std::endl;
      ss << "  if (use_pgo_tiling && (pgo_tiling.get_block_dim() == 0U || "
            "pgo_tiling.get_block_dim() > limit->aiv_num)) {"
         << std::endl;
      ss << "    OP_LOGW(OP_NAME, \"Loaded PGO block_dim %u is outside core budget [1, %u].\", "
            "pgo_tiling.get_block_dim(), limit->aiv_num);"
         << std::endl;
      ss << "    use_pgo_tiling = false;" << std::endl;
      ss << "  }" << std::endl;
      ss << "  if (use_pgo_tiling) {" << std::endl;
      ss << "    *tiling = pgo_tiling;" << std::endl;
      if (UseAttCodegen(fused_schedule_result)) {
        if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
          ss << "    optiling::EmitFinalTilingByCase(*tiling, tiling->get_block_dim(), \"pgo\", \"pgo\", 0U, -1);"
             << std::endl;
        } else {
          ss << "    optiling::EmitFinalTilingForPgo(*tiling, tiling->get_block_dim());" << std::endl;
        }
      }
      ss << "  } else {" << std::endl;
      ss << "    if (!optiling::GetTiling(*tiling, tiling_case_id, nullptr)) {" << std::endl;
      ss << "      return -1;" << std::endl;
      ss << "    }" << std::endl;
      ss << "  }" << std::endl;
    } else {
      ss << "  (void)config_file;" << std::endl;
      ss << "  if (!optiling::GetTiling(*tiling, tiling_case_id, nullptr)) {" << std::endl;
      ss << "    return -1;" << std::endl;
      ss << "  }" << std::endl;
    }
    ss << "  *blockDim = tiling->get_block_dim();" << std::endl;
    ss << "  using namespace optiling;" << std::endl;
  }
  ss << "  if (!GetWorkspaceSize(*tiling, *workspaceSize)) {" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  if (!is_inductor_scene) {
    ss << "  if (!CheckedWorkspaceAdd(*workspaceSize, 16ULL * 1024ULL * 1024ULL, *workspaceSize)) {" << std::endl;
    ss << "    return -1;" << std::endl;
    ss << "  }" << std::endl;
  }
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenProfilingAllTilingData(std::string tiling_data_list_name,
                                                 std::string tiling_data_perf_list_name,
                                                 const ascir::FusedScheduledResult &fused_schedule_result,
                                                 bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  *workspaceSize = 0;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> raw_search_candidates;" << std::endl;
  ss << "  for (const auto &tiling_data_item : " << tiling_data_list_name << ") {" << std::endl;
  ss << "    uint64_t candidate_workspace_size = 0U;" << std::endl;
  ss << "    if (!GetWorkspaceSize(tiling_data_item, candidate_workspace_size)) {" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    *workspaceSize = std::max(candidate_workspace_size, *workspaceSize);" << std::endl;
  ss << "    AutofuseTilingDataPerf tiling_data_perf;" << std::endl;
  ss << "    tiling_data_perf.tiling_data = tiling_data_item;" << std::endl;
  ss << "    tiling_data_perf.best_perf = DBL_MAX;" << std::endl;
  ss << "    raw_search_candidates.push_back(tiling_data_perf);" << std::endl;
  ss << "  }" << std::endl;
  if (!is_inductor_scene) {
    ss << "    if (!CheckedWorkspaceAdd(*workspaceSize, 16ULL * 1024ULL * 1024ULL, *workspaceSize)) {" << std::endl;
    ss << "      return -1;" << std::endl;
    ss << "    }" << std::endl;
  }
  ss << "  auto normalized_search_candidates = NormalizePgoMeasuredCandidates(std::move(raw_search_candidates));"
     << std::endl;
  ss << "  " << tiling_data_perf_list_name << ".insert(" << tiling_data_perf_list_name
     << ".end(), normalized_search_candidates.begin(), normalized_search_candidates.end());" << std::endl;
  ss << "  PgoConfig::Instance().batch_callback(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, *workspaceSize, &" << tiling_data_perf_list_name << ");" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearchByCoreNum(const ascir::FusedScheduledResult &fused_schedule_result,
                                                   codegen::PgoShapeStringStream &pgo_shape_dim,
                                                   const std::string &tiling, bool is_inductor_scene,
                                                   const std::string &core_num) const {
  std::stringstream ss;
  ss << "extern \"C\" int64_t PgoTilingSearchByCoreNum(char *search_file, char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint64_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr) {"
     << std::endl;
  ss << "  (void)prof_callback;" << std::endl;
  ss << "  (void)prof_batch_callback;" << std::endl;
  ss << "  const ResLimit effective_res_limit = GetResLimit(res_limit);" << std::endl;
  ss << "  const ResLimit *limit = &effective_res_limit;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput(core_num);
  ss << "  using namespace optiling;" << std::endl;
  ss << "  std::vector<AutofuseTilingData> tiling_data_list;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_perf_list;" << std::endl;
  ss << "  double axeorder_cost = DBL_MAX;" << std::endl;
  ss << "  AutofuseTiling(";
  ss << pgo_shape_dim.shape_dim_use.str();
  ss << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback(";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result);
  ss << "stream, *workspaceSize, tiling, &axeorder_cost);" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_data_axereorder_perf;" << std::endl;
  ss << "  tiling_data_axereorder_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_data_axereorder_perf.best_perf = axeorder_cost;" << std::endl;
  ss << "  tiling_data_perf_list.push_back(tiling_data_axereorder_perf);" << std::endl;
  ss << "  PgoConfig::Instance().need_change_solver_run = true;" << std::endl;
  ss << "  PgoConfig::Instance().pgo_threshold_index = 0;" << std::endl;
  ss << "  while (PgoConfig::Instance().pgo_threshold_index < PgoConfig::Instance().pgo_threshold_list_size) {"
     << std::endl;
  ss << "    if (!optiling::PGOByCoreNumSearchTilingKey(tiling_data_list, tiling, max_block_dim)) {" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    PgoConfig::Instance().pgo_threshold_index++;" << std::endl;
  ss << "  }" << std::endl;
  ss << GenProfilingAllTilingData("tiling_data_list", "tiling_data_perf_list", fused_schedule_result,
                                  is_inductor_scene);
  ss << "  best_perf = DBL_MAX;" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_perf_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_perf_list, best_perf);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}

std::string TilingLib::GenPgoTilingSearch(const ascir::FusedScheduledResult &fused_schedule_result,
                                          codegen::PgoShapeStringStream &pgo_shape_dim,
                                          const std::string &tiling) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearch(char *search_file, char *config_file, ";
  ss << pgo_shape_dim.shape_dim_def.str();
  ss << tiling << " *tiling, uint64_t *workspaceSize, uint32_t *blockDim,";
  ss << " ResLimit *res_limit = nullptr, ";
  ss << PGOSearchFuncInputOutputDef(fused_schedule_result);
  ss << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
        "prof_batch_callback=nullptr) {"
     << std::endl;
  ss << "  const char* var = std::getenv(\"AUTOFUSE_DFX_FLAGS\");" << std::endl;
  ss << "  if ((var != nullptr) && (std::string(var).find(\"autofuse_pgo_algo=pruning\") != std::string::npos)) {"
     << std::endl;
  ss << "    PgoConfig::Instance().pgo_algorithm = 0;" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    PgoConfig::Instance().pgo_algorithm = 1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  PgoConfig::Instance().single_callback = prof_callback;" << std::endl;
  ss << "  PgoConfig::Instance().batch_callback = prof_batch_callback;" << std::endl;
  ss << "  if (PgoConfig::Instance().pgo_algorithm == 0) {" << std::endl;
  ss << "    return PgoTilingSearchPGO(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str()
     << " tiling, workspaceSize, blockDim, res_limit, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  } else if (PgoConfig::Instance().pgo_algorithm == 1) {" << std::endl;
  ss << "    return PgoTilingSearchByCoreNum(search_file, config_file, " << pgo_shape_dim.shape_dim_use.str()
     << " tiling, workspaceSize, blockDim, res_limit, ";
  ss << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetMaxBlockDimFromInput(const std::string &core_num) const {
  std::stringstream ss;
  if (std::stoi(core_num) != 0) {
    ss << "  auto max_core_num = " << core_num << ";" << std::endl;
    ss << "  tiling->set_block_dim(max_core_num);" << std::endl;
    ss << "  max_block_dim = max_core_num;" << std::endl;
  }
  return ss.str();
}

std::string TilingLib::GenGetAutoFuseTilingInput(bool is_inductor_scene) const {
  std::stringstream ss;
  ss << "tiling, workspaceSize, blockDim, ";
  if (is_inductor_scene) {
    ss << "res_limit);" << std::endl;
  } else {
    ss << "limit->aiv_num, limit->ub_size - 256);" << std::endl;
  }

  return ss.str();
}

void TilingLib::GenPgoTilingKeySearch(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  if (ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    ss << "  // 不使用，仅保持接口一致" << std::endl;
    ss << "  std::unordered_map<int64_t, uint64_t> workspace_map;" << std::endl;
    ss << "  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, "
       << PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, *workspaceSize, best_perf, workspace_map)) {"
       << std::endl;
  } else {
    ss << "  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, "
       << PGOSearchFuncInputOutputCall(fused_schedule_result) << "stream, *workspaceSize, best_perf)) {" << std::endl;
  }
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
}

std::string TilingLib::GenPgoTilingSearchPGO(const ascir::FusedScheduledResult &fused_schedule_result,
                                             codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                             bool is_inductor_scene, const std::string &core_num) const {
  std::stringstream ss;

  ss << "extern \"C\" int64_t PgoTilingSearchPGO(char *search_file, char *config_file, "
     << pgo_shape_dim.shape_dim_def.str() << tiling << " *tiling, uint64_t *workspaceSize, uint32_t *blockDim,"
     << " ResLimit *res_limit = nullptr, " << PGOSearchFuncInputOutputDef(fused_schedule_result)
     << "void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback "
     << "prof_batch_callback=nullptr) {" << std::endl;

  ss << "  (void)prof_callback;" << std::endl;
  ss << "  (void)prof_batch_callback;" << std::endl;
  ss << "  const ResLimit effective_res_limit = GetResLimit(res_limit);" << std::endl;
  ss << "  const ResLimit *limit = &effective_res_limit;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> tiling_data_list;" << std::endl;
  ss << pgo_shape_dim.tiling_set_shape_dim.str();
  ss << "  double best_perf = DBL_MAX;" << std::endl;
  ss << "  uint32_t max_block_dim = limit->aiv_num;" << std::endl;
  ss << GenGetMaxBlockDimFromInput(core_num);
  ss << "  AutofuseTiling(" << pgo_shape_dim.shape_dim_use.str() << GenGetAutoFuseTilingInput(is_inductor_scene);
  ss << "  PgoConfig::Instance().single_callback(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "stream, *workspaceSize, tiling, &best_perf);" << std::endl;
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"axesreorder solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingDataPerf tiling_perf;" << std::endl;
  ss << "  tiling_perf.tiling_data = *tiling;" << std::endl;
  ss << "  tiling_perf.best_perf = best_perf;" << std::endl;
  ss << "  tiling_data_list.push_back(tiling_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"axesreorder solution base perf is %lf\", best_perf);" << std::endl;
  ss << "  tiling->set_block_dim(max_block_dim);" << std::endl;
  GenPgoTilingKeySearch(fused_schedule_result, ss);
  ss << "  if (optiling::IsEqual(best_perf, DBL_MAX)) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"pgo solution get perf failed %lf\", best_perf);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  SavePGOSearchTilingData(search_file, tiling_data_list);" << std::endl;
  ss << "  SavePGOConfigTilingData(config_file, tiling_data_list, best_perf);" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"pgo solution best perf is %lf\", best_perf);" << std::endl;
  ss << std::endl;

  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenGetResLimitStru(void) const {
  std::stringstream ss;
  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;

  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info));
  ss << "constexpr ResLimit g_no_limit_res = {1, " << platform_info.aiv_num << ", 0, " << platform_info.ub_size
     << ", {}};" << std::endl;
  ss << "inline ResLimit GetResLimit(const ResLimit *res_limit) {" << std::endl;
  ss << "  ResLimit limit = g_no_limit_res;" << std::endl;
  ss << "  if (res_limit == nullptr) {" << std::endl;
  ss << "    return limit;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (res_limit->valid_num > 0U) { limit.valid_num = res_limit->valid_num; }" << std::endl;
  ss << "  if (res_limit->aiv_num > 0U) { limit.aiv_num = res_limit->aiv_num; }" << std::endl;
  ss << "  if (res_limit->aic_num > 0U) { limit.aic_num = res_limit->aic_num; }" << std::endl;
  ss << "  if (res_limit->ub_size > 0U) { limit.ub_size = res_limit->ub_size; }" << std::endl;
  ss << "  for (uint32_t i = 0U; i < 10U; ++i) {" << std::endl;
  ss << "    if (res_limit->resv[i] > 0U) { limit.resv[i] = res_limit->resv[i]; }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return limit;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

bool TilingLib::IsMixKernelTaskType(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.workspace_nodes.size() != 0;
}
std::string TilingLib::GenPGOGetTilingKey(const std::string tiling) const {
  std::stringstream ss;
  ss << "bool PGOGetTilingKey(const char *config_file_path, " << tiling << " &tiling_data) {" << std::endl;
  ss << "  OP_LOGD(OP_NAME, \"PGOGetTilingKey from file:%s.\", config_file_path);" << std::endl;
  ss << "  static int best_config = 0;" << std::endl;
  ss << "  static " + tiling + " best_tiling;" << std::endl;
  ss << "  if (best_config == 0) {" << std::endl;
  ss << "    std::ifstream config_file(config_file_path);" << std::endl;
  ss << "    if (!config_file.is_open()) {" << std::endl;
  ss << "      OP_LOGD(OP_NAME, \"failed to open or not exist: %s.\", config_file_path);" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"[Start to use tiling result]: %s.\", config_file_path);" << std::endl;
  ss << "    std::string line;" << std::endl;
  ss << "    // first line: 0:read every time; 1:read first time" << std::endl;
  ss << "    std::getline(config_file, line);" << std::endl;
  ss << "    std::istringstream iss0(line);" << std::endl;
  ss << "    int flag = -1;" << std::endl;
  ss << "    if (!(iss0 >> flag) || (flag != 0 && flag != 1)) {" << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"Invalid PGO config flag.\");" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    OP_LOGD(OP_NAME, \"best_config %d.\", flag);" << std::endl;
  ss << "    // second line: tiling_data dumped as int32 decimals, space-separated" << std::endl;
  ss << "    std::getline(config_file, line);" << std::endl;
  ss << "    if (line.find('#') != std::string::npos) {" << std::endl;
  ss << "        line = line.substr(0, line.find('#'));" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::istringstream iss1(line);" << std::endl;
  ss << "    std::vector<int32_t> tiling_i32;" << std::endl;
  ss << "    tiling_i32.reserve((sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t));" << std::endl;
  ss << "    int64_t tmp = 0;" << std::endl;
  ss << "    while (iss1 >> tmp) {" << std::endl;
  ss << "      tiling_i32.push_back(static_cast<int32_t>(tmp));" << std::endl;
  ss << "    }" << std::endl;
  ss << "    const size_t expect_num = (sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t);" << std::endl;
  ss << "    if (!iss1.eof() || tiling_i32.size() != expect_num) {" << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"Invalid PGO tiling data length.\");" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (memcpy_s(&tiling_data, sizeof(tiling_data), tiling_i32.data(), sizeof(tiling_data)) != EOK) {"
     << std::endl;
  ss << "      OP_LOGW(OP_NAME, \"Failed to load PGO tiling data.\");" << std::endl;
  ss << "      return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    config_file.close();" << std::endl;
  ss << "    if (flag == 1) {" << std::endl;
  ss << "      best_tiling = tiling_data;" << std::endl;
  ss << "      best_config = flag;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    tiling_data = best_tiling;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return true;" << std::endl;
  ss << "}" << std::endl;
  ss << "" << std::endl;
  return ss.str();
}

std::string TilingLib::GenSavePGOSearchTilingDataFunc(const std::string tiling) const {
  std::stringstream ss;

  // SavePGOSearchTilingData
  ss << "void SavePGOSearchTilingData(char *search_file, std::vector<" << tiling << "Perf> &tiling_data_list, "
     << "std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOSearchTilingData to file:%s.\", search_file);" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "    PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string TilingLib::GenSavePGOConfigTilingDataFunc() const {
  std::stringstream ss;

  // SavePGOConfigTilingData
  ss << "void SavePGOConfigTilingData(char *file, std::vector<AutofuseTilingDataPerf> &tiling_data_list, "
     << "double best_perf, std::ios::openmode mode = std::ios::out) {" << std::endl;
  ss << "  OP_LOGI(OP_NAME, \"SavePGOConfigTilingData to file:%s.\", file);" << std::endl;
  ss << "  std::ofstream out_file(file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    OP_LOGE(OP_NAME, \"Failed to open file:%s.\", file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (PgoConfig::Instance().pgo_algorithm == 1) {" << std::endl;
  ss << "    for (auto item : tiling_data_list) {" << std::endl;
  ss << "      if (item.best_perf < best_perf) {" << std::endl;
  ss << "        best_perf = item.best_perf;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file << \"1\" << std::endl;" << std::endl;
  ss << "  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {" << std::endl;
  ss << "    if (optiling::IsEqual(item->best_perf, best_perf)) {" << std::endl;
  ss << "      PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  return;" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}
}  // namespace codegen
