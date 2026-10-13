/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_tiling_data.h"
#include "autofuse_tiling_func_pgo.h"
#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>

using namespace optiling;

struct ResLimit {
  uint32_t valid_num = 0U;
  uint32_t aiv_num = 0U;
  uint32_t aic_num = 0U;
  uint32_t ub_size = 0U;
  uint32_t resv[10] = {};
};

namespace {
using GenerateMeasuredTopnSolutionsType = int64_t (*)(const std::vector<std::map<std::string, std::string>> &, int64_t,
                                                      std::vector<AutofuseTilingData> &, std::vector<int64_t> &,
                                                      std::vector<int64_t> &, ResLimit *);
using GenerateTopnSolutionsType = int64_t (*)(const std::vector<std::map<std::string, std::string>> &, int64_t,
                                              std::vector<AutofuseTilingData> &, std::vector<int64_t> &,
                                              std::vector<int64_t> &, ResLimit *);
using SetTopnPgoContextType = int64_t (*)(PgoTensorArgs *, void *, ProfilingCallback, ProfilingBatchCallback,
                                          std::vector<AutofuseTilingDataPerf> *);
using ClearTopnPgoContextType = void (*)();
using GetTilingDataReprType = std::string (*)(const AutofuseTilingData *);
using TestParseInductorPgoResultType = int (*)(const char *, int64_t, bool);

struct SamplingStats {
  size_t batch_calls = 0U;
  size_t single_calls = 0U;
  size_t sampled_candidates = 0U;
  std::map<std::string, double> min_perf;
  std::set<uint32_t> sampled_block_dims;
};

struct TopnRunResult {
  SamplingStats stats;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  std::vector<std::string> reprs;
  std::vector<AutofuseTilingDataPerf> measured_candidates;
};

SamplingStats g_stats;
PgoTensorArgs g_tensor_args;
int32_t g_stream_token = 0;
void *g_tiling_handle = nullptr;
GenerateMeasuredTopnSolutionsType g_generate_measured_topn = nullptr;
GenerateTopnSolutionsType g_generate_topn = nullptr;
SetTopnPgoContextType g_set_context = nullptr;
ClearTopnPgoContextType g_clear_context = nullptr;
GetTilingDataReprType g_get_repr = nullptr;
TestParseInductorPgoResultType g_test_parse_result = nullptr;
enum class PerfMode { kBlockDim, kDefaultWorst, kDefaultBest };
PerfMode g_perf_mode = PerfMode::kBlockDim;
std::string g_default_repr;

template <typename T>
bool LoadSymbol(T &target, const char *name) {
  target = reinterpret_cast<T>(dlsym(g_tiling_handle, name));
  return target != nullptr;
}

bool LoadTiling(const char *path) {
  g_tiling_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  return g_tiling_handle != nullptr && LoadSymbol(g_generate_measured_topn, "GenerateMeasuredTopnSolutions") &&
         LoadSymbol(g_generate_topn, "GenerateTopnSolutions") && LoadSymbol(g_set_context, "SetTopnPgoContext") &&
         LoadSymbol(g_clear_context, "ClearTopnPgoContext") && LoadSymbol(g_get_repr, "GetTilingDataRepr") &&
         LoadSymbol(g_test_parse_result, "TestParseInductorPgoResult");
}

double GetFakePerf(const AutofuseTilingData &tiling_data) {
  const auto block_dim = tiling_data.get_block_dim();
  const auto repr = g_get_repr(&tiling_data);
  if (!g_default_repr.empty() && repr == g_default_repr) {
    if (g_perf_mode == PerfMode::kDefaultWorst) {
      return 1000.0;
    }
    if (g_perf_mode == PerfMode::kDefaultBest) {
      return 0.5;
    }
  }
  return static_cast<double>(block_dim);
}

void RecordPerf(const std::string &repr, double perf) {
  const auto iter = g_stats.min_perf.find(repr);
  if (iter == g_stats.min_perf.end() || perf < iter->second) {
    g_stats.min_perf[repr] = perf;
  }
}

void RecordBlockDim(const AutofuseTilingData &tiling_data) {
  g_stats.sampled_block_dims.insert(tiling_data.get_block_dim());
}

long FakeSingleCallback(PgoTensorArgs *, void *, uint64_t, AutofuseTilingData *tiling_data, double *cost_time) {
  if (tiling_data == nullptr || cost_time == nullptr) {
    return -1;
  }
  ++g_stats.single_calls;
  ++g_stats.sampled_candidates;
  const auto repr = g_get_repr(tiling_data);
  *cost_time = GetFakePerf(*tiling_data);
  RecordPerf(repr, *cost_time);
  RecordBlockDim(*tiling_data);
  return 0;
}

long FakeBatchCallback(PgoTensorArgs *, void *, uint64_t, std::vector<AutofuseTilingDataPerf> *profiles) {
  if (profiles == nullptr) {
    return -1;
  }
  ++g_stats.batch_calls;
  for (auto &profile : *profiles) {
    ++g_stats.sampled_candidates;
    const auto repr = g_get_repr(&profile.tiling_data);
    profile.best_perf = GetFakePerf(profile.tiling_data) + 1.0 / static_cast<double>(g_stats.batch_calls + 1U);
    RecordPerf(repr, profile.best_perf);
    RecordBlockDim(profile.tiling_data);
  }
  return 0;
}

long FailingBatchCallback(PgoTensorArgs *, void *, uint64_t, std::vector<AutofuseTilingDataPerf> *) {
  return -1;
}

long InvalidatingBatchCallback(PgoTensorArgs *, void *, uint64_t, std::vector<AutofuseTilingDataPerf> *profiles) {
  if (profiles == nullptr) {
    return -1;
  }
  for (auto &profile : *profiles) {
    profile.best_perf = std::numeric_limits<double>::max();
  }
  return 0;
}

bool SetCallbacks(ProfilingBatchCallback batch_callback, std::vector<AutofuseTilingDataPerf> *measured_candidates) {
  return g_set_context(&g_tensor_args, &g_stream_token, FakeSingleCallback, batch_callback, measured_candidates) == 0;
}

bool RunMeasuredTopn(int64_t topn, TopnRunResult &result, PerfMode perf_mode = PerfMode::kBlockDim,
                     ProfilingBatchCallback batch_callback = FakeBatchCallback) {
  g_stats = SamplingStats();
  g_perf_mode = perf_mode;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  std::vector<AutofuseTilingDataPerf> measured_candidates;
  if (!SetCallbacks(batch_callback, &measured_candidates)) {
    return false;
  }
  ResLimit limit = {1U, 4U, 0U, 1024U * 1024U, {}};
  const auto ret = g_generate_measured_topn({}, topn, tiling_datas, workspaces, block_dims, &limit);
  g_clear_context();
  if (ret != 0 || tiling_datas.empty() || tiling_datas.size() != workspaces.size() ||
      tiling_datas.size() != block_dims.size()) {
    return false;
  }
  result.stats = g_stats;
  result.tiling_datas = tiling_datas;
  result.workspaces = workspaces;
  result.block_dims = block_dims;
  result.measured_candidates = measured_candidates;
  for (const auto &tiling_data : tiling_datas) {
    result.reprs.push_back(g_get_repr(&tiling_data));
  }
  return true;
}

bool IsStablePerfOrder(const TopnRunResult &result) {
  for (size_t i = 1U; i < result.reprs.size(); ++i) {
    const auto &previous_repr = result.reprs[i - 1U];
    const auto &current_repr = result.reprs[i];
    const auto previous_perf = result.stats.min_perf.at(previous_repr);
    const auto current_perf = result.stats.min_perf.at(current_repr);
    if (previous_perf > current_perf || (previous_perf == current_perf && previous_repr > current_repr)) {
      return false;
    }
  }
  return true;
}

bool IsFallbackPrefixOrder(const TopnRunResult &result) {
  if (result.reprs.size() < 2U || result.reprs[1] != g_default_repr) {
    return false;
  }
  TopnRunResult without_fallback = result;
  without_fallback.reprs.erase(without_fallback.reprs.begin() + 1);
  return IsStablePerfOrder(without_fallback);
}

bool IsMeasuredCandidateOrderStable(const TopnRunResult &result) {
  for (size_t i = 1U; i < result.measured_candidates.size(); ++i) {
    const auto &previous = result.measured_candidates[i - 1U];
    const auto &current = result.measured_candidates[i];
    const auto previous_repr = g_get_repr(&previous.tiling_data);
    const auto current_repr = g_get_repr(&current.tiling_data);
    if (previous.best_perf > current.best_perf ||
        (previous.best_perf == current.best_perf && previous_repr > current_repr)) {
      return false;
    }
  }
  return true;
}

bool InvalidTopnIsRejected() {
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  return g_generate_measured_topn({}, 0, tiling_datas, workspaces, block_dims, nullptr) != 0;
}

bool FailingCallbackIsRejected() {
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  std::vector<AutofuseTilingDataPerf> measured_candidates;
  if (!SetCallbacks(FailingBatchCallback, &measured_candidates)) {
    return false;
  }
  const auto ret = g_generate_measured_topn({}, 1, tiling_datas, workspaces, block_dims, nullptr);
  g_clear_context();
  return ret != 0;
}

bool CoreLimitAbovePlatformIsClamped() {
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  std::vector<AutofuseTilingDataPerf> measured_candidates;
  if (!SetCallbacks(FakeBatchCallback, &measured_candidates)) {
    return false;
  }
  ResLimit limit = {1U, std::numeric_limits<uint32_t>::max(), 0U, 1024U * 1024U, {}};
  const auto ret = g_generate_measured_topn({}, 1, tiling_datas, workspaces, block_dims, &limit);
  g_clear_context();
  return ret == 0 && !tiling_datas.empty() && !g_stats.sampled_block_dims.empty() &&
         *g_stats.sampled_block_dims.rbegin() < limit.aiv_num;
}

template <typename T>
void WriteValue(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint64_t HashTiling(const AutofuseTilingData &tiling_data) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&tiling_data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < sizeof(tiling_data); ++i) {
    hash = (hash ^ bytes[i]) * 1099511628211ULL;
  }
  return hash;
}

bool WriteResult(const std::string &path, const TopnRunResult &result) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const uint32_t version = 2U;
  const uint32_t flags = 0U;
  const uint32_t count = static_cast<uint32_t>(result.tiling_datas.size());
  const uint32_t tiling_size = sizeof(AutofuseTilingData);
  const uint32_t record_header_size = 32U;
  out.write("AUTOFUSE_PGO_TOPN_V2", 20);
  WriteValue(out, version);
  WriteValue(out, flags);
  WriteValue(out, count);
  WriteValue(out, tiling_size);
  WriteValue(out, record_header_size);
  for (size_t i = 0; i < result.tiling_datas.size(); ++i) {
    const uint64_t repr_size = result.reprs[i].size();
    const uint64_t hash = HashTiling(result.tiling_datas[i]);
    WriteValue(out, repr_size);
    WriteValue(out, static_cast<uint64_t>(result.workspaces[i]));
    WriteValue(out, result.block_dims[i]);
    WriteValue(out, hash);
    WriteValue(out, result.tiling_datas[i]);
    out.write(result.reprs[i].data(), static_cast<std::streamsize>(result.reprs[i].size()));
  }
  return out.good();
}

bool VerifyBadResult(const std::string &path, const TopnRunResult &result, std::streamoff offset, uint8_t value) {
  if (!WriteResult(path, result)) {
    return false;
  }
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  uint8_t current = 0U;
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(&current), sizeof(current));
  current ^= value;
  file.seekp(offset);
  file.write(reinterpret_cast<const char *>(&current), sizeof(current));
  file.close();
  return g_test_parse_result(path.c_str(), 4, false) == 0;
}

bool VerifyResultParser(const TopnRunResult &result) {
  const std::string path = "/tmp/autofuse_pgo_parser_" + std::to_string(getpid());
  if (!WriteResult(path, result) || g_test_parse_result(path.c_str(), 4, true) != 0 ||
      g_test_parse_result(path.c_str(), 0, false) != 0 || !VerifyBadResult(path, result, 32, 0xFFU) ||
      !VerifyBadResult(path, result, 64, 0xFFU)) {
    unlink(path.c_str());
    return false;
  }
  if (!WriteResult(path, result) || truncate(path.c_str(), 41) != 0 ||
      g_test_parse_result(path.c_str(), 4, false) != 0 || !WriteResult(path, result)) {
    unlink(path.c_str());
    return false;
  }
  std::ofstream(path, std::ios::binary | std::ios::app).put('x');
  const bool valid = g_test_parse_result(path.c_str(), 4, false) == 0;
  unlink(path.c_str());
  return valid;
}

bool CheckProxyRejectsInvalidTopn(int64_t topn) {
  std::vector<AutofuseTilingData> tilings(1);
  std::vector<int64_t> workspaces = {123};
  std::vector<int64_t> block_dims = {456};
  const auto ret = g_generate_topn({}, topn, tilings, workspaces, block_dims, nullptr);
  return ret != 0 && tilings.empty() && workspaces.empty() && block_dims.empty();
}

bool CheckProxyFallback(int64_t topn) {
  std::vector<AutofuseTilingData> tilings;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
  const auto ret = g_generate_topn({}, topn, tilings, workspaces, block_dims, nullptr);
  return ret == 0 && !tilings.empty() && tilings.size() <= static_cast<size_t>(topn) &&
         tilings.size() == workspaces.size() && tilings.size() == block_dims.size();
}

bool VerifyProxyFallbacks() {
  return CheckProxyRejectsInvalidTopn(1025) && CheckProxyFallback(1) && CheckProxyFallback(4);
}

bool CollectTopnResults(TopnRunResult &top1, TopnRunResult &top2, TopnRunResult &top4,
                        TopnRunResult &default_best_top2) {
  TopnRunResult discovery_default;
  if (!InvalidTopnIsRejected() ||
      !RunMeasuredTopn(1, discovery_default, PerfMode::kBlockDim, InvalidatingBatchCallback) ||
      discovery_default.reprs.size() != 1U) {
    return false;
  }
  g_default_repr = discovery_default.reprs.front();
  return RunMeasuredTopn(1, top1, PerfMode::kDefaultWorst) && RunMeasuredTopn(2, top2, PerfMode::kDefaultWorst) &&
         RunMeasuredTopn(4, top4, PerfMode::kDefaultWorst) &&
         RunMeasuredTopn(2, default_best_top2, PerfMode::kDefaultBest);
}

int VerifyTopnResults(const TopnRunResult &top1, const TopnRunResult &top2, const TopnRunResult &top4,
                      const TopnRunResult &default_best_top2) {
  const std::set<std::string> unique_reprs(top4.reprs.begin(), top4.reprs.end());
  const std::set<uint32_t> expected_block_dims = {1U, 2U, 3U, 4U};
  const bool complete_sampling = top1.stats.batch_calls == 1U && top1.stats.batch_calls == top4.stats.batch_calls &&
                                 top1.stats.single_calls == top4.stats.single_calls &&
                                 top1.stats.sampled_candidates == top4.stats.sampled_candidates &&
                                 top1.stats.sampled_block_dims == expected_block_dims &&
                                 top4.stats.sampled_block_dims == expected_block_dims;
  if (!complete_sampling) {
    return 2;
  }
  if (top1.reprs.size() != 1U || top2.reprs.size() != 2U || top4.reprs.size() > 4U ||
      top1.reprs.front() != top2.reprs.front() || top1.reprs.front() != top4.reprs.front() ||
      unique_reprs.size() != top4.reprs.size()) {
    return 3;
  }
  if (top1.measured_candidates.size() != top1.stats.min_perf.size() ||
      top1.measured_candidates.size() != top4.measured_candidates.size() ||
      top1.measured_candidates.size() <= top1.tiling_datas.size()) {
    return 11;
  }
  if (top4.stats.sampled_candidates != top4.stats.min_perf.size()) {
    return 4;
  }
  if (top2.reprs[1] != g_default_repr) {
    return 5;
  }
  if (!IsFallbackPrefixOrder(top4)) {
    return 13;
  }
  if (!IsMeasuredCandidateOrderStable(top4)) {
    return 14;
  }
  const bool default_best_valid =
      default_best_top2.reprs.size() == 1U && default_best_top2.reprs.front() == g_default_repr;
  return default_best_valid ? 0 : 12;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2 || !LoadTiling(argv[1])) {
    return 7;
  }
  TopnRunResult top1;
  TopnRunResult top2;
  TopnRunResult top4;
  TopnRunResult default_best_top2;
  if (!CollectTopnResults(top1, top2, top4, default_best_top2)) {
    return 1;
  }
  const auto verify_ret = VerifyTopnResults(top1, top2, top4, default_best_top2);
  if (verify_ret != 0) {
    return verify_ret;
  }
  if (!VerifyResultParser(top4)) {
    return 8;
  }
  if (!CoreLimitAbovePlatformIsClamped()) {
    return 10;
  }
  if (!VerifyProxyFallbacks()) {
    return 9;
  }
  return FailingCallbackIsRejected() ? 0 : 6;
}
