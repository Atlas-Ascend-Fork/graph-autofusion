/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <gtest/gtest.h>

#include "common/final_tiling_selection_snapshot.h"
#include "common_gen_utils.h"
#include "final_tiling_log_gen.h"
#define private public
#include "high_perf_tiling_code_gen_impl.h"
#undef private

namespace att {
namespace {
class PgoTrialTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    std::string path = "/tmp/af_topn_trial_XXXXXX";
    ASSERT_NE(mkdtemp(path.data()), nullptr);
    directory_ = path;
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  void RunGeneratedSource(const std::string &name, const std::string &code) {
    const auto path = directory_.string() + "/" + name;
    std::ofstream source(path + ".cpp");
    ASSERT_TRUE(source.is_open());
    source << code;
    source.close();
    ASSERT_TRUE(source);
    ASSERT_EQ(test::CompileGeneratedTiling("'" + path + ".cpp' -o '" + path + "'"), 0);
    EXPECT_EQ(std::system(("'" + path + "'").c_str()), 0);
  }

  std::filesystem::path directory_;
};

class TrialCodeGenerator : public HighPerfTilingCodeGenImpl {
 public:
  using HighPerfTilingCodeGenImpl::HighPerfTilingCodeGenImpl;
  using TilingCodeGenImpl::GenCopyPgoReuseSelection;
  using TilingCodeGenImpl::GenerateInputParamsAndTiling;
  using TilingCodeGenImpl::GenPGOByCoreNumGetAllSchedulesResults;
  using TilingCodeGenImpl::GenPGOByCoreNumGetScheduleResult;
  using TilingCodeGenImpl::GenPGOCollectWorkspaceSizes;
  using TilingCodeGenImpl::GenPGOFusedScheduleResultsGetTilingDefine;
  using TilingCodeGenImpl::GenPGOGetAllSchedulesResults;
  using TilingCodeGenImpl::GenPGOGetScheduleResultPerGroup;
  using TilingCodeGenImpl::GenPGOTilingCase;
  using TilingCodeGenImpl::GenRestoreReuseTilingKey;
  using TilingCodeGenImpl::GenReuseGroupTilingWrapperGetTiling;
  using TilingCodeGenImpl::GenSearchAllTilingbyCaseId;
  using TilingCodeGenImpl::GenWorkspaceOffsetHelpers;
  using TilingCodeGenImpl::ObtainInnerParams;
  using TilingCodeGenImpl::TryGenPGOByCoreNumReuseTiling;

  void SetSegmentCache(bool enabled) {
    config_.enable_tiling_segment_cache = enabled;
  }

  void SetWorkspaceIds() {
    workspace_tensor_id_set_[0][0] = {3, 4, 7};
  }

  std::string Source() {
    return tiling_func_.GetOutputStr();
  }
};

constexpr char kScheduleBaselinePreamble[] = R"cpp(
#include <array>
#include <cassert>
#include <cfloat>
#include <cstdint>

#define OP_LOGD(...)
#define OP_LOGW(...)

  struct GroupTiling {
    uint32_t key = 2;
    uint32_t tile = 32;
    void set_tiling_key(uint32_t value) {
      key = value;
    }
  };

  struct CaseSolver {
    bool GetTiling(GroupTiling &data) const {
      data.tile = succeeds ? 64 : 0;
      return succeeds;
    }
    bool succeeds = false;
  };
  using TilingCaseImplPtr = CaseSolver *;

  CaseSolver *GetTilingImplPtr(int32_t case_id, uint32_t) {
    static CaseSolver solver;
    solver.succeeds = case_id == 0;
    return &solver;
  }

  [[maybe_unused]] CaseSolver *GetSelectedTilingImplPtr(uint32_t case_id, uint32_t cores, const char *) {
    return GetTilingImplPtr(case_id, cores);
  }

  bool SolveExplicitCase(GroupTiling &tiling_data, int32_t tiling_case_id) {
    const uint32_t corenum = 50;
    [[maybe_unused]] const char *segment_sub_case_tag = nullptr;
    bool ret = false;
    if (tiling_case_id == -1) {
      return true;
)cpp";

constexpr char kScheduleBaselineTypesAndMain[] = R"cpp(
  return ret;
  }

  struct AutofuseTilingData {
    int32_t result = 0;
    GroupTiling g0{1, 0};
    void set_graph0_tiling_key(int32_t value) {
      result = value;
    }
  };

  struct Baseline {
    int32_t result;
    bool operator()(uint32_t, int32_t case_id, AutofuseTilingData &data, double &, double &best, uint32_t &,
                    void * = nullptr) const {
      assert(data.result == result && case_id == -1 && best == -1.0);
      data.g0 = {2, 32U + static_cast<uint32_t>(result)};
      best = 20;
      return result != 2;
    }
  };
  const std::array<Baseline, 3> kScheduleResultFunctions{{{0}, {1}, {2}}};

  namespace AscGraph0 {
  [[maybe_unused]] bool GetTiling(AutofuseTilingData &data, int32_t case_id, void *, void * = nullptr) {
    (void)SolveExplicitCase(data.g0, case_id);
    return true;  // A different schedule result succeeded.
  }
  }  // namespace AscGraph0

  struct Search {
    int32_t result;
    bool operator()(uint32_t &seen, uint32_t, int32_t case_id, const AutofuseTilingData &data, double &, double &best,
                    uint32_t &, void *, uint64_t, void *, void *) const {
      assert(result != 2 && data.result == result && case_id == -1 && best == 10);
      assert(data.g0.key == 2 && data.g0.tile == 32U + static_cast<uint32_t>(result));
      seen |= 1U << result;
      return true;
    }
  };
  const std::array<Search, 3> kScheduleResultFunctionsPGO{{{0}, {1}, {2}}};

  int main() {
    GroupTiling g0;
    assert(!SolveExplicitCase(g0, 1) && g0.key == 2 && g0.tile == 32);
    assert(SolveExplicitCase(g0, 0) && g0.key == 0 && g0.tile == 64);

    uint32_t tiling_data_list = 0;
    const AutofuseTilingData tiling_data{};
    const uint32_t ori_block_dim = 50;
    const int32_t tiling_case_id = -1;
    double cur_perf = 0;
    double best_perf = 10;
    uint32_t cur_block_dim = 1;
    void *stream = nullptr;
    uint64_t workspaceSize = 0;
    void *block_dim_vec = nullptr;
    void *search_cfg = nullptr;
)cpp";

constexpr char kScheduleBaselineAssertions[] = R"cpp(
  assert(has_valid_tiling &&tiling_data_list == 3U);
  }
)cpp";

TEST_P(PgoTrialTest, ScheduleBaselineKeepsGroupCaseAndParametersTogether) {
  TilingCodeGenConfig config{};
  config.is_inductor_scene = true;
  const TilingModelInfo models;
  const ScoreFuncs scores;
  TrialCodeGenerator generator("ScheduleBaseline", config, models, scores, false);
  generator.SetSegmentCache(GetParam());
  ASSERT_EQ(generator.GenerateInputParamsAndTiling(), af::SUCCESS);
  const auto explicit_case = generator.Source();
  generator.GenPGOGetAllSchedulesResults(0, {{0, {}}, {1, {}}, {2, {}}});
  const auto baseline = generator.Source().substr(explicit_case.size());

  std::ostringstream program;
  program << kScheduleBaselinePreamble << explicit_case << kScheduleBaselineTypesAndMain << baseline
          << kScheduleBaselineAssertions;
  RunGeneratedSource("schedule_baseline", program.str());
}

TEST_P(PgoTrialTest, FailedCaseDoesNotPolluteFollowingGroup) {
  TilingCodeGenConfig config{};
  config.is_autofuse = true;
  config.is_inductor_scene = GetParam();
  const TilingModelInfo models;
  const ScoreFuncs scores;
  TrialCodeGenerator generator("TopnTrial", config, models, scores, false);
  ASSERT_EQ(generator.GenSearchAllTilingbyCaseId(), af::SUCCESS);
  const auto search = generator.Source();

  ModelInfo info{};
  ASSERT_EQ(generator.GenPGOTilingCase(info), af::SUCCESS);
  info.tiling_case_id = 1;
  ASSERT_EQ(generator.GenPGOTilingCase(info), af::SUCCESS);
  const auto cases_g0 = generator.Source().substr(search.size());
  const auto g1_start = generator.Source().size();
  info.schedule_group_ident.group_id = 1;
  info.tiling_case_id = 0;
  ASSERT_EQ(generator.GenPGOTilingCase(info), af::SUCCESS);
  const auto cases_g1 = generator.Source().substr(g1_start);

  std::ofstream source(directory_ / "topn_trial.cpp");
  ASSERT_TRUE(source.is_open());
  source << R"cpp(
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>

#define OP_LOGW(...)
#define OP_LOGD(...)

        struct SearchConfig {};
        struct TilingData {
          uint32_t key = 0;
          uint32_t block_dim = 8;
          uint32_t tile = 16;
          void set_tiling_key(uint32_t value) {
            key = value;
          }
        };

        struct AutofuseTilingData {
          TilingData graph0_result0_g0_tiling_data;
          TilingData graph0_result0_g1_tiling_data;
        };

        struct AutofuseTilingDataPerf {
          AutofuseTilingData tiling_data;
        };

        struct TilingCaseImpl {
          uint32_t corenum;
          bool ExecutePGOSolver(TilingData &trial, std::vector<AutofuseTilingDataPerf> &candidates,
                                AutofuseTilingData *snapshot, void *, std::unordered_map<int64_t, uint64_t> &,
                                const std::vector<uint32_t *> &, const SearchConfig *) {
            // Emulate a solver that writes fields before checking its core budget.
            trial.block_dim = trial.key == 0 ? 24 : 150;
            trial.tile = trial.key == 0 ? 32 : 0;
            if (trial.block_dim > corenum) {
              return false;
            }
            auto &group = search_g1 ? snapshot->graph0_result0_g1_tiling_data : snapshot->graph0_result0_g0_tiling_data;
            group = trial;
            candidates.push_back({*snapshot});
            return true;
          }
          static bool search_g1;
        };

        bool TilingCaseImpl::search_g1 = false;
        struct TilingCase0Impl : TilingCaseImpl {
          explicit TilingCase0Impl(uint32_t cores) : TilingCaseImpl{cores} {}
        };
        using TilingCase1Impl = TilingCase0Impl;
      )cpp"
         << search << R"cpp(
        int main() {
          constexpr uint32_t corenum = 50;
          AutofuseTilingData base;
          auto *output_tiling_data = &base;
          AutofuseTilingData autofuse_tiling_data_tmp;
          std::vector<AutofuseTilingDataPerf> tiling_data_list;
          std::unordered_map<int64_t, uint64_t> workspace_map;
          std::vector<uint32_t *> block_dim_vec;
          const SearchConfig *effective_cfg = nullptr;
          void *stream = nullptr;
          alignas(TilingCase0Impl) unsigned char memory[sizeof(TilingCase0Impl)];
          TilingCaseImpl *tilingCaseImplPtr;
          bool ret = false;
          {
            [[maybe_unused]] auto &tiling_data = base.graph0_result0_g0_tiling_data;
      )cpp"
         << cases_g0 << R"cpp(
        }
        assert(ret &&tiling_data_list.size() == 1);
        const auto &g0_candidate = tiling_data_list[0].tiling_data.graph0_result0_g0_tiling_data;
        assert(g0_candidate.key == 0 && g0_candidate.block_dim == 24 && g0_candidate.tile == 32);
        // Keep the previous best as the base when the new G0 candidate does not beat it.
        TilingCaseImpl::search_g1 = true;
        ret = false;
        {
          [[maybe_unused]] auto &tiling_data = base.graph0_result0_g1_tiling_data;
      )cpp"
         << cases_g1 << R"cpp(
    }
    assert(ret &&tiling_data_list.size() == 2);
    const auto &inherited = tiling_data_list[1].tiling_data.graph0_result0_g0_tiling_data;
    const auto &g1 = tiling_data_list[1].tiling_data.graph0_result0_g1_tiling_data;
    assert(inherited.key == 0 && inherited.block_dim == 8 && inherited.tile == 16);
    assert(g1.key == 0 && g1.block_dim == 24 && g1.tile == 32);
    }
  )cpp";
  source.close();
  ASSERT_TRUE(source);
  const auto path = directory_.string();
  ASSERT_EQ(test::CompileGeneratedTiling("'" + path + "/topn_trial.cpp' -o '" + path + "/topn_trial'"), 0);
  EXPECT_EQ(std::system(("'" + path + "/topn_trial'").c_str()), 0);
}

TEST_P(PgoTrialTest, WorkspaceSubCasesMustShareExpressions) {
  TilingCodeGenConfig config{};
  config.is_inductor_scene = GetParam();
  config.enable_autofuse_pgo = !GetParam();
  config.tiling_data_type_name = "AutofuseTilingData";
  ModelInfo base{};
  const auto tile_size = af::Symbol("tile_size");
  base.workspace_size_map[3] = tile_size;
  ModelInfo sub_case = base;
  sub_case.sub_case_tag = "R";
  const ScoreFuncs scores;
  const auto generate = [&](const ModelInfo &other) {
    const TilingModelInfo models{base, other};
    TrialCodeGenerator generator("TopnWorkspaceSubCases", config, models, scores, false);
    generator.SetWorkspaceIds();
    return generator.GenPGOCollectWorkspaceSizes(0, 0);
  };

  EXPECT_EQ(generate(sub_case), af::SUCCESS);
  sub_case.workspace_size_map[3] = tile_size * af::Symbol(2);
  EXPECT_NE(generate(sub_case), af::SUCCESS);
  sub_case.workspace_size_map = base.workspace_size_map;
  sub_case.variable_expr_map.emplace(tile_size, af::Symbol("other_tile_size"));
  EXPECT_NE(generate(sub_case), af::SUCCESS);
}

constexpr char kWorkspaceIncludes[] = R"cpp(
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
)cpp";

constexpr char kWorkspaceTypes[] = R"cpp(
#define OP_LOGE(...) (void)0
#define OP_LOGI(...) (void)0
#define OP_LOGW(...) (void)0
  using std::max;
  template <typename T>
  T Max(T a, T b) {
    return std::max(a, b);
  }
  struct GroupTiling {
    uint32_t key;
    uint32_t tile_size;
    uint32_t get_tiling_key() const {
      return key;
    }
    void set_tiling_key(uint32_t value) {
      key = value;
    }
    uint32_t get_tile_size() const {
      return tile_size;
    }
    uint32_t get_block_dim() const {
      return 8;
    }
  };
  struct AutofuseTilingData {
    GroupTiling graph0_result0_g0_tiling_data;
    GroupTiling graph0_result0_g1_tiling_data;
    uint64_t workspace3 = 0;
    uint64_t workspace4 = 0;
    uint64_t workspace7 = 0;
    uint64_t capacity = 128;
    uint64_t get_workspace3() const {
      return workspace3;
    }
    uint64_t get_workspace4() const {
      return workspace4;
    }
    uint64_t get_workspace7() const {
      return workspace7;
    }
    void set_workspace3(uint64_t value) {
      workspace3 = value;
    }
    void set_workspace4(uint64_t value) {
      workspace4 = value;
    }
    void set_workspace7(uint64_t value) {
      workspace7 = value;
    }
    uint32_t get_block_dim() const {
      return 8;
    }
    void set_block_dim(uint32_t) {}
  };
  struct AutofuseTilingDataPerf {
    AutofuseTilingData tiling_data;
    double best_perf;
    std::vector<FinalTilingSelectionSnapshot> final_tiling_selections;
  };
  using AscGraph0ScheduleResult0G0TilingData = GroupTiling;
  using AscGraph0ScheduleResult0G1TilingData = GroupTiling;
  template <typename From, typename To>
  To RefToRef(const From &value) {
    return value;
  }
  bool GetWorkspaceSize(const AutofuseTilingData &tiling, uint64_t &size) {
    size = tiling.capacity;
    return true;
  }
  void RestoreFinalTilingSelectionSnapshots(const std::vector<FinalTilingSelectionSnapshot> &) {}
  std::vector<FinalTilingSelectionSnapshot> GetFinalTilingSelectionSnapshots() {
    return {};
  }
  struct PgoTensorArgs {};
  struct SearchConfig {};
  struct PgoConfig {
    static PgoConfig &Instance() {
      static PgoConfig config;
      return config;
    }
    PgoTensorArgs *tensor_args = nullptr;
    int (*batch_callback)(PgoTensorArgs *, void *, uint64_t, std::vector<AutofuseTilingDataPerf> *) = nullptr;
  };
)cpp";

constexpr char kWorkspaceAssembleEntry[] = R"cpp(
  bool Assemble(std::vector<AutofuseTilingDataPerf> &tiling_data_list) {
    AutofuseTilingData tiling_data{};
    double best_perf = 100;
    void *stream = nullptr;
    uint64_t workspaceSize = 0;
    size_t candidate_begin_index1 = 0;
    std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{{3, 4096}, {4, 4096}, {7, 8192}};
    std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{{{{0, 16}, {0, 4}}, 1, {}}, {{{0, 4}, {1, 8}}, 2, {}}};
)cpp";

constexpr char kWorkspaceStubGraph[] = R"cpp(
  return true;
  }
  namespace AscGraph0 {
  bool GetPgoTilingSeed(AutofuseTilingData &data) {
    data.workspace3 = 96;
    data.workspace4 = 32;
    return true;
  }
  bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf> &candidates, AutofuseTilingData &data, int,
                          AutofuseTilingData *, void *, uint64_t, double &perf, std::vector<uint32_t *>,
                          const SearchConfig *) {
    auto candidate = data;
    candidate.capacity = 128;
    candidates.push_back({candidate, 1.0, {}});
    perf = 1.0;
    return true;
  }
  }
)cpp";

constexpr char kWorkspaceReuseNamespace[] = R"cpp(
  namespace AscGraph0ScheduleResult0G1 {
)cpp";

constexpr char kWorkspaceReuseLoop[] = R"cpp(
  }
  std::vector<AutofuseTilingDataPerf> ReuseByCore(std::vector<AutofuseTilingDataPerf> inputs) {
    std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp2;
    for (auto &candidate : inputs) {
      auto &tiling_data = candidate.tiling_data;
)cpp";

constexpr char kWorkspaceAssertions[] = R"cpp(
  }
  return tiling_data_list_tmp2;
  }
  int main() {
    std::vector<AutofuseTilingDataPerf> candidates;
    assert(Assemble(candidates));
    assert(candidates.size() == 2);
    auto &a = candidates[0].tiling_data;
    auto &b = candidates[1].tiling_data;
    assert(a.workspace3 == 64 && a.workspace4 == 32 && a.workspace7 == 0);
    assert(b.workspace3 == 16 && b.workspace4 == 16 && b.workspace7 == 0);
    auto undersized = a;
    undersized.capacity = 95;
    assert(!FinalizePGOWorkspaceOffsets(undersized));
    assert(FinalizePGOWorkspaceOffsets(a) && FinalizePGOWorkspaceOffsets(b));
    assert(a.workspace3 == 0 && a.workspace4 == 64);
    assert(b.workspace3 == 0 && b.workspace4 == 16);
    std::unordered_map<int64_t, uint64_t> workspace_map;
    auto key_check = a;
    key_check.graph0_result0_g1_tiling_data.key = 2;
    assert(CollectPGOWorkspaceSizesForResult0(key_check, workspace_map));
    assert(workspace_map.at(3) == 64 && workspace_map.count(4) == 0);
    key_check.graph0_result0_g1_tiling_data.key = 99;
    assert(!CollectPGOWorkspaceSizesForResult0(key_check, workspace_map));

    auto primary = candidates.front();
    primary.final_tiling_selections = {{true, 0, 0, 0, 0, 0, 1}, {}};
    auto unmapped = primary;
    unmapped.tiling_data.graph0_result0_g0_tiling_data.key = 1;
    auto reused = ReuseByCore({primary, unmapped});
    assert(reused.size() == 1);
    const auto &reuse_tiling = reused[0].tiling_data.graph0_result0_g1_tiling_data;
    assert(reuse_tiling.key == 8 && reuse_tiling.tile_size == 16);
    auto &snapshot = reused[0].final_tiling_selections[1];
    assert(snapshot.valid && snapshot.group == 1 && snapshot.tiling_key == 8 && snapshot.case_id == 8 &&
           snapshot.sub_case_id == 1);
    assert(CollectPGOWorkspaceSizesForResult0(reused[0].tiling_data, workspace_map));
    assert(workspace_map.at(3) == 64 && workspace_map.at(4) == 32 && workspace_map.count(7) == 0);
    primary.final_tiling_selections.clear();
    reused = ReuseByCore({primary});
    assert(reused.size() == 1 && reused[0].tiling_data.graph0_result0_g1_tiling_data.key == 8);

    AutofuseTilingData final_data{};
    final_data.capacity = 127;
    double best_perf = DBL_MAX;
    std::vector<AutofuseTilingDataPerf> root_candidates;
    assert(!PGOSearchTilingKey(root_candidates, final_data, -1, &final_data, nullptr, nullptr, 0, best_perf, nullptr));
    assert(root_candidates.size() == 1 && root_candidates[0].tiling_data.workspace4 == 96);
    final_data = {};
    best_perf = DBL_MAX;
    assert(PGOSearchTilingKey(root_candidates, final_data, -1, &final_data, nullptr, nullptr, 0, best_perf, nullptr));
    assert(final_data.workspace3 == 0 && final_data.workspace4 == 96);
  }
)cpp";

TilingModelInfo MakeWorkspaceModels() {
  ModelInfo g0{};
  g0.workspace_size_map[3] = af::Symbol("tile_size") * af::Symbol(4);
  ModelInfo g0_inactive = g0;
  g0_inactive.tiling_case_id = 1;
  g0_inactive.workspace_size_map = {{7, af::Symbol(512)}};
  ModelInfo g1{};
  g1.schedule_group_ident.group_id = 1;
  g1.workspace_size_map = {{3, af::Symbol(8)}, {4, af::Symbol("tile_size") * af::Symbol(8)}};
  ModelInfo g1_alternative = g1;
  g1_alternative.tiling_case_id = 1;
  g1_alternative.workspace_size_map = {{4, af::Symbol("tile_size") * af::Symbol(2)}};
  ModelInfo g1_without_workspace = g1;
  g1_without_workspace.tiling_case_id = 2;
  g1_without_workspace.workspace_size_map.clear();
  ModelInfo g1_reuse = g1_alternative;
  g1_reuse.tiling_case_id = 8;
  g1_reuse.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  const auto &reuse = g1_reuse.reuse_schedule_group;
  reuse->reuse_group_ident = g0.schedule_group_ident;
  reuse->info.tiling_keys = {0, 0};
  auto &reuse_info = reuse->schedule_group_to_info[g1_reuse.schedule_group_ident];
  reuse_info.tiling_keys = {8, 8};
  return {g0, g0_inactive, g1, g1_alternative, g1_without_workspace, g1_reuse};
}

TEST_P(PgoTrialTest, LastGroupCandidatesRebuildWorkspaceLayout) {
  TilingCodeGenConfig config{};
  config.is_autofuse = true;
  config.is_inductor_scene = GetParam();
  config.enable_autofuse_pgo = !GetParam();
  config.tiling_data_type_name = "AutofuseTilingData";

  const auto models = MakeWorkspaceModels();
  const auto &g1_reuse = models.back();
  const auto &reuse = g1_reuse.reuse_schedule_group;
  const auto &reuse_info = reuse->schedule_group_to_info.at(g1_reuse.schedule_group_ident);
  const ScoreFuncs scores;
  TrialCodeGenerator generator("TopnWorkspace", config, models, scores, false);
  generator.SetWorkspaceIds();
  generator.GenWorkspaceOffsetHelpers();
  ASSERT_EQ(generator.GenPGOCollectWorkspaceSizes(0, 0), af::SUCCESS);
  auto helpers = generator.Source();
  const std::map<size_t, std::pair<std::string, std::string>> groups = {
      {0, {"AscGraph0ScheduleResult0G0", "graph0_result0_g0"}},
      {1, {"AscGraph0ScheduleResult0G1", "graph0_result0_g1"}}};
  ASSERT_EQ(generator.GenPGOGetScheduleResultPerGroup(0, 0, groups, *groups.find(1), {}), af::SUCCESS);
  auto assemble = generator.Source().substr(helpers.size());
  ASSERT_EQ(generator.GenPGOFusedScheduleResultsGetTilingDefine({{0, {}}}), af::SUCCESS);
  auto finalize = generator.Source().substr(helpers.size() + assemble.size());
  ASSERT_TRUE(generator.TryGenPGOByCoreNumReuseTiling(0, 0, 1, 2));
  auto reuse_by_core = generator.Source().substr(helpers.size() + assemble.size() + finalize.size());

  auto reuse_config = config;
  reuse_config.tiling_data_type_name = "GroupTiling";
  const TilingModelInfo reuse_models{g1_reuse};
  TrialCodeGenerator reuse_generator("TopnWorkspaceReuse", reuse_config, reuse_models, scores, false);
  reuse_generator.SetFinalTilingObservationMetadata(1, 2, {{"R", 1}});
  ASSERT_EQ(reuse_generator.GenRestoreReuseTilingKey(reuse->info, reuse_info), af::SUCCESS);
  reuse_generator.GenCopyPgoReuseSelection();

  std::ostringstream program;
  program << kWorkspaceIncludes << ascgen_utils::GenFinalTilingSelectionSnapshotSnippet() << kWorkspaceTypes << helpers
          << kWorkspaceAssembleEntry << assemble << kWorkspaceStubGraph << finalize << kWorkspaceReuseNamespace
          << reuse_generator.Source() << kWorkspaceReuseLoop << reuse_by_core << kWorkspaceAssertions;
  RunGeneratedSource("topn_workspace", program.str());
}

constexpr char kByCoreReuseIncludes[] = R"cpp(
#include <cassert>
#include <cfloat>
#include <unordered_map>
#define OP_LOGE(...) (void)0
#define ATT_FINAL_TILING_LOGI(...) (void)0
#define CheckCachedLogLevel(...) true
)cpp";

constexpr char kByCoreReuseTypesAndPrimary[] = R"cpp(
  struct GroupTiling {
    uint32_t key = 0;
    uint32_t tile = 0;
    uint32_t shape = 0;
    uint32_t blocks = 0;
    uint32_t get_tiling_key() const {
      return key;
    }
    uint32_t get_shape_size() const {
      return shape;
    }
    uint32_t get_block_dim() const {
      return blocks;
    }
    void set_tiling_key(uint32_t value) {
      key = value;
    }
    void set_shape_size(uint32_t value) {
      shape = value;
    }
    void set_block_dim(uint32_t value) {
      blocks = value;
    }
  };
  using AscGraph0ScheduleResult0G0TilingData = GroupTiling;
  using AscGraph0ScheduleResult1G0TilingData = GroupTiling;
  using AscGraph1ScheduleResult0G0TilingData = GroupTiling;
  template <typename From, typename To>
  To RefToRef(const From &value) {
    return value;
  }
  struct AutofuseTilingData {
    GroupTiling graph0_result0_g0_tiling_data;
    GroupTiling CURRENT_FIELD;
    uint32_t blocks = 3;
    uint32_t get_block_dim() const {
      return blocks;
    }
    void set_block_dim(uint32_t value) {
      blocks = value;
    }
    void set_graph0_tiling_key(uint32_t) {}
    void set_graph1_tiling_key(uint32_t) {}
  };
  struct AutofuseTilingDataPerf {
    AutofuseTilingData tiling_data;
    double best_perf;
    std::vector<FinalTilingSelectionSnapshot> final_tiling_selections;
  };
  struct GroupLevelCache;
  namespace AscGraph0ScheduleResult0G0 {
  bool GetTiling(GroupTiling &data, std::unordered_map<int64_t, uint64_t> &, int32_t key, GroupLevelCache *) {
    assert(data.key == 7 && key == 7 && data.blocks == 3 && data.tile == 0);
    if (data.shape == 0) {
      return false;
    }
    data.tile = data.shape + data.blocks;
    FinalTilingObservationSlotGuard slot(0);
    CommitCurrentFinalTilingSelection({true, 0, 0, 0, 7, 7, 1});
    return true;
  }
  }
  namespace CURRENT_GROUP {
)cpp";

constexpr char kByCoreReuseSearchEntry[] = R"cpp(
  const auto kScheduleResultFunctionsPGOByCoreNum = std::vector{&GetScheduleResult0PGOByCoreNum
#if !CROSS_GRAPH
                                                                ,
                                                                &GetScheduleResult1PGOByCoreNum
#endif
  };
  bool Search(std::vector<AutofuseTilingDataPerf> &tiling_data_list, AutofuseTilingData tiling_data) {
)cpp";

constexpr char kByCoreReuseAssertions[] = R"cpp(
  return has_valid_tiling;
  }
  int main() {
    FinalTilingObservationContext context;
    FinalTilingObservationCallGuard observation(context, true, 2);
    AutofuseTilingData base;
    base.graph0_result0_g0_tiling_data = {7, 999, 12, 33};
    base.CURRENT_FIELD = {0, 999, 47, 33};
    std::vector<AutofuseTilingDataPerf> candidates;
    assert(Search(candidates, base) && candidates.size() == (CROSS_GRAPH ? 1 : 2));
    const auto &solved = candidates.back();
    assert(solved.tiling_data.CURRENT_FIELD.key == 8 && solved.tiling_data.CURRENT_FIELD.tile == 50);
    assert(solved.tiling_data.CURRENT_FIELD.blocks == 3);
    const auto &selection = solved.final_tiling_selections.at(1);
    assert(selection.valid && selection.graph == (CROSS_GRAPH ? 1 : 0) && selection.result == (CROSS_GRAPH ? 0 : 1) &&
           selection.group == 0);
    assert(selection.tiling_key == 8 && selection.case_id == 8 && selection.sub_case_id == 1);
    base.CURRENT_FIELD.shape = 0;
    candidates.clear();
    assert(Search(candidates, base) == !CROSS_GRAPH && candidates.size() == (CROSS_GRAPH ? 0 : 1));
  }
)cpp";

TilingModelInfo MakeCrossScopeReuseModels(bool cross_graph) {
  ModelInfo primary{};
  primary.tiling_case_id = 7;
  auto input = std::make_shared<AttAxis>();
  input->name = "shape";
  input->axis_pos = AxisPosition::ORIGIN;
  input->size = std::make_shared<SymVarInfo>(af::Symbol("shape_size"));
  primary.arg_list = {input};
  ModelInfo current = primary;
  current.schedule_group_ident = cross_graph ? ScheduleGroupIdent{1, 0, 0} : ScheduleGroupIdent{0, 1, 0};
  current.tiling_case_id = 8;
  auto reuse = std::make_shared<ReuseScheduleGroup>();
  reuse->reuse_group_ident = primary.schedule_group_ident;
  reuse->info.reuse_input_axes = {"shape_size"};
  reuse->info.tiling_keys = {7};
  auto &current_info = reuse->schedule_group_to_info[current.schedule_group_ident];
  current_info.reuse_input_axes = {"shape_size"};
  current_info.tiling_keys = {8};
  primary.reuse_schedule_group = reuse;
  current.reuse_schedule_group = reuse;
  return {primary, current};
}

TEST_P(PgoTrialTest, ByCoreNumSolvesReuseGroupOutsidePrimaryResult) {
  const bool cross_graph = GetParam();
  const auto models = MakeCrossScopeReuseModels(cross_graph);
  const auto &current = models.back();
  const auto &reuse = current.reuse_schedule_group;

  TilingCodeGenConfig config{};
  config.is_autofuse = true;
  config.enable_autofuse_pgo = true;
  config.tiling_data_type_name = "AutofuseTilingData";
  const ScoreFuncs scores;
  TrialCodeGenerator generator("ByCoreReuse", config, models, scores, false);
  using Groups = std::map<size_t, std::pair<std::string, std::string>>;
  std::map<size_t, std::map<size_t, Groups>> namespaces;
  std::map<std::string, std::set<std::string>> hardware;
  std::map<std::string, std::set<std::string>> input_vars;
  ASSERT_EQ(generator.ObtainInnerParams(hardware, namespaces, &input_vars), af::SUCCESS);
  const auto &ident = current.schedule_group_ident;
  const auto current_prefix = ident.GetItemPrefix();
  ASSERT_EQ(input_vars.at(current_prefix), std::set<std::string>{"shape_size"});
  const auto &results = namespaces.at(ident.asc_graph_id);
  for (const auto &[result_id, groups] : results) {
    hardware[groups.at(0).second].insert("block_dim");
    generator.GenPGOByCoreNumGetScheduleResult(ident.asc_graph_id, result_id, groups, hardware, {});
  }
  const auto solve = generator.Source();
  generator.GenPGOByCoreNumGetAllSchedulesResults(ident.asc_graph_id, results, input_vars);
  const auto search = generator.Source().substr(solve.size());

  config.tiling_data_type_name = "GroupTiling";
  const TilingModelInfo current_models{current};
  TrialCodeGenerator wrapper("ByCoreReuseWrapper", config, current_models, scores, false);
  wrapper.SetFinalTilingObservationMetadata(1, 2, {{"R", 1}});
  ASSERT_EQ(wrapper.GenReuseGroupTilingWrapperGetTiling(ident.GetGroupPrefix(), "AscGraph0ScheduleResult0G0",
                                                        reuse->info, reuse->schedule_group_to_info.cbegin()),
            af::SUCCESS);
  wrapper.GenCopyPgoReuseSelection();

  std::ostringstream program;
  program << "#define CURRENT_GROUP " << ident.GetGroupPrefix() << "\n#define CURRENT_FIELD " << current_prefix
          << "_tiling_data\n#define CROSS_GRAPH " << cross_graph << "\n";
  program << kByCoreReuseIncludes << final_tiling::GenFinalTilingLogHelpers(false) << kByCoreReuseTypesAndPrimary
          << wrapper.Source() << "}\n"
          << solve << kByCoreReuseSearchEntry << search << kByCoreReuseAssertions;
  RunGeneratedSource("by_core_reuse", program.str());
}

INSTANTIATE_TEST_SUITE_P(PgoModes, PgoTrialTest, testing::Bool());
}  // namespace
}  // namespace att
