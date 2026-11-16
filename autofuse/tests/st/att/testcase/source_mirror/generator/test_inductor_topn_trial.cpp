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
#include <gtest/gtest.h>

#include "common_gen_utils.h"
#include "high_perf_tiling_code_gen_impl.h"

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

  std::filesystem::path directory_;
};

class TrialCodeGenerator : public HighPerfTilingCodeGenImpl {
 public:
  using HighPerfTilingCodeGenImpl::HighPerfTilingCodeGenImpl;
  using TilingCodeGenImpl::GenPGOTilingCase;
  using TilingCodeGenImpl::GenSearchAllTilingbyCaseId;

  std::string Source() {
    return tiling_func_.GetOutputStr();
  }
};

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
INSTANTIATE_TEST_SUITE_P(PgoModes, PgoTrialTest, testing::Bool());
}  // namespace
}  // namespace att
