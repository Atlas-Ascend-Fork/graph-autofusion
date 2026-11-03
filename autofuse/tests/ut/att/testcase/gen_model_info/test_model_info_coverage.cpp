/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <unistd.h>
#include <gtest/gtest.h>
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "graph_construct_utils.h"
#include "nlohmann/json.hpp"
#include "schedule_result.h"
#include "test_fa_ascir_graph.h"
#include "util/thread_local_context.h"

namespace af {
namespace ascir {
namespace cg {
Status BuildWorkSpaceAscendGraph(af::AscGraph &graph);
}  // namespace cg
}  // namespace ascir
}  // namespace af

namespace att {
void to_json(nlohmann::json &json, const ATTConfig &config);
std::string GetRealPath(const std::string &path);
void DumpModelInfo(const std::vector<ModelInfo> &models, std::string dump_dir);

namespace {
af::AscGraph MakeScheduledGraph(const char *name) {
  af::AscGraph graph(name);
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return graph;
}

ascir::FusedScheduledResult ScheduleGraphs(const std::vector<af::AscGraph> &graphs) {
  ascir::ScheduleGroup group;
  group.impl_graphs = graphs;
  ascir::ScheduledResult result;
  result.schedule_groups.push_back(group);
  ascir::FusedScheduledResult fused;
  fused.node_idx_to_scheduled_results = {{result}};
  return fused;
}

std::string ReadText(const std::string &path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class ModelInfoCoverageTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_context_ = GetThreadLocalContext();
    previous_strategy_ = AutoFuseConfig::GetAttStrategyConfig();
    previous_pgo_strategy_ = AutoFuseConfig::GetPgoStrategyConfig();
    for (const auto *key : {"AUTOFUSE_FLAGS", "AUTOFUSE_DFX_FLAGS"}) {
      const auto *value = std::getenv(key);
      if (value != nullptr) {
        previous_environment_[key] = value;
      }
      unsetenv(key);
    }
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig{};
    char directory[] = "./att-model-coverage-XXXXXX";
    const auto created = mkdtemp(directory);
    ASSERT_NE(created, nullptr);
    directory_ = created;
    GetThreadLocalContext().SetOption({});
  }

  void TearDown() override {
    GetThreadLocalContext() = previous_context_;
    AutoFuseConfig::MutableAttStrategyConfig() = previous_strategy_;
    AutoFuseConfig::MutablePgoStrategyConfig() = previous_pgo_strategy_;
    for (const auto *key : {"AUTOFUSE_FLAGS", "AUTOFUSE_DFX_FLAGS"}) {
      const auto previous = previous_environment_.find(key);
      if (previous == previous_environment_.end()) {
        unsetenv(key);
      } else {
        setenv(key, previous->second.c_str(), 1);
      }
    }
    if (!directory_.empty()) {
      std::remove((directory_ + "/model_info.json").c_str());
      std::remove((directory_ + "/tuning_space.json").c_str());
      EXPECT_EQ(rmdir(directory_.c_str()), 0);
    }
  }

  void CreateDumpFiles() const {
    std::ofstream(directory_ + "/model_info.json") << "stale model";
    std::ofstream(directory_ + "/tuning_space.json") << "stale tuning";
  }

  std::string directory_;
  ThreadLocalContext previous_context_;
  AttStrategyConfig previous_strategy_;
  PgoStrategyConfig previous_pgo_strategy_;
  std::map<std::string, std::string> previous_environment_;
};

TEST_F(ModelInfoCoverageTest, SerializesConfigurationNamesAndAlignmentValues) {
  ATTConfig config;
  config.config_names = {"input0", "input1"};
  config.config_value = {{"input0", "32"}, {"input1", "64"}};
  nlohmann::json json;
  to_json(json, config);
  EXPECT_EQ(json.at("config_inputs"), nlohmann::json(config.config_names));
  EXPECT_EQ(json.at("align_config").at("input0"), "32");
  EXPECT_EQ(json.at("align_config").at("input1"), "64");
  config = {};
  to_json(json, config);
  EXPECT_TRUE(json.at("config_inputs").empty());
  EXPECT_TRUE(json.at("align_config").empty());
}

TEST_F(ModelInfoCoverageTest, RejectsEmptyAndOverlongPathsAndNormalizesExistingDirectory) {
  EXPECT_TRUE(GetRealPath("").empty());
  EXPECT_TRUE(GetRealPath(std::string(4096, 'x')).empty());
  EXPECT_TRUE(GetRealPath(directory_ + "/missing").empty());
  const auto resolved = GetRealPath(directory_);
  ASSERT_FALSE(resolved.empty());
  EXPECT_EQ(resolved.back(), '/');
  EXPECT_EQ(GetRealPath(directory_ + "/"), resolved);
}

TEST_F(ModelInfoCoverageTest, DebugDumpReplacesStaleFilesWithModelAndParsedGraphData) {
  CreateDumpFiles();
  auto graph = MakeScheduledGraph("coverage_dump");
  std::vector<ModelInfo> models;
  const std::map<std::string, std::string> options{{kDumpDebugInfo, directory_}};
  ASSERT_EQ(GenerateModelInfo({graph}, models, options), af::SUCCESS);
  ASSERT_FALSE(models.empty());
  const auto serialized = nlohmann::json::parse(ReadText(directory_ + "/model_info.json"));
  ASSERT_EQ(serialized.at("model_info").size(), models.size());
  EXPECT_EQ(serialized.at("model_info")[0].at("tiling_case_id"), models[0].tiling_case_id);
  EXPECT_FALSE(serialized.at("model_info")[0].at("arg_list").empty());
  const auto tuning_dump = ReadText(directory_ + "/tuning_space.json");
  EXPECT_EQ(tuning_dump.find("stale tuning"), std::string::npos);
  for (const auto &section :
       {"TuningSpace{", "sub_axes:", "containers:", "node_infos:", "tensors:", "block_dim:", "related_scopes:"}) {
    EXPECT_NE(tuning_dump.find(section), std::string::npos) << section;
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Load" || node->GetType() == "Store") {
      EXPECT_NE(tuning_dump.find(node->GetName()), std::string::npos);
    }
  }
  std::vector<ModelInfo> replacement(1);
  replacement.front().tiling_case_id = 71;
  DumpModelInfo(replacement, directory_ + "/");
  const auto replaced = nlohmann::json::parse(ReadText(directory_ + "/model_info.json"));
  ASSERT_EQ(replaced.at("model_info").size(), 1U);
  EXPECT_EQ(replaced.at("model_info")[0].at("tiling_case_id"), 71);
}

TEST_F(ModelInfoCoverageTest, OutputAliasRemovesOnlyItsWorkspaceAllocation) {
  af::AscGraph graph("coverage_alias");
  ASSERT_EQ(af::ascir::cg::BuildWorkSpaceAscendGraph(graph), af::SUCCESS);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  int64_t tensor_id = 40;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Workspace") {
      node->outputs[0].attr.mem.tensor_id = tensor_id++;
    }
  }
  auto output = graph.FindNode("output1");
  ASSERT_NE(output, nullptr);
  const auto aliased_id = output->inputs[0].attr.mem.tensor_id;
  auto schedules = ScheduleGraphs({graph});
  FusedParsedScheduleResult before;
  ASSERT_EQ(GetModelInfoMap(schedules, {}, before), af::SUCCESS);
  ASSERT_FALSE(before[0][0].groups_tiling_model_info[0].empty());
  const auto original = before[0][0].groups_tiling_model_info[0][0].workspace_size_map;
  ASSERT_NE(original.find(aliased_id), original.end());
  ASSERT_GT(original.size(), 1U);
  // Duplicate outputs must be idempotent; unrelated and null entries are ignored.
  schedules.output_nodes = {nullptr, graph.FindNode("workspace1"), output, output};
  FusedParsedScheduleResult after;
  ASSERT_EQ(GetModelInfoMap(schedules, {}, after), af::SUCCESS);
  ASSERT_EQ(after[0][0].groups_tiling_model_info[0].size(), before[0][0].groups_tiling_model_info[0].size());
  for (const auto &model : after[0][0].groups_tiling_model_info[0]) {
    auto expected = original;
    expected.erase(aliased_id);
    EXPECT_EQ(model.workspace_size_map, expected);
    EXPECT_EQ(model.output_nodes, schedules.output_nodes);
  }
}

TEST_F(ModelInfoCoverageTest, SkipsEmptyScheduleAndRejectsInconsistentExplicitTilingKeys) {
  auto graph = MakeScheduledGraph("coverage_schedule");
  auto schedules = ScheduleGraphs({graph});
  schedules.node_idx_to_scheduled_results[0][0].schedule_groups[0].graph_name_to_score_funcs[graph.GetName()] =
      "CoverageScheduleScore";
  schedules.node_idx_to_scheduled_results[0].insert(schedules.node_idx_to_scheduled_results[0].begin(),
                                                    ascir::ScheduledResult{});
  FusedParsedScheduleResult models;
  ASSERT_EQ(GetModelInfoMap(schedules, {}, models), af::SUCCESS);
  ASSERT_EQ(models.size(), 1U);
  EXPECT_EQ(models[0].count(0), 0U);
  ASSERT_EQ(models[0].count(1), 1U);
  EXPECT_FALSE(models[0][1].groups_tiling_model_info[0].empty());
  for (const auto &model : models[0][1].groups_tiling_model_info[0]) {
    EXPECT_EQ(model.score_func, "CoverageScheduleScore");
  }
  auto keyed = MakeScheduledGraph("keyed");
  keyed.SetTilingKey(7);
  auto unkeyed = MakeScheduledGraph("unkeyed");
  auto inconsistent = ScheduleGraphs({keyed, unkeyed});
  FusedParsedScheduleResult failed_models;
  EXPECT_NE(GetModelInfoMap(inconsistent, {}, failed_models), af::SUCCESS);
  EXPECT_TRUE(failed_models[0][0].groups_tiling_model_info[0].empty());
}
}  // namespace
}  // namespace att
