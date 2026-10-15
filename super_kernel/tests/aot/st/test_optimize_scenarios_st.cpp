/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "st_fixture.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::ordered_json;

class TemporaryWorkDirectory {
 public:
  TemporaryWorkDirectory() : previous_(std::filesystem::current_path()) {
    char pattern[] = "/tmp/sk-aot-st-XXXXXX";
    const char *directory = mkdtemp(pattern);
    if (directory == nullptr) {
      throw std::runtime_error("cannot create ST temporary directory");
    }
    path_ = directory;
    std::filesystem::current_path(path_);
  }
  ~TemporaryWorkDirectory() {
    std::error_code error;
    std::filesystem::current_path(previous_, error);
    std::filesystem::remove_all(path_, error);
  }

 private:
  std::filesystem::path previous_;
  std::filesystem::path path_;
};

Json ReadJson(const std::filesystem::path &path) {
  std::ifstream input(path);
  return Json::parse(input);
}

size_t EnabledEntries(const sk::test::Model &model, aclrtStream stream) {
  const auto tasks = model.Tasks(stream);
  return std::count_if(tasks.begin(), tasks.end(), [](const auto &task) {
    return !task.disabled && task.type == ACL_MODEL_RI_TASK_KERNEL && task.setParamsCount != 0;
  });
}
}  // namespace

TEST_F(AotSystemTest, OptimizeSelectsEntryForCubeAndBothMixRatios) {
  struct Case {
    sk::test::KernelSpec spec;
    const char *entry;
  };
  for (const auto &item : {Case{{ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0}, "sk_entry_aic"},
                           Case{{ACL_KERNEL_TYPE_MIX, 4, 1, 1, 0}, "sk_entry_mix11"},
                           Case{{ACL_KERNEL_TYPE_MIX, 4, 1, 2, 0}, "sk_entry_mix12"}}) {
    SCOPED_TRACE(item.entry);
    sk::test::Model model;
    auto stream = model.AddStream();
    auto first = model.AddKernel(stream, "typed_first", item.spec);
    auto second = model.AddKernel(stream, "typed_second", item.spec);
    ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
    EXPECT_EQ(EnabledEntries(model, stream), 1U);
    const auto tasks = model.Tasks(stream);
    auto entry = std::find_if(tasks.begin(), tasks.end(), [](const auto &task) { return !task.disabled; });
    ASSERT_NE(entry, tasks.end());
    EXPECT_EQ(entry->function, item.entry);
    EXPECT_EQ(entry->numBlocks, 4U);
    EXPECT_FALSE(entry->args.empty());
    EXPECT_NE(model.Snapshot(first).disabled, model.Snapshot(second).disabled);
    EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  }
}

TEST_F(AotSystemTest, OptimizeDebugPerOpKeepsSeparateEntries) {
  sk::test::Model model;
  auto stream = model.AddStream();
  model.AddKernel(stream, "debug_first");
  model.AddKernel(stream, "debug_second");
  aclskOption option{};
  option.optionType = aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM;
  option.debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 1;
  aclskOptions options{&option, 1};
  ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  EXPECT_EQ(EnabledEntries(model, stream), 2U);
  for (const auto &task : model.Tasks(stream)) {
    EXPECT_EQ(task.function, "sk_entry_mix12_op_trace");
    EXPECT_FALSE(task.args.empty());
  }
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
}

TEST_F(AotSystemTest, OptimizeSharedScopeAcrossStreamsUsesSingleEntry) {
  for (uint32_t enabled : {0U, 1U}) {
    SCOPED_TRACE(enabled);
    sk::test::Model model;
    auto first = model.AddStream();
    auto second = model.AddStream();
    ASSERT_EQ(aclskScopeBegin("shared", first), ACL_SUCCESS);
    ASSERT_EQ(aclskScopeBegin("shared", second), ACL_SUCCESS);
    model.AddKernel(first, "parallel_first");
    model.AddKernel(first, "parallel_first_next");
    model.AddKernel(second, "parallel_second");
    model.AddKernel(second, "parallel_second_next");
    ASSERT_EQ(aclskScopeEnd("shared", first), ACL_SUCCESS);
    ASSERT_EQ(aclskScopeEnd("shared", second), ACL_SUCCESS);
    aclskOption option{};
    option.optionType = aclskOptionType::STREAM_FUSION;
    option.streamFusion.streamFusion = enabled;
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    // STREAM_FUSION=0 currently warns without disabling a shared scope.
    EXPECT_EQ(EnabledEntries(model, first) + EnabledEntries(model, second), 1U);
    for (auto stream : {first, second}) {
      for (const auto &task : model.Tasks(stream)) {
        if (!task.disabled) {
          EXPECT_EQ(task.function, "sk_entry_aiv");
          EXPECT_FALSE(task.args.empty());
        } else {
          EXPECT_EQ(task.type, ACL_MODEL_RI_TASK_KERNEL);
        }
      }
    }
    EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  }
}

TEST_F(AotSystemTest, OptimizeDebugDumpPersistsOptionsAndGeneratedTaskQueues) {
  TemporaryWorkDirectory workDirectory;
  ASSERT_EQ(setenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META", "1", 1), 0);
  sk::test::Model model;
  auto stream = model.AddStream();
  auto first = model.AddKernel(stream, "dump_first");
  auto second = model.AddKernel(stream, "dump_second");
  char pattern[] = "dump_*";
  char *patterns[] = {pattern};
  char extend[] = "mix_kernel_split=1:simt_op_support=0";
  aclskOption values[5]{};
  values[0].optionType = aclskOptionType::DEBUG_SYNC_ALL;
  values[0].debugSync.debugSyncAll = 1;
  values[1].optionType = aclskOptionType::DCCI_BEFORE_KERNEL_START;
  values[1].dcciBeforeKernelStart = {patterns, 1};
  values[2].optionType = aclskOptionType::DCCI_AFTER_KERNEL_END;
  values[2].dcciAfterKernelEnd = {patterns, 1};
  values[3].optionType = aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK;
  values[3].debugCrossCoreSyncCheck.enableCrossCoreSyncCheck = 1;
  values[4].optionType = aclskOptionType::OPT_EXTEND_OPTION;
  values[4].optExtend.value = extend;
  aclskOptions options{values, 5};
  ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  ASSERT_EQ(SkUtGetDebugJsonPrintCallCount(), 2U);
  const auto directory = std::filesystem::path(SkUtGetDebugJsonPrintPath(0)).parent_path();
  EXPECT_EQ(directory, std::filesystem::path(SkUtGetDebugJsonPrintPath(1)).parent_path());
  const auto original = ReadJson(directory / "sk_graph_origin.json");
  EXPECT_EQ(original.at("options").at("debug_sync_all").at("value"), 1);
  EXPECT_EQ(original.at("options").at("dcci_before_kernel_start").at("value"), Json::array({"dump_*"}));
  EXPECT_EQ(original.at("options").at("opt_extend_option").at("value").at("mix_kernel_split"), Json::array({"1"}));
  const auto updated = ReadJson(directory / "sk_graph_updated.json");
  EXPECT_EQ(updated.at("options"), original.at("options"));
  const auto queues = ReadJson(directory / "sk_task_queue.json");
  ASSERT_EQ(queues.at("scopeCount"), 1);
  const auto &aiv = queues.at("scopes").at(0).at("taskQueues").at("aiv");
  EXPECT_EQ(aiv.at("funcCnt"), 2);
  std::vector<uint32_t> functionNodes;
  for (const auto &task : aiv.at("taskQue").at("taskInfos")) {
    if (task.at("type") == "FUNC") {
      functionNodes.push_back(task.at("nodeIndex").get<uint32_t>());
      EXPECT_NE(task.at("debugOptions").get<uint64_t>(), 0U);
    }
  }
  EXPECT_EQ(functionNodes, (std::vector<uint32_t>{model.Snapshot(first).id, model.Snapshot(second).id}));
  EXPECT_EQ(EnabledEntries(model, stream), 1U);
  const auto tasks = model.Tasks(stream);
  const auto entry = std::find_if(tasks.begin(), tasks.end(), [](const auto &task) {
    return !task.disabled && task.type == ACL_MODEL_RI_TASK_KERNEL && task.setParamsCount != 0;
  });
  ASSERT_NE(entry, tasks.end());
  ASSERT_EQ(entry->function.rfind("sk_entry_aiv", 0), 0U);
  const std::string expectedLaunchInfo =
      ", kernelType: AIV_ONLY, numBlocks: " + std::to_string(entry->numBlocks) + ", batchMode: 1";
  std::ifstream fusedNodesLog(directory / "sk_fused_nodes.log");
  ASSERT_TRUE(fusedNodesLog.is_open());
  std::vector<std::string> functionLines;
  size_t launchInfoLineCount = 0;
  for (std::string line; std::getline(fusedNodesLog, line);) {
    if (line.find("SK Function:") != std::string::npos) {
      functionLines.push_back(line);
    }
    if (line.find("SK LaunchInfo{") != std::string::npos) {
      ++launchInfoLineCount;
    }
  }
  ASSERT_EQ(functionLines.size(), 1U);
  ASSERT_GE(functionLines.front().size(), expectedLaunchInfo.size());
  EXPECT_EQ(functionLines.front().compare(functionLines.front().size() - expectedLaunchInfo.size(),
                                          expectedLaunchInfo.size(), expectedLaunchInfo),
            0);
  EXPECT_EQ(launchInfoLineCount, 0U);
  model.Destroy();
  // Reinitialize logging through the public API before removing its directory.
  ASSERT_EQ(unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META"), 0);
  sk::test::Model quiet;
  quiet.AddKernel(quiet.AddStream(), "quiet");
  ASSERT_EQ(aclskOptimize(quiet.Handle(), nullptr), ACL_SUCCESS);
}

TEST_F(AotSystemTest, OptimizeRejectsRuntimeEnumerationFailureWithoutTaskUpdates) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "query_failure");
  const auto original = model.Snapshot(task);
  SkUtSetAclmdlRIGetStreamsRet(1, ACL_ERROR_FAILURE);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  EXPECT_EQ(model.Snapshot(task).args, original.args);
  EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
}

TEST_F(AotSystemTest, OptimizeEntryResolutionFailureDoesNotCommitModel) {
  sk::test::Model model;
  auto stream = model.AddStream();
  model.AddKernel(stream, "missing_entry");
  SkUtSetEntryBinHandleNull(1);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  EXPECT_EQ(EnabledEntries(model, stream), 0U);
}

TEST_F(AotSystemTest, OptimizeSyncAllocationFailureReleasesRegisteredResources) {
  sk::test::Model model;
  auto producer = model.AddStream();
  auto consumer = model.AddStream();
  uint64_t eventStorage = 0;
  auto event = reinterpret_cast<aclrtEvent>(&eventStorage);
  ASSERT_EQ(aclskScopeBegin("allocation", producer), ACL_SUCCESS);
  model.AddKernel(producer, "allocation_first");
  model.AddEvent(producer, ACL_MODEL_RI_TASK_EVENT_RECORD, event);
  ASSERT_EQ(aclskScopeEnd("allocation", producer), ACL_SUCCESS);
  model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_WAIT, event);
  model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_RESET, event);
  SkUtSetAclrtMemsetRet(ACL_ERROR_FAILURE);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
  model.Destroy();
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
}
