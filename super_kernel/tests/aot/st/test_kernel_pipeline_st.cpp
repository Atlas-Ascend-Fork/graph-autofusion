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
#include "st_process.h"
#include "exception_fixture.h"
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>

namespace {
class KernelDumpDirectory {
 public:
  KernelDumpDirectory() : previous_(std::filesystem::current_path()) {
    std::filesystem::current_path(directory_.Path());
    if (setenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META", "1", 1) != 0) {
      throw std::runtime_error("cannot enable kernel dump");
    }
  }
  ~KernelDumpDirectory() {
    unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
    // The public entry disables further file logging.
    sk::test::Model quiet;
    quiet.AddKernel(quiet.AddStream(), "cap_quiet");
    EXPECT_EQ(aclskOptimize(quiet.Handle(), nullptr), ACL_SUCCESS);
    std::error_code error;
    std::filesystem::current_path(previous_, error);
  }
  nlohmann::json Queue() const {
    const auto path = std::filesystem::path(SkUtGetDebugJsonPrintPath(0)).parent_path() / "sk_task_queue.json";
    std::ifstream input(path);
    return nlohmann::json::parse(input);
  }

 private:
  std::filesystem::path previous_;
  sk::test::TemporaryDirectory directory_;
};
}  // namespace

TEST_F(AotSystemTest, MixedKernelPipelineBuildsBothQueuesAndReportsUnpairedEarlyStartSync) {
  for (bool earlyStart : {false, true}) {
    for (uint32_t parallel : {0U, 1U}) {
      SCOPED_TRACE(earlyStart);
      SCOPED_TRACE(parallel);
      SkUtResetTestControls();
      KernelDumpDirectory directory;
      sk::test::Model model;
      auto stream = model.AddStream();
      const sk::test::KernelSpec specs[] = {
          {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0, 3},   {ACL_KERNEL_TYPE_VECTOR, 4, 0, 0, 0, 3},
          {ACL_KERNEL_TYPE_MIX, 4, 1, 2, 0, 3},    {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0, 3},
          {ACL_KERNEL_TYPE_VECTOR, 4, 0, 0, 0, 3},
      };
      std::vector<uint32_t> expected;
      for (size_t i = 0; i < 5; ++i) {
        const auto task = model.AddKernel(stream, "pipeline_" + std::to_string(i), specs[i]);
        sk::test::RegisterExceptionBinary(task);
        expected.push_back(model.Snapshot(task).id);
      }
      aclskOption values[2]{};
      values[0].optionType = aclskOptionType::EARLY_START;
      values[0].earlyStart.enableEarlyStart = earlyStart;
      values[1].optionType = aclskOptionType::AUTO_OP_PARALLEL;
      values[1].autoOpParallel.enableAutoOpParallel = parallel;
      aclskOptions options{values, 2};
      const auto result = aclskOptimize(model.Handle(), &options);
      if (earlyStart) {
        // Current production rejects this mixed early-start chain because
        // its generated sync has no related node. Preserve the reproducer.
        EXPECT_EQ(result, ACL_ERROR_FAILURE);
        EXPECT_EQ(model.UpdateAttempts(), 0U);
        EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("DispatchSyncTasks failed: related node is null"),
                  std::string::npos);
        continue;
      }
      ASSERT_EQ(result, ACL_SUCCESS);
      size_t entries = 0;
      for (const auto &task : model.Tasks(stream)) {
        if (!task.disabled) {
          ++entries;
          EXPECT_EQ(task.function, "sk_entry_mix12");
          EXPECT_EQ(task.numBlocks, 4U);
        }
      }
      ASSERT_EQ(entries, 1U);
      const auto queue = directory.Queue();
      ASSERT_EQ(queue.at("scopeCount"), 1);
      const auto &scope = queue.at("scopes").at(0).at("taskQueues");
      std::set<uint32_t> observed;
      for (const char *core : {"aic", "aiv"}) {
        EXPECT_GT(scope.at(core).at("funcCnt").get<size_t>(), 0U);
        for (const auto &task : scope.at(core).at("taskQue").at("taskInfos")) {
          if (task.at("type") == "FUNC") {
            observed.insert(task.at("nodeIndex").get<uint32_t>());
          }
        }
      }
      EXPECT_EQ(observed, std::set<uint32_t>(expected.begin(), expected.end()));
    }
  }
}

TEST_F(AotSystemTest, ScaleUpCapabilityAllowsDifferentScheModeBlockCounts) {
  for (uint64_t cap : {0ULL, 16ULL}) {
    SCOPED_TRACE(cap);
    sk::test::Model model;
    auto stream = model.AddStream();
    auto first = model.AddKernel(stream, "scale_first", {ACL_KERNEL_TYPE_CUBE, 2, 0, 0, 1, cap});
    auto second = model.AddKernel(stream, "scale_second", {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 1, cap});
    for (auto handle : {first, second}) {
      aclmdlRITaskParams params{};
      ASSERT_EQ(aclmdlRITaskGetParams(handle, &params), ACL_SUCCESS);
      aclrtLaunchKernelAttr attr{};
      attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
      attr.value.schemMode = 1;
      aclrtLaunchKernelCfg cfg{&attr, 1};
      params.kernelTaskParams.cfg = &cfg;
      ASSERT_EQ(aclmdlRITaskSetParams(handle, &params), ACL_SUCCESS);
    }
    ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
    std::vector<uint32_t> blocks;
    for (const auto &task : model.Tasks(stream)) {
      if (!task.disabled) {
        EXPECT_EQ(task.function, "sk_entry_aic");
        blocks.push_back(task.numBlocks);
      }
    }
    EXPECT_EQ(blocks, cap == 0 ? (std::vector<uint32_t>{2, 4}) : (std::vector<uint32_t>{4}));
  }
}

TEST_F(AotSystemTest, CompilerDcciCapabilityIsReflectedInGeneratedFunctionTasks) {
  for (uint64_t cap : {0ULL, 4ULL}) {
    SCOPED_TRACE(cap);
    SkUtResetTestControls();
    KernelDumpDirectory directory;
    sk::test::Model model;
    auto stream = model.AddStream();
    model.AddKernel(stream, "dcci_capability", {ACL_KERNEL_TYPE_VECTOR, 1, 0, 0, 0, cap});
    ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
    const auto queue = directory.Queue();
    bool found = false;
    for (const auto &task : queue.at("scopes").at(0).at("taskQueues").at("aiv").at("taskQue").at("taskInfos")) {
      if (task.at("type") == "FUNC") {
        found = true;
        // Dump flags: default DCCI-after-function (bit 5), or disabled DCCI (bit 0).
        EXPECT_EQ(task.at("debugOptions").get<uint64_t>(), cap == 0 ? 32U : 1U);
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST_F(AotSystemTest, EarlyStartPipelineWithWaitOnlyTailsBuildsBothQueues) {
  KernelDumpDirectory directory;
  sk::test::Model model;
  auto stream = model.AddStream();
  const sk::test::KernelSpec specs[] = {
      {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0, 2},   {ACL_KERNEL_TYPE_VECTOR, 4, 0, 0, 0, 2},
      {ACL_KERNEL_TYPE_MIX, 4, 1, 2, 0, 3},    {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0, 1},
      {ACL_KERNEL_TYPE_VECTOR, 4, 0, 0, 0, 1},
  };
  for (size_t i = 0; i < 5; ++i) {
    model.AddKernel(stream, "early_tail_" + std::to_string(i), specs[i]);
  }
  aclskOption option{};
  option.optionType = aclskOptionType::EARLY_START;
  option.earlyStart.enableEarlyStart = 1;
  aclskOptions options{&option, 1};
  ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  size_t count = 0;
  for (const auto &task : model.Tasks(stream)) {
    if (!task.disabled) {
      ++count;
      EXPECT_EQ(task.function, "sk_entry_mix12_early_start");
      EXPECT_FALSE(task.args.empty());
    }
  }
  EXPECT_EQ(count, 1U);
  const auto queue = directory.Queue();
  const auto &queues = queue.at("scopes").at(0).at("taskQueues");
  EXPECT_EQ(queues.at("aic").at("funcCnt"), 3);
  EXPECT_EQ(queues.at("aiv").at("funcCnt"), 3);
}

TEST_F(AotSystemTest, InvalidCompilerBindingsPreserveOriginalKernels) {
  struct Case {
    const char *name;
    sk::test::KernelSpec spec;
    std::vector<sk::test::BinaryBinding> bindings;
    const char *reason;
  };
  const Case cases[] = {
      {"missing_bindings", {}, {}, "bindMap is empty"},
      {"unknown_binding", {}, {{0, 48, {32, 32, 32, 32}}}, "Function is not found in sk bind map"},
      {"conflicting_bindings",
       {},
       {{0, 16, {32, 32, 32, 32}}, {4, 16, {48, 48, 48, 48}}},
       "duplicated entries with different values"},
      {"mixed_capabilities",
       {ACL_KERNEL_TYPE_MIX, 4, 1, 2, 0},
       {{0, 16, {32, 32, 32, 32}}, {4, 24, {40, 40, 40, 40}}},
       "cap is inconsistent"},
  };
  for (const auto &item : cases) {
    SCOPED_TRACE(item.name);
    sk::test::Model model;
    auto stream = model.AddStream();
    auto task = model.AddKernel(stream, item.name, item.spec);
    sk::test::SetKernelBindings(task, item.bindings);
    const auto before = model.Snapshot(task);
    ut_log::LogBuffer::Instance().Clear();
    ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
    const auto after = model.Snapshot(task);
    EXPECT_EQ(after.args, before.args);
    EXPECT_EQ(after.function, before.function);
    EXPECT_FALSE(after.disabled);
    EXPECT_EQ(after.setParamsCount, 0U);
    EXPECT_EQ(model.UpdateAttempts(), 0U);
    EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(item.reason), std::string::npos);
  }
}

TEST_F(AotSystemTest, FourDistinctCompilerEntriesAreEmittedInRequestedSplitMode) {
  KernelDumpDirectory directory;
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "four_bindings");
  sk::test::SetKernelBindings(task, {{0, 16, {32, 48, 64, 80}}});
  sk::test::RegisterExceptionBinary(task);
  aclskOption option{};
  option.optionType = aclskOptionType::SPLIT_MODE;
  option.splitMode.splitCnt = 4;
  aclskOptions options{&option, 1};
  ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  const auto queue = directory.Queue();
  bool found = false;
  for (const auto &item : queue.at("scopes").at(0).at("taskQueues").at("aiv").at("taskQue").at("taskInfos")) {
    if (item.at("type") == "FUNC") {
      found = true;
      EXPECT_EQ(item.at("entryCnt"), 4);
      std::set<std::string> entries;
      for (const auto &entry : item.at("entries")) {
        entries.insert(entry.get<std::string>());
      }
      EXPECT_EQ(entries.size(), 4U);
    }
  }
  EXPECT_TRUE(found);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
}
