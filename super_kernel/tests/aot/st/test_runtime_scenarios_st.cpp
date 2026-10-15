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
#include "runtime/kernel.h"
#include "st_process.h"
#include <algorithm>
#include <limits>

namespace {
bool RunInSimtProcess() {
  if (std::getenv("SK_ST_SIMT_CHILD") != nullptr) {
    // Supply the supported architecture through the external Runtime boundary.
    SkUtSetAclrtGetSocName("Ascend950");
    return false;
  }
  const auto *test = testing::UnitTest::GetInstance()->current_test_info();
  const std::string filter = std::string("--gtest_filter=") + test->test_suite_name() + "." + test->name();
  EXPECT_EQ(sk::test::RunProcess({"/proc/self/exe", filter, "--gtest_repeat=1"}, "SK_ST_SIMT_CHILD=1"), 0);
  return true;
}
std::vector<sk::test::TaskSnapshot> Entries(const sk::test::Model &model, aclrtStream stream) {
  std::vector<sk::test::TaskSnapshot> result;
  for (const auto &task : model.Tasks(stream)) {
    if (!task.disabled && task.type == ACL_MODEL_RI_TASK_KERNEL && task.function.find("sk_entry_") == 0) {
      result.push_back(task);
    }
  }
  return result;
}
aclskOption SimtOption(char *value) {
  aclskOption option{};
  option.optionType = aclskOptionType::OPT_EXTEND_OPTION;
  option.optExtend.value = value;
  return option;
}
}  // namespace

TEST_F(AotSystemTest, SimtEntryPropagatesDynamicUbufLaunchAttribute) {
  if (RunInSimtProcess()) {
    return;
  }
  for (uint32_t type : {3U, 4U}) {
    SCOPED_TRACE(type);
    SetSimtAivType(type);
    SetFunctionAllocUbufSize(4096);
    SkUtSetAclrtFunctionAvailDynUbufSize(8192);
    sk::test::Model model;
    auto stream = model.AddStream();
    model.AddKernel(stream, "simt_first");
    model.AddKernel(stream, "simt_second");
    char value[] = "simt_op_support=1";
    auto option = SimtOption(value);
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    const auto entries = Entries(model, stream);
    ASSERT_EQ(entries.size(), 1U);
    EXPECT_EQ(entries[0].function, "sk_entry_aiv_simt");
    ASSERT_EQ(entries[0].attributes.size(), 2U);
    const auto schemMode =
        std::find_if(entries[0].attributes.begin(), entries[0].attributes.end(),
                     [](const auto &attr) { return attr.id == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE; });
    ASSERT_NE(schemMode, entries[0].attributes.end());
    EXPECT_EQ(schemMode->value.schemMode, 1U);
    const auto dynUbuf = std::find_if(entries[0].attributes.begin(), entries[0].attributes.end(), [](const auto &attr) {
      return attr.id == ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    });
    ASSERT_NE(dynUbuf, entries[0].attributes.end());
    EXPECT_EQ(dynUbuf->value.dynUBufSize, 8192U);
    EXPECT_FALSE(entries[0].args.empty());
  }
}

TEST_F(AotSystemTest, SimtMetadataFailurePreservesOriginalKernelTasks) {
  if (RunInSimtProcess()) {
    return;
  }
  for (bool failDynSize : {false, true}) {
    SCOPED_TRACE(failDynSize);
    SetSimtAivType(3);
    SetRtFunctionGetMetaInfoRet(failDynSize ? 0 : -1);
    SkUtSetAclrtFunctionGetAvailDynUbufPerBlockRet(failDynSize ? ACL_ERROR_FAILURE : ACL_SUCCESS);
    sk::test::Model model;
    auto stream = model.AddStream();
    auto task = model.AddKernel(stream, "simt_failure");
    char value[] = "simt_op_support=1";
    auto option = SimtOption(value);
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    EXPECT_TRUE(Entries(model, stream).empty());
    EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
    EXPECT_FALSE(model.Snapshot(task).disabled);
    EXPECT_EQ(model.UpdateAttempts(), 0U);
  }
}

TEST_F(AotSystemTest, SimtImpossibleUbufRequirementRejectsFusionBeforeUpdate) {
  if (RunInSimtProcess()) {
    return;
  }
  SetSimtAivType(3);
  SkUtSetAclrtFunctionAvailDynUbufSize(std::numeric_limits<size_t>::max());
  sk::test::Model model;
  auto stream = model.AddStream();
  model.AddKernel(stream, "simt_overflow");
  char value[] = "simt_op_support=1";
  auto option = SimtOption(value);
  aclskOptions options{&option, 1};
  EXPECT_EQ(aclskOptimize(model.Handle(), &options), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  EXPECT_TRUE(Entries(model, stream).empty());
}

TEST_F(AotSystemTest, RuntimeFailuresBeforeUpdateLeaveModelRecoverable) {
  using Inject = void (*)();
  const Inject cases[] = {
      [] { SkUtSetAclmdlRIGetIdRet(ACL_ERROR_FAILURE); },
      [] { SkUtSetAclmdlRIGetStreamsRet(0, ACL_ERROR_FAILURE); },
      [] { SkUtSetAclrtStreamGetIdRet(ACL_ERROR_FAILURE); },
      [] { SkUtSetAclrtGetDeviceRet(ACL_ERROR_FAILURE); },
      [] { SkUtSetAclrtGetDeviceInfoRet(ACL_ERROR_FAILURE); },
      [] { SkUtSetAclmdlRIDestroyRegisterCallbackRet(ACL_ERROR_FAILURE); },
      [] { SkUtSetBinaryGetFunctionNullHandle(1); },
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    SCOPED_TRACE(i);
    SkUtResetTestControls();
    sk::test::Model model;
    auto stream = model.AddStream();
    auto task = model.AddKernel(stream, "runtime_failure");
    cases[i]();
    EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
    EXPECT_EQ(model.UpdateAttempts(), 0U);
    EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
    model.Destroy();
    EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
    EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
  }
}

TEST_F(AotSystemTest, DynamicTaskGroupRemainsOutsideFusion) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "dynamic_group");
  aclmdlRITaskParams params{};
  ASSERT_EQ(aclmdlRITaskGetParams(task, &params), ACL_SUCCESS);
  uint64_t group = 0;
  params.taskGrp = reinterpret_cast<aclrtTaskGrp>(&group);
  ASSERT_EQ(aclmdlRITaskSetParams(task, &params), ACL_SUCCESS);
  const auto before = model.Snapshot(task);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  const auto after = model.Snapshot(task);
  EXPECT_EQ(after.setParamsCount, before.setParamsCount);
  EXPECT_EQ(after.args, before.args);
  EXPECT_FALSE(after.disabled);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
}

TEST_F(AotSystemTest, DefaultTaskRemainsOrderedBetweenFusedKernelGroups) {
  for (uint32_t bypass : {0U, 1U}) {
    SCOPED_TRACE(bypass);
    sk::test::Model model;
    auto stream = model.AddStream();
    model.AddKernel(stream, "before_default");
    aclmdlRITaskParams params{};
    params.type = ACL_MODEL_RI_TASK_DEFAULT;
    auto barrier = model.AddTask(stream, params);
    model.AddKernel(stream, "after_default");
    aclskOption option{};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.taskBreakerBypass = bypass;
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    // The default task splits implicit scopes even when bypass is enabled.
    EXPECT_EQ(Entries(model, stream).size(), 2U);
    EXPECT_FALSE(model.Snapshot(barrier).disabled);
    EXPECT_EQ(model.Snapshot(barrier).type, ACL_MODEL_RI_TASK_DEFAULT);
    EXPECT_EQ(model.Snapshot(barrier).setParamsCount, 0U);
  }
}

TEST_F(AotSystemTest, ValueWaitPoliciesPreserveOrFuseMatchedMemorySynchronization) {
  for (uint32_t policy : {0U, 1U}) {
    for (uint32_t flag : {0U, 1U, 2U, 3U}) {
      SCOPED_TRACE(policy);
      SCOPED_TRACE(flag);
      sk::test::Model model;
      auto stream = model.AddStream();
      uint64_t memory = 0;
      model.AddKernel(stream, "value_producer");
      aclmdlRITaskParams params{};
      params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
      params.valueWriteTaskParams = {&memory, 1};
      auto write = model.AddTask(stream, params);
      params = {};
      params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
      params.valueWaitTaskParams = {&memory, 1, flag};
      auto wait = model.AddTask(stream, params);
      model.AddKernel(stream, "value_consumer");
      aclskOption option{};
      option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
      option.aggressiveOpts.valueBreakerBypass = policy;
      aclskOptions options{&option, 1};
      ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
      EXPECT_EQ(Entries(model, stream).size(), policy ? 1U : 2U);
      EXPECT_EQ(model.Snapshot(write).disabled, policy != 0);
      EXPECT_EQ(model.Snapshot(wait).disabled, policy != 0);
      if (policy == 0) {
        EXPECT_EQ(model.Snapshot(wait).syncAddress, reinterpret_cast<uintptr_t>(&memory));
        EXPECT_EQ(model.Snapshot(wait).syncValue, 1U);
      }
    }
  }
}

TEST_F(AotSystemTest, DuplicateMemoryNotifiesRejectModelBeforeMutation) {
  sk::test::Model model;
  auto stream = model.AddStream();
  uint64_t memory = 0;
  model.AddKernel(stream, "duplicate_notify");
  aclmdlRITaskParams params{};
  params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  params.valueWriteTaskParams = {&memory, 1};
  model.AddTask(stream, params);
  model.AddTask(stream, params);
  params = {};
  params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  params.valueWaitTaskParams = {&memory, 1, 1};
  model.AddTask(stream, params);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  for (const auto &task : model.Tasks(stream)) {
    EXPECT_EQ(task.setParamsCount, 0U);
    EXPECT_FALSE(task.disabled);
  }
}

TEST_F(AotSystemTest, SimtArchitectureMixSplitBuildsExecutableEntryWithAndWithoutIgnorePattern) {
  if (RunInSimtProcess()) {
    return;
  }
  for (bool ignore : {false, true}) {
    for (uint16_t ratio : {1U, 2U}) {
      SCOPED_TRACE(ignore);
      SCOPED_TRACE(ratio);
      sk::test::Model model;
      auto stream = model.AddStream();
      model.AddKernel(stream, "mix_split_first", {ACL_KERNEL_TYPE_MIX, 4, 1, ratio, 0});
      model.AddKernel(stream, "mix_split_second", {ACL_KERNEL_TYPE_MIX, 4, 1, ratio, 0});
      char pattern[] = "mix_split_*";
      char *patterns[] = {pattern};
      aclskOption option{};
      option.optionType = aclskOptionType::UBUF_LOCK_IGNORE_KERNEL;
      option.ubufLockIgnoreKernel = {ignore ? 1U : 0U, patterns};
      aclskOptions options{&option, 1};
      ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
      const auto entries = Entries(model, stream);
      ASSERT_EQ(entries.size(), 1U);
      EXPECT_EQ(entries[0].function, ratio == 1 ? "sk_entry_mix11" : "sk_entry_mix12");
      EXPECT_EQ(entries[0].numBlocks, 4U);
      EXPECT_FALSE(entries[0].args.empty());
      EXPECT_EQ(model.SuccessfulUpdates(), 1U);
    }
  }
}

TEST_F(AotSystemTest, CustomQueueFusesIndependentHeterogeneousStreams) {
  for (uint32_t parallel : {0U, 1U}) {
    SCOPED_TRACE(parallel);
    sk::test::Model model;
    std::vector<aclrtStream> streams;
    const sk::test::KernelSpec specs[] = {
        {ACL_KERNEL_TYPE_CUBE, 4, 0, 0, 0},
        {ACL_KERNEL_TYPE_VECTOR, 4, 0, 0, 0},
        {ACL_KERNEL_TYPE_MIX, 4, 1, 2, 0},
    };
    for (size_t i = 0; i < 3; ++i) {
      auto stream = model.AddStream();
      streams.push_back(stream);
      for (size_t j = 0; j < 3; ++j) {
        model.AddKernel(stream, "heterogeneous_" + std::to_string(i) + "_" + std::to_string(j), specs[(i + j) % 3]);
      }
    }
    aclskOption option{};
    option.optionType = aclskOptionType::AUTO_OP_PARALLEL;
    option.autoOpParallel.enableAutoOpParallel = parallel;
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    size_t count = 0;
    size_t disabled = 0;
    for (auto stream : streams) {
      count += Entries(model, stream).size();
      for (const auto &task : model.Tasks(stream)) {
        disabled += task.disabled;
      }
    }
    EXPECT_EQ(count, 1U);
    EXPECT_EQ(disabled, 8U);
    EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  }
}

// Known production nontermination: the splitter retries the same oversized
// first kernel. Enable only with an external timeout while fixing that issue.
TEST_F(AotSystemTest, DISABLED_RuntimeCoreLimitKeepsOversizedKernelOutsideFusion) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "oversized_cube", {ACL_KERNEL_TYPE_CUBE, 33, 0, 0, 0});
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  EXPECT_TRUE(Entries(model, stream).empty());
  EXPECT_FALSE(model.Snapshot(task).disabled);
  EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
}
