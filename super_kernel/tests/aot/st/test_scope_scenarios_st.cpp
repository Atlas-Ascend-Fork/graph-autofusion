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
#include <set>

TEST_F(AotSystemTest, SharedScopePreservesSurroundingTasksAndAddsBoundarySynchronization) {
  for (uint32_t streamCount : {2U, 3U}) {
    for (uint32_t parallel : {0U, 1U}) {
      SCOPED_TRACE(streamCount);
      SCOPED_TRACE(parallel);
      sk::test::Model model;
      std::vector<aclrtStream> streams;
      std::vector<aclmdlRITask> outside;
      for (uint32_t i = 0; i < streamCount; ++i) {
        auto stream = model.AddStream();
        streams.push_back(stream);
        outside.push_back(model.AddKernel(stream, "outside_before_" + std::to_string(i)));
        ASSERT_EQ(aclskScopeBegin("bounded", stream), ACL_SUCCESS);
        for (uint32_t j = 0; j < 4; ++j) {
          model.AddKernel(stream, "bounded_" + std::to_string(i) + "_" + std::to_string(j));
        }
        ASSERT_EQ(aclskScopeEnd("bounded", stream), ACL_SUCCESS);
        outside.push_back(model.AddKernel(stream, "outside_after_" + std::to_string(i)));
      }
      aclskOption option{};
      option.optionType = aclskOptionType::AUTO_OP_PARALLEL;
      option.autoOpParallel.enableAutoOpParallel = parallel;
      aclskOptions options{&option, 1};
      ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
      for (auto task : outside) {
        EXPECT_FALSE(model.Snapshot(task).disabled);
        EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
      }
      size_t entries = 0;
      std::set<uintptr_t> writes;
      std::vector<uintptr_t> waits;
      for (auto stream : streams) {
        for (const auto &task : model.Tasks(stream)) {
          if (task.disabled) {
            continue;
          }
          if (task.type == ACL_MODEL_RI_TASK_KERNEL && task.function.find("sk_entry_") == 0) {
            ++entries;
            EXPECT_FALSE(task.args.empty());
          } else if (task.type == ACL_MODEL_RI_TASK_VALUE_WRITE) {
            writes.insert(task.syncAddress);
          } else if (task.type == ACL_MODEL_RI_TASK_VALUE_WAIT) {
            waits.push_back(task.syncAddress);
          }
        }
      }
      EXPECT_EQ(entries, 1U);
      EXPECT_FALSE(waits.empty());
      for (auto address : waits) {
        EXPECT_NE(address, 0U);
        EXPECT_EQ(writes.count(address), 1U);
      }
      EXPECT_GT(sk::test::OutstandingAllocations(), 0U);
      model.Destroy();
      EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
    }
  }
}

TEST_F(AotSystemTest, InternalNotifyWaitFuseWhileResetRemainsExternal) {
  for (uint32_t parallel : {0U, 1U}) {
    SCOPED_TRACE(parallel);
    sk::test::Model model;
    auto producer = model.AddStream();
    auto consumer = model.AddStream();
    uint64_t eventStorage = 0;
    auto event = reinterpret_cast<aclrtEvent>(&eventStorage);
    ASSERT_EQ(aclskScopeBegin("internal", producer), ACL_SUCCESS);
    ASSERT_EQ(aclskScopeBegin("internal", consumer), ACL_SUCCESS);
    model.AddKernel(producer, "internal_producer");
    auto notify = model.AddEvent(producer, ACL_MODEL_RI_TASK_EVENT_RECORD, event);
    auto wait = model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_WAIT, event);
    model.AddKernel(consumer, "internal_consumer");
    auto reset = model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_RESET, event);
    ASSERT_EQ(aclskScopeEnd("internal", producer), ACL_SUCCESS);
    ASSERT_EQ(aclskScopeEnd("internal", consumer), ACL_SUCCESS);
    aclskOption option{};
    option.optionType = aclskOptionType::AUTO_OP_PARALLEL;
    option.autoOpParallel.enableAutoOpParallel = parallel;
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    EXPECT_TRUE(model.Snapshot(notify).disabled);
    EXPECT_TRUE(model.Snapshot(wait).disabled);
    EXPECT_FALSE(model.Snapshot(reset).disabled);
    EXPECT_EQ(model.Snapshot(reset).type, ACL_MODEL_RI_TASK_EVENT_RESET);
    EXPECT_EQ(model.Snapshot(reset).setParamsCount, 0U);
    EXPECT_EQ(model.Snapshot(reset).syncAddress, reinterpret_cast<uintptr_t>(event));
    EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  }
}

TEST_F(AotSystemTest, UnpairedMemoryWaitPolicyKeepsExternalAddressWhenFused) {
  for (uint32_t policy : {0U, 2U}) {
    SCOPED_TRACE(policy);
    sk::test::Model model;
    auto stream = model.AddStream();
    uint64_t memory = 1;
    model.AddKernel(stream, "unpaired_before");
    aclmdlRITaskParams params{};
    params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
    params.valueWaitTaskParams = {&memory, 1, 1};
    auto wait = model.AddTask(stream, params);
    model.AddKernel(stream, "unpaired_after");
    aclskOption option{};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.valueBreakerBypass = policy;
    aclskOptions options{&option, 1};
    ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
    EXPECT_EQ(model.Snapshot(wait).disabled, policy != 0);
    if (policy == 0) {
      EXPECT_EQ(model.Snapshot(wait).syncAddress, reinterpret_cast<uintptr_t>(&memory));
      EXPECT_EQ(model.Snapshot(wait).syncValue, 1U);
    }
    EXPECT_EQ(memory, 1U);
  }
}

TEST_F(AotSystemTest, InconsistentMemoryWaitConditionsRejectModel) {
  sk::test::Model model;
  auto stream = model.AddStream();
  uint64_t memory = 0;
  model.AddKernel(stream, "inconsistent_waits");
  aclmdlRITaskParams params{};
  params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  params.valueWriteTaskParams = {&memory, 1};
  model.AddTask(stream, params);
  params = {};
  params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  params.valueWaitTaskParams = {&memory, 1, 1};
  model.AddTask(stream, params);
  params.valueWaitTaskParams.value = 2;
  model.AddTask(stream, params);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
}

TEST_F(AotSystemTest, EmptyScopeDoesNotReplaceSurroundingTasks) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto before = model.AddKernel(stream, "empty_before");
  ASSERT_EQ(aclskScopeBegin("empty", stream), ACL_SUCCESS);
  ASSERT_EQ(aclskScopeEnd("empty", stream), ACL_SUCCESS);
  auto after = model.AddKernel(stream, "empty_after");
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  for (auto task : {before, after}) {
    EXPECT_FALSE(model.Snapshot(task).disabled);
    EXPECT_EQ(model.Snapshot(task).setParamsCount, 0U);
  }
}

TEST_F(AotSystemTest, SyncAllocationFailureDoesNotCommitCrossStreamModel) {
  sk::test::Model model;
  auto producer = model.AddStream();
  auto consumer = model.AddStream();
  uint64_t eventStorage = 0;
  auto event = reinterpret_cast<aclrtEvent>(&eventStorage);
  ASSERT_EQ(aclskScopeBegin("allocation_failure", producer), ACL_SUCCESS);
  model.AddKernel(producer, "allocation_producer");
  model.AddEvent(producer, ACL_MODEL_RI_TASK_EVENT_RECORD, event);
  ASSERT_EQ(aclskScopeEnd("allocation_failure", producer), ACL_SUCCESS);
  model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_WAIT, event);
  SkUtSetAclrtMallocRet(ACL_ERROR_FAILURE);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
  EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
}
