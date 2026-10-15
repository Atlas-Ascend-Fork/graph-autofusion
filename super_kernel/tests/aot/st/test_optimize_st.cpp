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

TEST_F(AotSystemTest, OptimizeWithoutMarkersUsesWholeModelAsScope) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "ordinary");
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  const auto after = model.Snapshot(task);
  EXPECT_EQ(after.function, "sk_entry_aiv");
  EXPECT_FALSE(after.args.empty());
  EXPECT_EQ(after.numBlocks, 1U);
  EXPECT_EQ(after.setParamsCount, 1U);
  EXPECT_FALSE(after.disabled);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
}

TEST_F(AotSystemTest, OptimizeSingleScopeReplacesTasksAndReleasesModelResources) {
  sk::test::Model model;
  auto stream = model.AddStream();
  ASSERT_EQ(aclskScopeBegin("single", stream), ACL_SUCCESS);
  model.AddKernel(stream, "first");
  model.AddKernel(stream, "second");
  ASSERT_EQ(aclskScopeEnd("single", stream), ACL_SUCCESS);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  const auto tasks = model.Tasks(stream);
  ASSERT_EQ(tasks.size(), 8U);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  size_t replaced = 0;
  for (const auto &task : tasks) {
    if (task.setParamsCount != 0) {
      ++replaced;
      EXPECT_FALSE(task.disabled);
      EXPECT_EQ(task.type, ACL_MODEL_RI_TASK_KERNEL);
      EXPECT_GT(task.numBlocks, 0U);
      EXPECT_FALSE(task.args.empty());
      EXPECT_EQ(task.function, "sk_entry_aiv");
    } else {
      EXPECT_TRUE(task.disabled);
    }
  }
  EXPECT_EQ(replaced, 1U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 1U);
  model.Destroy();
  EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
  EXPECT_EQ(SkUtGetModelDestroyCallbackCount(), 0U);
}

TEST_F(AotSystemTest, OptimizeUnpairedScopeFailsBeforeCommit) {
  sk::test::Model model;
  auto stream = model.AddStream();
  ASSERT_EQ(aclskScopeBegin("broken", stream), ACL_SUCCESS);
  model.AddKernel(stream, "first");
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 0U);
}

TEST_F(AotSystemTest, OptimizeUpdateFailurePropagatesAndModelCanBeDestroyed) {
  sk::test::Model model;
  auto stream = model.AddStream();
  ASSERT_EQ(aclskScopeBegin("failure", stream), ACL_SUCCESS);
  model.AddKernel(stream, "first");
  model.AddKernel(stream, "second");
  ASSERT_EQ(aclskScopeEnd("failure", stream), ACL_SUCCESS);
  SkUtSetAclmdlRIUpdateRet(ACL_ERROR_FAILURE);
  EXPECT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(model.UpdateAttempts(), 1U);
  EXPECT_EQ(model.SuccessfulUpdates(), 0U);
  model.Destroy();
  EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
}

TEST_F(AotSystemTest, OptimizeTwoStreamsPreservesCrossStreamSynchronization) {
  sk::test::Model model;
  auto producer = model.AddStream();
  auto consumer = model.AddStream();
  uint64_t eventStorage = 0;
  auto event = reinterpret_cast<aclrtEvent>(&eventStorage);
  ASSERT_EQ(aclskScopeBegin("producer", producer), ACL_SUCCESS);
  model.AddKernel(producer, "first");
  model.AddKernel(producer, "second");
  ASSERT_EQ(aclskScopeEnd("producer", producer), ACL_SUCCESS);
  auto record = model.AddEvent(producer, ACL_MODEL_RI_TASK_EVENT_RECORD, event);
  auto wait = model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_WAIT, event);
  ASSERT_EQ(aclskScopeBegin("consumer", consumer), ACL_SUCCESS);
  model.AddKernel(consumer, "third");
  model.AddKernel(consumer, "fourth");
  ASSERT_EQ(aclskScopeEnd("consumer", consumer), ACL_SUCCESS);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  for (auto stream : {producer, consumer}) {
    const auto tasks = model.Tasks(stream);
    EXPECT_EQ(std::count_if(tasks.begin(), tasks.end(),
                            [](const auto &task) {
                              return task.type == ACL_MODEL_RI_TASK_KERNEL && !task.disabled && task.setParamsCount > 0;
                            }),
              1);
  }
  const auto notify = model.Snapshot(record);
  const auto awaited = model.Snapshot(wait);
  // Events outside both named scopes remain Runtime tasks in their original order.
  EXPECT_EQ(notify.type, ACL_MODEL_RI_TASK_EVENT_RECORD);
  EXPECT_EQ(awaited.type, ACL_MODEL_RI_TASK_EVENT_WAIT);
  EXPECT_FALSE(notify.disabled);
  EXPECT_FALSE(awaited.disabled);
  EXPECT_EQ(notify.syncAddress, reinterpret_cast<uintptr_t>(event));
  EXPECT_EQ(notify.syncAddress, awaited.syncAddress);
  EXPECT_EQ(notify.syncValue, awaited.syncValue);
}

TEST_F(AotSystemTest, OptimizeFusedRecordRewritesExternalWaitAndReleasesSyncMemory) {
  sk::test::Model model;
  auto producer = model.AddStream();
  auto consumer = model.AddStream();
  uint64_t eventStorage = 0;
  auto event = reinterpret_cast<aclrtEvent>(&eventStorage);
  ASSERT_EQ(aclskScopeBegin("producer", producer), ACL_SUCCESS);
  model.AddKernel(producer, "first");
  auto record = model.AddEvent(producer, ACL_MODEL_RI_TASK_EVENT_RECORD, event);
  ASSERT_EQ(aclskScopeEnd("producer", producer), ACL_SUCCESS);
  auto wait = model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_WAIT, event);
  auto reset = model.AddEvent(consumer, ACL_MODEL_RI_TASK_EVENT_RESET, event);
  ASSERT_EQ(aclskScopeBegin("consumer", consumer), ACL_SUCCESS);
  model.AddKernel(consumer, "second");
  ASSERT_EQ(aclskScopeEnd("consumer", consumer), ACL_SUCCESS);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  EXPECT_TRUE(model.Snapshot(record).disabled);
  const auto awaited = model.Snapshot(wait);
  EXPECT_EQ(awaited.type, ACL_MODEL_RI_TASK_VALUE_WAIT);
  EXPECT_FALSE(awaited.disabled);
  EXPECT_NE(awaited.syncAddress, 0U);
  EXPECT_NE(awaited.syncAddress, reinterpret_cast<uintptr_t>(event));
  uint64_t initialValue = 1;
  ASSERT_EQ(aclrtMemcpy(&initialValue, sizeof(initialValue), reinterpret_cast<void *>(awaited.syncAddress),
                        sizeof(initialValue), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
  EXPECT_EQ(initialValue, 0U);
  const auto resetTask = model.Snapshot(reset);
  EXPECT_EQ(resetTask.type, ACL_MODEL_RI_TASK_VALUE_WRITE);
  EXPECT_EQ(resetTask.syncAddress, awaited.syncAddress);
  EXPECT_FALSE(resetTask.disabled);
  EXPECT_GT(sk::test::OutstandingAllocations(), 0U);
  EXPECT_EQ(model.SuccessfulUpdates(), 1U);
  model.Destroy();
  EXPECT_EQ(sk::test::OutstandingAllocations(), 0U);
}
