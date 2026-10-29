/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "dlog_pub.h"
#include "sk_task_builder.h"
#include "ut_common_stubs.h"

namespace {
constexpr uint64_t kDebugCrossCoreSyncCheck = 0x10;

SkQueueType QueueFor(SkKernelType type) {
  switch (type) {
    case SkKernelType::AIC_ONLY:
    case SkKernelType::MIX_AIC_1_0:
      return SkQueueType::AIC;
    case SkKernelType::AIV_ONLY:
    case SkKernelType::MIX_AIV_1_0:
      return SkQueueType::AIV;
    case SkKernelType::MIX_AIC_1_1:
      return SkQueueType::MIX_1_1;
    case SkKernelType::MIX_AIC_1_2:
      return SkQueueType::MIX_1_2;
    default:
      return SkQueueType::UNKNOWN;
  }
}
SkTaskDesc MakeKernelTask(uint64_t nodeId, SkKernelType kernelType, uint32_t numBlocks = 1) {
  const SkQueueType queueType = QueueFor(kernelType);
  SkKernelDesc kernel;
  kernel.kernelType = kernelType;
  kernel.originalNumBlocks = numBlocks;
  kernel.cubeNum = queueType == SkQueueType::AIV ? 0 : numBlocks;
  kernel.vectorNum = queueType == SkQueueType::AIC ? 0 : numBlocks;
  kernel.functionName = "kernel";
  kernel.deviceArgs = reinterpret_cast<const void *>(0x4000 + nodeId);
  kernel.argsSize = 64;
  kernel.binaryHandle = reinterpret_cast<aclrtBinHandle>(0x2);
  kernel.originalFunction = reinterpret_cast<aclrtFuncHandle>(0x3);
  SkResolvedFunctionDesc function;
  function.functionAddress = {0x1000 + nodeId, 0x2000 + nodeId};
  function.functionOffset = {0x10, 0x20};
  function.prefetchCount = {2, 3};
  kernel.functions.push_back(function);

  SkTaskDesc task;
  task.nodeId = nodeId;
  task.kind = SkTaskKind::KERNEL;
  task.queueType = queueType;
  task.payload = std::move(kernel);
  return task;
}

SkTaskDesc MakeEventTask(uint64_t nodeId, SkTaskKind kind, SkQueueType queueType,
                         void *address = reinterpret_cast<void *>(0x1234)) {
  SkTaskDesc task;
  task.nodeId = nodeId;
  task.kind = kind;
  task.queueType = queueType;
  task.payload = SkEventDesc{address};
  return task;
}

SkTaskIR MakeInput(SkKernelType coreType = SkKernelType::AIC_ONLY) {
  SkTaskIR input;
  input.functionName = "lowered_test";
  input.scopeId = 3;
  input.coreSpec.type = coreType;
  input.coreSpec.numBlocks = 2;
  input.coreSpec.cubeNum = coreType == SkKernelType::AIV_ONLY ? 0 : 2;
  input.coreSpec.vectorNum = coreType == SkKernelType::AIC_ONLY ? 0 : (coreType == SkKernelType::MIX_AIC_1_2 ? 4 : 2);
  input.config.splitBinCount = 1;
  input.config.counterSlotCount = 1;
  input.tasks.push_back(MakeKernelTask(42, coreType));
  return input;
}

const TaskQue *GetQueue(const SkBuildResult &result, SkQueueType type) {
  const auto *args = result.launchInfo.devArgs.Get();
  if (args == nullptr) {
    return nullptr;
  }
  const auto *base = reinterpret_cast<const uint8_t *>(args);
  const size_t offset = type == SkQueueType::AIC ? args->skHeader.aicQueOffset : args->skHeader.aivQueOffset;
  return reinterpret_cast<const TaskQue *>(base + offset);
}

const TaskInfo *FindTask(const TaskQue *queue, SkTaskType type, uint32_t occurrence = 0) {
  if (queue == nullptr) {
    return nullptr;
  }
  for (uint32_t i = 0; i < queue->taskCnt; ++i) {
    if (queue->taskInfos[i].type == type && occurrence-- == 0) {
      return &queue->taskInfos[i];
    }
  }
  return nullptr;
}

uint32_t CountTasks(const TaskQue *queue, SkTaskType type) {
  uint32_t count = 0;
  if (queue != nullptr) {
    for (uint32_t i = 0; i < queue->taskCnt; ++i) {
      count += queue->taskInfos[i].type == type ? 1U : 0U;
    }
  }
  return count;
}

uint32_t CountSyncs(const TaskQue *queue, SkCoreSyncType type) {
  uint32_t count = 0;
  if (queue != nullptr) {
    for (uint32_t i = 0; i < queue->taskCnt; ++i) {
      const auto &task = queue->taskInfos[i];
      count += task.type == SkTaskType::TYPE_SYNC && task.args == static_cast<uint64_t>(type) ? 1U : 0U;
    }
  }
  return count;
}

const SkDfxInfo *GetDfxInfo(const SkBuildResult &result, size_t index = 0) {
  const auto *args = result.launchInfo.devArgs.Get();
  if (args == nullptr) {
    return nullptr;
  }
  const auto *base = reinterpret_cast<const uint8_t *>(args);
  return reinterpret_cast<const SkDfxInfo *>(base + args->skHeader.dfxOffset) + index;
}

void SetConfig(SkTaskIR &input, SkTaskAttributeId id, bool value = true) {
  SetBoolAttribute(input.config.attributes, id, value);
}

void SetKernelBool(SkTaskDesc &task, SkTaskAttributeId id, bool value = true) {
  SetBoolAttribute(std::get<SkKernelDesc>(task.payload).attributes, id, value);
}

void SetKernelUint64(SkTaskDesc &task, SkTaskAttributeId id, uint64_t value) {
  SetUint64Attribute(std::get<SkKernelDesc>(task.payload).attributes, id, value);
}
}  // namespace

class SkTaskBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
  }

  void TearDown() override {
    SkUtResetTestControls();
  }

  SkTaskBuilder builder;
};

TEST_F(SkTaskBuilderTest, RejectsEmptyInput) {
  const auto result = builder.Build({});
  EXPECT_EQ(result.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(result.launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsInvalidCoreAndConfig) {
  auto input = MakeInput();
  input.coreSpec = {};
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.config.splitBinCount = 0;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.config.counterSlotCount = 0;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsUnknownQueueAndKernelQueueMismatch) {
  auto input = MakeInput();
  input.tasks[0].queueType = SkQueueType::UNKNOWN;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.tasks[0].queueType = SkQueueType::AIV;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsInvalidTaskIdentityAndPayload) {
  auto input = MakeInput();
  input.tasks[0].nodeId = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.tasks.push_back(MakeKernelTask(42, SkKernelType::AIC_ONLY));
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.tasks[0].payload = SkEventDesc{};
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsInvalidEventAndUnknownTaskKind) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks.push_back(MakeEventTask(43, SkTaskKind::WAIT, SkQueueType::MIX_1_1));
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.tasks[0].kind = static_cast<SkTaskKind>(0xFFU);
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsInvalidAttributesAndSplitCount) {
  auto input = MakeInput();
  input.config.attributes.push_back(
      {static_cast<SkTaskAttributeId>(0xFFFFU), SkTaskAttributeType::UINT64, SkTaskAttributeFlag::REQUIRED, 1U});
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  input.config.splitBinCount = 2;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, AcceptsUnknownOptionalAttributeForForwardCompatibility) {
  auto input = MakeInput();
  input.config.attributes.push_back(
      {static_cast<SkTaskAttributeId>(0xFFFFU), SkTaskAttributeType::UINT64, SkTaskAttributeFlag::OPTIONAL, 7U});
  EXPECT_NE(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsEventOnlyInput) {
  auto input = MakeInput();
  input.tasks = {MakeEventTask(1, SkTaskKind::WAIT, SkQueueType::AIC)};
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsUnavailableCoreAndBlockCountOverflow) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.coreSpec.vectorNum = 0;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput(SkKernelType::MIX_AIC_1_2);
  std::get<SkKernelDesc>(input.tasks[0].payload).originalNumBlocks = std::numeric_limits<uint32_t>::max() / 2U + 1U;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, BuildsSingleKernelForEveryCoreType) {
  for (const auto type :
       {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY, SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_2}) {
    SCOPED_TRACE(to_string(type));
    const auto result = builder.Build(MakeInput(type));
    ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
    EXPECT_EQ(result.launchInfo.entryInfo.entryType, type);
    EXPECT_EQ(result.launchInfo.entryInfo.nodeCnt, 1U);
    EXPECT_EQ(result.launchInfo.skFuncName, "lowered_test");
  }
}

TEST_F(SkTaskBuilderTest, DispatchesKernelAndRollingPreloadToResolvedQueues) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  auto &kernel = std::get<SkKernelDesc>(input.tasks[0].payload);
  kernel.argsSize = 96;
  const auto argsAddress = reinterpret_cast<uintptr_t>(kernel.deviceArgs);
  const auto result = builder.Build(input);
  const auto *aic = GetQueue(result, SkQueueType::AIC);
  const auto *aiv = GetQueue(result, SkQueueType::AIV);
  ASSERT_NE(aic, nullptr);
  ASSERT_NE(aiv, nullptr);
  EXPECT_EQ(CountTasks(aic, SkTaskType::TYPE_FUNC), 1U);
  EXPECT_EQ(CountTasks(aiv, SkTaskType::TYPE_FUNC), 1U);
  EXPECT_EQ(CountTasks(aic, SkTaskType::TYPE_PRELOAD), 1U);
  EXPECT_EQ(CountTasks(aiv, SkTaskType::TYPE_PRELOAD), 1U);
  EXPECT_EQ(FindTask(aic, SkTaskType::TYPE_PRELOAD)->args, 2U);
  EXPECT_EQ(FindTask(aiv, SkTaskType::TYPE_PRELOAD)->args, 3U);
  for (const auto *queue : {aic, aiv}) {
    const auto *func = FindTask(queue, SkTaskType::TYPE_FUNC);
    const auto *preload = FindTask(queue, SkTaskType::TYPE_PRELOAD);
    ASSERT_NE(func, nullptr);
    ASSERT_NE(preload, nullptr);
    EXPECT_EQ(func->args, argsAddress);
    EXPECT_EQ(func->argsSize, 96U);
    EXPECT_EQ(preload->extraInfo, argsAddress);
    EXPECT_EQ(preload->argsSize, 96U);
  }
}

TEST_F(SkTaskBuilderTest, EncodesEventDefaultsAndCustomWaitValues) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks.push_back(MakeEventTask(43, SkTaskKind::NOTIFY, SkQueueType::AIC));
  input.tasks.push_back(MakeEventTask(44, SkTaskKind::WAIT, SkQueueType::AIV));
  input.tasks.push_back(MakeEventTask(45, SkTaskKind::RESET, SkQueueType::AIC));
  auto &wait = std::get<SkEventDesc>(input.tasks[2].payload);
  wait.value = 7;
  wait.waitFlag = static_cast<uint32_t>(SkMemoryWaitFlag::AND);

  const auto result = builder.Build(input);
  const auto *notify = FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_EVENT_NOTIFY);
  const auto *waitTask = FindTask(GetQueue(result, SkQueueType::AIV), SkTaskType::TYPE_EVENT_WAIT);
  const auto *reset = FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_EVENT_RESET);
  ASSERT_NE(notify, nullptr);
  ASSERT_NE(waitTask, nullptr);
  ASSERT_NE(reset, nullptr);
  EXPECT_EQ(GetEventTaskAddr(*notify), 0x1234U);
  EXPECT_EQ(GetEventTaskValue(*notify), SK_DEFAULT_NOTIFY_VALUE);
  EXPECT_EQ(GetEventTaskFlag(*notify), SK_DEFAULT_WRITE_FLAG);
  EXPECT_EQ(GetEventTaskValue(*waitTask), 7U);
  EXPECT_EQ(GetEventTaskFlag(*waitTask), static_cast<uint32_t>(SkMemoryWaitFlag::AND));
  EXPECT_EQ(GetEventTaskValue(*reset), SK_DEFAULT_RESET_VALUE);
}

TEST_F(SkTaskBuilderTest, DispatchesUnifiedEventsInTaskOrder) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks.push_back(MakeEventTask(50, SkTaskKind::NOTIFY, SkQueueType::AIC));
  input.tasks.push_back(MakeEventTask(51, SkTaskKind::WAIT, SkQueueType::AIV));
  input.tasks.push_back(MakeEventTask(52, SkTaskKind::RESET, SkQueueType::AIC));
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_EVENT_NOTIFY), nullptr);
  EXPECT_NE(FindTask(GetQueue(result, SkQueueType::AIV), SkTaskType::TYPE_EVENT_WAIT), nullptr);
  EXPECT_NE(FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_EVENT_RESET), nullptr);
}

TEST_F(SkTaskBuilderTest, ExplicitSameCoreDependencyProducesFrameworkSyncAndScheMode) {
  for (const auto type : {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY}) {
    auto input = MakeInput(type);
    input.tasks.push_back(MakeKernelTask(43, type));
    input.tasks[0].successorIds = {43};
    const auto result = builder.Build(input);
    ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
    const auto expected =
        type == SkKernelType::AIC_ONLY ? SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC : SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV;
    EXPECT_EQ(
        CountSyncs(GetQueue(result, type == SkKernelType::AIC_ONLY ? SkQueueType::AIC : SkQueueType::AIV), expected),
        1U);
    EXPECT_TRUE(result.launchInfo.isScheModeOn);
  }
}

TEST_F(SkTaskBuilderTest, TasksWithoutSuccessorsRetainDefaultCrossCoreSync) {
  auto input = MakeInput(SkKernelType::AIC_ONLY);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::AIC_ONLY));

  ut_log::LogBuffer::Instance().Clear();
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("ExtractSyncRelationsFromGraph complete: edgeCount=0"),
            std::string::npos);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIC), SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC), 1U);
}

TEST_F(SkTaskBuilderTest, RejectsSelfAndReverseTopologicalDependencies) {
  auto input = MakeInput(SkKernelType::AIC_ONLY);
  input.tasks[0].successorIds = {input.tasks[0].nodeId};
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);

  input = MakeInput(SkKernelType::AIC_ONLY);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::AIC_ONLY));
  input.tasks[1].successorIds = {input.tasks[0].nodeId};
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, MixSplitRejectsReverseTopologicalDependencyAcrossGroups) {
  auto input = MakeInput(SkKernelType::AIC_ONLY);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::AIC_ONLY));
  SetConfig(input, SkTaskAttributeId::CONFIG_MIX_KERNEL_SPLIT_ENABLED);
  SetKernelBool(input.tasks[1], SkTaskAttributeId::KERNEL_MIX_SPLIT);
  input.tasks[1].successorIds = {input.tasks[0].nodeId};

  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, ProcessesExplicitDependencyFromEventTask) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks = {MakeEventTask(10, SkTaskKind::NOTIFY, SkQueueType::AIC), MakeKernelTask(11, SkKernelType::AIV_ONLY)};
  input.tasks[0].successorIds = {11};

  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIC), SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV), 1U);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIV), SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV), 1U);
}

TEST_F(SkTaskBuilderTest, CrossCoreDependencyProducesSetWaitAndFrameworkScheMode) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks = {MakeKernelTask(1, SkKernelType::AIC_ONLY), MakeKernelTask(2, SkKernelType::AIV_ONLY)};
  input.tasks[0].successorIds = {2};
  ut_log::LogBuffer::Instance().Clear();
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIC), SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV), 1U);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIV), SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV), 1U);
  EXPECT_TRUE(result.launchInfo.isScheModeOn);
  const auto log = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(log.find("[queue=AIC], [type=SYNC], [flag=SET_AIC_TO_AIV], [prev=AIC, next=AIV]"), std::string::npos);
  EXPECT_NE(log.find("[queue=AIV], [type=SYNC], [flag=WAIT_AIC_TO_AIV], [prev=AIC, next=AIV]"), std::string::npos);
}

TEST_F(SkTaskBuilderTest, DebugSyncAllReplacesRelationsAndRequiresScheMode) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  SetConfig(input, SkTaskAttributeId::CONFIG_DEBUG_ENABLED);
  SetConfig(input, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL);
  ut_log::LogBuffer::Instance().Clear();
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIC), SkCoreSyncType::ALL_SYNC), 1U);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIV), SkCoreSyncType::ALL_SYNC), 1U);
  EXPECT_TRUE(result.launchInfo.isScheModeOn);
  EXPECT_EQ(FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_SYNC)->debugOptions & 0x2U, 0x2U);
  const auto log = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(log.find("[queue=AIC], [type=SYNC], [flag=ALL_SYNC], [prev=UNKNOWN, next=UNKNOWN]"), std::string::npos);
  EXPECT_NE(log.find("[queue=AIV], [type=SYNC], [flag=ALL_SYNC], [prev=UNKNOWN, next=UNKNOWN]"), std::string::npos);
}

TEST_F(SkTaskBuilderTest, KernelScheModeAttributeControlsLaunchMode) {
  for (const auto type :
       {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY, SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_2}) {
    for (const bool enabled : {false, true}) {
      auto input = MakeInput(type);
      SetKernelBool(input.tasks[0], SkTaskAttributeId::KERNEL_SCHE_MODE, enabled);
      const auto result = builder.Build(input);
      ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
      EXPECT_EQ(result.launchInfo.isScheModeOn, enabled);
    }
  }
}

TEST_F(SkTaskBuilderTest, CrossCoreDebugRequiresScheModeOnlyForMixKernel) {
  for (const auto type :
       {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY, SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_2}) {
    auto input = MakeInput(type);
    SetKernelUint64(input.tasks[0], SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS, kDebugCrossCoreSyncCheck);
    const auto result = builder.Build(input);
    ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
    EXPECT_EQ(result.launchInfo.isScheModeOn, type == SkKernelType::MIX_AIC_1_1 || type == SkKernelType::MIX_AIC_1_2);
  }
}

TEST_F(SkTaskBuilderTest, BlockDimScaleUpUsesResolvedCoreCounts) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_2);
  input.coreSpec.cubeNum = 3;
  input.coreSpec.vectorNum = 6;
  std::get<SkKernelDesc>(input.tasks[0].payload).originalNumBlocks = 2;
  SetKernelBool(input.tasks[0], SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP);
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_FUNC)->numBlocks, 3U);
  EXPECT_EQ(FindTask(GetQueue(result, SkQueueType::AIV), SkTaskType::TYPE_FUNC)->numBlocks, 6U);
}

TEST_F(SkTaskBuilderTest, Mix12DoublesVectorBlockCountWithoutScaleUp) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_2);
  std::get<SkKernelDesc>(input.tasks[0].payload).originalNumBlocks = 2;
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_FUNC)->numBlocks, 2U);
  EXPECT_EQ(FindTask(GetQueue(result, SkQueueType::AIV), SkTaskType::TYPE_FUNC)->numBlocks, 4U);
}

TEST_F(SkTaskBuilderTest, MixGroupBoundaryAddsGlobalBarrier) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::MIX_AIC_1_1));
  SetKernelBool(input.tasks[1], SkTaskAttributeId::KERNEL_MIX_SPLIT);
  SetConfig(input, SkTaskAttributeId::CONFIG_MIX_KERNEL_SPLIT_ENABLED);
  ut_log::LogBuffer::Instance().Clear();
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIC), SkCoreSyncType::ALL_SYNC), 1U);
  EXPECT_EQ(CountSyncs(GetQueue(result, SkQueueType::AIV), SkCoreSyncType::ALL_SYNC), 1U);
  const auto log = ut_log::LogBuffer::Instance().GetContent();
  EXPECT_NE(log.find("taskCount=2, splitGroupCount=2, hasMixKernel=1"), std::string::npos);
  EXPECT_NE(log.find("AddBoundaryAllSync: add boundary all-sync at group=0, localTask=0, globalTask=0, nodeId=42"),
            std::string::npos);
  EXPECT_NE(log.find("PrecomputeSyncRelationsByMixGroups complete: taskCount=2, splitGroupCount=2"), std::string::npos);
}

TEST_F(SkTaskBuilderTest, FunctionTaskPreservesDebugOptionsAndSimtFlag) {
  auto input = MakeInput();
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS, 0x39U);
  SetKernelBool(input.tasks[0], SkTaskAttributeId::SIMT_ENABLED);
  const auto result = builder.Build(input);
  const auto *func = FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_FUNC);
  const auto *preload = FindTask(GetQueue(result, SkQueueType::AIC), SkTaskType::TYPE_PRELOAD);
  ASSERT_NE(func, nullptr);
  ASSERT_NE(preload, nullptr);
  EXPECT_EQ(func->debugOptions, 0x39U);
  EXPECT_EQ(func->isSimtKernel, 1U);
  EXPECT_EQ(preload->isSimtKernel, 0U);
}

TEST_F(SkTaskBuilderTest, SimtComputesMaximumDcacheAndSelectsEntry) {
  auto input = MakeInput();
  SetConfig(input, SkTaskAttributeId::CONFIG_SIMT_ENABLED);
  SetKernelBool(input.tasks[0], SkTaskAttributeId::SIMT_ENABLED);
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, 4096);
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE, 1024);
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_TRUE(result.launchInfo.useSimtEntry);
  EXPECT_EQ(result.launchInfo.skMaxDcacheSize, SK_TOTAL_UB_SIZE - 4096U - 1024U);
  EXPECT_STREQ(SkUtGetLastBinaryGetFunctionName(), "sk_entry_aic_simt");
}

TEST_F(SkTaskBuilderTest, SimtDisabledIgnoresKernelSimtMetadata) {
  auto input = MakeInput();
  SetKernelBool(input.tasks[0], SkTaskAttributeId::SIMT_ENABLED);
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_FALSE(result.launchInfo.useSimtEntry);
  EXPECT_EQ(result.launchInfo.skMaxDcacheSize, 0U);
  EXPECT_STREQ(SkUtGetLastBinaryGetFunctionName(), "sk_entry_aic");
}

TEST_F(SkTaskBuilderTest, SimtUsesMostRestrictiveDcacheLimitAcrossKernels) {
  auto input = MakeInput(SkKernelType::AIV_ONLY);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::AIV_ONLY));
  SetConfig(input, SkTaskAttributeId::CONFIG_SIMT_ENABLED);
  for (auto &task : input.tasks) {
    SetKernelBool(task, SkTaskAttributeId::SIMT_ENABLED);
  }
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, 8192);
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE, 4096);
  SetKernelUint64(input.tasks[1], SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, 24576);
  SetKernelUint64(input.tasks[1], SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE, 8192);

  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(result.launchInfo.skMaxDcacheSize, SK_TOTAL_UB_SIZE - 24576U - 8192U);
}

TEST_F(SkTaskBuilderTest, RejectsIncompleteOrOverflowingSimtMetadata) {
  auto input = MakeInput();
  SetConfig(input, SkTaskAttributeId::CONFIG_SIMT_ENABLED);
  SetKernelBool(input.tasks[0], SkTaskAttributeId::SIMT_ENABLED);
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_DYN_UBUF_SIZE, 1);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "SIMT kernel lacks ubuf size info, nodeId=42, hasDynUbufSize=1, hasAllocUbufSize=0"),
            std::string::npos);
  SetKernelUint64(input.tasks[0], SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE, SK_TOTAL_UB_SIZE);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "SIMT kernel ubuf size exceeds total ub size, nodeId=42, dynUbufSize=1, allocUbufSize="),
            std::string::npos);
}

TEST_F(SkTaskBuilderTest, EntryNameReflectsCoreAndFeatureAttributes) {
  struct Case {
    SkKernelType type;
    const char *base;
  };
  for (const auto &testCase : std::vector<Case>{{SkKernelType::AIC_ONLY, "sk_entry_aic"},
                                                {SkKernelType::AIV_ONLY, "sk_entry_aiv"},
                                                {SkKernelType::MIX_AIC_1_1, "sk_entry_mix11"},
                                                {SkKernelType::MIX_AIC_1_2, "sk_entry_mix12"}}) {
    auto input = MakeInput(testCase.type);
    SetConfig(input, SkTaskAttributeId::CONFIG_DEBUG_ENABLED);
    SetConfig(input, SkTaskAttributeId::CONFIG_PROFILING_ENABLED);
    SetConfig(input, SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED);
    SetConfig(input, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED);
    const auto result = builder.Build(input);
    ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
    EXPECT_EQ(std::string(SkUtGetLastBinaryGetFunctionName()),
              std::string(testCase.base) + "_debug_dump_profiling_op_trace_early_start");
  }
}

TEST_F(SkTaskBuilderTest, DebugBuildPreservesDeviceArgsDumpPath) {
  auto input = MakeInput();
  SetConfig(input, SkTaskAttributeId::CONFIG_DEBUG_ENABLED);
  SetConfig(input, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL);

  const auto result = builder.Build(input);
  EXPECT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(result.launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, EarlyStartEmitsMaskedSyncTasks) {
  auto input = MakeInput(SkKernelType::AIC_ONLY);
  input.tasks.push_back(MakeKernelTask(43, SkKernelType::AIC_ONLY));
  SetConfig(input, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED);
  SetKernelBool(input.tasks[0], SkTaskAttributeId::EARLY_START_SET);
  SetKernelBool(input.tasks[1], SkTaskAttributeId::EARLY_START_WAIT);
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  const auto *queue = GetQueue(result, SkQueueType::AIC);
  EXPECT_EQ(CountSyncs(queue, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC), 2U);
  const auto *first = FindTask(queue, SkTaskType::TYPE_SYNC, 0);
  const auto *second = FindTask(queue, SkTaskType::TYPE_SYNC, 1);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->extraInfo, 0U);
  EXPECT_NE(second->extraInfo, 0U);
  EXPECT_TRUE(result.launchInfo.isScheModeOn);
}

TEST_F(SkTaskBuilderTest, EarlyStartFirstAicQueueTaskAbsorbsReceiveSync) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks = {MakeKernelTask(42, SkKernelType::AIV_ONLY), MakeKernelTask(43, SkKernelType::AIC_ONLY)};
  input.tasks[0].successorIds = {43};
  SetConfig(input, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED);
  SetKernelBool(input.tasks[1], SkTaskAttributeId::EARLY_START_WAIT);

  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  const auto *queue = GetQueue(result, SkQueueType::AIC);
  const auto *func = FindTask(queue, SkTaskType::TYPE_FUNC);
  ASSERT_NE(func, nullptr);
  EXPECT_NE(func->extraInfo & static_cast<uint32_t>(SkEarlyStartMask::AIV_TO_AIC_WAIT), 0U);
  EXPECT_EQ(CountSyncs(queue, SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC), 1U);
  const auto *sync = FindTask(queue, SkTaskType::TYPE_SYNC);
  ASSERT_NE(sync, nullptr);
  EXPECT_EQ(sync->extraInfo, static_cast<uint32_t>(SkEarlyStartMask::AIV_TO_AIC_WAIT));
  EXPECT_EQ(sync->index, 1U);
  EXPECT_EQ(sync->numBlocks, 1U);
}

TEST_F(SkTaskBuilderTest, EarlyStartFirstAivQueueTaskAbsorbsReceiveSync) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks = {MakeKernelTask(42, SkKernelType::AIC_ONLY), MakeKernelTask(43, SkKernelType::AIV_ONLY)};
  input.tasks[0].successorIds = {43};
  SetConfig(input, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED);
  SetKernelBool(input.tasks[1], SkTaskAttributeId::EARLY_START_WAIT);

  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  const auto *queue = GetQueue(result, SkQueueType::AIV);
  const auto *func = FindTask(queue, SkTaskType::TYPE_FUNC);
  ASSERT_NE(func, nullptr);
  EXPECT_NE(func->extraInfo & static_cast<uint32_t>(SkEarlyStartMask::AIC_TO_AIV_WAIT), 0U);
  EXPECT_EQ(CountSyncs(queue, SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV), 1U);
  const auto *sync = FindTask(queue, SkTaskType::TYPE_SYNC);
  ASSERT_NE(sync, nullptr);
  EXPECT_EQ(sync->extraInfo, static_cast<uint32_t>(SkEarlyStartMask::AIC_TO_AIV_WAIT));
  EXPECT_EQ(sync->index, 1U);
  EXPECT_EQ(sync->numBlocks, 1U);
}

TEST_F(SkTaskBuilderTest, PreservesEarlyStartAndTaskDispatchDiagnostics) {
  for (const auto coreType : {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY}) {
    SCOPED_TRACE(to_string(coreType));
    auto input = MakeInput(coreType);
    input.tasks.push_back(MakeKernelTask(43, coreType));
    input.tasks[0].debugLabel = "producer kernel";
    input.tasks[1].debugLabel = "consumer kernel";
    SetConfig(input, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED);
    SetKernelBool(input.tasks[0], SkTaskAttributeId::EARLY_START_SET);
    SetKernelBool(input.tasks[1], SkTaskAttributeId::EARLY_START_WAIT);
    ut_log::LogBuffer::Instance().Clear();

    const auto result = builder.Build(input);
    ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
    const auto log = ut_log::LogBuffer::Instance().GetContent();
    EXPECT_NE(log.find("ApplyEarlyStartSyncPass begin: taskCount=2"), std::string::npos);
    EXPECT_NE(log.find("early-start final: task[0](nodeId=42), funcConfig="), std::string::npos);
    EXPECT_NE(log.find("relatedSetNodeId=42, relatedWaitNodeId="), std::string::npos);
    EXPECT_NE(log.find(coreType == SkKernelType::AIC_ONLY ? "[flag=AIC_TO_AIC]" : "[flag=AIV_TO_AIV]"),
              std::string::npos);
    EXPECT_NE(log.find("[activeCoreCount=1]"), std::string::npos);
    EXPECT_NE(log.find("[nodeInfo=producer kernel]"), std::string::npos);
    EXPECT_NE(log.find("add task sync for vec recv, vec recv size"), std::string::npos);
    EXPECT_NE(log.find("add sync task for cub send, cub send size"), std::string::npos);
  }
}

TEST_F(SkTaskBuilderTest, EntryArgsContainAlignedRegionsAndConfiguredCounterSlots) {
  auto input = MakeInput();
  input.config.counterSlotCount = 7;
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  const auto &header = result.launchInfo.devArgs.Get()->skHeader;
  EXPECT_EQ(header.counterOffset % 64U, 0U);
  EXPECT_EQ(header.dfxOffset - header.counterOffset, 7U * sizeof(SkCounterInfo));
  EXPECT_EQ(header.eventConfigOffset - header.dfxOffset, sizeof(SkDfxInfo));
  EXPECT_EQ(header.nodeCnt, 1U);
  EXPECT_GE(header.totalSize, header.eventConfigOffset + sizeof(SkEventConfig));
}

TEST_F(SkTaskBuilderTest, DfxInfoPreservesLoweredKernelMetadata) {
  auto input = MakeInput();
  auto &kernel = std::get<SkKernelDesc>(input.tasks[0].payload);
  kernel.originalNumBlocks = 5;
  kernel.cubeNum = 4;
  const auto result = builder.Build(input);
  const auto *dfx = GetDfxInfo(result);
  ASSERT_NE(dfx, nullptr);
  EXPECT_EQ(dfx->binHdl, reinterpret_cast<uintptr_t>(kernel.binaryHandle));
  EXPECT_EQ(dfx->funcHdlOri, reinterpret_cast<uintptr_t>(kernel.originalFunction));
  EXPECT_EQ(dfx->numBlocks, 5U);
  EXPECT_EQ(dfx->cubeNum, 4U);
  EXPECT_EQ(dfx->vecNum, 0U);
}

TEST_F(SkTaskBuilderTest, RejectsUnresolvedFunctionAddressAndTooLargeBlockDim) {
  auto input = MakeInput();
  std::get<SkKernelDesc>(input.tasks[0].payload).functions[0].functionAddress[0] = 0;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  input = MakeInput();
  std::get<SkKernelDesc>(input.tasks[0].payload).originalNumBlocks = 256;
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, RejectsEntryResolutionFailures) {
  const auto input = MakeInput();
  SkUtSetEntryBinHandleNull(1);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "failed to get entry bin handle: AscendGetEntryBinHandle() returned null"),
            std::string::npos);
  SkUtResetTestControls();
  SkUtSetBinaryGetFunctionNullHandle(1);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "failed to resolve entry func handle: funcName=sk_entry_aic, binHandle="),
            std::string::npos);
}

TEST_F(SkTaskBuilderTest, RejectsDfxInitializationAndEntryArgsCopyFailures) {
  const auto input = MakeInput();
  SkUtSetSecurecMemsetFailOnCall(1);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find("init dfxInfos failed, ret="), std::string::npos);
  for (int call : {1, 5, 9}) {
    SCOPED_TRACE(call);
    SkUtResetTestControls();
    SkUtSetSecurecMemcpyFailOnCall(call);
    ut_log::LogBuffer::Instance().Clear();
    EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
    const auto log = ut_log::LogBuffer::Instance().GetContent();
    const char *expected = call == 1   ? "GenEntryArgs memcpy_s AIC queue1 failed, ret="
                           : call == 5 ? "GenEntryArgs memcpy_s AIV queue1 failed, ret="
                                       : "GenEntryArgs memcpy_s dfx failed, ret=";
    EXPECT_NE(log.find(expected), std::string::npos);
    EXPECT_NE(log.find("Build failed: GenEntryArgs failed"), std::string::npos);
  }
}

TEST_F(SkTaskBuilderTest, RejectsCounterAndEventConfigInitializationFailures) {
  const auto input = MakeInput();
  for (int call : {2, 3}) {
    SCOPED_TRACE(call);
    SkUtResetTestControls();
    SkUtSetSecurecMemsetFailOnCall(call);
    EXPECT_EQ(builder.Build(input).launchInfo.devArgs.Get(), nullptr);
  }
}

TEST_F(SkTaskBuilderTest, RepeatedBuildsDoNotShareState) {
  const auto input = MakeInput();
  const auto first = builder.Build(input);
  const auto second = builder.Build(input);
  ASSERT_NE(first.launchInfo.devArgs.Get(), nullptr);
  ASSERT_NE(second.launchInfo.devArgs.Get(), nullptr);
  EXPECT_EQ(GetQueue(first, SkQueueType::AIC)->taskCnt, GetQueue(second, SkQueueType::AIC)->taskCnt);
  EXPECT_EQ(GetQueue(first, SkQueueType::AIV)->taskCnt, GetQueue(second, SkQueueType::AIV)->taskCnt);
  EXPECT_EQ(first.taskQueueJson, second.taskQueueJson);
}

TEST_F(SkTaskBuilderTest, ExpandsQueuesForDenseSynchronization) {
  auto input = MakeInput(SkKernelType::MIX_AIC_1_1);
  input.tasks.clear();
  for (uint64_t id = 1; id <= 20; ++id) {
    input.tasks.push_back(MakeKernelTask(id, id % 2 == 0 ? SkKernelType::AIV_ONLY : SkKernelType::AIC_ONLY));
    if (id > 1) {
      input.tasks[id - 2].successorIds.push_back(id);
    }
  }
  const auto result = builder.Build(input);
  ASSERT_NE(result.launchInfo.devArgs.Get(), nullptr);
  EXPECT_GT(GetQueue(result, SkQueueType::AIC)->taskCnt, 20U);
  EXPECT_GT(GetQueue(result, SkQueueType::AIV)->taskCnt, 20U);
}
