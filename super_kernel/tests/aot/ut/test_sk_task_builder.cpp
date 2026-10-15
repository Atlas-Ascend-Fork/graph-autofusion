/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <map>
#include <string>
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_task_builder.h"
#include "sk_options_manager.h"
#include "sk_node.h"
#include "ut_common_stubs.h"

namespace {

void SetInnerOption(SuperKernelOptionsManager &options, SkInnerOptionType optionType, uint32_t value) {
  options.RegisterDefaultInnerOptions();
  options.innerOptionMap[optionType]->SetValue(value);
}

}  // namespace

class SkTaskBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    graph = std::make_unique<SuperKernelGraph>();
    opts = std::make_unique<SuperKernelOptionsManager>();
    builder = std::make_unique<SkTaskBuilder>(*opts, *graph);
  }

  void TearDown() override {
    SkUtResetTestControls();
  }

  SuperKernelBaseNode *CreateKernelNode(uint64_t nodeId) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateKernelNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId, uint64_t nextNodeId,
                                          SkKernelType kernelType = SkKernelType::AIC_ONLY) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, preNodeId);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.kernelInfos.kernelType = kernelType;
    node->nodeInfos.kernelInfos.needMixKernelSplit =
        kernelType == SkKernelType::MIX_AIC_1_1 || kernelType == SkKernelType::MIX_AIC_1_2;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    if (kernelType == SkKernelType::AIC_ONLY || kernelType == SkKernelType::MIX_AIC_1_0) {
      node->nodeInfos.kernelInfos.cubeNum = 1;
    } else if (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0) {
      node->nodeInfos.kernelInfos.vecNum = 1;
    } else if (kernelType == SkKernelType::MIX_AIC_1_1) {
      node->nodeInfos.kernelInfos.cubeNum = 1;
      node->nodeInfos.kernelInfos.vecNum = 1;
    } else if (kernelType == SkKernelType::MIX_AIC_1_2) {
      node->nodeInfos.kernelInfos.cubeNum = 1;
      node->nodeInfos.kernelInfos.vecNum = 2;
    }
    node->nodeInfos.kernelInfos.funcName = "k";
    // Allocate valid devArgs with proper skHeader.totalSize for AddFuncTask access
    constexpr size_t devArgsSize = 256;
    auto *devArgsBuffer = static_cast<uint8_t *>(calloc(1, devArgsSize));
    if (devArgsBuffer != nullptr) {
      auto *devArgs = reinterpret_cast<SkDeviceEntryArgs *>(devArgsBuffer);
      devArgs->skHeader.totalSize = devArgsSize;
      node->nodeInfos.kernelInfos.devArgs = devArgsBuffer;
    } else {
      node->nodeInfos.kernelInfos.devArgs = nullptr;
    }
    node->nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(0x2);
    node->nodeInfos.kernelInfos.funcHdl = reinterpret_cast<aclrtFuncHandle>(0x3);
    node->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0x1000;
    node->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0x2000;
    node->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[0] = 2;
    node->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[1] = 1;
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateNotifyNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId, uint64_t nextNodeId,
                                          uint64_t eventId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, streamIdx,
                                                        INVALID_STREAM_ID, preNodeId);
    node->SetNodeType(SkNodeType::NODE_NOTIFY);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0xABC);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateWaitNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId, uint64_t nextNodeId,
                                        uint64_t eventId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, streamIdx,
                                                        INVALID_STREAM_ID, preNodeId);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0xDEF);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateResetNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId, uint64_t nextNodeId) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, streamIdx,
                                                        INVALID_STREAM_ID, preNodeId);
    node->SetNodeType(SkNodeType::NODE_RESET);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateDefaultNodeEx(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                           uint64_t nextNodeId) {
    auto node = std::make_unique<SuperKernelDefaultNode>(nullptr, ACL_MODEL_RI_TASK_DEFAULT, 0, streamIdx,
                                                         INVALID_STREAM_ID, preNodeId);
    node->SetNodeType(SkNodeType::NODE_DEFAULT);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  ScopeCoreInfo BuildCoreInfo(SkKernelType type, uint32_t numBlocks) {
    return {type, numBlocks};
  }

  SkHostEntryInfo GenEntryInfoForTest(SkTask &aicTask, SkTask &aivTask, bool useSimtEntry = false) {
    SkKernelType type = SkKernelType::DEFAULT;
    uint32_t numBlocks = 0;
    if (aicTask.funcCnt == 0 && aivTask.funcCnt != 0) {
      type = SkKernelType::AIV_ONLY;
      numBlocks = aivTask.numBlocks;
    } else if (aicTask.funcCnt != 0 && aivTask.funcCnt == 0) {
      type = SkKernelType::AIC_ONLY;
      numBlocks = aicTask.numBlocks;
    } else if (aicTask.funcCnt != 0 && aivTask.funcCnt != 0) {
      const uint32_t mix12Blocks = aivTask.numBlocks / 2U + aivTask.numBlocks % 2U;
      if ((aicTask.nodeType == SkKernelType::MIX_AIC_1_2 && aivTask.nodeType == SkKernelType::MIX_AIC_1_2) ||
          aivTask.numBlocks > aicTask.numBlocks) {
        type = SkKernelType::MIX_AIC_1_2;
        numBlocks = std::max(aicTask.numBlocks, mix12Blocks);
      } else {
        type = SkKernelType::MIX_AIC_1_1;
        numBlocks = aicTask.numBlocks;
      }
    }
    auto scopeCoreInfo = BuildCoreInfo(type, numBlocks);
    return builder->GenEntryInfo(aicTask, aivTask, scopeCoreInfo, useSimtEntry);
  }

  SkBuildResult BuildWithCoreResult(std::string skFuncName, const std::vector<SuperKernelBaseNode *> &tasks,
                                    const std::vector<SuperKernelBaseNode *> &customTasks, uint16_t scopeId) {
    uint32_t maxCubeNum = 0;
    uint32_t maxVectorNum = 0;
    bool hasNativeMix12 = false;
    for (const auto *node : tasks) {
      if (node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
        maxCubeNum = std::max(maxCubeNum, node->GetCubeNum());
        maxVectorNum = std::max(maxVectorNum, node->GetVecNum());
        hasNativeMix12 = hasNativeMix12 || node->GetKernelType() == SkKernelType::MIX_AIC_1_2;
      }
    }
    ScopeCoreInfo result;
    if (maxCubeNum == 0) {
      result = {SkKernelType::AIV_ONLY, maxVectorNum};
    } else if (maxVectorNum == 0) {
      result = {SkKernelType::AIC_ONLY, maxCubeNum};
    } else if (hasNativeMix12 || maxVectorNum > maxCubeNum) {
      result = {SkKernelType::MIX_AIC_1_2, std::max(maxCubeNum, maxVectorNum / 2U + maxVectorNum % 2U)};
    } else {
      result = {SkKernelType::MIX_AIC_1_1, std::max(maxCubeNum, maxVectorNum)};
    }
    auto scopeCoreInfo = BuildCoreInfo(result.type, result.numBlocks);
    return builder->Build(std::move(skFuncName), tasks, customTasks, scopeId, scopeCoreInfo);
  }

  std::unique_ptr<SuperKernelGraph> graph;
  std::unique_ptr<SuperKernelOptionsManager> opts;
  std::unique_ptr<SkTaskBuilder> builder;
};

TEST_F(SkTaskBuilderTest, Build_ScheModeFollowsSubKernelRegardlessOfBlockDimScaleUp) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  uint64_t nodeId = 9000;
  for (auto type :
       {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY, SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_2}) {
    for (bool scheModeOn : {false, true}) {
      for (bool scaleUp : {false, true}) {
        SCOPED_TRACE(testing::Message() << to_string(type) << " scheMode=" << scheModeOn << " scaleUp=" << scaleUp);
        auto *node = CreateKernelNodeEx(nodeId++, 0, INVALID_TASK_ID, INVALID_TASK_ID, type);
        node->nodeInfos.kernelInfos.isScheModeOn = scheModeOn;
        node->nodeInfos.kernelInfos.capBits.blockDimScaleUp = scaleUp;
        auto result = BuildWithCoreResult("Unknown", {node}, {}, 0);
        ASSERT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
        EXPECT_EQ(result.launchInfo.isScheModeOn, scheModeOn);
      }
    }
  }
}

TEST_F(SkTaskBuilderTest, Build_FrameworkSyncRequiresScheModeWithAndWithoutEarlyStart) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>("early_start", aclskOptionType::EARLY_START, 0, 0, 1));
  for (auto type : {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY}) {
    auto *first = CreateKernelNodeEx(9050, 0, INVALID_TASK_ID, 9051, type);
    auto *second = CreateKernelNodeEx(9051, 0, 9050, INVALID_TASK_ID, type);
    first->nodeInfos.kernelInfos.capBits.earlyStartSetFlag = true;
    second->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;
    first->nodeInfos.kernelInfos.capBits.disableScheMode = true;
    second->nodeInfos.kernelInfos.capBits.disableScheMode = true;
    for (uint32_t earlyStart : {0U, 1U}) {
      SCOPED_TRACE(testing::Message() << to_string(type) << " earlyStart=" << earlyStart);
      opts->GetOption(aclskOptionType::EARLY_START)->SetValue(earlyStart);
      auto result = BuildWithCoreResult("Unknown", {first, second}, {}, 0);
      ASSERT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
      EXPECT_TRUE(result.launchInfo.isScheModeOn);

      const auto *args = result.launchInfo.devArgs.Get();
      ASSERT_NE(args, nullptr);
      const bool isCube = type == SkKernelType::AIC_ONLY;
      const auto offset = isCube ? args->skHeader.aicQueOffset : args->skHeader.aivQueOffset;
      const auto *queue = reinterpret_cast<const TaskQue *>(reinterpret_cast<const uint8_t *>(args) + offset);
      const auto expectedSync = isCube ? SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC : SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV;
      uint32_t syncCount = 0;
      for (uint32_t i = 0; i < queue->taskCnt; ++i) {
        const auto &task = queue->taskInfos[i];
        if (task.type == SkTaskType::TYPE_SYNC && task.args == static_cast<uint64_t>(expectedSync)) {
          ++syncCount;
          EXPECT_EQ(task.extraInfo != 0, earlyStart != 0);
        }
      }
      // Early-start retains separate SET/WAIT tasks even when no surplus cores need to execute them.
      EXPECT_EQ(syncCount, earlyStart != 0 ? 2U : 1U);
    }
  }
}

TEST_F(SkTaskBuilderTest, Build_DebugSyncAllRequiresScheModeForSingleKernel) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1, 0, 1));
  auto *node = CreateKernelNodeEx(9070, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto result = BuildWithCoreResult("Unknown", {node}, {}, 0);
  ASSERT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_TRUE(result.launchInfo.isScheModeOn);
}

TEST_F(SkTaskBuilderTest, Build_CubeToVectorDependencyIncludesGlobalSync) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  auto *first = CreateKernelNodeEx(9080, 0, INVALID_TASK_ID, 9081, SkKernelType::AIC_ONLY);
  auto *second = CreateKernelNodeEx(9081, 0, 9080, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto result = BuildWithCoreResult("Unknown", {first, second}, {}, 0);
  ASSERT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_TRUE(result.launchInfo.isScheModeOn);
}

TEST_F(SkTaskBuilderTest, CalculateSkScheMode_DistinguishesGlobalSyncFromCoreGroupHandshake) {
  const std::pair<SkCoreSyncType, bool> cases[] = {
      {SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC, true},
      {SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV, true},
      {SkCoreSyncType::ALL_SYNC, true},
      {SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV, false},
      {SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV, false},
      {SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC, false},
      {SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC, false},
  };
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.Init(1));
  ASSERT_TRUE(aiv.Init(1));
  for (SkTask *task : {&aic, &aiv}) {
    auto *queue = task->taskQue.get();
    queue->taskCnt = 1;
    for (const auto &testCase : cases) {
      SCOPED_TRACE(to_string(testCase.first));
      queue->taskInfos[0] = {};
      queue->taskInfos[0].type = SkTaskType::TYPE_SYNC;
      queue->taskInfos[0].args = static_cast<uint64_t>(testCase.first);
      EXPECT_EQ(builder->CalculateSkScheMode({}, aic, aiv), testCase.second);
    }
    queue->taskCnt = 0;
  }
}

TEST_F(SkTaskBuilderTest, Build_DebugCrossCoreSyncCheckRequiresScheModeOnlyForMixKernel) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));
  uint64_t nodeId = 9200;
  for (auto type :
       {SkKernelType::AIC_ONLY, SkKernelType::AIV_ONLY, SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_2}) {
    SCOPED_TRACE(to_string(type));
    auto *node = CreateKernelNodeEx(nodeId++, 0, INVALID_TASK_ID, INVALID_TASK_ID, type);
    auto result = BuildWithCoreResult("Unknown", {node}, {}, 0);
    ASSERT_NE(result.launchInfo.entryInfo.skEntryFunc, nullptr);
    EXPECT_EQ(result.launchInfo.isScheModeOn, type == SkKernelType::MIX_AIC_1_1 || type == SkKernelType::MIX_AIC_1_2);
  }
}

TEST_F(SkTaskBuilderTest, CalculateSkScheMode_PreloadAndEventsDoNotRequireBatchScheduling) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.Init(1));
  ASSERT_TRUE(aiv.Init(1));
  EXPECT_FALSE(builder->CalculateSkScheMode({}, aic, aiv));
  auto *queue = aic.taskQue.get();
  queue->taskCnt = 1;
  for (auto type : {SkTaskType::TYPE_PRELOAD, SkTaskType::TYPE_EVENT_NOTIFY, SkTaskType::TYPE_EVENT_WAIT,
                    SkTaskType::TYPE_EVENT_RESET}) {
    queue->taskInfos[0] = {};
    queue->taskInfos[0].type = type;
    queue->taskInfos[0].index = std::numeric_limits<uint32_t>::max();
    queue->taskInfos[0].extraInfo = 0xFF;  // Addresses/event values are not early-start flags.
    EXPECT_FALSE(builder->CalculateSkScheMode({}, aic, aiv));
  }
}

TEST_F(SkTaskBuilderTest, Build_EmptyTasks_ReturnEmptyLaunchInfo) {
  std::vector<SuperKernelBaseNode *> tasks;
  std::vector<SuperKernelBaseNode *> customTasks;

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.entryInfo.nodeCnt, 0U);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_KernelNotifyWait_Success) {
  auto *k1 = CreateKernelNodeEx(1, 0, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);
  auto *n2 = CreateNotifyNodeEx(2, 0, 1, 3, 99);
  auto *w3 = CreateWaitNodeEx(3, 1, INVALID_TASK_ID, 4, 99);
  auto *k4 = CreateKernelNodeEx(4, 1, 3, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  (void)k1;
  (void)n2;
  (void)w3;
  (void)k4;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(1), graph->GetNodeById(2), graph->GetNodeById(3),
                                              graph->GetNodeById(4)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_.size(), tasks.size());
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_ResetUsesPreviousKernelQueue) {
  auto *k1 = CreateKernelNodeEx(11, 0, INVALID_TASK_ID, 12, SkKernelType::AIC_ONLY);
  auto *r2 = CreateResetNodeEx(12, 0, 11, 13);
  auto *k3 = CreateKernelNodeEx(13, 0, 12, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  (void)k1;
  (void)r2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(11), graph->GetNodeById(12), graph->GetNodeById(13)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  EXPECT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIC);
  EXPECT_TRUE(builder->taskSyncInfos_[1].crossSyncInfo.empty());
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_ResetFallbackQueue) {
  auto *k1 = CreateKernelNodeEx(21, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *reset = CreateResetNodeEx(22, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  (void)k1;
  (void)reset;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(21), graph->GetNodeById(22)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  EXPECT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_ResetWithoutFallbackQueue_ReturnsFalse) {
  auto *reset = CreateResetNodeEx(23, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  std::vector<SuperKernelBaseNode *> tasks = {reset};

  EXPECT_FALSE(builder->InitTaskSyncInfos(tasks));
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MixWaitDependentPreviousTask_UsesSameAicQueue) {
  auto *k1 = CreateKernelNodeEx(101, 0, INVALID_TASK_ID, 102, SkKernelType::AIC_ONLY);
  auto *w2 = CreateWaitNodeEx(102, 0, 101, 103, 55);
  auto *k3 = CreateKernelNodeEx(103, 0, 102, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  (void)k1;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(101), graph->GetNodeById(102),
                                              graph->GetNodeById(103)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MixWaitDependentPreviousTask_UsesSameAivQueue) {
  auto *k1 = CreateKernelNodeEx(201, 0, INVALID_TASK_ID, 202, SkKernelType::AIV_ONLY);
  auto *w2 = CreateWaitNodeEx(202, 0, 201, 203, 66);
  auto *k3 = CreateKernelNodeEx(203, 0, 202, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  (void)k1;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(201), graph->GetNodeById(202),
                                              graph->GetNodeById(203)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MixWaitAccurateDirectDependency_UsesSameQueue) {
  auto *k1 = CreateKernelNodeEx(211, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *w2 = CreateWaitNodeEx(212, 1, 211, 213, 67);
  auto *k3 = CreateKernelNodeEx(213, 1, 212, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  k1->sendToNodeId.insert(213);
  k3->receiveNodeId.insert(211);
  (void)k1;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(211), graph->GetNodeById(212),
                                              graph->GetNodeById(213)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MixWaitAfterMixTask_FallbackAivQueue) {
  auto *k1 = CreateKernelNodeEx(301, 0, INVALID_TASK_ID, 302, SkKernelType::MIX_AIC_1_1);
  auto *w2 = CreateWaitNodeEx(302, 0, 301, 303, 77);
  auto *k3 = CreateKernelNodeEx(303, 0, 302, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  (void)k1;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(301), graph->GetNodeById(302),
                                              graph->GetNodeById(303)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MixWaitNonDependentPreviousTask_UsesOppositeQueue) {
  auto *k4 = CreateKernelNodeEx(404, 1, INVALID_TASK_ID, 402, SkKernelType::AIC_ONLY);
  auto *w2 = CreateWaitNodeEx(402, 1, 404, 403, 88);
  auto *k3 = CreateKernelNodeEx(403, 0, 402, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  (void)k4;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(404), graph->GetNodeById(402),
                                              graph->GetNodeById(403)};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskBuilderTest, PrecomputeAndOptimizeSyncRelations_Smoke) {
  auto *k1 = CreateKernelNodeEx(10, 0, INVALID_TASK_ID, 11, SkKernelType::AIC_ONLY);
  auto *n2 = CreateNotifyNodeEx(11, 0, 10, INVALID_TASK_ID, 77);
  auto *w3 = CreateWaitNodeEx(20, 1, INVALID_TASK_ID, 21, 77);
  auto *k4 = CreateKernelNodeEx(21, 1, 20, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  (void)k1;
  (void)n2;
  (void)w3;
  (void)k4;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(10), graph->GetNodeById(11), graph->GetNodeById(20),
                                              graph->GetNodeById(21)};

  ASSERT_TRUE(builder->PrecomputeSyncRelationsFromGraph(tasks));
  builder->PrintSyncInfo("ut-before-opt");
  builder->OptimizeSyncRelations(tasks);
  builder->PrintSyncInfo("ut-after-opt");
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelationsByMixGroups_AddsBoundarySyncAndRebasesGroupRelations) {
  auto *aic = CreateKernelNodeEx(91001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *mix0 = CreateKernelNodeEx(91002, 1, INVALID_TASK_ID, 91003, SkKernelType::MIX_AIC_1_1);
  auto *mix1 = CreateKernelNodeEx(91003, 1, 91002, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  auto *aiv = CreateKernelNodeEx(91004, 2, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);

  std::vector<SuperKernelBaseNode *> tasks = {aic, mix0, mix1, aiv};

  ASSERT_TRUE(builder->PrecomputeSyncRelationsByMixGroups(tasks));
  ASSERT_EQ(builder->taskSyncInfos_.size(), tasks.size());

  EXPECT_EQ(builder->taskSyncInfos_[0].queueType, SkQueueType::AIC);
  EXPECT_EQ(builder->taskSyncInfos_[1].queueType, SkQueueType::MIX_1_1);
  EXPECT_EQ(builder->taskSyncInfos_[2].queueType, SkQueueType::MIX_1_2);
  EXPECT_EQ(builder->taskSyncInfos_[3].queueType, SkQueueType::AIV);

  ASSERT_EQ(builder->taskSyncInfos_[0].crossSyncInfo.count(static_cast<size_t>(SkQueueType::AIC)), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[0].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)], SyncDirection::ALL_SYNC);
  EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.empty());
  EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.empty());

  ASSERT_EQ(builder->taskSyncInfos_[1].cubSendInfo.count(2), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[1].cubSendInfo[2], SyncDirection::CUB_TO_VEC);
  ASSERT_EQ(builder->taskSyncInfos_[1].vecSendInfo.count(2), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[1].vecSendInfo[2], SyncDirection::VEC_TO_CUB);

  ASSERT_EQ(builder->taskSyncInfos_[2].crossSyncInfo.count(static_cast<size_t>(SkQueueType::AIC)), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[2].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)], SyncDirection::ALL_SYNC);
  ASSERT_EQ(builder->taskSyncInfos_[2].cubRecvInfo.count(1), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[2].cubRecvInfo[1], SyncDirection::VEC_TO_CUB);
  ASSERT_EQ(builder->taskSyncInfos_[2].vecRecvInfo.count(1), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[2].vecRecvInfo[1], SyncDirection::CUB_TO_VEC);
  EXPECT_TRUE(builder->taskSyncInfos_[2].cubSendInfo.empty());
  EXPECT_TRUE(builder->taskSyncInfos_[2].vecSendInfo.empty());
}

TEST_F(SkTaskBuilderTest, SplitTasksByMixGroups_UsesKernelInfoMixSplitFlag) {
  aclskOption option{};
  option.optionType = aclskOptionType::UBUF_LOCK_IGNORE_KERNEL;
  char ignoredMix[] = "IgnoredMix";
  char *ignoredMixKernels[] = {ignoredMix};
  option.ubufLockIgnoreKernel.ubufLockIgnoreKernelCnt = 1;
  option.ubufLockIgnoreKernel.ubufLockIgnoreKernel = ignoredMixKernels;
  opts->SetOptOptionValue(&option);

  auto *aic = CreateKernelNodeEx(91401, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *ignored = CreateKernelNodeEx(91402, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  auto *splitMix = CreateKernelNodeEx(91403, 2, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  ignored->nodeInfos.kernelInfos.funcName = "IgnoredMix";
  ignored->nodeInfos.kernelInfos.needMixKernelSplit = false;
  splitMix->nodeInfos.kernelInfos.funcName = "NeedSplitMix";

  std::vector<SuperKernelBaseNode *> tasks = {aic, ignored, splitMix};

  EXPECT_FALSE(aic->nodeInfos.kernelInfos.needMixKernelSplit);
  EXPECT_FALSE(ignored->nodeInfos.kernelInfos.needMixKernelSplit);
  EXPECT_TRUE(splitMix->nodeInfos.kernelInfos.needMixKernelSplit);

  std::vector<std::vector<SuperKernelBaseNode *>> splitTasks;
  bool hasMixKernel = false;
  ASSERT_TRUE(builder->SplitTasksByMixGroups(tasks, splitTasks, hasMixKernel));
  ASSERT_EQ(splitTasks.size(), 2U);
  EXPECT_EQ(splitTasks[0].size(), 2U);
  EXPECT_EQ(splitTasks[1].size(), 1U);
  EXPECT_TRUE(hasMixKernel);
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelationsByMixGroups_InvalidTasks_ReturnFalse) {
  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups({}));

  auto *kernel = CreateKernelNodeEx(91101, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  std::vector<SuperKernelBaseNode *> tasks = {kernel, nullptr};
  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups(tasks));
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelationsByMixGroups_UnsupportedNodeType_ReturnFalse) {
  auto *defaultNode = CreateDefaultNodeEx(91201, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  std::vector<SuperKernelBaseNode *> tasks = {defaultNode};

  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups(tasks));
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelationsByMixGroups_OnlyEventWithoutFallbackKernel_ReturnFalse) {
  auto *notify = CreateNotifyNodeEx(91301, 0, INVALID_TASK_ID, INVALID_TASK_ID, 1);
  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups({notify}));

  auto *wait = CreateWaitNodeEx(91302, 0, INVALID_TASK_ID, INVALID_TASK_ID, 2);
  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups({wait}));

  auto *reset = CreateResetNodeEx(91303, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  EXPECT_FALSE(builder->PrecomputeSyncRelationsByMixGroups({reset}));
}

TEST_F(SkTaskBuilderTest, InsertAndRemoveSyncInfo_Works) {
  builder->taskSyncInfos_.resize(3);
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;

  builder->InsertSyncEvent(0, 1);
  builder->InsertSyncEvent(0, 2);

  EXPECT_FALSE(builder->taskSyncInfos_[0].cubSendInfo.empty());
  builder->RemoveSyncInfo(0, 1, false, SyncDirection::CUB_TO_VEC);
  EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) == builder->taskSyncInfos_[0].cubSendInfo.end());
}

TEST_F(SkTaskBuilderTest, JudgeRemoveCrossSync_CanDetectCrossing) {
  builder->taskSyncInfos_.resize(4);
  builder->taskSyncInfos_[1].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;

  EXPECT_TRUE(builder->JudgeRemoveCrossSync(0, 2, true));
}

TEST_F(SkTaskBuilderTest, AddAndDispatchTasks_Smoke) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(16));
  ASSERT_TRUE(aiv.taskQue.Init(16));

  auto *kernel = CreateKernelNodeEx(1001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *notify = CreateNotifyNodeEx(1002, 0, 1001, INVALID_TASK_ID, 55);
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 1);

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddSyncTask(aic, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
  ASSERT_TRUE(builder->AddEventTask(aiv, notify, 1, SkTaskType::TYPE_EVENT_NOTIFY));
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 2, 0, 1, SkTaskType::TYPE_FUNC, 1));

  ASSERT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 3, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIV));
  ASSERT_TRUE(
      builder->DispatchFuncTask(aic, aiv, kernel, &dfx, 4, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC, scopeCoreInfo));

  builder->taskSyncInfos_.resize(2);
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
  std::map<size_t, SyncDirection> syncInfo{{1, SyncDirection::CUB_TO_VEC}};
  ASSERT_TRUE(builder->DispatchSyncTasks(aic, aiv, 5, syncInfo, true, SkQueueType::AIC));
}

TEST_F(SkTaskBuilderTest, Build_RejectsScheModeCoreMismatchAgainstScopeResult) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  auto *kernel = CreateKernelNodeEx(1012, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  kernel->nodeInfos.kernelInfos.cubeNum = 0;
  kernel->nodeInfos.kernelInfos.vecNum = 5;
  kernel->nodeInfos.kernelInfos.numBlocks = 5;
  kernel->nodeInfos.kernelInfos.isScheModeOn = true;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 3);

  SkBuildResult result = builder->Build("Unknown", {kernel}, {}, 0, scopeCoreInfo);
  EXPECT_EQ(result.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(result.launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_AppliesBlockDimScaleUpFromScopeResult) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));
  auto *kernel = CreateKernelNodeEx(1011, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.cubeNum = 3;
  kernel->nodeInfos.kernelInfos.vecNum = 0;
  kernel->nodeInfos.kernelInfos.numBlocks = 3;
  kernel->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 8);

  SkDfxInfo dfx{};
  ASSERT_TRUE(
      builder->DispatchFuncTask(aic, aiv, kernel, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC, scopeCoreInfo));
  ASSERT_EQ(aic.GetTaskQue()->taskCnt, 1U);
  EXPECT_EQ(aic.GetTaskQue()->taskInfos[0].numBlocks, 8U);
}

TEST_F(SkTaskBuilderTest, AddEventTask_EncodesCustomValueAndWaitFlag) {
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *notify = CreateNotifyNodeEx(1101, 0, INVALID_TASK_ID, INVALID_TASK_ID, 66);
  notify->nodeInfos.syncInfos.memoryValue = 9;
  ASSERT_TRUE(builder->AddEventTask(aic, notify, 0, SkTaskType::TYPE_EVENT_NOTIFY));

  auto *wait = CreateWaitNodeEx(1102, 0, INVALID_TASK_ID, INVALID_TASK_ID, 67);
  wait->nodeInfos.syncInfos.memoryValue = 12;
  wait->nodeInfos.syncInfos.memoryWaitFlag = static_cast<uint32_t>(SkMemoryWaitFlag::AND);
  ASSERT_TRUE(builder->AddEventTask(aic, wait, 1, SkTaskType::TYPE_EVENT_WAIT));

  auto *reset = CreateResetNodeEx(1103, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  reset->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);
  reset->nodeInfos.syncInfos.memoryValue = 0;
  ASSERT_TRUE(builder->AddEventTask(aic, reset, 2, SkTaskType::TYPE_EVENT_RESET));

  TaskQue *taskQue = aic.GetTaskQue();
  ASSERT_NE(taskQue, nullptr);
  ASSERT_EQ(taskQue->taskCnt, 3U);

  EXPECT_EQ(taskQue->taskInfos[0].args,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(notify->nodeInfos.syncInfos.addrValue)));
  EXPECT_EQ(taskQue->taskInfos[0].entry[0], 9U);
  EXPECT_EQ(taskQue->taskInfos[0].extraInfo, 0U);

  EXPECT_EQ(taskQue->taskInfos[1].args,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(wait->nodeInfos.syncInfos.addrValue)));
  EXPECT_EQ(taskQue->taskInfos[1].entryCnt, 0U);
  EXPECT_EQ(taskQue->taskInfos[1].entry[0], 12U);
  EXPECT_EQ(taskQue->taskInfos[1].extraInfo, static_cast<uint64_t>(SkMemoryWaitFlag::AND));

  EXPECT_EQ(taskQue->taskInfos[2].args,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(reset->nodeInfos.syncInfos.addrValue)));
  EXPECT_EQ(taskQue->taskInfos[2].entry[0], 0U);
  EXPECT_EQ(taskQue->taskInfos[2].extraInfo, 0U);
}

TEST_F(SkTaskBuilderTest, AddEventTask_UnexpectedEventTaskTypeUsesZeroDefaultValue) {
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(2));

  auto *notify = CreateNotifyNodeEx(1110, 0, INVALID_TASK_ID, INVALID_TASK_ID, 66);
  notify->nodeInfos.syncInfos.memoryValue = std::numeric_limits<uint64_t>::max();
  notify->nodeInfos.syncInfos.memoryWaitFlag = std::numeric_limits<uint32_t>::max();

  ASSERT_TRUE(builder->AddEventTask(aic, notify, 0, SkTaskType::TYPE_MAX));

  TaskQue *taskQue = aic.GetTaskQue();
  ASSERT_NE(taskQue, nullptr);
  ASSERT_EQ(taskQue->taskCnt, 1U);
  EXPECT_EQ(taskQue->taskInfos[0].entry[0], 0U);
  EXPECT_EQ(taskQue->taskInfos[0].extraInfo, SK_DEFAULT_WRITE_FLAG);
}

TEST_F(SkTaskBuilderTest, EntryInfoAndArgs_GenerateSuccessfully) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(8));
  ASSERT_TRUE(aiv.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(2001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  EXPECT_NE(entryInfo.skEntryFunc, nullptr);

  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  EXPECT_NE(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_SimtKernelSetsTaskInfoFlagForFuncOnly) {
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(2002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  kernel->nodeInfos.kernelInfos.isSimtOp = true;
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_PRELOAD, 1));

  TaskQue *taskQue = aic.GetTaskQue();
  ASSERT_NE(taskQue, nullptr);
  ASSERT_EQ(taskQue->taskCnt, 2U);
  EXPECT_EQ(taskQue->taskInfos[0].isSimtKernel, 1U);
  EXPECT_EQ(taskQue->taskInfos[1].isSimtKernel, 0U);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_BlockDimScaleUpOverwritesFuncAndPreloadAfterSkBlockDimResolved) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(8));
  ASSERT_TRUE(aiv.taskQue.Init(8));

  auto *scaled = CreateKernelNodeEx(2011, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *unchanged = CreateKernelNodeEx(2012, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *skDriver = CreateKernelNodeEx(2013, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  scaled->nodeInfos.kernelInfos.numBlocks = 6;
  scaled->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  unchanged->nodeInfos.kernelInfos.numBlocks = 6;
  skDriver->nodeInfos.kernelInfos.numBlocks = 12;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 12);

  SkDfxInfo dfxInfos[3]{};
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, scaled, &dfxInfos[0], 0, 1, SkTaskType::TYPE_PRELOAD,
                                        SkQueueType::AIC, scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, scaled, &dfxInfos[0], 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC,
                                        scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, unchanged, &dfxInfos[1], 1, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::AIC, scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, skDriver, &dfxInfos[2], 2, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC,
                                        scopeCoreInfo));

  TaskQue *taskQue = aic.GetTaskQue();
  ASSERT_EQ(taskQue->taskInfos[0].numBlocks, 12U);
  ASSERT_EQ(taskQue->taskInfos[1].numBlocks, 12U);
  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);

  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.numBlocks, 12U);
  EXPECT_EQ(taskQue->taskInfos[0].numBlocks, 12U);
  EXPECT_EQ(taskQue->taskInfos[1].numBlocks, 12U);
  EXPECT_EQ(taskQue->taskInfos[2].numBlocks, 6U);
  EXPECT_EQ(taskQue->taskInfos[3].numBlocks, 12U);
  EXPECT_EQ(dfxInfos[0].numBlocks, 6U);
  EXPECT_EQ(scaled->nodeInfos.kernelInfos.numBlocks, 6U);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_BlockDimScaleUpUsesFinalMix12QueueBlockDims) {
  auto *aicNode = CreateKernelNodeEx(2021, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *scaledAiv = CreateKernelNodeEx(2022, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *aivDriver = CreateKernelNodeEx(2023, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  aicNode->nodeInfos.kernelInfos.numBlocks = 8;
  scaledAiv->nodeInfos.kernelInfos.numBlocks = 6;
  scaledAiv->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  aivDriver->nodeInfos.kernelInfos.numBlocks = 12;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 8);

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));
  SkDfxInfo dfxInfos[3]{};
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, aicNode, &dfxInfos[0], 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC,
                                        scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, scaledAiv, &dfxInfos[1], 1, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::AIV, scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, aivDriver, &dfxInfos[2], 2, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::AIV, scopeCoreInfo));

  ASSERT_EQ(aiv.GetTaskQue()->taskInfos[0].numBlocks, 16U);
  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);

  EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(entryInfo.numBlocks, 8U);
  EXPECT_EQ(aic.numBlocks, 8U);
  EXPECT_EQ(aiv.numBlocks, 16U);
  EXPECT_EQ(aiv.GetTaskQue()->taskInfos[0].numBlocks, 16U);
  EXPECT_EQ(aiv.GetTaskQue()->taskInfos[1].numBlocks, 12U);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_BlockDimScaleUpOverridesOriginalMix12Adaptation) {
  auto *scaledMix11 = CreateKernelNodeEx(2031, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  auto *mix12Driver = CreateKernelNodeEx(2032, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  scaledMix11->nodeInfos.kernelInfos.numBlocks = 4;
  scaledMix11->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  mix12Driver->nodeInfos.kernelInfos.numBlocks = 6;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 6);

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));
  SkDfxInfo dfxInfos[2]{};
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, scaledMix11, &dfxInfos[0], 0, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::MIX_1_1, scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, mix12Driver, &dfxInfos[1], 1, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::MIX_1_2, scopeCoreInfo));

  ASSERT_EQ(aiv.GetTaskQue()->taskInfos[0].numBlocks, 12U);
  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);

  EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aic.GetTaskQue()->taskInfos[0].numBlocks, 6U);
  EXPECT_EQ(aiv.GetTaskQue()->taskInfos[0].numBlocks, 12U);
  EXPECT_EQ(aiv.GetTaskQue()->taskInfos[1].numBlocks, 12U);
}

TEST_F(SkTaskBuilderTest, Build_BlockDimScaleUpUsesResolvedBlockDimInTaskQueueJson) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *scaled = CreateKernelNodeEx(2041, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *unchanged = CreateKernelNodeEx(2042, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *driver = CreateKernelNodeEx(2043, 2, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  scaled->nodeInfos.kernelInfos.numBlocks = 6;
  scaled->nodeInfos.kernelInfos.cubeNum = 6;
  scaled->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  unchanged->nodeInfos.kernelInfos.numBlocks = 8;
  unchanged->nodeInfos.kernelInfos.cubeNum = 8;
  driver->nodeInfos.kernelInfos.numBlocks = 12;
  driver->nodeInfos.kernelInfos.cubeNum = 12;
  std::vector<SuperKernelBaseNode *> tasks{scaled, unchanged, driver};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);

  ASSERT_NE(buildResult.launchInfo.entryInfo.skEntryFunc, nullptr);
  const Json &taskInfos = buildResult.taskQueueJson["taskQueues"]["aic"]["taskQue"]["taskInfos"];
  std::map<uint32_t, uint32_t> funcTaskBlockDims;
  for (const auto &taskInfo : taskInfos) {
    if (taskInfo["type"] == to_string(SkTaskType::TYPE_FUNC)) {
      funcTaskBlockDims[taskInfo["nodeIndex"].get<uint32_t>()] = taskInfo["numBlocks"].get<uint32_t>();
    }
  }
  EXPECT_EQ(funcTaskBlockDims[0], 12U);
  EXPECT_EQ(funcTaskBlockDims[1], 8U);
  EXPECT_EQ(funcTaskBlockDims[2], 12U);
  EXPECT_EQ(scaled->nodeInfos.kernelInfos.numBlocks, 6U);
}

TEST_F(SkTaskBuilderTest, Build_Mix11ScheModeAdaptsVectorQueueToMix12Result) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *mix11 = CreateKernelNodeEx(2051, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  auto *vector = CreateKernelNodeEx(2052, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  mix11->nodeInfos.kernelInfos.numBlocks = 12;
  mix11->nodeInfos.kernelInfos.cubeNum = 12;
  mix11->nodeInfos.kernelInfos.vecNum = 12;
  mix11->nodeInfos.kernelInfos.isScheModeOn = true;
  vector->nodeInfos.kernelInfos.numBlocks = 20;
  vector->nodeInfos.kernelInfos.vecNum = 20;
  std::vector<SuperKernelBaseNode *> tasks{mix11, vector};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);

  ASSERT_NE(buildResult.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(buildResult.launchInfo.entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(buildResult.launchInfo.entryInfo.numBlocks, 12U);

  const Json &aicTaskInfos = buildResult.taskQueueJson["taskQueues"]["aic"]["taskQue"]["taskInfos"];
  const Json &aivTaskInfos = buildResult.taskQueueJson["taskQueues"]["aiv"]["taskQue"]["taskInfos"];
  std::map<uint32_t, uint32_t> aicFuncBlockDims;
  std::map<uint32_t, uint32_t> aivFuncBlockDims;
  for (const auto &taskInfo : aicTaskInfos) {
    if (taskInfo["type"] == to_string(SkTaskType::TYPE_FUNC)) {
      aicFuncBlockDims[taskInfo["nodeIndex"].get<uint32_t>()] = taskInfo["numBlocks"].get<uint32_t>();
    }
  }
  for (const auto &taskInfo : aivTaskInfos) {
    if (taskInfo["type"] == to_string(SkTaskType::TYPE_FUNC)) {
      aivFuncBlockDims[taskInfo["nodeIndex"].get<uint32_t>()] = taskInfo["numBlocks"].get<uint32_t>();
    }
  }
  EXPECT_EQ(aicFuncBlockDims[0], 12U);
  EXPECT_EQ(aivFuncBlockDims[0], 24U);
  EXPECT_EQ(aivFuncBlockDims[1], 20U);
}

TEST_F(SkTaskBuilderTest, GetPreFetchCnt_OptionBranches) {
  opts->AddOption(std::make_unique<NumberOptOption>("preload", aclskOptionType::PRELOAD_CODE, 1, 0, 2));
  ResolvedFunctionInfo resolved{};
  resolved.prefetchCnt[0] = 3;
  resolved.prefetchCnt[1] = 4;

  auto p1 = builder->GetPreFetchCnt(resolved);
  EXPECT_EQ(p1.first, 3);
  EXPECT_EQ(p1.second, 4);

  opts->GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(0);
  auto p2 = builder->GetPreFetchCnt(resolved);
  EXPECT_EQ(p2.first, 16);
  EXPECT_EQ(p2.second, 8);

  opts->GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(2);
  auto p3 = builder->GetPreFetchCnt(resolved);
  EXPECT_EQ(p3.first, 0);
  EXPECT_EQ(p3.second, 0);
}

TEST_F(SkTaskBuilderTest, Build_DebugMode_TriggerDumpHelpers) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1, 0, 1));

  std::vector<SuperKernelBaseNode *> tasks;
  tasks.push_back(CreateKernelNodeEx(3001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY));

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_NotifyMissingKernel_TriggerSearchFailureLogPath) {
  auto *notify = CreateNotifyNodeEx(4001, 0, INVALID_TASK_ID, INVALID_TASK_ID, 123);
  std::vector<SuperKernelBaseNode *> tasks = {notify};

  EXPECT_FALSE(builder->InitTaskSyncInfos(tasks));
}

TEST_F(SkTaskBuilderTest, AddSyncTask_WhenQueueFull_TriggerExpand) {
  SkTask task;
  ASSERT_TRUE(task.taskQue.Init(1));
  task.taskQue.get()->taskCnt = 1;

  EXPECT_TRUE(builder->AddSyncTask(task, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
  EXPECT_GE(task.taskQue.get()->cap, 2U);
  EXPECT_EQ(task.taskQue.get()->taskCnt, 2U);
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_UnknownQueueType_ReturnFalse) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));
  auto *kernel = CreateKernelNodeEx(5001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 1);
  SkDfxInfo dfx{};

  EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, kernel, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::UNKNOWN,
                                         scopeCoreInfo));
}

TEST_F(SkTaskBuilderTest, DispatchEventTask_UnsupportedNodeType_ReturnFalse) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));
  auto *kernel = CreateKernelNodeEx(5002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  EXPECT_FALSE(builder->DispatchEventTask(aic, aiv, kernel, 0, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIC));
}

TEST_F(SkTaskBuilderTest, AddFuncTask_UnresolvedAddress_ReturnFalse) {
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(4));
  auto *kernel = CreateKernelNodeEx(5003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *kernelNode = dynamic_cast<SuperKernelKernelNode *>(kernel);
  ASSERT_NE(kernelNode, nullptr);
  kernelNode->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0;

  SkDfxInfo dfx{};
  EXPECT_FALSE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_UnknownSyncType_SkipsAndReturnsTrue) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(8));
  ASSERT_TRUE(aiv.taskQue.Init(8));

  builder->taskSyncInfos_.resize(2);
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;

  std::map<size_t, SyncDirection> syncInfo{{1, SyncDirection::MIX_TO_MIX}};
  EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, syncInfo, true, SkQueueType::AIC));
  EXPECT_EQ(aic.taskQue.get()->taskCnt, 0U);
  EXPECT_EQ(aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskMissingPreviousKernel_FallbackAndSucceed) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *kernel = CreateKernelNodeEx(5004, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *customWait = CreateWaitNodeEx(5005, 0, INVALID_TASK_ID, INVALID_TASK_ID, 100);
  (void)kernel;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(5004)};
  std::vector<SuperKernelBaseNode *> customTasks = {customWait};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  // Missing previous kernel for custom WAIT now falls back to AIV queue instead of hard-fail.
  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskUnsupportedNodeType_ReturnEmpty) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *kernel = CreateKernelNodeEx(5006, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *customKernel = CreateKernelNodeEx(5007, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  (void)kernel;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(5006)};
  std::vector<SuperKernelBaseNode *> customTasks = {customKernel};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_Mix12Branch_Selected) {
  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aiv.funcCnt = 1;
  aic.nodeType = SkKernelType::MIX_AIC_1_2;
  aiv.nodeType = SkKernelType::MIX_AIC_1_2;
  aic.numBlocks = 4;
  aiv.numBlocks = 10;
  ASSERT_TRUE(aiv.taskQue.Init(2));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_SearchDirectionFailureBranches) {
  auto *resetNoNext = CreateResetNodeEx(6001, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  auto *notifyNoNext = CreateNotifyNodeEx(6002, 0, 6001, INVALID_TASK_ID, 1);
  (void)resetNoNext;

  std::vector<SuperKernelBaseNode *> tasksNoNext = {notifyNoNext};
  EXPECT_FALSE(builder->InitTaskSyncInfos(tasksNoNext));

  auto *resetLoopA = CreateResetNodeEx(6101, 0, 6102, INVALID_TASK_ID);
  auto *resetLoopB = CreateResetNodeEx(6102, 0, 6101, INVALID_TASK_ID);
  auto *notifyLoop = CreateNotifyNodeEx(6103, 0, 6101, INVALID_TASK_ID, 2);
  (void)resetLoopA;
  (void)resetLoopB;

  std::vector<SuperKernelBaseNode *> tasksLoop = {notifyLoop};
  EXPECT_FALSE(builder->InitTaskSyncInfos(tasksLoop));

  const uint64_t chainStart = 6201;
  const uint64_t chainLen = 105;
  for (uint64_t i = 0; i < chainLen; ++i) {
    uint64_t nodeId = chainStart + i;
    uint64_t preNodeId = (i + 1 < chainLen) ? (nodeId + 1) : (nodeId + 1);
    CreateResetNodeEx(nodeId, 0, preNodeId, INVALID_TASK_ID);
  }
  auto *tailReset = CreateResetNodeEx(chainStart + chainLen, 0, chainStart + chainLen, INVALID_TASK_ID);
  (void)tailReset;
  auto *notifyMaxHop = CreateNotifyNodeEx(6300, 0, chainStart, INVALID_TASK_ID, 3);

  std::vector<SuperKernelBaseNode *> tasksMaxHop = {notifyMaxHop};
  EXPECT_FALSE(builder->InitTaskSyncInfos(tasksMaxHop));
}

TEST_F(SkTaskBuilderTest, SyncOptimizationAndDispatchSyncBranches) {
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(5);
  for (auto &info : builder->taskSyncInfos_) {
    info.queueType = SkQueueType::AIC;
  }
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[3].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[4].queueType = SkQueueType::AIV;

  builder->taskSyncInfos_[0].cubSendInfo[3] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[1].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[2].cubSendInfo[4] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[2].cubSendInfo[3] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[3].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;

  builder->taskSyncInfos_[2].vecSendInfo[4] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[1].cubRecvInfo[3] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[0].vecSendInfo[4] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[0].vecSendInfo[3] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[3].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[4].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

  builder->taskSyncInfos_[4].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].vecRecvInfo[0] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].cubRecvInfo[2] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[4].cubRecvInfo[1] = SyncDirection::VEC_TO_CUB;

  std::vector<SuperKernelBaseNode *> tasksForOptimize;
  tasksForOptimize.reserve(builder->taskSyncInfos_.size());
  for (size_t idx = 0; idx < builder->taskSyncInfos_.size(); ++idx) {
    auto *node = CreateKernelNodeEx(6500 + idx, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
    tasksForOptimize.push_back(node);
  }
  builder->OptimizeSyncRelations(tasksForOptimize);

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(32));
  ASSERT_TRUE(aiv.taskQue.Init(32));

  std::map<size_t, SyncDirection> recvCubToVec{{1, SyncDirection::CUB_TO_VEC}};
  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, recvCubToVec, false, SkQueueType::AIV));

  std::map<size_t, SyncDirection> sendVecToCub{{1, SyncDirection::VEC_TO_CUB}};
  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendVecToCub, true, SkQueueType::AIV));

  std::map<size_t, SyncDirection> sendCubToCub{{1, SyncDirection::CUB_TO_CUB}};
  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendCubToCub, true, SkQueueType::AIC));

  std::map<size_t, SyncDirection> sendVecToVec{{1, SyncDirection::VEC_TO_VEC}};
  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, sendVecToVec, true, SkQueueType::AIV));

  EXPECT_GT(aic.taskQue.get()->taskCnt + aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, Build_WithCustomNotifyWaitReset_Success) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  SetInnerOption(*opts, SkInnerOptionType::MIX_KERNEL_SPLIT, 1);

  auto *k0 = CreateKernelNodeEx(7001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  auto *k1 = CreateKernelNodeEx(7002, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  (void)k0;
  (void)k1;

  auto *cNotify = CreateNotifyNodeEx(7010, 0, 7001, INVALID_TASK_ID, 91);
  auto *cWait = CreateWaitNodeEx(7011, 0, 7001, INVALID_TASK_ID, 91);
  auto *cReset = CreateResetNodeEx(7012, 0, 7002, INVALID_TASK_ID);

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(7001), graph->GetNodeById(7002)};
  std::vector<SuperKernelBaseNode *> customTasks = {cNotify, cWait, cReset};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_WithSimtTasks_RecordsMaxDcacheSize) {
  opts->RegisterDefaultOptions();
  auto *splitOpt = opts->GetOption(aclskOptionType::SPLIT_MODE);
  ASSERT_NE(splitOpt, nullptr);
  splitOpt->SetValue(1);
  SetInnerOption(*opts, SkInnerOptionType::SIMT_OP_SUPPORT, 1);

  auto *k0 = CreateKernelNodeEx(7051, 0, INVALID_TASK_ID, 7052, SkKernelType::AIV_ONLY);
  auto *k1 = CreateKernelNodeEx(7052, 0, 7051, 7053, SkKernelType::AIV_ONLY);
  auto *k2 = CreateKernelNodeEx(7053, 0, 7052, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  k0->nodeInfos.kernelInfos.isSimtOp = true;
  k0->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k0->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k0->nodeInfos.kernelInfos.dynUbufSize = 8192;
  k0->nodeInfos.kernelInfos.allocUbufSize = 4096;
  k1->nodeInfos.kernelInfos.isSimtOp = true;
  k1->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k1->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k1->nodeInfos.kernelInfos.dynUbufSize = 24576;
  k1->nodeInfos.kernelInfos.allocUbufSize = 8192;
  k2->nodeInfos.kernelInfos.isSimtOp = false;
  k2->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k2->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k2->nodeInfos.kernelInfos.dynUbufSize = 65536;
  k2->nodeInfos.kernelInfos.allocUbufSize = 65536;

  std::vector<SuperKernelBaseNode *> tasks = {k0, k1, k2};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_TRUE(launchInfo.useSimtEntry);
  EXPECT_EQ(launchInfo.skMaxDcacheSize, SK_TOTAL_UB_SIZE - 24576U - 8192U);
  EXPECT_STREQ(SkUtGetLastBinaryGetFunctionName(), "sk_entry_mix11_simt");
}

TEST_F(SkTaskBuilderTest, Build_WithSimtTasks_SimtOpSupportDisabledSkipsRecord) {
  opts->RegisterDefaultOptions();
  auto *splitOpt = opts->GetOption(aclskOptionType::SPLIT_MODE);
  ASSERT_NE(splitOpt, nullptr);
  splitOpt->SetValue(1);
  SetInnerOption(*opts, SkInnerOptionType::SIMT_OP_SUPPORT, 0);

  auto *k0 = CreateKernelNodeEx(7061, 0, INVALID_TASK_ID, 7062, SkKernelType::AIV_ONLY);
  auto *k1 = CreateKernelNodeEx(7062, 0, 7061, INVALID_TASK_ID, SkKernelType::AIV_ONLY);

  k0->nodeInfos.kernelInfos.isSimtOp = true;
  k0->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k0->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k0->nodeInfos.kernelInfos.dynUbufSize = 8192;
  k0->nodeInfos.kernelInfos.allocUbufSize = 4096;
  k1->nodeInfos.kernelInfos.isSimtOp = true;
  k1->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k1->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k1->nodeInfos.kernelInfos.dynUbufSize = 24576;
  k1->nodeInfos.kernelInfos.allocUbufSize = 8192;

  std::vector<SuperKernelBaseNode *> tasks = {k0, k1};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_FALSE(launchInfo.useSimtEntry);
  EXPECT_EQ(launchInfo.skMaxDcacheSize, 0U);
  EXPECT_STREQ(SkUtGetLastBinaryGetFunctionName(), "sk_entry_aiv");
}

TEST_F(SkTaskBuilderTest, Build_WithSimtTaskMissingAllocUbufSize_ReturnsEmpty) {
  opts->RegisterDefaultOptions();
  auto *splitOpt = opts->GetOption(aclskOptionType::SPLIT_MODE);
  ASSERT_NE(splitOpt, nullptr);
  splitOpt->SetValue(1);
  SetInnerOption(*opts, SkInnerOptionType::SIMT_OP_SUPPORT, 1);

  auto *k0 = CreateKernelNodeEx(7071, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  k0->nodeInfos.kernelInfos.isSimtOp = true;
  k0->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k0->nodeInfos.kernelInfos.dynUbufSize = 8192;

  std::vector<SuperKernelBaseNode *> tasks = {k0};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);

  EXPECT_EQ(buildResult.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_FALSE(buildResult.launchInfo.useSimtEntry);
}

TEST_F(SkTaskBuilderTest, Build_WithSimtTaskUbufSizeOverflow_ReturnsEmpty) {
  opts->RegisterDefaultOptions();
  auto *splitOpt = opts->GetOption(aclskOptionType::SPLIT_MODE);
  ASSERT_NE(splitOpt, nullptr);
  splitOpt->SetValue(1);
  SetInnerOption(*opts, SkInnerOptionType::SIMT_OP_SUPPORT, 1);

  auto *k0 = CreateKernelNodeEx(7072, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  k0->nodeInfos.kernelInfos.isSimtOp = true;
  k0->nodeInfos.kernelInfos.hasDynUbufSize = true;
  k0->nodeInfos.kernelInfos.hasAllocUbufSize = true;
  k0->nodeInfos.kernelInfos.dynUbufSize = SK_TOTAL_UB_SIZE - 1024U;
  k0->nodeInfos.kernelInfos.allocUbufSize = 2048U;

  std::vector<SuperKernelBaseNode *> tasks = {k0};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);

  EXPECT_EQ(buildResult.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_FALSE(buildResult.launchInfo.useSimtEntry);
}

TEST_F(SkTaskBuilderTest, Build_WithResetTask_Success) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *k0 = CreateKernelNodeEx(7021, 0, INVALID_TASK_ID, 7022, SkKernelType::MIX_AIC_1_1);
  auto *reset = CreateResetNodeEx(7022, 0, 7021, 7023);
  auto *k1 = CreateKernelNodeEx(7023, 0, 7022, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  reset->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x2345);
  reset->nodeInfos.syncInfos.memoryValue = 0;

  std::vector<SuperKernelBaseNode *> tasks = {k0, reset, k1};
  std::vector<SuperKernelBaseNode *> customTasks;

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, AddTasks_NullQueueAndDispatchQueueSelection) {
  SkTask emptyTask;
  auto *notify = CreateNotifyNodeEx(8001, 0, INVALID_TASK_ID, INVALID_TASK_ID, 11);
  auto *kernel = CreateKernelNodeEx(8002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  EXPECT_FALSE(builder->AddSyncTask(emptyTask, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));
  EXPECT_FALSE(builder->AddEventTask(emptyTask, notify, 0, SkTaskType::TYPE_EVENT_NOTIFY));
  EXPECT_FALSE(builder->AddFuncTask(emptyTask, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(8));
  ASSERT_TRUE(aiv.taskQue.Init(8));

  EXPECT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 1, SkTaskType::TYPE_EVENT_NOTIFY, SkQueueType::AIC));
  EXPECT_EQ(aic.taskQue.get()->taskCnt, 1U);
  EXPECT_TRUE(builder->DispatchEventTask(aic, aiv, notify, 2, SkTaskType::TYPE_EVENT_WAIT, SkQueueType::MIX_1_1));
  EXPECT_EQ(aiv.taskQue.get()->taskCnt, 1U);
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_MixFailureAndDcciBranch) {
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(16));
  ASSERT_TRUE(aiv.taskQue.Init(16));

  auto *mix11 = CreateKernelNodeEx(8101, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);
  auto *mix12 = CreateKernelNodeEx(8102, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  auto *n1 = dynamic_cast<SuperKernelKernelNode *>(mix11);
  auto *n2 = dynamic_cast<SuperKernelKernelNode *>(mix12);
  ASSERT_NE(n1, nullptr);
  ASSERT_NE(n2, nullptr);
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 1);

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, mix11, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_1,
                                        scopeCoreInfo));
  TaskInfo &firstFunc = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((firstFunc.debugOptions & 0x1U), 0U);

  n1->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0;
  EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, mix11, &dfx, 1, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_1,
                                         scopeCoreInfo));

  n2->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0;
  EXPECT_FALSE(builder->DispatchFuncTask(aic, aiv, mix12, &dfx, 2, 1, SkTaskType::TYPE_FUNC, SkQueueType::MIX_1_2,
                                         scopeCoreInfo));
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DcciBeforeKernelStart_SetsDebugFlag) {
  opts->AddOption(std::make_unique<StringListOptOption>(
      "dcci_before_kernel_start", aclskOptionType::DCCI_BEFORE_KERNEL_START, std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(8103, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x4U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DcciAfterKernelEnd_SetsDebugFlag) {
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(8104, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DcciAfterKernelEnd_OverridesDisableFlag) {
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(8105, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_AllModeBranches) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aiv.taskQue.Init(8));

  aic.funcCnt = 0;
  aiv.funcCnt = 1;
  aiv.numBlocks = 5;
  SkHostEntryInfo onlyAiv = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(onlyAiv.skEntryFunc, nullptr);
  EXPECT_EQ(onlyAiv.entryType, SkKernelType::AIV_ONLY);

  aic.funcCnt = 1;
  aiv.funcCnt = 0;
  aic.numBlocks = 6;
  SkHostEntryInfo onlyAic = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(onlyAic.skEntryFunc, nullptr);
  EXPECT_EQ(onlyAic.entryType, SkKernelType::AIC_ONLY);

  aic.funcCnt = 1;
  aiv.funcCnt = 1;
  aic.nodeType = SkKernelType::AIC_ONLY;
  aiv.nodeType = SkKernelType::AIV_ONLY;
  aic.numBlocks = 8;
  aiv.numBlocks = 7;
  SkHostEntryInfo mix11 = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(mix11.skEntryFunc, nullptr);
  EXPECT_EQ(mix11.entryType, SkKernelType::MIX_AIC_1_1);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_UsesScopeCoreResultWithoutRecalculation) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aiv.taskQue.Init(8));

  aic.funcCnt = 0;
  aiv.funcCnt = 1;
  aiv.numBlocks = 5;

  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIV_ONLY, 5);
  SkHostEntryInfo entryInfo = builder->GenEntryInfo(aic, aiv, scopeCoreInfo);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::AIV_ONLY);
  EXPECT_EQ(entryInfo.numBlocks, 5U);
}

TEST_F(SkTaskBuilderTest, Build_WithNotifyAndWaitNodes_Success) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *k0 = CreateKernelNodeEx(8201, 0, INVALID_TASK_ID, 8202, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNodeEx(8202, 0, 8201, 8203, 101);
  auto *w2 = CreateWaitNodeEx(8203, 1, INVALID_TASK_ID, 8204, 101);
  auto *k3 = CreateKernelNodeEx(8204, 1, 8203, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  (void)k0;
  (void)n1;
  (void)w2;
  (void)k3;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(8201), graph->GetNodeById(8202),
                                              graph->GetNodeById(8203), graph->GetNodeById(8204)};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_UnsupportedKernelType_FallbackAic) {
  auto *kernel = CreateKernelNodeEx(8301, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::DEFAULT);
  std::vector<SuperKernelBaseNode *> tasks = {kernel};

  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  ASSERT_EQ(builder->taskSyncInfos_.size(), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[0].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskBuilderTest, RemoveMultiSendRecv_DenseBranches) {
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(5);
  for (auto &info : builder->taskSyncInfos_) {
    info.queueType = SkQueueType::AIV;
  }

  builder->taskSyncInfos_[0].vecSendInfo[2] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[0].vecSendInfo[3] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[2].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[3].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

  builder->taskSyncInfos_[1].cubSendInfo[2] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[1].cubSendInfo[4] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[2].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;

  builder->taskSyncInfos_[4].vecRecvInfo[0] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[4].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[4].cubRecvInfo[2] = SyncDirection::VEC_TO_CUB;

  builder->RemoveMultiSendSync();
  builder->RemoveMultiRecvSync();

  EXPECT_LE(builder->taskSyncInfos_[0].vecSendInfo.size(), 1U);
  EXPECT_LE(builder->taskSyncInfos_[1].cubSendInfo.size(), 1U);
  EXPECT_LE(builder->taskSyncInfos_[4].vecRecvInfo.size(), 1U);
  EXPECT_LE(builder->taskSyncInfos_[4].cubRecvInfo.size(), 1U);
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_NoneAndUnknownDirection_SkipAndSucceed) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  builder->taskSyncInfos_.resize(2);
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;

  std::map<size_t, SyncDirection> noneDir{{1, SyncDirection::NONE}};
  EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, noneDir, true, SkQueueType::AIC));

  std::map<size_t, SyncDirection> unknownDir{{1, static_cast<SyncDirection>(999)}};
  EXPECT_FALSE(builder->DispatchSyncTasks(aic, aiv, 0, unknownDir, false, SkQueueType::AIC));

  EXPECT_EQ(aic.taskQue.get()->taskCnt, 0U);
  EXPECT_EQ(aiv.taskQue.get()->taskCnt, 0U);
}

TEST_F(SkTaskBuilderTest, AddTasks_RemainsFullAfterExpand_AndUnsupportedNodeType) {
  auto *notify = CreateNotifyNodeEx(8601, 0, INVALID_TASK_ID, INVALID_TASK_ID, 66);
  auto *kernel = CreateKernelNodeEx(8602, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  SkTask syncTask;
  ASSERT_TRUE(syncTask.taskQue.Init(1));
  syncTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(builder->AddSyncTask(syncTask, 0, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC));

  SkTask eventTask;
  ASSERT_TRUE(eventTask.taskQue.Init(1));
  eventTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(builder->AddEventTask(eventTask, notify, 0, SkTaskType::TYPE_EVENT_NOTIFY));

  SkTask funcTask;
  ASSERT_TRUE(funcTask.taskQue.Init(1));
  funcTask.taskQue.get()->taskCnt = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(builder->AddFuncTask(funcTask, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  SkTask normalTask;
  ASSERT_TRUE(normalTask.taskQue.Init(4));
  EXPECT_FALSE(builder->AddFuncTask(normalTask, notify, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));
}

TEST_F(SkTaskBuilderTest, DispatchFuncTask_UnsupportedNodeTypeAndQueueFailure) {
  auto *notify = CreateNotifyNodeEx(8701, 0, INVALID_TASK_ID, INVALID_TASK_ID, 77);
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 1);
  SkDfxInfo dfx{};

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(1));
  ASSERT_TRUE(aiv.taskQue.Init(1));

  EXPECT_FALSE(
      builder->DispatchFuncTask(aic, aiv, notify, &dfx, 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC, scopeCoreInfo));
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_Mix12ElseBranch_Selected) {
  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aiv.funcCnt = 1;
  aic.nodeType = SkKernelType::AIC_ONLY;
  aiv.nodeType = SkKernelType::AIV_ONLY;
  aic.numBlocks = 2;
  aiv.numBlocks = 9;
  ASSERT_TRUE(aiv.taskQue.Init(2));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
}

TEST_F(SkTaskBuilderTest, Build_OnlyEventTask_GenEntryInfoFail) {
  auto *notify = CreateNotifyNodeEx(8801, 0, INVALID_TASK_ID, INVALID_TASK_ID, 201);
  std::vector<SuperKernelBaseNode *> tasks = {notify};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_ResolveEntryHandleAtBuildStage_WhenBinNotReady_ReturnEmpty) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  SkUtSetEntryBinHandleNull(1);

  auto *kernel = CreateKernelNodeEx(8911, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  std::vector<SuperKernelBaseNode *> tasks = {kernel};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, PrecomputeSyncRelations_WaitWithoutNotify_Success) {
  auto *wait = CreateWaitNodeEx(9101, 0, INVALID_TASK_ID, INVALID_TASK_ID, 333);
  std::vector<SuperKernelBaseNode *> tasks = {wait};

  EXPECT_FALSE(builder->PrecomputeSyncRelationsFromGraph(tasks));
}

TEST_F(SkTaskBuilderTest, Build_RollingPreloadDispatchFail_ReturnEmpty) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *notify = CreateNotifyNodeEx(9501, 0, INVALID_TASK_ID, 9502, 5001);
  auto *kernel = CreateKernelNodeEx(9502, 0, 9501, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *k = dynamic_cast<SuperKernelKernelNode *>(kernel);
  ASSERT_NE(k, nullptr);
  k->nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0;

  std::vector<SuperKernelBaseNode *> tasks = {notify, kernel};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_CustomTaskUnsupportedType_ReturnEmpty) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));

  auto *kernel = CreateKernelNodeEx(9001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *customKernel = CreateKernelNodeEx(9002, 0, 9001, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelBaseNode *> tasks = {kernel};
  std::vector<SuperKernelBaseNode *> customTasks = {customKernel};

  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, customTasks, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_ResolveFailWhenEntryBinNull_ReturnEmpty) {
  SkUtSetEntryBinHandleNull(1);

  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aic.numBlocks = 1;

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  EXPECT_EQ(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::DEFAULT);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_ResolveFailWhenBinaryGetNullHandle_ReturnEmpty) {
  SkUtSetBinaryGetFunctionNullHandle(1);

  SkTask aic;
  SkTask aiv;
  aiv.funcCnt = 1;
  aiv.numBlocks = 2;

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  EXPECT_EQ(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::DEFAULT);
}

TEST_F(SkTaskBuilderTest, InitTaskSyncInfos_MaxHopsExceededBranch_ReturnsTrueWithFallback) {
  const uint64_t chainStart = 10000;
  const uint64_t chainLen = 120;
  for (uint64_t i = 0; i < chainLen; ++i) {
    uint64_t nodeId = chainStart + i;
    uint64_t preNodeId = nodeId + 1;
    CreateResetNodeEx(nodeId, 0, preNodeId, INVALID_TASK_ID);
  }

  auto *notify = CreateNotifyNodeEx(20000, 0, chainStart, INVALID_TASK_ID, 77);
  std::vector<SuperKernelBaseNode *> tasks = {notify};

  EXPECT_FALSE(builder->InitTaskSyncInfos(tasks));
}

TEST_F(SkTaskBuilderTest, PrintSyncInfo_CoversVecAndCubFormattingPaths) {
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(2);
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;

  builder->taskSyncInfos_[0].vecSendInfo[1] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[0].vecRecvInfo[1] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[1].cubSendInfo[0] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[1].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;

  builder->PrintSyncInfo("ut-print-sync");
}

TEST_F(SkTaskBuilderTest, ExtractIntraStreamSync_StreamFusionOffMultiStreamBranch) {
  opts->AddOption(std::make_unique<NumberOptOption>("stream_fusion", aclskOptionType::STREAM_FUSION, 0, 0, 1));

  auto *k0 = CreateKernelNodeEx(30001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *k1 = CreateKernelNodeEx(30002, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  (void)k0;
  (void)k1;

  std::vector<SuperKernelBaseNode *> tasks = {graph->GetNodeById(30001), graph->GetNodeById(30002)};
  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));
  builder->ExtractIntraStreamSync(tasks);
}

TEST_F(SkTaskBuilderTest, InsertSyncEvent_GenSyncDirectionDenseBranches) {
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(2);

  // pre=MIX, curr=MIX -> MIX_TO_MIX
  builder->taskSyncInfos_[0] = TaskSyncInfo{};
  builder->taskSyncInfos_[1] = TaskSyncInfo{};
  builder->taskSyncInfos_[0].queueType = SkQueueType::MIX_1_1;
  builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_2;
  builder->InsertSyncEvent(0, 1);
  EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) != builder->taskSyncInfos_[0].cubSendInfo.end());
  EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

  // pre=MIX, curr=AIC -> VEC_TO_CUB
  builder->taskSyncInfos_[0] = TaskSyncInfo{};
  builder->taskSyncInfos_[1] = TaskSyncInfo{};
  builder->taskSyncInfos_[0].queueType = SkQueueType::MIX_1_2;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->InsertSyncEvent(0, 1);
  EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

  // pre=AIC, curr=MIX -> CUB_TO_VEC
  builder->taskSyncInfos_[0] = TaskSyncInfo{};
  builder->taskSyncInfos_[1] = TaskSyncInfo{};
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_1;
  builder->InsertSyncEvent(0, 1);
  EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.find(1) != builder->taskSyncInfos_[0].cubSendInfo.end());

  // pre=AIV, curr=MIX -> VEC_TO_CUB
  builder->taskSyncInfos_[0] = TaskSyncInfo{};
  builder->taskSyncInfos_[1] = TaskSyncInfo{};
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[1].queueType = SkQueueType::MIX_1_1;
  builder->InsertSyncEvent(0, 1);
  EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());

  // pre=UNKNOWN(default) -> fallback VEC_TO_CUB
  builder->taskSyncInfos_[0] = TaskSyncInfo{};
  builder->taskSyncInfos_[1] = TaskSyncInfo{};
  builder->taskSyncInfos_[0].queueType = SkQueueType::UNKNOWN;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
  builder->InsertSyncEvent(0, 1);
  EXPECT_TRUE(builder->taskSyncInfos_[0].vecSendInfo.find(1) != builder->taskSyncInfos_[0].vecSendInfo.end());
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyAicFail_ReturnEmpty) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  SkDfxInfo dfx{};
  SkUtSetSecurecMemcpyFailOnCall(1);
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyAivFail_ReturnEmpty) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  SkDfxInfo dfx{};
  SkUtSetSecurecMemcpyFailOnCall(2);
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemsetCounterFail_ReturnEmpty) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  SkDfxInfo dfx{};
  SkUtSetSecurecMemsetFailOnCall(1);
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_WithoutWorkspace_IgnoresSecondMemsetFailure) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  SkDfxInfo dfx{};
  SkUtSetSecurecMemsetFailOnCall(2);
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  ASSERT_NE(devArgs.Get(), nullptr);
  EXPECT_EQ(devArgs.Get()->skHeader.dfxOffset,
            devArgs.Get()->skHeader.counterOffset + GetSkRuntimeConfig().eventCoreNum * sizeof(SkCounterInfo));
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_CounterOffsetAlignedTo64Bytes) {
  constexpr size_t queueShardCount = 4;
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(1));
  ASSERT_TRUE(aiv.taskQue.Init(1));

  SkDfxInfo dfx{};
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  ASSERT_NE(devArgs.Get(), nullptr);

  size_t expectedAicOffset = sizeof(SkHeaderInfo);
  size_t expectedAivOffset = expectedAicOffset + aic.GetTaskQueSize() * queueShardCount;
  size_t rawCounterOffset = expectedAivOffset + aiv.GetTaskQueSize() * queueShardCount;
  EXPECT_EQ(devArgs.Get()->skHeader.aicQueSize, aic.GetTaskQueSize());
  EXPECT_EQ(devArgs.Get()->skHeader.aivQueSize, aiv.GetTaskQueSize());
  EXPECT_EQ(devArgs.Get()->skHeader.aicQueOffset, expectedAicOffset);
  EXPECT_EQ(devArgs.Get()->skHeader.aivQueOffset, expectedAivOffset);
  EXPECT_NE(rawCounterOffset % 64, 0U);
  EXPECT_EQ(devArgs.Get()->skHeader.counterOffset % 64, 0U);
  EXPECT_GE(devArgs.Get()->skHeader.counterOffset, rawCounterOffset);
}

TEST_F(SkTaskBuilderTest, GenEntryArgs_MemcpyDfxFail_ReturnEmpty) {
  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(4));
  ASSERT_TRUE(aiv.taskQue.Init(4));

  SkDfxInfo dfx{};
  SkUtSetSecurecMemcpyFailOnCall(3);
  DeviceArgsPtr devArgs = builder->GenEntryArgs(aic, aiv, &dfx, 1);
  EXPECT_EQ(devArgs.Get(), nullptr);
}

TEST_F(SkTaskBuilderTest, Build_DfxMemsetFail_ReturnEmpty) {
  std::vector<SuperKernelBaseNode *> tasks;
  tasks.push_back(CreateKernelNodeEx(40001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY));

  SkUtSetSecurecMemsetFailOnCall(1);
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;
  EXPECT_EQ(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(launchInfo.devArgs.Get(), nullptr);
}

// ==================== DEBUG_CROSS_CORE_SYNC_CHECK: AddFuncTask debugOptions bit 0x10 测试 ====================

TEST_F(SkTaskBuilderTest, AddFuncTask_CrossCoreSyncCheck_SetsDebugFlag) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(41001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x10U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CrossCoreSyncCheckNotSet_NoDebugFlag) {
  // 不设置 DEBUG_CROSS_CORE_SYNC_CHECK option
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(41002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_EQ((funcTask.debugOptions & 0x10U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CrossCoreSyncCheckZero_NoDebugFlag) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 0, 0, 1));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(41003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_EQ((funcTask.debugOptions & 0x10U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CrossCoreSyncCheck_CombinedWithOtherDebugFlags) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(41004, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  SkDfxInfo dfx{};

  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  // Both bit 0x1 (dcci_disable) and bit 0x10 (cross_core_sync_check) should be set
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x10U), 0U);
}

// ==================== DEBUG_OP_EXEC_TRACE: GenEntryInfo enableOpTrace 测试 ====================

TEST_F(SkTaskBuilderTest, GenEntryInfo_DebugOpExecTrace_EnablesOpTrace) {
  opts->AddOption(
      std::make_unique<NumberOptOption>("debug_op_exec_trace", aclskOptionType::DEBUG_OP_EXEC_TRACE, 1, 0, 1));

  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aic.numBlocks = 1;
  ASSERT_TRUE(aic.taskQue.Init(8));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_DebugOpExecTraceZero_DisablesOpTrace) {
  opts->AddOption(
      std::make_unique<NumberOptOption>("debug_op_exec_trace", aclskOptionType::DEBUG_OP_EXEC_TRACE, 0, 0, 1));

  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aic.numBlocks = 1;
  ASSERT_TRUE(aic.taskQue.Init(8));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
}

// ==================== DEBUG_CROSS_CORE_SYNC_CHECK: GenEntryInfo force enableOpTrace 测试 ====================

TEST_F(SkTaskBuilderTest, GenEntryInfo_DebugCrossCoreSyncCheck_ForceEnablesOpTrace) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));

  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aic.numBlocks = 1;
  ASSERT_TRUE(aic.taskQue.Init(8));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_DebugOpExecTraceAndCrossCoreSyncCheck_BothEnableOpTrace) {
  opts->AddOption(
      std::make_unique<NumberOptOption>("debug_op_exec_trace", aclskOptionType::DEBUG_OP_EXEC_TRACE, 1, 0, 1));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));

  SkTask aic;
  SkTask aiv;
  aic.funcCnt = 1;
  aic.numBlocks = 1;
  ASSERT_TRUE(aic.taskQue.Init(8));

  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);
  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
}

// ==================== Build 集成测试: DEBUG_CROSS_CORE_SYNC_CHECK ====================

TEST_F(SkTaskBuilderTest, Build_WithDebugCrossCoreSyncCheck_Success) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));

  auto *kernel = CreateKernelNodeEx(42001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelBaseNode *> tasks = {kernel};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  EXPECT_NE(launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(launchInfo.devArgs.Get(), nullptr);
}

// ==================== capBits.disableDcci: AddFuncTask debugOptions bit 0 测试 ====================

TEST_F(SkTaskBuilderTest, AddFuncTask_CapBitsDisableDcci_SetsBit0) {
  // 不设置任何 disableDcciList，仅通过 capBits.disableDcci 禁用 dcci
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(43001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CapBitsDisableDcciFalse_NoBit0) {
  // capBits.disableDcci = false, 且不设置 disableDcciList
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(43002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = false;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_EQ((funcTask.debugOptions & 0x1U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CapBitsDisableDcci_CombinedWithDisableDcciList) {
  // 同时设置 capBits.disableDcci 和 disableDcciList
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(43003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
}

// ==================== enable_dcci_after_func: AddFuncTask debugOptions bit 5 (0x20) 测试 ====================

TEST_F(SkTaskBuilderTest, AddFuncTask_NoDisableDcci_SetsBit5) {
  // 默认情况：未禁用 dcci，bit 5 应置位
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = false;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DisableDcciByCapBits_ClearsBit5) {
  // 通过 capBits.disableDcci 禁用 dcci，bit 5 应清除
  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_EQ((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DisableDcciByListAndCapAllowsDcci_AddsAfterFallback) {
  // SK_BIND 未声明 disable_dcci，但 option 禁用 dcci 时，记录 mismatch warning 并兜底追加 after_kernel_end。
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = false;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DisableDcciByListAndCapDisablesDcci_NoAfterFallback) {
  // SK_BIND 已声明 disable_dcci 时，沿用原语义，不因为 disable list 额外追加 after_kernel_end。
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44009, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_EQ((funcTask.debugOptions & 0x8U), 0U);
  EXPECT_EQ((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_AfterKernelEndOverridesDisable_SetsBit5) {
  // 同时禁用 dcci 和设置 after_kernel_end，bit 5 应置位 (强制执行)
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44004, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_AfterKernelEndOnly_SetsBit3AndBit5) {
  // 仅设置 after_kernel_end，bit 3 和 bit 5 都应置位
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44005, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_CapBitsDisableDcciAndAfterKernelEnd_SetsBit5) {
  // capBits.disableDcci=true + after_kernel_end，bit 5 应置位 (强制执行)
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44006, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  // bit 0 (disable) 和 bit 3 (after_end) 都置位，但 bit 5 应置位
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_DisableDcciListNotMatch_NoBit0AndBit5Set) {
  // disableDcciList 不匹配当前 kernelName，bit 0 不置位，bit 5 应置位
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"other_kernel"}));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44007, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.capBits.disableDcci = false;

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_EQ((funcTask.debugOptions & 0x1U), 0U);
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);
}

TEST_F(SkTaskBuilderTest, AddFuncTask_AllDebugBitsCombined) {
  // 组合所有 debugOptions bits：0, 3, 4, 5, 16, 验证 bit 5 正确计算
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                        std::vector<std::string>{"k"}));
  opts->AddOption(std::make_unique<StringListOptOption>(
      "dcci_before_kernel_start", aclskOptionType::DCCI_BEFORE_KERNEL_START, std::vector<std::string>{"k"}));
  opts->AddOption(std::make_unique<StringListOptOption>("dcci_after_kernel_end", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                        std::vector<std::string>{"k"}));
  opts->AddOption(std::make_unique<NumberOptOption>("debug_cross_core_sync_check",
                                                    aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1, 0, 1));

  SkTask aic;
  ASSERT_TRUE(aic.taskQue.Init(8));

  auto *kernel = CreateKernelNodeEx(44008, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);

  SkDfxInfo dfx{};
  ASSERT_TRUE(builder->AddFuncTask(aic, kernel, &dfx, 0, 0, 1, SkTaskType::TYPE_FUNC, 1));

  TaskInfo &funcTask = aic.taskQue.get()->taskInfos[aic.taskQue.get()->taskCnt - 1];
  EXPECT_NE((funcTask.debugOptions & 0x1U), 0U);   // dcci_disable
  EXPECT_NE((funcTask.debugOptions & 0x4U), 0U);   // before_kernel_start
  EXPECT_NE((funcTask.debugOptions & 0x8U), 0U);   // after_kernel_end
  EXPECT_NE((funcTask.debugOptions & 0x10U), 0U);  // cross_core_sync_check
  EXPECT_NE((funcTask.debugOptions & 0x20U), 0U);  // enable_dcci_after_func (after_end overrides disable)
}

TEST_F(SkTaskBuilderTest, ApplyEarlyStartSyncPass_RewritesAicAndAivSyncInfo) {
  auto *aicSet = CreateKernelNodeEx(43001, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *aicWait = CreateKernelNodeEx(43002, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *aivSet = CreateKernelNodeEx(43003, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *aivWait = CreateKernelNodeEx(43004, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  aicSet->nodeInfos.kernelInfos.capBits.earlyStartSetFlag = true;
  aicWait->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;
  aivSet->nodeInfos.kernelInfos.capBits.earlyStartSetFlag = true;
  aivWait->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;

  std::vector<SuperKernelBaseNode *> tasks = {aicSet, aicWait, aivSet, aivWait};
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[3].queueType = SkQueueType::AIV;

  builder->taskSyncInfos_[0].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::CUB_TO_CUB;
  builder->taskSyncInfos_[0].cubSendInfo[1] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[1].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[2].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)] = SyncDirection::VEC_TO_VEC;
  builder->taskSyncInfos_[2].vecSendInfo[3] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[3].vecRecvInfo[2] = SyncDirection::CUB_TO_VEC;

  ASSERT_TRUE(builder->ApplyEarlyStartSyncPass(tasks));

  const auto &firstAicInfo = builder->taskSyncInfos_[0].earlyStartInfo;
  EXPECT_EQ(firstAicInfo.relatedSetNode, aicSet);
  EXPECT_EQ(firstAicInfo.relatedWaitNode, nullptr);
  EXPECT_EQ(firstAicInfo.nextAicRelatedNode, aicWait);
  EXPECT_TRUE(firstAicInfo.CheckFuncMask(SkEarlyStartMask::AIC_TO_AIC_SET));
  EXPECT_TRUE(firstAicInfo.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIC_SET));
  EXPECT_TRUE(firstAicInfo.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIC_WAIT));
  EXPECT_TRUE(firstAicInfo.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIV_SET));
  EXPECT_TRUE(builder->taskSyncInfos_[0].crossSyncInfo.empty());
  EXPECT_TRUE(builder->taskSyncInfos_[0].cubSendInfo.empty());

  const auto &secondAicInfo = builder->taskSyncInfos_[1].earlyStartInfo;
  EXPECT_EQ(secondAicInfo.relatedSetNode, nullptr);
  EXPECT_EQ(secondAicInfo.relatedWaitNode, aicWait);
  EXPECT_TRUE(secondAicInfo.CheckFuncMask(SkEarlyStartMask::AIC_TO_AIC_WAIT));
  EXPECT_TRUE(secondAicInfo.CheckFuncMask(SkEarlyStartMask::AIC_TO_AIV_SET));
  EXPECT_TRUE(secondAicInfo.CheckFuncMask(SkEarlyStartMask::AIV_TO_AIC_WAIT));
  EXPECT_TRUE(secondAicInfo.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIC_WAIT));
  EXPECT_TRUE(builder->taskSyncInfos_[1].cubRecvInfo.empty());

  const auto &firstAivInfo = builder->taskSyncInfos_[2].earlyStartInfo;
  EXPECT_EQ(firstAivInfo.relatedSetNode, aivSet);
  EXPECT_EQ(firstAivInfo.relatedWaitNode, nullptr);
  EXPECT_EQ(firstAivInfo.nextAivRelatedNode, aivWait);
  EXPECT_TRUE(firstAivInfo.CheckFuncMask(SkEarlyStartMask::AIV_TO_AIV_SET));
  EXPECT_TRUE(firstAivInfo.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIV_SET));
  EXPECT_TRUE(firstAivInfo.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIV_WAIT));
  EXPECT_TRUE(firstAivInfo.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIC_SET));
  EXPECT_TRUE(builder->taskSyncInfos_[2].crossSyncInfo.empty());
  EXPECT_TRUE(builder->taskSyncInfos_[2].vecSendInfo.empty());

  const auto &secondAivInfo = builder->taskSyncInfos_[3].earlyStartInfo;
  EXPECT_EQ(secondAivInfo.relatedSetNode, nullptr);
  EXPECT_EQ(secondAivInfo.relatedWaitNode, aivWait);
  EXPECT_TRUE(secondAivInfo.CheckFuncMask(SkEarlyStartMask::AIV_TO_AIV_WAIT));
  EXPECT_TRUE(secondAivInfo.CheckFuncMask(SkEarlyStartMask::AIV_TO_AIC_SET));
  EXPECT_TRUE(secondAivInfo.CheckFuncMask(SkEarlyStartMask::AIC_TO_AIV_WAIT));
  EXPECT_TRUE(secondAivInfo.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIV_WAIT));
  EXPECT_TRUE(builder->taskSyncInfos_[3].vecRecvInfo.empty());
}

TEST_F(SkTaskBuilderTest, ApplyEarlyStartSyncPass_RewritesPreviousCrossSyncWithoutSetFlag) {
  auto *prev = CreateKernelNodeEx(43101, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *cur = CreateKernelNodeEx(43102, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  cur->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;

  std::vector<SuperKernelBaseNode *> tasks = {prev, cur};
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[0].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::CUB_TO_CUB;

  ASSERT_TRUE(builder->ApplyEarlyStartSyncPass(tasks));

  const auto &prevEarlyStart = builder->taskSyncInfos_[0].earlyStartInfo;
  const auto &curEarlyStart = builder->taskSyncInfos_[1].earlyStartInfo;
  EXPECT_EQ(prevEarlyStart.relatedSetNode, nullptr);
  EXPECT_EQ(curEarlyStart.relatedWaitNode, cur);
  EXPECT_EQ(prevEarlyStart.nextAicRelatedNode, cur);
  EXPECT_TRUE(prevEarlyStart.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIC_SET));
  EXPECT_TRUE(prevEarlyStart.CheckSyncMask(SkEarlyStartMask::AIC_TO_AIC_WAIT));
  EXPECT_TRUE(curEarlyStart.CheckFuncMask(SkEarlyStartMask::AIC_TO_AIC_WAIT));
  EXPECT_TRUE(builder->taskSyncInfos_[0].crossSyncInfo.empty());
}

TEST_F(SkTaskBuilderTest, ApplyEarlyStartSyncPass_AllSyncBarrierStopsCrossBarrierRewrite) {
  auto *prev = CreateKernelNodeEx(43111, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *barrier = CreateKernelNodeEx(43112, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *cur = CreateKernelNodeEx(43113, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  cur->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;

  std::vector<SuperKernelBaseNode *> tasks = {prev, barrier, cur};
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[2].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[0].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)] = SyncDirection::VEC_TO_VEC;
  builder->taskSyncInfos_[1].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;

  ASSERT_TRUE(builder->ApplyEarlyStartSyncPass(tasks));

  const auto &prevEarlyStart = builder->taskSyncInfos_[0].earlyStartInfo;
  const auto &barrierEarlyStart = builder->taskSyncInfos_[1].earlyStartInfo;
  const auto &curEarlyStart = builder->taskSyncInfos_[2].earlyStartInfo;
  EXPECT_EQ(prevEarlyStart.nextAivRelatedNode, nullptr);
  ASSERT_EQ(builder->taskSyncInfos_[0].crossSyncInfo.count(static_cast<size_t>(SkQueueType::AIV)), 1U);
  EXPECT_EQ(builder->taskSyncInfos_[0].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)], SyncDirection::VEC_TO_VEC);
  EXPECT_EQ(barrierEarlyStart.nextAivRelatedNode, cur);
  EXPECT_EQ(curEarlyStart.relatedWaitNode, cur);
  EXPECT_FALSE(curEarlyStart.CheckFuncMask(SkEarlyStartMask::AIV_TO_AIV_WAIT));
  EXPECT_FALSE(prevEarlyStart.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIV_SET));
  EXPECT_FALSE(prevEarlyStart.CheckSyncMask(SkEarlyStartMask::AIV_TO_AIV_WAIT));
}

TEST_F(SkTaskBuilderTest, ApplyEarlyStartSyncPass_NonKernelTaskDoesNotFail) {
  auto *node = CreateDefaultNodeEx(43201, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  std::vector<SuperKernelBaseNode *> tasks = {node};
  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;

  EXPECT_TRUE(builder->ApplyEarlyStartSyncPass(tasks));
}

TEST_F(SkTaskBuilderTest, ApplyEarlyStartSyncPass_MultipleSendOrRecvReturnsFalse) {
  auto *prev = CreateKernelNodeEx(43301, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *cur = CreateKernelNodeEx(43302, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  cur->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;
  std::vector<SuperKernelBaseNode *> tasks = {prev, cur};

  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[0].cubSendInfo[1] = SyncDirection::CUB_TO_VEC;
  builder->taskSyncInfos_[0].cubSendInfo[2] = SyncDirection::CUB_TO_VEC;
  EXPECT_FALSE(builder->ApplyEarlyStartSyncPass(tasks));

  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(tasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIC;
  builder->taskSyncInfos_[1].cubRecvInfo[0] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[1].cubRecvInfo[2] = SyncDirection::VEC_TO_CUB;
  EXPECT_FALSE(builder->ApplyEarlyStartSyncPass(tasks));

  auto *aivPrev = CreateKernelNodeEx(43303, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *aivCur = CreateKernelNodeEx(43304, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  aivCur->nodeInfos.kernelInfos.capBits.earlyStartWaitFlag = true;
  std::vector<SuperKernelBaseNode *> aivTasks = {aivPrev, aivCur};

  builder->taskSyncInfos_.clear();
  builder->taskSyncInfos_.resize(aivTasks.size());
  builder->taskSyncInfos_[0].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[1].queueType = SkQueueType::AIV;
  builder->taskSyncInfos_[0].vecSendInfo[1] = SyncDirection::VEC_TO_CUB;
  builder->taskSyncInfos_[0].vecSendInfo[2] = SyncDirection::VEC_TO_CUB;
  EXPECT_FALSE(builder->ApplyEarlyStartSyncPass(aivTasks));
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_EarlyStartMasksEnqueueTasks) {
  auto *related = CreateKernelNodeEx(43401, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  auto *nextAic = CreateKernelNodeEx(43402, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *nextAiv = CreateKernelNodeEx(43403, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  related->nodeInfos.kernelInfos.numBlocks = 2;
  nextAic->nodeInfos.kernelInfos.numBlocks = 3;
  nextAiv->nodeInfos.kernelInfos.numBlocks = 4;
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 4);

  EarlyStartInfo info;
  info.relatedSetNode = related;
  info.relatedWaitNode = related;
  info.nextAicRelatedNode = nextAic;
  info.nextAivRelatedNode = nextAiv;
  info.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIC_WAIT);
  info.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIV_WAIT);
  info.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIC_SET);
  info.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIC_WAIT);
  info.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIV_SET);
  info.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIV_SET);
  info.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIV_WAIT);
  info.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIC_SET);

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(16));
  ASSERT_TRUE(aiv.taskQue.Init(16));

  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, info, false, SkQueueType::AIC, scopeCoreInfo));
  EXPECT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, info, true, SkQueueType::AIC, scopeCoreInfo));
  ASSERT_EQ(aic.taskQue.get()->taskCnt, 4U);
  ASSERT_EQ(aiv.taskQue.get()->taskCnt, 4U);

  EXPECT_EQ(aic.taskQue.get()->taskInfos[0].relatedType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[0].numBlocks, 2U);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[0].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIC_WAIT));
  EXPECT_EQ(aic.taskQue.get()->taskInfos[1].relatedType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[1].numBlocks, 2U);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[1].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIC_SET));
  EXPECT_EQ(aic.taskQue.get()->taskInfos[2].relatedType, SkKernelType::AIC_ONLY);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[2].numBlocks, 3U);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[2].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIC_WAIT));
  EXPECT_EQ(aic.taskQue.get()->taskInfos[3].relatedType, SkKernelType::AIC_ONLY);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[3].numBlocks, 3U);
  EXPECT_EQ(aic.taskQue.get()->taskInfos[3].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIV_SET));

  EXPECT_EQ(aiv.taskQue.get()->taskInfos[0].relatedType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[0].numBlocks, 4U);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[0].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIC_TO_AIV_WAIT));
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[1].relatedType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[1].numBlocks, 4U);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[1].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIV_SET));
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[2].relatedType, SkKernelType::AIV_ONLY);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[2].numBlocks, 4U);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[2].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIV_WAIT));
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[3].relatedType, SkKernelType::AIV_ONLY);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[3].numBlocks, 4U);
  EXPECT_EQ(aiv.taskQue.get()->taskInfos[3].extraInfo, static_cast<uint64_t>(SkEarlyStartMask::AIV_TO_AIC_SET));

  auto *defaultNode = CreateDefaultNodeEx(43404, 0, INVALID_TASK_ID, INVALID_TASK_ID);
  EarlyStartInfo defaultInfo;
  defaultInfo.relatedWaitNode = defaultNode;
  defaultInfo.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIC_WAIT);
  defaultInfo.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIV_WAIT);

  SkTask defaultAic;
  SkTask defaultAiv;
  ASSERT_TRUE(defaultAic.taskQue.Init(4));
  ASSERT_TRUE(defaultAiv.taskQue.Init(4));
  EXPECT_TRUE(
      builder->DispatchSyncTasks(defaultAic, defaultAiv, 1, defaultInfo, false, SkQueueType::AIC, scopeCoreInfo));
}

TEST_F(SkTaskBuilderTest, GenEntryInfo_BlockDimScaleUpUpdatesRelatedEarlyStartSyncTask) {
  auto *scaled = CreateKernelNodeEx(43405, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *aicDriver = CreateKernelNodeEx(43406, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *aivDriver = CreateKernelNodeEx(43407, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  scaled->nodeInfos.kernelInfos.numBlocks = 6;
  scaled->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  aicDriver->nodeInfos.kernelInfos.numBlocks = 8;
  aivDriver->nodeInfos.kernelInfos.numBlocks = 12;
  std::vector<SuperKernelBaseNode *> tasks = {scaled, aicDriver, aivDriver};
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::MIX_AIC_1_2, 8);
  ASSERT_TRUE(builder->InitTaskSyncInfos(tasks));

  EarlyStartInfo &earlyStartInfo = builder->taskSyncInfos_[0].earlyStartInfo;
  earlyStartInfo.relatedSetNode = scaled;
  earlyStartInfo.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIC_SET);

  SkTask aic;
  SkTask aiv;
  ASSERT_TRUE(aic.taskQue.Init(6));
  ASSERT_TRUE(aiv.taskQue.Init(6));
  ASSERT_TRUE(builder->DispatchSyncTasks(aic, aiv, 0, earlyStartInfo, true, SkQueueType::AIC, scopeCoreInfo));
  SkDfxInfo dfxInfos[3]{};
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, scaled, &dfxInfos[0], 0, 1, SkTaskType::TYPE_FUNC, SkQueueType::AIC,
                                        scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, aicDriver, &dfxInfos[1], 1, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::AIC, scopeCoreInfo));
  ASSERT_TRUE(builder->DispatchFuncTask(aic, aiv, aivDriver, &dfxInfos[2], 2, 1, SkTaskType::TYPE_FUNC,
                                        SkQueueType::AIV, scopeCoreInfo));

  ASSERT_EQ(aic.GetTaskQue()->taskInfos[0].numBlocks, 8U);
  SkHostEntryInfo entryInfo = GenEntryInfoForTest(aic, aiv);

  ASSERT_NE(entryInfo.skEntryFunc, nullptr);
  EXPECT_EQ(entryInfo.entryType, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(aic.GetTaskQue()->taskInfos[0].numBlocks, 8U);
  EXPECT_EQ(aic.GetTaskQue()->taskInfos[0].relatedType, SkKernelType::AIC_ONLY);
}

TEST_F(SkTaskBuilderTest, DispatchSyncTasks_EarlyStartAddFailureReturnsFalse) {
  auto *related = CreateKernelNodeEx(43411, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto scopeCoreInfo = BuildCoreInfo(SkKernelType::AIC_ONLY, 1);
  EarlyStartInfo info;
  info.relatedWaitNode = related;

  SkTask missingAicQueue;
  SkTask aiv;
  ASSERT_TRUE(aiv.taskQue.Init(4));
  info.ApplySyncMask(SkEarlyStartMask::AIV_TO_AIC_WAIT);
  EXPECT_FALSE(builder->DispatchSyncTasks(missingAicQueue, aiv, 0, info, false, SkQueueType::AIC, scopeCoreInfo));

  EarlyStartInfo aivInfo;
  aivInfo.relatedWaitNode = related;
  aivInfo.ApplySyncMask(SkEarlyStartMask::AIC_TO_AIV_WAIT);
  SkTask aic;
  SkTask missingAivQueue;
  ASSERT_TRUE(aic.taskQue.Init(4));
  EXPECT_FALSE(builder->DispatchSyncTasks(aic, missingAivQueue, 0, aivInfo, false, SkQueueType::AIV, scopeCoreInfo));
}

TEST_F(SkTaskBuilderTest, Build_WithEarlyStartOption_Success) {
  opts->AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1, 1, 4));
  opts->AddOption(std::make_unique<NumberOptOption>(
      "early_start", aclskOptionType::EARLY_START, aclskEarlyStartValue::ACLSK_EARLY_START_ENABLED,
      aclskEarlyStartValue::ACLSK_EARLY_START_DISABLED, aclskEarlyStartValue::ACLSK_EARLY_START_ENABLED));

  auto *kernel = CreateKernelNodeEx(43501, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  std::vector<SuperKernelBaseNode *> tasks = {kernel};
  SkBuildResult buildResult = BuildWithCoreResult("Unknown", tasks, {}, 0);

  EXPECT_NE(buildResult.launchInfo.entryInfo.skEntryFunc, nullptr);
  EXPECT_NE(buildResult.launchInfo.devArgs.Get(), nullptr);
}
