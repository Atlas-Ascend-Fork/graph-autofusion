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

#include "dlog_pub.h"

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_options_manager.h"
#include "sk_scope_info.h"
#include "sk_task_ir.h"
#include "sk_task_lowering.h"

class SkTaskLoweringTest : public testing::Test {
 protected:
  SuperKernelBaseNode *AddKernel(uint64_t id, uint32_t stream, uint64_t previous, uint64_t next, SkKernelType type) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, stream, INVALID_STREAM_ID,
                                                        previous);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(id);
    node->SetPreNodeId(previous);
    node->SetNextNodeId(next);
    node->nodeInfos.kernelInfos.kernelType = type;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.cubeNum = type == SkKernelType::AIV_ONLY || type == SkKernelType::MIX_AIV_1_0 ? 0 : 1;
    node->nodeInfos.kernelInfos.vecNum = type == SkKernelType::AIC_ONLY || type == SkKernelType::MIX_AIC_1_0
                                             ? 0
                                             : (type == SkKernelType::MIX_AIC_1_2 ? 2 : 1);
    node->nodeInfos.kernelInfos.funcName = "kernel";
    auto *result = node.get();
    graph.graphMap[id] = std::move(node);
    return result;
  }

  SuperKernelBaseNode *AddEvent(uint64_t id, uint32_t stream, uint64_t previous, uint64_t next, SkNodeType type) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, stream,
                                                        INVALID_STREAM_ID, previous);
    node->SetNodeType(type);
    node->SetNodeId(id);
    node->SetPreNodeId(previous);
    node->SetNextNodeId(next);
    node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);
    auto *result = node.get();
    graph.graphMap[id] = std::move(node);
    return result;
  }

  bool RunLowering(const std::vector<SuperKernelBaseNode *> &tasks, SkKernelType coreType) {
    SkTaskLowering lowering(options, graph);
    return lowering.Run("test_sk", tasks, 7, ScopeCoreInfo{coreType, 1}, input);
  }

  SuperKernelGraph graph;
  SuperKernelOptionsManager options;
  SkTaskIR input;
};

TEST_F(SkTaskLoweringTest, MapsKernelTypesToResolvedQueues) {
  struct Case {
    SkKernelType kernelType;
    SkKernelType coreType;
    SkQueueType expected;
  };
  const std::vector<Case> cases = {
      {SkKernelType::AIC_ONLY, SkKernelType::AIC_ONLY, SkQueueType::AIC},
      {SkKernelType::AIV_ONLY, SkKernelType::AIV_ONLY, SkQueueType::AIV},
      {SkKernelType::MIX_AIC_1_1, SkKernelType::MIX_AIC_1_1, SkQueueType::MIX_1_1},
      {SkKernelType::MIX_AIC_1_2, SkKernelType::MIX_AIC_1_2, SkQueueType::MIX_1_2},
  };
  uint64_t id = 100;
  for (const auto &testCase : cases) {
    graph.graphMap.clear();
    auto *kernel = AddKernel(id++, 0, INVALID_TASK_ID, INVALID_TASK_ID, testCase.kernelType);
    ASSERT_TRUE(RunLowering({kernel}, testCase.coreType));
    EXPECT_EQ(input.tasks[0].queueType, testCase.expected);
  }
}

TEST_F(SkTaskLoweringTest, UsesCopiedDeviceArgsAndSize) {
  auto *kernel = AddKernel(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *kernelNode = static_cast<SuperKernelKernelNode *>(kernel);
  kernelNode->taskParams.kernelTaskParams.args = reinterpret_cast<void *>(0x1000);
  kernelNode->taskParams.kernelTaskParams.argsSize = 64;
  kernel->nodeInfos.kernelInfos.devArgs = reinterpret_cast<void *>(0x2000);
  kernel->nodeInfos.kernelInfos.devArgsSize = 96;

  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  ASSERT_EQ(input.tasks.size(), 1U);
  const auto &lowered = std::get<SkKernelDesc>(input.tasks[0].payload);
  EXPECT_EQ(lowered.deviceArgs, kernel->nodeInfos.kernelInfos.devArgs);
  EXPECT_EQ(lowered.argsSize, 96U);
}

TEST_F(SkTaskLoweringTest, GroupsLoweredKernelPropertiesByBuilderSemantics) {
  auto *kernel = AddKernel(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto &kernelInfo = kernel->nodeInfos.kernelInfos;
  kernelInfo.isScheModeOn = true;
  kernelInfo.capBits.blockDimScaleUp = true;
  kernelInfo.needMixKernelSplit = true;
  kernelInfo.capBits.earlyStartSetFlag = true;
  kernelInfo.capBits.earlyStartWaitFlag = true;
  kernelInfo.isSimtOp = true;
  kernelInfo.hasDynUbufSize = true;
  kernelInfo.hasAllocUbufSize = true;
  kernelInfo.dynUbufSize = 4096U;
  kernelInfo.allocUbufSize = 1024U;

  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  const auto &lowered = std::get<SkKernelDesc>(input.tasks[0].payload);
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, false));
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP, false));
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::KERNEL_MIX_SPLIT, false));
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::EARLY_START_SET, false));
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::EARLY_START_WAIT, false));
  EXPECT_TRUE(GetBoolAttribute(lowered.attributes, SkTaskAttributeId::SIMT_ENABLED, false));
  ASSERT_TRUE(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE).has_value());
  ASSERT_TRUE(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE).has_value());
  EXPECT_EQ(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE).value(), 4096U);
  EXPECT_EQ(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE).value(), 1024U);
}

TEST_F(SkTaskLoweringTest, ResolvesEventQueuesBeforeBuilder) {
  auto *aic = AddKernel(1, 0, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);
  auto *notify = AddEvent(2, 0, 1, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);
  auto *wait = AddEvent(3, 1, INVALID_TASK_ID, 4, SkNodeType::NODE_WAIT);
  auto *aiv = AddKernel(4, 1, 3, INVALID_TASK_ID, SkKernelType::AIV_ONLY);

  ASSERT_TRUE(RunLowering({aic, notify, wait, aiv}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[0].queueType, SkQueueType::AIC);
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIC);
  EXPECT_EQ(input.tasks[2].queueType, SkQueueType::AIV);
  EXPECT_EQ(input.tasks[3].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, MixWaitUsesPreviousScheduledKernelQueueOnSameStream) {
  auto *aic = AddKernel(10, 0, INVALID_TASK_ID, 11, SkKernelType::AIC_ONLY);
  auto *wait = AddEvent(11, 0, 10, 12, SkNodeType::NODE_WAIT);
  auto *mix = AddKernel(12, 0, 11, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);

  ASSERT_TRUE(RunLowering({aic, wait, mix}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskLoweringTest, UnifiedEventUsesKernelResourceAndFallback) {
  auto *aiv = AddKernel(20, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *event = AddEvent(21, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkNodeType::NODE_WAIT);

  ASSERT_TRUE(RunLowering({aiv, event}, SkKernelType::AIV_ONLY));
  ASSERT_EQ(input.tasks.size(), 2U);
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, CopiesSuccessorsForEveryTaskKindInStableOrder) {
  auto *kernel = AddKernel(22, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *notify = AddEvent(23, 0, 22, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);
  auto *wait = AddEvent(24, 0, INVALID_TASK_ID, 22, SkNodeType::NODE_WAIT);
  auto *reset = AddEvent(25, 0, 22, INVALID_TASK_ID, SkNodeType::NODE_RESET);
  for (auto *task : {kernel, notify, wait, reset}) {
    task->sendToNodeId.insert(100);
    task->sendToNodeId.insert(90);
  }
  kernel->sendToNodeId.insert(90);
  kernel->sendToNodeId.insert(100);

  ASSERT_TRUE(RunLowering({kernel, notify, wait, reset}, SkKernelType::AIC_ONLY));
  ASSERT_EQ(input.tasks.size(), 4U);
  for (const auto &task : input.tasks) {
    EXPECT_EQ(task.successorIds, (std::vector<uint64_t>{90, 100}));
  }
}

TEST_F(SkTaskLoweringTest, ResetUsesPreviousKernelQueue) {
  auto *aic = AddKernel(30, 0, INVALID_TASK_ID, 31, SkKernelType::AIC_ONLY);
  auto *reset = AddEvent(31, 0, 30, 32, SkNodeType::NODE_RESET);
  auto *aiv = AddKernel(32, 0, 31, INVALID_TASK_ID, SkKernelType::AIV_ONLY);

  ASSERT_TRUE(RunLowering({aic, reset, aiv}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskLoweringTest, EventWithoutGraphNeighborUsesFirstKernelQueue) {
  auto *aiv = AddKernel(40, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  auto *reset = AddEvent(41, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkNodeType::NODE_RESET);

  ASSERT_TRUE(RunLowering({aiv, reset}, SkKernelType::AIV_ONLY));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, SearchBeyond100HopsUsesFirstKernelFallback) {
  auto *fallback = AddKernel(300, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *distant = AddKernel(301, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  uint64_t predecessor = distant->GetNodeId();
  for (uint64_t id = 400; id < 501; ++id) {
    auto *reset = AddEvent(id, 1, predecessor, INVALID_TASK_ID, SkNodeType::NODE_RESET);
    predecessor = reset->GetNodeId();
  }
  auto *notify = AddEvent(600, 1, predecessor, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);

  ASSERT_TRUE(RunLowering({fallback, notify, distant}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskLoweringTest, ReusesKernelSearchCacheWithin100Hops) {
  auto *fallback = AddKernel(700, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *distant = AddKernel(701, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  uint64_t predecessor = distant->GetNodeId();
  for (uint64_t id = 1000; id < 1180; ++id) {
    auto *reset = AddEvent(id, 1, predecessor, INVALID_TASK_ID, SkNodeType::NODE_RESET);
    predecessor = reset->GetNodeId();
  }
  auto *nearNotify = AddEvent(800, 1, 1079, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);
  auto *farNotify = AddEvent(801, 1, 1178, INVALID_TASK_ID, SkNodeType::NODE_NOTIFY);

  ASSERT_TRUE(RunLowering({fallback, nearNotify, farNotify, distant}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
  EXPECT_EQ(input.tasks[2].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, RejectsNullTask) {
  EXPECT_FALSE(RunLowering({nullptr}, SkKernelType::AIC_ONLY));
}

TEST_F(SkTaskLoweringTest, FailureClearsPreviouslyLoweredOutput) {
  auto *kernel = AddKernel(900, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  ASSERT_FALSE(input.tasks.empty());

  EXPECT_FALSE(RunLowering({nullptr}, SkKernelType::AIC_ONLY));
  EXPECT_TRUE(input.tasks.empty());
  EXPECT_TRUE(input.functionName.empty());
}

TEST_F(SkTaskLoweringTest, EventOnlyInputHasNoQueueFallback) {
  auto *reset = AddEvent(50, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkNodeType::NODE_RESET);

  EXPECT_FALSE(RunLowering({reset}, SkKernelType::AIC_ONLY));
}

TEST_F(SkTaskLoweringTest, MixWaitUsesSameAivQueueOnSameStream) {
  auto *aiv = AddKernel(60, 0, INVALID_TASK_ID, 61, SkKernelType::AIV_ONLY);
  auto *wait = AddEvent(61, 0, 60, 62, SkNodeType::NODE_WAIT);
  auto *mix = AddKernel(62, 0, 61, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);

  ASSERT_TRUE(RunLowering({aiv, wait, mix}, SkKernelType::MIX_AIC_1_2));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, MixWaitUsesDirectDependencyQueueAcrossStreams) {
  auto *aic = AddKernel(70, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *wait = AddEvent(71, 1, 70, 72, SkNodeType::NODE_WAIT);
  auto *mix = AddKernel(72, 1, 71, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);
  aic->sendToNodeId.insert(72);
  mix->receiveNodeId.insert(70);

  ASSERT_TRUE(RunLowering({aic, wait, mix}, SkKernelType::MIX_AIC_1_2));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIC);
}

TEST_F(SkTaskLoweringTest, MixWaitUsesOppositeQueueWithoutDependency) {
  auto *aic = AddKernel(80, 1, INVALID_TASK_ID, 81, SkKernelType::AIC_ONLY);
  auto *wait = AddEvent(81, 1, 80, 82, SkNodeType::NODE_WAIT);
  auto *mix = AddKernel(82, 0, 81, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_1);

  ASSERT_TRUE(RunLowering({aic, wait, mix}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, MixWaitAfterMixKernelFallsBackToAiv) {
  auto *mix11 = AddKernel(90, 0, INVALID_TASK_ID, 91, SkKernelType::MIX_AIC_1_1);
  auto *wait = AddEvent(91, 0, 90, 92, SkNodeType::NODE_WAIT);
  auto *mix12 = AddKernel(92, 0, 91, INVALID_TASK_ID, SkKernelType::MIX_AIC_1_2);

  ASSERT_TRUE(RunLowering({mix11, wait, mix12}, SkKernelType::MIX_AIC_1_2));
  EXPECT_EQ(input.tasks[1].queueType, SkQueueType::AIV);
}

TEST_F(SkTaskLoweringTest, LowersPreloadModesToResolvedFunctionDescriptors) {
  auto *kernel = AddKernel(100, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[0] = 9;
  kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[1] = 7;
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE,
                                                      ACLSK_PRELOAD_PER_FUNC, ACLSK_PRELOAD_MAX,
                                                      ACLSK_PRELOAD_PER_FUNC_MIN_AIV));
  struct Case {
    aclskPreloadMode mode;
    std::array<uint64_t, kSkCoreTypeCount> expected;
  };
  const std::vector<Case> cases = {
      {ACLSK_PRELOAD_MAX, {16, 8}}, {ACLSK_PRELOAD_PER_FUNC, {9, 7}},          {ACLSK_PRELOAD_DISABLED, {0, 0}},
      {ACLSK_PRELOAD_MIN, {1, 1}},  {ACLSK_PRELOAD_PER_FUNC_HALF_AIV, {9, 4}}, {ACLSK_PRELOAD_PER_FUNC_MIN_AIV, {9, 1}},
  };
  for (const auto &testCase : cases) {
    SCOPED_TRACE(static_cast<uint32_t>(testCase.mode));
    options.GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(testCase.mode);
    ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
    EXPECT_EQ(std::get<SkKernelDesc>(input.tasks[0].payload).functions[0].prefetchCount, testCase.expected);
  }
}

TEST_F(SkTaskLoweringTest, FunctionSizedPreloadModesRespectHardwareLimits) {
  auto *kernel = AddKernel(100, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[0] = 20;
  kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[1] = 18;
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE,
                                                      ACLSK_PRELOAD_PER_FUNC, ACLSK_PRELOAD_MAX,
                                                      ACLSK_PRELOAD_PER_FUNC_MIN_AIV));
  struct Case {
    aclskPreloadMode mode;
    std::array<uint64_t, kSkCoreTypeCount> expected;
  };
  const std::vector<Case> cases = {
      {ACLSK_PRELOAD_PER_FUNC, {16, 8}},
      {ACLSK_PRELOAD_PER_FUNC_HALF_AIV, {16, 4}},
      {ACLSK_PRELOAD_PER_FUNC_MIN_AIV, {16, 1}},
  };
  for (const auto &testCase : cases) {
    SCOPED_TRACE(static_cast<uint32_t>(testCase.mode));
    options.GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(testCase.mode);
    ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
    EXPECT_EQ(std::get<SkKernelDesc>(input.tasks[0].payload).functions[0].prefetchCount, testCase.expected);
  }
}

TEST_F(SkTaskLoweringTest, ReducedAivPreloadModesHandleZeroOneAndEvenCounts) {
  auto *kernel = AddKernel(100, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[0] = 3;
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<NumberOptOption>("preload_code", aclskOptionType::PRELOAD_CODE,
                                                      ACLSK_PRELOAD_PER_FUNC, ACLSK_PRELOAD_MAX,
                                                      ACLSK_PRELOAD_PER_FUNC_MIN_AIV));
  struct Case {
    uint64_t aivCount;
    uint64_t halfAivCount;
    uint64_t minAivCount;
  };
  const std::vector<Case> cases = {{0, 0, 0}, {1, 1, 1}, {4, 2, 1}};
  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.aivCount);
    kernel->nodeInfos.kernelInfos.resolvedFuncs[0].prefetchCnt[1] = testCase.aivCount;
    options.GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(ACLSK_PRELOAD_PER_FUNC_HALF_AIV);
    ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
    EXPECT_EQ(std::get<SkKernelDesc>(input.tasks[0].payload).functions[0].prefetchCount,
              (std::array<uint64_t, kSkCoreTypeCount>{3, testCase.halfAivCount}));
    options.GetOption(aclskOptionType::PRELOAD_CODE)->SetValue(ACLSK_PRELOAD_PER_FUNC_MIN_AIV);
    ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
    EXPECT_EQ(std::get<SkKernelDesc>(input.tasks[0].payload).functions[0].prefetchCount,
              (std::array<uint64_t, kSkCoreTypeCount>{3, testCase.minAivCount}));
  }
}

TEST_F(SkTaskLoweringTest, LowersDcciCapabilityAndOptionBits) {
  auto *kernel = AddKernel(110, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.funcName = "matched_kernel";
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<StringListOptOption>("dcci_disable", aclskOptionType::DCCI_DISABLE_ON_KERNEL,
                                                          std::vector<std::string>{"matched.*"}));

  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  const auto &lowered = std::get<SkKernelDesc>(input.tasks[0].payload);
  ASSERT_TRUE(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS).has_value());
  EXPECT_EQ(GetUint64Attribute(lowered.attributes, SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS).value(), 0x29U);

  kernel->nodeInfos.kernelInfos.capBits.disableDcci = true;
  kernel->nodeInfos.kernelInfos.funcName = "unmatched_kernel";
  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  EXPECT_EQ(GetUint64Attribute(std::get<SkKernelDesc>(input.tasks[0].payload).attributes,
                               SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS)
                .value(),
            0x1U);
}

TEST_F(SkTaskLoweringTest, LowersDcciBeforeAfterAndCrossCoreDebugBits) {
  auto *kernel = AddKernel(120, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  kernel->nodeInfos.kernelInfos.funcName = "debug_kernel";
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<StringListOptOption>("dcci_before", aclskOptionType::DCCI_BEFORE_KERNEL_START,
                                                          std::vector<std::string>{"debug.*"}));
  options.AddOption(std::make_unique<StringListOptOption>("dcci_after", aclskOptionType::DCCI_AFTER_KERNEL_END,
                                                          std::vector<std::string>{"debug.*"}));
  options.AddOption(std::make_unique<NumberOptOption>("cross_core", aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK, 1));

  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  const auto debugOptions = GetUint64Attribute(std::get<SkKernelDesc>(input.tasks[0].payload).attributes,
                                               SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS);
  ASSERT_TRUE(debugOptions.has_value());
  EXPECT_EQ(debugOptions.value(), 0x3CU);
  EXPECT_TRUE(
      GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_CROSS_CORE_SYNC_CHECK_ENABLED, false));
  EXPECT_TRUE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED, false));
}

TEST_F(SkTaskLoweringTest, LowersBuilderConfigurationOptions) {
  auto *kernel = AddKernel(130, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  options.AddOption(std::make_unique<NumberOptOption>("split_mode", aclskOptionType::SPLIT_MODE, 1));
  options.AddOption(std::make_unique<NumberOptOption>("debug_sync_all", aclskOptionType::DEBUG_SYNC_ALL, 1));
  options.AddOption(std::make_unique<NumberOptOption>("stream_fusion", aclskOptionType::STREAM_FUSION, 0));
  options.AddOption(std::make_unique<NumberOptOption>("early_start", aclskOptionType::EARLY_START, 1));
  options.AddOption(std::make_unique<NumberOptOption>("op_trace", aclskOptionType::DEBUG_OP_EXEC_TRACE, 1));

  ASSERT_TRUE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  EXPECT_EQ(input.config.splitBinCount, 1U);
  EXPECT_TRUE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_ENABLED, false));
  EXPECT_TRUE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL, false));
  EXPECT_FALSE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_STREAM_FUSION_ENABLED, true));
  EXPECT_TRUE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED, false));
  EXPECT_TRUE(GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED, false));
}

TEST_F(SkTaskLoweringTest, RejectsUnsupportedNodeAndCoreMismatch) {
  auto *kernel = AddKernel(140, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  ut_log::LogBuffer::Instance().Clear();
  EXPECT_FALSE(RunLowering({kernel}, SkKernelType::AIC_ONLY));
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "kernel 140 core requirement does not match final SK result, taskCore={cube:0,vec:1}, "
                "scheMode=0, blockDimScaleUp=0, skCore={cube:1,vec:0}, skType=AIC_ONLY"),
            std::string::npos);

  kernel->SetNodeType(SkNodeType::NODE_DEFAULT);
  EXPECT_FALSE(RunLowering({kernel}, SkKernelType::AIV_ONLY));
}

TEST_F(SkTaskLoweringTest, PreservesMissingReverseRelationWarning) {
  auto *producer = AddKernel(150, 0, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIC_ONLY);
  auto *consumer = AddKernel(151, 1, INVALID_TASK_ID, INVALID_TASK_ID, SkKernelType::AIV_ONLY);
  producer->sendToNodeId.insert(151);
  ut_log::LogBuffer::Instance().Clear();

  ASSERT_TRUE(RunLowering({producer, consumer}, SkKernelType::MIX_AIC_1_1));
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(
                "ValidateGraphRelations: reverse relation missing for edge 150 -> 151"),
            std::string::npos);

  consumer->receiveNodeId.insert(150);
  ut_log::LogBuffer::Instance().Clear();
  ASSERT_TRUE(RunLowering({producer, consumer}, SkKernelType::MIX_AIC_1_1));
  EXPECT_EQ(ut_log::LogBuffer::Instance().GetContent().find("reverse relation missing"), std::string::npos);
}
