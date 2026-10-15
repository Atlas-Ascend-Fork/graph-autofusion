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
#include <gmock/gmock.h>
#include <memory>
#include <limits>
#include <vector>

#define private public
#define protected public
#include "super_kernel.h"
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_lock_detector.h"
#include "sk_options_manager.h"
#include "dlog_pub.h"
#include "ut_common_stubs.h"

namespace {

struct TestRITask {
  uint32_t taskId;
  aclmdlRITaskType type;
  aclmdlRITaskParams params;
};

std::unique_ptr<aclmdlRITask> MakeTaskHandle(TestRITask &task) {
  return std::make_unique<aclmdlRITask>(reinterpret_cast<aclmdlRITask>(&task));
}

}  // namespace

/**
 * @brief Test fixture class for SuperKernelGraph unit tests
 */
class SuperKernelGraphTest : public testing::Test {
 protected:
  void SetUp() override {
    opts = std::make_unique<SuperKernelOptionsManager>();
    graph = std::make_unique<SuperKernelGraph>(nullptr, *opts);
  }

  void TearDown() override {
    graph.reset();
    opts.reset();
  }

  void ResetGraph() {
    graph = std::make_unique<SuperKernelGraph>(nullptr, *opts);
  }

  void ConfigureValueBreakerBypass(uint32_t value) {
    aclskOption option{};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.valueBreakerBypass = value;
    opts->SetOptOptionValue(&option);
  }

  SuperKernelMemoryNode *CreateMemoryNode(uint64_t nodeId, SkNodeType nodeType, uint64_t eventId, uint64_t memoryValue,
                                          uint32_t waitFlag = std::numeric_limits<uint32_t>::max()) {
    const auto taskType =
        (nodeType == SkNodeType::NODE_MEMORY_WAIT) ? ACL_MODEL_RI_TASK_VALUE_WAIT : ACL_MODEL_RI_TASK_VALUE_WRITE;
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, taskType, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(nodeType);
    node->SetIsFusible(false);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1000 + nodeId);
    node->nodeInfos.syncInfos.memoryValue = memoryValue;
    node->nodeInfos.syncInfos.memoryWaitFlag = waitFlag;
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelMemoryNode *CreateEventNodeWithTask(TestRITask &task, uint64_t nodeId, SkNodeType nodeType,
                                                 aclmdlRITaskType taskType, void *addrValue) {
    task.taskId = static_cast<uint32_t>(nodeId);
    task.type = taskType;
    auto node = std::make_unique<SuperKernelMemoryNode>(MakeTaskHandle(task), taskType, 0, 0, INVALID_STREAM_ID,
                                                        INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNodeType(nodeType);
    node->nodeInfos.syncInfos.addrValue = addrValue;
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  std::unique_ptr<SuperKernelOptionsManager> opts;
  std::unique_ptr<SuperKernelGraph> graph;
};

// ==================== GetModelId 测试 ====================

TEST_F(SuperKernelGraphTest, GetModelId_DefaultConstructedIsEmpty) {
  EXPECT_TRUE(graph->GetModelId().empty());
}

TEST_F(SuperKernelGraphTest, GetModelId_ReturnsExplicitConstructorValue) {
  graph = std::make_unique<SuperKernelGraph>(nullptr, *opts, "model_42_3");
  EXPECT_EQ(graph->GetModelId(), "model_42_3");
}

TEST_F(SuperKernelGraphTest, GetModelId_MoveConstructorPreservesValue) {
  SuperKernelGraph source(nullptr, *opts, "model_77_1");
  SuperKernelGraph target(std::move(source));
  EXPECT_EQ(target.GetModelId(), "model_77_1");
}

// ==================== GetSortedNodeIds Empty Graph Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_EmptyGraph) {
  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_TRUE(sortedIds.empty());
  EXPECT_EQ(sortedIds.size(), 0);
}

// ==================== GetSortedNodeIds Single Node Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNode) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(5);
  graph->graphMap[5] = std::move(node);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 1);
  EXPECT_EQ(sortedIds[0], 5);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNodeWithMinId) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(0);
  graph->graphMap[0] = std::move(node);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 1);
  EXPECT_EQ(sortedIds[0], 0);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_SingleNodeWithMaxId) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(100);
  graph->graphMap[100] = std::move(node);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 1);
  EXPECT_EQ(sortedIds[0], 100);
}

// ==================== GetSortedNodeIds Multiple Nodes Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_Sequential) {
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
  EXPECT_EQ(sortedIds[2], 3);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_ReverseOrder) {
  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
  EXPECT_EQ(sortedIds[2], 3);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleNodes_RandomOrder) {
  graph->graphMap[5] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[8] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[10] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 5);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
  EXPECT_EQ(sortedIds[2], 5);
  EXPECT_EQ(sortedIds[3], 8);
  EXPECT_EQ(sortedIds[4], 10);
}

// ==================== GetSortedNodeIds Boundary Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_ZeroId) {
  graph->graphMap[0] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 0);
  EXPECT_EQ(sortedIds[1], 1);
  EXPECT_EQ(sortedIds[2], 2);
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_LargeIdGap) {
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1000] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                  INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[500] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                 INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 500);
  EXPECT_EQ(sortedIds[2], 1000);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWriteTurnsIntoNotifyAndKeepsFusible) {
  constexpr uint64_t kEventId = 0x101;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 7);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  graph->BuildEventNodeAssociations();

  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_TRUE(writeNode->IsFusible());
  EXPECT_EQ(graph->eventToNodes[kEventId].notifyNodeId, writeNode->GetNodeId());
  EXPECT_NE(writeNode->GetNodeInfos().syncInfos.addrValue, nullptr);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWaitTurnsIntoWaitAndUnfusible) {
  constexpr uint64_t kEventId = 0x151;
  auto *waitNode =
      CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WAIT, kEventId, 5, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode->IsFusible());
  ASSERT_NE(graph->GetEventInfo(kEventId), nullptr);
  EXPECT_EQ(graph->GetEventInfo(kEventId)->waitNodeIdList.count(waitNode->GetNodeId()), 1);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_ValueBreakerBypassPairedWaitDisabledBlocksFusion) {
  constexpr uint64_t kEventId = 0x202;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(writeNode->IsFusible());
  EXPECT_FALSE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_NoNotifyAfterRuleCheckKeepsWaitUnfusibleWithoutBypass) {
  constexpr uint64_t kEventId = 0x203;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(writeNode->IsFusible());
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_ValueBreakerBypassPairedWaitEnabledConvertsMemoryEvent) {
  constexpr uint64_t kEventId = 0x202;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(writeNode->IsFusible());
  EXPECT_TRUE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_ValueBreakerBypassPairedWaitEnabledConvertsReset) {
  constexpr uint64_t kEventId = 0x204;
  auto *notifyNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  auto *resetNode = CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  auto *waitNode =
      CreateMemoryNode(3, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(notifyNode->GetNodeId());
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(resetNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(notifyNode->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_TRUE(notifyNode->IsFusible());
  EXPECT_EQ(resetNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(resetNode->IsFusible());
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_PairedBypassDoesNotFuseWaitWithoutNotify) {
  constexpr uint64_t kEventId = 0x233;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(writeNode->IsFusible());
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_UnpairedBypassFusesWaitWithoutNotifyAfterRuleCheck) {
  constexpr uint64_t kEventId = 0x234;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(writeNode->IsFusible());
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Boundary_NonContiguousIds) {
  graph->graphMap[10] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[20] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[30] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 10);
  EXPECT_EQ(sortedIds[1], 20);
  EXPECT_EQ(sortedIds[2], 30);
}

// ==================== GetSortedNodeIds Multiple Calls Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleCalls_SameGraph) {
  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds1 = graph->GetSortedNodeIds();
  auto sortedIds2 = graph->GetSortedNodeIds();

  EXPECT_EQ(sortedIds1.size(), sortedIds2.size());
  for (size_t i = 0; i < sortedIds1.size(); ++i) {
    EXPECT_EQ(sortedIds1[i], sortedIds2[i]);
  }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MultipleCalls_AfterModification) {
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds1 = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds1.size(), 2);
  EXPECT_EQ(sortedIds1[0], 1);
  EXPECT_EQ(sortedIds1[1], 2);

  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds2 = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds2.size(), 3);
  EXPECT_EQ(sortedIds2[0], 1);
  EXPECT_EQ(sortedIds2[1], 2);
  EXPECT_EQ(sortedIds2[2], 3);
}

// ==================== GetSortedNodeIds Mixed Node Types Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_MixedNodeTypes) {
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[3] = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
  EXPECT_EQ(sortedIds[2], 3);
}

// ==================== GetSortedNodeIds Large Scale Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_LargeScale) {
  const uint32_t numNodes = 100;
  for (uint32_t i = 0; i < numNodes; ++i) {
    uint64_t nodeId = numNodes - i;  // Insert in reverse order
    graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                      INVALID_STREAM_ID, INVALID_TASK_ID);
  }

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), numNodes);

  for (uint32_t i = 0; i < numNodes; ++i) {
    EXPECT_EQ(sortedIds[i], i + 1);
  }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_LargeScale_RandomInsertion) {
  const uint32_t numNodes = 50;
  std::vector<uint64_t> nodeIds = {5,  23, 1,  45, 12, 8,  33, 17, 50, 3,  27, 9,  41, 2,  37, 19, 48,
                                   6,  30, 14, 25, 11, 39, 4,  35, 20, 44, 7,  29, 13, 49, 10, 42, 18,
                                   36, 15, 46, 22, 38, 26, 47, 21, 40, 24, 34, 16, 43, 31, 32, 28};

  for (uint64_t nodeId : nodeIds) {
    graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                      INVALID_STREAM_ID, INVALID_TASK_ID);
  }

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), numNodes);

  std::vector<uint64_t> expectedIds = nodeIds;
  std::sort(expectedIds.begin(), expectedIds.end());

  for (uint32_t i = 0; i < numNodes; ++i) {
    EXPECT_EQ(sortedIds[i], expectedIds[i]);
  }
}

// ==================== GetSortedNodeIds Const Correctness Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_ConstMethod) {
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  const SuperKernelGraph *constGraph = graph.get();
  auto sortedIds = constGraph->GetSortedNodeIds();

  EXPECT_EQ(sortedIds.size(), 2);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
}

// ==================== GetSortedNodeIds Duplicate Values Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_NoDuplicateIds) {
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);

  std::sort(sortedIds.begin(), sortedIds.end());
  auto last = std::unique(sortedIds.begin(), sortedIds.end());
  EXPECT_EQ(last, sortedIds.end());
}

// ==================== GetSortedNodeIds Integrity Tests ====================

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Integrity_AllIdsPresent) {
  std::vector<uint64_t> expectedIds = {5, 10, 15, 20, 25};

  for (uint64_t nodeId : expectedIds) {
    graph->graphMap[nodeId] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                                      INVALID_STREAM_ID, INVALID_TASK_ID);
  }

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), expectedIds.size());

  for (uint64_t expectedId : expectedIds) {
    auto it = std::find(sortedIds.begin(), sortedIds.end(), expectedId);
    EXPECT_NE(it, sortedIds.end());
  }
}

TEST_F(SuperKernelGraphTest, GetSortedNodeIds_Integrity_NoExtraIds) {
  graph->graphMap[3] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[1] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  graph->graphMap[2] = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);

  auto sortedIds = graph->GetSortedNodeIds();
  EXPECT_EQ(sortedIds.size(), 3);
  EXPECT_EQ(sortedIds[0], 1);
  EXPECT_EQ(sortedIds[1], 2);
  EXPECT_EQ(sortedIds[2], 3);
}

// ==================== BitsetToString Tests ====================

TEST_F(SuperKernelGraphTest, BitsetToString_EmptyScopeNameToIdx) {
  graph->scopeNameToIdx.clear();
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(0);
  std::string result = graph->BitsetToString(flags);
  EXPECT_EQ(result, "0");
}

TEST_F(SuperKernelGraphTest, BitsetToString_WithScopeNames) {
  graph->scopeNameToIdx["scope_a"] = 0;
  graph->scopeNameToIdx["scope_b"] = 1;
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(0);
  flags.set(1);
  std::string result = graph->BitsetToString(flags);
  EXPECT_EQ(result.size(), 2);
}

TEST_F(SuperKernelGraphTest, BitsetToString_SingleFlag) {
  graph->scopeNameToIdx["scope_a"] = 0;
  std::bitset<MAX_SCOPE_NUM> flags;
  flags.set(0);
  std::string result = graph->BitsetToString(flags);
  EXPECT_EQ(result, "1");
}

// ==================== GetNodeById Tests ====================

TEST_F(SuperKernelGraphTest, GetNodeById_ExistingNode) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(5);
  graph->graphMap[5] = std::move(node);

  auto *retrievedNode = graph->GetNodeById(5);
  EXPECT_NE(retrievedNode, nullptr);
  EXPECT_EQ(retrievedNode->GetNodeId(), 5);
}

TEST_F(SuperKernelGraphTest, GetNodeById_NonExistingNode) {
  auto *retrievedNode = graph->GetNodeById(999);
  EXPECT_EQ(retrievedNode, nullptr);
}

// ==================== AddNode Tests ====================

TEST_F(SuperKernelGraphTest, AddNode_KernelNode) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);

  bool result = graph->AddNode(std::move(node));
  EXPECT_TRUE(result);
  EXPECT_EQ(graph->graphMap.size(), 1);
  EXPECT_NE(graph->GetNodeById(10), nullptr);
}

TEST_F(SuperKernelGraphTest, AddNode_DuplicateNodeId) {
  auto node1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node1->SetNodeId(10);
  graph->graphMap[10] = std::move(node1);

  auto node2 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node2->SetNodeId(10);

  bool result = graph->AddNode(std::move(node2));
  EXPECT_FALSE(result);
  EXPECT_EQ(graph->graphMap.size(), 1);
}

// ==================== ExpandUpdateNodes Tests ====================

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_EmptyList) {
  std::vector<SuperKernelBaseNode *> customNodes;
  bool result = graph->ExpandUpdateNodes(customNodes);
  EXPECT_TRUE(result);
  EXPECT_EQ(graph->needUpdateNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_AddNodes) {
  auto node1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node1->SetNodeId(10);
  SuperKernelBaseNode *node1Ptr = node1.get();
  graph->graphMap[10] = std::move(node1);

  auto node2 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node2->SetNodeId(20);
  SuperKernelBaseNode *node2Ptr = node2.get();
  graph->graphMap[20] = std::move(node2);

  std::vector<SuperKernelBaseNode *> customNodes = {node1Ptr, node2Ptr};
  bool result = graph->ExpandUpdateNodes(customNodes);
  EXPECT_TRUE(result);
  EXPECT_EQ(graph->needUpdateNodes.size(), 2);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_NullNode) {
  std::vector<SuperKernelBaseNode *> customNodes = {nullptr};
  bool result = graph->ExpandUpdateNodes(customNodes);
  EXPECT_TRUE(result);
  EXPECT_EQ(graph->needUpdateNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, ExpandUpdateNodes_DuplicateAdd) {
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  std::vector<SuperKernelBaseNode *> customNodes1 = {nodePtr};
  graph->ExpandUpdateNodes(customNodes1);
  EXPECT_EQ(graph->needUpdateNodes.size(), 1);

  std::vector<SuperKernelBaseNode *> customNodes2 = {nodePtr};
  graph->ExpandUpdateNodes(customNodes2);
  EXPECT_EQ(graph->needUpdateNodes.size(), 1);
}

// ==================== GetStreamByIndex Tests ====================

TEST_F(SuperKernelGraphTest, GetStreamByIndex_ValidIndex) {
  graph->streams.resize(3);
  graph->streams[0] = reinterpret_cast<aclrtStream>(0x100);
  graph->streams[1] = reinterpret_cast<aclrtStream>(0x200);
  graph->streams[2] = reinterpret_cast<aclrtStream>(0x300);

  auto *stream = graph->GetStreamByIndex(1);
  EXPECT_EQ(stream, reinterpret_cast<aclrtStream>(0x200));
}

TEST_F(SuperKernelGraphTest, GetStreamByIndex_OutOfBounds) {
  graph->streams.resize(2);
  auto *stream = graph->GetStreamByIndex(5);
  EXPECT_EQ(stream, nullptr);
}

// ==================== GetScopeNameByIdx Tests ====================

TEST_F(SuperKernelGraphTest, GetScopeNameByIdx_ExistingIdx) {
  graph->scopeIdxToName[0] = "scope_a";
  graph->scopeIdxToName[1] = "scope_b";

  std::string scopeName;
  bool result = graph->GetScopeNameByIdx(0, scopeName);
  EXPECT_TRUE(result);
  EXPECT_EQ(scopeName, "scope_a");
}

TEST_F(SuperKernelGraphTest, GetScopeNameByIdx_NonExistingIdx) {
  graph->scopeIdxToName[0] = "scope_a";

  std::string scopeName;
  bool result = graph->GetScopeNameByIdx(5, scopeName);
  EXPECT_FALSE(result);
}

// ==================== GetOriginalScopeInfos Tests ====================

TEST_F(SuperKernelGraphTest, GetOriginalScopeInfos_Empty) {
  const auto &scopeInfos = graph->GetOriginalScopeInfos();
  EXPECT_EQ(scopeInfos.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetOriginalScopeInfos_WithData) {
  OriginalScopeInfo info;
  info.scopeId = 0;
  info.nodeIds = {1, 2, 3};
  graph->originalScopeInfos_.push_back(info);

  const auto &scopeInfos = graph->GetOriginalScopeInfos();
  EXPECT_EQ(scopeInfos.size(), 1);
  EXPECT_EQ(scopeInfos[0].nodeIds.size(), 3);
}

// ==================== GetStreams Tests ====================

TEST_F(SuperKernelGraphTest, GetStreams_Empty) {
  const auto &streams = graph->GetStreams();
  EXPECT_EQ(streams.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetStreams_WithStreams) {
  graph->streams.resize(3);
  graph->streams[0] = reinterpret_cast<aclrtStream>(0x100);
  graph->streams[1] = reinterpret_cast<aclrtStream>(0x200);
  graph->streams[2] = reinterpret_cast<aclrtStream>(0x300);

  const auto &streams = graph->GetStreams();
  EXPECT_EQ(streams.size(), 3);
}

TEST_F(SuperKernelGraphTest, InitFromModelRI_AllStreamsEmptyReturnsTrue) {
  SkUtResetTestControls();
  SkUtSetModelStreamNum(2);
  SkUtSetStreamTaskNum(0, 0);
  SkUtSetStreamTaskNum(1, 0);

  EXPECT_TRUE(graph->InitFromModelRI());
  EXPECT_TRUE(graph->GetStreams().empty());
  EXPECT_TRUE(graph->GetNodeSizeInStream().empty());
  EXPECT_TRUE(graph->GetHeadNodes().empty());
  SkUtResetTestControls();
}

TEST_F(SuperKernelGraphTest, InitFromModelRI_FiltersEmptyStreams) {
  SkUtResetTestControls();
  SkUtSetModelStreamNum(4);
  SkUtSetStreamTaskNum(0, 0);
  SkUtSetStreamTaskNum(1, 2);
  SkUtSetStreamTaskNum(2, 0);
  SkUtSetStreamTaskNum(3, 3);

  EXPECT_TRUE(graph->InitFromModelRI());
  ASSERT_EQ(graph->GetStreams().size(), 2U);
  EXPECT_EQ(graph->GetStreams()[0], reinterpret_cast<aclrtStream>(2));
  EXPECT_EQ(graph->GetStreams()[1], reinterpret_cast<aclrtStream>(4));
  EXPECT_EQ(graph->GetNodeSizeInStream(), (std::vector<uint64_t>{2, 3}));
  EXPECT_EQ(graph->GetHeadNodes().size(), 2U);
  EXPECT_EQ(graph->GetSortedNodeIds().size(), 5U);
  SkUtResetTestControls();
}

TEST_F(SuperKernelGraphTest, InitFromModelRI_TaskNumExceedsOldBufferSizeReturnsTrue) {
  constexpr uint32_t taskNum = 1025;
  SkUtResetTestControls();
  SkUtSetModelStreamNum(1);
  SkUtSetStreamTaskNum(0, taskNum);

  EXPECT_TRUE(graph->InitFromModelRI());
  EXPECT_EQ(graph->GetNodeSizeInStream(), (std::vector<uint64_t>{taskNum}));
  EXPECT_EQ(graph->GetSortedNodeIds().size(), taskNum);
  SkUtResetTestControls();
}

// ==================== GetHeadNodes Tests ====================

TEST_F(SuperKernelGraphTest, GetHeadNodes_Empty) {
  const auto &headNodes = graph->GetHeadNodes();
  EXPECT_EQ(headNodes.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetHeadNodes_WithNodes) {
  graph->headNodes = {10, 20, 30};
  const auto &headNodes = graph->GetHeadNodes();
  EXPECT_EQ(headNodes.size(), 3);
  EXPECT_EQ(headNodes[0], 10);
  EXPECT_EQ(headNodes[1], 20);
  EXPECT_EQ(headNodes[2], 30);
}

// ==================== GetNodeSizeInStream Tests ====================

TEST_F(SuperKernelGraphTest, GetNodeSizeInStream_Empty) {
  const auto &nodeSize = graph->GetNodeSizeInStream();
  EXPECT_EQ(nodeSize.size(), 0);
}

TEST_F(SuperKernelGraphTest, GetNodeSizeInStream_WithData) {
  graph->nodeSizeInStream = {5, 10, 3};
  const auto &nodeSize = graph->GetNodeSizeInStream();
  EXPECT_EQ(nodeSize.size(), 3);
  EXPECT_EQ(nodeSize[0], 5);
  EXPECT_EQ(nodeSize[1], 10);
  EXPECT_EQ(nodeSize[2], 3);
}

// ==================== EventAssociation Tests ====================

TEST_F(SuperKernelGraphTest, AddEventAssociateNotify_Success) {
  auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  node->SetNodeType(SkNodeType::NODE_NOTIFY);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  bool result = graph->AddEventAssociateNotify(100, nodePtr);
  EXPECT_TRUE(result);
  EXPECT_EQ(graph->eventToNodes[100].notifyNodeId, 10);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateNotify_Duplicate) {
  auto node1 = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node1->SetNodeId(10);
  node1->SetNodeType(SkNodeType::NODE_NOTIFY);
  graph->graphMap[10] = std::move(node1);
  graph->eventToNodes[100].notifyNodeId = 10;

  auto node2 = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node2->SetNodeId(20);
  node2->SetNodeType(SkNodeType::NODE_NOTIFY);
  SuperKernelBaseNode *node2Ptr = node2.get();
  graph->graphMap[20] = std::move(node2);

  bool result = graph->AddEventAssociateNotify(100, node2Ptr);
  EXPECT_FALSE(result);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateWait_Success) {
  auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  node->SetNodeType(SkNodeType::NODE_WAIT);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  bool result = graph->AddEventAssociateWait(100, nodePtr);
  EXPECT_TRUE(result);
  EXPECT_TRUE(graph->eventToNodes[100].waitNodeIdList.count(10) > 0);
}

TEST_F(SuperKernelGraphTest, AddEventAssociateReset_Success) {
  auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  node->SetNodeType(SkNodeType::NODE_RESET);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  bool result = graph->AddEventAssociateReset(100, nodePtr);
  EXPECT_TRUE(result);
  EXPECT_TRUE(graph->eventToNodes[100].resetNodeIdList.count(10) > 0);
}

// ==================== MemoryAssociation Tests ====================

TEST_F(SuperKernelGraphTest, AddMemoryAssociateWrite_Success) {
  auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  node->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  bool result = graph->AddMemoryAssociateWrite(100, nodePtr);
  EXPECT_TRUE(result);
  EXPECT_TRUE(graph->memoryToNodes[100].writeNodeIdList.count(10) > 0);
}

TEST_F(SuperKernelGraphTest, AddMemoryAssociateWait_Success) {
  auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  node->SetNodeType(SkNodeType::NODE_MEMORY_WAIT);
  SuperKernelBaseNode *nodePtr = node.get();
  graph->graphMap[10] = std::move(node);

  bool result = graph->AddMemoryAssociateWait(100, nodePtr);
  EXPECT_TRUE(result);
  EXPECT_TRUE(graph->memoryToNodes[100].waitNodeIdList.count(10) > 0);
}

// ==================== SuperKernelNodeFactory Tests ====================

TEST_F(SuperKernelGraphTest, CreateNode_KernelTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_EventRecordTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node =
      SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_EventWaitTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node =
      SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_ValueWriteTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node =
      SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_ValueWaitTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node =
      SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
}

TEST_F(SuperKernelGraphTest, CreateNode_DefaultTask) {
  auto task = std::make_unique<aclmdlRITask>(nullptr);
  auto node = SuperKernelNodeFactory::CreateNode(std::move(task), ACL_MODEL_RI_TASK_DEFAULT, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_NE(node, nullptr);
  EXPECT_EQ(node->GetNodeType(), SkNodeType::NODE_DEFAULT);
}

// ==================== CollectFusionFailStats Tests ====================

TEST_F(SuperKernelGraphTest, CollectFusionFailStats_EmptyGraph) {
  auto stats = graph->CollectFusionFailStats();
  EXPECT_EQ(stats.fusibleCount, 0);
  EXPECT_EQ(stats.unfusibleCount, 0);
}

TEST_F(SuperKernelGraphTest, CollectFusionFailStats_WithNodes) {
  auto node1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node1->SetNodeId(10);
  node1->SetNodeType(SkNodeType::NODE_KERNEL);
  node1->SetIsFusible(true);
  graph->graphMap[10] = std::move(node1);

  auto node2 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node2->SetNodeId(20);
  node2->SetNodeType(SkNodeType::NODE_KERNEL);
  node2->SetIsFusible(false);
  node2->SetFusionFailReason(FusionFailReason::UNSUPPORT_EVENT_TYPE);
  graph->graphMap[20] = std::move(node2);

  auto node3 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node3->SetNodeId(30);
  node3->SetNodeType(SkNodeType::NODE_KERNEL);
  node3->SetIsFusible(false);
  node3->SetFusionFailReason(BindmapFailReason::BINHDL_NULL);
  graph->graphMap[30] = std::move(node3);

  auto node4 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node4->SetNodeId(40);
  node4->SetNodeType(SkNodeType::NODE_KERNEL);
  node4->SetIsFusible(false);
  node4->SetFusionFailReason(ScopeProcessStatus::RESOURCE_INSUFFICIENT);
  graph->graphMap[40] = std::move(node4);

  auto node5 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                       INVALID_TASK_ID);
  node5->SetNodeId(50);
  node5->SetNodeType(SkNodeType::NODE_KERNEL);
  node5->SetIsFusible(false);
  node5->SetFusionFailReason(DeadlockFailReason::KERNEL_INSUFFICIENT_CORES);
  graph->graphMap[50] = std::move(node5);

  auto stats = graph->CollectFusionFailStats();
  EXPECT_EQ(stats.fusibleCount, 1);
  EXPECT_EQ(stats.unfusibleCount, 4);
  EXPECT_EQ(stats.reasonStats["OP_UNSUPPORT [BINHDL_NULL]"], 1);
  EXPECT_EQ(stats.reasonStats["SCOPE_FUSE_PART [RESOURCE_INSUFFICIENT]"], 1);
  EXPECT_EQ(stats.reasonStats["EXIST_DEADLOCK [KERNEL_INSUFFICIENT_CORES]"], 1);
  ASSERT_EQ(stats.unfusibleNodeLogEntries.size(), 4);
  bool hasBindmapDetail = false;
  bool hasScopeDetail = false;
  bool hasDeadlockDetail = false;
  for (const auto &entry : stats.unfusibleNodeLogEntries) {
    hasBindmapDetail = hasBindmapDetail || entry.find(
                                               "reasonDetail: Failed to resolve SuperKernel bind map for the operator, "
                                               "binHdl is null") != std::string::npos;
    hasScopeDetail =
        hasScopeDetail || entry.find(
                              "reasonDetail: scope fuse failed, "
                              "Insufficient stream task slots or event memory resources") != std::string::npos;
    const std::string deadlockDetail =
        "reasonDetail: deadlock exists, "
        "The wait node depends on a kernel node that requires more cores than available";
    hasDeadlockDetail = hasDeadlockDetail || entry.find(deadlockDetail) != std::string::npos;
  }
  EXPECT_TRUE(hasBindmapDetail);
  EXPECT_TRUE(hasScopeDetail);
  EXPECT_TRUE(hasDeadlockDetail);
}

// ==================== PostProcessMemoryNode Extended Tests ====================

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_EmptyMemoryMap_ReturnsTrue) {
  // No entries in memoryToNodes
  EXPECT_TRUE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_EmptyMemoryEvent_ReturnsFalse) {
  constexpr uint64_t kEventId = 0x300;
  graph->memoryToNodes[kEventId] = MemoryInfos{};

  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWriteWithResetValue_TurnsIntoReset) {
  constexpr uint64_t kEventId = 0x301;
  // memoryValue = 0 means reset value (SK_DEFAULT_RESET_VALUE = 0)
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  graph->BuildEventNodeAssociations();

  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_TRUE(writeNode->IsFusible());
  EXPECT_TRUE(graph->eventToNodes[kEventId].resetNodeIdList.count(writeNode->GetNodeId()) > 0);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWaitNullptrNode_Skipped) {
  constexpr uint64_t kEventId = 0x401;
  // Insert a waitNodeId that has no corresponding node in graphMap
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(9999);

  // Should not crash; the nullptr check skips the node
  EXPECT_TRUE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWaitNonMemoryWaitType_Skipped) {
  constexpr uint64_t kEventId = 0x402;
  // Create a kernel node and add its id to waitNodeIdList
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(10);
  graph->graphMap[10] = std::move(node);
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(10);

  // The node type is not NODE_MEMORY_WAIT, so SetIsFusible(false) is skipped
  EXPECT_TRUE(graph->PostProcessMemoryNode());
  // Kernel node's IsFusible should remain at its default
  EXPECT_EQ(graph->GetNodeById(10)->IsFusible(), false);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWriteNonMemoryWriteType_Skipped) {
  constexpr uint64_t kEventId = 0x403;
  // Create a kernel node and add its id to writeNodeIdList
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  node->SetNodeId(20);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  graph->graphMap[20] = std::move(node);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(20);

  // The node type is not NODE_MEMORY_WRITE, so it's skipped entirely
  EXPECT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(graph->GetNodeById(20)->GetNodeType(), SkNodeType::NODE_KERNEL);
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_InconsistentWaitParams_ReturnsFalse) {
  constexpr uint64_t kEventId = 0x501;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  // Two wait nodes with different memoryValue
  auto *waitNode1 =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  auto *waitNode2 =
      CreateMemoryNode(3, SkNodeType::NODE_MEMORY_WAIT, kEventId, 7, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));

  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode1->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode2->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_InconsistentWaitFlag_ReturnsFalse) {
  constexpr uint64_t kEventId = 0x502;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  // Two wait nodes with same memoryValue but different flags
  auto *waitNode1 =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  auto *waitNode2 =
      CreateMemoryNode(3, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::GEQ));

  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode1->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode2->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_ValueBreakerBypassUnpairedWait_UpgradesWaitToNodeWait) {
  constexpr uint64_t kEventId = 0x603;
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 5, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_InvalidWaitNodeInConsistencyCheck_ReturnsFalse) {
  constexpr uint64_t kEventId = 0x701;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  // Insert a non-existent nodeId and a valid wait node
  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(8888);  // non-existent nodeId
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  // The consistency check should fail because GetNodeById(8888) returns nullptr
  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_WrongNodeTypeInConsistencyCheck_ReturnsFalse) {
  constexpr uint64_t kEventId = 0x702;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  // Create a kernel node and put it in the wait list
  auto kernelNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                            INVALID_TASK_ID);
  kernelNode->SetNodeId(99);
  graph->graphMap[99] = std::move(kernelNode);

  auto *waitNode =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(99);  // kernel node in wait list
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  // The consistency check should fail because node 99 is not NODE_MEMORY_WAIT
  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_MultipleEvents_ProcessedIndependently) {
  constexpr uint64_t kEvent1 = 0x801;
  constexpr uint64_t kEvent2 = 0x802;

  // Event1: only write
  auto *writeNode1 = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEvent1, 5);
  graph->memoryToNodes[kEvent1].writeNodeIdList.insert(writeNode1->GetNodeId());

  // Event2: only wait
  auto *waitNode2 =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEvent2, 5, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEvent2].waitNodeIdList.insert(waitNode2->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  // Event1: write should become notify
  EXPECT_EQ(writeNode1->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_TRUE(writeNode1->IsFusible());

  // Event2: wait should become an unfusible event wait
  EXPECT_EQ(waitNode2->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode2->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_MultipleOnlyWriteNodes_SameEvent) {
  constexpr uint64_t kEventId = 0x901;
  auto *writeNode1 = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 5);
  auto *writeNode2 = CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WRITE, kEventId, 0);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode1->GetNodeId());
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode2->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  graph->BuildEventNodeAssociations();

  // writeNode1: memoryValue=5 != 0 → NODE_NOTIFY
  EXPECT_EQ(writeNode1->GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_TRUE(writeNode1->IsFusible());

  // writeNode2: memoryValue=0 == SK_DEFAULT_RESET_VALUE → NODE_RESET
  EXPECT_EQ(writeNode2->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_TRUE(writeNode2->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_MultipleOnlyWaitNodes_AllUnfusible) {
  constexpr uint64_t kEventId = 0x902;
  auto *waitNode1 =
      CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WAIT, kEventId, 5, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  auto *waitNode2 =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 7, static_cast<uint32_t>(SkMemoryWaitFlag::GEQ));
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode1->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode2->GetNodeId());

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  EXPECT_EQ(waitNode1->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode1->IsFusible());
  EXPECT_EQ(waitNode2->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode2->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_ConsistentWaitParams_Succeeds) {
  constexpr uint64_t kEventId = 0xB01;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  // Two wait nodes with same memoryValue and same flag
  auto *waitNode1 =
      CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  auto *waitNode2 =
      CreateMemoryNode(3, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode1->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode2->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  EXPECT_TRUE(writeNode->IsFusible());
  EXPECT_EQ(waitNode1->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(waitNode1->IsFusible());
  EXPECT_EQ(waitNode2->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_TRUE(waitNode2->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_PairedWaitSupportsAllMemoryWaitFlags) {
  struct Case {
    uint64_t eventId;
    uint64_t writeValue;
    uint64_t waitValue;
    SkMemoryWaitFlag flag;
  };
  const std::vector<Case> cases = {
      {0xC01, 5, 3, SkMemoryWaitFlag::GEQ},
      {0xC02, 0x4, 0xC, SkMemoryWaitFlag::AND},
      {0xC03, 0x1, 0x2, SkMemoryWaitFlag::NOR},
  };

  for (const auto &item : cases) {
    const uint64_t baseNodeId = item.eventId & 0xFF;
    auto *writeNode = CreateMemoryNode(baseNodeId, SkNodeType::NODE_MEMORY_WRITE, item.eventId, item.writeValue);
    auto *waitNode = CreateMemoryNode(baseNodeId + 100, SkNodeType::NODE_MEMORY_WAIT, item.eventId, item.waitValue,
                                      static_cast<uint32_t>(item.flag));
    graph->memoryToNodes[item.eventId].writeNodeIdList.insert(writeNode->GetNodeId());
    graph->memoryToNodes[item.eventId].waitNodeIdList.insert(waitNode->GetNodeId());
  }
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ASSERT_TRUE(graph->PostProcessMemoryNode());

  for (const auto &item : cases) {
    const uint64_t baseNodeId = item.eventId & 0xFF;
    EXPECT_EQ(graph->GetNodeById(baseNodeId)->GetNodeType(), SkNodeType::NODE_NOTIFY);
    EXPECT_EQ(graph->GetNodeById(baseNodeId + 100)->GetNodeType(), SkNodeType::NODE_WAIT);
  }
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_UnknownFlagTreatedAsNoNotify) {
  constexpr uint64_t kEventId = 0xC10;
  auto *writeNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kEventId, 3);
  auto *waitNode = CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WAIT, kEventId, 3, 0xFF);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(writeNode->GetNodeId());
  graph->memoryToNodes[kEventId].waitNodeIdList.insert(waitNode->GetNodeId());
  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  ut_log::LogBuffer::Instance().Clear();
  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_THAT(ut_log::LogBuffer::Instance().GetContent(), testing::HasSubstr("event 0xc10"));
  EXPECT_EQ(writeNode->GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(writeNode->IsFusible());
  EXPECT_EQ(waitNode->GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(waitNode->IsFusible());
}

TEST_F(SuperKernelGraphTest, PostProcessMemoryNode_OnlyWriteDuplicateAssociationsFail) {
  constexpr uint64_t kNotifyEvent = 0xC13;
  auto *notifyNode = CreateMemoryNode(1, SkNodeType::NODE_MEMORY_WRITE, kNotifyEvent, 7);
  graph->memoryToNodes[kNotifyEvent].writeNodeIdList.insert(notifyNode->GetNodeId());
  graph->eventToNodes[kNotifyEvent].notifyNodeId = 999;
  EXPECT_FALSE(graph->PostProcessMemoryNode());

  ResetGraph();
  constexpr uint64_t kResetEvent = 0xC14;
  auto *resetNode = CreateMemoryNode(2, SkNodeType::NODE_MEMORY_WRITE, kResetEvent, 0);
  graph->memoryToNodes[kResetEvent].writeNodeIdList.insert(resetNode->GetNodeId());
  graph->eventToNodes[kResetEvent].resetNodeIdList.insert(resetNode->GetNodeId());
  EXPECT_FALSE(graph->PostProcessMemoryNode());
}

TEST_F(SuperKernelGraphTest, Update_ValueBackedEventUsesDefaultValueAndFlag) {
  TestRITask notifyTask{};
  TestRITask waitTask{};
  TestRITask resetTask{};
  auto *notifyNode = CreateEventNodeWithTask(notifyTask, 10, SkNodeType::NODE_NOTIFY, ACL_MODEL_RI_TASK_EVENT_RECORD,
                                             reinterpret_cast<void *>(0x1000));
  auto *waitNode = CreateEventNodeWithTask(waitTask, 11, SkNodeType::NODE_WAIT, ACL_MODEL_RI_TASK_EVENT_WAIT,
                                           reinterpret_cast<void *>(0x2000));
  auto *resetNode = CreateEventNodeWithTask(resetTask, 12, SkNodeType::NODE_RESET, ACL_MODEL_RI_TASK_EVENT_RESET,
                                            reinterpret_cast<void *>(0x3000));
  std::vector<SuperKernelBaseNode *> updateNodes = {notifyNode, waitNode, resetNode};
  ASSERT_TRUE(graph->ExpandUpdateNodes(updateNodes));

  EXPECT_EQ(graph->Update(), ACL_SUCCESS);

  EXPECT_EQ(notifyTask.params.type, ACL_MODEL_RI_TASK_VALUE_WRITE);
  EXPECT_EQ(notifyTask.params.valueWriteTaskParams.devAddr, reinterpret_cast<void *>(0x1000));
  EXPECT_EQ(notifyTask.params.valueWriteTaskParams.value, SK_DEFAULT_NOTIFY_VALUE);

  EXPECT_EQ(waitTask.params.type, ACL_MODEL_RI_TASK_VALUE_WAIT);
  EXPECT_EQ(waitTask.params.valueWaitTaskParams.devAddr, reinterpret_cast<void *>(0x2000));
  EXPECT_EQ(waitTask.params.valueWaitTaskParams.value, SK_DEFAULT_WAIT_VALUE);
  EXPECT_EQ(waitTask.params.valueWaitTaskParams.flag, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));

  EXPECT_EQ(resetTask.params.type, ACL_MODEL_RI_TASK_VALUE_WRITE);
  EXPECT_EQ(resetTask.params.valueWriteTaskParams.devAddr, reinterpret_cast<void *>(0x3000));
  EXPECT_EQ(resetTask.params.valueWriteTaskParams.value, SK_DEFAULT_RESET_VALUE);
}

TEST_F(SuperKernelGraphTest, RegisterFusibleScope_ExceedMaxScopeNum_MarksUnfusible) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM + 1; ++i) {
    auto node = std::unique_ptr<SuperKernelBaseNode>(
        new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, i, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
    node->SetNodeId(i);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetIsScopeNode(true);
    static_cast<SuperKernelKernelNode *>(node.get())->isScopeBegin = true;
    node->SetIsFusible(true);
    static_cast<SuperKernelKernelNode *>(node.get())->scopeName = "scope_" + std::to_string(i);

    graph->RegisterFusibleScope(node);
    ASSERT_TRUE(graph->AddNode(std::move(node)));
  }

  auto *addedNode = graph->GetNodeById(MAX_SCOPE_NUM);
  ASSERT_NE(addedNode, nullptr);
  EXPECT_FALSE(addedNode->IsFusible());
  EXPECT_EQ(addedNode->GetFusionFailReason(), FusionFailReason::EXCEED_SCOPE_MAX);
  EXPECT_EQ(graph->graphMap.size(), MAX_SCOPE_NUM + 1);
  EXPECT_EQ(graph->scopeNameToIdx.size(), MAX_SCOPE_NUM);
}

TEST_F(SuperKernelGraphTest, RegisterFusibleScope_MoreMarkersThanMaxScopeNum_ExistingScopeNameRemainsFusible) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM + 1; ++i) {
    auto node = std::unique_ptr<SuperKernelBaseNode>(
        new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, i, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
    node->SetNodeId(i);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetIsScopeNode(true);
    static_cast<SuperKernelKernelNode *>(node.get())->isScopeBegin = true;
    node->SetIsFusible(true);
    static_cast<SuperKernelKernelNode *>(node.get())->scopeName =
        (i == MAX_SCOPE_NUM) ? "scope_0" : "scope_" + std::to_string(i);

    graph->RegisterFusibleScope(node);
    ASSERT_TRUE(graph->AddNode(std::move(node)));
  }

  auto *addedNode = graph->GetNodeById(MAX_SCOPE_NUM);
  ASSERT_NE(addedNode, nullptr);
  EXPECT_TRUE(addedNode->IsFusible());
  EXPECT_EQ(addedNode->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(graph->graphMap.size(), MAX_SCOPE_NUM + 1);
  EXPECT_EQ(graph->scopeNameToIdx.size(), MAX_SCOPE_NUM);
}

TEST_F(SuperKernelGraphTest, RegisterFusibleScope_ExceedMaxScopeNum_NonScopeNodeIgnored) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM; ++i) {
    graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
  }

  auto node = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  node->SetNodeId(1002);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->SetIsScopeNode(false);
  node->SetIsFusible(true);

  graph->RegisterFusibleScope(node);
  graph->graphMap[1002] = std::move(node);

  auto *addedNode = graph->GetNodeById(1002);
  EXPECT_NE(addedNode, nullptr);
  EXPECT_TRUE(addedNode->IsFusible());
  EXPECT_EQ(addedNode->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SuperKernelGraphTest, RegisterFusibleScope_ExceedMaxScopeNum_NonFusibleScopeIgnored) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM; ++i) {
    graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
  }

  auto node = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  node->SetNodeId(1003);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->SetIsScopeNode(true);
  static_cast<SuperKernelKernelNode *>(node.get())->isScopeBegin = true;
  node->SetIsFusible(false);
  static_cast<SuperKernelKernelNode *>(node.get())->scopeName = "scope_unfusible_new";

  graph->RegisterFusibleScope(node);
  graph->graphMap[1003] = std::move(node);

  auto *addedNode = graph->GetNodeById(1003);
  EXPECT_NE(addedNode, nullptr);
  EXPECT_FALSE(addedNode->IsFusible());
  EXPECT_NE(addedNode->GetFusionFailReason(), FusionFailReason::EXCEED_SCOPE_MAX);
  EXPECT_EQ(graph->scopeNameToIdx.size(), MAX_SCOPE_NUM);
}
TEST_F(SuperKernelGraphTest, RegisterFusibleScope_ExceedMaxScopeNum_EmptyScopeNameIgnored) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM; ++i) {
    graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
  }

  auto node = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  node->SetNodeId(1004);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->SetIsScopeNode(true);
  static_cast<SuperKernelKernelNode *>(node.get())->isScopeBegin = true;
  node->SetIsFusible(true);
  static_cast<SuperKernelKernelNode *>(node.get())->scopeName = "";

  graph->RegisterFusibleScope(node);
  graph->graphMap[1004] = std::move(node);

  auto *addedNode = graph->GetNodeById(1004);
  EXPECT_NE(addedNode, nullptr);
  EXPECT_TRUE(addedNode->IsFusible());
  EXPECT_EQ(graph->scopeNameToIdx.size(), MAX_SCOPE_NUM);
}

TEST_F(SuperKernelGraphTest, UpdateNodeScopeBitFlags_ExceedScopeMaxReasonPropagatesToInnerNode) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM; ++i) {
    graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
  }

  auto scopeBegin = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  scopeBegin->SetNodeId(1000);
  scopeBegin->SetNodeType(SkNodeType::NODE_KERNEL);
  scopeBegin->SetIsScopeNode(true);
  static_cast<SuperKernelKernelNode *>(scopeBegin.get())->isScopeBegin = true;
  scopeBegin->SetIsFusible(true);
  static_cast<SuperKernelKernelNode *>(scopeBegin.get())->scopeName = "scope_exceed_limit";
  graph->RegisterFusibleScope(scopeBegin);
  graph->graphMap[1000] = std::move(scopeBegin);
  EXPECT_EQ(graph->GetNodeById(1000)->GetFusionFailReason(), FusionFailReason::EXCEED_SCOPE_MAX);

  auto innerNode = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  innerNode->SetNodeId(1001);
  innerNode->SetNodeType(SkNodeType::NODE_KERNEL);
  innerNode->SetIsFusible(true);
  graph->graphMap[1001] = std::move(innerNode);

  auto scopeEnd = std::unique_ptr<SuperKernelBaseNode>(
      new SuperKernelKernelNode(nullptr, ACL_MODEL_RI_TASK_KERNEL, 2, 0, INVALID_STREAM_ID, INVALID_TASK_ID));
  scopeEnd->SetNodeId(1002);
  scopeEnd->SetNodeType(SkNodeType::NODE_KERNEL);
  scopeEnd->SetIsScopeNode(true);
  static_cast<SuperKernelKernelNode *>(scopeEnd.get())->isScopeEnd = true;
  scopeEnd->SetIsFusible(false);
  static_cast<SuperKernelKernelNode *>(scopeEnd.get())->scopeName = "scope_exceed_limit";
  graph->graphMap[1002] = std::move(scopeEnd);

  graph->UpdateNodeScopeBitFlags();

  auto *updatedInnerNode = graph->GetNodeById(1001);
  ASSERT_NE(updatedInnerNode, nullptr);
  EXPECT_FALSE(updatedInnerNode->IsFusible());
  EXPECT_EQ(updatedInnerNode->GetFusionFailReason(), FusionFailReason::EXCEED_SCOPE_MAX);
  EXPECT_EQ(updatedInnerNode->GetScopeBitFlags().count(), 0);
}
