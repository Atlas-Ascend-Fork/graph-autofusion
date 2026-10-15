/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_node_scope_tagging.cpp
 * \brief Unit tests for verifying scope tagging functionality on nodes
 */

#include <gtest/gtest.h>
#include <memory>
#include <bitset>

#define private public
#define protected public
#include "super_kernel.h"
#include "sk_node.h"
#include "sk_types.h"
#include "sk_graph.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_lock_detector.h"

class TestLockDetector : public ::testing::Test {
 protected:
  void SetUp() override {
    opts = std::make_unique<SuperKernelOptionsManager>();
    graph = std::make_unique<SuperKernelGraph>();
    lockDetector = std::make_unique<LockDetector>(*graph, *opts);
    lockDetector->Reset();
    // Initialize device core numbers for LockDetector
    LockDetector::GetDeviceCores();
  }

  void ConfigureValueBreakerBypass(uint32_t value) {
    aclskOption option{};
    option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
    option.aggressiveOpts.valueBreakerBypass = value;
    opts->SetOptOptionValue(&option);
  }

  void TearDown() override {
    lockDetector->Reset();
    graph.reset();
  }

  // Helper function to create a wait node
  SuperKernelBaseNode *CreateWaitNode(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId = INVALID_TASK_ID,
                                      uint64_t nextNodeId = INVALID_TASK_ID, uint64_t notifyNodeId = INVALID_TASK_ID,
                                      uint64_t nodeIdxInStream = 0) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, nodeIdxInStream,
                                                        streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->SetPreNodeId(preNodeId);
    node->nodeInfos.syncInfos.correspondingNotifyNodeId = notifyNodeId;
    // eventId is not used in LockDetector, but needed for eventToNodes mapping
    // We'll set it based on the corresponding notify node's eventId
    node->isFusible = true;
    node->nodeType = SkNodeType::NODE_WAIT;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a notify node
  SuperKernelBaseNode *CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId = INVALID_TASK_ID,
                                        uint64_t nextNodeId = INVALID_TASK_ID, uint64_t eventId = INVALID_TASK_ID,
                                        std::vector<uint64_t> waitNodeIds = {}, uint64_t nodeIdxInStream = 0) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeIdxInStream,
                                                        streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->SetPreNodeId(preNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->isFusible = true;
    node->nodeType = SkNodeType::NODE_NOTIFY;
    node->nodeInfos.syncInfos.correspondingWaitNodeIds = waitNodeIds;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateResetNode(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId = INVALID_TASK_ID,
                                       uint64_t nextNodeId = INVALID_TASK_ID, uint64_t eventId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->SetPreNodeId(preNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->nodeInfos.syncInfos.memoryValue = SK_DEFAULT_RESET_VALUE;
    node->isFusible = true;
    node->nodeType = SkNodeType::NODE_RESET;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a kernel node with custom core counts
  SuperKernelBaseNode *CreateKernelNodeWithCores(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId,
                                                 uint64_t nextNodeId, uint32_t numBlocks, SkKernelType kernelType,
                                                 uint64_t nodeIdxInStream = 0) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeIdxInStream, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->SetPreNodeId(preNodeId);
    node->nodeType = SkNodeType::NODE_KERNEL;
    // Mark as fusible for testing
    node->isFusible = true;
    // Set custom kernel parameters
    node->nodeInfos.kernelInfos.numBlocks = numBlocks;
    node->nodeInfos.kernelInfos.kernelType = kernelType;
    // Calculate vecNum and cubeNum based on kernelType and numBlocks
    if (kernelType == SkKernelType::AIC_ONLY || kernelType == SkKernelType::MIX_AIC_1_0) {
      node->nodeInfos.kernelInfos.cubeNum = numBlocks;
      node->nodeInfos.kernelInfos.vecNum = 0;
    } else if (kernelType == SkKernelType::AIV_ONLY || kernelType == SkKernelType::MIX_AIV_1_0) {
      node->nodeInfos.kernelInfos.cubeNum = 0;
      node->nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kernelType == SkKernelType::MIX_AIC_1_1) {
      node->nodeInfos.kernelInfos.cubeNum = numBlocks;
      node->nodeInfos.kernelInfos.vecNum = numBlocks;
    } else if (kernelType == SkKernelType::MIX_AIC_1_2) {
      node->nodeInfos.kernelInfos.cubeNum = numBlocks;
      node->nodeInfos.kernelInfos.vecNum = numBlocks << 1;
    }
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to setup streams in graph
  void SetupStreams(const std::vector<std::vector<uint64_t>> &streamNodes) {
    graph->streams.clear();
    graph->headNodes.clear();

    for (const auto &nodes : streamNodes) {
      graph->streams.emplace_back();
      if (!nodes.empty()) {
        graph->headNodes.push_back(nodes[0]);
      } else {
        graph->headNodes.push_back(INVALID_TASK_ID);
      }
    }
  }

  // Helper function to setup event mapping
  void SetupEvent(uint64_t eventId, uint64_t notifyNodeId, const std::vector<uint64_t> &waitNodeIds) {
    EventInfos eventInfo;
    eventInfo.notifyNodeId = notifyNodeId;
    for (auto waitNodeId : waitNodeIds) {
      eventInfo.waitNodeIdList.insert(waitNodeId);
    }
    graph->eventToNodes[eventId] = eventInfo;
    std::cout << "[SetupEvent] event=1 , notifyNodeId=" << graph->eventToNodes[1].notifyNodeId << ", waitNodeIds= [";
    for (size_t i = 0; i < graph->eventToNodes[1].waitNodeIdList.size(); ++i) {
      if (i > 0) std::cout << ",";
      std::cout << waitNodeIds[i];
    }
    std::cout << "]" << std::endl;
    std::cout.flush();  // 立即刷新缓冲区
  }
  std::unique_ptr<SuperKernelGraph> graph;
  std::unique_ptr<SuperKernelOptionsManager> opts;
  std::unique_ptr<LockDetector> lockDetector;
};

// Test 1: one stream, kernel node (after wait node) exceeds max sk cube/vec num
TEST_F(TestLockDetector, SingleStreamKernelFirst) {
  // ======================= graph =======================
  /*                               sk
  stream0: k0(8c) -> n1(eventid=1) -> k2(4c) -> n3(notify) -> k4(8v) -> k5(6c,6v) -> w6(eventid=1) -> k7(4c,8v) ->
  k8(8c,8v) ↑                                                                   ↓
                          └───────────────────────────────────────────────────────────────────┘
  */
  // 依次传kernel(4c), notify, kernel(8v), kernel(6c,6v), wait, kernel(4c,8v), kernel(8c,8v)
  // 验证LockDetector值的状态、返回的融合结果、图上节点初始visited状态、reset后的状态
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 1);
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 4,
                                       SkKernelType::AIC_ONLY);  // nodeid, streamid, nextnodeid, numblocks, kerneltype
  auto *n3 = CreateNotifyNode(3, 0, 2, 4, 2);                    // nodeid, streamid, eventid, nextnodeid
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 1);  // nodeid, streamid, eventid, nextnodeid
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 8, SkKernelType::MIX_AIC_1_1);
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}});
  SetupEvent(1, 1, {6});  // eventid, notifynodeid, waitnodeidlist
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0);
  // sk - node 2
  EXPECT_TRUE(lockDetector->IsFusible(*n3));
  EXPECT_TRUE(n3->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0);
  // sk - node 3
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(k4->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 8);
  // sk - node 4
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 6});
  EXPECT_TRUE(lockDetector->IsFusible(*k5));
  EXPECT_TRUE(k5->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 6);
  EXPECT_EQ(lockDetector->superKernelVecNum, 12);
  // sk - node 5
  EXPECT_TRUE(lockDetector->IsFusible(*w6));
  EXPECT_TRUE(k5->isVisited);
  // sk - node 6
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 6});
  EXPECT_TRUE(lockDetector->IsFusible(*k7));
  EXPECT_TRUE(k7->isVisited);
  // sk - node 7
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 8});
  EXPECT_TRUE(lockDetector->IsFusible(*k8));
  EXPECT_TRUE(k8->isVisited);
  EXPECT_EQ(lockDetector->skStreamIds, std::unordered_set<uint32_t>{0});
  lockDetector->Reset();
  EXPECT_FALSE(k2->isVisited);
  EXPECT_FALSE(n3->isVisited);
  EXPECT_FALSE(k4->isVisited);
  EXPECT_FALSE(k5->isVisited);
  EXPECT_FALSE(w6->isVisited);
  EXPECT_FALSE(k7->isVisited);
  EXPECT_FALSE(k8->isVisited);
}

// Test 2: the cross-stream notify path has enough remaining cores
TEST_F(TestLockDetector, CrossStreamNotifyHasEnoughCores) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=1) -> k2(4c) -> n3(notify) -> k4(8v) -> k5(6c,6v) -> w6(eventid=1) -> k7(4c,8v) ->
  k8(8c,8v) ↑ ┌───────────────────────────────────────────────────────────────┘ ↑ stream1: k9(4c,8v) ->
  notify(eventid=1) -> k11(4c,4v)
  */
  // The candidate SK uses 6 cube cores and 12 vector cores at w6; k9 can run outside the SK and release w6.
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 4, SkKernelType::AIC_ONLY);
  auto *n3 = CreateNotifyNode(3, 0, 2, 4, 11, {});
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 10);
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 8, SkKernelType::MIX_AIC_1_1);
  auto *k9 = CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 10, 4, SkKernelType::MIX_AIC_1_2);
  auto *n10 = CreateNotifyNode(10, 1, 9, 11, 1, {6});
  auto *k11 = CreateKernelNodeWithCores(11, 1, 10, INVALID_TASK_ID, 4, SkKernelType::MIX_AIC_1_1);
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11}});
  SetupEvent(1, 10, {6});
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0);
  // sk - node 2
  EXPECT_TRUE(lockDetector->IsFusible(*n3));
  EXPECT_TRUE(n3->isVisited);
  // sk - node 3
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(k4->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 8);
  // sk - node 4
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 6});
  EXPECT_TRUE(lockDetector->IsFusible(*k5));
  EXPECT_TRUE(k5->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 6);
  EXPECT_EQ(lockDetector->superKernelVecNum, 12);
  // // sk - node 5
  EXPECT_TRUE(lockDetector->IsFusible(*w6));
  EXPECT_TRUE(w6->isVisited);
  EXPECT_TRUE(k9->isVisited);
  EXPECT_TRUE(n10->isVisited);
  // sk - node 6
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 6});
  EXPECT_TRUE(lockDetector->IsFusible(*k7));
  EXPECT_TRUE(k7->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 6);
  EXPECT_EQ(lockDetector->superKernelVecNum, 12);
  // sk - node 7
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 8});
  EXPECT_TRUE(lockDetector->IsFusible(*k8));
  EXPECT_EQ(lockDetector->superKernelCubeNum, 8);
  EXPECT_EQ(lockDetector->superKernelVecNum, 16);

  EXPECT_EQ(lockDetector->skStreamIds, std::unordered_set<uint32_t>{0});
  lockDetector->Reset();
  EXPECT_FALSE(k0->isVisited);
  EXPECT_FALSE(n1->isVisited);
  EXPECT_FALSE(k2->isVisited);
  EXPECT_FALSE(n3->isVisited);
  EXPECT_FALSE(k4->isVisited);
  EXPECT_FALSE(k5->isVisited);
  EXPECT_FALSE(w6->isVisited);
  EXPECT_FALSE(k7->isVisited);
  EXPECT_FALSE(k8->isVisited);
  EXPECT_FALSE(k9->isVisited);
  EXPECT_FALSE(n10->isVisited);
  EXPECT_FALSE(k11->isVisited);
}

// Test 3: the cross-stream notify path does not have enough remaining cores
TEST_F(TestLockDetector, CrossStreamNotifyHasInsufficientCores) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=1) -> k2(4c) -> n3(notify) -> k4(8v) -> k5(6c,6v) -> w6(eventid=1) -> k7(4c,8v) ->
  k8(8c,8v) ↑ ┌───────────────────────────────────────────────────────────────┘ ↑ stream1: k9(24c,24v) ->
  notify(eventid=1) -> k11(4c,4v)
  */
  // The candidate SK uses 6 cube cores and 12 vector cores at w6; k9 cannot run outside the SK to release w6.
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 4, SkKernelType::AIC_ONLY);
  auto *n3 = CreateNotifyNode(3, 0, 2, 4, 11, {});
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 10);
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 8, SkKernelType::MIX_AIC_1_1);
  auto *k9 = CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 10, 24, SkKernelType::MIX_AIC_1_2);
  auto *n10 = CreateNotifyNode(10, 1, 9, 11, 1, {6});
  auto *k11 = CreateKernelNodeWithCores(11, 1, 10, INVALID_TASK_ID, 4, SkKernelType::MIX_AIC_1_1);
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11}});
  SetupEvent(1, 10, {6});
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0);
  // sk - node 2
  EXPECT_TRUE(lockDetector->IsFusible(*n3));
  EXPECT_TRUE(n3->isVisited);
  // sk - node 3
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(k4->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 4);
  EXPECT_EQ(lockDetector->superKernelVecNum, 8);
  // sk - node 4
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 6});
  EXPECT_TRUE(lockDetector->IsFusible(*k5));
  EXPECT_TRUE(k5->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 6);
  EXPECT_EQ(lockDetector->superKernelVecNum, 12);
  // // sk - node 5
  EXPECT_FALSE(lockDetector->IsFusible(*w6));
  EXPECT_FALSE(n10->isVisited);

  EXPECT_EQ(lockDetector->skStreamIds, std::unordered_set<uint32_t>{0});
  lockDetector->Reset();
  EXPECT_FALSE(k0->isVisited);
  EXPECT_FALSE(n1->isVisited);
  EXPECT_FALSE(k2->isVisited);
  EXPECT_FALSE(n3->isVisited);
  EXPECT_FALSE(k4->isVisited);
  EXPECT_FALSE(k5->isVisited);
  EXPECT_FALSE(w6->isVisited);
  EXPECT_FALSE(k7->isVisited);
  EXPECT_FALSE(k8->isVisited);
  EXPECT_FALSE(k9->isVisited);
  EXPECT_FALSE(n10->isVisited);
  EXPECT_FALSE(k11->isVisited);
}

// Test 4: one stream multi wait
TEST_F(TestLockDetector, SingleStreamMultiWait) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=3) -> k2(4c) -> w3(eid=1) -> k4(8v) -> k5(6c,6v) -> w6(eid=2) -> k7(4c,8v) ->
  k8(8c,8v) ↑                                      ↑ ┌─────────────────────────┘      ┌───────────────────────────────┘
                              ↑                                ↑
  stream1: k9(4c,8v) -> notify(eventid=1) -> k11(4c,4v) -> notify(eid=2)
  */
  // 依次传wait、kernel(4c,8v)、wait、kernel(4c,4v)
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 10, SkKernelType::MIX_AIC_1_2);
  auto *w3 = CreateWaitNode(3, 0, 2, 4, 10);
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 12);
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 12, SkKernelType::MIX_AIC_1_1);
  auto *k9 = CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 10, 4, SkKernelType::MIX_AIC_1_2);
  auto *n10 = CreateNotifyNode(10, 1, 9, 11, 1, {3});
  auto *k11 = CreateKernelNodeWithCores(11, 1, 10, INVALID_TASK_ID, 4, SkKernelType::MIX_AIC_1_1);
  auto *n12 = CreateNotifyNode(12, 1, 11, INVALID_TASK_ID, 2, {6});
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11}});
  SetupEvent(1, 10, {6});
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // sk - node 2
  EXPECT_TRUE(lockDetector->IsFusible(*w3));
  EXPECT_TRUE(w3->isVisited);
  EXPECT_TRUE(k9->isVisited);
  // sk - node 3
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(k4->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // sk - node 4
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k5));
  EXPECT_TRUE(k5->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // // sk - node 5
  EXPECT_TRUE(lockDetector->IsFusible(*w6));
  EXPECT_TRUE(w6->isVisited);
  EXPECT_TRUE(k9->isVisited);
  EXPECT_TRUE(n10->isVisited);
  // sk - node 6
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k7));
  EXPECT_TRUE(k7->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // sk - node 7
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 12});
  EXPECT_TRUE(lockDetector->IsFusible(*k8));
  EXPECT_EQ(lockDetector->superKernelCubeNum, 12);
  EXPECT_EQ(lockDetector->superKernelVecNum, 24);

  EXPECT_EQ(lockDetector->skStreamIds, std::unordered_set<uint32_t>{0});
  lockDetector->Reset();
  EXPECT_FALSE(k0->isVisited);
  EXPECT_FALSE(n1->isVisited);
  EXPECT_FALSE(k2->isVisited);
  EXPECT_FALSE(w3->isVisited);
  EXPECT_FALSE(k4->isVisited);
  EXPECT_FALSE(k5->isVisited);
  EXPECT_FALSE(w6->isVisited);
  EXPECT_FALSE(k7->isVisited);
  EXPECT_FALSE(k8->isVisited);
  EXPECT_FALSE(k9->isVisited);
  EXPECT_FALSE(n10->isVisited);
  EXPECT_FALSE(k11->isVisited);
  EXPECT_FALSE(n12->isVisited);
}

TEST_F(TestLockDetector, UnchangedScopeCoreInfoSkipsResourceRecheck) {
  auto *firstKernel = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 4, SkKernelType::AIC_ONLY);
  auto *secondKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, 4, SkKernelType::AIC_ONLY);
  SetupStreams({{0, 1}});
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_TRUE(lockDetector->scopeCoreInfoChanged_);

  EXPECT_TRUE(lockDetector->IsFusible(*firstKernel));
  EXPECT_TRUE(firstKernel->isVisited);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_FALSE(lockDetector->scopeCoreInfoChanged_);
  EXPECT_TRUE(lockDetector->IsFusible(*secondKernel));
  EXPECT_TRUE(secondKernel->isVisited);
}

// Test 6: one stream, notify node of wait node not in graph
TEST_F(TestLockDetector, SingleStreamNotifyOutsideSK) {
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 20, SkKernelType::AIC_ONLY);
  auto *w1 = CreateWaitNode(1, 0, 0, INVALID_TASK_ID, 10);
  SetupStreams({{0}});
  SetupEvent(1, 10, {1});
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});
  EXPECT_TRUE(lockDetector->IsFusible(*k0));
  EXPECT_FALSE(lockDetector->IsFusible(*w1));
  lockDetector->Reset();
}

TEST_F(TestLockDetector, SingleStreamNotifyHasCore) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=1) -> k2(4c) -> n3(notify) -> W4
  stream1: n5
  */
  // kernel(4c)、wait、kernel(4c,4v)
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 4, SkKernelType::AIC_ONLY);
  auto *n3 = CreateNotifyNode(3, 0, 2, 4, 11, {});
  auto *w4 = CreateWaitNode(4, 0, 3, INVALID_TASK_ID, 5);
  auto *n5 = CreateNotifyNode(5, 1, INVALID_TASK_ID, INVALID_TASK_ID, 1, {4});
  SetupStreams({{0, 1, 2, 3, 4}, {5}});

  SetupEvent(1, 5, {4});

  n5->SetNotifyExpandVecNum(40);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(lockDetector->IsFusible(*n3));
  EXPECT_FALSE(lockDetector->IsFusible(*w4));
  lockDetector->Reset();
}

TEST_F(TestLockDetector, ResetNodeNoCoreResourceCanFuse) {
  auto *reset = CreateResetNode(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, 1);
  SetupStreams({{1}});

  EXPECT_TRUE(lockDetector->IsFusible(*reset));
  EXPECT_TRUE(reset->isVisited);
  EXPECT_EQ(lockDetector->GetDeadlockReason(), DeadlockFailReason::NOT_FIND_DEADLOCK);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 0);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0);
  EXPECT_EQ(lockDetector->skStreamIds, std::unordered_set<uint32_t>{0});
}

TEST_F(TestLockDetector, ScopeMarkerNoCoreResourceCanFuse) {
  auto *scopeMarker = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY);
  scopeMarker->SetIsScopeNode(true);
  SetupStreams({{1}});

  EXPECT_TRUE(lockDetector->IsFusible(*scopeMarker));
  EXPECT_TRUE(scopeMarker->isVisited);
  EXPECT_EQ(lockDetector->kernelNodeNum, 0U);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 0U);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0U);
}

TEST_F(TestLockDetector, ValueBackedPairedWaitKeepsDeadlockDetectionWithValueBreakerPairedFlag) {
  auto *notify = CreateNotifyNode(10, 1, INVALID_TASK_ID, INVALID_TASK_ID, 1, {1});
  auto *wait = CreateWaitNode(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, 10);
  wait->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);

  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT);

  EXPECT_FALSE(lockDetector->IsFusible(*wait));
  EXPECT_FALSE(wait->isVisited);
  EXPECT_EQ(lockDetector->GetDeadlockReason(), DeadlockFailReason::FIRST_WAIT);
  (void)notify;
}

TEST_F(TestLockDetector, ValueBackedUnpairedWaitBypassesDeadlockDetectionWithValueBreakerBypass) {
  auto *wait = CreateWaitNode(1, 0, INVALID_TASK_ID, INVALID_TASK_ID, INVALID_TASK_ID);
  wait->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);

  ConfigureValueBreakerBypass(ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT);

  EXPECT_TRUE(lockDetector->IsFusible(*wait));
  EXPECT_TRUE(wait->isVisited);
  EXPECT_EQ(lockDetector->GetDeadlockReason(), DeadlockFailReason::NOT_FIND_DEADLOCK);
}

TEST_F(TestLockDetector, notifyInOtherSKWithSameStream) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=3) -> k2(4c) -> w3(eid=1) -> k4(8v) -> k5(6c,6v) -> w6(eid=2) -> k7(4c,8v) ->
  k8(8c,8v) ↑                                      ↑ ┌─────────────────────────┘      ┌───────────────────────────────┘
                              ↑                                ↑
  stream1: k9(4c,8v) -> notify(eventid=1) -> k11(4c,4v) -> notify(eid=2)
  */
  // 依次传wait、kernel(4c,8v)、wait、kernel(4c,4v)
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 10, SkKernelType::MIX_AIC_1_2);
  auto *w3 = CreateWaitNode(3, 0, 2, 4, 10);
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 12);
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 12, SkKernelType::MIX_AIC_1_1);
  auto *k9 = CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 10, 20, SkKernelType::MIX_AIC_1_2);
  auto *n10 = CreateNotifyNode(10, 1, 9, 11, 1, {3});
  n10->SetScopeStreamIds({0, 1});
  k9->SetScopeStreamIds({0, 1});
  n10->SetNotifyExpandVecNum(20);
  n10->SetNotifyExpandCubeNum(40);
  auto *k11 = CreateKernelNodeWithCores(11, 1, 10, INVALID_TASK_ID, 4, SkKernelType::MIX_AIC_1_1);
  auto *n12 = CreateNotifyNode(12, 1, 11, INVALID_TASK_ID, 2, {6});
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11}});
  SetupEvent(1, 10, {6});
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // sk - node 2
  EXPECT_TRUE(lockDetector->IsFusible(*w3));
  EXPECT_TRUE(w3->isVisited);
  EXPECT_FALSE(k9->isVisited);
  // sk - node 3
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(k4->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
}

TEST_F(TestLockDetector, notifyInOtherSKWithoutSameStream) {
  // ======================= graph =======================
  /*
  stream0: k0(8c) -> notify(eventid=3) -> k2(4c) -> w3(eid=1) -> k4(8v) -> k5(6c,6v) -> w6(eid=2) -> k7(4c,8v) ->
  k8(8c,8v) ↑                                      ↑ ┌─────────────────────────┘      ┌───────────────────────────────┘
                              ↑                                ↑
  stream1: k9(4c,8v) -> notify(eventid=1) -> k11(4c,4v) -> notify(eid=2)
  */
  // 依次传wait、kernel(4c,8v)、wait、kernel(4c,4v)
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 8, SkKernelType::AIC_ONLY);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 10, {});  // nodeid, streamid, next, eventid
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 10, SkKernelType::MIX_AIC_1_2);
  auto *w3 = CreateWaitNode(3, 0, 2, 4, 10);
  auto *k4 = CreateKernelNodeWithCores(4, 0, 3, 5, 8, SkKernelType::AIV_ONLY);
  auto *k5 = CreateKernelNodeWithCores(5, 0, 4, 6, 6, SkKernelType::MIX_AIC_1_1);
  auto *w6 = CreateWaitNode(6, 0, 5, 7, 12);
  auto *k7 = CreateKernelNodeWithCores(7, 0, 6, 8, 4, SkKernelType::MIX_AIC_1_2);
  auto *k8 = CreateKernelNodeWithCores(8, 0, 7, INVALID_TASK_ID, 12, SkKernelType::MIX_AIC_1_1);
  auto *k9 = CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 10, 20, SkKernelType::MIX_AIC_1_2);
  auto *n10 = CreateNotifyNode(10, 1, 9, 11, 1, {3});
  n10->SetScopeStreamIds({1});
  k9->SetScopeStreamIds({1});
  n10->SetNotifyExpandVecNum(20);
  n10->SetNotifyExpandCubeNum(40);
  auto *k11 = CreateKernelNodeWithCores(11, 1, 10, INVALID_TASK_ID, 4, SkKernelType::MIX_AIC_1_1);
  auto *n12 = CreateNotifyNode(12, 1, 11, INVALID_TASK_ID, 2, {6});
  SetupStreams({{0, 1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11}});
  SetupEvent(1, 10, {6});
  // sk - node 1
  lockDetector->SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k2));
  EXPECT_TRUE(k2->isVisited);
  EXPECT_EQ(lockDetector->superKernelCubeNum, 10);
  EXPECT_EQ(lockDetector->superKernelVecNum, 20);
  // sk - node 2
  EXPECT_FALSE(lockDetector->IsFusible(*w3));
  EXPECT_FALSE(w3->isVisited);
  EXPECT_FALSE(k9->isVisited);
}

TEST_F(TestLockDetector, ThreeStreamsCrossScopeNotifyUsesCurrentScopeId) {
  // stream 0: k1 -> k2 -> k3 -> notify1
  // stream 1: k4 -> k5 -> k6 -> wait1 -> k7 (already in SK)
  // stream 2: k8 -> k9 -> k10 -> wait2 -> k11 (current scope)
  // notify1 signals both wait1 and wait2.
  auto *k1 = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 2, SkKernelType::AIC_ONLY);
  auto *k2 = CreateKernelNodeWithCores(2, 0, 1, 3, 2, SkKernelType::AIC_ONLY);
  auto *k3 = CreateKernelNodeWithCores(3, 0, 2, 4, 2, SkKernelType::AIC_ONLY);
  auto *notify1 = CreateNotifyNode(4, 0, 3, INVALID_TASK_ID, 1, {8, 13});

  auto *k4 = CreateKernelNodeWithCores(5, 1, INVALID_TASK_ID, 6, 2, SkKernelType::AIC_ONLY);
  auto *k5 = CreateKernelNodeWithCores(6, 1, 5, 7, 2, SkKernelType::AIC_ONLY);
  auto *k6 = CreateKernelNodeWithCores(7, 1, 6, 8, 2, SkKernelType::AIC_ONLY);
  auto *wait1 = CreateWaitNode(8, 1, INVALID_TASK_ID, 9, 4);
  auto *k7 = CreateKernelNodeWithCores(9, 1, 8, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);

  auto *k8 = CreateKernelNodeWithCores(10, 2, INVALID_TASK_ID, 11, 2, SkKernelType::AIC_ONLY);
  auto *k9 = CreateKernelNodeWithCores(11, 2, 10, 12, 2, SkKernelType::AIC_ONLY);
  auto *k10 = CreateKernelNodeWithCores(12, 2, 11, 13, 2, SkKernelType::AIC_ONLY);
  auto *wait2 = CreateWaitNode(13, 2, INVALID_TASK_ID, 14, 4);
  auto *k11 = CreateKernelNodeWithCores(14, 2, 13, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);

  SetupStreams({{1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}});
  SetupEvent(1, 4, {8, 13});

  constexpr uint16_t stream2ScopeId = 101;
  constexpr uint16_t stream3ScopeId = 102;
  for (auto *node : {k4, k5, k6, wait1, k7}) {
    node->SetScopeId(stream2ScopeId);
  }
  for (auto *node : {k8, k9, k10, wait2, k11}) {
    node->SetScopeId(stream3ScopeId);
  }

  // Stream 2 is already fused into the SK.
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 10});
  EXPECT_TRUE(lockDetector->IsFusible(*k4));
  EXPECT_TRUE(lockDetector->IsFusible(*k5));
  EXPECT_TRUE(lockDetector->IsFusible(*k6));
  EXPECT_TRUE(lockDetector->IsFusible(*wait1));
  EXPECT_TRUE(lockDetector->IsFusible(*k7));

  // The stream 3 scope must use its current scope core requirement while the
  // notify on stream 1 still has waits in both scopes.
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 2});
  EXPECT_EQ(k8->GetScopeId(), stream3ScopeId);
  EXPECT_EQ(k11->GetScopeId(), stream3ScopeId);
  EXPECT_TRUE(lockDetector->IsFusible(*k8));
  EXPECT_TRUE(lockDetector->IsFusible(*k9));
  EXPECT_TRUE(lockDetector->IsFusible(*k10));
  EXPECT_TRUE(lockDetector->IsFusible(*wait2));
  EXPECT_TRUE(lockDetector->IsFusible(*k11));
  EXPECT_EQ(lockDetector->superKernelCubeNum, 2U);
  EXPECT_EQ(lockDetector->superKernelVecNum, 0U);
  (void)notify1;
}

TEST_F(TestLockDetector, NotifyPredecessorStreamExcludesCandidateScope) {
  auto *currentKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 4, SkKernelType::AIC_ONLY, 0);
  auto *currentWait = CreateWaitNode(2, 0, 1, 3, 10, 1);
  auto *nextKernel = CreateKernelNodeWithCores(3, 0, 2, INVALID_TASK_ID, 5, SkKernelType::AIC_ONLY, 2);
  CreateKernelNodeWithCores(9, 1, INVALID_TASK_ID, 20, 4, SkKernelType::AIC_ONLY, 0);
  CreateNotifyNode(20, 1, 9, 11, 2, {21}, 1);
  auto *candidateWait = CreateWaitNode(11, 1, 20, 12, 10, 2);
  auto *candidateKernel = CreateKernelNodeWithCores(12, 1, 11, INVALID_TASK_ID, 4, SkKernelType::AIC_ONLY, 3);
  CreateWaitNode(21, 2, INVALID_TASK_ID, 10, 20, 0);
  CreateNotifyNode(10, 2, 21, INVALID_TASK_ID, 1, {2, 11}, 1);
  SetupStreams({{1, 2, 3}, {9, 20, 11, 12}, {21, 10}});

  constexpr uint16_t currentScopeId = 1;
  constexpr uint16_t candidateScopeId = 2;
  for (auto *node : {currentKernel, currentWait, nextKernel}) {
    node->SetScopeId(currentScopeId);
  }
  candidateWait->SetScopeId(candidateScopeId);
  candidateKernel->SetScopeId(candidateScopeId);
  LockDetector::ScopeRuntimeInfo candidateScope;
  candidateScope.coreInfo = {SkKernelType::AIC_ONLY, 4};
  candidateScope.streamIds.insert(1);
  candidateScope.headNodeIds.emplace(1, 12);
  candidateScope.tailNodeIds.emplace(1, 12);
  lockDetector->scopeInfos_.emplace(candidateScopeId, candidateScope);
  lockDetector->SetCurrentScopeId(currentScopeId);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});

  const auto savedDeviceCubeNum = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  EXPECT_TRUE(lockDetector->IsFusible(*currentKernel));
  EXPECT_TRUE(lockDetector->IsFusible(*currentWait));
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 5});
  EXPECT_TRUE(lockDetector->IsFusible(*nextKernel));
  LockDetector::deviceRealCubeNum = savedDeviceCubeNum;
}

TEST_F(TestLockDetector, SerializedCandidateFoundByEarlierWaitBeforeCurrentScope) {
  auto *matchingWait = CreateWaitNode(1, 0, INVALID_TASK_ID, 2, 20, 0);
  auto *unrelatedWait = CreateWaitNode(2, 0, 1, 3, 30, 1);
  auto *currentKernel = CreateKernelNodeWithCores(3, 0, 2, 4, 1, SkKernelType::AIC_ONLY, 2);
  auto *currentWait = CreateWaitNode(4, 0, 3, INVALID_TASK_ID, 40, 3);

  auto *candidateHead = CreateKernelNodeWithCores(10, 1, INVALID_TASK_ID, 11, 1, SkKernelType::AIC_ONLY, 0);
  auto *candidateTail = CreateKernelNodeWithCores(11, 1, 10, 20, 1, SkKernelType::AIC_ONLY, 1);
  auto *matchingNotify = CreateNotifyNode(20, 1, 11, INVALID_TASK_ID, 1, {1}, 2);
  auto *unrelatedNotify = CreateNotifyNode(30, 2, INVALID_TASK_ID, INVALID_TASK_ID, 2, {2}, 0);
  SetupStreams({{1, 2, 3, 4}, {10, 11, 20}, {30}});

  lockDetector->skRangeInStream[0] = {2, 2};
  lockDetector->parallelScopeHeadNodeIds_[0] = 3;
  lockDetector->parallelScopeTailNodeIds_[0] = 3;
  LockDetector::ScopeRuntimeInfo candidate;
  candidate.headNodeIds[1] = 10;
  candidate.tailNodeIds[1] = 11;

  EXPECT_TRUE(lockDetector->IsSerializedWithCandidateScope(*currentWait, 1, candidate));
  (void)matchingWait;
  (void)unrelatedWait;
  (void)currentKernel;
  (void)candidateHead;
  (void)candidateTail;
  (void)matchingNotify;
  (void)unrelatedNotify;
}

TEST_F(TestLockDetector, SerializedCandidateFoundByLaterNotifyAfterCurrentWait) {
  auto *currentKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 1, SkKernelType::AIC_ONLY, 0);
  auto *currentWait = CreateWaitNode(2, 0, 1, 3, 40, 1);
  auto *unrelatedNotify = CreateNotifyNode(3, 0, 2, 4, 1, {}, 2);
  auto *matchingNotify = CreateNotifyNode(4, 0, 3, INVALID_TASK_ID, 2, {10}, 3);

  auto *candidateWait = CreateWaitNode(10, 1, INVALID_TASK_ID, 11, 4, 0);
  auto *candidateHead = CreateKernelNodeWithCores(11, 1, 10, 12, 1, SkKernelType::AIC_ONLY, 1);
  auto *candidateTail = CreateKernelNodeWithCores(12, 1, 11, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY, 2);
  SetupStreams({{1, 2, 3, 4}, {10, 11, 12}});

  lockDetector->skRangeInStream[0] = {0, 0};
  lockDetector->parallelScopeHeadNodeIds_[0] = 1;
  lockDetector->parallelScopeTailNodeIds_[0] = 1;
  LockDetector::ScopeRuntimeInfo candidate;
  candidate.headNodeIds[1] = 11;
  candidate.tailNodeIds[1] = 12;

  EXPECT_TRUE(lockDetector->IsSerializedWithCandidateScope(*currentWait, 10, candidate));
  (void)currentKernel;
  (void)unrelatedNotify;
  (void)matchingNotify;
  (void)candidateWait;
  (void)candidateHead;
  (void)candidateTail;
}

TEST_F(TestLockDetector, CandidateWithoutOrderingEvidenceMayRunInParallel) {
  auto *currentKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 1, SkKernelType::AIC_ONLY, 0);
  auto *currentWait = CreateWaitNode(2, 0, 1, 3, 30, 1);
  auto *notify = CreateNotifyNode(3, 0, 2, INVALID_TASK_ID, 1, {10}, 2);

  auto *candidateHead = CreateKernelNodeWithCores(10, 1, INVALID_TASK_ID, 11, 1, SkKernelType::AIC_ONLY, 0);
  auto *candidateWait = CreateWaitNode(11, 1, 10, INVALID_TASK_ID, 3, 1);
  SetupStreams({{1, 2, 3}, {10, 11}});

  lockDetector->skRangeInStream[0] = {0, 0};
  lockDetector->parallelScopeHeadNodeIds_[0] = 1;
  lockDetector->parallelScopeTailNodeIds_[0] = 1;
  LockDetector::ScopeRuntimeInfo candidate;
  candidate.headNodeIds[1] = 10;
  candidate.tailNodeIds[1] = 11;

  EXPECT_FALSE(lockDetector->IsSerializedWithCandidateScope(*currentWait, 11, candidate));
  (void)currentKernel;
  (void)notify;
  (void)candidateHead;
  (void)candidateWait;
}

TEST_F(TestLockDetector, MergedCandidateStreamsParticipateInLaterSerializationCheck) {
  auto *currentKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 1, SkKernelType::AIC_ONLY, 0);
  auto *currentWait = CreateWaitNode(2, 0, 1, INVALID_TASK_ID, 30, 1);

  auto *mergedHead = CreateKernelNodeWithCores(10, 1, INVALID_TASK_ID, 11, 1, SkKernelType::AIC_ONLY, 0);
  auto *mergedTail = CreateKernelNodeWithCores(11, 1, 10, 12, 1, SkKernelType::AIC_ONLY, 1);
  auto *notify = CreateNotifyNode(12, 1, 11, INVALID_TASK_ID, 1, {20}, 2);

  auto *candidateWait = CreateWaitNode(20, 2, INVALID_TASK_ID, 21, 12, 0);
  auto *candidateHead = CreateKernelNodeWithCores(21, 2, 20, 22, 1, SkKernelType::AIC_ONLY, 1);
  auto *candidateTail = CreateKernelNodeWithCores(22, 2, 21, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY, 2);
  SetupStreams({{1, 2}, {10, 11, 12}, {20, 21, 22}});

  lockDetector->skRangeInStream[0] = {0, 0};
  lockDetector->parallelScopeHeadNodeIds_[0] = 1;
  lockDetector->parallelScopeTailNodeIds_[0] = 1;
  LockDetector::ScopeRuntimeInfo mergedScope;
  mergedScope.streamIds = {1};
  mergedScope.headNodeIds[1] = 10;
  mergedScope.tailNodeIds[1] = 11;
  lockDetector->MergeCandidateScopeRange(mergedScope);

  LockDetector::ScopeRuntimeInfo candidate;
  candidate.headNodeIds[2] = 21;
  candidate.tailNodeIds[2] = 22;

  EXPECT_TRUE(lockDetector->IsSerializedWithCandidateScope(*currentWait, 20, candidate));
  (void)currentKernel;
  (void)mergedHead;
  (void)mergedTail;
  (void)notify;
  (void)candidateWait;
  (void)candidateHead;
  (void)candidateTail;
}
