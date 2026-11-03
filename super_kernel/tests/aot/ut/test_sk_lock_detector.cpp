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
#include <map>
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
#include "sk_scope_split.h"

class TestLockDetector : public ::testing::Test {
 protected:
  void SetUp() override {
    opts = std::make_unique<SuperKernelOptionsManager>();
    graph = std::make_unique<SuperKernelGraph>();
    lockDetector = std::make_unique<LockDetector>(*graph, *opts);
    scopeDetector = std::make_unique<ScopeDeadlockDetector>(*graph, *opts);
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
    scopeDetector.reset();
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
                                       uint64_t nextNodeId = INVALID_TASK_ID, uint64_t eventId = INVALID_TASK_ID,
                                       uint64_t nodeIdxInStream = 0) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeIdxInStream,
                                                        streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
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

  SuperKernelBaseNode *CreateDefaultNode(uint64_t nodeId, uint32_t streamIdx, uint64_t preNodeId, uint64_t nextNodeId,
                                         uint64_t nodeIdxInStream) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeIdxInStream,
                                                        streamIdx, INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(preNodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeType = SkNodeType::NODE_DEFAULT;
    node->isFusible = true;
    auto *ptr = node.get();
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
    graph->tailNodes.clear();

    for (const auto &nodes : streamNodes) {
      graph->streams.emplace_back();
      if (!nodes.empty()) {
        graph->headNodes.push_back(nodes[0]);
        graph->tailNodes.push_back(nodes.back());
      } else {
        graph->headNodes.push_back(INVALID_TASK_ID);
        graph->tailNodes.push_back(INVALID_TASK_ID);
      }
    }
  }

  void CreateGraphFromStreams(const std::vector<std::vector<uint64_t>> &streams,
                              const std::map<uint64_t, std::vector<uint64_t>> &events,
                              const std::unordered_set<uint64_t> &defaultNodes = {},
                              const std::unordered_set<uint64_t> &resetNodes = {}) {
    std::unordered_map<uint64_t, uint64_t> notifyForWait;
    for (const auto &[notifyId, waitIds] : events) {
      for (const auto waitId : waitIds) {
        notifyForWait.emplace(waitId, notifyId);
      }
    }
    for (uint32_t streamId = 0; streamId < streams.size(); ++streamId) {
      const auto &ids = streams[streamId];
      for (size_t index = 0; index < ids.size(); ++index) {
        const auto id = ids[index];
        const auto previous = index == 0 ? INVALID_TASK_ID : ids[index - 1];
        const auto next = index + 1 == ids.size() ? INVALID_TASK_ID : ids[index + 1];
        if (const auto notify = events.find(id); notify != events.end()) {
          CreateNotifyNode(id, streamId, previous, next, id, notify->second, index);
        } else if (const auto wait = notifyForWait.find(id); wait != notifyForWait.end()) {
          CreateWaitNode(id, streamId, previous, next, wait->second, index);
        } else if (defaultNodes.count(id) != 0) {
          CreateDefaultNode(id, streamId, previous, next, index);
        } else if (resetNodes.count(id) != 0) {
          CreateResetNode(id, streamId, previous, next, id, index);
        } else {
          CreateKernelNodeWithCores(id, streamId, previous, next, 0, SkKernelType::AIC_ONLY, index);
        }
      }
    }
    SetupStreams(streams);
  }

  void ExpectConcurrencyRange(const std::vector<SuperKernelBaseNode *> &members,
                              const std::map<uint32_t, std::pair<uint64_t, uint64_t>> &expected) {
    const auto actual = lockDetector->CalculateScopeConcurrentRanges(members, members.front()->GetScopeId());
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto &[streamId, bounds] : expected) {
      SCOPED_TRACE(streamId);
      ASSERT_TRUE(actual.count(streamId) != 0);
      EXPECT_EQ(actual.at(streamId).firstNodeId, bounds.first);
      EXPECT_EQ(actual.at(streamId).lastNodeId, bounds.second);
    }
  }

  void SetScopeMembers(SuperKernelScopeInfo &scope, const std::vector<uint64_t> &ids) {
    std::map<uint32_t, ScopeStreamInfo> byStream;
    for (const auto id : ids) {
      auto *node = graph->GetNodeById(id);
      ASSERT_NE(node, nullptr);
      scope.AddNode(node);
      const auto streamId = node->GetStreamIdxInGraph();
      auto &info = byStream[streamId];
      info.streamIdx = streamId;
      if (info.nodeSize == 0) {
        info.headNodeIdx = id;
      }
      info.tailNodeIdx = id;
      ++info.nodeSize;
    }
    std::vector<ScopeStreamInfo> infos;
    for (const auto &[streamId, info] : byStream) {
      infos.push_back(info);
    }
    scope.SetScopeStreamInfos(std::move(infos));
    scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 1});
  }

  bool CanScopesOverlap(std::vector<SuperKernelScopeInfo> &scopes) {
    lockDetector->SetScopeInfos(scopes);
    const auto firstId = scopes[0].GetScopeId();
    const auto secondId = scopes[1].GetScopeId();
    const auto firstConcurrentRange = lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), firstId);
    const auto secondConcurrentRange = lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), secondId);
    return lockDetector->CanRunTogether(firstId, secondId,
                                        {{firstId, firstConcurrentRange}, {secondId, secondConcurrentRange}});
  }

  enum class UpstreamScopeWait { NONE, INTERNAL, EXTERNAL };

  void ExpectUpstreamScopeConcurrencyRange(bool notifyInScope, UpstreamScopeWait scopeWait,
                                           const std::map<uint32_t, std::pair<uint64_t, uint64_t>> &expected) {
    // stream0: w1 -> candidate k2
    // stream1: k10 -> n11/k11 -> notifyf n20 -> k21 -> k22
    // stream2: k30 -> w31/k31 -> k32 -> k33
    // stream3: n40/k40 -> k41
    // scopeM owns stream1 through k11 or k21, and stream2 through k32.
    std::map<uint64_t, std::vector<uint64_t>> events{{20, {1}}};
    if (scopeWait == UpstreamScopeWait::INTERNAL) {
      events[11] = {31};
    } else if (scopeWait == UpstreamScopeWait::EXTERNAL) {
      events[40] = {31};
    }
    // ======================= graph =======================
    /*
    stream0: [w1 -> k2]
    stream1: k10 -> [n11/w11] -> n20 -> k21 -> k22
    stream2: k30 -> [w31/k31] -> k32 -> k33
    stream3: n40 -> k41
    event20: n20 -> w1
    event11: n11 -> w31 (internal wait variant)
    event40: n40 -> w31 (external wait variant)
    */
    CreateGraphFromStreams({{1, 2}, {10, 11, 20, 21, 22}, {30, 31, 32, 33}, {40, 41}}, events);

    std::vector<SuperKernelScopeInfo> scopes(1);
    std::vector<uint64_t> members{10, 11, 30, 31, 32};
    if (notifyInScope) {
      members.insert(members.end(), {20, 21});
    }
    SetScopeMembers(scopes[0], members);
    lockDetector->SetScopeInfos(scopes);
    ExpectConcurrencyRange({graph->GetNodeById(2)}, expected);
  }

  void AppendRange(std::vector<uint64_t> &ids, uint64_t first, uint64_t last) {
    for (auto id = first; id <= last; ++id) {
      ids.push_back(id);
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
  std::unique_ptr<ScopeDeadlockDetector> scopeDetector;
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

TEST_F(TestLockDetector, CrossStreamWaitScopeConcurrencyRange) {
  // ======================= graph =======================
  /*                           sk
  stream0: k0(1c) -> [k1(1c) -> w2(event=1) -> k3(10c20v)]
  stream1: k4(4v) -> n5(event=1) -> k7(6c)
  stream2: k6(4v)
  event1: n5 -> w2

  */
  CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 1, SkKernelType::AIC_ONLY, 0);
  CreateKernelNodeWithCores(1, 0, 0, 2, 1, SkKernelType::AIC_ONLY, 1);
  CreateWaitNode(2, 0, 1, 3, 5, 2);
  CreateKernelNodeWithCores(3, 0, 2, INVALID_TASK_ID, 10, SkKernelType::MIX_AIC_1_2, 3);
  CreateKernelNodeWithCores(4, 1, INVALID_TASK_ID, 5, 4, SkKernelType::AIV_ONLY, 0);
  CreateNotifyNode(5, 1, 4, 7, 1, {2}, 1);
  CreateKernelNodeWithCores(6, 2, INVALID_TASK_ID, INVALID_TASK_ID, 4, SkKernelType::AIV_ONLY, 0);
  CreateKernelNodeWithCores(7, 1, 5, INVALID_TASK_ID, 6, SkKernelType::AIC_ONLY, 2);
  SetupStreams({{0, 1, 2, 3}, {4, 5, 7}, {6}});

  // Expected (first node ID, last node ID) for the WAIT-containing scope on every stream.
  const std::map<uint32_t, std::pair<uint64_t, uint64_t>> expectedConcurrentRange = {
      {0, {1, 2}},  // k1 -> w2
      {1, {4, 7}},  // k4 -> k7
      {2, {6, 6}},  // k6 -> k6
  };
  ExpectConcurrencyRange({graph->GetNodeById(1), graph->GetNodeById(2)}, expectedConcurrentRange);
}

TEST_F(TestLockDetector, CandidatePrefixConcurrencyRangeIncludesExternalNotifySuccessor) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> w2]
  stream1: k3 -> n4 -> k5
  event1: n4 -> w2
  */
  auto *kernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 2, SkKernelType::AIC_ONLY, 0);
  auto *wait = CreateWaitNode(2, 0, 1, INVALID_TASK_ID, 4, 1);
  CreateKernelNodeWithCores(3, 1, INVALID_TASK_ID, 4, 1, SkKernelType::AIC_ONLY, 0);
  CreateNotifyNode(4, 1, 3, 5, 1, {2}, 1);
  CreateKernelNodeWithCores(5, 1, 4, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY, 2);
  SetupStreams({{1, 2}, {3, 4, 5}});

  auto concurrentRange = lockDetector->CalculateScopeConcurrentRanges({kernel, wait}, kernel->GetScopeId());
  ASSERT_EQ(concurrentRange.size(), 2U);
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 2U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 3U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 5U);
}

TEST_F(TestLockDetector, CandidateConcurrencyRangeFollowsNotifyToWaitOnRight) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> w2] -> n3
  stream1: w4
  stream2: n5
  event2: n3 -> w4
  event1: n5 -> w2
  */
  auto *kernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 1, SkKernelType::AIC_ONLY, 0);
  auto *wait = CreateWaitNode(2, 0, 1, 3, 5, 1);
  CreateNotifyNode(3, 0, 2, INVALID_TASK_ID, 2, {4}, 2);
  CreateWaitNode(4, 1, INVALID_TASK_ID, INVALID_TASK_ID, 3, 0);
  CreateNotifyNode(5, 2, INVALID_TASK_ID, INVALID_TASK_ID, 1, {2}, 0);
  SetupStreams({{1, 2, 3}, {4}, {5}});

  const auto concurrentRange = lockDetector->CalculateScopeConcurrentRanges({kernel, wait}, kernel->GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 2U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, INVALID_TASK_ID);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, INVALID_TASK_ID);
}

TEST_F(TestLockDetector, RightSearchIncludesScopeContainingMatchedWait) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2
  stream1: k10 -> [w11 -> k12] -> n13
  stream2: k20 -> [n21 -> k22] -> k23
  stream3: k30 -> k31 -> k32
  event2: n2 -> w11
  event21: n21 -> w31
  scope1: stream1[w11,k12], stream2[n21,k22]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32}}, {{2, {11}}, {21, {31}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 21, 22});
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {10, 13}}, {2, {20, 23}}, {3, {30, 30}}});
}

TEST_F(TestLockDetector, RightSearchStopsBeforeTraversedMultiStreamScope) {
  // ======================= graph =======================
  /*
  stream0: k1 -> [k2 -> k3]
  stream1: [k10 -> k11 -> k12]
  scope1: stream0[k2,k3], stream1[k11,k12]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}}, {});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {2, 3, 11, 12});
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {10, 10}}});
}

TEST_F(TestLockDetector, NotifyPredecessorChecksOtherScopeStream) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2
  stream1: k10 -> n11
  stream2: k20 -> k21
  event11: n11 -> w2
  scope1: stream1[k10,n11], stream2[k21]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}}, {{11, {2}}});
  auto *candidate = graph->GetNodeById(1);
  candidate->nodeInfos.kernelInfos.numBlocks = 20;
  candidate->nodeInfos.kernelInfos.cubeNum = 20;
  auto *otherPredecessor = graph->GetNodeById(20);
  otherPredecessor->nodeInfos.kernelInfos.numBlocks = 20;
  otherPredecessor->nodeInfos.kernelInfos.cubeNum = 20;

  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {10, 11, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});

  ASSERT_TRUE(lockDetector->IsFusible(*candidate));
  EXPECT_FALSE(lockDetector->IsFusible(*graph->GetNodeById(2)));
}

TEST_F(TestLockDetector, NotifyPredecessorChecksExternalWaitInsideScope) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2
  stream1: k10 -> n11
  stream2: k20 -> k21
  stream3: k29 -> n30
  event11: n11 -> w2
  event30: n30 -> w21
  scope1: stream1[k10,n11], stream2[k21]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}, {29, 30}}, {{11, {2}}, {30, {21}}});
  auto *candidate = graph->GetNodeById(1);
  candidate->nodeInfos.kernelInfos.numBlocks = 20;
  candidate->nodeInfos.kernelInfos.cubeNum = 20;
  auto *notifyPredecessor = graph->GetNodeById(29);
  notifyPredecessor->nodeInfos.kernelInfos.numBlocks = 20;
  notifyPredecessor->nodeInfos.kernelInfos.cubeNum = 20;

  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {10, 11, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});

  ASSERT_TRUE(lockDetector->IsFusible(*candidate));
  EXPECT_FALSE(lockDetector->IsFusible(*graph->GetNodeById(2)));
}

TEST_F(TestLockDetector, NotifyAtScopeHeadChecksOtherScopeStream) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2
  stream1: n11
  stream2: k20 -> k21
  event11: n11 -> w2
  scope1: stream1[n11], stream2[k21]
  */
  CreateGraphFromStreams({{1, 2}, {11}, {20, 21}}, {{11, {2}}});
  auto *candidate = graph->GetNodeById(1);
  candidate->nodeInfos.kernelInfos.numBlocks = 20;
  candidate->nodeInfos.kernelInfos.cubeNum = 20;
  auto *otherPredecessor = graph->GetNodeById(20);
  otherPredecessor->nodeInfos.kernelInfos.numBlocks = 20;
  otherPredecessor->nodeInfos.kernelInfos.cubeNum = 20;

  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});

  ASSERT_TRUE(lockDetector->IsFusible(*candidate));
  EXPECT_FALSE(lockDetector->IsFusible(*graph->GetNodeById(2)));
}

TEST_F(TestLockDetector, NotifyPredecessorCountsScopeCoresOnce) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2
  stream1: k10 -> n11
  stream2: k20 -> k21
  event11: n11 -> w2
  scope1: stream1[k10,n11], stream2[k21]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}}, {{11, {2}}});
  auto *candidate = graph->GetNodeById(1);
  candidate->nodeInfos.kernelInfos.numBlocks = 20;
  candidate->nodeInfos.kernelInfos.cubeNum = 20;
  auto *otherPredecessor = graph->GetNodeById(20);
  otherPredecessor->nodeInfos.kernelInfos.numBlocks = 4;
  otherPredecessor->nodeInfos.kernelInfos.cubeNum = 4;

  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {10, 11, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});

  ASSERT_TRUE(lockDetector->IsFusible(*candidate));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(2)));
  EXPECT_EQ(lockDetector->depOpCubeNum, 8U);
}

TEST_F(TestLockDetector, LeftSearchNotifyAfterMultiStreamScopeWithoutWait) {
  ExpectUpstreamScopeConcurrencyRange(false, UpstreamScopeWait::NONE,
                                      {{0, {2, 2}}, {1, {21, 22}}, {2, {33, 33}}, {3, {40, 41}}});
}

TEST_F(TestLockDetector, LeftSearchNotifyAfterMultiStreamScopeWithExternalWait) {
  ExpectUpstreamScopeConcurrencyRange(false, UpstreamScopeWait::EXTERNAL,
                                      {{0, {2, 2}}, {1, {21, 22}}, {2, {33, 33}}, {3, {41, 41}}});
}

TEST_F(TestLockDetector, LeftSearchNotifyInMultiStreamScopeWithoutWait) {
  ExpectUpstreamScopeConcurrencyRange(true, UpstreamScopeWait::NONE,
                                      {{0, {2, 2}}, {1, {10, 22}}, {2, {30, 33}}, {3, {40, 41}}});
}

TEST_F(TestLockDetector, LeftSearchNotifyInMultiStreamScopeWithInternalWait) {
  ExpectUpstreamScopeConcurrencyRange(true, UpstreamScopeWait::INTERNAL,
                                      {{0, {2, 2}}, {1, {10, 22}}, {2, {30, 33}}, {3, {40, 41}}});
}

TEST_F(TestLockDetector, LeftSearchNotifyInMultiStreamScopeWithExternalWait) {
  ExpectUpstreamScopeConcurrencyRange(true, UpstreamScopeWait::EXTERNAL,
                                      {{0, {2, 2}}, {1, {10, 22}}, {2, {30, 33}}, {3, {41, 41}}});
}

TEST_F(TestLockDetector, LeftSearchIncludesUpstreamScopeWithLaterWait) {
  // ======================= graph =======================
  /*
  stream0: k0 -> n1 -> w2 -> k3 -> w8
  stream1: w4 -> k5 -> w6 -> n7
  stream2: n9
  stream3: k11 -> n10 -> k12
  event1: n1 -> w4
  event2: n10 -> w2
  event6: n9 -> w6
  event7: n7 -> w8
  */
  auto *k0 = CreateKernelNodeWithCores(0, 0, INVALID_TASK_ID, 1, 1, SkKernelType::AIC_ONLY, 0);
  auto *n1 = CreateNotifyNode(1, 0, 0, 2, 1, {4}, 1);
  auto *w2 = CreateWaitNode(2, 0, 1, 3, 10, 2);
  auto *k3 = CreateKernelNodeWithCores(3, 0, 2, 8, 1, SkKernelType::AIC_ONLY, 3);
  CreateWaitNode(8, 0, 3, INVALID_TASK_ID, 7, 4);
  CreateWaitNode(4, 1, INVALID_TASK_ID, 5, 1, 0);
  auto *k5 = CreateKernelNodeWithCores(5, 1, 4, 6, 1, SkKernelType::AIC_ONLY, 1);
  auto *w6 = CreateWaitNode(6, 1, 5, 7, 9, 2);
  auto *n7 = CreateNotifyNode(7, 1, 6, INVALID_TASK_ID, 7, {8}, 3);
  CreateNotifyNode(9, 2, INVALID_TASK_ID, INVALID_TASK_ID, 6, {6}, 0);
  CreateKernelNodeWithCores(11, 3, INVALID_TASK_ID, 10, 1, SkKernelType::AIC_ONLY, 0);
  CreateNotifyNode(10, 3, 11, 12, 2, {2}, 1);
  CreateKernelNodeWithCores(12, 3, 10, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY, 2);
  SetupStreams({{0, 1, 2, 3, 8}, {4, 5, 6, 7}, {9}, {11, 10, 12}});

  std::vector<SuperKernelScopeInfo> scopes(2);
  for (auto *node : {k0, n1, w2, k3}) {
    scopes[0].AddNode(node);
  }
  scopes[0].SetScopeStreamInfos({{0, 0, 3, 4}});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 1});
  for (auto *node : {k5, w6, n7}) {
    scopes[1].AddNode(node);
  }
  scopes[1].SetScopeStreamInfos({{1, 5, 7, 3}});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[1].GetScopeId());

  const auto concurrentRange = lockDetector->CalculateScopeConcurrentRanges({k5, w6, n7}, scopes[1].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 0U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 8U);
  EXPECT_EQ(concurrentRange.at(3).firstNodeId, 11U);
  EXPECT_EQ(concurrentRange.at(3).lastNodeId, 12U);
}

TEST_F(TestLockDetector, ParallelWaitScopeCombinationRespectsSharedStreams) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k3 -> n4
  stream2: k10 -> w30
  stream3: w11 -> k20
  stream4: w21
  event1: n4 -> w2
  event100: w11 waits for external event
  event101: w21 waits for external event
  event102: w30 waits for external event
  scope1: stream2[k10], stream3[w11]
  scope2: stream3[k20], stream4[w21]
  scope3: stream2[w30]
  */
  auto *currentKernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 2, 4, SkKernelType::AIC_ONLY, 0);
  auto *currentWait = CreateWaitNode(2, 0, 1, INVALID_TASK_ID, 4, 1);
  CreateKernelNodeWithCores(3, 1, INVALID_TASK_ID, 4, 1, SkKernelType::AIC_ONLY, 0);
  CreateNotifyNode(4, 1, 3, INVALID_TASK_ID, 1, {2}, 1);
  auto *w1s4 = CreateKernelNodeWithCores(10, 2, INVALID_TASK_ID, 30, 2, SkKernelType::AIC_ONLY, 0);
  auto *w1s5 = CreateWaitNode(11, 3, INVALID_TASK_ID, 20, 100, 0);
  auto *w2s5 = CreateKernelNodeWithCores(20, 3, 11, INVALID_TASK_ID, 3, SkKernelType::AIC_ONLY, 1);
  auto *w2s6 = CreateWaitNode(21, 4, INVALID_TASK_ID, INVALID_TASK_ID, 101, 0);
  auto *w3s4 = CreateWaitNode(30, 2, 10, INVALID_TASK_ID, 102, 1);
  SetupStreams({{1, 2}, {3, 4}, {10, 30}, {11, 20}, {21}});

  std::vector<SuperKernelScopeInfo> scopes(3);
  scopes[0].AddNode(w1s4);
  scopes[0].AddNode(w1s5);
  scopes[0].SetScopeStreamInfos({{2, 10, 10, 1}, {3, 11, 11, 1}});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  scopes[1].AddNode(w2s5);
  scopes[1].AddNode(w2s6);
  scopes[1].SetScopeStreamInfos({{3, 20, 20, 1}, {4, 21, 21, 1}});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});
  scopes[2].AddNode(w3s4);
  scopes[2].SetScopeStreamInfos({{2, 30, 30, 1}});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  lockDetector->SetScopeInfos(scopes);
  const uint16_t currentScopeId = 500;
  currentKernel->SetScopeId(currentScopeId);
  currentWait->SetScopeId(currentScopeId);
  lockDetector->SetCurrentScopeId(currentScopeId);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});

  const auto savedDeviceCubeNum = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 12;
  EXPECT_TRUE(lockDetector->IsFusible(*currentKernel));
  EXPECT_TRUE(lockDetector->IsFusible(*currentWait));  // 4 + 1 + max(6, 3 + 4) = 12

  lockDetector->Reset();
  lockDetector->SetCurrentScopeId(currentScopeId);
  lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  LockDetector::deviceRealCubeNum = 11;
  EXPECT_TRUE(lockDetector->IsFusible(*currentKernel));
  EXPECT_TRUE(lockDetector->IsFusible(*currentWait));
  LockDetector::deviceRealCubeNum = savedDeviceCubeNum;
}

TEST_F(TestLockDetector, CandidateNotifyWithoutWaitKeepsConcurrentRangeUnbounded) {
  // stream0: k1 -> n1 -> k2; stream1: k3 -> w1 -> n2 -> k4; stream2: k5 -> w2 -> k6.
  // Original stream IDs are 0, 1, and 2, and the compact test stream indices are the same.
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22}}, {{2, {11}}, {12, {21}}});

  auto *k1 = graph->GetNodeById(1);
  auto *n1 = graph->GetNodeById(2);
  auto *k2 = graph->GetNodeById(3);
  k1->nodeInfos.kernelInfos.cubeNum = 16;
  k2->nodeInfos.kernelInfos.cubeNum = 16;

  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {1, 2, 3});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  SetScopeMembers(scopes[1], {10, 11, 12, 13});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  SetScopeMembers(scopes[2], {20, 21, 22});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 15});
  lockDetector->SetScopeInfos(scopes);

  // sk2/sk3 both contain WAIT nodes, so each scope's concurrentRange should cover all three streams.
  lockDetector->SetCurrentScopeId(scopes[1].GetScopeId());
  const auto scope2ConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), scopes[1].GetScopeId());
  ASSERT_EQ(scope2ConcurrentRange.size(), 3U);
  EXPECT_EQ(scope2ConcurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(scope2ConcurrentRange.at(0).lastNodeId, 3U);
  EXPECT_EQ(scope2ConcurrentRange.at(1).firstNodeId, 10U);
  EXPECT_EQ(scope2ConcurrentRange.at(1).lastNodeId, 13U);
  EXPECT_EQ(scope2ConcurrentRange.at(2).firstNodeId, 20U);
  EXPECT_EQ(scope2ConcurrentRange.at(2).lastNodeId, 22U);

  lockDetector->SetCurrentScopeId(scopes[2].GetScopeId());
  const auto scope3ConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[2].GetNodes(), scopes[2].GetScopeId());
  ASSERT_EQ(scope3ConcurrentRange.size(), 3U);
  EXPECT_EQ(scope3ConcurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(scope3ConcurrentRange.at(0).lastNodeId, 3U);
  EXPECT_EQ(scope3ConcurrentRange.at(1).firstNodeId, 10U);
  EXPECT_EQ(scope3ConcurrentRange.at(1).lastNodeId, 13U);
  EXPECT_EQ(scope3ConcurrentRange.at(2).firstNodeId, 20U);
  EXPECT_EQ(scope3ConcurrentRange.at(2).lastNodeId, 22U);
}

TEST_F(TestLockDetector, InternalNotifyKeepsConservativeScopeOverlap) {
  // ======================= graph =======================
  /*
  stream0: [n1 -> k2]
  stream1: [w3]
  event1: n1 -> w3
  scope0: stream0[n1,k2]
  scope1: stream1[w3]
  */
  CreateNotifyNode(1, 0, INVALID_TASK_ID, 2, 1, {3}, 0);
  CreateKernelNodeWithCores(2, 0, 1, INVALID_TASK_ID, 1, SkKernelType::AIC_ONLY, 1);
  CreateWaitNode(3, 1, INVALID_TASK_ID, INVALID_TASK_ID, 1, 0);
  SetupStreams({{1, 2}, {3}});

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {3});
  EXPECT_TRUE(CanScopesOverlap(scopes));

  scopes[0] = SuperKernelScopeInfo{};
  SetScopeMembers(scopes[0], {1});
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

TEST_F(TestLockDetector, SameStreamNotifyAfterSKRangeRejectsWait) {
  // ======================= graph =======================
  /*        sk
  stream0: [k0(1c)] -> w1(event=1)

  stream1: k2(4c) -> n1(event=1)
  */
}

TEST_F(TestLockDetector, SKRangeIsTrackedIndependentlyPerStream) {
  // ======================= graph dsv4 =======================
  /*
  stream15: w6(e0) -> w41(e40) -> k42...k45 -> n46(e46) -> k47...k106(4c) -> n107(e107) -> w203(e186) -> k204...k206 ->
  w207(e202) -> k208 -> n209(e209) stream16: w5(e0) -> w108(e46) -> default109 -> n110(e110) -> default111 -> default112
  stream17: w4(e0) -> w127(e126) -> default128 -> w138(e137) -> default139
  stream18: w3(e0) -> w141(e134) -> k142 -> k143 -> n144(e144) -> k145 -> k146(4c) -> n147(e147)
  stream19: w2(e0) -> w159(e153) -> k160...k166 -> w167(e107) -> k168 -> n169(e169) -> k170 -> k171 -> k172
  stream20: w1(e0)
  stream21: n0(e0) -> k7...k39 -> n40(e40) -> k113...k125 -> n126(e126) -> k129 -> n130(e130) ->
            k131...k133 -> n134(e134) -> k135 -> k136 -> n137(e137) -> k140 -> w151(e144) ->
            k152 -> n153(e153) -> k154 -> k155 -> w156(e147) -> w157(e107) -> k158 ->
            w173(e169) -> w174(e110) -> k175...k185 -> n186(e186) -> k187...k201 -> n202(e202) ->
            k210 -> k211 -> k212 -> default213 -> k214 -> w215(e209) -> k216 -> k217

  event0: n0 -> w1, w2, w3, w4, w5, w6
  event40: n40 -> w41       event46: n46 -> w108       event107: n107 -> w157, w167
  event110: n110 -> w174   event126: n126 -> w127       event134: n134 -> w141
  event137: n137 -> w138   event144: n144 -> w151       event147: n147 -> w156
  event130: n130 -> (no wait)
  event153: n153 -> w159   event169: n169 -> w173       event186: n186 -> w203
  event202: n202 -> w207   event209: n209 -> w215

  Possible concurrent ranges with the scope memberships below (empty means no overlapping task):
  scope 1951 (WAIT nodes 156,157):
    stream15: node 6 -> node 107
    stream16: node 5 -> node 112
    stream17: node 4 -> node 139
    stream18: node 144 -> node 147
    stream19: node 2 -> node 172
    stream20: node 1 -> node 1
    stream21: node 152 -> node 158
  scope 1192 (WAIT node 167):
    stream15: node 6 -> node 209
    stream16: node 5 -> node 112
    stream17: node 4 -> node 139
    stream18: node 144 -> node 147
    stream19: node 163 -> node 169
    stream20: node 1 -> node 1
    stream21: node 0 -> node 217

  Test stream mapping: original stream15 -> streamIdxInGraph 0, stream16 -> 1, stream17 -> 2,
  stream18 -> 3, stream19 -> 4, stream20 -> 5, stream21 -> 6.
  */
  // scope 1190：125、126、129
  // scope 1951：152、153、154、155、156、157、158（stream21）
  // scope 1192：163、164、165、166、167、168、169（stream19）
  // scope 1955：175-183
  // scope 1649：184、185、186
  // scope 1194：188
  // scope 1195：196、197、198
  // scope 1956：214
  // scope 1957：216、217

  std::vector<std::vector<uint64_t>> streams(7);
  streams[0] = {6, 41};
  AppendRange(streams[0], 42, 107);
  AppendRange(streams[0], 203, 209);
  streams[1] = {5};
  AppendRange(streams[1], 108, 112);
  streams[2] = {4, 127, 128, 138, 139};
  streams[3] = {3};
  AppendRange(streams[3], 141, 147);
  streams[4] = {2};
  AppendRange(streams[4], 159, 172);
  streams[5] = {1};
  streams[6] = {0};
  AppendRange(streams[6], 7, 40);
  AppendRange(streams[6], 113, 126);
  AppendRange(streams[6], 129, 137);
  streams[6].push_back(140);
  AppendRange(streams[6], 151, 158);
  AppendRange(streams[6], 173, 202);
  AppendRange(streams[6], 210, 217);
  CreateGraphFromStreams(streams,
                         {{0, {1, 2, 3, 4, 5, 6}},
                          {40, {41}},
                          {46, {108}},
                          {107, {157, 167}},
                          {110, {174}},
                          {126, {127}},
                          {130, {}},
                          {134, {141}},
                          {137, {138}},
                          {144, {151}},
                          {147, {156}},
                          {153, {159}},
                          {169, {173}},
                          {186, {203}},
                          {202, {207}},
                          {209, {215}}},
                         {109, 111, 112, 128, 139, 213});
  for (const auto id : {152U, 154U, 155U, 158U, 163U, 164U, 165U, 166U, 168U}) {
    auto *kernel = graph->GetNodeById(id);
    kernel->nodeInfos.kernelInfos.numBlocks = 16;
    kernel->nodeInfos.kernelInfos.cubeNum = 16;
  }
  for (const auto id : {106U, 146U}) {
    auto *kernel = graph->GetNodeById(id);
    kernel->nodeInfos.kernelInfos.numBlocks = 4;
    kernel->nodeInfos.kernelInfos.cubeNum = 4;
  }

  std::vector<SuperKernelScopeInfo> scopes(9);
  SetScopeMembers(scopes[0], {125, 126, 129});
  std::vector<uint64_t> scope1951Ids;
  AppendRange(scope1951Ids, 152, 158);
  SetScopeMembers(scopes[1], scope1951Ids);
  std::vector<uint64_t> scope1192Ids;
  AppendRange(scope1192Ids, 163, 169);
  SetScopeMembers(scopes[2], scope1192Ids);
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  std::vector<uint64_t> ids;
  AppendRange(ids, 175, 183);
  SetScopeMembers(scopes[3], ids);
  SetScopeMembers(scopes[4], {184, 185, 186});
  SetScopeMembers(scopes[5], {188});
  SetScopeMembers(scopes[6], {196, 197, 198});
  SetScopeMembers(scopes[7], {214});
  SetScopeMembers(scopes[8], {216, 217});
  lockDetector->SetScopeInfos(scopes);

  // Check the possible concurrent ranges used by both resource passes.
  ExpectConcurrencyRange(
      scopes[1].GetNodes(),
      {{0, {6, 107}}, {1, {5, 112}}, {2, {4, 139}}, {3, {145, 147}}, {4, {2, 172}}, {5, {1, 1}}, {6, {152, 158}}});
  ExpectConcurrencyRange(
      scopes[2].GetNodes(),
      {{0, {6, 209}}, {1, {5, 112}}, {2, {4, 139}}, {3, {145, 147}}, {4, {163, 169}}, {5, {1, 1}}, {6, {151, 217}}});

  const auto savedDeviceCubeNum = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  for (const auto &[scopeIndex, firstId, waitId] :
       {std::tuple<size_t, uint64_t, uint64_t>{1, 152, 156}, {2, 163, 167}}) {
    lockDetector->Reset();
    lockDetector->SetCurrentScopeId(scopes[scopeIndex].GetScopeId());
    lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
    for (uint64_t id = firstId; id < waitId; ++id) {
      EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(id))) << id;
    }
    // A single WAIT scope and its notify predecessor fit in 32 cores.
    EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(waitId))) << waitId;
    EXPECT_EQ(lockDetector->depOpCubeNum, 4U) << waitId;
  }
  uint64_t riskWaitId = INVALID_TASK_ID;
  const auto *riskScope = scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, 156U);
  EXPECT_EQ(riskScope, &scopes[1]);

  // Match the parallel pass's first split: remove w156 and retain both sides as separate scopes.
  scopes[1].ClearNodeScopeIds();
  scopes.erase(scopes.begin() + 1);
  SuperKernelScopeInfo scopeBeforeWait;
  SetScopeMembers(scopeBeforeWait, {152, 153, 154, 155});
  scopeBeforeWait.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes.insert(scopes.begin() + 1, std::move(scopeBeforeWait));
  SuperKernelScopeInfo scopeAfterWait;
  SetScopeMembers(scopeAfterWait, {158});
  scopeAfterWait.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes.insert(scopes.begin() + 2, std::move(scopeAfterWait));
  graph->GetNodeById(156)->SetIsFusible(false);
  graph->GetNodeById(157)->SetIsFusible(false);

  riskWaitId = INVALID_TASK_ID;
  const auto *nextRiskScope = scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
  EXPECT_EQ(nextRiskScope, nullptr);
  LockDetector::deviceRealCubeNum = savedDeviceCubeNum;
}

TEST_F(TestLockDetector, SampleScopeConcurrencyRange) {
  // ======================= graph =======================
  /*
  stream180: n0(e0) -> n4(e4) -> k8..s184【.k21 -> n22(e22) -> k23...k26 -> w34(e30) -> k35...k37 -> w38(e33) -> k39 ->
  k40 -> w41(e7) -> k42.】..k56 -> n57(e57) -> k58...k63 -> n64(e64) -> k65...k73 -> w74(e71) -> k75...k94 -> n95(e95)
  -> w96(e95) -> w98(e97) -> w100(e99) -> reset101...reset112 stream179: w1(e0) -> w65(e57) -> k66 -> k67 -> k68 ->
  w69(e64) -> k70 -> n71(e71) -> n99(e99) stream178: w2(e0) -> 【w27(e22) -> k28 -> k29 -> n30(e30) -> k31 -> k32 ->
  n33(e33) -> n97(e97) 】 stream177: w3(e0) -> w5(e4) -> default6 -> n7(e7) -> n95(e95)

  event0: n0 -> w1, w2, w3
  event4: n4 -> w5       event7: n7 -> w41
  event22: n22 -> w27    event30: n30 -> w34
  event33: n33 -> w38    event57: n57 -> w65
  event64: n64 -> w69    event71: n71 -> w74
  event95: n95 -> w96    event97: n97 -> w98

  scope184: 11-44, 97
  scope185: 46-71, 99
  scope186: 73-88
  scope187: 90-94, 96

  Possible concurrent ranges with the scope memberships below (empty means no overlapping task):
  scope 184:
    stream177: node 3 -> node 95
    stream178: node 27 -> node 97
    stream179: node 1 -> node 1
    stream180: node 11 -> node 44
  scope 185:
    stream177: node 95 -> node 95
    stream178: empty
    stream179: node 65 -> node 69
    stream180: node 46 -> node 64
  scope 186:
    stream177: node 95 -> node 95
    stream178: empty
    stream179: empty
    stream180: node 73 -> node 88
  scope 187:
    stream177: node 95 -> node 95
    stream178: empty
    stream179: empty
    stream180: node 90 -> node 96

  Test stream mapping: original stream177 -> streamIdxInGraph 0, stream178 -> 1, stream179 -> 2,
  stream180 -> 3.
  */

  // Bounds are inclusive node IDs in the possible concurrent range.
  const std::map<uint32_t, std::pair<uint64_t, uint64_t>> scope184ConcurrentRange = {
      {0, {3, 95}}, {1, {27, 97}}, {2, {1, 1}}, {3, {11, 44}}};
  const std::map<uint32_t, std::pair<uint64_t, uint64_t>> scope185ConcurrentRange = {
      {0, {95, 95}}, {1, {INVALID_TASK_ID, INVALID_TASK_ID}}, {2, {65, 99}}, {3, {46, 64}}};
  const std::map<uint32_t, std::pair<uint64_t, uint64_t>> scope186ConcurrentRange = {
      {0, {95, 95}}, {1, {INVALID_TASK_ID, INVALID_TASK_ID}}, {2, {INVALID_TASK_ID, INVALID_TASK_ID}}, {3, {73, 88}}};
  const std::map<uint32_t, std::pair<uint64_t, uint64_t>> scope187ConcurrentRange = {
      {0, {95, 95}}, {1, {INVALID_TASK_ID, INVALID_TASK_ID}}, {2, {INVALID_TASK_ID, INVALID_TASK_ID}}, {3, {90, 96}}};

  std::vector<std::vector<uint64_t>> streams(4);
  streams[0] = {3, 5, 6, 7, 95};
  streams[1] = {2};
  AppendRange(streams[1], 27, 33);
  streams[1].push_back(97);
  streams[2] = {1};
  AppendRange(streams[2], 65, 71);
  streams[2].push_back(99);
  streams[3] = {0, 4};
  AppendRange(streams[3], 8, 26);
  AppendRange(streams[3], 34, 64);
  AppendRange(streams[3], 72, 94);
  streams[3].push_back(96);
  streams[3].push_back(98);
  AppendRange(streams[3], 100, 112);
  CreateGraphFromStreams(streams,
                         {{0, {1, 2, 3}},
                          {4, {5}},
                          {7, {41}},
                          {22, {27}},
                          {30, {34}},
                          {33, {38}},
                          {57, {65}},
                          {64, {69}},
                          {71, {74}},
                          {95, {96}},
                          {97, {98}},
                          {99, {100}}},
                         {6, 72}, {101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112});
  for (const auto &stream : streams) {
    for (const auto id : stream) {
      auto *node = graph->GetNodeById(id);
      if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
        node->nodeInfos.kernelInfos.numBlocks = 4;
        node->nodeInfos.kernelInfos.cubeNum = 4;
      }
    }
  }

  std::vector<SuperKernelScopeInfo> scopes(4);
  std::vector<uint64_t> ids;
  AppendRange(ids, 11, 44);
  ids.push_back(97);
  SetScopeMembers(scopes[0], ids);
  ids.clear();
  AppendRange(ids, 46, 71);
  ids.push_back(99);
  SetScopeMembers(scopes[1], ids);
  ids.clear();
  AppendRange(ids, 73, 88);
  SetScopeMembers(scopes[2], ids);
  SetScopeMembers(scopes[3], {90, 91, 92, 93, 94, 96});
  for (auto &scope : scopes) {
    scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  }
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange(scopes[0].GetNodes(), scope184ConcurrentRange);
  ExpectConcurrencyRange(scopes[1].GetNodes(), scope185ConcurrentRange);
  ExpectConcurrencyRange(scopes[2].GetNodes(), scope186ConcurrentRange);
  ExpectConcurrencyRange(scopes[3].GetNodes(), scope187ConcurrentRange);

  for (size_t scopeIndex = 0; scopeIndex < scopes.size(); ++scopeIndex) {
    lockDetector->Reset();
    lockDetector->SetCurrentScopeId(scopes[scopeIndex].GetScopeId());
    lockDetector->SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
    for (auto *node : scopes[scopeIndex].GetNodes()) {
      if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
        EXPECT_TRUE(lockDetector->IsFusible(*node)) << node->GetNodeId();
      } else {
        ASSERT_TRUE(lockDetector->IsFusible(*node)) << node->GetNodeId();
      }
    }
  }
}

// 依赖闭包如何算核数
/*
eq1：only kernel
stream1：k1 -> n1 -> k3 -> k9
stream2：k2 -> w1 -> k6 -> n2
stream3：k4 -> k5 -> [k7 -> w2 -> k8]
*/
TEST_F(TestLockDetector, DependencyEq1KernelOnlyPredecessors) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}, {20, 21, 22, 23, 24}}, {{2, {11}}, {13, {23}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {22, 23, 24});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 8});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(22)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(23)));
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 24U);
}

/*
eq2：only single stream scope
stream1：k1 -> n1 -> k3 -> k9
stream2：k2 -> [w1 -> k6] -> n2
stream3：k4 -> k5 -> [k7 -> w2 -> k8]
*/
TEST_F(TestLockDetector, DependencyEq2SingleStreamWaitScopes) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}, {20, 21, 22, 23, 24}}, {{2, {11}}, {13, {23}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12});
  SetScopeMembers(scopes[1], {22, 23, 24});
  EXPECT_TRUE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
}

/*
eq3：exist multi stream scope
stream1：k1 -> n1 -> k3 -> k9
stream2：k10 -> [k11 -> k12] -> k13
stream3：k2 -> [w1 -> k6] -> n2
stream4：k4 -> k5 -> [k7 -> w2 -> k8]

scope1: w1 k6 k11 k12
*/
TEST_F(TestLockDetector, DependencyEq3MultiStreamProducerScope) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33, 34}},
                         {{2, {21}}, {23, {33}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12, 21, 22});
  SetScopeMembers(scopes[1], {32, 33, 34});
  EXPECT_TRUE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
}

/*
eq4：multiple single stream scopes in the dependency closure
stream1：k1 -> n1
stream2：k2 -> [w1 -> k3 -> n2]
stream3：k4 -> [w2 -> k5 -> n3]
stream4：k6 -> [w3 -> k7]

scope1: w1 k3 n2
scope2: w2 k5 n3
scope3: w3 k7
*/
TEST_F(TestLockDetector, DependencyEq4ThreeWaitScopesInDependencyChain) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32}},
                         {{2, {11}}, {13, {21}}, {23, {31}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {11, 12, 13});
  SetScopeMembers(scopes[1], {21, 22, 23});
  SetScopeMembers(scopes[2], {31, 32});
  lockDetector->SetScopeInfos(scopes);
  const auto firstId = scopes[0].GetScopeId();
  const auto secondId = scopes[1].GetScopeId();
  EXPECT_TRUE(lockDetector->CanRunTogether(
      firstId, secondId,
      {{firstId, lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), firstId)},
       {secondId, lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), secondId)}}));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq5：kernel nodes between scopes in the dependency closure
stream1：k1 -> n1
stream2：k2 -> [w1 -> k3] -> k4 -> k5 -> n2
stream3：k6 -> [k7 -> w2 -> k8]

scope1: w1 k3
scope2: k7 w2 k8
*/
TEST_F(TestLockDetector, DependencyEq5KernelsBetweenWaitScopes) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13, 14, 15}, {20, 21, 22, 23, 24}}, {{2, {11}}, {15, {23}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12});
  SetScopeMembers(scopes[1], {22, 23, 24});
  EXPECT_TRUE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq6：multi stream scope with kernels after the scope in the dependency closure
stream1：k1 -> n1
stream2：k2 -> [w1 -> k3] -> k4 -> n2
stream3：k5 -> [k6 -> k7] -> k8
stream4：k9 -> [k10 -> w2 -> k11]

scope1: w1 k3 k6 k7
scope2: k10 w2 k11
*/
TEST_F(TestLockDetector, DependencyEq6MultiStreamScopePrecedesExternalKernels) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13, 14}, {20, 21, 22, 23}, {30, 31, 32, 33}}, {{2, {11}}, {14, {32}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12, 21, 22});
  SetScopeMembers(scopes[1], {31, 32, 33});
  EXPECT_TRUE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq7：outside wait scope with a shared dependency
stream1：k1 -> n1
stream2：[k2 -> w1 -> k3 -> n2]
stream3：[k4 -> w2 -> k5]
stream4：[k6 -> w3 -> k7]

scope1: k2 w1 k3 n2
scope2: k4 w2 k5
scope3: k6 w3 k7
candidate: scope2
event: n1 -> w1, w3; n2 -> w2
*/
TEST_F(TestLockDetector, DependencyEq7SharedNotifyAcrossWaitScopes) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13}, {20, 21, 22}, {30, 31, 32}}, {{2, {11, 31}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {10, 11, 12, 13});
  SetScopeMembers(scopes[1], {20, 21, 22});
  SetScopeMembers(scopes[2], {30, 31, 32});
  lockDetector->SetScopeInfos(scopes);
  const auto firstId = scopes[0].GetScopeId();
  const auto secondId = scopes[1].GetScopeId();
  EXPECT_TRUE(lockDetector->CanRunTogether(
      firstId, secondId,
      {{firstId, lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), firstId)},
       {secondId, lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), secondId)}}));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq8：outside wait scope with an independent dependency
stream1：k1 -> n1
stream2：[k2 -> w1 -> k3]
stream3：k4 -> n2
stream4：[k5 -> w2 -> k6]

scope1: k2 w1 k3
scope2: k5 w2 k6
candidate: scope1
*/
TEST_F(TestLockDetector, DependencyEq8IndependentNotifyBranches) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}, {20, 21}, {30, 31, 32}}, {{2, {11}}, {21, {31}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {10, 11, 12});
  SetScopeMembers(scopes[1], {30, 31, 32});
  EXPECT_TRUE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq9：fusion introduces a resource deadlock on an acyclic graph
stream1：[k1 -> n1 -> w1 -> k2]
stream2：[k3 -> w2 -> n2 -> k4]

scope1: k1 n1 w1 k2，20c
scope2: k3 w2 n2 k4，20c
event: n1 -> w2; n2 -> w1
device: 32c
*/
TEST_F(TestLockDetector, DependencyEq9CyclicScopesCannotBothStart) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}}, {{2, {11}}, {12, {3}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2, 3, 4});
  SetScopeMembers(scopes[1], {10, 11, 12, 13});
  for (auto &scope : scopes) {
    scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});
  }
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 20;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 20;
  EXPECT_TRUE(CanScopesOverlap(scopes));
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(1)));
  EXPECT_FALSE(lockDetector->IsFusible(*graph->GetNodeById(3)));
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq10：producer scope without a wait in the dependency closure
stream1：[k1 -> n1 -> k2]
stream2：[k3 -> w1 -> n2 -> k4]
stream3：[k5 -> w2 -> k6]

scope1: k1 n1 k2，16c
scope2: k3 w1 n2 k4，16c
scope3: k5 w2 k6，15c
candidate: scope3
event: n1 -> w1; n2 -> w2
device: 32c
schedule: scope2 and scope3 enter their waits before scope1 starts
*/
TEST_F(TestLockDetector, DependencyEq10WaitFreeProducerCannotStartWithBothWaitScopesResident) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22}}, {{2, {11}}, {12, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {1, 2, 3});
  SetScopeMembers(scopes[1], {10, 11, 12, 13});
  SetScopeMembers(scopes[2], {20, 21, 22});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 15});
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 15;
  lockDetector->SetScopeInfos(scopes);
  const auto secondId = scopes[1].GetScopeId();
  const auto thirdId = scopes[2].GetScopeId();
  EXPECT_TRUE(lockDetector->CanRunTogether(
      secondId, thirdId,
      {{secondId, lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), secondId)},
       {thirdId, lockDetector->CalculateScopeConcurrentRanges(scopes[2].GetNodes(), thirdId)}}));
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, 11U);
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq11：waiting scopes serialized on the same stream
stream1：k1 -> n1
stream2：[k2 -> w1 -> k3] -> [k4 -> w2 -> k5]
stream3：k6 -> n2

scope1: k2 w1 k3
scope2: k4 w2 k5
event: n1 -> w1; n2 -> w2
*/
TEST_F(TestLockDetector, DependencyEq11SameStreamWaitScopesCannotOverlap) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13, 14, 15}, {20, 21}}, {{2, {11}}, {21, {14}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {10, 11, 12});
  SetScopeMembers(scopes[1], {13, 14, 15});
  EXPECT_FALSE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
}

/*
eq12：outside wait scope can release while both scopes hold cores
stream1：k1(1c) -> n1
stream2：[k2 -> w1 -> k3]
stream3：k4(1c) -> n2
stream4：[k5 -> w2 -> k6]

scope1: k2 w1 k3，16c
scope2: k5 w2 k6，15c
candidate: scope1
event: n1 -> w1; n2 -> w2
device: 32c
*/
TEST_F(TestLockDetector, DependencyEq12TwoWaitScopesLeaveOneCoreForProducer) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}, {20, 21}, {30, 31, 32}}, {{2, {11}}, {21, {31}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {10, 11, 12});
  SetScopeMembers(scopes[1], {30, 31, 32});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 15});
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(30)->nodeInfos.kernelInfos.cubeNum = 15;
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 1;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 1;
  EXPECT_TRUE(CanScopesOverlap(scopes));
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq13：outside scope without a wait can finish and release its cores
stream1：k1(1c) -> n1
stream2：[k2 -> w1 -> k3]
stream3：[k4 -> k5]

scope1: k2 w1 k3，16c
scope2: k4 k5，20c
candidate: scope1
event: n1 -> w1
device: 32c
*/
TEST_F(TestLockDetector, DependencyEq13UnrelatedWaitFreeScopeDoesNotBlockNotify) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}, {20, 21}}, {{2, {11}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {10, 11, 12});
  SetScopeMembers(scopes[1], {20, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 20});
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 20;
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 1;
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(10)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(11)));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq14：two waits in one scope need producers at different stages
stream1：k1(16c) -> n1
stream2：w3 -> k2(16c) -> n2
stream3：[k3 -> w1 -> n3 -> k4 -> w2 -> k5]

scope1: k3 w1 n3 k4 w2 k5，16c
event: n1 -> w1; n3 -> w3; n2 -> w2
device: 32c
*/
TEST_F(TestLockDetector, DependencyEq14TwoWaitsUseDifferentProducerStages) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}, {20, 21, 22, 23, 24, 25}}, {{2, {21}}, {22, {10}}, {12, {24}}});
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(11)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 16;
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {20, 21, 22, 23, 24, 25});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(20)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(21)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(22)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(23)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(24)));
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq15：multi stream wait scopes sharing a member stream cannot overlap
stream1：k1 -> n1
stream2：k2 -> [w1 -> k3] -> [k4 -> w2 -> k5]
stream3：k6 -> [k7 -> k8]
stream4：k9 -> [k10 -> k11]
stream5：k12 -> n2

scope1: w1 k3 k7 k8
scope2: k4 w2 k5 k10 k11
candidate: scope1
event: n1 -> w1; n2 -> w2
*/
TEST_F(TestLockDetector, DependencyEq15MultiStreamScopesShareOneStream) {
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22}, {30, 31, 32}, {40, 41}},
                         {{2, {11}}, {41, {15}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12, 21, 22});
  SetScopeMembers(scopes[1], {13, 14, 15, 16, 31, 32});
  EXPECT_FALSE(CanScopesOverlap(scopes));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq16：
stream1：k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> [k42 -> w0 -> k43]
stream2：k -> k -> k -> k -> n1 -> k -> k -> k -> k -> k37 -> k39 -> k -> n0 -> k -> k
stream3：k -> k -> k -> k -> k -> k27 -> k28 -> w1 -> k -> k38 -> k40 -> k -> k -> k -> k
stream4：k -> k -> k -> w2 -> k -> k -> k29 -> k30 -> k -> k -> k -> k -> k -> k -> k
stream5：k -> n2 -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k
stream6：k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k

候选scope：[k42 -> w0 -> k43]
scope1：k37 k38 k39 k40
scope2: k27 k28 w1 k29 k30
*/
TEST_F(TestLockDetector, DependencyEq16SharedStreamPreventsProducerScopeCombination) {
  CreateGraphFromStreams({{1, 42, 43, 44}, {51, 37, 39, 52}, {27, 28, 55, 38, 40}, {65, 29, 30}, {61}, {70}},
                         {{51, {55}}, {52, {43}}, {61, {65}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {42, 43, 44});
  SetScopeMembers(scopes[1], {37, 38, 39, 40});
  SetScopeMembers(scopes[2], {27, 28, 55, 29, 30});
  lockDetector->SetScopeInfos(scopes);
  const auto rootId = scopes[0].GetScopeId();
  const auto producerId = scopes[1].GetScopeId();
  const auto waitId = scopes[2].GetScopeId();
  const auto ranges = LockDetector::ScopeConcurrentRanges{
      {rootId, lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), rootId)},
      {producerId, lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), producerId)},
      {waitId, lockDetector->CalculateScopeConcurrentRanges(scopes[2].GetNodes(), waitId)}};
  EXPECT_FALSE(lockDetector->CanRunTogether(producerId, waitId, ranges));
  EXPECT_EQ(ranges.at(rootId).at(0).firstNodeId, 42U);
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq16-change1：
stream1：k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> [k42 -> w0 -> k43]
stream2：k -> k -> k -> k -> n1 -> k -> k -> k -> k -> k37 -> k39 -> k -> n0 -> k -> k
stream3：k -> k -> k -> k -> k -> k27 -> k28 -> w1 -> k -> n3 -> k -> k -> k -> k -> k
stream4：k -> k -> k -> w2 -> k -> k -> k29 -> k30 -> k -> k -> k -> k -> k -> k -> k
stream5：k -> n2 -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k
stream6：k -> k -> k -> k -> k -> k -> k -> k -> w3 -> k -> k39 -> k40 -> k -> k -> k

候选scope：[k42 -> w0 -> k43]
scope1：k37 k38 k39 k40
scope2: k27 k28 w1 k29 k30
*/
TEST_F(TestLockDetector, DependencyEq16Change1ExtraEventMovesProducerTail) {
  CreateGraphFromStreams({{1, 42, 43, 44}, {51, 37, 52}, {27, 28, 55, 38, 57}, {65, 29, 30}, {61}, {58, 39, 40}},
                         {{51, {55}}, {52, {43}}, {57, {58}}, {61, {65}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {42, 43, 44});
  SetScopeMembers(scopes[1], {37, 38, 39, 40});
  SetScopeMembers(scopes[2], {27, 28, 55, 29, 30});
  lockDetector->SetScopeInfos(scopes);
  const auto producerId = scopes[1].GetScopeId();
  const auto waitId = scopes[2].GetScopeId();
  EXPECT_FALSE(lockDetector->CanRunTogether(
      producerId, waitId,
      {{producerId, lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), producerId)},
       {waitId, lockDetector->CalculateScopeConcurrentRanges(scopes[2].GetNodes(), waitId)}}));
  const auto rootConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(rootConcurrentRange.at(0).firstNodeId, 42U);
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq16-change2：
stream1：k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> [k42 -> w0 -> k43]
stream2：k -> k -> k -> k -> n1 -> k -> k -> k -> k -> k37 -> k39 -> k -> n0 -> k -> k
stream3：k -> k -> k -> k -> k -> k27 -> k28 -> w1 -> k -> k38 -> k40 -> k -> k -> k -> k
stream4：k -> k -> k -> w2 -> k -> k -> k29 -> k30 -> k -> k -> k -> k -> k -> k -> k
stream5：k -> k -> w3 -> k14 -> n2 -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k
stream6：k -> n3 -> k15 -> k16 -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k -> k

候选scope：[k42 -> w0 -> k43]
scope1：k37 k38 k39 k40
scope2: k27 k28 w1 k29 k30
scope3: w3 k14 k15 k16
*/
TEST_F(TestLockDetector, DependencyEq16Change2ThirdWaitScope) {
  CreateGraphFromStreams(
      {{1, 42, 43, 44}, {51, 37, 39, 52}, {27, 28, 55, 38, 40}, {65, 29, 30}, {58, 14, 61}, {59, 15, 16}},
      {{51, {55}}, {52, {43}}, {59, {58}}, {61, {65}}});
  std::vector<SuperKernelScopeInfo> scopes(4);
  SetScopeMembers(scopes[0], {42, 43, 44});
  SetScopeMembers(scopes[1], {37, 38, 39, 40});
  SetScopeMembers(scopes[2], {27, 28, 55, 29, 30});
  SetScopeMembers(scopes[3], {58, 14, 15, 16});
  lockDetector->SetScopeInfos(scopes);
  const auto rootId = scopes[0].GetScopeId();
  const auto thirdId = scopes[3].GetScopeId();
  const auto ranges = LockDetector::ScopeConcurrentRanges{
      {rootId, lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), rootId)},
      {thirdId, lockDetector->CalculateScopeConcurrentRanges(scopes[3].GetNodes(), thirdId)}};
  EXPECT_TRUE(lockDetector->CanRunTogether(rootId, thirdId, ranges));
  EXPECT_EQ(ranges.at(rootId).at(0).firstNodeId, 42U);
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
}

/*
eq19：两个scope分别等待外部notify，共同占核后外部kernel无法运行
stream1：[k1 -> w1 -> k5]
stream2：[k2 -> w2 -> k3]
stream3：k4(1c) -> n1
stream4：k6(1c) -> n2

scope1: k1 w1 k5，16c
scope2: k2 w2 k3，16c
event: n1 -> w1; n2 -> w2
device: 32c
schedule: scope1和scope2可同时进入各自的wait
expected: 检查w1时计入scope1(16c)+scope2(16c)+k4(1c)=33c>32c；scope2虽等待不同的notify，仍占用同一设备资源
*/
TEST_F(TestLockDetector, DependencyEq19IndependentWaitScopesExhaustCores) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}, {20, 21}, {30, 31}}, {{21, {2}}, {31, {11}}});
  for (const auto id : {1U, 10U}) {
    graph->GetNodeById(id)->nodeInfos.kernelInfos.cubeNum = 16;
  }
  for (const auto id : {20U, 30U}) {
    graph->GetNodeById(id)->nodeInfos.kernelInfos.cubeNum = 1;
  }
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2, 3});
  SetScopeMembers(scopes[1], {10, 11, 12});
  for (auto &scope : scopes) {
    scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  }
  EXPECT_TRUE(CanScopesOverlap(scopes));
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, 2U);
  LockDetector::deviceRealCubeNum = savedCores;
}

/*
eq20：同一scope有两个外部wait，解除链多次经过同一外部scope

stream1：[k1 -> w1 -> k2 -> w2 -> k3]
stream2：[k4 -> w3 -> k5]
stream3：[k6 -> k7 -> n1]
stream4：k8(1c) -> n2
stream5：k9(1c) -> n3

scope1: k1 w1 k2 w2 k3，16c
scope2: k4 w3 k5，12c
scope3: k6 k7 n1，4c
event: n1 -> w1; n2 -> w2; n3 -> w3
device: 32c
schedule: scope1和scope2可以同时持核；scope3没有外部wait
*/
TEST_F(TestLockDetector, DependencyEq20WaitFreeProducerCanReleaseCores) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12}, {20, 21, 22}, {30, 31}, {40, 41}},
                         {{22, {2}}, {31, {4}}, {41, {11}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {1, 2, 3, 4, 5});
  SetScopeMembers(scopes[1], {10, 11, 12});
  SetScopeMembers(scopes[2], {20, 21, 22});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 16});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 12});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 16;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 12;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 4;
  graph->GetNodeById(30)->nodeInfos.kernelInfos.cubeNum = 1;
  graph->GetNodeById(40)->nodeInfos.kernelInfos.cubeNum = 1;
  EXPECT_TRUE(CanScopesOverlap(scopes));
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 32;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
  LockDetector::deviceRealCubeNum = savedCores;
}

// scope间可达的例子
/*
eq1：
stream1：[k1]-> k2 -> [k3]
scope1: k1
scope2: k3
可达：是。若最终计划保证同流任务完成顺序，则scope1释放核后，才经过k2启动scope2。
*/
TEST_F(TestLockDetector, ReachabilityEq1SameStreamScopesCannotOverlap) {
  CreateGraphFromStreams({{1, 2, 3}}, {});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1});
  SetScopeMembers(scopes[1], {3});
  EXPECT_FALSE(CanScopesOverlap(scopes));
}

/*
eq2:
stream1: [k1 -> n1] -> k2
stream2: w1 -> [k3]
scope1: k1 n1
scope2: k3
event: n1 -> w1
可达：否。n1仍在scope1内部；n1解除启动前的w1，不保证scope1已释放核。
*/
TEST_F(TestLockDetector, ReachabilityEq2InternalNotifyDoesNotSerializeScopes) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11}}, {{2, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {11});
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

/*
eq3:
stream1: [k1 -> n1] -> k2
stream2: [w1 -> k3]
scope1: k1 n1
scope2: w1 k3
event: n1 -> w1
可达：否。w1在scope2内部，scope2可以先启动并占核等待n1。
*/
TEST_F(TestLockDetector, ReachabilityEq3InternalWaitCanStartBeforeNotify) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11}}, {{2, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11});
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

/*
eq4:
stream1: [k1] -> n1 -> k2
stream2: [w1 -> k3]
scope1: k1
scope2: w1 k3
event: n1 -> w1
可达：否。n1虽在scope1释放核之后，但w1在scope2内部，不能推出scope2在scope1释放后才启动。
*/
TEST_F(TestLockDetector, ReachabilityEq4LaterNotifyDoesNotDelayWaitScopeLaunch) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11}}, {{2, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1});
  SetScopeMembers(scopes[1], {10, 11});
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

/*
eq5:
stream1: [k1] -> n1
stream2: w1 -> [k2]
scope1: k1
scope2: k2
event: n1 -> w1
可达：是。scope1释放核后才执行n1，外部w1完成后scope2才启动。
*/
TEST_F(TestLockDetector, ReachabilityEq5ExternalWaitSerializesScopes) {
  CreateGraphFromStreams({{1, 2}, {10, 11}}, {{2, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1});
  SetScopeMembers(scopes[1], {11});
  EXPECT_FALSE(CanScopesOverlap(scopes));
}

/*
eq6:
stream1: [k1 -> n1] -> k2
stream2: k3 -> w1 -> k4 -> n2
stream3: k6 -> [w2 -> k7]
scope1: k1 n1
scope2: w2 k7
event: n1 -> w1; n2 -> w2
可达：否。n1到w2的依赖链只到scope2内部的wait；scope2可先启动并占核等待w2。
*/
TEST_F(TestLockDetector, ReachabilityEq6DependencyEndsAtInternalWait) {
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22}}, {{2, {11}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {21, 22});
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

// 原流程误判场景
/*
eq7:
stream1: [k1 -> n1] -> k2 -> n3
stream2: k3 -> w1 -> k4 -> n2
stream3: k6 -> w3 -> [w2 -> k7]
终止
*/
TEST_F(TestLockDetector, ReachabilityEq7PrecedingScopeCannotOverlap) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}, {20, 21, 22, 23}}, {{2, {11}}, {4, {21}}, {13, {22}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {22, 23});
  EXPECT_FALSE(CanScopesOverlap(scopes));
}

/*
eq8:
stream1: k1 -> n1 -> k2 -> n3 -> k8
stream2: k3 -> w1 -> k4 -> n2 -> k9
stream3: k6 -> w3 -> [w2 -> k7]
终止
*/
TEST_F(TestLockDetector, ReachabilityEq8EarlierNotifyIsOutsideConcurrentRange) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22, 23}}, {{2, {11}}, {4, {21}}, {13, {22}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {22, 23});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_LT(2U, concurrentRange.at(0).firstNodeId);
}

/*
eq9：scope内部wait对应的notify在scope启动前已完成
stream1: k1 -> n1 -> n0 -> k2
stream2: w0 -> [k3 -> w1 -> k4]

scopew: k3 w1 k4
event: n0 -> w0; n1 -> w1
结论：n1先于n0完成，w0必须等n0完成，scopew必须等w0完成才启动。
因此n1位于scopew启动前已完成的左区间；检查内部w1时，无需计入n1及其前驱k1的资源。
*/
TEST_F(TestLockDetector, ReachabilityEq9InternalNotifyCompletedBeforeScopeStarts) {
  CreateGraphFromStreams({{1, 2, 3, 4}, {10, 11, 12, 13}}, {{3, {10}}, {2, {12}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 4U);
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 7;
  graph->GetNodeById(11)->nodeInfos.kernelInfos.cubeNum = 6;
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());
  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(11)));
  EXPECT_TRUE(lockDetector->IsFusible(*graph->GetNodeById(12)));
  LockDetector::deviceRealCubeNum = savedCores;
}

// 计算可达性的样例(遇到scope的情况)
/*
eq1：
stream1: k1 -> n1 -> k2 -> n3 -> k8
stream2: k3 -> w1 -> k4 -> n2 -> k9
stream3: k6 -> w3 -> [w2 -> k7]

scope1：w2 k7
scope2: n1 k2 w1 k4

scope1的可能并行区间：
stream1：[k8,k8]
stream2：[n2,k9]
stream3：[w2,k7]
scope2的可能并行区间：
stream1：[n1,k2]
stream2：[w1,k4]
stream3：[k6,k6]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq1CompletedProducerScope) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22, 23}}, {{2, {11}}, {4, {21}}, {13, {22}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {2, 3, 11, 12});
  lockDetector->SetScopeInfos(scopes);
  const auto rootConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(rootConcurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(rootConcurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(rootConcurrentRange.at(1).firstNodeId, 13U);
  EXPECT_EQ(rootConcurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(rootConcurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(rootConcurrentRange.at(2).lastNodeId, 23U);
  const auto producerConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), scopes[1].GetScopeId());
  EXPECT_EQ(producerConcurrentRange.at(0).firstNodeId, 2U);
  EXPECT_EQ(producerConcurrentRange.at(0).lastNodeId, 3U);
  EXPECT_EQ(producerConcurrentRange.at(1).firstNodeId, 11U);
  EXPECT_EQ(producerConcurrentRange.at(1).lastNodeId, 12U);
  EXPECT_EQ(producerConcurrentRange.at(2).firstNodeId, 20U);
  EXPECT_EQ(producerConcurrentRange.at(2).lastNodeId, 20U);
  EXPECT_FALSE(CanScopesOverlap(scopes));
}

/*

eq2-1：
stream1: k1 -> n1 -> k2 -> n3 -> k8
stream2: k3 -> w1 -> k4 -> n2 -> k9
stream3: k6 -> w3 -> [w2 -> k7]

scope1：w2 k7
scope2: n1 k2 w1 k4 n3

scope1的可能并行区间：
stream1：[k1,k8]
stream2：[k3,k9]
stream3：[w2,k7]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq2WholeProducerScopeIncluded) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22}}, {{2, {11}}, {4, {20}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {21, 22});
  SetScopeMembers(scopes[1], {2, 3, 4, 11, 12});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 10U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 21U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 22U);
  EXPECT_TRUE(CanScopesOverlap(scopes));
}

/*

eq2-2：不采用：
stream1: k1 -> n1 -> k2 -> n3 -> k8
stream2: k3 -> w1 -> k4 -> n2 -> k9
stream3: k6 -> w3 -> [w2 -> k7]

scope1：w2 k7
scope2: n1 k2 w1 k4 n3

scope1的可能并行区间：
stream1：[k8,k8]
stream2：[n2,k9]
stream3：[w2,k7]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq2RejectedNarrowAlternative) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22}}, {{2, {11}}, {4, {20}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {21, 22});
  SetScopeMembers(scopes[1], {2, 3, 4, 11, 12});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_NE(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_NE(concurrentRange.at(1).firstNodeId, 13U);
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 10U);
}

/*

eq3：
stream1: k1 -> n1 -> k2 -> n3 -> k8
stream2: k3 -> w1 -> k4 -> n2 -> k9
stream3: k6 -> w3 -> [w2 -> k7]

scope1：w2 k7
scope2: k3 k6 w1 w3

scope1的可能并行区间：
stream1：[k8,k8]
stream2：[k4,k9]
stream3：[w2,k7]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq3PrecedingMultiStreamScope) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22, 23}}, {{2, {11}}, {4, {21}}, {13, {22}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
  EXPECT_FALSE(CanScopesOverlap(scopes));
}

/*

eq4：
stream1: k1 -> n1 -> k2 -> n3 -> k8 -> k11
stream2: k3 -> w1 -> k4 -> n2 -> k9 -> w4 -> k10
stream3: k6 -> w3 -> [w2 -> k7] -> n4

scope1：w2 k7
scope2: k3 k6 w1 w3

scope1的可能并行区间：
stream1：[k8,k12]
stream2：[k4,k9]
stream3：[w2,k7]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq4LaterWaitBlocksTail) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22, 23, 24}},
                         {{2, {11}}, {4, {21}}, {13, {22}}, {24, {15}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 6U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
}

/*
eq5：
stream1: k1 -> n1 -> k2 -> n3 -> k8 -> k11 -> k12
stream2: k3 -> w1 -> k4 -> n2 -> k9 -> w4 -> k10
stream3: k6 -> w3 -> [w2 -> k7] -> n4

scope1：w2 k7
scope2: k3 k6 w1 w3
scope3: k11 w4 k10

scope1的可能并行区间：
stream1：[k8,k11]
stream2：[k4,k10]
stream3：[w2,k7]
*/
TEST_F(TestLockDetector, ConcurrentRangeEq5LaterWaitScopeExtendsBoundary) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6, 7}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22, 23, 24}},
                         {{2, {11}}, {4, {21}}, {13, {22}}, {24, {15}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  SetScopeMembers(scopes[2], {6, 15, 16});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 7U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 16U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
}

/*
eq6：
stream1: k1 -> n1 -> k2 -> n3 -> k8 -> k11 -> w5 -> k12
stream2: k3 -> w1 -> k4 -> n2 -> k9 -> w4 -> k10 -> n5
stream3: k6 -> w3 -> [w2 -> k7] -> n4

scope1：w2 k7
scope2: k3 k6 w1 w3
scope3: n5 k11

scope1的可能并行区间：
stream1：[k8,k8]
stream2：[k4,k9]
stream3：[w2,k7]

*/
TEST_F(TestLockDetector, ConcurrentRangeEq6LaterScopeCannotStartBeforeRootFinishes) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6, 7, 8}, {10, 11, 12, 13, 14, 15, 16, 17}, {20, 21, 22, 23, 24}},
                         {{2, {11}}, {4, {21}}, {13, {22}}, {24, {15}}, {17, {7}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  SetScopeMembers(scopes[2], {6, 17});
  lockDetector->SetScopeInfos(scopes);
  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
}

TEST_F(TestLockDetector, StartedWaitScopeBlocksItsNotifyDependency) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k10 -> k11 -> n12
  stream2: k20 -> n21
  event21: n21 -> w2
  event12: n12 -> w2
  scope1: stream1[k10,k11,n12]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}, {20, 21}}, {{21, {11}}, {12, {2}}});
  auto *root = graph->GetNodeById(1);
  auto *wait = graph->GetNodeById(2);
  root->nodeInfos.kernelInfos.cubeNum = 6;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 3;

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11, 12});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  EXPECT_TRUE(lockDetector->IsFusible(*root));
  EXPECT_FALSE(lockDetector->IsFusible(*wait));
  LockDetector::deviceRealCubeNum = savedCores;
}

TEST_F(TestLockDetector, StartedWaitScopeExcludesLaterKernelOnItsStream) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k10 -> k11 -> k12 -> k13 -> k14
  stream2: k20
  event20: n20 -> w11
  event14: n14 -> w2
  scope1: stream1[k10,k11,k12]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11, 12, 13, 14}, {20}}, {{20, {11}}, {14, {2}}});
  auto *root = graph->GetNodeById(1);
  auto *wait = graph->GetNodeById(2);
  root->nodeInfos.kernelInfos.cubeNum = 4;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(13)->nodeInfos.kernelInfos.cubeNum = 5;

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11, 12});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  EXPECT_TRUE(lockDetector->IsFusible(*root));
  EXPECT_TRUE(lockDetector->IsFusible(*wait));
  LockDetector::deviceRealCubeNum = savedCores;
}

TEST_F(TestLockDetector, IndependentWaitScopesCanStarveExternalKernelTogether) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k10 -> w11
  stream2: k20 -> w21
  stream3: k30
  event21: n21 -> w2
  event30: n30 -> w11
  scope1: stream0[k1,w2]
  scope2: stream1[k10,w11]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}, {30}}, {{21, {2}}, {30, {11}}});
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 6;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 3;

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, 2U);
  LockDetector::deviceRealCubeNum = savedCores;
}

TEST_F(TestLockDetector, CollectWaitScopeCandidatesSkipsInternalWaitOnlyScope) {
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21, 22}, {30}}, {{1, {11}}, {20, {22}}, {30, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2, 10, 11});
  SetScopeMembers(scopes[1], {20, 21, 22});

  const auto candidates = scopeDetector->CollectWaitScopeCandidates(scopes);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(candidates[0], &scopes[1]);
}

TEST_F(TestLockDetector, NotifyCompletedBeforeScopeLaunchStopsResourceSearch) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> n3
  stream1: k10 -> w11 -> w12 -> k13
  event3: n3 -> w10
  event2: n2 -> w12
  scope1: stream1[w11,w12,k13]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13}}, {{3, {10}}, {2, {12}}});
  auto *root = graph->GetNodeById(11);
  auto *wait = graph->GetNodeById(12);
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 7;
  root->nodeInfos.kernelInfos.cubeNum = 6;

  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  lockDetector->SetScopeInfos(scopes);
  lockDetector->SetCurrentScopeId(scopes[0].GetScopeId());
  lockDetector->SetScopeCoreInfo(scopes[0].GetScopeCoreInfo());

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  EXPECT_TRUE(lockDetector->IsFusible(*root));
  EXPECT_TRUE(lockDetector->IsFusible(*wait));
  LockDetector::deviceRealCubeNum = savedCores;
}

TEST_F(TestLockDetector, ConcurrentRangeExcludesScopeCompletedBeforeLaunch) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2 -> k3 -> n4 -> k5
  stream1: k10 -> w11 -> k12 -> n13 -> k14
  stream2: k20 -> w21 -> k22
  event2: n2 -> w11
  event4: n4 -> w21
  event13: n13 -> w22
  scope1: stream0[n2,k3], stream1[w11,k12]
  scope2: stream2[w21,k22]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22}}, {{2, {11}}, {4, {20}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {21, 22});
  SetScopeMembers(scopes[1], {2, 3, 11, 12});
  lockDetector->SetScopeInfos(scopes);

  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  ASSERT_EQ(concurrentRange.size(), 3U);
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 13U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 21U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 22U);
}

TEST_F(TestLockDetector, ConcurrentRangeIncludesWholeScopeReachedThroughInternalEvent) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2 -> k3 -> n4 -> k5
  stream1: k10 -> w11 -> k12 -> n13 -> k14
  stream2: k20 -> w21 -> k22
  event2: n2 -> w11
  event4: n4 -> w21
  event13: n13 -> w22
  scope1: stream0[n2,k3,n4], stream1[w11,k12]
  scope2: stream2[w21,k22]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22}}, {{2, {11}}, {4, {20}}, {13, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {21, 22});
  SetScopeMembers(scopes[1], {2, 3, 4, 11, 12});
  lockDetector->SetScopeInfos(scopes);

  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  ASSERT_EQ(concurrentRange.size(), 3U);
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 1U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 10U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 21U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 22U);
}

TEST_F(TestLockDetector, ConcurrentRangeMovesPastCompletedMultiStreamScope) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2 -> k3 -> n4 -> k5
  stream1: k10 -> w11 -> k12 -> n13 -> k14
  stream2: k20 -> w21 -> k22 -> k23
  event2: n2 -> w11
  event4: n4 -> w21
  event13: n13 -> w22
  scope1: stream1[k10,w11], stream2[k20,w21]
  scope2: stream2[k22,k23]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11, 12, 13, 14}, {20, 21, 22, 23}}, {{2, {11}}, {4, {21}}, {13, {22}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  lockDetector->SetScopeInfos(scopes);

  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
}

TEST_F(TestLockDetector, ConcurrentRangeIncludesLaterWaitScopeAndItsSuccessor) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2 -> k3 -> n4 -> k5 -> k6 -> k7
  stream1: k10 -> w11 -> k12 -> n13 -> k14 -> w15 -> k16
  stream2: k20 -> w21 -> k22 -> k23 -> n24
  event2: n2 -> w11
  event4: n4 -> w21
  event13: n13 -> w22
  event24: n24 -> w15
  scope1: stream1[k10,w11], stream2[k20,w21]
  scope2: stream1[k6,w15,k16]
  scope3: stream2[k22,k23]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6, 7}, {10, 11, 12, 13, 14, 15, 16}, {20, 21, 22, 23, 24}},
                         {{2, {11}}, {4, {21}}, {13, {22}}, {24, {15}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  SetScopeMembers(scopes[2], {6, 15, 16});
  lockDetector->SetScopeInfos(scopes);

  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 7U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 16U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
}

TEST_F(TestLockDetector, ConcurrentRangeExcludesScopeStartingAfterRootFinishes) {
  // ======================= graph =======================
  /*
  stream0: k1 -> n2 -> k3 -> n4 -> k5 -> k6 -> k7 -> w8
  stream1: k10 -> w11 -> k12 -> n13 -> k14 -> w15 -> k16 -> n17
  stream2: k20 -> w21 -> k22 -> k23 -> n24
  event2: n2 -> w11
  event4: n4 -> w21
  event13: n13 -> w22
  event24: n24 -> w15
  event17: n17 -> w7
  scope1: stream1[k10,w11], stream2[k20,w21]
  scope2: stream0[k6], stream1[n17]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6, 7, 8}, {10, 11, 12, 13, 14, 15, 16, 17}, {20, 21, 22, 23, 24}},
                         {{2, {11}}, {4, {21}}, {13, {22}}, {24, {15}}, {17, {7}}});
  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {22, 23});
  SetScopeMembers(scopes[1], {10, 20, 11, 21});
  SetScopeMembers(scopes[2], {6, 17});
  lockDetector->SetScopeInfos(scopes);

  const auto concurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  EXPECT_EQ(concurrentRange.at(0).firstNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(0).lastNodeId, 5U);
  EXPECT_EQ(concurrentRange.at(1).firstNodeId, 12U);
  EXPECT_EQ(concurrentRange.at(1).lastNodeId, 14U);
  EXPECT_EQ(concurrentRange.at(2).firstNodeId, 22U);
  EXPECT_EQ(concurrentRange.at(2).lastNodeId, 23U);
}

TEST_F(TestLockDetector, ParallelResourcePassSplitsAtRiskWait) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k10 -> w11
  stream2: k20 -> n21
  stream3: n30
  event21: n21 -> w2
  event30: n30 -> w11
  scope1: stream0[k1,w2]
  scope2: stream1[k10,w11]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}, {30}}, {{21, {2}}, {30, {11}}});
  auto *root = graph->GetNodeById(1);
  auto *other = graph->GetNodeById(10);
  auto *dependency = graph->GetNodeById(20);
  root->nodeInfos.kernelInfos.numBlocks = 20;
  root->nodeInfos.kernelInfos.cubeNum = 20;
  other->nodeInfos.kernelInfos.numBlocks = 10;
  other->nodeInfos.kernelInfos.cubeNum = 10;
  dependency->nodeInfos.kernelInfos.numBlocks = 4;
  dependency->nodeInfos.kernelInfos.cubeNum = 4;

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11});
  auto *riskWait = graph->GetNodeById(2);
  ASSERT_TRUE(riskWait->IsFusible());

  ParallelScopeDeadlockRefinePass pass(*graph, *opts);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 2U);
  ASSERT_EQ(scopes[0].GetNodes().size(), 1U);
  EXPECT_EQ(scopes[0].GetNodes()[0]->GetNodeId(), 1U);
  EXPECT_EQ(scopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::DEADLOCK_DETECTED);
  EXPECT_EQ(scopes[1].GetNodes().size(), 2U);
  EXPECT_EQ(scopes[1].GetNodes()[0]->GetScopeId(), scopes[1].GetScopeId());
  EXPECT_EQ(riskWait->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_FALSE(riskWait->IsFusible());
  EXPECT_EQ(riskWait->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
}

TEST_F(TestLockDetector, ParallelResourceSplitKeepsLaterWaitNotifiedByPrefix) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {8}, {30}}, {{8, {4}}, {30, {2}}});
  SuperKernelScopeInfo scope;
  SetScopeMembers(scope, {1, 8, 2, 3, 4, 5});
  auto *riskWait = graph->GetNodeById(2);
  auto *laterWait = graph->GetNodeById(4);

  ParallelScopeDeadlockRefinePass pass(*graph, *opts);
  std::vector<SuperKernelScopeInfo> splitScopes;
  pass.SplitScopeAtRiskWait(scope, *riskWait, splitScopes);

  ASSERT_EQ(splitScopes.size(), 2U);
  EXPECT_EQ(splitScopes[0].GetNodes().back()->GetNodeId(), 8U);
  ASSERT_EQ(splitScopes[1].GetNodes().size(), 3U);
  EXPECT_EQ(splitScopes[1].GetNodes()[1], laterWait);
  EXPECT_FALSE(riskWait->IsFusible());
  EXPECT_TRUE(laterWait->IsFusible());
}

TEST_F(TestLockDetector, ParallelResourceSplitCutsLaterWaitNotifiedAfterRiskWait) {
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {8}, {30}}, {{8, {4}}, {30, {2}}});
  SuperKernelScopeInfo scope;
  SetScopeMembers(scope, {1, 2, 3, 8, 4, 5});
  auto *riskWait = graph->GetNodeById(2);
  auto *laterWait = graph->GetNodeById(4);

  ParallelScopeDeadlockRefinePass pass(*graph, *opts);
  std::vector<SuperKernelScopeInfo> splitScopes;
  pass.SplitScopeAtRiskWait(scope, *riskWait, splitScopes);

  ASSERT_EQ(splitScopes.size(), 3U);
  EXPECT_EQ(splitScopes[1].GetNodes().back()->GetNodeId(), 8U);
  EXPECT_EQ(laterWait->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_FALSE(laterWait->IsFusible());
  EXPECT_EQ(laterWait->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
}

TEST_F(TestLockDetector, ParallelResourcePassSplitsLaterWaitsAfterRiskWait) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> w2 -> k3 -> w4 -> k5]
  stream1: [k10 -> w11]
  stream2: k20 -> n21 -> k30 -> n31
  stream3: n40
  event21: n21 -> w2
  event31: n31 -> w4
  event40: n40 -> w11
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11}, {20, 21, 30, 31}, {40}}, {{21, {2}}, {31, {4}}, {40, {11}}});
  for (const auto nodeId : {1U, 3U, 5U}) {
    auto *node = graph->GetNodeById(nodeId);
    node->nodeInfos.kernelInfos.numBlocks = 20;
    node->nodeInfos.kernelInfos.cubeNum = 20;
  }
  graph->GetNodeById(10)->nodeInfos.kernelInfos.numBlocks = 10;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 10;
  for (const auto nodeId : {20U, 30U}) {
    auto *node = graph->GetNodeById(nodeId);
    node->nodeInfos.kernelInfos.numBlocks = 4;
    node->nodeInfos.kernelInfos.cubeNum = 4;
  }
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2, 3, 4, 5});
  SetScopeMembers(scopes[1], {10, 11});

  ParallelScopeDeadlockRefinePass pass(*graph, *opts);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 4U);
  EXPECT_EQ(scopes[0].GetNodes().front()->GetNodeId(), 1U);
  EXPECT_EQ(scopes[1].GetNodes().front()->GetNodeId(), 3U);
  EXPECT_EQ(scopes[2].GetNodes().front()->GetNodeId(), 5U);
  EXPECT_EQ(scopes[3].GetNodes().front()->GetNodeId(), 10U);
  for (const auto waitId : {2U, 4U}) {
    const auto *wait = graph->GetNodeById(waitId);
    EXPECT_EQ(wait->GetScopeId(), INVALID_SCOPE_ID);
    EXPECT_FALSE(wait->IsFusible());
    EXPECT_EQ(wait->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  }
}

TEST_F(TestLockDetector, ParallelResourcePassSplitsRiskWaitAndLaterWaits) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> w2 -> k3 -> w4 -> k5 -> w6 -> k7]
  stream1: [k10 -> w11]
  stream2: n21
  stream3: k20 -> n31
  stream4: n40
  stream5: n50
  event21: n21 -> w2
  event31: n31 -> w4
  event50: n50 -> w6
  event40: n40 -> w11
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5, 6, 7}, {10, 11}, {21}, {20, 31}, {40}, {50}},
                         {{21, {2}}, {31, {4}}, {50, {6}}, {40, {11}}});
  for (const auto nodeId : {1U, 3U, 5U, 7U}) {
    auto *node = graph->GetNodeById(nodeId);
    node->nodeInfos.kernelInfos.numBlocks = 20;
    node->nodeInfos.kernelInfos.cubeNum = 20;
  }
  graph->GetNodeById(10)->nodeInfos.kernelInfos.numBlocks = 10;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 10;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.numBlocks = 4;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 4;
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {1, 2, 3, 4, 5, 6, 7});
  SetScopeMembers(scopes[1], {10, 11});

  ParallelScopeDeadlockRefinePass pass(*graph, *opts);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 4U);
  EXPECT_EQ(scopes[0].GetNodes().front()->GetNodeId(), 1U);
  EXPECT_EQ(scopes[0].GetNodes().back()->GetNodeId(), 3U);
  ASSERT_EQ(scopes[1].GetNodes().size(), 1U);
  EXPECT_EQ(scopes[1].GetNodes()[0]->GetNodeId(), 5U);
  ASSERT_EQ(scopes[2].GetNodes().size(), 1U);
  EXPECT_EQ(scopes[2].GetNodes()[0]->GetNodeId(), 7U);
  EXPECT_EQ(scopes[3].GetNodes().front()->GetNodeId(), 10U);
  EXPECT_TRUE(graph->GetNodeById(2)->IsFusible());
  for (const auto waitId : {4U, 6U}) {
    const auto *wait = graph->GetNodeById(waitId);
    EXPECT_EQ(wait->GetScopeId(), INVALID_SCOPE_ID);
    EXPECT_FALSE(wait->IsFusible());
    EXPECT_EQ(wait->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  }
}

TEST_F(TestLockDetector, ThreeWaitScopesOnlyFailWhenAllAreResident) {
  // ======================= graph =======================
  /*
  stream0: k1 -> w2
  stream1: k10 -> w11
  stream2: k20 -> w21
  stream3: k30 -> n31
  stream4: n40
  stream5: n50
  event31: n31 -> w2
  event40: n40 -> w11
  event50: n50 -> w21
  scope1: stream0[k1,w2]
  scope2: stream1[k10,w11]
  scope3: stream2[k20,w21]
  */
  CreateGraphFromStreams({{1, 2}, {10, 11}, {20, 21}, {30, 31}, {40}, {50}}, {{31, {2}}, {40, {11}}, {50, {21}}});
  graph->GetNodeById(1)->nodeInfos.kernelInfos.cubeNum = 4;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(30)->nodeInfos.kernelInfos.cubeNum = 1;

  std::vector<SuperKernelScopeInfo> scopes(3);
  SetScopeMembers(scopes[0], {1, 2});
  SetScopeMembers(scopes[1], {10, 11});
  SetScopeMembers(scopes[2], {20, 21});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});
  scopes[2].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, 2U);
  LockDetector::deviceRealCubeNum = savedCores;
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftEventExcludesUnscopedNotify) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3]
  stream1: k10 -> n11 -> k12
  event11: n11 -> w2
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}}, {{11, {2}}});
  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {12, 12}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftEventUsesScopeHeadPredecessor) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3]
  stream1: k9 -> k10 -> [n12 -> k13] -> k14
  event12: n12 -> w2
  scope1: stream1[n12,k13]
  */
  CreateGraphFromStreams({{1, 2, 3}, {9, 10, 11, 12, 13, 14}}, {{12, {2}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {10, 14}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryRightEventUsesScopeTailSuccessor) {
  // ======================= graph =======================
  /*
  stream0: [k1] -> k2 -> k3
  stream1: k9 -> k10 -> [w11 -> k12 -> w13] -> k14 -> k15
  event2: n2 -> w12
  scope1: stream1[w11,k12,w13]
  */
  CreateGraphFromStreams({{1, 2, 3}, {9, 10, 11, 12, 13, 14, 15}}, {{2, {12}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {9, 14}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftEventScopeAtStreamHeadKeepsRange) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3]
  stream1: [k10 -> n11] -> k12
  event11: n11 -> w2
  scope1: stream1[k10,n11]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}}, {{11, {2}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {10, 11});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {10, 12}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryRightEventScopeAtStreamTailKeepsRange) {
  // ======================= graph =======================
  /*
  stream0: [k1] -> k2 -> k3
  stream1: k10 -> [w11 -> k12]
  event2: n2 -> w11
  scope1: stream1[w11,k12]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}}, {{2, {11}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {10, 12}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftScansScopeTailAfterEventEntry) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3]
  stream1: k10 -> [n11 -> k12 -> w13] -> k14
  stream2: k30 -> [k31] -> k32
  stream3: k40 -> n41 -> k42
  event11: n11 -> w2; event41: n41 -> w13
  scope1: stream1[n11,k12,w13], stream2[k31]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13, 14}, {30, 31, 32}, {40, 41, 42}}, {{11, {2}}, {41, {13}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13, 31});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {10, 14}}, {2, {30, 32}}, {3, {42, 42}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftScopeOnlyTightensOtherStreams) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3]
  stream1: k9 -> k10 -> [k11 -> k12 -> n13] -> k14
  stream2: k30 -> [k31] -> k32
  event13: n13 -> w2
  scope1: stream1[k11,k12,n13]
  scope2: stream1[k10], stream2[k31]
  */
  CreateGraphFromStreams({{1, 2, 3}, {9, 10, 11, 12, 13, 14}, {30, 31, 32}}, {{13, {2}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {11, 12, 13});
  SetScopeMembers(scopes[1], {10, 31});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {10, 14}}, {2, {32, 32}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryRightScopeOnlyTightensOtherStreams) {
  // ======================= graph =======================
  /*
  stream0: [k1] -> k2 -> k3
  stream1: k9 -> [w10 -> k11] -> [k12] -> k13
  stream2: k30 -> [k31] -> k32
  event2: n2 -> w10
  scope1: stream1[w10,k11]
  scope2: stream1[k12], stream2[k31]
  */
  CreateGraphFromStreams({{1, 2, 3}, {9, 10, 11, 12, 13}, {30, 31, 32}}, {{2, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {10, 11});
  SetScopeMembers(scopes[1], {12, 31});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {9, 12}}, {2, {30, 30}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryRightScansScopeHeadBeforeEventEntry) {
  // ======================= graph =======================
  /*
  stream0: [k1] -> k2 -> k3
  stream1: k10 -> [n11 -> k12 -> w13] -> k14
  stream2: k30 -> [k31] -> k32
  stream3: k40 -> n41 -> k42
  event2: n2 -> w13; event11: n11 -> w41
  scope1: stream1[w11,k12,n13], stream2[k31]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13, 14}, {30, 31, 32}, {40, 41, 42}}, {{2, {13}}, {11, {41}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {11, 12, 13, 31});
  lockDetector->SetScopeInfos(scopes);
  ExpectConcurrencyRange({graph->GetNodeById(1)}, {{0, {1, 1}}, {1, {10, 14}}, {2, {30, 32}}, {3, {40, 40}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryLeftScopePropagatesWaitToOtherStream) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> k2] -> k3
  stream1: [k10 -> k11] -> k12
  stream2: k20 -> n21 -> k22
  event21: n21 -> w10
  scope1: stream0[k1,k2], stream1[k10,k11]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}, {20, 21, 22}}, {{21, {10}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {1, 2, 10, 11});
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange({graph->GetNodeById(3)}, {{0, {3, 3}}, {1, {12, 12}}, {2, {22, 22}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryRightScopePropagatesNotifyToOtherStream) {
  // ======================= graph =======================
  /*
  stream0: k1 -> [k2 -> k3]
  stream1: k10 -> [n11] -> k12
  stream2: k20 -> w21 -> k22
  event11: n11 -> w21
  scope1: stream0[k2,k3], stream1[n11]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12}, {20, 21, 22}}, {{11, {21}}});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {2, 3, 10, 11});
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange({graph->GetNodeById(1)},
                         {{0, {1, 1}}, {1, {INVALID_TASK_ID, INVALID_TASK_ID}}, {2, {20, 20}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryInternalWaitDoesNotTightenOtherStream) {
  // ======================= graph =======================
  /*
  stream0: [k1 -> w2]
  stream1: k10 -> n11 -> k12
  event11: n11 -> w2
  */
  CreateGraphFromStreams({{1, 2}, {10, 11, 12}}, {{11, {2}}});
  ExpectConcurrencyRange({graph->GetNodeById(1), graph->GetNodeById(2)}, {{0, {1, 2}}, {1, {10, 12}}});
}

TEST_F(TestLockDetector, ConcurrentBoundaryKeepsCandidateStreamRange) {
  // ======================= graph =======================
  /*
  stream0: k1 -> k2 -> [k3 -> k4] -> k5
  stream1: k10 -> k11
  scope1: stream0[k3,k4]
  */
  CreateGraphFromStreams({{1, 2, 3, 4, 5}, {10, 11}}, {});
  std::vector<SuperKernelScopeInfo> scopes(1);
  SetScopeMembers(scopes[0], {3, 4});
  lockDetector->SetScopeInfos(scopes);

  ExpectConcurrencyRange(scopes[0].GetNodes(), {{0, {3, 4}}, {1, {10, 11}}});
}

TEST_F(TestLockDetector, SerializedWaitScopesDoNotFormParallelCombination) {
  // ======================= graph =======================
  /*
  stream0: w1 -> k2 -> w3
  stream1: k10 -> w11 -> k12 -> n13
  stream2: k20 -> n21
  stream3: n30
  event13: n13 -> w1
  event30: n30 -> w11
  event21: n21 -> w3
  scope1: stream0[w2,w3]
  scope2: stream1[k10,k11,k12]
  */
  CreateGraphFromStreams({{1, 2, 3}, {10, 11, 12, 13}, {20, 21}, {30}}, {{13, {1}}, {30, {11}}, {21, {3}}});
  graph->GetNodeById(2)->nodeInfos.kernelInfos.cubeNum = 6;
  graph->GetNodeById(10)->nodeInfos.kernelInfos.cubeNum = 3;
  graph->GetNodeById(20)->nodeInfos.kernelInfos.cubeNum = 3;

  std::vector<SuperKernelScopeInfo> scopes(2);
  SetScopeMembers(scopes[0], {2, 3});
  SetScopeMembers(scopes[1], {10, 11, 12});
  scopes[0].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 6});
  scopes[1].SetScopeCoreInfo({SkKernelType::AIC_ONLY, 3});

  const auto savedCores = LockDetector::deviceRealCubeNum;
  LockDetector::deviceRealCubeNum = 10;
  lockDetector->SetScopeInfos(scopes);
  const auto aConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[0].GetNodes(), scopes[0].GetScopeId());
  const auto bConcurrentRange =
      lockDetector->CalculateScopeConcurrentRanges(scopes[1].GetNodes(), scopes[1].GetScopeId());
  EXPECT_EQ(aConcurrentRange.at(1).firstNodeId, INVALID_TASK_ID);
  EXPECT_EQ(aConcurrentRange.at(1).lastNodeId, INVALID_TASK_ID);
  EXPECT_EQ(bConcurrentRange.at(0).firstNodeId, INVALID_TASK_ID);
  EXPECT_FALSE(lockDetector->CanRunTogether(
      scopes[0].GetScopeId(), scopes[1].GetScopeId(),
      {{scopes[0].GetScopeId(), aConcurrentRange}, {scopes[1].GetScopeId(), bConcurrentRange}}));
  uint64_t riskWaitId = INVALID_TASK_ID;
  scopeDetector->CheckParallelScopeResources(scopes, riskWaitId);
  EXPECT_EQ(riskWaitId, INVALID_TASK_ID);
  LockDetector::deviceRealCubeNum = savedCores;
}
