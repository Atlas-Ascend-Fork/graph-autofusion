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
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <vector>
#include <algorithm>

#define private public
#define protected public
#include "super_kernel.h"
#include "sk_graph.h"
#include "sk_scope_split.h"
#include "sk_scope_split_result_reporter.h"
#include "sk_node.h"
#include "sk_lock_detector.h"
#include "sk_options_manager.h"

/**
 * @brief Test fixture class for SuperKernelScopeSplitter unit tests
 */
class SuperKernelScopeSplitterTest : public testing::Test {
 protected:
  void SetUp() override {
    // Clear any lingering mock state from previous tests
    GlobalMockObject::verify();
    opts = std::make_unique<SuperKernelOptionsManager>();
    graph = std::make_unique<SuperKernelGraph>();
    LockDetector::GetDeviceCores();
  }

  void TearDown() override {
    graph.reset();
    opts.reset();
    // Clear mock state
    GlobalMockObject::verify();
  }

  // Helper function to create a kernel node
  SuperKernelBaseNode *CreateKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    // Mark as fusible for testing
    node->isFusible = true;
    // Set kernel parameters to pass LockDetector check
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->nodeInfos.kernelInfos.vecNum = 0;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to set preNodeId for a node
  void SetPreNodeId(uint64_t nodeId, uint64_t preNodeId) {
    auto *node = graph->GetNodeById(nodeId);
    if (node != nullptr) {
      node->SetPreNodeId(preNodeId);
    }
  }

  // Helper function to create a wait node
  SuperKernelBaseNode *CreateWaitNode(uint64_t nodeId, uint32_t streamIdx, uint64_t notifyNodeId,
                                      uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.syncInfos.correspondingNotifyNodeId = notifyNodeId;
    // eventId is not used in LockDetector, but needed for eventToNodes mapping
    // We'll set it based on the corresponding notify node's eventId
    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_WAIT;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a notify node
  SuperKernelBaseNode *CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx, uint64_t eventId,
                                        uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_NOTIFY;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a kernel node with high core requirement
  SuperKernelBaseNode *CreateLargeKernelNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID,
                                             uint32_t cubeNum = 1000, uint32_t vecNum = 0) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = true;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = cubeNum;
    node->nodeInfos.kernelInfos.vecNum = vecNum;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create an unfusible kernel node
  SuperKernelBaseNode *CreateUnfusibleKernelNode(uint64_t nodeId, uint32_t streamIdx,
                                                 uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->isFusible = false;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 0;
    node->nodeInfos.kernelInfos.vecNum = 0;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a reset node
  SuperKernelBaseNode *CreateResetNode(uint64_t nodeId, uint32_t streamIdx, uint64_t eventId,
                                       uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->nodeInfos.syncInfos.eventId = eventId;
    node->isFusible = true;
    // Manually set nodeType since InitNode() is not called
    node->nodeType = SkNodeType::NODE_RESET;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a kernel node with custom core counts
  SuperKernelBaseNode *CreateKernelNodeWithCores(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId,
                                                 uint32_t numBlocks, SkKernelType kernelType) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
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

  // Helper function to create a scope begin node (fusible)
  SuperKernelBaseNode *CreateScopeBeginNode(uint64_t nodeId, uint32_t streamIdx, const std::string &scopeName,
                                            uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = true;
    node->SetIsScopeNode(true);
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->nodeInfos.kernelInfos.vecNum = 0;
    // Set scope name through reflection (need to access private member)
    SuperKernelKernelNode *kernelNode = static_cast<SuperKernelKernelNode *>(node.get());
    kernelNode->scopeName = scopeName;
    kernelNode->isScopeBegin = true;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a scope end node (fusible)
  SuperKernelBaseNode *CreateScopeEndNode(uint64_t nodeId, uint32_t streamIdx, const std::string &scopeName,
                                          uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = true;
    node->SetIsScopeNode(true);
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->nodeInfos.kernelInfos.vecNum = 0;
    // Set scope name through reflection
    SuperKernelKernelNode *kernelNode = static_cast<SuperKernelKernelNode *>(node.get());
    kernelNode->scopeName = scopeName;
    kernelNode->isScopeBegin = false;
    kernelNode->isScopeEnd = true;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create an unfusible scope begin node
  SuperKernelBaseNode *CreateUnfusibleScopeBeginNode(uint64_t nodeId, uint32_t streamIdx,
                                                     uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = false;
    node->SetIsScopeNode(true);
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->nodeInfos.kernelInfos.vecNum = 0;
    SuperKernelKernelNode *kernelNode = static_cast<SuperKernelKernelNode *>(node.get());
    kernelNode->scopeName = DEFAULT_SK_SCOPE_NAME;
    kernelNode->isScopeBegin = true;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create an unfusible scope end node
  SuperKernelBaseNode *CreateUnfusibleScopeEndNode(uint64_t nodeId, uint32_t streamIdx,
                                                   uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = false;
    node->SetIsScopeNode(true);
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = 1;
    node->nodeInfos.kernelInfos.vecNum = 0;
    SuperKernelKernelNode *kernelNode = static_cast<SuperKernelKernelNode *>(node.get());
    kernelNode->scopeName = DEFAULT_SK_SCOPE_NAME;
    kernelNode->isScopeBegin = false;
    kernelNode->isScopeEnd = true;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a default node
  SuperKernelBaseNode *CreateDefaultNode(uint64_t nodeId, uint32_t streamIdx, uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelDefaultNode>(nullptr, ACL_MODEL_RI_TASK_DEFAULT, 0, streamIdx,
                                                         INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = false;  // Default is unfusible, will be set by InitDefaultNodeFusibility
    node->nodeType = SkNodeType::NODE_DEFAULT;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to setup streams in graph
  // This function also sets up preNodeId, nextNodeId, and nodeIdxInStream for each node
  void SetupStreams(const std::vector<std::vector<uint64_t>> &streamNodes) {
    graph->streams.clear();
    graph->headNodes.clear();

    for (size_t streamIdx = 0; streamIdx < streamNodes.size(); ++streamIdx) {
      const auto &nodes = streamNodes[streamIdx];
      graph->streams.emplace_back();
      if (!nodes.empty()) {
        graph->headNodes.push_back(nodes[0]);
        // Set up nodeIdxInStream, preNodeId and nextNodeId for each node
        for (size_t i = 0; i < nodes.size(); ++i) {
          auto *currentNode = graph->GetNodeById(nodes[i]);
          if (currentNode != nullptr) {
            currentNode->nodeIdxInStream = i;
            currentNode->streamIdxInGraph = static_cast<uint32_t>(streamIdx);
            if (i > 0) {
              currentNode->SetPreNodeId(nodes[i - 1]);
            }
            if (i < nodes.size() - 1) {
              currentNode->SetNextNodeId(nodes[i + 1]);
            }
          }
        }
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
  }

  // Helper function to verify scope result
  void VerifyScope(const SuperKernelScopeInfo &scope, const std::vector<uint64_t> &expectedNodeIds) {
    EXPECT_EQ(scope.nodes_.size(), expectedNodeIds.size());
    std::vector<uint64_t> actualNodeIds;
    for (const auto *node : scope.nodes_) {
      actualNodeIds.push_back(node->GetNodeId());
    }
    std::sort(actualNodeIds.begin(), actualNodeIds.end());
    std::vector<uint64_t> sortedExpected = expectedNodeIds;
    std::sort(sortedExpected.begin(), sortedExpected.end());
    EXPECT_EQ(actualNodeIds, sortedExpected);
  }

  // Helper function to create a kernel node with ScheMode configuration
  SuperKernelBaseNode *CreateScheModeKernelNode(uint64_t nodeId, uint32_t streamIdx, uint32_t cubeNum, uint32_t vecNum,
                                                bool isScheModeOn = true, uint64_t nextNodeId = INVALID_TASK_ID) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->nodeType = SkNodeType::NODE_KERNEL;
    node->SetNodeId(nodeId);
    node->SetNextNodeId(nextNodeId);
    node->isFusible = true;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
    node->nodeInfos.kernelInfos.cubeNum = cubeNum;
    node->nodeInfos.kernelInfos.vecNum = vecNum;
    // 打桩：直接设置 isScheModeOn
    node->nodeInfos.kernelInfos.isScheModeOn = isScheModeOn;
    SuperKernelBaseNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to build a test scope from a node list
  SuperKernelScopeInfo BuildTestScope(const std::vector<SuperKernelBaseNode *> &nodeList) {
    SuperKernelScopeInfo scope;
    scope.nodes_ = nodeList;
    return scope;
  }

  bool RunDeadlockRefine(std::vector<SuperKernelScopeInfo> &scopes) {
    DeadlockRefinePass pass(*graph, *opts);
    pass.SetScopeSplitResultReporter(&reporter);
    return pass.Run(scopes);
  }

  bool FinalizeScopeCoreInfo(std::vector<SuperKernelScopeInfo> &scopes) {
    SuperKernelScopeSplitter splitter(*graph, *opts);
    splitter.scopeInfos_ = std::move(scopes);
    const bool result = splitter.FinalizeScopeCoreInfo();
    scopes = std::move(splitter.scopeInfos_);
    return result;
  }

  std::unique_ptr<SuperKernelGraph> graph;
  std::unique_ptr<SuperKernelOptionsManager> opts;
  ScopeSplitResultReporter reporter;
};

TEST_F(SuperKernelScopeSplitterTest, ScopeCoreInfoCalculatorWholeScopeMatchesProgressiveResult) {
  auto *cubeNode = CreateKernelNodeWithCores(1, 0, 2, 3, SkKernelType::AIC_ONLY);
  auto *vectorNode = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 5, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  ScopeCoreInfoCalculator calculator(25, 50);

  ScopeCoreInfo wholeScopeCoreInfo;
  ASSERT_TRUE(calculator.GetValidScopeCoreInfo({cubeNode, vectorNode}, wholeScopeCoreInfo));
  EXPECT_EQ(wholeScopeCoreInfo.type, SkKernelType::MIX_AIC_1_1);
  EXPECT_EQ(wholeScopeCoreInfo.numBlocks, 5U);

  calculator.Reset();
  ScopeCoreInfo progressiveScopeCoreInfo;
  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*cubeNode, progressiveScopeCoreInfo));
  EXPECT_EQ(progressiveScopeCoreInfo.type, SkKernelType::AIC_ONLY);
  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*vectorNode, progressiveScopeCoreInfo));
  EXPECT_EQ(progressiveScopeCoreInfo.type, wholeScopeCoreInfo.type);
  EXPECT_EQ(progressiveScopeCoreInfo.numBlocks, wholeScopeCoreInfo.numBlocks);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeCoreInfoCalculatorFailureDoesNotCommitIncrementalState) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *mix12Node = CreateKernelNodeWithCores(2, 0, 3, 3, SkKernelType::MIX_AIC_1_2);
  auto *cubeNode = CreateKernelNodeWithCores(3, 0, INVALID_TASK_ID, 2, SkKernelType::AIC_ONLY);
  ScopeCoreInfoCalculator calculator(25, 50);

  ScopeCoreInfo scopeCoreInfo;
  ASSERT_FALSE(calculator.GetValidScopeCoreInfo({vectorNode, mix12Node}, scopeCoreInfo));
  EXPECT_EQ(scopeCoreInfo.type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopeCoreInfo.numBlocks, 3U);

  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*vectorNode, scopeCoreInfo));
  ASSERT_FALSE(calculator.UpdateScopeCoreInfo(*mix12Node, scopeCoreInfo));
  EXPECT_EQ(scopeCoreInfo.type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopeCoreInfo.numBlocks, 3U);

  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*cubeNode, scopeCoreInfo));
  EXPECT_EQ(scopeCoreInfo.type, SkKernelType::MIX_AIC_1_1);
  EXPECT_EQ(scopeCoreInfo.numBlocks, 5U);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeCoreInfoCalculatorResetClearsIncrementalState) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *mix12Node = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 3, SkKernelType::MIX_AIC_1_2);
  ScopeCoreInfoCalculator calculator(25, 50);

  ScopeCoreInfo scopeCoreInfo;
  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*vectorNode, scopeCoreInfo));
  calculator.Reset();
  ASSERT_TRUE(calculator.UpdateScopeCoreInfo(*mix12Node, scopeCoreInfo));
  EXPECT_EQ(scopeCoreInfo.type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopeCoreInfo.numBlocks, 3U);
}

TEST_F(SuperKernelScopeSplitterTest, RequiresExactCoreMatchExcludesBlockDimScaleUp) {
  auto *node = CreateScheModeKernelNode(1, 0, 3, 3);
  ASSERT_TRUE(node->RequiresExactCoreMatch());

  node->nodeInfos.kernelInfos.capBits.blockDimScaleUp = true;
  EXPECT_FALSE(node->RequiresExactCoreMatch());
}

// ==================== Basic Tests: 基础图切分和融合 ====================

TEST_F(SuperKernelScopeSplitterTest, PassOrder_ScheModeSplitBeforeDeadlockRefine) {
  SuperKernelScopeSplitter splitter(*graph, *opts);

  ASSERT_GE(splitter.passes_.size(), 3);
  EXPECT_NE(dynamic_cast<InitialScopeSplitPass *>(splitter.passes_[0].get()), nullptr);
  EXPECT_NE(dynamic_cast<ScheModeKernelSplitPass *>(splitter.passes_[1].get()), nullptr);
  EXPECT_NE(dynamic_cast<DeadlockRefinePass *>(splitter.passes_[2].get()), nullptr);
}

TEST_F(SuperKernelScopeSplitterTest, LockDetectorRecordsScopeInfoByScopeId) {
  auto *notify = CreateNotifyNode(1, 0, 7);
  auto *wait = CreateWaitNode(2, 1, notify->GetNodeId());

  SuperKernelScopeInfo scope;
  scope.AddNode(wait);
  scope.AddScopeStreamInfo({1, wait->GetNodeId(), wait->GetNodeId(), 1});

  DeadlockRefinePass pass(*graph, *opts);
  const ScopeCoreInfo scopeCoreInfo{SkKernelType::MIX_AIC_1_2, 4};
  scope.SetScopeCoreInfo(scopeCoreInfo);
  pass.lockDetector_.UpdateScopeInfo(scope);

  const auto &scopeInfo = pass.lockDetector_.scopeInfos_.at(scope.GetScopeId());
  EXPECT_EQ(scopeInfo.streamIds, (std::unordered_set<uint32_t>{1}));
  EXPECT_EQ(scopeInfo.headNodeIds.at(1), wait->GetNodeId());
  EXPECT_EQ(scopeInfo.tailNodeIds.at(1), wait->GetNodeId());
  EXPECT_EQ(scopeInfo.coreInfo.GetCubeNum(), 4U);
  EXPECT_EQ(scopeInfo.coreInfo.GetVectorNum(), 8U);
}

/**
 * @brief 基础多流融合 - 无跨流依赖
 *
 * 图结构:
 *   stream0: [K1(id=1) -> K2(id=2) -> K3(id=3)]
 *   stream1: [K4(id=4) -> K5(id=5) -> K6(id=6)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6} (所有节点融合)
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_MultiStreamNoCrossStreamDependency) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, 3);
  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *k4 = CreateKernelNode(4, 1, 5);
  auto *k5 = CreateKernelNode(5, 1, 6);
  auto *k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4, 5, 6}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

/**
 * @brief 单个跨流Wait-Notify依赖
 *
 * 图结构:
 *   stream0: [K1(id=1) -> Wait1(id=3) -> K2(id=5)]
 *   stream1: [K3(id=2) -> Notify1(id=4) -> K4(id=6)]
 *   Event1(eventId=100): Wait1(3) 等待 Notify1(4)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6}
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_SingleCrossStreamWaitNotify) {
  auto *k1 = CreateKernelNode(1, 0, 3);
  auto *wait1 = CreateWaitNode(3, 0, 4, 5);
  auto *k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 4);
  auto *notify1 = CreateNotifyNode(4, 1, 100, 6);
  auto *k4 = CreateKernelNode(6, 1, INVALID_TASK_ID);

  SetupStreams({{1, 3, 5}, {2, 4, 6}});
  SetupEvent(100, 4, {3});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

/**
 * @brief 多个Wait-Notify依赖链
 *
 * 图结构:
 *   stream0: [K1(id=1) -> Wait1(id=4) -> K2(id=7)]
 *   stream1: [K3(id=2) -> Notify1(id=5) -> Wait2(id=8) -> K4(id=10)]
 *   stream2: [K5(id=3) -> Notify2(id=6) -> K6(id=9)]
 *   Event1: Wait1(4) 等待 Notify1(5)
 *   Event2: Wait2(8) 等待 Notify2(6)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_MultipleWaitNotifyDependencies) {
  auto *k1 = CreateKernelNode(1, 0, 4);
  auto *wait1 = CreateWaitNode(4, 0, 5, 7);
  auto *k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 5);
  auto *notify1 = CreateNotifyNode(5, 1, 100, 8);
  auto *wait2 = CreateWaitNode(8, 1, 6, 10);
  auto *k4 = CreateKernelNode(10, 1, INVALID_TASK_ID);

  auto *k5 = CreateKernelNode(3, 2, 6);
  auto *notify2 = CreateNotifyNode(6, 2, 200, 9);
  auto *k6 = CreateKernelNode(9, 2, INVALID_TASK_ID);

  SetupStreams({{1, 4, 7}, {2, 5, 8, 10}, {3, 6, 9}});
  SetupEvent(100, 5, {4});
  SetupEvent(200, 6, {8});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

/**
 * @brief 不可融合节点单流测试 - 触发scope切分
 *
 * 图结构:
 *   stream0: [K1(id=1, fusible) -> UF1(id=2, unfusible) -> K2(id=3, fusible)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成2个scope: Scope0={K1(1)}, Scope1={K2(3)}
 *   - UF1作为切分边界
 */
TEST_F(SuperKernelScopeSplitterTest, Unfusible_SingleStream) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *uf1 = CreateUnfusibleKernelNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);
  VerifyScope(scopeInfos[0], {1});
  VerifyScope(scopeInfos[1], {3});
}

/**
 * @brief 不可融合节点多流测试
 *
 * 图结构:
 *   stream0: [K1(id=1, fusible) -> UF1(id=2, unfusible) -> K2(id=3, fusible)]
 *   stream1: [K3(id=4, fusible)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成2个scope: {K1(1), K3(4)}, {K2(3)}
 */
TEST_F(SuperKernelScopeSplitterTest, Unfusible_MultiStream) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *uf1 = CreateUnfusibleKernelNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(4, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);
  VerifyScope(scopeInfos[0], {1, 4});
  VerifyScope(scopeInfos[1], {3});
}

/**
 * @brief 多Wait等待同一Notify
 *
 * 图结构:
 *   stream0: [K1(id=1) -> Wait1(id=4) -> K2(id=7)]
 *   stream1: [K3(id=2) -> Wait2(id=5) -> K4(id=8)]
 *   stream2: [K5(id=3) -> Notify1(id=6) -> K6(id=9)]
 *   Event1: Wait1(4)和Wait2(5) 都等待 Notify1(6)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6, 7, 8, 9}
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_MultipleWaitsOnSameNotify) {
  auto *k1 = CreateKernelNode(1, 0, 4);
  auto *wait1 = CreateWaitNode(4, 0, 6, 7);
  auto *k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 5);
  auto *wait2 = CreateWaitNode(5, 1, 6, 8);
  auto *k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

  auto *k5 = CreateKernelNode(3, 2, 6);
  auto *notify1 = CreateNotifyNode(6, 2, 100, 9);
  auto *k6 = CreateKernelNode(9, 2, INVALID_TASK_ID);

  SetupStreams({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});
  SetupEvent(100, 6, {4, 5});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

/**
 * @brief 连续不可融合节点
 *
 * 图结构:
 *   stream0: [K1(id=1, fusible) -> UF1(id=2) -> UF2(id=3) -> K2(id=4, fusible)]
 *   stream1: [K3(id=5, fusible)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成2个scope: {K1(1), K3(5)}, {K2(4)}
 */
TEST_F(SuperKernelScopeSplitterTest, Unfusible_ConsecutiveUnfusibleNodes) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *uf1 = CreateUnfusibleKernelNode(2, 0, 3);
  auto *uf2 = CreateUnfusibleKernelNode(3, 0, 4);
  auto *k2 = CreateKernelNode(4, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(5, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4}, {5}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);
  VerifyScope(scopeInfos[0], {1, 5});
  VerifyScope(scopeInfos[1], {4});
}

/**
 * @brief EventOnlyStreamRemovePass检测纯Event流 - 独立Pass验证
 *
 * 手动构建scope包含纯Event流:
 *   Stream0: [Wait1(id=1) -> K1(id=3)] (混合)
 *   Stream1: [Notify1(id=5) -> Notify2(id=6)] (纯Event)
 *   Stream2: [Wait2(id=2) -> K2(id=4)] (混合)
 *
 * 预期结果:
 *   - EventOnlyStreamRemovePass.Run返回true
 *   - Stream1纯Event节点标记为non-fusible
 *   - scopes清空触发重切分
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_PassDetectsPureEventStream) {
  auto *wait1 = CreateWaitNode(1, 0, 5, 3);
  auto *k1 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *wait2 = CreateWaitNode(2, 1, 6, 4);
  auto *k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(5, 2, 100, 6);
  auto *notify2 = CreateNotifyNode(6, 2, 200, INVALID_TASK_ID);

  SetupStreams({{1, 3}, {2, 4}, {5, 6}});
  SetupEvent(100, 5, {1});
  SetupEvent(200, 6, {2});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.AddNode(wait1);
  scope.AddNode(k1);
  scope.AddNode(wait2);
  scope.AddNode(k2);
  scope.AddNode(notify1);
  scope.AddNode(notify2);
  ScopeStreamInfo info0{0, 1, 3, 2};
  ScopeStreamInfo info1{1, 2, 4, 2};
  ScopeStreamInfo info2{2, 5, 6, 2};
  scope.scopeStreamInfos_.push_back(info0);
  scope.scopeStreamInfos_.push_back(info1);
  scope.scopeStreamInfos_.push_back(info2);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_FALSE(notify2->IsFusible());
  EXPECT_TRUE(k1->IsFusible());
  EXPECT_TRUE(k2->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
  EXPECT_EQ(wait1->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(k1->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(wait2->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(k2->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(notify1->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(notify2->GetScopeId(), INVALID_SCOPE_ID);
}

/**
 * @brief 三流多Wait-Notify + 不可融合节点
 *
 * 图结构:
 *   stream0: [K1(1) -> K2(3) -> Wait1(5) -> K3(7) -> UF1(9) -> K4(11) -> K5(13)]
 *   stream1: [K6(2) -> Notify1(4) -> K7(6) -> Wait2(8) -> K8(10)]
 *   stream2: [K9(12) -> Notify2(14) -> K10(16)]
 *   Event1: Wait1(5) 等待 Notify1(4)
 *   Event2: Wait2(8) 等待 Notify2(14)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - UF1(9)触发切分，生成2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_ThreeStreamsWithWaitNotifyAndUnfusible) {
  auto *k1 = CreateKernelNode(1, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 5);
  auto *wait1 = CreateWaitNode(5, 0, 4, 7);
  auto *k3 = CreateKernelNode(7, 0, 9);
  auto *uf1 = CreateUnfusibleKernelNode(9, 0, 11);
  auto *k4 = CreateKernelNode(11, 0, 13);
  auto *k5 = CreateKernelNode(13, 0, INVALID_TASK_ID);

  auto *k6 = CreateKernelNode(2, 1, 4);
  auto *notify1 = CreateNotifyNode(4, 1, 100, 6);
  auto *k7 = CreateKernelNode(6, 1, 8);
  auto *wait2 = CreateWaitNode(8, 1, 14, 10);
  auto *k8 = CreateKernelNode(10, 1, INVALID_TASK_ID);

  auto *k9 = CreateKernelNode(12, 2, 14);
  auto *notify2 = CreateNotifyNode(14, 2, 200, 16);
  auto *k10 = CreateKernelNode(16, 2, INVALID_TASK_ID);

  SetupStreams({{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10}, {12, 14, 16}});
  SetupEvent(100, 4, {5});
  SetupEvent(200, 14, {8});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16});
  VerifyScope(scopeInfos[1], {11, 13});
}

/**
 * @brief 嵌套Wait-Notify
 *
 * 图结构:
 *   stream0: [K1(1) -> Wait1(4) -> K2(7)]
 *   stream1: [K3(2) -> Notify1(5) -> Wait2(8) -> K4(10)]
 *   stream2: [K5(3) -> K6(6) -> Notify2(9)]
 *   Event1: Wait1(4) 等待 Notify1(5)
 *   Event2: Wait2(8) 等待 Notify2(9)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_NestedWaitNotify) {
  auto *k1 = CreateKernelNode(1, 0, 4);
  auto *wait1 = CreateWaitNode(4, 0, 5, 7);
  auto *k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 5);
  auto *notify1 = CreateNotifyNode(5, 1, 100, 8);
  auto *wait2 = CreateWaitNode(8, 1, 9, 10);
  auto *k4 = CreateKernelNode(10, 1, INVALID_TASK_ID);

  auto *k5 = CreateKernelNode(3, 2, 6);
  auto *k6 = CreateKernelNode(6, 2, 9);
  auto *notify2 = CreateNotifyNode(9, 2, 200, INVALID_TASK_ID);

  SetupStreams({{1, 4, 7}, {2, 5, 8, 10}, {3, 6, 9}});
  SetupEvent(100, 5, {4});
  SetupEvent(200, 9, {8});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
}

/**
 * @brief 单流多节点 - 统一SplitGraph路径
 *
 * 图结构:
 *   stream0: [K1(1) -> K2(2) -> K3(3)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {K1(1), K2(2), K3(3)}
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_SingleStreamMultipleNodes) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, 3);
  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3});
}

// ==================== ScopeBitFlags 相关测试用例 ====================

/**
 * @brief 全部不可融合节点 - 无scope生成
 *
 * 图结构:
 *   stream0: [UF1(1, unfusible) -> UF2(2, unfusible)]
 *   stream1: [UF3(3, unfusible)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成0个scope (无可融合节点)
 */
TEST_F(SuperKernelScopeSplitterTest, Unfusible_AllNodesUnfusible) {
  auto *uf1 = CreateUnfusibleKernelNode(1, 0, 2);
  auto *uf2 = CreateUnfusibleKernelNode(2, 0, INVALID_TASK_ID);
  auto *uf3 = CreateUnfusibleKernelNode(3, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2}, {3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 0);
}

/**
 * @brief 流内节点执行顺序保持验证
 *
 * 图结构:
 *   stream0: [K1(10) -> K2(30) -> K3(50)]
 *   stream1: [K4(20) -> K5(40) -> K6(60)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {10, 20, 30, 40, 50, 60}
 *   - Stream0节点顺序: K1(10)在K2(30)之前，K2(30)在K3(50)之前
 *   - Stream1节点顺序: K4(20)在K5(40)之前，K5(40)在K6(60)之前
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_StreamOrderPreserved) {
  auto *k1 = CreateKernelNode(10, 0, 30);
  auto *k2 = CreateKernelNode(30, 0, 50);
  auto *k3 = CreateKernelNode(50, 0, INVALID_TASK_ID);

  auto *k4 = CreateKernelNode(20, 1, 40);
  auto *k5 = CreateKernelNode(40, 1, 60);
  auto *k6 = CreateKernelNode(60, 1, INVALID_TASK_ID);

  SetupStreams({{10, 30, 50}, {20, 40, 60}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {10, 20, 30, 40, 50, 60});

  const auto &scopeNodes = scopeInfos[0].nodes_;
  std::vector<size_t> stream0Positions;
  std::vector<size_t> stream1Positions;

  for (size_t i = 0; i < scopeNodes.size(); ++i) {
    uint32_t streamIdx = scopeNodes[i]->GetStreamIdxInGraph();
    if (streamIdx == 0) {
      stream0Positions.push_back(i);
    } else if (streamIdx == 1) {
      stream1Positions.push_back(i);
    }
  }

  ASSERT_EQ(stream0Positions.size(), 3);
  ASSERT_EQ(stream1Positions.size(), 3);

  std::vector<uint64_t> stream0NodeIds;
  std::vector<uint64_t> stream1NodeIds;
  for (size_t pos : stream0Positions) {
    stream0NodeIds.push_back(scopeNodes[pos]->GetNodeId());
  }
  for (size_t pos : stream1Positions) {
    stream1NodeIds.push_back(scopeNodes[pos]->GetNodeId());
  }

  EXPECT_TRUE(stream0NodeIds[0] < stream0NodeIds[1]);
  EXPECT_TRUE(stream0NodeIds[1] < stream0NodeIds[2]);

  EXPECT_TRUE(stream1NodeIds[0] < stream1NodeIds[1]);
  EXPECT_TRUE(stream1NodeIds[1] < stream1NodeIds[2]);
}

/**
 * @brief 多个Scope切分 - Wait-Notify在不同Scope
 *
 * 图结构:
 *   stream0: [K1(1) -> Wait1(3) -> K2(5)]
 *   stream1: [K3(2) -> K4(4) -> Notify1(6)]
 *   Event1: Wait1(3) 等待 Notify1(6)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成多个scope，所有节点 {1, 2, 3, 4, 5, 6} 被融合
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_MultipleScopesWithWaitNotify) {
  auto *k1 = CreateKernelNode(1, 0, 3);
  auto *wait1 = CreateWaitNode(3, 0, 6, 5);
  auto *k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 4);
  auto *k4 = CreateKernelNode(4, 1, 6);
  auto *notify1 = CreateNotifyNode(6, 1, 100, INVALID_TASK_ID);

  SetupStreams({{1, 3, 5}, {2, 4, 6}});
  SetupEvent(100, 6, {3});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_GE(scopeInfos.size(), 1);

  std::vector<uint64_t> allNodeIds;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allNodeIds.push_back(node->GetNodeId());
    }
  }
  std::sort(allNodeIds.begin(), allNodeIds.end());
  std::vector<uint64_t> expectedIds = {1, 2, 3, 4, 5, 6};
  std::sort(expectedIds.begin(), expectedIds.end());
  EXPECT_EQ(allNodeIds, expectedIds);
}

/**
 * @brief 验证四流无跨流依赖时全部节点融合到单个scope
 *
 * 图结构:
 *   stream0: [K1(1) -> K5(5)]
 *   stream1: [K2(2) -> K6(6)]
 *   stream2: [K3(3) -> K7(7)]
 *   stream3: [K4(4) -> K8(8)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6, 7, 8}
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_FourStreamParallelFusion) {
  auto *k1 = CreateKernelNode(1, 0, 5);
  auto *k2 = CreateKernelNode(5, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(2, 1, 6);
  auto *k4 = CreateKernelNode(6, 1, INVALID_TASK_ID);

  auto *k5 = CreateKernelNode(3, 2, 7);
  auto *k6 = CreateKernelNode(7, 2, INVALID_TASK_ID);

  auto *k7 = CreateKernelNode(4, 3, 8);
  auto *k8 = CreateKernelNode(8, 3, INVALID_TASK_ID);

  SetupStreams({{1, 5}, {2, 6}, {3, 7}, {4, 8}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6, 7, 8});
}

/**
 * @brief Wait在Notify之前 - 触发suspend机制
 *
 * 图结构:
 *   stream0: [Wait1(1) -> K1(3)]
 *   stream1: [K2(2) -> Notify1(4)]
 *   Event1: Wait1(1) 等待 Notify1(4)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {Wait1(1), K2(2), K1(3), Notify1(4)}
 *   - Wait1在Notify1之后处理（通过suspend/resume）
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_WaitBeforeNotifySuspend) {
  auto *wait1 = CreateWaitNode(1, 0, 4, 3);
  auto *k1 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *k2 = CreateKernelNode(2, 1, 4);
  auto *notify1 = CreateNotifyNode(4, 1, 100, INVALID_TASK_ID);

  SetupStreams({{1, 3}, {2, 4}});
  SetupEvent(100, 4, {1});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4});

  const auto &scopeNodes = scopeInfos[0].nodes_;
  size_t waitPos = 0, notifyPos = 0;
  for (size_t i = 0; i < scopeNodes.size(); ++i) {
    if (scopeNodes[i]->GetNodeId() == 1) {
      waitPos = i;
    } else if (scopeNodes[i]->GetNodeId() == 4) {
      notifyPos = i;
    }
  }
  EXPECT_GT(waitPos, notifyPos);
}

/**
 * @brief 不可融合节点在流中间 - 多Scope切分
 *
 * 图结构:
 *   stream0: [K1(1) -> UF1(2, unfusible) -> K2(3) -> K3(4)]
 *   stream1: [K4(6) -> K5(5)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成2个scope: {K1(1), K5(5), K4(6)}, {K2(3), K3(4)}
 */
TEST_F(SuperKernelScopeSplitterTest, Unfusible_NodeInMiddle) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *uf1 = CreateUnfusibleKernelNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);

  auto *k4 = CreateKernelNode(6, 1, 5);
  auto *k5 = CreateKernelNode(5, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4}, {6, 5}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);
  VerifyScope(scopeInfos[0], {1, 5, 6});
  VerifyScope(scopeInfos[1], {3, 4});
}

/**
 * @brief 复杂Wait-Notify链验证多流依赖处理
 *
 * 图结构:
 *   stream0: [K1(1) -> Wait1(5) -> K2(9) -> Wait2(13) -> K3(17)]
 *   stream1: [K4(2) -> Notify1(6) -> K5(10) -> Notify2(14) -> K7(18)]
 *   stream2: [K6(3) -> Notify3(15) -> Wait3(7) -> K8(11) -> K9(19)]
 *   stream3: [K10(4) -> Notify4(8) -> K11(12) -> Wait4(16) -> K12(20)]
 *   Event100: Wait1(5) 等待 Notify1(6)
 *   Event200: Wait2(13) 等待 Notify2(14)
 *   Event300: Wait3(7) 等待 Notify3(15)
 *   Event400: Wait4(16) 等待 Notify4(8)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成>=1个scope
 *   - 所有节点 {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20} 被处理
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_ComplexWaitNotifyChain) {
  auto *k1 = CreateKernelNode(1, 0, 5);
  auto *wait1 = CreateWaitNode(5, 0, 6, 9);
  auto *k2 = CreateKernelNode(9, 0, 13);
  auto *wait2 = CreateWaitNode(13, 0, 14, 17);
  auto *k3 = CreateKernelNode(17, 0, INVALID_TASK_ID);

  auto *k4 = CreateKernelNode(2, 1, 6);
  auto *notify1 = CreateNotifyNode(6, 1, 100, 10);
  auto *k5 = CreateKernelNode(10, 1, 14);
  auto *notify2 = CreateNotifyNode(14, 1, 200, 18);
  auto *k7 = CreateKernelNode(18, 1, INVALID_TASK_ID);

  auto *k6 = CreateKernelNode(3, 2, 15);
  auto *notify3 = CreateNotifyNode(15, 2, 300, 7);
  auto *wait3 = CreateWaitNode(7, 2, 15, 11);
  auto *k8 = CreateKernelNode(11, 2, 19);
  auto *k9 = CreateKernelNode(19, 2, INVALID_TASK_ID);

  auto *k10 = CreateKernelNode(4, 3, 8);
  auto *notify4 = CreateNotifyNode(8, 3, 400, 12);
  auto *k11 = CreateKernelNode(12, 3, 16);
  auto *wait4 = CreateWaitNode(16, 3, 8, 20);
  auto *k12 = CreateKernelNode(20, 3, INVALID_TASK_ID);

  SetupStreams({{1, 5, 9, 13, 17}, {2, 6, 10, 14, 18}, {3, 15, 7, 11, 19}, {4, 8, 12, 16, 20}});
  SetupEvent(100, 6, {5});
  SetupEvent(200, 14, {13});
  SetupEvent(300, 15, {7});
  SetupEvent(400, 8, {16});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_GE(scopeInfos.size(), 1);

  std::vector<uint64_t> allNodeIds;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allNodeIds.push_back(node->GetNodeId());
    }
  }
  std::sort(allNodeIds.begin(), allNodeIds.end());
  std::vector<uint64_t> expectedIds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  std::sort(expectedIds.begin(), expectedIds.end());
  EXPECT_EQ(allNodeIds, expectedIds);
}

/**
 * @brief 单流单节点
 *
 * 图结构:
 *   stream0: [K1(1)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {K1(1)}
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_SingleStreamSingleNode) {
  auto *k1 = CreateKernelNode(1, 0, INVALID_TASK_ID);

  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1});
}

/**
 * @brief 多流每流单节点
 *
 * 图结构:
 *   stream0: [K1(1)]
 *   stream1: [K2(2)]
 *   stream2: [K3(3)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {K1(1), K2(2), K3(3)}
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_MultiStreamSingleNodeEach) {
  auto *k1 = CreateKernelNode(1, 0, INVALID_TASK_ID);
  auto *k2 = CreateKernelNode(2, 1, INVALID_TASK_ID);
  auto *k3 = CreateKernelNode(3, 2, INVALID_TASK_ID);

  SetupStreams({{1}, {2}, {3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3});
}

/**
 * @brief ResetStreamStates恢复暂停的流
 *
 * 图结构:
 *   stream0: [K1(1, 4 cores) -> Wait1(3) -> K2(5, 4 cores)]
 *   stream1: [K3(2, 2 cores) -> Notify1(4)]
 *   stream2: [K4(6, 2 cores) -> Notify2(8)]
 *   Event100: Wait1(3) 等待 Notify1(4)
 *   Event200: Notify2(8) 无对应Wait(用于测试无依赖Notify)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - Wait1(3)触发stream0暂停
 *   - Notify1(4)处理时恢复stream0
 *   - 所有节点 {1, 2, 3, 4, 5, 6, 8} 被处理
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_ResetStreamStatesResume) {
  auto *k1 = CreateKernelNodeWithCores(1, 0, 3, 4, SkKernelType::AIC_ONLY);
  auto *wait1 = CreateWaitNode(3, 0, 4, 5);
  auto *k2 = CreateKernelNodeWithCores(5, 0, INVALID_TASK_ID, 4, SkKernelType::AIC_ONLY);

  auto *k3 = CreateKernelNodeWithCores(2, 1, 4, 2, SkKernelType::AIC_ONLY);
  auto *notify1 = CreateNotifyNode(4, 1, 100, INVALID_TASK_ID);

  auto *k4 = CreateKernelNodeWithCores(6, 2, 8, 2, SkKernelType::AIC_ONLY);
  auto *notify2 = CreateNotifyNode(8, 2, 200, INVALID_TASK_ID);

  SetupStreams({{1, 3, 5}, {2, 4}, {6, 8}});
  SetupEvent(100, 4, {3});
  SetupEvent(200, 8, {});

  graph->BuildEventNodeAssociations();

  EXPECT_TRUE(notify2->IsFusible());
  EXPECT_TRUE(notify2->GetCorrespondingWaitNodeIds().empty());

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_GE(scopeInfos.size(), 1);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 8};
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief 多次暂停恢复
 *
 * 图结构:
 *   stream0: [Wait1(1) -> K1(3) -> Wait2(5) -> K2(7)]
 *   stream1: [Notify1(2) -> K3(4) -> Notify2(6) -> K4(8)]
 *   stream2: [K5(9)]
 *   Event1: Wait1(1) 等待 Notify1(2)
 *   Event2: Wait2(5) 等待 Notify2(6)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 全部节点被处理: {1, 2, 3, 4, 5, 6, 7, 8, 9}
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_MultipleSuspendResume) {
  auto *wait1 = CreateWaitNode(1, 0, 2, 3);
  auto *k1 = CreateKernelNode(3, 0, 5);
  auto *wait2 = CreateWaitNode(5, 0, 6, 7);
  auto *k2 = CreateKernelNode(7, 0, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(2, 1, 100, 4);
  auto *k3 = CreateKernelNode(4, 1, 6);
  auto *notify2 = CreateNotifyNode(6, 1, 200, 8);
  auto *k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

  auto *k5 = CreateKernelNode(9, 2, INVALID_TASK_ID);

  SetupStreams({{1, 3, 5, 7}, {2, 4, 6, 8}, {9}});
  SetupEvent(100, 2, {1});
  SetupEvent(200, 6, {5});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief 无Scope节点全图融合
 *
 * 图结构:
 *   stream0: [K1(1) -> K2(2) -> K3(3)]
 *   stream1: [K4(4) -> K5(5) -> K6(6)]
 *   无Scope标记节点
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6}
 */
TEST_F(SuperKernelScopeSplitterTest, Basic_NoScopeNodesFullGraphFusion) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, 3);
  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *k4 = CreateKernelNode(4, 1, 5);
  auto *k5 = CreateKernelNode(5, 1, 6);
  auto *k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4, 5, 6}});

  graph->scopeNameToIdx.clear();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);
  VerifyScope(scopeInfos[0], {1, 2, 3, 4, 5, 6});
}

/**
 * @brief 单个可融合Scope - ScopeBitFlags标记
 *
 * 图结构:
 *   stream0: [K1(1) -> ScopeBegin_A(2) -> K2(3) -> K3(4) -> ScopeEnd_A(5) -> K4(6)]
 *   scope_A为可融合Scope
 *
 * 预期结果:
 *   - UpdateNodeScopeBitFlags标记:
 *     - K1/K4: scopeBitFlags=0 (不属于任何scope)
 *     - ScopeBegin_A/K2/K3/ScopeEnd_A: scopeBitFlags第0位为1
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_SingleFusibleScope) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *scopeBegin = CreateScopeBeginNode(2, 0, "scope_A", 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *k3 = CreateKernelNode(4, 0, 5);
  auto *scopeEnd = CreateScopeEndNode(5, 0, "scope_A", 6);
  auto *k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);

  EXPECT_TRUE(scopeBegin->GetScopeBitFlags().test(0));
  EXPECT_EQ(scopeBegin->GetScopeBitFlags().count(), 1);

  EXPECT_TRUE(k2->GetScopeBitFlags().test(0));
  EXPECT_EQ(k2->GetScopeBitFlags().count(), 1);
  EXPECT_TRUE(k3->GetScopeBitFlags().test(0));
  EXPECT_EQ(k3->GetScopeBitFlags().count(), 1);

  EXPECT_TRUE(scopeEnd->GetScopeBitFlags().test(0));
  EXPECT_EQ(scopeEnd->GetScopeBitFlags().count(), 1);

  EXPECT_EQ(k4->GetScopeBitFlags().count(), 0);
}

/**
 * @brief 多个可融合Scope - 不同ScopeBitFlags
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> ScopeEnd_A(3) -> K2(4) -> ScopeBegin_B(5) -> K3(6) -> ScopeEnd_B(7)]
 *
 * 预期结果:
 *   - Scope_A节点: scopeBitFlags第0位为1
 *   - K2(4): scopeBitFlags=0
 *   - Scope_B节点: scopeBitFlags第1位为1
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_MultipleFusibleScopes) {
  auto *scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *scopeEndA = CreateScopeEndNode(3, 0, "scope_A", 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *scopeBeginB = CreateScopeBeginNode(5, 0, "scope_B", 6);
  auto *k3 = CreateKernelNode(6, 0, 7);
  auto *scopeEndB = CreateScopeEndNode(7, 0, "scope_B", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

  graph->scopeNameToIdx["scope_A"] = 0;
  graph->scopeNameToIdx["scope_B"] = 1;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(scopeBeginA->GetScopeBitFlags().test(0));
  EXPECT_TRUE(k1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeEndA->GetScopeBitFlags().test(0));

  EXPECT_EQ(k2->GetScopeBitFlags().count(), 0);

  EXPECT_TRUE(scopeBeginB->GetScopeBitFlags().test(1));
  EXPECT_TRUE(k3->GetScopeBitFlags().test(1));
  EXPECT_TRUE(scopeEndB->GetScopeBitFlags().test(1));

  EXPECT_FALSE(scopeBeginA->GetScopeBitFlags().test(1));
  EXPECT_FALSE(k1->GetScopeBitFlags().test(1));
  EXPECT_FALSE(scopeEndA->GetScopeBitFlags().test(1));

  EXPECT_FALSE(scopeBeginB->GetScopeBitFlags().test(0));
  EXPECT_FALSE(k3->GetScopeBitFlags().test(0));
  EXPECT_FALSE(scopeEndB->GetScopeBitFlags().test(0));
}

/**
 * @brief 不可融合Scope - 中间节点标记为不可融合
 *
 * 图结构:
 *   stream0: [K1(1) -> UnfusibleBegin(2) -> K2(3) -> K3(4) -> UnfusibleEnd(5) -> K4(6)]
 *   UnfusibleBegin/End为不可融合Scope节点
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 生成2个scope: {K1(1), UnfusibleBegin(2)}, {UnfusibleEnd(5), K4(6)}
 *   - K2(3)和K3(4)被标记为不可融合
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnfusibleScope) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *k3 = CreateKernelNode(4, 0, 5);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
  auto *k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6}});

  graph->UpdateNodeScopeBitFlags();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 2);
  EXPECT_EQ(scopeInfos[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(scopeInfos[0].nodes_[1]->GetNodeId(), 2);

  EXPECT_EQ(scopeInfos[1].nodes_.size(), 2);
  EXPECT_EQ(scopeInfos[1].nodes_[0]->GetNodeId(), 5);
  EXPECT_EQ(scopeInfos[1].nodes_[1]->GetNodeId(), 6);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 5, 6};  // K1, UnfusibleBegin, UnfusibleEnd, K4
  EXPECT_EQ(allProcessedNodes, expectedNodes);

  EXPECT_FALSE(k2->IsFusible());
  EXPECT_FALSE(k3->IsFusible());
  EXPECT_TRUE(k1->IsFusible());
  EXPECT_TRUE(k4->IsFusible());
}

/**
 * @brief 可融合+不可融合Scope混合
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> ScopeEnd_A(3) -> UnfusibleBegin(4) -> K2(5) -> UnfusibleEnd(6) -> K3(7)]
 *
 * 预期结果:
 *   - 生成3个scope
 *   - K2(5)在unfusible scope内不可融合
 *   - K3(7)在命名scope外不可融合
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_MixedFusibleAndUnfusible) {
  auto *scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *scopeEndA = CreateScopeEndNode(3, 0, "scope_A", 4);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(4, 0, 5);
  auto *k2 = CreateKernelNode(5, 0, 6);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(6, 0, 7);
  auto *k3 = CreateKernelNode(7, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 3);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 3);
  EXPECT_EQ(scopeInfos[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(scopeInfos[0].nodes_[1]->GetNodeId(), 2);
  EXPECT_EQ(scopeInfos[0].nodes_[2]->GetNodeId(), 3);

  EXPECT_EQ(scopeInfos[1].nodes_.size(), 1);
  EXPECT_EQ(scopeInfos[1].nodes_[0]->GetNodeId(), 4);

  EXPECT_EQ(scopeInfos[2].nodes_.size(), 1);
  EXPECT_EQ(scopeInfos[2].nodes_[0]->GetNodeId(), 6);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 6};  // ScopeBegin_A, K1, ScopeEnd_A, UnfusibleBegin, UnfusibleEnd
  EXPECT_EQ(allProcessedNodes, expectedNodes);

  EXPECT_TRUE(k1->IsFusible());
  EXPECT_FALSE(k2->IsFusible());
  EXPECT_FALSE(k3->IsFusible());  // K3在scope外，被标记为不可融合
}

/**
 * @brief 相同Scope名称跨流
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> ScopeEnd_A(3)]
 *   stream1: [K2(4) -> ScopeBegin_A(5) -> K3(6) -> ScopeEnd_A(7)]
 *   两流使用相同scope名称"scope_A"
 *
 * 预期结果:
 *   - 相同scope名称节点使用相同ScopeBitFlags
 *   - 生成1个scope: {1, 2, 3, 4, 5, 6, 7}
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_SameNameAcrossStreams) {
  auto *scopeBeginA0 = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *scopeEndA0 = CreateScopeEndNode(3, 0, "scope_A", INVALID_TASK_ID);

  auto *k2 = CreateKernelNode(4, 1, 5);
  auto *scopeBeginA1 = CreateScopeBeginNode(5, 1, "scope_A", 6);
  auto *k3 = CreateKernelNode(6, 1, 7);
  auto *scopeEndA1 = CreateScopeEndNode(7, 1, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4, 5, 6, 7}});

  graph->scopeNameToIdx["scope_A"] = 0;

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 7};  // 所有节点
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief 嵌套Scope - ScopeBitFlags多位设置
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> ScopeBegin_B(2) -> K1(3) -> ScopeEnd_B(4) -> K2(5) -> ScopeEnd_A(6)]
 *   scope_B嵌套在scope_A内部
 *
 * 预期结果:
 *   - K1: scopeBitFlags同时包含scope_A和scope_B
 *   - K2: scopeBitFlags只包含scope_A
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_NestedScopes) {
  auto *scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *scopeBeginB = CreateScopeBeginNode(2, 0, "scope_B", 3);
  auto *k1 = CreateKernelNode(3, 0, 4);
  auto *scopeEndB = CreateScopeEndNode(4, 0, "scope_B", 5);
  auto *k2 = CreateKernelNode(5, 0, 6);
  auto *scopeEndA = CreateScopeEndNode(6, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6}});

  graph->scopeNameToIdx["scope_A"] = 0;
  graph->scopeNameToIdx["scope_B"] = 1;

  graph->UpdateNodeScopeBitFlags();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 3);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6};  // 所有节点
  EXPECT_EQ(allProcessedNodes, expectedNodes);

  auto k1BitFlags = k1->GetScopeBitFlags();
  EXPECT_TRUE(k1BitFlags.test(0));  // scope_A
  EXPECT_TRUE(k1BitFlags.test(1));  // scope_B

  auto k2BitFlags = k2->GetScopeBitFlags();
  EXPECT_TRUE(k2BitFlags.test(0));   // scope_A
  EXPECT_FALSE(k2BitFlags.test(1));  // scope_B
}

/**
 * @brief Scope与跨流依赖
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> Wait1(3) -> K2(4) -> ScopeEnd_A(5)]
 *   stream1: [K3(6) -> Notify1(7) -> K4(8)]
 *   Event1: Wait1(3) 等待 Notify1(7)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 所有节点 {1, 2, 3, 4, 5, 6, 7, 8} 融合到同一个scope
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_WithCrossStreamDependency) {
  auto *scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *wait1 = CreateWaitNode(3, 0, 7, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *scopeEndA = CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(6, 1, 7);
  auto *notify1 = CreateNotifyNode(7, 1, 100, 8);
  auto *k4 = CreateKernelNode(8, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5}, {6, 7, 8}});
  SetupEvent(100, 7, {3});

  graph->scopeNameToIdx["scope_A"] = 0;

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6, 7, 8};  // 所有节点
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief 超过最大Scope数量限制
 *
 * 图结构:
 *   创建超过MAX_SCOPE_NUM(64)个scope定义
 *   stream0: [K1(1) -> ScopeBegin_64(2) -> K2(3)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 超过限制的scope不生效
 *   - 所有节点 {1, 2, 3} 被处理
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_ExceedMaxScopeNumLimit) {
  for (uint32_t i = 0; i < MAX_SCOPE_NUM + 1; ++i) {
    graph->scopeNameToIdx["scope_" + std::to_string(i)] = i;
  }

  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *scopeBegin = CreateScopeBeginNode(2, 0, "scope_64", 3);
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3};  // 所有节点（包括scopeBegin）
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief Scope内包含不可融合节点
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> UF1(3, unfusible) -> K2(4) -> ScopeEnd_A(5)]
 *
 * 预期结果:
 *   - 生成2个scope: {ScopeBegin_A(1), K1(2)}, {K2(4), ScopeEnd_A(5)}
 *   - UF1作为切分边界
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_WithUnfusibleNodes) {
  auto *scopeBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *unfusibleK = CreateUnfusibleKernelNode(3, 0, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *scopeEndA = CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 2);
  EXPECT_EQ(scopeInfos[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(scopeInfos[0].nodes_[1]->GetNodeId(), 2);

  EXPECT_EQ(scopeInfos[1].nodes_.size(), 2);
  EXPECT_EQ(scopeInfos[1].nodes_[0]->GetNodeId(), 4);
  EXPECT_EQ(scopeInfos[1].nodes_[1]->GetNodeId(), 5);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {1, 2, 4, 5};  // ScopeBegin_A, K1, K2, ScopeEnd_A
  EXPECT_EQ(allProcessedNodes, expectedNodes);

  EXPECT_TRUE(k1->IsFusible());
  EXPECT_FALSE(unfusibleK->IsFusible());
  EXPECT_TRUE(k2->IsFusible());
}

/**
 * @brief Scope begin/end不成对 - 只有begin
 *
 * 图结构:
 *   stream0: [K1(1) -> ScopeBegin_A(2) -> K2(3) -> K3(4)]
 *   scope_A缺少ScopeEnd节点
 *
 * 预期结果:
 *   - UpdateNodeScopeBitFlags检测到未关闭scope
 *   - Scope bit flag更新失败
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnpairedScopeBegin) {
  CreateKernelNode(1, 0, 2);
  CreateScopeBeginNode(2, 0, "scope_A", 3);
  CreateKernelNode(3, 0, 4);
  CreateKernelNode(4, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4}});

  graph->scopeNameToIdx["scope_A"] = 0;

  EXPECT_FALSE(graph->UpdateNodeScopeBitFlags());
}

/**
 * @brief Scope begin/end不成对 - 只有end
 *
 * 图结构:
 *   stream0: [K1(1) -> ScopeEnd_A(2) -> K2(3) -> K3(4)]
 *   scope_A缺少ScopeBegin节点
 *
 * 预期结果:
 *   - UpdateNodeScopeBitFlags检测到没有匹配begin的scope end
 *   - Scope bit flag更新失败
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnpairedScopeEnd) {
  CreateKernelNode(1, 0, 2);
  CreateScopeEndNode(2, 0, "scope_A", 3);
  CreateKernelNode(3, 0, 4);
  CreateKernelNode(4, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4}});

  graph->scopeNameToIdx["scope_A"] = 0;

  EXPECT_FALSE(graph->UpdateNodeScopeBitFlags());
}

/**
 * @brief 重复的Scope begin
 *
 * 图结构:
 *   stream0: [ScopeBegin_A(1) -> K1(2) -> ScopeBegin_A(3) -> K2(4) -> ScopeEnd_A(5)]
 *   scope_A有两个ScopeBegin节点
 *
 * 预期结果:
 *   - UpdateNodeScopeBitFlags检测到未关闭的scope begin
 *   - Scope bit flag更新失败
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_DuplicateScopeBegin) {
  CreateScopeBeginNode(1, 0, "scope_A", 2);
  CreateKernelNode(2, 0, 3);
  CreateScopeBeginNode(3, 0, "scope_A", 4);
  CreateKernelNode(4, 0, 5);
  CreateScopeEndNode(5, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5}});

  graph->scopeNameToIdx["scope_A"] = 0;

  EXPECT_FALSE(graph->UpdateNodeScopeBitFlags());
}

/**
 * @brief 多个Scope的ScopeBitFlags验证
 *
 * 图结构:
 *   stream0: [K1(1) -> ScopeBegin_A(2) -> K2(3) -> ScopeEnd_A(4) -> K3(5) -> ScopeBegin_B(6) -> K4(7) -> ScopeEnd_B(8)
 * -> K5(9)]
 *
 * 预期结果:
 *   - K1(1): scopeBitFlags=0 (不在任何scope内)
 *   - Scope_A节点: {ScopeBegin_A(2), K2(3), ScopeEnd_A(4)} scopeBitFlags第0位为1
 *   - K3(5): scopeBitFlags=0 (不在任何scope内)
 *   - Scope_B节点: {ScopeBegin_B(6), K4(7), ScopeEnd_B(8)} scopeBitFlags第1位为1
 *   - K5(9): scopeBitFlags=0 (不在任何scope内)
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_MultipleScopeBitFlags) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *scopeBeginA = CreateScopeBeginNode(2, 0, "scope_A", 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *scopeEndA = CreateScopeEndNode(4, 0, "scope_A", 5);
  auto *k3 = CreateKernelNode(5, 0, 6);
  auto *scopeBeginB = CreateScopeBeginNode(6, 0, "scope_B", 7);
  auto *k4 = CreateKernelNode(7, 0, 8);
  auto *scopeEndB = CreateScopeEndNode(8, 0, "scope_B", 9);
  auto *k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

  graph->scopeNameToIdx["scope_A"] = 0;
  graph->scopeNameToIdx["scope_B"] = 1;

  EXPECT_TRUE(graph->UpdateNodeScopeBitFlags());

  EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);

  EXPECT_TRUE(scopeBeginA->GetScopeBitFlags().test(0));
  EXPECT_TRUE(k2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeEndA->GetScopeBitFlags().test(0));

  EXPECT_EQ(k3->GetScopeBitFlags().count(), 0);

  EXPECT_TRUE(scopeBeginB->GetScopeBitFlags().test(1));
  EXPECT_TRUE(k4->GetScopeBitFlags().test(1));
  EXPECT_TRUE(scopeEndB->GetScopeBitFlags().test(1));

  EXPECT_EQ(k5->GetScopeBitFlags().count(), 0);

  EXPECT_FALSE(scopeBeginA->GetScopeBitFlags().test(1));
  EXPECT_FALSE(k2->GetScopeBitFlags().test(1));
  EXPECT_FALSE(scopeEndA->GetScopeBitFlags().test(1));

  EXPECT_FALSE(scopeBeginB->GetScopeBitFlags().test(0));
  EXPECT_FALSE(k4->GetScopeBitFlags().test(0));
  EXPECT_FALSE(scopeEndB->GetScopeBitFlags().test(0));
}

/**
 * @brief Unfusible Scope内部节点标记为不可融合，外部节点可融合
 *
 * 图结构:
 *   stream0: [K1(1) -> UnfusibleBegin(2) -> K2(3) -> K3(4) -> UnfusibleEnd(5) -> K4(6)]
 *   UnfusibleBegin/End为不可融合Scope节点
 *
 * 预期结果:
 *   - K1/K4可融合（在scope外）
 *   - UnfusibleBegin/End可融合（Scope节点始终可融合）
 *   - K2/K3不可融合（在unfusible scope内）
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_PureUnfusibleScope) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *scopeBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *k3 = CreateKernelNode(4, 0, 5);
  auto *scopeEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
  auto *k4 = CreateKernelNode(6, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6}});

  EXPECT_TRUE(graph->scopeNameToIdx.empty());

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(k1->IsFusible()) << "K1 should be fusible (outside scope)";
  EXPECT_EQ(k1->GetScopeBitFlags().count(), 0) << "K1 should have no scope flags";

  EXPECT_TRUE(scopeBegin->IsFusible()) << "ScopeBegin should be fusible (scope nodes are always fusible)";

  EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible (inside unfusible scope)";
  EXPECT_EQ(k2->GetScopeBitFlags().count(), 0) << "K2 should have no scope flags (unfusible scope)";

  EXPECT_FALSE(k3->IsFusible()) << "K3 should be unfusible (inside unfusible scope)";
  EXPECT_EQ(k3->GetScopeBitFlags().count(), 0) << "K3 should have no scope flags (unfusible scope)";

  EXPECT_TRUE(scopeEnd->IsFusible()) << "ScopeEnd should be fusible (scope nodes are always fusible)";

  EXPECT_TRUE(k4->IsFusible()) << "K4 should be fusible (outside scope)";
  EXPECT_EQ(k4->GetScopeBitFlags().count(), 0) << "K4 should have no scope flags";
}

/**
 * @brief 多个不可融合Scope
 *
 * 图结构:
 *   stream0: [K1(1) -> UnfusibleBegin1(2) -> K2(3) -> UnfusibleEnd1(4) -> K3(5) -> UnfusibleBegin2(6) -> K4(7) ->
 * UnfusibleEnd2(8) -> K5(9)]
 *
 * 预期结果:
 *   - K1/K3/K5可融合（在scope外）
 *   - K2/K4不可融合（在unfusible scope内）
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_MultipleUnfusibleScopes) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *scopeBegin1 = CreateUnfusibleScopeBeginNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *scopeEnd1 = CreateUnfusibleScopeEndNode(4, 0, 5);
  auto *k3 = CreateKernelNode(5, 0, 6);
  auto *scopeBegin2 = CreateUnfusibleScopeBeginNode(6, 0, 7);
  auto *k4 = CreateKernelNode(7, 0, 8);
  auto *scopeEnd2 = CreateUnfusibleScopeEndNode(8, 0, 9);
  auto *k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(k1->IsFusible());
  EXPECT_TRUE(k3->IsFusible());
  EXPECT_TRUE(k5->IsFusible());

  EXPECT_FALSE(k2->IsFusible());
  EXPECT_FALSE(k4->IsFusible());
}

/**
 * @brief Fusible和Unfusible Scope混合
 *
 * 图结构:
 *   stream0: [FusibleBegin_A(1) -> K1(2) -> UnfusibleBegin(3) -> K2(4) -> UnfusibleEnd(5) -> K3(6) -> FusibleEnd_A(7)]
 *
 * 预期结果:
 *   - K1/K3可融合（在fusible scope_A中）
 *   - K2不可融合（unfusible覆盖fusible）
 *   - ScopeBitFlags包含scope_A标记
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_MixedFusibleUnfusible) {
  auto *fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(3, 0, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
  auto *k3 = CreateKernelNode(6, 0, 7);
  auto *fusibleEndA = CreateScopeEndNode(7, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(k1->IsFusible()) << "K1 should be fusible";
  EXPECT_TRUE(k1->GetScopeBitFlags().test(0)) << "K1 should have scope_A flag";
  EXPECT_EQ(k1->GetScopeBitFlags().count(), 1) << "K1 should have exactly 1 scope flag";

  EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible (unfusible scope overrides fusible)";
  EXPECT_TRUE(k2->GetScopeBitFlags().test(0)) << "K2 should have scope_A flag";
  EXPECT_EQ(k2->GetScopeBitFlags().count(), 1) << "K2 should have exactly 1 scope flag";

  EXPECT_TRUE(k3->IsFusible()) << "K3 should be fusible";
  EXPECT_TRUE(k3->GetScopeBitFlags().test(0)) << "K3 should have scope_A flag";
  EXPECT_EQ(k3->GetScopeBitFlags().count(), 1) << "K3 should have exactly 1 scope flag";
}

/**
 * @brief 嵌套Fusible和Unfusible Scope
 *
 * 图结构:
 *   stream0: [FusibleBegin_A(1) -> UnfusibleBegin(2) -> FusibleBegin_B(3) -> K1(4) -> FusibleEnd_B(5) -> K2(6) ->
 * UnfusibleEnd(7) -> K3(8) -> FusibleEnd_A(9)]
 *
 * 预期结果:
 *   - K1/K2不可融合（外层unfusible覆盖内层fusible）
 *   - K3可融合（回到fusible scope_A）
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_NestedFusibleUnfusible) {
  auto *fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
  auto *fusibleBeginB = CreateScopeBeginNode(3, 0, "scope_B", 4);
  auto *k1 = CreateKernelNode(4, 0, 5);
  auto *fusibleEndB = CreateScopeEndNode(5, 0, "scope_B", 6);
  auto *k2 = CreateKernelNode(6, 0, 7);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(7, 0, 8);
  auto *k3 = CreateKernelNode(8, 0, 9);
  auto *fusibleEndA = CreateScopeEndNode(9, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

  graph->scopeNameToIdx["scope_A"] = 0;
  graph->scopeNameToIdx["scope_B"] = 1;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_FALSE(k1->IsFusible()) << "K1 should be unfusible (unfusible outer scope overrides)";
  EXPECT_TRUE(k1->GetScopeBitFlags().test(0)) << "K1 should have scope_A flag";
  EXPECT_TRUE(k1->GetScopeBitFlags().test(1)) << "K1 should have scope_B flag";
  EXPECT_EQ(k1->GetScopeBitFlags().count(), 2) << "K1 should have 2 scope flags";

  EXPECT_FALSE(k2->IsFusible()) << "K2 should be unfusible";
  EXPECT_TRUE(k2->GetScopeBitFlags().test(0)) << "K2 should have scope_A flag";
  EXPECT_EQ(k2->GetScopeBitFlags().count(), 1) << "K2 should have 1 scope flag";

  EXPECT_TRUE(k3->IsFusible()) << "K3 should be fusible";
  EXPECT_TRUE(k3->GetScopeBitFlags().test(0)) << "K3 should have scope_A flag";
  EXPECT_EQ(k3->GetScopeBitFlags().count(), 1) << "K3 should have 1 scope flag";
}

/**
 * @brief Unfusible Scope跨越整个图 - Scope节点可融合，内部节点不可融合
 *
 * 图结构:
 *   stream0: [UnfusibleBegin(1) -> K1(2) -> K2(3) -> K3(4) -> UnfusibleEnd(5)]
 *
 * 预期结果:
 *   - Scope节点(UnfusibleBegin/End)可融合
 *   - 所有普通节点(K1/K2/K3)不可融合
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnfusibleSpanningWholeGraph) {
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(1, 0, 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *k3 = CreateKernelNode(4, 0, 5);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5}});

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(unfusibleBegin->IsFusible());
  EXPECT_TRUE(unfusibleEnd->IsFusible());
  EXPECT_FALSE(k1->IsFusible());
  EXPECT_FALSE(k2->IsFusible());
  EXPECT_FALSE(k3->IsFusible());
}

/**
 * @brief 重复的Unfusible Scope begin/end
 *
 * 图结构:
 *   stream0: [UnfusibleBegin1(1) -> K1(2) -> UnfusibleBegin2(3) -> K2(4) -> UnfusibleEnd1(5) -> K3(6) ->
 * UnfusibleEnd2(7)]
 *
 * 预期结果:
 *   - 所有 kernel 节点 {K1(2), K2(4), K3(6)} 不可融合
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_DuplicateUnfusibleScopeBeginEnd) {
  auto *unfusibleBegin1 = CreateUnfusibleScopeBeginNode(1, 0, 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *unfusibleBegin2 = CreateUnfusibleScopeBeginNode(3, 0, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *unfusibleEnd1 = CreateUnfusibleScopeEndNode(5, 0, 6);
  auto *k3 = CreateKernelNode(6, 0, 7);
  auto *unfusibleEnd2 = CreateUnfusibleScopeEndNode(7, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

  graph->UpdateNodeScopeBitFlags();

  EXPECT_FALSE(k1->IsFusible());
  EXPECT_FALSE(k2->IsFusible());
  EXPECT_FALSE(k3->IsFusible());
}

/**
 * @brief Unfusible Scope在Fusible Scope内部
 *
 * 图结构:
 *   stream0: [FusibleBegin_A(1) -> K1(2) -> UnfusibleBegin(3) -> K2(4) -> UnfusibleEnd(5) -> K3(6) -> FusibleEnd_A(7)]
 *
 * 预期结果:
 *   - K1/K3可融合（只在fusible scope中）
 *   - K2不可融合（在unfusible scope中）
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnfusibleInsideFusible) {
  auto *fusibleBeginA = CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(3, 0, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(5, 0, 6);
  auto *k3 = CreateKernelNode(6, 0, 7);
  auto *fusibleEndA = CreateScopeEndNode(7, 0, "scope_A", INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(k1->IsFusible());
  EXPECT_TRUE(k1->GetScopeBitFlags().test(0));

  EXPECT_FALSE(k2->IsFusible());
  EXPECT_TRUE(k2->GetScopeBitFlags().test(0));  // 仍在fusible scope_A中

  EXPECT_TRUE(k3->IsFusible());
  EXPECT_TRUE(k3->GetScopeBitFlags().test(0));
}

/**
 * @brief Unfusible和Fusible Scope并列
 *
 * 图结构:
 *   stream0: [K1(1) -> UnfusibleBegin(2) -> K2(3) -> UnfusibleEnd(4) -> K3(5) -> FusibleBegin_A(6) -> K4(7) ->
 * FusibleEnd_A(8) -> K5(9)]
 *
 * 预期结果:
 *   - K1/K3/K5不可融合（在命名scope外或unfusible scope内）
 *   - K4可融合（在fusible scope中）
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnfusibleFusibleSideBySide) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(4, 0, 5);
  auto *k3 = CreateKernelNode(5, 0, 6);
  auto *fusibleBeginA = CreateScopeBeginNode(6, 0, "scope_A", 7);
  auto *k4 = CreateKernelNode(7, 0, 8);
  auto *fusibleEndA = CreateScopeEndNode(8, 0, "scope_A", 9);
  auto *k5 = CreateKernelNode(9, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7, 8, 9}});

  graph->scopeNameToIdx["scope_A"] = 0;

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(k4->IsFusible());  // 在fusible scope中

  EXPECT_FALSE(k1->IsFusible());  // 在命名scope外
  EXPECT_FALSE(k2->IsFusible());  // 在unfusible scope中
  EXPECT_FALSE(k3->IsFusible());  // 在命名scope外
  EXPECT_FALSE(k5->IsFusible());  // 在命名scope外

  EXPECT_EQ(k1->GetScopeBitFlags().count(), 0);
  EXPECT_TRUE(k4->GetScopeBitFlags().test(0));
  EXPECT_EQ(k3->GetScopeBitFlags().count(), 0);
  EXPECT_EQ(k5->GetScopeBitFlags().count(), 0);
}

/**
 * @brief 未关闭的Unfusible Scope
 *
 * 图结构:
 *   stream0: [K1(1) -> UnfusibleBegin(2) -> K2(3) -> K3(4)]
 *   Unfusible Scope缺少End节点
 *
 * 预期结果:
 *   - UpdateNodeScopeBitFlags检测到未关闭scope
 *   - Scope bit flag更新失败
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_UnclosedUnfusibleScope) {
  CreateKernelNode(1, 0, 2);
  CreateUnfusibleScopeBeginNode(2, 0, 3);
  CreateKernelNode(3, 0, 4);
  CreateKernelNode(4, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3, 4}});

  EXPECT_FALSE(graph->UpdateNodeScopeBitFlags());
}

/**
 * @brief 只有Unfusible Scope节点 - 无普通节点
 *
 * 图结构:
 *   stream0: [UnfusibleBegin(1) -> UnfusibleEnd(2)]
 *
 * 预期结果:
 *   - Scope节点总是可融合
 */
TEST_F(SuperKernelScopeSplitterTest, Scope_OnlyUnfusibleScopeNodes) {
  auto *unfusibleBegin = CreateUnfusibleScopeBeginNode(1, 0, 2);
  auto *unfusibleEnd = CreateUnfusibleScopeEndNode(2, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2}});

  graph->UpdateNodeScopeBitFlags();

  EXPECT_TRUE(unfusibleBegin->IsFusible());
  EXPECT_TRUE(unfusibleEnd->IsFusible());
}

// ==================== ScopeBitFlags 相关测试用例 ====================

/**
 * @brief 不同ScopeBitFlags的节点分配到不同Scope
 *
 * 图结构:
 *   stream0: K1(scope=0, id=1) -> K2(scope=1, id=2)
 *   stream1: K3(scope=0, id=3)
 *
 * 预期结果:
 *   - K1/K3分配到Scope0 (相同ScopeBitFlags)
 *   - K2分配到Scope1
 *   - 生成2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_DifferentFlagsSplit) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  std::bitset<MAX_SCOPE_NUM> flags0;
  flags0.set(0);
  k1->SetScopeBitFlags(flags0);

  auto *k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
  std::bitset<MAX_SCOPE_NUM> flags1;
  flags1.set(1);
  k2->SetScopeBitFlags(flags1);

  auto *k3 = CreateKernelNode(3, 1, INVALID_TASK_ID);
  k3->SetScopeBitFlags(flags0);

  SetupStreams({{1, 2}, {3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 2);
  EXPECT_EQ(scopeInfos[1].nodes_.size(), 1);

  EXPECT_EQ(scopeInfos[0].scopeBitFlags_, flags0);
  EXPECT_EQ(scopeInfos[1].scopeBitFlags_, flags1);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_UnfusibleNodeOutsideScopeDoesNotSetBreakReason) {
  auto *inScopeKernel = CreateKernelNode(1, 0, 2);
  std::bitset<MAX_SCOPE_NUM> scopeFlags;
  scopeFlags.set(0);
  inScopeKernel->SetScopeBitFlags(scopeFlags);

  CreateUnfusibleKernelNode(2, 0, INVALID_TASK_ID);
  SetupStreams({{1, 2}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  ASSERT_TRUE(splitter.SplitGraph());
  ASSERT_EQ(splitter.GetScopeInfos().size(), 1U);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetBreakInfo().GetReason(), ScopeBreakReason::NONE);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_UnfusibleNodeInsideScopeSetsBreakReason) {
  auto *inScopeKernel = CreateKernelNode(1, 0, 2);
  auto *unfusibleKernel = CreateUnfusibleKernelNode(2, 0, INVALID_TASK_ID);
  std::bitset<MAX_SCOPE_NUM> scopeFlags;
  scopeFlags.set(0);
  inScopeKernel->SetScopeBitFlags(scopeFlags);
  unfusibleKernel->SetScopeBitFlags(scopeFlags);
  SetupStreams({{1, 2}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  ASSERT_TRUE(splitter.SplitGraph());
  ASSERT_EQ(splitter.GetScopeInfos().size(), 1U);
  const auto &breakInfo = splitter.GetScopeInfos()[0].GetBreakInfo();
  EXPECT_EQ(breakInfo.GetReason(), ScopeBreakReason::UNFUSIBLE_NODE);
  EXPECT_EQ(breakInfo.GetTriggerNodeId(), 2U);
}

/**
 * @brief 无ScopeBitFlags节点只能和无ScopeBitFlags节点融合
 *
 * 图结构:
 *   stream0: K1(scope=0, id=1) -> K2(no flags, id=2) -> K3(scope=1, id=3)
 *
 * 预期结果:
 *   - K1在Scope0
 *   - K2在独立Scope（无flags）
 *   - K3在Scope1
 *   - 生成3个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_NoFlagsOnlyFuseWithNoFlags) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  std::bitset<MAX_SCOPE_NUM> flags0;
  flags0.set(0);
  k1->SetScopeBitFlags(flags0);

  auto *k2 = CreateKernelNode(2, 0, 3);

  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  std::bitset<MAX_SCOPE_NUM> flags1;
  flags1.set(1);
  k3->SetScopeBitFlags(flags1);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 3);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 1);
  EXPECT_EQ(scopeInfos[1].nodes_.size(), 1);
  EXPECT_EQ(scopeInfos[2].nodes_.size(), 1);
  EXPECT_EQ(scopeInfos[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(scopeInfos[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(scopeInfos[2].nodes_[0]->GetNodeId(), 3);
}

/**
 * @brief 多个ScopeBitFlags位同时设置 - 不同flags分配到不同scope
 *
 * 图结构:
 *   stream0: K1(flags=011, id=1) -> K2(flags=011, id=2) -> K3(flags=001, id=3)
 *   stream1: K4(flags=011, id=4) -> K5(flags=111, id=5)
 *
 * 预期结果:
 *   - K1/K2/K4分配到Scope0 (flags=011)
 *   - K3分配到Scope1 (flags=001)
 *   - K5分配到Scope2 (flags=111)
 *   - 生成3个scope，每个scope的scopeBitFlags正确
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_MultipleFlagsBits) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  std::bitset<MAX_SCOPE_NUM> flags011;
  flags011.set(0);
  flags011.set(1);
  k1->SetScopeBitFlags(flags011);

  auto *k2 = CreateKernelNode(2, 0, 3);
  k2->SetScopeBitFlags(flags011);

  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  std::bitset<MAX_SCOPE_NUM> flags001;
  flags001.set(0);
  k3->SetScopeBitFlags(flags001);

  auto *k4 = CreateKernelNode(4, 1, 5);
  k4->SetScopeBitFlags(flags011);

  auto *k5 = CreateKernelNode(5, 1, INVALID_TASK_ID);
  std::bitset<MAX_SCOPE_NUM> flags111;
  flags111.set(0);
  flags111.set(1);
  flags111.set(2);
  k5->SetScopeBitFlags(flags111);

  SetupStreams({{1, 2, 3}, {4, 5}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 3);

  size_t scope011Count = 0;
  size_t scope001Count = 0;
  size_t scope111Count = 0;
  for (const auto &scope : scopeInfos) {
    if (scope.scopeBitFlags_ == flags011) {
      scope011Count++;
      EXPECT_EQ(scope.nodes_.size(), 3);
    } else if (scope.scopeBitFlags_ == flags001) {
      scope001Count++;
      EXPECT_EQ(scope.nodes_.size(), 1);
    } else if (scope.scopeBitFlags_ == flags111) {
      scope111Count++;
      EXPECT_EQ(scope.nodes_.size(), 1);
    }
  }
  EXPECT_EQ(scope011Count, 1);
  EXPECT_EQ(scope001Count, 1);
  EXPECT_EQ(scope111Count, 1);
}

/**
 * @brief 复杂场景 - 多流多ScopeBitFlags
 *
 * 图结构:
 *   stream0-3: 各8个节点，4种ScopeBitFlags，包含事件同步
 *
 * 预期结果:
 *   - 生成2个scope
 *   - 每个scope16个节点
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_ComplexMultiStream) {
  std::bitset<MAX_SCOPE_NUM> scope0;
  scope0.set(0);
  std::bitset<MAX_SCOPE_NUM> scope1;
  scope1.set(1);

  for (int i = 1; i <= 8; ++i) {
    auto *node = CreateKernelNode(i, 0, i < 8 ? i + 1 : INVALID_TASK_ID);
    node->SetScopeBitFlags(scope0);
  }

  for (int i = 9; i <= 16; ++i) {
    auto *node = CreateKernelNode(i, 1, i < 16 ? i + 1 : INVALID_TASK_ID);
    node->SetScopeBitFlags(scope1);
  }

  for (int i = 17; i <= 24; ++i) {
    if (i == 19) {
      auto *node = CreateNotifyNode(19, 2, 100, 20);  // eventId=100
      node->SetScopeBitFlags(scope0);
    } else {
      auto *node = CreateKernelNode(i, 2, i < 24 ? i + 1 : INVALID_TASK_ID);
      node->SetScopeBitFlags(scope0);
    }
  }

  for (int i = 25; i <= 32; ++i) {
    if (i == 27) {
      auto *node = CreateWaitNode(27, 3, 19, 28);  // 等待 Notify1(K19)
      node->SetScopeBitFlags(scope1);
    } else {
      auto *node = CreateKernelNode(i, 3, i < 32 ? i + 1 : INVALID_TASK_ID);
      node->SetScopeBitFlags(scope1);
    }
  }

  SetupStreams({
      {1, 2, 3, 4, 5, 6, 7, 8},          // Stream 0
      {9, 10, 11, 12, 13, 14, 15, 16},   // Stream 1
      {17, 18, 19, 20, 21, 22, 23, 24},  // Stream 2
      {25, 26, 27, 28, 29, 30, 31, 32}   // Stream 3
  });

  SetupEvent(100, 19, {27});  // Event 100: Notify1(19) → Wait1(27)

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);

  int scope0Count = 0, scope1Count = 0;
  for (const auto &scope : scopeInfos) {
    if (scope.scopeBitFlags_ == scope0)
      scope0Count++;
    else if (scope.scopeBitFlags_ == scope1)
      scope1Count++;
  }
  EXPECT_EQ(scope0Count, 1);
  EXPECT_EQ(scope1Count, 1);

  size_t totalNodes = 0;
  for (const auto &scope : scopeInfos) {
    totalNodes += scope.nodes_.size();
  }
  EXPECT_EQ(totalNodes, 32);

  EXPECT_EQ(scopeInfos[0].nodes_.size(), 16);
  EXPECT_EQ(scopeInfos[1].nodes_.size(), 16);
}

/**
 * @brief ScopeBitFlags与事件同步交互
 *
 * 图结构:
 *   stream0: K1(scope=0, id=1) -> Wait1(scope=0, id=2) -> K3(scope=1, id=3)
 *   stream1: K4(scope=0, id=4) -> Notify1(scope=0, id=5) -> K6(scope=1, id=6)
 *   Event1: Notify1(5) -> Wait1(2)
 *
 * 预期结果:
 *   - Wait/Notify因scopeBitFlags匹配加入Scope0
 *   - K3/K6因scopeBitFlags不匹配加入Scope1
 *   - 生成2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScopeBitFlags_WithEventSynchronization) {
  std::bitset<MAX_SCOPE_NUM> scope0;
  scope0.set(0);
  std::bitset<MAX_SCOPE_NUM> scope1;
  scope1.set(1);

  auto *k1 = CreateKernelNode(1, 0, 2);
  k1->SetScopeBitFlags(scope0);
  auto *wait1 = CreateWaitNode(2, 0, 5, 3);  // 等待 Notify1(id=5)
  wait1->SetScopeBitFlags(scope0);
  auto *k3 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  k3->SetScopeBitFlags(scope1);

  auto *k4 = CreateKernelNode(4, 1, 5);
  k4->SetScopeBitFlags(scope0);
  auto *notify1 = CreateNotifyNode(5, 1, 100, 6);  // eventId=100
  notify1->SetScopeBitFlags(scope0);
  auto *k6 = CreateKernelNode(6, 1, INVALID_TASK_ID);
  k6->SetScopeBitFlags(scope1);

  SetupStreams({{1, 2, 3}, {4, 5, 6}});
  SetupEvent(100, 5, {2});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 2);

  EXPECT_EQ(scopeInfos[0].scopeBitFlags_, scope0);
  EXPECT_EQ(scopeInfos[0].nodes_.size(), 4);

  EXPECT_EQ(scopeInfos[1].scopeBitFlags_, scope1);
  EXPECT_EQ(scopeInfos[1].nodes_.size(), 2);
}

/**
 * @brief 跨流Wait-Notify依赖处理 - suspend/resume机制验证
 *
 * 图结构:
 *   stream0: [K1(1) -> Wait1(2) -> K2(3)]
 *   stream1: [K3(4) -> Notify1(5) -> K4(6)]
 *   Event1: Wait1(2) 等待 Notify1(5)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - InitialScopeSplitPass处理Wait的suspend/resume
 *   - Wait1(2)先于Notify1(5)处理时触发stream0暂停
 *   - Notify1(5)处理时恢复stream0
 *   - 所有节点 {1, 2, 3, 4, 5, 6} 融合到一个scope
 */
TEST_F(SuperKernelScopeSplitterTest, Sync_CrossStreamWaitNotifySuspendResume) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *wait1 = CreateWaitNode(2, 0, 5, 3);  // 等待 Notify1(id=5)
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *k3 = CreateKernelNode(4, 1, 5);
  auto *notify1 = CreateNotifyNode(5, 1, 100, 6);
  auto *k4 = CreateKernelNode(6, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4, 5, 6}});
  SetupEvent(100, 5, {2});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  EXPECT_EQ(scopeInfos.size(), 1);

  std::set<uint64_t> allProcessedNodes;
  for (const auto *node : scopeInfos[0].nodes_) {
    allProcessedNodes.insert(node->GetNodeId());
  }
  std::set<uint64_t> expectedNodes = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief DeadlockRefinePass资源不足导致死锁切分
 *
 * 场景描述:
 *   设备资源: cube=25, vec=32 (强制设置)
 *   stream A: K1(cube=12,id=1) -> Wait1(id=2) -> K2(cube=15,id=3)
 *   stream B: K3(cube=12,id=4,unfusible) -> Notify1(id=5,unfusible)
 *   Event100: Wait1等待Notify1
 *
 * 执行分析:
 *   1. InitialScopeSplitPass: Notify1(unfusible)加入visitedNotifies，不进scope
 *   2. Wait1发现Notify1已visited，融合进scope
 *   3. scope节点: K1 -> Wait1 -> K2
 *   4. DeadlockRefinePass检查:
 *      - 加入K1后，候选SK核数为12C
 *      - GetWaitNodeFusibleStatus(Wait1):
 *        -> HasDeadlock(Notify1) -> CheckKernelNodeDeadlock(K3)
 *        -> HasEnoughCores(K3, false): available={25-12=13}, 12<=13
 *        -> 记录外部依赖占用12C，Wait1可以继续融合
 *      - 加入K2后，候选SK核数增长为15C
 *        -> SK available={25-12=13}, 15>13
 *        -> K2触发资源死锁风险
 *   5. FindDeadlockInScope将K2记录为触发节点，将Wait1记录为切分点
 *   6. 在Wait1处切分（Wait是最近的Wait节点）
 *
 * 预期结果:
 *   - 检测到资源死锁，切分为2个scope
 *   - scope0: {K1(1)}，核心需求12
 *   - scope1: {K2(3)}，核心需求15
 *   - K3(12)可与scope0(12)并发: 12+12=24 < 25
 */
TEST_F(SuperKernelScopeSplitterTest, DeadlockRefine_ResourceDeadlockSplitAtWait) {
  // 强制设置设备核心数为25（匹配用户场景）
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;

  auto *k1 = CreateKernelNode(1, 0, 2);
  k1->nodeInfos.kernelInfos.cubeNum = 12;
  k1->nodeInfos.kernelInfos.numBlocks = 12;

  auto *wait1 = CreateWaitNode(2, 0, 5, 3);

  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  k2->nodeInfos.kernelInfos.cubeNum = 15;
  k2->nodeInfos.kernelInfos.numBlocks = 15;

  auto *k3 = CreateUnfusibleKernelNode(4, 1, 5);
  k3->nodeInfos.kernelInfos.cubeNum = 12;
  k3->nodeInfos.kernelInfos.numBlocks = 12;

  auto *notify1 = CreateNotifyNode(5, 1, 100, INVALID_TASK_ID);
  notify1->isFusible = false;

  SetupStreams({{1, 2, 3}, {4, 5}});
  SetupEvent(100, 5, {2});

  graph->BuildEventNodeAssociations();

  // 验证 Notify1 的前置节点关系是否正确设置
  auto *notifyNodeInGraph = graph->GetNodeById(5);
  ASSERT_NE(notifyNodeInGraph, nullptr);
  uint64_t notifyPreNodeId = notifyNodeInGraph->GetPreNodeId();
  EXPECT_EQ(notifyPreNodeId, 4) << "Notify1(5) preNodeId should be K3(4), but got " << notifyPreNodeId;

  // 验证 K3 的核需求是否正确设置
  auto *k3NodeInGraph = graph->GetNodeById(4);
  ASSERT_NE(k3NodeInGraph, nullptr);
  uint64_t k3CubeNum = k3NodeInGraph->GetCubeNum();
  EXPECT_EQ(k3CubeNum, 12) << "K3(4) cubeNum should be 12, but got " << k3CubeNum;

  // 关键：在 splitter 构造后设置设备核心数（LockDetector 构造时会调用 Init->GetDeviceCores）
  SuperKernelScopeSplitter splitter(*graph, *opts);
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  ASSERT_EQ(scopeInfos.size(), 2U);
  ASSERT_EQ(scopeInfos[0].GetNodes().size(), 1U);
  ASSERT_EQ(scopeInfos[1].GetNodes().size(), 1U);
  EXPECT_EQ(scopeInfos[0].GetNodes()[0]->GetNodeId(), 1U);
  EXPECT_EQ(scopeInfos[1].GetNodes()[0]->GetNodeId(), 3U);
  EXPECT_EQ(scopeInfos[0].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopeInfos[0].GetScopeCoreInfo().numBlocks, 12U);
  EXPECT_EQ(scopeInfos[1].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopeInfos[1].GetScopeCoreInfo().numBlocks, 15U);
  EXPECT_EQ(wait1->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(k2->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  EXPECT_EQ(scopeInfos[0].GetBreakInfo().GetReason(), ScopeBreakReason::DEADLOCK_DETECTED);
  EXPECT_EQ(scopeInfos[0].GetBreakInfo().GetTriggerNodeId(), 3U);

  // 恢复默认核心数
  LockDetector::deviceRealCubeNum = 32;
  LockDetector::deviceRealVecNum = 32;
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefine_SplitUsesCandidateSnapshotBeforeWait) {
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;

  auto *k1 = CreateKernelNode(1, 0, 2);
  k1->nodeInfos.kernelInfos.cubeNum = 5;
  k1->nodeInfos.kernelInfos.numBlocks = 5;
  auto *wait = CreateWaitNode(2, 0, 6, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  k2->nodeInfos.kernelInfos.cubeNum = 10;
  k2->nodeInfos.kernelInfos.numBlocks = 10;
  auto *k3 = CreateKernelNode(4, 0, INVALID_TASK_ID);
  k3->nodeInfos.kernelInfos.cubeNum = 15;
  k3->nodeInfos.kernelInfos.numBlocks = 15;

  auto *dependency = CreateUnfusibleKernelNode(5, 1, 6);
  dependency->nodeInfos.kernelInfos.cubeNum = 12;
  dependency->nodeInfos.kernelInfos.numBlocks = 12;
  auto *notify = CreateNotifyNode(6, 1, 100, INVALID_TASK_ID);
  notify->isFusible = false;

  SetupStreams({{1, 2, 3, 4}, {5, 6}});
  SetupEvent(100, 6, {2});
  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;
  ASSERT_TRUE(splitter.SplitGraph());

  const auto &scopes = splitter.GetScopeInfos();
  ASSERT_EQ(scopes.size(), 2U);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 5U);
  EXPECT_EQ(scopes[1].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopes[1].GetScopeCoreInfo().numBlocks, 15U);
  EXPECT_EQ(wait->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(k2->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(k3->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  EXPECT_EQ(scopes[0].GetBreakInfo().GetTriggerNodeId(), 4U);

  LockDetector::deviceRealCubeNum = 32;
  LockDetector::deviceRealVecNum = 32;
}

/**
 * @brief EventOnlyStreamRemovePass - Pipeline单独Event
 *
 * 图结构:
 *   stream0: [Notify1(1)] (只有Event)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 无可融合Kernel，生成0个scope
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_FullPipelineOnlyEventNodes) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, INVALID_TASK_ID);

  SetupStreams({{1}});
  SetupEvent(0x100, 1, {});

  graph->BuildEventNodeAssociations();

  EXPECT_TRUE(notify1->IsFusible());
  EXPECT_TRUE(notify1->GetCorrespondingWaitNodeIds().empty());

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);

  const auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 0);
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefinePassWithoutValueBreakerDropsUnpairedMemoryWaitOnSplit) {
  auto *wait1 = CreateWaitNode(1, 0, INVALID_TASK_ID, 2);
  auto *k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
  wait1->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);
  SuperKernelScopeSplitter splitter(*graph, *opts);
  auto *deadlockPass = dynamic_cast<DeadlockRefinePass *>(splitter.passes_[2].get());
  ASSERT_NE(deadlockPass, nullptr);

  SuperKernelScopeInfo scope;
  scope.AddNode(wait1);
  scope.AddNode(k2);
  scope.SetScopeStreamInfos({
      ScopeStreamInfo{0, 1, 2, 2},
  });
  scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 1});

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.emplace_back(std::move(scope));

  ASSERT_TRUE(deadlockPass->Run(scopes));
  ASSERT_EQ(scopes.size(), 1);

  std::set<uint64_t> actualNodes;
  for (const auto *node : scopes[0].GetNodes()) {
    actualNodes.insert(node->GetNodeId());
  }
  EXPECT_EQ(actualNodes, (std::set<uint64_t>{2}));
  EXPECT_EQ(wait1->GetScopeId(), INVALID_SCOPE_ID);
  EXPECT_EQ(k2->GetScopeId(), scopes[0].GetScopeId());
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefinePassValueBreakerBypassKeepsUnpairedMemoryWait) {
  auto *wait1 = CreateWaitNode(1, 0, INVALID_TASK_ID, 2);
  auto *k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
  wait1->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1234);

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.valueBreakerBypass = ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT;
  opts->SetOptOptionValue(&option);
  SuperKernelScopeSplitter splitter(*graph, *opts);
  auto *deadlockPass = dynamic_cast<DeadlockRefinePass *>(splitter.passes_[2].get());
  ASSERT_NE(deadlockPass, nullptr);

  SuperKernelScopeInfo scope;
  scope.AddNode(wait1);
  scope.AddNode(k2);
  scope.SetScopeStreamInfos({
      ScopeStreamInfo{0, 1, 2, 2},
  });
  scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 1});

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.emplace_back(std::move(scope));

  ASSERT_TRUE(deadlockPass->Run(scopes));
  ASSERT_EQ(scopes.size(), 1);

  std::set<uint64_t> actualNodes;
  for (const auto *node : scopes[0].GetNodes()) {
    actualNodes.insert(node->GetNodeId());
  }
  EXPECT_EQ(actualNodes, (std::set<uint64_t>{1, 2}));
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefinePassStoresScopeCoreInfoAfterCheck) {
  auto *kernel = CreateKernelNode(1, 0, INVALID_TASK_ID);
  SuperKernelScopeSplitter splitter(*graph, *opts);
  auto *deadlockPass = dynamic_cast<DeadlockRefinePass *>(splitter.passes_[2].get());
  ASSERT_NE(deadlockPass, nullptr);

  SuperKernelScopeInfo scope;
  scope.AddNode(kernel);
  scope.SetScopeStreamInfos({ScopeStreamInfo{0, 1, 1, 1}});
  std::vector<SuperKernelScopeInfo> scopes;
  scopes.emplace_back(std::move(scope));

  ASSERT_TRUE(deadlockPass->Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 1U);
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefinePassRefreshesStaleScopeCoreInfoAfterCheck) {
  auto *kernel = CreateKernelNode(1, 0, INVALID_TASK_ID);
  SuperKernelScopeSplitter splitter(*graph, *opts);
  auto *deadlockPass = dynamic_cast<DeadlockRefinePass *>(splitter.passes_[2].get());
  ASSERT_NE(deadlockPass, nullptr);

  SuperKernelScopeInfo scope;
  scope.AddNode(kernel);
  scope.SetScopeStreamInfos({ScopeStreamInfo{0, 1, 1, 1}});
  scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 2});
  std::vector<SuperKernelScopeInfo> scopes;
  scopes.emplace_back(std::move(scope));

  ASSERT_TRUE(deadlockPass->Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 1U);
  EXPECT_EQ(kernel->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SuperKernelScopeSplitterTest, DeadlockRefinePassClearsStaleCoreInfoWhenUnavailable) {
  auto *kernel1 = CreateScheModeKernelNode(1, 0, 4, 0, true, 2);
  auto *kernel2 = CreateScheModeKernelNode(2, 0, 8, 0);
  SetupStreams({{1, 2}});

  SuperKernelScopeInfo scope;
  scope.AddNode(kernel1);
  scope.AddNode(kernel2);
  scope.SetScopeStreamInfos({ScopeStreamInfo{0, 1, 2, 2}});
  scope.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 4});
  std::vector<SuperKernelScopeInfo> scopes;
  scopes.emplace_back(std::move(scope));

  ASSERT_TRUE(RunDeadlockRefine(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  VerifyScope(scopes[0], {1, 2});
  EXPECT_FALSE(scopes[0].GetScopeCoreInfo().IsValid());
  EXPECT_EQ(kernel1->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(kernel2->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SuperKernelScopeSplitterTest, FinalizeScopeCoreInfoRejectsConflictingScheModeRequirements) {
  auto *kernel1 = CreateScheModeKernelNode(1, 0, 4, 0, true, 2);
  auto *kernel2 = CreateScheModeKernelNode(2, 0, 8, 0);
  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({kernel1, kernel2}));

  EXPECT_FALSE(FinalizeScopeCoreInfo(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  EXPECT_FALSE(scopes[0].GetScopeCoreInfo().IsValid());
}

/**
 * @brief 孤立Notify节点(无对应Wait)保持当前融合状态
 *
 * 图结构:
 *   stream0: [Notify1(1, eventId=100) -> K1(2)]
 *   Event100: Notify1(1) 无对应Wait节点(空等待列表)
 *
 * 预期结果:
 *   - Notify1保持当前融合状态
 *   - Notify1无对应Wait节点ID
 *   - SplitGraph返回true
 *   - 生成1个scope包含Notify1(1)和K1(2)
 */
TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_OrphanNotifyMarkedUnfusible) {
  auto *notify1 = CreateNotifyNode(1, 0, 100, 2);
  CreateKernelNode(2, 0, INVALID_TASK_ID);

  SetupStreams({{1, 2}});
  SetupEvent(100, 1, {});

  graph->BuildEventNodeAssociations();

  EXPECT_TRUE(notify1->IsFusible());
  EXPECT_TRUE(notify1->GetCorrespondingWaitNodeIds().empty());

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : splitter.GetScopeInfos()) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }

  std::set<uint64_t> expectedNodes = {1, 2};
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief ResumeSuspendedWaitStreams错误处理 - Wait节点不存在
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2)]
 *   stream1: [Notify1(3,unfusible)]
 *   Event1: Notify1对应不存在的Wait(id=999)
 *
 * 预期结果:
 *   - ResumeSuspendedWaitStreams发现Wait不存在
 *   - SplitGraph返回false
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_WaitNodeNotFound) {
  auto *wait1 = CreateWaitNode(1, 0, 3, 2);
  wait1->isFusible = false;  // 标记为不可融合，触发 suspend/resume 逻辑
  auto *k1 = CreateKernelNode(2, 0, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(3, 1, 100, INVALID_TASK_ID);
  notify1->isFusible = false;  // 标记为不可融合

  SetupStreams({{1, 2}, {3}});

  SetupEvent(100, 3, {999});

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_FALSE(result);
}

/**
 * @brief 验证SkipUnfusibleNodesForStream跳过不可融合节点后正确处理可融合节点
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2,unfusible) -> K2(3,fusible)]
 *   stream1: [Notify1(4,unfusible)]
 *   Event1: Wait1等待Notify1
 *
 * 预期结果:
 *   - SkipUnfusibleNodesForStream跳过Wait和K1
 *   - SplitGraph返回true
 *   - 生成scope包含K2(3)
 */
TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_SkipUnfusibleNodesFailure) {
  auto *wait1 = CreateWaitNode(1, 0, 4, 2);
  wait1->isFusible = false;
  auto *k1 = CreateUnfusibleKernelNode(2, 0, 3);  // 不可融合
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(4, 1, 100, INVALID_TASK_ID);
  notify1->isFusible = false;

  SetupStreams({{1, 2, 3}, {4}});
  SetupEvent(100, 4, {1});

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
}

/**
 * @brief HandleUnfusibleNotifyNode错误传播
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2)]
 *   stream1: [Notify1(3,unfusible)]
 *   Event1: Notify1对应不存在的Wait(id=999)
 *
 * 预期结果:
 *   - HandleUnfusibleNotifyNode返回false
 *   - SplitGraph返回false
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_UnfusibleNotifyErrorPropagation) {
  auto *wait1 = CreateWaitNode(1, 0, 3, 2);
  wait1->isFusible = false;
  auto *k1 = CreateKernelNode(2, 0, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(3, 1, 100, INVALID_TASK_ID);
  notify1->isFusible = false;

  SetupStreams({{1, 2}, {3}});
  SetupEvent(100, 3, {999});

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_FALSE(result);
}

/**
 * @brief 多Wait节点部分不存在
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2)]
 *   stream1: [Wait2(3,unfusible) -> K2(4)]
 *   stream2: [Notify1(5,unfusible)]
 *   Event1: Notify1对应Wait1(1)和不存在的Wait(id=999)
 *
 * 预期结果:
 *   - SplitGraph返回false
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_MultipleWaitNodesPartialExistence) {
  auto *wait1 = CreateWaitNode(1, 0, 5, 2);
  wait1->isFusible = false;
  auto *k1 = CreateKernelNode(2, 0, INVALID_TASK_ID);

  auto *wait2 = CreateWaitNode(3, 1, 5, 4);
  wait2->isFusible = false;
  auto *k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(5, 2, 100, INVALID_TASK_ID);
  notify1->isFusible = false;

  SetupStreams({{1, 2}, {3, 4}, {5}});
  SetupEvent(100, 5, {1, 999});  // 第二个 wait 节点不存在

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_FALSE(result);
}

/**
 * @brief ResumeSuspendedWaitStreams成功 - 单流暂停
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2)]
 *   stream1: [Notify1(3,unfusible) -> K2(4)]
 *   Event1: Wait1等待Notify1
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 只有Kernel节点被处理: {K1(2), K2(4)}
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_ResumeSuspendedWaitSuccess_Single) {
  auto *wait1 = CreateWaitNode(1, 0, 3, 2);
  wait1->isFusible = false;  // 不可融合，触发 suspend
  auto *k1 = CreateKernelNode(2, 0, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(3, 1, 100, 4);
  notify1->isFusible = false;  // 不可融合
  auto *k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2}, {3, 4}});
  SetupEvent(100, 3, {1});

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {2, 4};  // 只有可融合的 Kernel 节点
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief ResumeSuspendedWaitStreams成功 - 多流暂停
 *
 * 图结构:
 *   stream0: [Wait1(1,unfusible) -> K1(2)]
 *   stream1: [Wait2(3,unfusible) -> K2(4)]
 *   stream2: [Notify1(5,unfusible) -> K3(6)]
 *   Event1: Wait1和Wait2都等待Notify1
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 只有Kernel节点被处理: {K1(2), K2(4), K3(6)}
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_ResumeSuspendedWaitSuccess_Multiple) {
  auto *wait1 = CreateWaitNode(1, 0, 5, 2);
  wait1->isFusible = false;
  auto *k1 = CreateKernelNode(2, 0, INVALID_TASK_ID);

  auto *wait2 = CreateWaitNode(3, 1, 5, 4);
  wait2->isFusible = false;
  auto *k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);

  auto *notify1 = CreateNotifyNode(5, 2, 100, 6);
  notify1->isFusible = false;
  auto *k3 = CreateKernelNode(6, 2, INVALID_TASK_ID);

  SetupStreams({{1, 2}, {3, 4}, {5, 6}});
  SetupEvent(100, 5, {1, 3});

  graph->BuildEventNodeAssociations();

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }
  std::set<uint64_t> expectedNodes = {2, 4, 6};  // 只有可融合的 Kernel 节点
  EXPECT_EQ(allProcessedNodes, expectedNodes);
}

/**
 * @brief SkipUnfusibleNodesForStream错误处理 - 节点不存在
 *
 * 图结构:
 *   stream0: [K1(1,unfusible) -> K2(id=999,不存在)]
 *
 * 预期结果:
 *   - SkipUnfusibleNodesForStream遍历到不存在节点
 *   - SplitGraph返回false
 */

TEST_F(SuperKernelScopeSplitterTest, ErrorHandling_SkipUnfusibleNodeNotFound) {
  auto *k1 = CreateUnfusibleKernelNode(1, 0, 999);  // 不可融合，指向不存在的节点

  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_FALSE(result);
}

// ==================== EventOnly Tests: EventOnly流移除 ====================

// ==================== EventOnly: 单个纯Event流 ====================

/**
 * @brief EventOnlyStreamRemovePass - 单个纯Event流
 *
 * 手动构建scope:
 *   Stream0: [Notify1(id=1)] (纯Event)
 *   Stream1: [K1(id=2)] (有Kernel)
 *
 * 预期结果:
 *   - Notify1标记为non-fusible
 *   - scopes清空触发重新切分
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_SingleEventOnlyStream) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, INVALID_TASK_ID);
  auto *k1 = CreateKernelNode(2, 1, INVALID_TASK_ID);

  SetupStreams({{1}, {2}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(notify1);
  scope.nodes_.push_back(k1);
  ScopeStreamInfo info0{0, 1, 1, 1};
  ScopeStreamInfo info1{1, 2, 2, 1};
  scope.scopeStreamInfos_.push_back(info0);
  scope.scopeStreamInfos_.push_back(info1);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_TRUE(k1->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== EventOnly: 全部纯Event流 ====================

/**
 * @brief EventOnlyStreamRemovePass - 全部纯Event流
 *
 * 手动构建scope:
 *   Stream0: [Notify1(id=1)]
 *   Stream1: [Wait1(id=2)]
 *
 * 预期结果:
 *   - 所有Event节点标记为non-fusible
 *   - scopes清空
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_AllEventOnlyStreams) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, INVALID_TASK_ID);
  auto *wait1 = CreateWaitNode(2, 1, 100, INVALID_TASK_ID);

  SetupStreams({{1}, {2}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(notify1);
  scope.nodes_.push_back(wait1);
  ScopeStreamInfo info0{0, 1, 1, 1};
  ScopeStreamInfo info1{1, 2, 2, 1};
  scope.scopeStreamInfos_.push_back(info0);
  scope.scopeStreamInfos_.push_back(info1);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_FALSE(wait1->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== EventOnly: 混合Stream部分移除 ====================

/**
 * @brief EventOnlyStreamRemovePass - 混合Stream部分移除
 *
 * 手动构建scope:
 *   Stream0: [Notify1(id=1)] (纯Event)
 *   Stream1: [K1(2) -> Notify2(3) -> K2(4)] (混合)
 *   Stream2: [Wait1(5) -> Reset1(6)] (纯Event)
 *
 * 预期结果:
 *   - Stream0/Stream2纯Event标记为non-fusible
 *   - Stream1混合保持不变
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_MixedStreamsPartialRemove) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, INVALID_TASK_ID);
  auto *k1 = CreateKernelNode(2, 1, 3);
  auto *notify2 = CreateNotifyNode(3, 1, 0x200, 4);
  auto *k2 = CreateKernelNode(4, 1, INVALID_TASK_ID);
  auto *wait1 = CreateWaitNode(5, 2, 100, 6);
  auto *reset1 = CreateResetNode(6, 2, 0x300, INVALID_TASK_ID);

  SetPreNodeId(3, 2);
  SetPreNodeId(4, 3);
  SetPreNodeId(6, 5);

  SetupStreams({{1}, {2}, {5}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(notify1);
  scope.nodes_.push_back(k1);
  scope.nodes_.push_back(notify2);
  scope.nodes_.push_back(k2);
  scope.nodes_.push_back(wait1);
  scope.nodes_.push_back(reset1);
  ScopeStreamInfo info0{0, 1, 1, 1};
  ScopeStreamInfo info1{1, 2, 4, 3};
  ScopeStreamInfo info2{2, 5, 6, 2};
  scope.scopeStreamInfos_.push_back(info0);
  scope.scopeStreamInfos_.push_back(info1);
  scope.scopeStreamInfos_.push_back(info2);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_FALSE(wait1->IsFusible());
  EXPECT_FALSE(reset1->IsFusible());
  EXPECT_TRUE(k1->IsFusible());
  EXPECT_TRUE(notify2->IsFusible());  // 混合stream中的event节点不被标记
  EXPECT_TRUE(k2->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== EventOnly: 无Event节点Stream保留 ====================

/**
 * @brief EventOnlyStreamRemovePass - 无Event节点Stream保留
 *
 * 手动构建scope:
 *   Stream0: [K1(1) -> K2(2)] (无Event)
 *
 * 预期结果:
 *   - Stream无Event节点，scope保留
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_NoEventNodesStreamKept) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
  SetPreNodeId(2, 1);

  SetupStreams({{1}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(k1);
  scope.nodes_.push_back(k2);
  ScopeStreamInfo info0{0, 1, 2, 2};
  scope.scopeStreamInfos_.push_back(info0);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(scopes.size(), 1);
  EXPECT_EQ(scopes[0].nodes_.size(), 2);
}

// ==================== EventOnly: Kernel+Event混合保留 ====================

/**
 * @brief EventOnlyStreamRemovePass - Kernel+Event混合保留
 *
 * 手动构建scope:
 *   Stream0: [K1(1) -> Notify1(2) -> K2(3)] (混合)
 *
 * 预期结果:
 *   - Stream有Kernel和Event混合，scope保留
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_KernelWithEventStreamKept) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *notify1 = CreateNotifyNode(2, 0, 0x100, 3);
  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);

  SetupStreams({{1}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(k1);
  scope.nodes_.push_back(notify1);
  scope.nodes_.push_back(k2);
  ScopeStreamInfo info0{0, 1, 3, 3};
  scope.scopeStreamInfos_.push_back(info0);
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(scopes.size(), 1);
  EXPECT_EQ(scopes[0].nodes_.size(), 3);
}

// ==================== EventOnly: 多个Scope处理 ====================

/**
 * @brief EventOnlyStreamRemovePass - 多个Scope处理
 *
 * 手动构建3个scope:
 *   Scope0: [K1(1)] (有Kernel)
 *   Scope1: [Notify1(2)] (纯Event)
 *   Scope2: [K2(3)] (有Kernel)
 *
 * 预期结果:
 *   - Scope1纯Event标记为non-fusible
 *   - scopes清空触发重切分
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_MultipleScopesProcessed) {
  auto *k1 = CreateKernelNode(1, 0, INVALID_TASK_ID);
  auto *notify1 = CreateNotifyNode(2, 1, 0x100, INVALID_TASK_ID);
  auto *k2 = CreateKernelNode(3, 2, INVALID_TASK_ID);

  SetupStreams({{1}, {2}, {3}});

  std::vector<SuperKernelScopeInfo> scopes;

  SuperKernelScopeInfo scope0;
  scope0.nodes_.push_back(k1);
  scope0.scopeStreamInfos_.push_back({0, 1, 1, 1});
  scopes.push_back(std::move(scope0));

  SuperKernelScopeInfo scope1;
  scope1.nodes_.push_back(notify1);
  scope1.scopeStreamInfos_.push_back({1, 2, 2, 1});
  scopes.push_back(std::move(scope1));

  SuperKernelScopeInfo scope2;
  scope2.nodes_.push_back(k2);
  scope2.scopeStreamInfos_.push_back({2, 3, 3, 1});
  scopes.push_back(std::move(scope2));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== EventOnly: FullPipeline集成 ====================

/**
 * @brief EventOnlyStreamRemovePass - FullPipeline集成验证
 *
 * 图结构:
 *   stream0: [K1(1) -> K2(2)] (有Kernel)
 *   stream1: [Notify1(3)] (只有Event)
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - Notify1在BuildEventNodeAssociations后保持当前融合状态
 *   - Stream1只有Event节点，后续由EventOnlyStreamRemovePass移除
 *   - 生成1个scope仅包含K1/K2
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_FullPipelineWithEventOnlyStream) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, INVALID_TASK_ID);
  auto *notify1 = CreateNotifyNode(3, 1, 0x100, INVALID_TASK_ID);

  SetPreNodeId(2, 1);

  SetupStreams({{1}, {3}});
  SetupEvent(0x100, 3, {});

  graph->BuildEventNodeAssociations();

  EXPECT_TRUE(notify1->IsFusible());
  EXPECT_TRUE(notify1->GetCorrespondingWaitNodeIds().empty());

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);

  const auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 1);
  VerifyScope(scopes[0], {1, 2});
}

// ==================== EventOnly: 相邻Event节点 ====================

/**
 * @brief EventOnlyStreamRemovePass - 相邻Event节点
 *
 * 手动构建scope:
 *   Stream0: [Notify1(1) -> Wait1(2) -> Reset1(3)] (纯Event)
 *
 * 预期结果:
 *   - 所有Event节点标记为non-fusible
 *   - scopes清空
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_AdjacentEventNodesRemoved) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, 2);
  auto *wait1 = CreateWaitNode(2, 0, 100, 3);
  auto *reset1 = CreateResetNode(3, 0, 0x200, INVALID_TASK_ID);

  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);

  SetupStreams({{1}});

  std::vector<SuperKernelScopeInfo> scopes;
  SuperKernelScopeInfo scope;
  scope.nodes_.push_back(notify1);
  scope.nodes_.push_back(wait1);
  scope.nodes_.push_back(reset1);
  scope.scopeStreamInfos_.push_back({0, 1, 3, 3});
  scopes.push_back(std::move(scope));

  EventOnlyStreamRemovePass removePass(*graph);
  removePass.SetScopeSplitResultReporter(&reporter);
  bool result = removePass.Run(scopes);

  EXPECT_TRUE(result);
  EXPECT_FALSE(notify1->IsFusible());
  EXPECT_FALSE(wait1->IsFusible());
  EXPECT_FALSE(reset1->IsFusible());
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== EventOnly: 多个单独Event节点 ====================

/**
 * @brief EventOnlyStreamRemovePass - 多个单独Event节点
 *
 * 图结构:
 *   stream0: [Notify1(1)]
 *   stream1: [Notify2(2)]
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 无可融合Kernel，生成0个scope
 */
TEST_F(SuperKernelScopeSplitterTest, EventOnly_MultipleOnlyEventNodes) {
  auto *notify1 = CreateNotifyNode(1, 0, 0x100, INVALID_TASK_ID);
  auto *notify2 = CreateNotifyNode(2, 1, 0x200, INVALID_TASK_ID);

  SetupStreams({{1}, {2}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);

  const auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 0);
}

// ==================== ScheMode 测试 1: 空Scope被丢弃 ====================
/**
 * @brief ScheMode空Scope丢弃测试
 *
 * 输入scope: 空
 *
 * 预期结果:
 *   - Run返回true
 *   - 空scope被丢弃，输出0个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_EmptyScope_Dropped) {
  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 0);
}

// ==================== ScheMode: 单节点不分割 ====================
/**
 * @brief ScheMode单节点不分割测试
 *
 * 输入scope: [K1(cube=4, vec=2, ScheMode=true)]
 *
 * 预期结果:
 *   - Run返回true
 *   - 单节点不分割，输出1个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_SingleNode_NoSplit) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);            // 只有一个scope
  EXPECT_EQ(inputScopes[0].nodes_.size(), 1);  // 包含该节点
  EXPECT_TRUE(inputScopes[0].GetScopeCoreInfo().IsValid());
}

// ==================== ScheMode: 非ScheMode不分割 ====================
/**
 * @brief ScheMode非ScheMode节点不分割
 *
 * 输入scope: [K1(cube=2,vec=1,ScheMode=false) -> K2(cube=4,vec=2,ScheMode=false) -> K3(cube=8,vec=4,ScheMode=false)]
 *
 * 预期结果:
 *   - Run返回true
 *   - 非ScheMode节点不分割
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_AllNonScheModeNodes_NoSplit) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, false, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, false, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 8, 4, false);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);            // 不分割，保持1个scope
  EXPECT_EQ(inputScopes[0].nodes_.size(), 3);  // 3个节点全部保留
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_NoReporterSkipsBreakInfoReporting) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true);
  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  ASSERT_TRUE(pass.Run(inputScopes));

  ASSERT_EQ(inputScopes.size(), 2U);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::NONE);
}

// ==================== ScheMode: Core递增分割(CORE_RISE) ====================
/**
 * @brief ScheMode Core递增触发分割(CORE_RISE)
 *
 * 输入scope: [K1(cube=2,vec=1) -> K2(cube=4,vec=2) -> K3(cube=8,vec=4)] (ScheMode=true)
 *
 * 预期结果:
 *   - Run返回true
 *   - 前面融合了ScheMode算子后，核数递增也需要断开
 *   - K2比K1核数大(CORE_RISE)，K3比K2核数大(CORE_RISE)，分割为3个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_IncreasingCores_SplitAtRisePoint) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 8, 4, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 3);  // core递增，每个上升点都分割
  EXPECT_TRUE(inputScopes[0].GetScopeCoreInfo().IsValid());
  EXPECT_TRUE(inputScopes[1].GetScopeCoreInfo().IsValid());
  EXPECT_TRUE(inputScopes[2].GetScopeCoreInfo().IsValid());
  ASSERT_GE(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({1}));
  ASSERT_GE(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({2}));
  ASSERT_GE(inputScopes[2].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[2].nodes_[0]->GetNodeId(), 3);
}

// ==================== ScheMode: Core相等不分割 ====================
/**
 * @brief ScheMode Core相等不分割
 *
 * 输入scope: [K1(cube=4,vec=2) -> K2(cube=4,vec=2) -> K3(cube=4,vec=2)] (ScheMode=true)
 *
 * 预期结果:
 *   - Run返回true
 *   - Core相等，不分割
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_EqualCores_NoSplit) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 2, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);            // core相等，不分割
  EXPECT_EQ(inputScopes[0].nodes_.size(), 3);  // 全部在一个scope中
}

// ==================== ScheMode: Cube下降分割 ====================
/**
 * @brief ScheMode Cube下降分割
 *
 * 输入scope: [K1(cube=8,vec=4) -> K2(cube=4,vec=2) -> K3(cube=2,vec=1)] (ScheMode=true)
 *
 * 预期结果:
 *   - Run返回true
 *   - Cube连续下降，分割为3个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_DecreasingCube_SplitAtDropPoint) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 8, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);  // 第一次分割点
  auto *k3 = CreateScheModeKernelNode(3, 0, 2, 1, true);     // 第二次分割点

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 3);  // 连续下降，分成3个scope
  ASSERT_GE(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({1}));
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetDetail(),
            "beforeSk=MIX_1_1(blocks=8,C=8,V=8), candidateSk=MIX_1_1(blocks=8,C=8,V=8), "
            "mismatch=ScheMode C(expected=4,actual=8)");
  ASSERT_GE(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({2}));
  ASSERT_GE(inputScopes[2].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[2].nodes_[0]->GetNodeId(), 3);
}

// ==================== ScheMode: C核相同且Vec下降 ====================
/**
 * @brief 非pure-V ScheMode算子按C核精确匹配
 *
 * 输入scope: [K1(cube=4,vec=8) -> K2(cube=4,vec=4) -> K3(cube=4,vec=2)] (ScheMode=true)
 *
 * 预期结果:
 *   - Run返回true
 *   - 三个算子的C核数相同，不因Vec下降切分
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_SameCubeDecreasingVec_DoesNotSplit) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 8, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 4, true, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  ASSERT_EQ(inputScopes.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_.size(), 3);
  ASSERT_TRUE(FinalizeScopeCoreInfo(inputScopes));
  EXPECT_EQ(inputScopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(inputScopes[0].GetScopeCoreInfo().numBlocks, 4U);
}

// ==================== ScheMode: 混合非ScheMode ====================
/**
 * @brief ScheMode混合非ScheMode节点
 *
 * 输入scope: [K1(cube=8,vec=4,ScheMode=true) -> K2(cube=2,vec=1,ScheMode=false) -> K3(cube=4,vec=2,ScheMode=true)]
 *
 * 预期结果:
 *   - K2按非ScheMode上限规则参与合并，但不改变当前最大核数
 *   - K3要求C精确为4，与K1要求C精确为8冲突
 *   - 输出2个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_MixedWithNonScheMode_UsesUpperBound) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 8, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 2, 1, false, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 2);                     // 在k3处分割
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);  // scopeBefore含k1,k2
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 3);  // scopeAfter从k3开始
}

// ==================== ScheMode: 多次分割 ====================
/**
 * @brief ScheMode多次分割
 *
 * 输入scope: [K1(cube=8,vec=4) -> K2(cube=4,vec=2)[分割点] -> K3(cube=2,vec=1)[分割点]]
 *
 * 预期结果:
 *   - 输出3个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_MultipleSplits_ConsecutiveDrops) {
  // k1(8,4) -> k2(4,2)[降,分割] -> k3(2,1)[再降,再分割]
  auto *k1 = CreateScheModeKernelNode(1, 0, 8, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);  // 第一次分割点
  auto *k3 = CreateScheModeKernelNode(3, 0, 2, 1, true);     // 第二次分割点

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 3);  // 连续下降，分成3个scope
  EXPECT_EQ(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(inputScopes[2].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[2].nodes_[0]->GetNodeId(), 3);
}

// ==================== ScheMode: 先增后降 ====================
/**
 * @brief ScheMode先增后降
 *
 * 输入scope: [K1(cube=2,vec=1)[增] -> K2(cube=4,vec=2)[增] ->
 *             K3(cube=8,vec=4)[增] -> K4(cube=4,vec=2)[降,分割]]
 *
 * 预期结果:
 *   - 相邻ScheMode算子的精确C核数均不相同
 *   - 输出4个单算子scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_IncreaseThenDecrease_SplitAtRiseAndDrop) {
  // k1(2,1) -> k2(4,2)[增,CORE_RISE] -> k3(8,4)[增,CORE_RISE] -> k4(4,2)[降,CORE_DROP]
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 8, 4, true, 4);
  auto *k4 = CreateScheModeKernelNode(4, 0, 4, 2, true);  // 分割点

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3, k4}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 4);  // k2,k3核数上升分割，k4核数下降分割
  EXPECT_EQ(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(inputScopes[2].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[2].nodes_[0]->GetNodeId(), 3);
  EXPECT_EQ(inputScopes[3].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[3].nodes_[0]->GetNodeId(), 4);
}

// ==================== ScheMode: 多Scope独立处理 ====================
/**
 * @brief ScheMode多Scope独立处理
 *
 * 输入:
 *   - ScopeA: [K1(cube=4,vec=2) -> K2(cube=2,vec=1)] (下降，应分割)
 *   - ScopeB: [K3(cube=2,vec=1) -> K4(cube=4,vec=2)] (精确C核冲突，应分割)
 *
 * 预期结果:
 *   - 输出共4个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_MultipleInputScopes_ProcessedIndependently) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 2, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 2, 1, true);
  auto *k3 = CreateScheModeKernelNode(3, 0, 2, 1, true, 4);
  auto *k4 = CreateScheModeKernelNode(4, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));
  inputScopes.push_back(BuildTestScope({k3, k4}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 4);
  bool foundK1 = false, foundK2 = false, foundK3 = false, foundK4 = false;
  for (const auto &scope : inputScopes) {
    if (scope.nodes_.size() == 1 && scope.nodes_[0]->GetNodeId() == 1) foundK1 = true;
    if (scope.nodes_.size() == 1 && scope.nodes_[0]->GetNodeId() == 2) foundK2 = true;
    if (scope.nodes_.size() == 1 && scope.nodes_[0]->GetNodeId() == 3) foundK3 = true;
    if (scope.nodes_.size() == 1 && scope.nodes_[0]->GetNodeId() == 4) foundK4 = true;
  }
  EXPECT_TRUE(foundK1);
  EXPECT_TRUE(foundK2);
  EXPECT_TRUE(foundK3);
  EXPECT_TRUE(foundK4);
}

// ==================== ScheMode: Cube相同Vec小分割 ====================
/**
 * @brief ScheMode Cube相同Vec更小分割
 *
 * 输入scope: [K1(cube=4,vec=4) -> K2(cube=4,vec=2)] (ScheMode=true)
 *
 * 预期结果:
 *   - Cube相同但Vec更小，分割
 *   - 输出2个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_SameCubeSmallerVec_Split) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);  // C/MIX ScheMode only requires exact cube cores
}

// ==================== ScheMode: Cube大Vec小分割 ====================
/**
 * @brief ScheMode精确C核数变化时分割
 *
 * 输入scope: [K1(cube=4,vec=4) -> K2(cube=8,vec=2)] (ScheMode=true)
 *
 * 预期结果:
 *   - C核从4变为8，精确C核约束冲突并触发分割
 *   - 输出2个scope
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_LargerCubeSmallerVec_Split) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 8, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 2);
  EXPECT_EQ(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_.size(), 1);
}

// ==================== ScheMode: 0值维度 ====================
/**
 * @brief ScheMode使用非零维度建立精确约束
 *
 * 输入scope: [k1(4,4) -> k2(0,8) -> k3(8,0)]
 *   - k2是pure-V，要求V精确为8，可与k1组成MIX1:2
 *   - k3要求C精确为8，与k1已有的C精确为4冲突
 *
 * 预期结果:
 *   - Run返回true
 *   - 在k3前切分，输出2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_ZeroDimensionUsesNonZeroExactRequirement) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 0, 8, true, 3);
  auto *k3 = CreateScheModeKernelNode(3, 0, 8, 0, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  ASSERT_EQ(inputScopes.size(), 2);
  ASSERT_GE(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  ASSERT_GE(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 3);
}

// ==================== ScheMode: 非ScheMode算子核数上升分割(CORE_RISE) ====================
/**
 * @brief 前面融合了ScheMode算子后，后续非ScheMode算子核数上升也分割
 *
 * 输入scope: [K1(cube=4,vec=2,ScheMode=true) -> K2(cube=8,vec=4,ScheMode=false)]
 *
 * 预期结果:
 *   - K2非ScheMode但核数大于merged(4,2)，触发CORE_RISE分割
 *   - 输出2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_NonScheModeCoreRise_Split) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 2, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 8, 4, false);  // 非ScheMode但核数大

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 2);  // K2核数上升触发CORE_RISE分割
  EXPECT_EQ(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
}

// ==================== ScheMode: ScheMode算子后非ScheMode核数上升Cube分割 ====================
/**
 * @brief 前面融合了ScheMode算子后，后续非ScheMode算子仅cube上升也分割
 *
 * 输入scope: [K1(cube=4,vec=4,ScheMode=true) -> K2(cube=8,vec=2,ScheMode=false)]
 *
 * 预期结果:
 *   - K2的cube=8 > merged_cube=4，触发CORE_RISE分割
 *   - 输出2个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_NonScheModeCubeRise_Split) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 4, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 8, 2, false);  // cube上升，vec下降

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 2);  // K2的cube上升触发CORE_RISE分割
}

// ==================== ScheMode: 无ScheMode算子时核数上升不分割 ====================
/**
 * @brief 所有算子都未开启ScheMode时，核数上升不分割
 *
 * 输入scope: [K1(cube=2,vec=1,ScheMode=false) -> K2(cube=4,vec=2,ScheMode=false)]
 *
 * 预期结果:
 *   - 没有ScheMode算子，核数上升不触发分割
 *   - 输出1个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_NoScheModeCoreRise_NoSplit) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, false, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, false);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);  // 无ScheMode，核数上升不分割
  EXPECT_EQ(inputScopes[0].nodes_.size(), 2);
}

// ==================== ScheMode: 混合ScheMode先降后升分割 ====================
/**
 * @brief ScheMode算子后先降后升，分别触发CORE_DROP和CORE_RISE分割
 *
 * 输入scope: [K1(cube=8,vec=4,ScheMode=true) -> K2(cube=4,vec=2,ScheMode=true) -> K3(cube=6,vec=3,ScheMode=false)]
 *
 * 预期结果:
 *   - K2比K1核数小，CORE_DROP分割
 *   - K3比K2核数大且K2开启了ScheMode，CORE_RISE分割
 *   - 输出3个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_CoreDropThenRise_SplitAtDropAndRise) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 8, 4, true, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true, 3);  // CORE_DROP分割点
  auto *k3 = CreateScheModeKernelNode(3, 0, 6, 3, false);    // CORE_RISE分割点

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2, k3}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 3);  // k2处CORE_DROP分割，k3处CORE_RISE分割
  ASSERT_GE(inputScopes[0].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_[0]->GetNodeId(), 1);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[0].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({1}));
  ASSERT_GE(inputScopes[1].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[1].nodes_[0]->GetNodeId(), 2);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(inputScopes[1].GetBreakInfo().GetSyncAllNodeIds(), std::vector<uint64_t>({2}));
  ASSERT_GE(inputScopes[2].nodes_.size(), 1);
  EXPECT_EQ(inputScopes[2].nodes_[0]->GetNodeId(), 3);
}

// ==================== ScheMode: 非ScheMode算子作为首节点 ====================
/**
 * @brief 首节点非ScheMode需求不超过后续ScheMode精确核数
 *
 * 输入scope: [K1(cube=2,vec=1,ScheMode=false) -> K2(cube=4,vec=2,ScheMode=true)]
 *
 * 预期结果:
 *   - K1按非ScheMode上限规则要求C/V不超过最终SK
 *   - K2要求C精确为4，可形成满足两者的最终SK
 *   - 输出1个scope
 */
TEST_F(SuperKernelScopeSplitterTest, ScheMode_NonScheModeFirst_FitsFinalExactCore) {
  auto *k1 = CreateScheModeKernelNode(1, 0, 2, 1, false, 2);
  auto *k2 = CreateScheModeKernelNode(2, 0, 4, 2, true);

  std::vector<SuperKernelScopeInfo> inputScopes;
  inputScopes.push_back(BuildTestScope({k1, k2}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  bool result = pass.Run(inputScopes);

  EXPECT_TRUE(result);
  EXPECT_EQ(inputScopes.size(), 1);
  EXPECT_EQ(inputScopes[0].nodes_.size(), 2);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_OddVectorAndNativeMix12_SplitBeforeMix12) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 31, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *mix12Node = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 8, SkKernelType::MIX_AIC_1_2);

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({vectorNode, mix12Node}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 2);
  EXPECT_EQ(scopes[0].GetBreakInfo().GetDetail(),
            "beforeSk=AIV_ONLY(blocks=31,C=0,V=31), candidateSk=MIX_1_2(blocks=16,C=16,V=32), "
            "mismatch=ScheMode V(expected=31,actual=32)");
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::AIV_ONLY);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 31U);
  EXPECT_EQ(scopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::SYNCALL_OP_DROP);
  EXPECT_EQ(scopes[1].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopes[1].GetScopeCoreInfo().numBlocks, 8U);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_OddVectorAndNonScheCube_UseMix11) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *cubeNode = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 3, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({vectorNode, cubeNode}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 1);
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_1);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 5U);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_NonScheOddVectorAndCube_UseMix12) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::AIV_ONLY);
  auto *cubeNode = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 3, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({vectorNode, cubeNode}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 3U);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_OddVectorAndScheCube_SplitForExactMismatch) {
  auto *vectorNode = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *cubeNode = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 3, SkKernelType::AIC_ONLY);
  cubeNode->nodeInfos.kernelInfos.isScheModeOn = true;

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({vectorNode, cubeNode}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 2);
  EXPECT_EQ(scopes[0].GetBreakInfo().GetDetail(),
            "beforeSk=AIV_ONLY(blocks=5,C=0,V=5), candidateSk=MIX_1_2(blocks=3,C=3,V=6), "
            "mismatch=ScheMode V(expected=5,actual=6)");
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::AIV_ONLY);
  EXPECT_EQ(scopes[1].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_Mix11AdaptsVectorSideToMix12) {
  auto *mix11Node = CreateKernelNodeWithCores(1, 0, 2, 5, SkKernelType::MIX_AIC_1_1);
  mix11Node->nodeInfos.kernelInfos.isScheModeOn = true;
  auto *vectorNode = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 10, SkKernelType::AIV_ONLY);
  vectorNode->nodeInfos.kernelInfos.isScheModeOn = true;

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({mix11Node, vectorNode}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 5U);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_CandidatesUseRawTaskCoreRequirements) {
  auto *cube8 = CreateKernelNodeWithCores(1, 0, 2, 8, SkKernelType::AIC_ONLY);
  auto *vector10 = CreateKernelNodeWithCores(2, 0, 3, 10, SkKernelType::AIV_ONLY);
  auto *cube12 = CreateKernelNodeWithCores(3, 0, INVALID_TASK_ID, 12, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({cube8, vector10, cube12}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_1);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 12U);
}

TEST_F(SuperKernelScopeSplitterTest, ScheMode_NativeMix12ConstrainsFinalSkType) {
  auto *mix12 = CreateKernelNodeWithCores(1, 0, 2, 3, SkKernelType::MIX_AIC_1_2);
  auto *cube = CreateKernelNodeWithCores(2, 0, INVALID_TASK_ID, 8, SkKernelType::AIC_ONLY);

  std::vector<SuperKernelScopeInfo> scopes;
  scopes.push_back(BuildTestScope({mix12, cube}));

  ScheModeKernelSplitPass pass(*graph);
  pass.SetScopeSplitResultReporter(&reporter);
  ASSERT_TRUE(pass.Run(scopes));
  ASSERT_EQ(scopes.size(), 1U);
  ASSERT_TRUE(FinalizeScopeCoreInfo(scopes));
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(scopes[0].GetScopeCoreInfo().numBlocks, 8U);
}

// ==================== ScheMode: Pass名称验证 ====================
/**
 * @brief ScheMode Pass名称验证
 *
 * 预期结果:
 *   - Pass名称应为"ScheModeKernelSplitPass"
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_PassName_Verification) {
  ScheModeKernelSplitPass pass(*graph);
  EXPECT_EQ(pass.GetName(), "ScheModeKernelSplitPass");
}

// ==================== ScheMode: Run返回值 ====================
/**
 * @brief ScheMode Run返回值验证
 *
 * 预期结果:
 *   - 空scope和合法scope场景下Run返回true
 */

TEST_F(SuperKernelScopeSplitterTest, ScheMode_RunReturnsTrueForEmptyAndValidScopes) {
  {  // 空输入
    std::vector<SuperKernelScopeInfo> emptyScopes;
    ScheModeKernelSplitPass pass(*graph);
    pass.SetScopeSplitResultReporter(&reporter);
    EXPECT_TRUE(pass.Run(emptyScopes));
  }
  {  // 正常输入
    auto *k1 = CreateScheModeKernelNode(1, 0, 4, 2, true);
    std::vector<SuperKernelScopeInfo> scopes;
    scopes.push_back(BuildTestScope({k1}));
    ScheModeKernelSplitPass pass(*graph);
    pass.SetScopeSplitResultReporter(&reporter);
    EXPECT_TRUE(pass.Run(scopes));
  }
}

// ==================== DefaultNode Tests: Default节点处理 ====================
/**
 * @file DefaultNode Tests
 *
 * 测试AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用时，Default节点的处理逻辑：
 * - 有Kernel的流中Default触发scope重切分
 * - 无Kernel的流中Default被移除
 */

// 测试1: 选项禁用时，default节点保持原有行为（不进入scope）
/**
 * @brief DefaultNode选项禁用 - 原有行为
 *
 * 图结构:
 *   stream0: [K1(1) -> Default1(2) -> K2(3)]
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass禁用
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - Default不进入scope，触发切分
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_OptionDisabled_Legacy) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *d1 = CreateDefaultNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0);
  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
}

// 测试2: 选项启用时，default在有kernel流中触发重切分
/**
 * @brief DefaultNode选项启用 - 有Kernel流触发重切分
 *
 * 图结构:
 *   stream0: [K1(1) -> Default1(2) -> K2(3)]
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - Default标记为不可融合，触发重切分
 *   - 生成2个scope: {K1(1)}, {K2(3)}
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_OptionEnabled_StreamWithKernel_Resplit) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *d1 = CreateDefaultNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0);
  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetupStreams({{1, 2, 3}});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 2);
}

// 测试3: 选项启用时，default在无kernel流中被移除
/**
 * @brief DefaultNode选项启用 - 无Kernel流被移除
 *
 * 图结构:
 *   stream0: [K1(1)]
 *   stream1: [Default1(2) -> Notify1(3) -> Wait1(4)] (无Kernel)
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - stream1无Kernel，被移除
 *   - 生成1个scope: {K1(1)}
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_OptionEnabled_StreamWithoutKernel_Removed) {
  auto *k1 = CreateKernelNode(1, 0);
  auto *d1 = CreateDefaultNode(2, 1, 3);
  auto *n1 = CreateNotifyNode(3, 1, 100, 4);
  auto *w1 = CreateWaitNode(4, 1, 100);
  SetPreNodeId(3, 2);
  SetPreNodeId(4, 3);
  SetupStreams({{1}, {2, 3, 4}});
  SetupEvent(100, 3, {4});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 1);
  EXPECT_EQ(scopes[0].GetNodes().size(), 1);  // 只有kernel1
}

// 测试4: 用户场景完整验证
/**
 * @brief DefaultNode用户场景完整验证
 *
 * 复杂图结构，验证实际使用场景
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - 正确处理Default和Event依赖
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_UserScenario) {
  auto *n1 = CreateNotifyNode(1, 0, 101, 2);
  auto *k1 = CreateKernelNode(2, 0, 3);
  auto *w2 = CreateWaitNode(3, 0, 102, 4);
  auto *k2 = CreateKernelNode(4, 0, 5);
  auto *n3 = CreateNotifyNode(5, 0, 103, 6);
  auto *k3 = CreateKernelNode(6, 0, 7);
  auto *w4 = CreateWaitNode(7, 0, 104);

  auto *w1 = CreateWaitNode(11, 1, 101, 12);
  auto *d1 = CreateDefaultNode(12, 1, 13);
  auto *n2 = CreateNotifyNode(13, 1, 102, 14);
  auto *w3 = CreateWaitNode(14, 1, 103, 15);
  auto *d2 = CreateDefaultNode(15, 1, 16);
  auto *n4 = CreateNotifyNode(16, 1, 104);

  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetPreNodeId(4, 3);
  SetPreNodeId(5, 4);
  SetPreNodeId(6, 5);
  SetPreNodeId(7, 6);
  SetPreNodeId(12, 11);
  SetPreNodeId(13, 12);
  SetPreNodeId(14, 13);
  SetPreNodeId(15, 14);
  SetPreNodeId(16, 15);

  SetupStreams({{1, 2, 3, 4, 5, 6, 7}, {11, 12, 13, 14, 15, 16}});
  SetupEvent(101, 1, {11});
  SetupEvent(102, 13, {3});
  SetupEvent(103, 5, {14});
  SetupEvent(104, 16, {7});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 1);
}

// 测试5: 多个default节点在同一条有kernel流中
/**
 * @brief DefaultNode多Default有Kernel流
 *
 * 图结构:
 *   stream0: [K1(1) -> Default1(2) -> K2(3) -> Default2(4) -> K3(5)]
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - 所有Default作为边界
 *   - 生成3个scope
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_MultipleDefaults_StreamWithKernel) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *d1 = CreateDefaultNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *d2 = CreateDefaultNode(4, 0, 5);
  auto *k3 = CreateKernelNode(5, 0);
  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetPreNodeId(4, 3);
  SetPreNodeId(5, 4);
  SetupStreams({{1, 2, 3, 4, 5}});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 3);
}

// 测试6: 空scope在移除default后被丢弃
/**
 * @brief DefaultNode移除后空Scope丢弃
 *
 * 图结构:
 *   stream0: [Default1(1) -> Notify1(2) -> Wait1(3)] (无Kernel)
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - 移除Default后只剩Event，整体移除
 *   - 生成0个scope
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_EmptyScopeAfterRemoval_Dropped) {
  auto *d1 = CreateDefaultNode(1, 0, 2);
  auto *n1 = CreateNotifyNode(2, 0, 100, 3);
  auto *w1 = CreateWaitNode(3, 0, 100);
  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetupStreams({{1, 2, 3}});
  SetupEvent(100, 2, {3});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 0);
}

// 测试7: DefaultNodeProcessPass名称验证
/**
 * @brief DefaultNode Pass名称验证
 *
 * 预期结果:
 *   - Pass名称应为"DefaultNodeProcessPass"
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_PassName_Verification) {
  DefaultNodeProcessPass pass(*graph);
  EXPECT_EQ(pass.GetName(), "DefaultNodeProcessPass");
}

// 测试8: 多scope独立处理
/**
 * @brief DefaultNode多Scope独立处理
 *
 * 图结构:
 *   stream0: [K1(1) -> Default1(2) -> K2(3)]
 *   stream1: [Default2(4)]
 *   stream2: [K3(5) -> Default3(6) -> K4(7)]
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - SplitGraph返回true
 *   - Default1/Default3触发切分
 *   - Default2所在流移除
 */

TEST_F(SuperKernelScopeSplitterTest, DefaultNode_MultipleScopes_Independent) {
  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *d1 = CreateDefaultNode(2, 0, 3);
  auto *k2 = CreateKernelNode(3, 0);
  auto *d2 = CreateDefaultNode(4, 1);
  auto *k3 = CreateKernelNode(5, 2, 6);
  auto *d3 = CreateDefaultNode(6, 2, 7);
  auto *k4 = CreateKernelNode(7, 2);

  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetPreNodeId(6, 5);
  SetPreNodeId(7, 6);
  SetupStreams({{1, 2, 3}, {4}, {5, 6, 7}});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
}

// 测试9: 有default的stream和无default的纯event流混合场景
/**
 * @brief DefaultNode混合EventOnly流
 *
 * 图结构:
 *   stream0: [Default1(1) -> Notify1(2) -> Wait1(3)] (有Default无Kernel)
 *   stream1: [Notify2(11) -> Wait2(12)] (纯Event无Default)
 *   stream2: [K1(21)]
 *   AGGRESSIVE_OPT_STRATEGIES.taskBreakerBypass启用
 *
 * 预期结果:
 *   - stream0由DefaultNodeProcessPass移除
 *   - stream1由EventOnlyStreamRemovePass处理
 *   - stream2单独成scope
 */

// streamA(default+event)应被移除，streamB(纯event)应交给EventOnlyStreamRemovePass
TEST_F(SuperKernelScopeSplitterTest, DefaultNode_StreamWithDefaultAndStreamOnlyEvent) {
  auto *d1 = CreateDefaultNode(1, 0, 2);
  auto *n1 = CreateNotifyNode(2, 0, 100, 3);
  auto *w1 = CreateWaitNode(3, 0, 100);

  auto *n2 = CreateNotifyNode(11, 1, 101, 12);
  auto *w2 = CreateWaitNode(12, 1, 101);

  auto *k1 = CreateKernelNode(21, 2);

  SetPreNodeId(2, 1);
  SetPreNodeId(3, 2);
  SetPreNodeId(12, 11);
  SetupStreams({{1, 2, 3}, {11, 12}, {21}});
  SetupEvent(100, 2, {3});
  SetupEvent(101, 11, {12});

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  auto &scopes = splitter.GetScopeInfos();
  EXPECT_EQ(scopes.size(), 1);
  EXPECT_EQ(scopes[0].GetNodes().size(), 1);  // 只有kernel1
}

/**
 * @brief DefaultNode所在流有unfusible Kernel+fusible Notify，流剔除后触发死锁切分
 *
 * 场景描述:
 *   - 流1有unfusible Kernel(K3)和fusible Notify1
 *   - K3是unfusible不进入scope，Notify1是fusible进入初始scope
 *   - DefaultNodeProcessPass检测scope内stream 1没有Kernel，移除Default1及Notify1
 *   - Wait1依赖Notify1(已被移除)，检查Notify1前驱K3(12)占用资源
 *   - K2(15)检查时资源不足，触发死锁切分
 *
 * 图结构:
 *   stream0: K1(1, cube=12) -> Wait1(2, 等待Notify1) -> K2(3, cube=15)
 *   stream1: K3(4, cube=12, unfusible) -> Notify1(5, eventId=100, fusible) -> Default1(6)
 *   Event100: Notify1(5) -> Wait1(2)
 *
 * 执行流程:
 *   1. taskBreakerBypass启用，Default1初始化为fusible=true
 *   2. InitialScopeSplitPass跳过K3(unfusible)，Notify1(fusible)进入scope
 *      - scope: {K1,Notify1,Wait1,K2,Default1} (按nodeId排序加入)
 *   3. DefaultNodeProcessPass检测scope内stream 1:
 *      - stream 1在scope内只有Notify1和Default1，没有Kernel(K3不在scope)
 *      - 执行RemoveDefaultsAndStreams移除Default1
 *      - RemoveStreamFromScope移除stream 1所有节点（Notify1被移除）
 *   4. 剩余scope: {K1,Wait1,K2}
 *   5. DeadlockRefinePass检查:
 *      - 加入K1后，候选SK核数为12C
 *      - Wait1检查Notify1(不在scope内):
 *        -> GetWaitNodeFusibleStatus: Notify1不在SK stream
 *        -> HasDeadlock(Notify1): preNode=K3(4, unfusible)
 *        -> CheckKernelNodeDeadlock(K3):
 *           HasEnoughCores(K3, false): available={25-12=13}, 12<=13
 *        -> 记录外部依赖占用12C，Wait1可以继续融合
 *      - 加入K2后，候选SK核数增长为15C，大于剩余13C，K2触发资源死锁风险
 *   6. 报告K2为触发节点，在Wait1处切分
 *
 * 预期结果:
 *   - 最终2个scope: {K1}, {K2}
 *   - K3(unfusible), Notify1(被移除), Default1(被移除), Wait1(切分边界)不在scope中
 */
TEST_F(SuperKernelScopeSplitterTest, DefaultNode_StreamWithKernelNotify_TriggerDeadlockAfterResplit) {
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;

  auto *k1 = CreateKernelNode(1, 0, 2);
  k1->nodeInfos.kernelInfos.cubeNum = 12;
  k1->nodeInfos.kernelInfos.numBlocks = 12;

  auto *wait1 = CreateWaitNode(2, 0, 5, 3);

  auto *k2 = CreateKernelNode(3, 0, INVALID_TASK_ID);
  k2->nodeInfos.kernelInfos.cubeNum = 15;
  k2->nodeInfos.kernelInfos.numBlocks = 15;

  auto *k3 = CreateUnfusibleKernelNode(4, 1, 5);
  k3->nodeInfos.kernelInfos.cubeNum = 12;
  k3->nodeInfos.kernelInfos.numBlocks = 12;

  auto *notify1 = CreateNotifyNode(5, 1, 100, 6);

  auto *d1 = CreateDefaultNode(6, 1, INVALID_TASK_ID);

  SetupStreams({{1, 2, 3}, {4, 5, 6}});
  SetupEvent(100, 5, {2});

  graph->BuildEventNodeAssociations();

  aclskOption option{};
  option.optionType = aclskOptionType::AGGRESSIVE_OPT_STRATEGIES;
  option.aggressiveOpts.taskBreakerBypass = 1;
  opts->SetOptOptionValue(&option);

  SuperKernelScopeSplitter splitter(*graph, *opts);
  LockDetector::deviceRealCubeNum = 25;
  LockDetector::deviceRealVecNum = 32;
  bool result = splitter.SplitGraph();

  ASSERT_TRUE(result);
  const auto &scopeInfos = splitter.GetScopeInfos();

  ASSERT_EQ(scopeInfos.size(), 2U);
  ASSERT_EQ(scopeInfos[0].GetNodes().size(), 1U);
  ASSERT_EQ(scopeInfos[1].GetNodes().size(), 1U);
  EXPECT_EQ(scopeInfos[0].GetNodes()[0]->GetNodeId(), 1U);
  EXPECT_EQ(scopeInfos[1].GetNodes()[0]->GetNodeId(), 3U);
  EXPECT_EQ(scopeInfos[0].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopeInfos[0].GetScopeCoreInfo().numBlocks, 12U);
  EXPECT_EQ(scopeInfos[1].GetScopeCoreInfo().type, SkKernelType::AIC_ONLY);
  EXPECT_EQ(scopeInfos[1].GetScopeCoreInfo().numBlocks, 15U);
  EXPECT_EQ(wait1->GetFusionFailReason(), FusionFailReason::CAN_FUSE);
  EXPECT_EQ(k2->GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  EXPECT_EQ(scopeInfos[0].GetBreakInfo().GetReason(), ScopeBreakReason::DEADLOCK_DETECTED);
  EXPECT_EQ(scopeInfos[0].GetBreakInfo().GetTriggerNodeId(), 3U);

  std::set<uint64_t> allProcessedNodes;
  for (const auto &scope : scopeInfos) {
    for (const auto *node : scope.nodes_) {
      allProcessedNodes.insert(node->GetNodeId());
    }
  }

  EXPECT_TRUE(allProcessedNodes.find(1) != allProcessedNodes.end()) << "K1 should be in scope";
  EXPECT_TRUE(allProcessedNodes.find(3) != allProcessedNodes.end()) << "K2 should be in scope";
  EXPECT_TRUE(allProcessedNodes.find(4) == allProcessedNodes.end()) << "K3(unfusible) should not be in scope";
  EXPECT_TRUE(allProcessedNodes.find(5) == allProcessedNodes.end())
      << "Notify1 should be removed by DefaultNodeProcessPass";
  EXPECT_TRUE(allProcessedNodes.find(6) == allProcessedNodes.end())
      << "Default1 should be removed by DefaultNodeProcessPass";

  LockDetector::deviceRealCubeNum = 32;
  LockDetector::deviceRealVecNum = 32;
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_EmptyGraph_ReturnsTrue) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  SetupStreams({});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 0);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_SingleKernel_CreatesSingleScope) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0);

  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 1);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetNodes().size(), 1);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_PreservesScopeName) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));
  CreateScopeBeginNode(1, 0, "scope_A", 2);
  auto *kernel = CreateKernelNode(2, 0, 3);
  CreateScopeEndNode(3, 0, "scope_A");
  SetupStreams({{1, 2, 3}});
  graph->scopeNameToIdx["scope_A"] = 0;
  graph->scopeIdxToName[0] = "scope_A";
  ASSERT_TRUE(graph->UpdateNodeScopeBitFlags());

  SuperKernelScopeSplitter splitter(*graph, *opts);
  ASSERT_TRUE(splitter.SplitGraph());
  ASSERT_EQ(splitter.GetScopeInfos().size(), 3U);

  const auto &scope = splitter.GetScopeInfos()[1];
  EXPECT_EQ(scope.GetScopeBitFlags(), kernel->GetScopeBitFlags());
  EXPECT_EQ(ScopeSplitPass::GetScopeNamesFromBitFlags(scope.GetScopeBitFlags(), *graph), "scope_A");
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_OddVectorScheModeKeepsAivResult) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));
  auto *kernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 5, SkKernelType::AIV_ONLY);
  kernel->nodeInfos.kernelInfos.isScheModeOn = true;
  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  ASSERT_TRUE(splitter.SplitGraph());
  ASSERT_EQ(splitter.GetScopeInfos().size(), 1U);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetScopeCoreInfo().type, SkKernelType::AIV_ONLY);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetScopeCoreInfo().numBlocks, 5U);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_EvenVectorScheModeKeepsMix12Behavior) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));
  auto *kernel = CreateKernelNodeWithCores(1, 0, INVALID_TASK_ID, 6, SkKernelType::AIV_ONLY);
  kernel->nodeInfos.kernelInfos.isScheModeOn = true;
  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  ASSERT_TRUE(splitter.SplitGraph());
  ASSERT_EQ(splitter.GetScopeInfos().size(), 1U);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetScopeCoreInfo().type, SkKernelType::MIX_AIC_1_2);
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetScopeCoreInfo().numBlocks, 3U);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_MultipleKernels_CreatesMultipleScopes) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateKernelNode(2, 0, 3);
  auto *k3 = CreateKernelNode(3, 0);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 3);
  for (size_t i = 0; i < splitter.GetScopeInfos().size(); ++i) {
    EXPECT_EQ(splitter.GetScopeInfos()[i].GetNodes().size(), 1);
  }
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_UnfusibleKernel_Skipped) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *k2 = CreateUnfusibleKernelNode(2, 0, 3);
  auto *k3 = CreateKernelNode(3, 0);

  SetupStreams({{1, 2, 3}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 2);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_NotifyWaitNodes_Skipped) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0, 2);
  auto *n1 = CreateNotifyNode(2, 0, 100, 3);
  auto *k2 = CreateKernelNode(3, 0, 4);
  auto *w1 = CreateWaitNode(4, 0, 100);

  SetupStreams({{1, 2, 3, 4}});
  SetupEvent(100, 2, {4});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 2);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_MultipleHeadNodes_CreatesAllScopes) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0);
  auto *k2 = CreateKernelNode(11, 1);
  auto *k3 = CreateKernelNode(21, 2);

  SetupStreams({{1}, {11}, {21}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 3);
}

TEST_F(SuperKernelScopeSplitterTest, PerOpMaxCoreSplitPass_DoesNotSetBreakReason) {
  opts->AddOption(std::make_unique<NumberOptOption>("debug_per_op_max_core_num",
                                                    aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM, 1, 0, 1));

  auto *k1 = CreateKernelNode(1, 0);

  SetupStreams({{1}});

  SuperKernelScopeSplitter splitter(*graph, *opts);
  bool result = splitter.SplitGraph();

  EXPECT_TRUE(result);
  EXPECT_EQ(splitter.GetScopeInfos().size(), 1);
  EXPECT_TRUE(splitter.IsDebugPerOpMaxCoreEnabled());
  EXPECT_EQ(splitter.GetScopeInfos()[0].GetBreakInfo().GetReason(), ScopeBreakReason::NONE);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeSplitResultReporter_RestoreOnlyMatchesCompleteScopeStructure) {
  auto *kernel = CreateKernelNode(1, 0);
  auto *defaultNode = CreateDefaultNode(2, 0);
  std::bitset<MAX_SCOPE_NUM> scopeFlags;
  scopeFlags.set(1);

  SuperKernelScopeInfo originalScope;
  originalScope.SetNodes({kernel, defaultNode});
  originalScope.SetScopeBitFlags(scopeFlags);
  ScopeBreakInfo originalBreakInfo;
  originalBreakInfo.SetReason(ScopeBreakReason::UNFUSIBLE_NODE)
      .SetTriggerNode(3, 0)
      .SetDetail("unfused node causes scope break")
      .SetParentScopeId(1);
  originalScope.SetBreakInfo(originalBreakInfo);

  std::vector<SuperKernelScopeInfo> previousScopes;
  previousScopes.push_back(std::move(originalScope));
  ScopeSplitResultReporter reporter;
  reporter.CaptureResplitScopes(previousScopes);

  SuperKernelScopeInfo matchedScope;
  matchedScope.SetNodes({kernel, defaultNode});
  matchedScope.SetScopeBitFlags(scopeFlags);
  SuperKernelScopeInfo defaultChangedScope;
  defaultChangedScope.SetNodes({kernel});
  defaultChangedScope.SetScopeBitFlags(scopeFlags);
  SuperKernelScopeInfo flagsChangedScope;
  flagsChangedScope.SetNodes({kernel, defaultNode});
  scopeFlags.set(2);
  flagsChangedScope.SetScopeBitFlags(scopeFlags);

  std::vector<SuperKernelScopeInfo> currentScopes;
  currentScopes.push_back(std::move(matchedScope));
  currentScopes.push_back(std::move(defaultChangedScope));
  currentScopes.push_back(std::move(flagsChangedScope));
  reporter.RestoreResplitBreakInfos(currentScopes);

  EXPECT_EQ(currentScopes[0].GetBreakInfo().GetReason(), ScopeBreakReason::UNFUSIBLE_NODE);
  EXPECT_EQ(currentScopes[0].GetBreakInfo().GetParentScopeId(), originalBreakInfo.GetParentScopeId());
  EXPECT_EQ(currentScopes[1].GetBreakInfo().GetReason(), ScopeBreakReason::NONE);
  EXPECT_EQ(currentScopes[2].GetBreakInfo().GetReason(), ScopeBreakReason::NONE);
}

TEST_F(SuperKernelScopeSplitterTest, ScopeSplitResultReporter_InheritedBreakPreservesCompleteBreakInfo) {
  ScopeBreakInfo sourceBreakInfo;
  sourceBreakInfo.SetReason(ScopeBreakReason::DEADLOCK_DETECTED)
      .SetTriggerNode(3, 2)
      .SetSyncAllNodeIds({1, 2, 3})
      .SetDetail("deadlock causes scope break")
      .SetParentScopeId(1);
  SuperKernelScopeInfo inheritedScope;
  ScopeSplitResultReporter reporter;

  reporter.ReportInheritedBreak(inheritedScope, sourceBreakInfo);

  const auto &inheritedBreakInfo = inheritedScope.GetBreakInfo();
  EXPECT_EQ(inheritedBreakInfo.GetReason(), sourceBreakInfo.GetReason());
  EXPECT_EQ(inheritedBreakInfo.GetTriggerNodeId(), sourceBreakInfo.GetTriggerNodeId());
  EXPECT_EQ(inheritedBreakInfo.GetTriggerStreamIdx(), sourceBreakInfo.GetTriggerStreamIdx());
  EXPECT_EQ(inheritedBreakInfo.GetSyncAllNodeIds(), sourceBreakInfo.GetSyncAllNodeIds());
  EXPECT_EQ(inheritedBreakInfo.GetDetail(), sourceBreakInfo.GetDetail());
  EXPECT_EQ(inheritedBreakInfo.GetParentScopeId(), sourceBreakInfo.GetParentScopeId());
}
