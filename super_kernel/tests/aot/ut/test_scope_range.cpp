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
 * \file test_scope_range.cpp
 * \brief Unit tests for verifying scope tagging functionality on nodes
 */

#include <gtest/gtest.h>
#include <memory>
#include <bitset>

#define private public
#define protected public
#include "sk_node.h"
#include "sk_types.h"
#include "sk_graph.h"

class TestScopeSplitRange : public ::testing::Test {
 protected:
  std::unique_ptr<SuperKernelGraph> graph;
  void SetUp() override {
    graph = std::make_unique<SuperKernelGraph>();
  }

  void TearDown() override {
    // Cleanup code if needed
  }

  // Helper function to create a kernel node
  SuperKernelKernelNode *CreateKernelNode(uint64_t nodeId = 1, uint32_t streamIdx = 0, bool isFusible = false) {
    auto task = std::make_unique<aclrtTask>();
    auto node = std::make_unique<SuperKernelKernelNode>(std::move(task), ACL_MODEL_RI_TASK_KERNEL, nodeId, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->SetIsFusible(isFusible);
    SuperKernelKernelNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  // Helper function to create a kernel node
  SuperKernelKernelNode *CreateScopeKernelNode(uint64_t nodeId = 1, uint32_t streamIdx = 0, std::string scopeName = "",
                                               bool isFusible = true, bool isScopeNode = true, bool isBegin = false,
                                               bool isEnd = false, bool isPlaceholder = false) {
    auto task = std::make_unique<aclrtTask>();
    auto node = std::make_unique<SuperKernelKernelNode>(std::move(task), ACL_MODEL_RI_TASK_KERNEL, nodeId, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeId(nodeId);
    node->scopeName = scopeName;
    node->isFusible = isFusible;
    node->isScopeNode = isScopeNode;
    node->isScopeBegin = isBegin;
    node->isScopeEnd = isEnd;
    node->isPlaceholder = isPlaceholder;
    SuperKernelKernelNode *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }
};
// Test 1: Verify node with single scope
TEST_F(TestScopeSplitRange, TestNodeWithSingleScope) {
  auto node1 = CreateKernelNode(1, 0, true);
  auto scopeBeginNode = CreateScopeKernelNode(2, 0, "scopeA", true, true, true, false, false);
  auto scopeNodePlaceholder_1 = CreateScopeKernelNode(3, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholder_2 = CreateScopeKernelNode(4, 0, "scopeA", true, true, false, false, true);
  auto node2 = CreateKernelNode(5, 0, true);
  auto node3 = CreateKernelNode(6, 0, false);
  auto scopeNodePlaceholder_3 = CreateScopeKernelNode(7, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholder_4 = CreateScopeKernelNode(8, 0, "scopeA", true, true, false, false, true);
  auto scopeEndNode = CreateScopeKernelNode(9, 0, "scopeA", true, true, false, true, false);
  auto node4 = CreateKernelNode(10, 0, true);
  graph->scopeNameToIdx["scopeA"] = 0;

  graph->UpdateNodeScopeBitFlags();
  EXPECT_TRUE(node1->GetScopeBitFlags().none());
  EXPECT_FALSE(node1->IsFusible());
  EXPECT_TRUE(scopeBeginNode->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeNodePlaceholder_1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeNodePlaceholder_2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node2->IsFusible());
  EXPECT_FALSE(node3->IsFusible());
  EXPECT_TRUE(scopeNodePlaceholder_3->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeNodePlaceholder_4->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeEndNode->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node4->GetScopeBitFlags().none());
  EXPECT_FALSE(node4->IsFusible());
}

TEST_F(TestScopeSplitRange, MemoryEventScopePolicyAppliesAfterPostProcess) {
  constexpr uint64_t kEventId = 0x301;
  auto writeNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0,
                                                           INVALID_STREAM_ID, INVALID_TASK_ID);
  writeNode->SetNodeId(1);
  writeNode->SetNodeType(SkNodeType::NODE_MEMORY_WRITE);
  writeNode->SetIsFusible(false);
  writeNode->nodeInfos.syncInfos.eventId = kEventId;
  writeNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1001);
  writeNode->nodeInfos.syncInfos.memoryValue = 7;
  SuperKernelMemoryNode *writeNodePtr = writeNode.get();
  graph->graphMap[1] = std::move(writeNode);
  graph->memoryToNodes[kEventId].writeNodeIdList.insert(1);

  CreateScopeKernelNode(2, 0, "scopeA", true, true, true, false, false);
  CreateScopeKernelNode(3, 0, "scopeA", true, true, false, true, false);
  graph->scopeNameToIdx["scopeA"] = 0;

  ASSERT_TRUE(graph->PostProcessMemoryNode());
  EXPECT_EQ(writeNodePtr->GetNodeType(), SkNodeType::NODE_NOTIFY);
  ASSERT_TRUE(writeNodePtr->IsFusible());

  ASSERT_TRUE(graph->UpdateNodeScopeBitFlags());
  EXPECT_FALSE(writeNodePtr->IsFusible());
  EXPECT_EQ(writeNodePtr->GetFusionFailReason(), FusionFailReason::NOT_IN_SCOPE);
}

// Test 2: Verify node with unfusible scope (scopeName传nullptr时不可融合）
TEST_F(TestScopeSplitRange, TestNodeWithUnfusibleScope) {
  auto scopeBeginNode = CreateScopeKernelNode(1, 0, "", false, true, true, false, false);
  auto scopeNodePlaceholder_1 = CreateScopeKernelNode(2, 0, "", false, true, false, false, true);
  auto scopeNodePlaceholder_2 = CreateScopeKernelNode(3, 0, "", false, true, false, false, true);
  auto node1 = CreateKernelNode(4, 0, true);
  auto scopeNodePlaceholder_3 = CreateScopeKernelNode(5, 0, "", false, true, false, false, true);
  auto scopeNodePlaceholder_4 = CreateScopeKernelNode(6, 0, "", false, true, false, false, true);
  auto scopeEndNode = CreateScopeKernelNode(7, 0, "", false, true, false, true, false);
  auto node2 = CreateKernelNode(8, 0, true);

  graph->UpdateNodeScopeBitFlags();

  // Scope nodes are always fusible (they mark fusion boundaries)
  EXPECT_TRUE(scopeBeginNode->IsFusible());
  EXPECT_TRUE(scopeNodePlaceholder_1->IsFusible());
  EXPECT_TRUE(scopeNodePlaceholder_2->IsFusible());
  EXPECT_FALSE(node1->IsFusible());  // node inside unfusible scope
  EXPECT_TRUE(scopeNodePlaceholder_3->IsFusible());
  EXPECT_TRUE(scopeNodePlaceholder_4->IsFusible());
  EXPECT_TRUE(scopeEndNode->IsFusible());
  EXPECT_TRUE(node2->IsFusible());
}

// Test 3: Verify node with multi fusible scope has intersection (非子集)
TEST_F(TestScopeSplitRange, TestNodeWithMultiScopeIntersection) {
  graph->scopeNameToIdx["scopeA"] = 0;
  graph->scopeNameToIdx["scopeB"] = 1;

  // scopeA begin
  auto scopeBeginA = CreateScopeKernelNode(1, 0, "scopeA", true, true, true, false, false);
  auto scopeNodePlaceholderA_1 = CreateScopeKernelNode(2, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_2 = CreateScopeKernelNode(3, 0, "scopeA", true, true, false, false, true);

  // scopeB begin (intersection, not subset)
  auto scopeBeginB = CreateScopeKernelNode(4, 0, "scopeB", true, true, true, false, false);
  auto scopeNodePlaceholderB_1 = CreateScopeKernelNode(5, 0, "scopeB", true, true, false, false, true);
  auto scopeNodePlaceholderB_2 = CreateScopeKernelNode(6, 0, "scopeB", true, true, false, false, true);

  // node in intersection (scopeA + scopeB)
  auto node1 = CreateKernelNode(7, 0, true);

  // scopeA end
  auto scopeNodePlaceholderA_3 = CreateScopeKernelNode(8, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_4 = CreateScopeKernelNode(9, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA = CreateScopeKernelNode(10, 0, "scopeA", true, true, false, true, false);

  // node still in scopeB
  auto node2 = CreateKernelNode(11, 0, true);

  // scopeB end
  auto scopeNodePlaceholderB_3 = CreateScopeKernelNode(12, 0, "scopeB", true, true, false, false, true);
  auto scopeNodePlaceholderB_4 = CreateScopeKernelNode(13, 0, "scopeB", true, true, false, false, true);
  auto scopeEndB = CreateScopeKernelNode(14, 0, "scopeB", true, true, false, true, false);

  graph->UpdateNodeScopeBitFlags();

  // node1 is in both scopeA and scopeB (intersection)
  EXPECT_TRUE(node1->IsFusible());
  EXPECT_EQ(node1->GetScopeBitFlags().count(), 2);
  EXPECT_TRUE(node1->GetScopeBitFlags().test(0));  // scopeA
  EXPECT_TRUE(node1->GetScopeBitFlags().test(1));  // scopeB
}

// Test 4: Verify real scenario with mixed fusible/unfusible scope interleaving
TEST_F(TestScopeSplitRange, TestRealScenarioWithMixedScopeInterleaving) {
  graph->scopeNameToIdx["sk2"] = 0;

  // Node 0: scope begin for unfusible scope (empty name)
  auto node0 = CreateScopeKernelNode(0, 0, "", false, true, true, false, false);

  // Node 1-2: placeholders in unfusible scope
  auto node1 = CreateScopeKernelNode(1, 0, "", false, true, false, false, true);
  auto node2 = CreateScopeKernelNode(2, 0, "", false, true, false, false, true);

  // Node 3: regular node in unfusible scope
  auto node3 = CreateKernelNode(3, 0, true);

  // Node 4: scope begin for fusible scope "sk2" (nested in unfusible scope)
  auto node4 = CreateScopeKernelNode(4, 0, "sk2", true, true, true, false, false);

  // Node 5-6: placeholders in fusible scope "sk2"
  auto node5 = CreateScopeKernelNode(5, 0, "sk2", true, true, false, false, true);
  auto node6 = CreateScopeKernelNode(6, 0, "sk2", true, true, false, false, true);

  // Node 7: placeholder in both scopes (unfusible + fusible)
  auto node7 = CreateScopeKernelNode(7, 0, "", false, true, false, false, true);

  // Node 8: placeholder in both scopes
  auto node8 = CreateScopeKernelNode(8, 0, "", false, true, false, false, true);

  // Node 9: scope end for unfusible scope
  auto node9 = CreateScopeKernelNode(9, 0, "", false, true, false, true, false);

  // Node 10: regular node still in fusible scope "sk2"
  auto node10 = CreateKernelNode(10, 0, true);

  // Node 11-12: placeholders in fusible scope "sk2"
  auto node11 = CreateScopeKernelNode(11, 0, "sk2", true, true, false, false, true);
  auto node12 = CreateScopeKernelNode(12, 0, "sk2", true, true, false, false, true);

  // Node 13: scope end for fusible scope "sk2"
  auto node13 = CreateScopeKernelNode(13, 0, "sk2", true, true, false, true, false);

  graph->UpdateNodeScopeBitFlags();

  // Node 0: unfusible scope begin - scope nodes are always fusible
  EXPECT_TRUE(node0->IsFusible());
  EXPECT_TRUE(node0->GetScopeBitFlags().none());

  // Node 1-2: inside unfusible scope, scope nodes are always fusible
  EXPECT_TRUE(node1->IsFusible());
  EXPECT_TRUE(node1->GetScopeBitFlags().none());
  EXPECT_TRUE(node2->IsFusible());
  EXPECT_TRUE(node2->GetScopeBitFlags().none());

  // Node 3: regular node in unfusible scope
  EXPECT_FALSE(node3->IsFusible());
  EXPECT_TRUE(node3->GetScopeBitFlags().none());

  // Node 4: fusible scope begin, but still inside unfusible scope
  EXPECT_TRUE(node4->IsFusible());
  EXPECT_TRUE(node4->GetScopeBitFlags().test(0));

  // Node 5-6: inside both unfusible and fusible scopes, scope nodes are always fusible
  EXPECT_TRUE(node5->IsFusible());
  EXPECT_TRUE(node5->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node6->IsFusible());
  EXPECT_TRUE(node6->GetScopeBitFlags().test(0));

  // Node 7-8: inside both scopes, scope nodes are always fusible
  EXPECT_TRUE(node7->IsFusible());
  EXPECT_TRUE(node7->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node8->IsFusible());
  EXPECT_TRUE(node8->GetScopeBitFlags().test(0));

  // Node 9: scope end for unfusible scope - scope nodes are always fusible
  EXPECT_TRUE(node9->IsFusible());
  EXPECT_TRUE(node9->GetScopeBitFlags().test(0));

  // Node 10: regular node, only in fusible scope now (unfusible scope ended)
  EXPECT_TRUE(node10->IsFusible());
  EXPECT_TRUE(node10->GetScopeBitFlags().test(0));

  // Node 11-12: only in fusible scope, scope nodes are always fusible
  EXPECT_TRUE(node11->IsFusible());
  EXPECT_TRUE(node11->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node12->IsFusible());
  EXPECT_TRUE(node12->GetScopeBitFlags().test(0));

  // Node 13: scope end for fusible scope
  EXPECT_TRUE(node13->IsFusible());
  EXPECT_TRUE(node13->GetScopeBitFlags().test(0));
}

// Test 5: Verify node with multi scope has intersection (子集)
TEST_F(TestScopeSplitRange, TestNodeWithMultiScopeSubset) {
  graph->scopeNameToIdx["scopeA"] = 0;
  graph->scopeNameToIdx["scopeB"] = 1;

  // scopeA begin (outer scope)
  auto scopeBeginA = CreateScopeKernelNode(1, 0, "scopeA", true, true, true, false, false);
  auto scopeNodePlaceholderA_1 = CreateScopeKernelNode(2, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_2 = CreateScopeKernelNode(3, 0, "scopeA", true, true, false, false, true);

  // scopeB begin (inner scope, subset of scopeA)
  auto scopeBeginB = CreateScopeKernelNode(4, 0, "scopeB", true, true, true, false, false);
  auto scopeNodePlaceholderB_1 = CreateScopeKernelNode(5, 0, "scopeB", true, true, false, false, true);
  auto scopeNodePlaceholderB_2 = CreateScopeKernelNode(6, 0, "scopeB", true, true, false, false, true);

  // node in both scopeA and scopeB (scopeB is subset of scopeA)
  auto node1 = CreateKernelNode(7, 0, true);

  // scopeB end
  auto scopeNodePlaceholderB_3 = CreateScopeKernelNode(8, 0, "scopeB", true, true, false, false, true);
  auto scopeNodePlaceholderB_4 = CreateScopeKernelNode(9, 0, "scopeB", true, true, false, false, true);
  auto scopeEndB = CreateScopeKernelNode(10, 0, "scopeB", true, true, false, true, false);

  // node only in scopeA (scopeB already ended)
  auto node2 = CreateKernelNode(11, 0, true);

  // scopeA end
  auto scopeNodePlaceholderA_3 = CreateScopeKernelNode(12, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_4 = CreateScopeKernelNode(13, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA = CreateScopeKernelNode(14, 0, "scopeA", true, true, false, true, false);

  graph->UpdateNodeScopeBitFlags();

  // node1 is in both scopeA and scopeB (scopeB is subset of scopeA)
  EXPECT_TRUE(node1->IsFusible());
  EXPECT_EQ(node1->GetScopeBitFlags().count(), 2);
  EXPECT_TRUE(node1->GetScopeBitFlags().test(0));  // scopeA
  EXPECT_TRUE(node1->GetScopeBitFlags().test(1));  // scopeB

  // node2 is only in scopeA (scopeB already ended)
  EXPECT_TRUE(node2->IsFusible());
  EXPECT_EQ(node2->GetScopeBitFlags().count(), 1);
  EXPECT_TRUE(node2->GetScopeBitFlags().test(0));  // only scopeA
}

// Test 8: Verify multiple scopes with the same name can be executed sequentially
TEST_F(TestScopeSplitRange, TestMultipleSequentialSameScope) {
  graph->scopeNameToIdx["scopeA"] = 0;

  // First scopeA
  auto scopeBeginA1 = CreateScopeKernelNode(1, 0, "scopeA", true, true, true, false, false);
  auto placeholderA1_1 = CreateScopeKernelNode(2, 0, "scopeA", true, true, false, false, true);
  auto node1 = CreateKernelNode(3, 0, true);
  auto placeholderA1_2 = CreateScopeKernelNode(4, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA1 = CreateScopeKernelNode(5, 0, "scopeA", true, true, false, true, false);

  // Node between two scopeA instances
  auto nodeBetween = CreateKernelNode(6, 0, true);

  // Second scopeA (same name, sequential execution)
  auto scopeBeginA2 = CreateScopeKernelNode(7, 0, "scopeA", true, true, true, false, false);
  auto placeholderA2_1 = CreateScopeKernelNode(8, 0, "scopeA", true, true, false, false, true);
  auto node2 = CreateKernelNode(9, 0, true);
  auto placeholderA2_2 = CreateScopeKernelNode(10, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA2 = CreateScopeKernelNode(11, 0, "scopeA", true, true, false, true, false);

  // Node after all scopes
  auto nodeAfter = CreateKernelNode(12, 0, true);

  graph->UpdateNodeScopeBitFlags();

  // First scopeA: nodes should be in scopeA
  EXPECT_TRUE(scopeBeginA1->IsFusible());
  EXPECT_TRUE(scopeBeginA1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(placeholderA1_1->IsFusible());
  EXPECT_TRUE(placeholderA1_1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node1->IsFusible());
  EXPECT_TRUE(node1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(placeholderA1_2->IsFusible());
  EXPECT_TRUE(placeholderA1_2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeEndA1->IsFusible());
  EXPECT_TRUE(scopeEndA1->GetScopeBitFlags().test(0));

  // Node between two scopeA instances: should NOT be in scopeA, marked unfusible (outside named scope)
  EXPECT_FALSE(nodeBetween->IsFusible());
  EXPECT_TRUE(nodeBetween->GetScopeBitFlags().none());

  // Second scopeA: nodes should be in scopeA again
  EXPECT_TRUE(scopeBeginA2->IsFusible());
  EXPECT_TRUE(scopeBeginA2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(placeholderA2_1->IsFusible());
  EXPECT_TRUE(placeholderA2_1->GetScopeBitFlags().test(0));
  EXPECT_TRUE(node2->IsFusible());
  EXPECT_TRUE(node2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(placeholderA2_2->IsFusible());
  EXPECT_TRUE(placeholderA2_2->GetScopeBitFlags().test(0));
  EXPECT_TRUE(scopeEndA2->IsFusible());
  EXPECT_TRUE(scopeEndA2->GetScopeBitFlags().test(0));

  // Node after all scopes: should NOT be in scopeA, marked unfusible (outside named scope)
  EXPECT_FALSE(nodeAfter->IsFusible());
  EXPECT_TRUE(nodeAfter->GetScopeBitFlags().none());
}

// Test 6: Verify node with multi scope has intersection (非子集) 存在不可融合的scope
TEST_F(TestScopeSplitRange, TestNodeWithMultiScopeIntersectionUnfusible) {
  graph->scopeNameToIdx["scopeA"] = 0;

  // scopeA begin (fusible)
  auto scopeBeginA = CreateScopeKernelNode(1, 0, "scopeA", true, true, true, false, false);
  auto scopeNodePlaceholderA_1 = CreateScopeKernelNode(2, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_2 = CreateScopeKernelNode(3, 0, "scopeA", true, true, false, false, true);

  // scopeB begin (unfusible, intersection with scopeA)
  auto scopeBeginB = CreateScopeKernelNode(4, 0, "", false, true, true, false, false);
  auto scopeNodePlaceholderB_1 = CreateScopeKernelNode(5, 0, "", true, true, false, false, true);
  auto scopeNodePlaceholderB_2 = CreateScopeKernelNode(6, 0, "", true, true, false, false, true);

  // node in intersection (scopeA + unfusible scopeB)
  auto node1 = CreateKernelNode(7, 0, true);

  // scopeA end
  auto scopeNodePlaceholderA_3 = CreateScopeKernelNode(8, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_4 = CreateScopeKernelNode(9, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA = CreateScopeKernelNode(10, 0, "scopeA", true, true, false, true, false);

  // node still in unfusible scopeB
  auto node2 = CreateKernelNode(11, 0, true);

  // scopeB end (unfusible)
  auto scopeNodePlaceholderB_3 = CreateScopeKernelNode(12, 0, "", true, true, false, false, true);
  auto scopeNodePlaceholderB_4 = CreateScopeKernelNode(13, 0, "", true, true, false, false, true);
  auto scopeEndB = CreateScopeKernelNode(14, 0, "", false, true, false, true, false);

  graph->UpdateNodeScopeBitFlags();

  // node1 is in scopeA + unfusible scopeB
  EXPECT_FALSE(node1->IsFusible());                 // unfusible due to scopeB
  EXPECT_EQ(node1->GetScopeBitFlags().count(), 1);  // only fusible scopes
  EXPECT_TRUE(node1->GetScopeBitFlags().test(0));   // only scopeA

  // node2 is only in unfusible scopeB
  EXPECT_FALSE(node2->IsFusible());
  EXPECT_EQ(node2->GetScopeBitFlags().count(), 0);  // no fusible scopes
}

// Test 7: Verify node with multi scope has intersection (子集) 存在不可融合的scope
TEST_F(TestScopeSplitRange, TestNodeWithMultiScopeSubsetUnfusible) {
  graph->scopeNameToIdx["scopeA"] = 0;

  // scopeA begin (outer fusible scope)
  auto scopeBeginA = CreateScopeKernelNode(1, 0, "scopeA", true, true, true, false, false);
  auto scopeNodePlaceholderA_1 = CreateScopeKernelNode(2, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_2 = CreateScopeKernelNode(3, 0, "scopeA", true, true, false, false, true);

  // scopeB begin (inner unfusible scope, subset of scopeA)
  auto scopeBeginB = CreateScopeKernelNode(4, 0, "", false, true, true, false, false);
  auto scopeNodePlaceholderB_1 = CreateScopeKernelNode(5, 0, "", true, true, false, false, true);
  auto scopeNodePlaceholderB_2 = CreateScopeKernelNode(6, 0, "", true, true, false, false, true);

  // node in both scopeA and unfusible scopeB (scopeB is subset of scopeA)
  auto node1 = CreateKernelNode(7, 0, true);

  // scopeB end (unfusible)
  auto scopeNodePlaceholderB_3 = CreateScopeKernelNode(8, 0, "", true, true, false, false, true);
  auto scopeNodePlaceholderB_4 = CreateScopeKernelNode(9, 0, "", true, true, false, false, true);
  auto scopeEndB = CreateScopeKernelNode(10, 0, "", false, true, false, true, false);

  // node only in scopeA (scopeB already ended)
  auto node2 = CreateKernelNode(11, 0, true);

  // scopeA end
  auto scopeNodePlaceholderA_3 = CreateScopeKernelNode(12, 0, "scopeA", true, true, false, false, true);
  auto scopeNodePlaceholderA_4 = CreateScopeKernelNode(13, 0, "scopeA", true, true, false, false, true);
  auto scopeEndA = CreateScopeKernelNode(14, 0, "scopeA", true, true, false, true, false);

  graph->UpdateNodeScopeBitFlags();

  // node1 is in scopeA + unfusible scopeB (scopeB is subset of scopeA)
  EXPECT_FALSE(node1->IsFusible());                 // unfusible due to scopeB
  EXPECT_EQ(node1->GetScopeBitFlags().count(), 1);  // only fusible scopeA
  EXPECT_TRUE(node1->GetScopeBitFlags().test(0));   // only scopeA

  // node2 is only in scopeA (scopeB already ended)
  EXPECT_TRUE(node2->IsFusible());
  EXPECT_EQ(node2->GetScopeBitFlags().count(), 1);
  EXPECT_TRUE(node2->GetScopeBitFlags().test(0));  // only scopeA
}
