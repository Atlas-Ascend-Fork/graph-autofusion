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
#include <memory>
#include <vector>
#include <algorithm>
#include <bitset>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_optimizer.h"
#include "sk_options_manager.h"
#include "sk_node.h"
#include "sk_candidate_heap.h"

class SkOptimizerTaskReorderTest : public testing::Test {
 protected:
  void SetUp() override {
    graph = std::make_unique<SuperKernelGraph>();
    opts = std::make_unique<SuperKernelOptionsManager>();
    optimizer = std::make_unique<SuperKernelOptimizer>(*opts);
  }

  SuperKernelBaseNode *CreateKernelNode(uint64_t nodeId, uint32_t streamIdx) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    node->nodeInfos.kernelInfos.funcName = "k";
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateWaitNode(uint64_t nodeId, uint32_t streamIdx) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  std::vector<uint64_t> ToNodeIds(const std::vector<SuperKernelBaseNode *> &nodes) {
    std::vector<uint64_t> nodeIds;
    nodeIds.reserve(nodes.size());
    for (const auto *node : nodes) {
      nodeIds.push_back(node == nullptr ? INVALID_TASK_ID : node->GetNodeId());
    }
    return nodeIds;
  }

  void EnableCustomizeQueueReorder() {
    auto option = std::make_unique<NumberOptOption>("auto_op_parallel", aclskOptionType::AUTO_OP_PARALLEL,
                                                    static_cast<uint32_t>(SkHeapType::CUSTOMIZE_QUEUE), 0, 0xFFFFFFFF);
    opts->AddOption(std::move(option));
  }

  std::unique_ptr<SuperKernelGraph> graph;
  std::unique_ptr<SuperKernelOptionsManager> opts;
  std::unique_ptr<SuperKernelOptimizer> optimizer;
};

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_DefaultDisabled_KeepOriginalOrder) {
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 0),
      CreateWaitNode(2, 1),
      CreateKernelNode(3, 1),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_WaitMovesBeforeLaterSameStreamKernel) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 0),
      CreateWaitNode(2, 1),
      CreateKernelNode(3, 1),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest,
       ReorderWaitNodesForTaskBuild_CustomizeQueue_WaitWithoutLaterSameStreamKernelStaysInPlace) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 0),
      CreateWaitNode(2, 1),
      CreateKernelNode(3, 0),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3}));
}

TEST_F(SkOptimizerTaskReorderTest,
       ReorderWaitNodesForTaskBuild_CustomizeQueue_ConsecutiveWaitsSameStreamKeepRelativeOrder) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateWaitNode(10, 1),
      CreateKernelNode(20, 0),
      CreateWaitNode(11, 1),
      CreateKernelNode(12, 1),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{20, 10, 11, 12}));
}

TEST_F(SkOptimizerTaskReorderTest,
       ReorderWaitNodesForTaskBuild_CustomizeQueue_ConsecutiveWaitsDifferentStreamsAttachToOwnKernel) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateWaitNode(10, 1),   CreateWaitNode(11, 2),   CreateKernelNode(20, 0),
      CreateKernelNode(12, 1), CreateKernelNode(13, 2),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{20, 10, 12, 11, 13}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_OutOfOrderTargetsDoNotFlushPrefix) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateWaitNode(10, 1),
      CreateWaitNode(11, 2),
      CreateKernelNode(13, 2),
      CreateKernelNode(12, 1),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{11, 13, 10, 12}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_AllNodesSameStream_KeepOriginalOrder) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 3), CreateWaitNode(2, 3),   CreateKernelNode(3, 3),
      CreateWaitNode(4, 3),   CreateKernelNode(5, 3),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 2, 3, 4, 5}));
}

TEST_F(SkOptimizerTaskReorderTest,
       ReorderWaitNodesForTaskBuild_CustomizeQueue_MultipleWaitsTargetSameKernelKeepRelativeOrder) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 0), CreateWaitNode(2, 2),   CreateKernelNode(3, 1),
      CreateWaitNode(4, 2),   CreateKernelNode(5, 2),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 3, 2, 4, 5}));
}

TEST_F(SkOptimizerTaskReorderTest, ReorderWaitNodesForTaskBuild_CustomizeQueue_NonWaitRelativeOrderPreserved) {
  EnableCustomizeQueueReorder();
  std::vector<SuperKernelBaseNode *> taskNodes = {
      CreateKernelNode(1, 0), CreateWaitNode(2, 1),   CreateKernelNode(3, 0),
      CreateKernelNode(4, 1), CreateKernelNode(5, 0),
  };

  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = optimizer->ReorderWaitNodesForTaskBuild(taskNodes);

  EXPECT_EQ(ToNodeIds(reorderedTaskNodes), (std::vector<uint64_t>{1, 3, 2, 4, 5}));
}

/**
 * @brief Test case: only op4-op9 are fusible, others are not
 *
 * Graph structure:
 *   stream1: op1 -> op4 -> op5 -> op10
 *   stream2: op2 -> op6 -> op7 -> op11
 *   stream3: op3 -> op8 -> op9 -> op12
 *
 * op4-op9 are fusible (SetIsFusible(true))
 * op1, op2, op3, op10, op11, op12 are NOT fusible (default isFusible=false)
 *
 * Expected: SplitGraph should split op4-op9 into one scope
 *          In Process() loop at line 310, this scope should be processed
 */
TEST_F(SkOptimizerTaskReorderTest, ThreeStreamParallel_OnlyOp4ToOp9Fusible) {
  // Create nodes for stream1: op1, op4, op5, op10
  auto *op1 = CreateKernelNode(1, 0);
  auto *op4 = CreateKernelNode(4, 0);
  auto *op5 = CreateKernelNode(5, 0);
  auto *op10 = CreateKernelNode(10, 0);

  // Create nodes for stream2: op2, op6, op7, op11
  auto *op2 = CreateKernelNode(2, 1);
  auto *op6 = CreateKernelNode(6, 1);
  auto *op7 = CreateKernelNode(7, 1);
  auto *op11 = CreateKernelNode(11, 1);

  // Create nodes for stream3: op3, op8, op9, op12
  auto *op3 = CreateKernelNode(3, 2);
  auto *op8 = CreateKernelNode(8, 2);
  auto *op9 = CreateKernelNode(9, 2);
  auto *op12 = CreateKernelNode(12, 2);

  // Only op4-op9 are fusible (others default to isFusible=false)
  op4->SetIsFusible(true);
  op5->SetIsFusible(true);
  op6->SetIsFusible(true);
  op7->SetIsFusible(true);
  op8->SetIsFusible(true);
  op9->SetIsFusible(true);
  for (auto *node : {op4, op5, op6, op7, op8, op9}) {
    node->nodeInfos.kernelInfos.kernelType = SkKernelType::AIV_ONLY;
    node->nodeInfos.kernelInfos.numBlocks = 1;
    node->nodeInfos.kernelInfos.vecNum = 1;
  }

  // Setup stream connections
  graph->streams.clear();
  graph->headNodes.clear();

  // Setup stream 0: op1 -> op4 -> op5 -> op10
  graph->streams.emplace_back();
  graph->headNodes.push_back(1);
  op1->SetNextNodeId(4);
  op1->SetPreNodeId(INVALID_TASK_ID);
  op1->nodeIdxInStream = 0;
  op1->streamIdxInGraph = 0;

  op4->SetNextNodeId(5);
  op4->SetPreNodeId(1);
  op4->nodeIdxInStream = 1;
  op4->streamIdxInGraph = 0;

  op5->SetNextNodeId(10);
  op5->SetPreNodeId(4);
  op5->nodeIdxInStream = 2;
  op5->streamIdxInGraph = 0;

  op10->SetNextNodeId(INVALID_TASK_ID);
  op10->SetPreNodeId(5);
  op10->nodeIdxInStream = 3;
  op10->streamIdxInGraph = 0;

  // Setup stream 1: op2 -> op6 -> op7 -> op11
  graph->streams.emplace_back();
  graph->headNodes.push_back(2);
  op2->SetNextNodeId(6);
  op2->SetPreNodeId(INVALID_TASK_ID);
  op2->nodeIdxInStream = 0;
  op2->streamIdxInGraph = 1;

  op6->SetNextNodeId(7);
  op6->SetPreNodeId(2);
  op6->nodeIdxInStream = 1;
  op6->streamIdxInGraph = 1;

  op7->SetNextNodeId(11);
  op7->SetPreNodeId(6);
  op7->nodeIdxInStream = 2;
  op7->streamIdxInGraph = 1;

  op11->SetNextNodeId(INVALID_TASK_ID);
  op11->SetPreNodeId(7);
  op11->nodeIdxInStream = 3;
  op11->streamIdxInGraph = 1;

  // Setup stream 2: op3 -> op8 -> op9 -> op12
  graph->streams.emplace_back();
  graph->headNodes.push_back(3);
  op3->SetNextNodeId(8);
  op3->SetPreNodeId(INVALID_TASK_ID);
  op3->nodeIdxInStream = 0;
  op3->streamIdxInGraph = 2;

  op8->SetNextNodeId(9);
  op8->SetPreNodeId(3);
  op8->nodeIdxInStream = 1;
  op8->streamIdxInGraph = 2;

  op9->SetNextNodeId(12);
  op9->SetPreNodeId(8);
  op9->nodeIdxInStream = 2;
  op9->streamIdxInGraph = 2;

  op12->SetNextNodeId(INVALID_TASK_ID);
  op12->SetPreNodeId(9);
  op12->nodeIdxInStream = 3;
  op12->streamIdxInGraph = 2;

  // Call Process method
  // Process will call SplitGraph() which should split op4-op9 into one scope
  bool result = optimizer->Process(*graph);

  // Log result for debugging
  SK_LOGI("Process result: %d", result);

  // Get scope infos after processing
  const auto &scopeInfos = optimizer->GetScopeInfos();
  SK_LOGI("Number of scopes: %zu", scopeInfos.size());

  // Print scope details for debugging
  for (size_t i = 0; i < scopeInfos.size(); ++i) {
    const auto &scope = scopeInfos[i];
    SK_LOGI("Scope %zu: scopeId=%u, nodeCount=%zu", i, scope.GetScopeId(), scope.GetNodes().size());
    for (const auto *node : scope.GetNodes()) {
      if (node != nullptr) {
        SK_LOGI("  Node: %s", node->Format().c_str());
      }
    }
  }

  // Verify that op4-op9 are in one scope
  bool foundScopeWithOp4ToOp9 = false;
  for (size_t i = 0; i < scopeInfos.size(); ++i) {
    const auto &scope = scopeInfos[i];
    const auto &nodes = scope.GetNodes();

    bool hasOp4 = false, hasOp5 = false, hasOp6 = false;
    bool hasOp7 = false, hasOp8 = false, hasOp9 = false;

    for (const auto *node : nodes) {
      if (node == nullptr) continue;
      uint64_t id = node->GetNodeId();
      if (id == 4) hasOp4 = true;
      if (id == 5) hasOp5 = true;
      if (id == 6) hasOp6 = true;
      if (id == 7) hasOp7 = true;
      if (id == 8) hasOp8 = true;
      if (id == 9) hasOp9 = true;
    }

    if (hasOp4 && hasOp5 && hasOp6 && hasOp7 && hasOp8 && hasOp9) {
      foundScopeWithOp4ToOp9 = true;
      SK_LOGI("Found scope with op4-op9 at index %zu", i);
      break;
    }
  }

  EXPECT_TRUE(foundScopeWithOp4ToOp9);
}
