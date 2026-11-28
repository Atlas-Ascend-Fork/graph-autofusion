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
#include "sk_task_builder.h"
#include "sk_task_lowering.h"
#include "sk_event_recorder.h"
#include "sk_queue_type_resolver.h"
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

  SuperKernelBaseNode *CreateKernelNode(uint64_t nodeId, uint32_t streamIdx,
                                        SkKernelType kernelType = SkKernelType::AIC_ONLY) {
    auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    node->nodeInfos.kernelInfos.funcName = "k";
    node->nodeInfos.kernelInfos.kernelType = kernelType;
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

  SuperKernelBaseNode *CreateNotifyNode(uint64_t nodeId, uint32_t streamIdx) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_NOTIFY);
    node->SetNodeId(nodeId);
    node->SetPreNodeId(INVALID_TASK_ID);
    node->SetNextNodeId(INVALID_TASK_ID);
    auto *ptr = node.get();
    graph->graphMap[nodeId] = std::move(node);
    return ptr;
  }

  SuperKernelBaseNode *CreateResetNode(uint64_t nodeId, uint32_t streamIdx) {
    auto node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, streamIdx,
                                                        INVALID_STREAM_ID, INVALID_TASK_ID);
    node->SetNodeType(SkNodeType::NODE_RESET);
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

  static void BindSyntheticEvent(SuperKernelBaseNode *owner, SuperKernelBaseNode *event) {
    event->SetPreNodeId(owner->GetNodeId());
  }

  std::vector<SuperKernelBaseNode *> BuildTasksWithCustomEvents(
      const std::vector<SuperKernelBaseNode *> &taskNodes,
      const std::vector<std::unique_ptr<SuperKernelBaseNode>> &eventNodes) {
    std::vector<SuperKernelBaseNode *> tasks;
    if (optimizer->BuildTaskListWithCustomEvents(taskNodes, eventNodes, tasks)) {
      EXPECT_TRUE(optimizer->BuildScopeDependencies(taskNodes, eventNodes, *graph));
    } else {
      ADD_FAILURE() << "failed to assemble tasks with custom events";
    }
    return tasks;
  }

  bool AnalyzeTasks(const std::vector<SuperKernelBaseNode *> &tasks, SkTaskBuilder::SkTaskContext &context) {
    SkTaskIR ir;
    ScopeCoreInfo coreInfo{SkKernelType::MIX_AIC_1_2, 1};
    if (!SkTaskLowering(*opts, *graph).Run("test", tasks, 0, coreInfo, ir)) {
      return false;
    }
    return SkTaskBuilder().PrecomputeSyncRelationsFromGraph(ir.tasks, ir.config, context, true);
  }

  SkQueueType QueueFor(const std::vector<SuperKernelBaseNode *> &tasks, const std::vector<SkQueueType> &queues,
                       const SuperKernelBaseNode *node) {
    auto it = std::find(tasks.begin(), tasks.end(), node);
    if (it == tasks.end() || queues.size() != tasks.size()) {
      return SkQueueType::UNKNOWN;
    }
    return queues[static_cast<size_t>(std::distance(tasks.begin(), it))];
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

TEST_F(SkOptimizerTaskReorderTest, ScheduleMapsProfilingByFinalTaskIndexAfterCustomEvents) {
  const std::string modelId = "model_custom_event_profiling";
  graph->modelId = modelId;
  auto *owner = CreateKernelNode(3001, 0, SkKernelType::AIV_ONLY);
  auto *later = CreateKernelNode(3002, 1, SkKernelType::AIV_ONLY);
  owner->nodeInfos.kernelInfos.funcName = "owner_kernel";
  later->nodeInfos.kernelInfos.funcName = "later_kernel";
  owner->nodeInfos.kernelInfos.numBlocks = 1;
  later->nodeInfos.kernelInfos.numBlocks = 1;
  auto *reset = CreateResetNode(3003, 0);
  BindSyntheticEvent(owner, reset);
  auto *notify = CreateNotifyNode(3004, 0);
  BindSyntheticEvent(owner, notify);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetScopeCoreInfo({SkKernelType::AIV_ONLY, 1});
  scopeInfo.MutableExtInfo().filteredNodes = {owner, later};
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[reset->GetNodeId()]));
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[notify->GetNodeId()]));

  auto &recorder = SkEventRecorder::Instance();
  const auto modelCount = recorder.modelIdIndexMap.size();
  recorder.enabled = true;
  recorder.deviceCtxs.active = 1;
  SkTaskBuilder builder;
  (void)optimizer->Schedule(scopeInfo, *graph, builder);
  std::vector<SuperKernelBaseNode *> tasks;
  ASSERT_TRUE(optimizer->BuildTaskListWithCustomEvents(scopeInfo.GetExtInfo().filteredNodes,
                                                       scopeInfo.GetExtInfo().eventNodes, tasks));
  SkTaskBuilder::SkTaskContext context;
  EXPECT_TRUE(AnalyzeTasks(tasks, context));
  const auto laterIndex = context.nodeIdToIndex.find(later->GetNodeId());
  EXPECT_NE(laterIndex, context.nodeIdToIndex.end());
  if (laterIndex != context.nodeIdToIndex.end()) {
    EXPECT_EQ(laterIndex->second, 3U);
    EXPECT_EQ(recorder.GetNodeInfo(modelId, scopeInfo.GetScopeId(), laterIndex->second).nodeName, "later_kernel");
  }
  EXPECT_EQ(recorder.GetNodeInfo(modelId, scopeInfo.GetScopeId(), 0).nodeName, "owner_kernel");
  EXPECT_TRUE(recorder.GetNodeInfo(modelId, scopeInfo.GetScopeId(), 1).nodeName.empty());
  EXPECT_TRUE(recorder.GetNodeInfo(modelId, scopeInfo.GetScopeId(), 2).nodeName.empty());
  recorder.enabled = false;
  recorder.deviceCtxs.active = 0;
  recorder.nodeInfoMap.erase(modelId);
  recorder.skNameMap.erase(modelId);
  if (recorder.modelIdIndexMap.size() > modelCount && recorder.modelIdIndexMap.back() == modelId) {
    recorder.modelIdIndexMap.pop_back();
    recorder.modelIdToIndexMap.erase(modelId);
  }
}

TEST_F(SkOptimizerTaskReorderTest, FrontResetAndBackNotifyFollowOwnTailWithoutDependingOnEachOther) {
  auto *owner = CreateKernelNode(7001, 2, SkKernelType::MIX_AIC_1_2);
  auto *other = CreateKernelNode(7002, 3, SkKernelType::AIC_ONLY);
  auto *reset = CreateResetNode(7003, 2);
  BindSyntheticEvent(owner, reset);
  auto *notify = CreateNotifyNode(7004, 2);
  BindSyntheticEvent(owner, notify);
  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetScopeCoreInfo({SkKernelType::MIX_AIC_1_2, 1});
  scopeInfo.MutableExtInfo().filteredNodes = {owner, other};
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[reset->GetNodeId()]));
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[notify->GetNodeId()]));
  SkTaskBuilder builder;
  (void)optimizer->Schedule(scopeInfo, *graph, builder);
  std::vector<SuperKernelBaseNode *> tasks;
  ASSERT_TRUE(optimizer->BuildTaskListWithCustomEvents(scopeInfo.GetExtInfo().filteredNodes,
                                                       scopeInfo.GetExtInfo().eventNodes, tasks));
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{7001, 7004, 7003, 7002}));
  EXPECT_EQ(owner->sendToNodeId.count(notify->GetNodeId()), 1U);
  EXPECT_EQ(owner->sendToNodeId.count(reset->GetNodeId()), 1U);
  EXPECT_EQ(other->sendToNodeId.count(reset->GetNodeId()), 0U);
  EXPECT_TRUE(notify->sendToNodeId.empty());
  EXPECT_TRUE(reset->sendToNodeId.empty());
  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[0].cubSendInfo.at(1), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[0].cubSendInfo.at(2), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[1].vecRecvInfo.at(0), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[2].vecRecvInfo.at(0), SyncDirection::CUB_TO_VEC);
}

TEST_F(SkOptimizerTaskReorderTest, FrontResetFollowsWaitAtStreamTail) {
  auto *owner = CreateKernelNode(1411, 2, SkKernelType::AIV_ONLY);
  auto *wait = CreateWaitNode(1412, 2);
  auto *other = CreateKernelNode(1413, 3, SkKernelType::AIC_ONLY);
  auto *reset = CreateResetNode(1414, 2);
  BindSyntheticEvent(wait, reset);
  wait->SetPreNodeId(owner->GetNodeId());
  auto *successor = CreateKernelNode(1415, 2, SkKernelType::AIC_ONLY);
  wait->SetNextNodeId(successor->GetNodeId());
  std::vector<std::unique_ptr<SuperKernelBaseNode>> events;
  events.push_back(std::move(graph->graphMap[reset->GetNodeId()]));
  auto tasks = BuildTasksWithCustomEvents({owner, wait, other}, events);
  EXPECT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{1411, 1412, 1414, 1413}));
  EXPECT_EQ(wait->sendToNodeId.count(reset->GetNodeId()), 1U);
  EXPECT_EQ(other->sendToNodeId.count(reset->GetNodeId()), 0U);
  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[1].cubSendInfo.at(2), SyncDirection::CUB_TO_VEC);
}

TEST_F(SkOptimizerTaskReorderTest, CrossStreamPredecessor_SynchronizesBeforeOwnedNotify) {
  auto *aicProducer = CreateKernelNode(1321, 3, SkKernelType::AIC_ONLY);
  auto *aivConsumer = CreateKernelNode(1322, 2, SkKernelType::AIV_ONLY);
  auto *unrelatedTail = CreateKernelNode(1323, 1, SkKernelType::AIC_ONLY);
  auto *notify = CreateNotifyNode(1324, 2);
  BindSyntheticEvent(aivConsumer, notify);
  aicProducer->sendToNodeId.insert(aivConsumer->GetNodeId());
  aivConsumer->receiveNodeId.insert(aicProducer->GetNodeId());
  std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;
  eventNodes.push_back(std::move(graph->graphMap[notify->GetNodeId()]));

  auto tasks = BuildTasksWithCustomEvents({aicProducer, aivConsumer, unrelatedTail}, eventNodes);
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{1321, 1322, 1324, 1323}));
  std::vector<SkQueueType> queues;
  ASSERT_TRUE(SkQueueTypeResolver(*graph).Resolve(tasks, queues));
  EXPECT_EQ(QueueFor(tasks, queues, notify), SkQueueType::AIV);

  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[0].cubSendInfo.at(1), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[1].vecRecvInfo.at(0), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[1].vecSendInfo.count(2), 0U);
  EXPECT_EQ(context.syncInfos[2].cubRecvInfo.count(1), 0U);
  EXPECT_EQ(aicProducer->sendToNodeId.count(aivConsumer->GetNodeId()), 1U);
  EXPECT_EQ(aivConsumer->receiveNodeId.count(aicProducer->GetNodeId()), 1U);
  EXPECT_EQ(aivConsumer->sendToNodeId.count(notify->GetNodeId()), 1U);
  EXPECT_EQ(aicProducer->sendToNodeId.count(notify->GetNodeId()), 0U);
}

TEST_F(SkOptimizerTaskReorderTest, BackwardNotifyDependencyIsRejectedByBuilder) {
  auto *owner = CreateKernelNode(1331, 2);
  auto *lateProducer = CreateKernelNode(1332, 3);
  auto *notify = CreateNotifyNode(1333, 2);
  BindSyntheticEvent(owner, notify);
  lateProducer->sendToNodeId.insert(notify->GetNodeId());
  notify->receiveNodeId.insert(lateProducer->GetNodeId());
  std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;
  eventNodes.push_back(std::move(graph->graphMap[notify->GetNodeId()]));

  auto tasks = BuildTasksWithCustomEvents({owner, lateProducer}, eventNodes);
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{1331, 1333, 1332}));
  EXPECT_EQ(owner->sendToNodeId.count(notify->GetNodeId()), 1U);
  EXPECT_EQ(notify->GetPreNodeId(), owner->GetNodeId());
  std::vector<SkQueueType> queues;
  ASSERT_TRUE(SkQueueTypeResolver(*graph).Resolve(tasks, queues));
  SkTaskBuilder::SkTaskContext context;
  EXPECT_FALSE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.nodeIdToIndex.size(), tasks.size());
}

TEST_F(SkOptimizerTaskReorderTest, InvalidSyntheticEventRejectsEntireTaskList) {
  auto *owner = CreateKernelNode(1311, 1);
  auto *other = CreateKernelNode(1312, 2);
  auto *later = CreateKernelNode(1319, 1);
  auto *valid = CreateNotifyNode(1313, 1);
  auto *missingOwner = CreateResetNode(1314, 1);
  auto *absentOwner = CreateKernelNode(1317, 1);
  auto *eventWithAbsentOwner = CreateResetNode(1318, 1);
  eventWithAbsentOwner->SetPreNodeId(absentOwner->GetNodeId());
  auto *wrongStreamOwner = CreateResetNode(1316, 1);
  wrongStreamOwner->SetPreNodeId(other->GetNodeId());
  valid->SetPreNodeId(owner->GetNodeId());
  auto *unsupported = CreateWaitNode(1315, 1);
  auto *earlyEmptyStreamEvent = CreateNotifyNode(1320, 3);
  BindSyntheticEvent(owner, earlyEmptyStreamEvent);
  SuperKernelScopeInfo scopeInfo;
  scopeInfo.MutableExtInfo().filteredNodes = {owner, other, later};
  auto &events = scopeInfo.MutableExtInfo().eventNodes;
  events.push_back(std::move(graph->graphMap[valid->GetNodeId()]));
  events.push_back(nullptr);

  for (auto *invalid : {missingOwner, eventWithAbsentOwner, wrongStreamOwner, earlyEmptyStreamEvent, unsupported,
                        static_cast<SuperKernelBaseNode *>(nullptr)}) {
    SCOPED_TRACE(invalid == nullptr ? INVALID_TASK_ID : invalid->GetNodeId());
    events.back() = invalid == nullptr ? nullptr : std::move(graph->graphMap[invalid->GetNodeId()]);
    std::vector<SuperKernelBaseNode *> tasks;
    EXPECT_FALSE(optimizer->BuildTaskListWithCustomEvents(scopeInfo.GetExtInfo().filteredNodes, events, tasks));
    EXPECT_TRUE(tasks.empty());
    SkTaskBuilder builder;
    EXPECT_FALSE(optimizer->Schedule(scopeInfo, *graph, builder));
    EXPECT_TRUE(owner->sendToNodeId.empty());
    EXPECT_TRUE(other->sendToNodeId.empty());
    EXPECT_TRUE(later->receiveNodeId.empty());
    EXPECT_TRUE(valid->receiveNodeId.empty());
  }
}

TEST_F(SkOptimizerTaskReorderTest, ReorderedWaitAndFrontReset_DoNotJoinUnrelatedTail) {
  EnableCustomizeQueueReorder();
  auto *aicFirst = CreateKernelNode(1351, 1, SkKernelType::AIC_ONLY);
  auto *wait = CreateWaitNode(1352, 2);
  auto *aicTail = CreateKernelNode(1353, 1, SkKernelType::AIC_ONLY);
  auto *aivTail = CreateKernelNode(1354, 2, SkKernelType::AIV_ONLY);
  auto *reset = CreateResetNode(1355, 2);
  BindSyntheticEvent(aivTail, reset);
  wait->SetNextNodeId(aivTail->GetNodeId());
  std::vector<std::unique_ptr<SuperKernelBaseNode>> eventNodes;
  eventNodes.push_back(std::move(graph->graphMap[reset->GetNodeId()]));

  auto reordered = optimizer->ReorderWaitNodesForTaskBuild({aicFirst, wait, aicTail, aivTail});
  ASSERT_EQ(ToNodeIds(reordered), (std::vector<uint64_t>{1351, 1353, 1352, 1354}));
  auto tasks = BuildTasksWithCustomEvents(reordered, eventNodes);
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{1351, 1353, 1352, 1354, 1355}));
  std::vector<SkQueueType> queues;
  ASSERT_TRUE(SkQueueTypeResolver(*graph).Resolve(tasks, queues));
  EXPECT_EQ(QueueFor(tasks, queues, wait), SkQueueType::AIV);
  EXPECT_EQ(QueueFor(tasks, queues, reset), SkQueueType::AIV);

  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[1].cubSendInfo.count(4), 0U);
  EXPECT_EQ(context.syncInfos[4].vecRecvInfo.count(1), 0U);
  EXPECT_EQ(reset->receiveNodeId.count(aivTail->GetNodeId()), 1U);
  EXPECT_EQ(wait->sendToNodeId.count(aivTail->GetNodeId()), 1U);
}

TEST_F(SkOptimizerTaskReorderTest, Schedule_NotifyOnlyDoesNotJoinUnrelatedStreamTails) {
  auto *stream1Tail = CreateKernelNode(321, 1);
  auto *stream2Tail = CreateKernelNode(322, 2);
  auto *stream3Tail = CreateKernelNode(323, 3);
  auto *stream2Notify = CreateNotifyNode(324, 2);
  auto *stream3Notify = CreateNotifyNode(325, 3);
  BindSyntheticEvent(stream2Tail, stream2Notify);
  BindSyntheticEvent(stream3Tail, stream3Notify);
  for (auto *kernel : {stream1Tail, stream2Tail, stream3Tail}) {
    kernel->nodeInfos.kernelInfos.numBlocks = 1;
    kernel->nodeInfos.kernelInfos.cubeNum = 1;
  }

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.SetScopeCoreInfo({SkKernelType::AIC_ONLY, 1});
  scopeInfo.MutableExtInfo().filteredNodes = {stream1Tail, stream2Tail, stream3Tail};
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[stream2Notify->GetNodeId()]));
  scopeInfo.MutableExtInfo().eventNodes.push_back(std::move(graph->graphMap[stream3Notify->GetNodeId()]));

  SkTaskBuilder builder;
  (void)optimizer->Schedule(scopeInfo, *graph, builder);
  std::vector<SuperKernelBaseNode *> tasks;
  ASSERT_TRUE(optimizer->BuildTaskListWithCustomEvents(scopeInfo.GetExtInfo().filteredNodes,
                                                       scopeInfo.GetExtInfo().eventNodes, tasks));
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{321, 322, 324, 323, 325}));
  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(stream2Tail->sendToNodeId.count(stream2Notify->GetNodeId()), 1U);
  EXPECT_EQ(stream3Tail->sendToNodeId.count(stream3Notify->GetNodeId()), 1U);
  EXPECT_EQ(stream2Notify->receiveNodeId.size(), 1U);
  EXPECT_EQ(stream2Notify->receiveNodeId.count(322), 1U);
  EXPECT_EQ(stream3Notify->receiveNodeId.size(), 1U);
  EXPECT_EQ(stream3Notify->receiveNodeId.count(323), 1U);
  EXPECT_TRUE(stream1Tail->sendToNodeId.empty());
  EXPECT_EQ(stream3Tail->sendToNodeId.count(stream2Notify->GetNodeId()), 0U);
}

TEST_F(SkOptimizerTaskReorderTest, BuildScopeDependencies_ConnectsAdjacentNodesPerStream) {
  auto *stream0First = CreateKernelNode(201, 0);
  auto *stream1First = CreateKernelNode(202, 1);
  auto *stream0Second = CreateWaitNode(203, 0);
  auto *stream1Second = CreateKernelNode(204, 1);

  auto tasks = BuildTasksWithCustomEvents({stream0First, stream1First, stream0Second, stream1Second}, {});
  EXPECT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{201, 202, 203, 204}));

  EXPECT_EQ(stream0First->sendToNodeId.count(203), 1);
  EXPECT_EQ(stream0Second->receiveNodeId.count(201), 1);
  EXPECT_EQ(stream1First->sendToNodeId.count(204), 1);
  EXPECT_EQ(stream1Second->receiveNodeId.count(202), 1);
  EXPECT_EQ(stream0First->sendToNodeId.count(202), 0);
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
  graph->tailNodes.clear();

  // Setup stream 0: op1 -> op4 -> op5 -> op10
  graph->streams.emplace_back();
  graph->headNodes.push_back(1);
  graph->tailNodes.push_back(10);
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
  graph->tailNodes.push_back(11);
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
  graph->tailNodes.push_back(12);
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

TEST_F(SkOptimizerTaskReorderTest, CustomNotifyAfterWaitUsesBoundOwner) {
  auto *first = CreateKernelNode(9111, 1, SkKernelType::AIC_ONLY);
  auto *owner = CreateKernelNode(9112, 2, SkKernelType::AIV_ONLY);
  auto *wait = CreateWaitNode(9113, 2);
  auto *other = CreateKernelNode(9114, 3, SkKernelType::AIV_ONLY);
  auto *notify = CreateNotifyNode(9115, 2);
  BindSyntheticEvent(wait, notify);
  wait->SetPreNodeId(owner->GetNodeId());
  auto *successor = CreateKernelNode(9116, 2, SkKernelType::AIC_ONLY);
  wait->SetNextNodeId(successor->GetNodeId());
  std::vector<SuperKernelBaseNode *> original{first, owner, wait, other};
  std::vector<std::unique_ptr<SuperKernelBaseNode>> events;
  events.push_back(std::move(graph->graphMap[notify->GetNodeId()]));
  auto tasks = BuildTasksWithCustomEvents(original, events);
  EXPECT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{9111, 9112, 9113, 9115, 9114}));
  EXPECT_EQ(notify->GetPreNodeId(), wait->GetNodeId());
  EXPECT_EQ(wait->sendToNodeId.count(notify->GetNodeId()), 1U);
  EXPECT_EQ(notify->receiveNodeId.count(wait->GetNodeId()), 1U);
  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[2].cubSendInfo.at(3), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[3].vecRecvInfo.at(2), SyncDirection::CUB_TO_VEC);
}

TEST_F(SkOptimizerTaskReorderTest, TaskAssemblyUsesBoundOwnerWithoutAddingDependency) {
  auto *owner = CreateKernelNode(9201, 2, SkKernelType::AIV_ONLY);
  auto *later = CreateKernelNode(9202, 2, SkKernelType::AIC_ONLY);
  auto *notify = CreateNotifyNode(9203, 2);
  notify->SetPreNodeId(owner->GetNodeId());
  std::vector<std::unique_ptr<SuperKernelBaseNode>> events;
  events.push_back(std::move(graph->graphMap[notify->GetNodeId()]));
  std::vector<SuperKernelBaseNode *> tasks;
  ASSERT_TRUE(optimizer->BuildTaskListWithCustomEvents({owner, later}, events, tasks));
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{9201, 9203, 9202}));
  EXPECT_EQ(notify->GetPreNodeId(), owner->GetNodeId());
  EXPECT_EQ(owner->sendToNodeId.count(notify->GetNodeId()), 0U);
  EXPECT_EQ(later->sendToNodeId.count(notify->GetNodeId()), 0U);
  EXPECT_TRUE(notify->receiveNodeId.empty());
}

TEST_F(SkOptimizerTaskReorderTest, EmptyStreamEventsFollowBoundScopeTail) {
  auto *first = CreateKernelNode(9301, 0);
  auto *tail = CreateKernelNode(9302, 1, SkKernelType::MIX_AIC_1_2);
  auto *reset = CreateResetNode(9303, 2);
  auto *notify = CreateNotifyNode(9304, 2);
  BindSyntheticEvent(tail, reset);
  BindSyntheticEvent(tail, notify);
  std::vector<std::unique_ptr<SuperKernelBaseNode>> events;
  events.push_back(std::move(graph->graphMap[reset->GetNodeId()]));
  events.push_back(std::move(graph->graphMap[notify->GetNodeId()]));

  auto tasks = BuildTasksWithCustomEvents({first, tail}, events);
  ASSERT_EQ(ToNodeIds(tasks), (std::vector<uint64_t>{9301, 9302, 9304, 9303}));
  SkTaskBuilder::SkTaskContext context;
  ASSERT_TRUE(AnalyzeTasks(tasks, context));
  EXPECT_EQ(context.syncInfos[1].cubSendInfo.at(2), SyncDirection::CUB_TO_VEC);
  EXPECT_EQ(context.syncInfos[1].cubSendInfo.at(3), SyncDirection::CUB_TO_VEC);
}
