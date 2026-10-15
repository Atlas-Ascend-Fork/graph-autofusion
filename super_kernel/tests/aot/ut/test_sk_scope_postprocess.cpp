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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_scope_postprocess.h"
#include "sk_node.h"
#include "sk_resource_manager.h"
#include "ut_common_stubs.h"

namespace {
class ScopedModelContext {
 public:
  explicit ScopedModelContext(aclmdlRI model) : model_(model) {
    SkResourceManager::SetCurrentModel(model_);
    EXPECT_EQ(SkResourceManager::CallbackRegister(model_, "model_scope_postprocess"), ACL_SUCCESS);
  }

  ~ScopedModelContext() {
    (void)SkUtInvokeModelDestroyCallback(model_);
    SkResourceManager::SetCurrentModel(nullptr);
  }

 private:
  aclmdlRI model_ = nullptr;
};
}  // namespace

class SuperKernelScopePostprocessTest : public testing::Test {
 protected:
  void SetUp() override {
    graph = std::make_unique<SuperKernelGraph>();
  }

  std::unique_ptr<SuperKernelGraph> graph;
};

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyWaitPairCancelled_Success) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(1);
  notifyNode->SetNextNodeId(2);
  notifyNode->nodeInfos.syncInfos.eventId = 100;
  notifyNode->nodeInfos.syncInfos.correspondingWaitNodeIds = {2};
  notifyNode->isFusible = true;

  auto waitNode =
      std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 1, 0, INVALID_STREAM_ID, 1);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->SetNodeId(2);
  waitNode->SetPreNodeId(1);
  waitNode->SetNextNodeId(INVALID_TASK_ID);
  waitNode->nodeInfos.syncInfos.eventId = 100;
  waitNode->nodeInfos.syncInfos.correspondingNotifyNodeId = 1;
  waitNode->isFusible = true;

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(notifyNode.get());
  scopeInfo.nodes_.push_back(waitNode.get());

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 0;
  streamInfo.headNodeIdx = 1;
  streamInfo.tailNodeIdx = 2;
  streamInfo.nodeSize = 2;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[1] = std::move(notifyNode);
  graph->graphMap[2] = std::move(waitNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyOneToManyWaits_AllCancelled_Success) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(3);
  notifyNode->SetNextNodeId(4);
  notifyNode->nodeInfos.syncInfos.eventId = 200;
  notifyNode->nodeInfos.syncInfos.correspondingWaitNodeIds = {4, 5};
  notifyNode->isFusible = true;

  auto waitNode1 =
      std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 1, 0, INVALID_STREAM_ID, 3);
  waitNode1->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode1->SetNodeId(4);
  waitNode1->SetPreNodeId(3);
  waitNode1->SetNextNodeId(5);
  waitNode1->nodeInfos.syncInfos.eventId = 200;
  waitNode1->nodeInfos.syncInfos.correspondingNotifyNodeId = 3;
  waitNode1->isFusible = true;

  auto waitNode2 = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 2, 1,
                                                           INVALID_STREAM_ID, INVALID_TASK_ID);
  waitNode2->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode2->SetNodeId(5);
  waitNode2->SetPreNodeId(INVALID_TASK_ID);
  waitNode2->SetNextNodeId(INVALID_TASK_ID);
  waitNode2->nodeInfos.syncInfos.eventId = 200;
  waitNode2->nodeInfos.syncInfos.correspondingNotifyNodeId = 3;
  waitNode2->isFusible = true;

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(notifyNode.get());
  scopeInfo.nodes_.push_back(waitNode1.get());
  scopeInfo.nodes_.push_back(waitNode2.get());

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 3;
  stream0.tailNodeIdx = 4;
  stream0.nodeSize = 2;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 5;
  stream1.tailNodeIdx = 5;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[3] = std::move(notifyNode);
  graph->graphMap[4] = std::move(waitNode1);
  graph->graphMap[5] = std::move(waitNode2);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NotifyWithoutWait_NoKernelCandidate_SkipSuccess) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(6);
  notifyNode->SetPreNodeId(INVALID_TASK_ID);
  notifyNode->SetNextNodeId(INVALID_TASK_ID);
  notifyNode->nodeInfos.syncInfos.eventId = 300;
  notifyNode->isFusible = true;

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(notifyNode.get());

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 6;
  stream0.tailNodeIdx = 6;
  stream0.nodeSize = 1;
  scopeInfo.scopeStreamInfos_.push_back(stream0);

  graph->graphMap[6] = std::move(notifyNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_EmptyUserScopeWithOnlyScopeMarkers_NoTargetNode) {
  auto beginNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                           INVALID_TASK_ID);
  beginNode->SetNodeType(SkNodeType::NODE_KERNEL);
  beginNode->SetNodeId(7);
  beginNode->SetPreNodeId(INVALID_TASK_ID);
  beginNode->SetNextNodeId(8);
  beginNode->SetIsFusible(true);
  beginNode->SetIsScopeNode(true);
  beginNode->isScopeBegin = true;
  beginNode->scopeName = "empty_scope";

  auto placeholderNode =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, 7);
  placeholderNode->SetNodeType(SkNodeType::NODE_KERNEL);
  placeholderNode->SetNodeId(8);
  placeholderNode->SetPreNodeId(7);
  placeholderNode->SetNextNodeId(9);
  placeholderNode->SetIsFusible(true);
  placeholderNode->SetIsScopeNode(true);
  placeholderNode->isPlaceholder = true;
  placeholderNode->scopeName = "empty_scope";

  auto endNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 2, 0, INVALID_STREAM_ID, 8);
  endNode->SetNodeType(SkNodeType::NODE_KERNEL);
  endNode->SetNodeId(9);
  endNode->SetPreNodeId(8);
  endNode->SetNextNodeId(INVALID_TASK_ID);
  endNode->SetIsFusible(true);
  endNode->SetIsScopeNode(true);
  endNode->isScopeEnd = true;
  endNode->scopeName = "empty_scope";

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {beginNode.get(), placeholderNode.get(), endNode.get()};

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 0;
  streamInfo.headNodeIdx = 7;
  streamInfo.tailNodeIdx = 9;
  streamInfo.nodeSize = 3;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[7] = std::move(beginNode);
  graph->graphMap[8] = std::move(placeholderNode);
  graph->graphMap[9] = std::move(endNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_SingleKernel_SelectMainNode_Success) {
  auto kernelNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 3, INVALID_STREAM_ID,
                                                            INVALID_TASK_ID);
  kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);
  kernelNode->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  kernelNode->SetNodeId(11);
  kernelNode->SetPreNodeId(INVALID_TASK_ID);
  kernelNode->SetNextNodeId(INVALID_TASK_ID);
  kernelNode->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(kernelNode.get());

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 3;
  streamInfo.headNodeIdx = 11;
  streamInfo.tailNodeIdx = 11;
  streamInfo.nodeSize = 1;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[11] = std::move(kernelNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  ASSERT_EQ(scopeInfo.extInfo_.filteredNodes.size(), 1U);
  EXPECT_EQ(scopeInfo.extInfo_.filteredNodes[0]->GetNodeId(), 11U);
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, 11U);

  ASSERT_EQ(scopeInfo.scopeStreamInfos_.size(), 1U);
  EXPECT_EQ(scopeInfo.scopeStreamInfos_[0].streamIdx, 3U);
  EXPECT_EQ(scopeInfo.scopeStreamInfos_[0].headNodeIdx, 11U);
  EXPECT_EQ(scopeInfo.scopeStreamInfos_[0].tailNodeIdx, 11U);
  EXPECT_EQ(scopeInfo.scopeStreamInfos_[0].nodeSize, 1U);
  EXPECT_TRUE(scopeInfo.extInfo_.customParamsList.empty() || scopeInfo.extInfo_.customParamsList[0].empty());
  EXPECT_TRUE(scopeInfo.extInfo_.eventNodes.empty());
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_StreamHeadMissing_Failed) {
  auto kernelNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                            INVALID_TASK_ID);
  kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);
  kernelNode->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  kernelNode->SetNodeId(31);
  kernelNode->SetPreNodeId(INVALID_TASK_ID);
  kernelNode->SetNextNodeId(INVALID_TASK_ID);
  kernelNode->isFusible = true;

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(kernelNode.get());

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 0;
  streamInfo.headNodeIdx = 999;
  streamInfo.tailNodeIdx = 31;
  streamInfo.nodeSize = 1;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[31] = std::move(kernelNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::UNRECOVERABLE_FAIL);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_FrontWait_Success) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  stream0Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Node0->SetNodeId(10);
  stream0Node0->SetPreNodeId(INVALID_TASK_ID);
  stream0Node0->SetNextNodeId(11);
  stream0Node0->SetIsFusible(true);

  auto stream0Node1 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, 10);
  stream0Node1->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Node1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Node1->SetNodeId(11);
  stream0Node1->SetPreNodeId(10);
  stream0Node1->SetNextNodeId(INVALID_TASK_ID);
  stream0Node1->SetIsFusible(true);

  auto stream1Node0 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 19);
  stream1Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Node0->SetNodeId(20);
  stream1Node0->SetPreNodeId(19);
  stream1Node0->SetNextNodeId(INVALID_TASK_ID);
  stream1Node0->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Node0.get(), stream0Node1.get(), stream1Node0.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 10;
  stream0.tailNodeIdx = 11;
  stream0.nodeSize = 2;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 20;
  stream1.tailNodeIdx = 20;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[10] = std::move(stream0Node0);
  graph->graphMap[11] = std::move(stream0Node1);
  graph->graphMap[20] = std::move(stream1Node0);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  EXPECT_FALSE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_FALSE(scopeInfo.scopeStreamInfos_.empty());
  EXPECT_NE(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_BackBlock_Success) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  stream0Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Node0->SetNodeId(40);
  stream0Node0->SetPreNodeId(INVALID_TASK_ID);
  stream0Node0->SetNextNodeId(INVALID_TASK_ID);
  stream0Node0->SetIsFusible(true);

  auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  stream1Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Node0->SetNodeId(50);
  stream1Node0->SetPreNodeId(INVALID_TASK_ID);
  stream1Node0->SetNextNodeId(51);
  stream1Node0->SetIsFusible(true);

  auto stream1Node1 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, INVALID_STREAM_ID, 50);
  stream1Node1->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Node1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Node1->SetNodeId(51);
  stream1Node1->SetPreNodeId(50);
  stream1Node1->SetNextNodeId(INVALID_TASK_ID);
  stream1Node1->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Node0.get(), stream1Node0.get(), stream1Node1.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 40;
  stream0.tailNodeIdx = 40;
  stream0.nodeSize = 1;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 50;
  stream1.tailNodeIdx = 50;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[40] = std::move(stream0Node0);
  graph->graphMap[50] = std::move(stream1Node0);
  graph->graphMap[51] = std::move(stream1Node1);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  EXPECT_FALSE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_FALSE(scopeInfo.scopeStreamInfos_.empty());
  EXPECT_NE(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NoKernelCandidate_SkipSuccess) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(60);
  notifyNode->SetPreNodeId(INVALID_TASK_ID);
  notifyNode->SetNextNodeId(INVALID_TASK_ID);
  notifyNode->nodeInfos.syncInfos.eventId = 101;
  notifyNode->isFusible = true;

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(notifyNode.get());

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 0;
  streamInfo.headNodeIdx = 60;
  streamInfo.tailNodeIdx = 60;
  streamInfo.nodeSize = 1;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[60] = std::move(notifyNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_NoKernelAfterFilter_SkipSuccess) {
  auto defaultNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID,
                                                             INVALID_TASK_ID);
  defaultNode->SetNodeType(SkNodeType::NODE_DEFAULT);
  defaultNode->SetNodeId(61);
  defaultNode->SetPreNodeId(INVALID_TASK_ID);
  defaultNode->SetNextNodeId(INVALID_TASK_ID);
  defaultNode->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_.push_back(defaultNode.get());

  ScopeStreamInfo streamInfo;
  streamInfo.streamIdx = 0;
  streamInfo.headNodeIdx = 61;
  streamInfo.tailNodeIdx = 61;
  streamInfo.nodeSize = 1;
  scopeInfo.scopeStreamInfos_.push_back(streamInfo);

  graph->graphMap[61] = std::move(defaultNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_MultiStreamKernelOnly_Success) {
  auto stream0Node0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  stream0Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Node0->SetNodeId(70);
  stream0Node0->SetPreNodeId(INVALID_TASK_ID);
  stream0Node0->SetNextNodeId(INVALID_TASK_ID);
  stream0Node0->SetIsFusible(true);

  auto stream1Node0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1,
                                                              INVALID_STREAM_ID, INVALID_TASK_ID);
  stream1Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Node0->SetNodeId(80);
  stream1Node0->SetPreNodeId(INVALID_TASK_ID);
  stream1Node0->SetNextNodeId(INVALID_TASK_ID);
  stream1Node0->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Node0.get(), stream1Node0.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 70;
  stream0.tailNodeIdx = 70;
  stream0.nodeSize = 1;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 80;
  stream1.tailNodeIdx = 80;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[70] = std::move(stream0Node0);
  graph->graphMap[80] = std::move(stream1Node0);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  ASSERT_EQ(scopeInfo.extInfo_.filteredNodes.size(), 2U);
  ASSERT_EQ(scopeInfo.scopeStreamInfos_.size(), 2U);
  EXPECT_TRUE(scopeInfo.extInfo_.skMainNodeId == 70U || scopeInfo.extInfo_.skMainNodeId == 80U);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_MainSelectReserveBoundaryAndMissingNext_Failed) {
  auto stream0Node0 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  stream0Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Node0->SetNodeId(90);
  stream0Node0->SetPreNodeId(1);
  stream0Node0->SetNextNodeId(INVALID_TASK_ID);
  stream0Node0->SetIsFusible(true);

  auto stream1Node0 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 2);
  stream1Node0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Node0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Node0->SetNodeId(100);
  stream1Node0->SetPreNodeId(2);
  stream1Node0->SetNextNodeId(101);
  stream1Node0->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Node0.get(), stream1Node0.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 90;
  stream0.tailNodeIdx = 90;
  stream0.nodeSize = 1;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 100;
  stream1.tailNodeIdx = 999;  // Tail missing in graph; reserve-step traversal uses missing next path.
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[90] = std::move(stream0Node0);
  graph->graphMap[100] = std::move(stream1Node0);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::UNRECOVERABLE_FAIL);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_FrontWaitMoveWorkNodePath_Success) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  auto s0n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  s0n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n0->SetNodeId(200);
  s0n0->SetPreNodeId(1);
  s0n0->SetNextNodeId(201);
  s0n0->SetIsFusible(true);

  auto s0n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, 200);
  s0n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n1->SetNodeId(201);
  s0n1->SetPreNodeId(200);
  s0n1->SetNextNodeId(INVALID_TASK_ID);
  s0n1->SetIsFusible(true);

  auto s1n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 2);
  s1n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n0->SetNodeId(300);
  s1n0->SetPreNodeId(2);
  s1n0->SetNextNodeId(301);
  s1n0->SetIsFusible(true);

  auto s1n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, INVALID_STREAM_ID, 300);
  s1n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n1->SetNodeId(301);
  s1n1->SetPreNodeId(300);
  s1n1->SetNextNodeId(INVALID_TASK_ID);
  s1n1->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {s0n0.get(), s0n1.get(), s1n0.get(), s1n1.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 200;
  stream0.tailNodeIdx = 201;
  stream0.nodeSize = 2;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 300;
  stream1.tailNodeIdx = 301;
  stream1.nodeSize = 2;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[200] = std::move(s0n0);
  graph->graphMap[201] = std::move(s0n1);
  graph->graphMap[300] = std::move(s1n0);
  graph->graphMap[301] = std::move(s1n1);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  EXPECT_FALSE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_FALSE(scopeInfo.scopeStreamInfos_.empty());
  EXPECT_NE(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_MidScopeTwoStreamsThreePlusTwo_SelectSecondAsMain_Success) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  auto s0n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 900);
  s0n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n0->SetNodeId(1000);
  s0n0->SetPreNodeId(900);
  s0n0->SetNextNodeId(1001);
  s0n0->SetIsFusible(true);

  auto s0n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, 1000);
  s0n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n1->SetNodeId(1001);
  s0n1->SetPreNodeId(1000);
  s0n1->SetNextNodeId(1002);
  s0n1->SetIsFusible(true);

  auto s0n2 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 2, 0, INVALID_STREAM_ID, 1001);
  s0n2->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n2->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n2->SetNodeId(1002);
  s0n2->SetPreNodeId(1001);
  s0n2->SetNextNodeId(1003);
  s0n2->SetIsFusible(true);

  auto s1n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 1900);
  s1n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n0->SetNodeId(2000);
  s1n0->SetPreNodeId(1900);
  s1n0->SetNextNodeId(2001);
  s1n0->SetIsFusible(true);

  auto s1n1 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, INVALID_STREAM_ID, 2000);
  s1n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n1->SetNodeId(2001);
  s1n1->SetPreNodeId(2000);
  s1n1->SetNextNodeId(2002);
  s1n1->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {s0n0.get(), s0n1.get(), s0n2.get(), s1n0.get(), s1n1.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 1000;
  stream0.tailNodeIdx = 1002;
  stream0.nodeSize = 3;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 2000;
  stream1.tailNodeIdx = 2001;
  stream1.nodeSize = 2;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[1000] = std::move(s0n0);
  graph->graphMap[1001] = std::move(s0n1);
  graph->graphMap[1002] = std::move(s0n2);
  graph->graphMap[2000] = std::move(s1n0);
  graph->graphMap[2001] = std::move(s1n1);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  EXPECT_FALSE(scopeInfo.extInfo_.filteredNodes.empty());
  ASSERT_EQ(scopeInfo.scopeStreamInfos_.size(), 2U);
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, 2001U);
  EXPECT_FALSE(scopeInfo.extInfo_.eventNodes.empty());
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_ThreeStreams_TwoMidOneFull_Success) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  // stream0: middle scope with 3 kernels (has pre and next)
  auto s0n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 9000);
  s0n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n0->SetNodeId(10000);
  s0n0->SetPreNodeId(9000);
  s0n0->SetNextNodeId(10001);
  s0n0->SetIsFusible(true);

  auto s0n1 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 0, INVALID_STREAM_ID, 10000);
  s0n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n1->SetNodeId(10001);
  s0n1->SetPreNodeId(10000);
  s0n1->SetNextNodeId(10002);
  s0n1->SetIsFusible(true);

  auto s0n2 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 2, 0, INVALID_STREAM_ID, 10001);
  s0n2->SetNodeType(SkNodeType::NODE_KERNEL);
  s0n2->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s0n2->SetNodeId(10002);
  s0n2->SetPreNodeId(10001);
  s0n2->SetNextNodeId(10003);
  s0n2->SetIsFusible(true);

  // stream1: middle scope with 3 kernels (has pre and next)
  auto s1n0 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 19000);
  s1n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n0->SetNodeId(20000);
  s1n0->SetPreNodeId(19000);
  s1n0->SetNextNodeId(20001);
  s1n0->SetIsFusible(true);

  auto s1n1 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 1, INVALID_STREAM_ID, 20000);
  s1n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n1->SetNodeId(20001);
  s1n1->SetPreNodeId(20000);
  s1n1->SetNextNodeId(20002);
  s1n1->SetIsFusible(true);

  auto s1n2 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 2, 1, INVALID_STREAM_ID, 20001);
  s1n2->SetNodeType(SkNodeType::NODE_KERNEL);
  s1n2->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s1n2->SetNodeId(20002);
  s1n2->SetPreNodeId(20001);
  s1n2->SetNextNodeId(20003);
  s1n2->SetIsFusible(true);

  // stream2: full scope with 2 kernels (no pre and no next)
  auto s2n0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 2, INVALID_STREAM_ID,
                                                      INVALID_TASK_ID);
  s2n0->SetNodeType(SkNodeType::NODE_KERNEL);
  s2n0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s2n0->SetNodeId(30000);
  s2n0->SetPreNodeId(INVALID_TASK_ID);
  s2n0->SetNextNodeId(30001);
  s2n0->SetIsFusible(true);

  auto s2n1 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 1, 2, INVALID_STREAM_ID, 30000);
  s2n1->SetNodeType(SkNodeType::NODE_KERNEL);
  s2n1->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  s2n1->SetNodeId(30001);
  s2n1->SetPreNodeId(30000);
  s2n1->SetNextNodeId(INVALID_TASK_ID);
  s2n1->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {s0n0.get(), s0n1.get(), s0n2.get(), s1n0.get(), s1n1.get(), s1n2.get(), s2n0.get(), s2n1.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 10000;
  stream0.tailNodeIdx = 10002;
  stream0.nodeSize = 3;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 20000;
  stream1.tailNodeIdx = 20002;
  stream1.nodeSize = 3;

  ScopeStreamInfo stream2;
  stream2.streamIdx = 2;
  stream2.headNodeIdx = 30000;
  stream2.tailNodeIdx = 30001;
  stream2.nodeSize = 2;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1, stream2};

  graph->graphMap[10000] = std::move(s0n0);
  graph->graphMap[10001] = std::move(s0n1);
  graph->graphMap[10002] = std::move(s0n2);
  graph->graphMap[20000] = std::move(s1n0);
  graph->graphMap[20001] = std::move(s1n1);
  graph->graphMap[20002] = std::move(s1n2);
  graph->graphMap[30000] = std::move(s2n0);
  graph->graphMap[30001] = std::move(s2n1);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_TRUE(result);
  EXPECT_FALSE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.filteredNodes.size(), 8U);
  ASSERT_EQ(scopeInfo.scopeStreamInfos_.size(), 3U);
  EXPECT_NE(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_StreamSelectFailed_EventAddrNotPrepared) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  auto stream0Kernel =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  stream0Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Kernel->SetNodeId(10);
  stream0Kernel->SetPreNodeId(1);
  stream0Kernel->SetNextNodeId(11);
  stream0Kernel->SetIsFusible(true);

  auto stream1Kernel =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 2);
  stream1Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Kernel->SetNodeId(20);
  stream1Kernel->SetPreNodeId(2);
  stream1Kernel->SetNextNodeId(21);
  stream1Kernel->SetIsFusible(true);

  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 1, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(30);
  notifyNode->nodeInfos.syncInfos.eventId = 100;
  notifyNode->nodeInfos.syncInfos.correspondingWaitNodeIds = {40};
  notifyNode->SetIsFusible(true);

  auto resetNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 2, 0,
                                                           INVALID_STREAM_ID, INVALID_TASK_ID);
  resetNode->SetNodeType(SkNodeType::NODE_RESET);
  resetNode->SetNodeId(31);
  resetNode->nodeInfos.syncInfos.eventId = 100;
  resetNode->SetIsFusible(true);

  auto waitNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 2,
                                                          INVALID_STREAM_ID, INVALID_TASK_ID);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->SetNodeId(40);
  waitNode->nodeInfos.syncInfos.eventId = 100;
  waitNode->nodeInfos.syncInfos.correspondingNotifyNodeId = 30;
  waitNode->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Kernel.get(), stream1Kernel.get(), notifyNode.get(), resetNode.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 10;
  stream0.tailNodeIdx = 10;
  stream0.nodeSize = 1;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 20;
  stream1.tailNodeIdx = 20;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[10] = std::move(stream0Kernel);
  graph->graphMap[20] = std::move(stream1Kernel);
  graph->graphMap[30] = std::move(notifyNode);
  graph->graphMap[31] = std::move(resetNode);
  graph->graphMap[40] = std::move(waitNode);

  EventInfos eventInfos;
  eventInfos.notifyNodeId = 30;
  eventInfos.waitNodeIdList.insert(40);
  eventInfos.resetNodeIdList.insert(31);
  graph->eventToNodes[100] = std::move(eventInfos);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::RESOURCE_INSUFFICIENT);

  auto *notifyAfter = graph->GetNodeById(30);
  auto *resetAfter = graph->GetNodeById(31);
  auto *waitAfter = graph->GetNodeById(40);
  ASSERT_NE(notifyAfter, nullptr);
  ASSERT_NE(resetAfter, nullptr);
  ASSERT_NE(waitAfter, nullptr);
  EXPECT_EQ(notifyAfter->nodeInfos.syncInfos.addrValue, nullptr);
  EXPECT_EQ(resetAfter->nodeInfos.syncInfos.addrValue, nullptr);
  EXPECT_EQ(waitAfter->nodeInfos.syncInfos.addrValue, nullptr);
}

TEST_F(SuperKernelScopePostprocessTest, ApplyEventMemoryForFilteredNodes_SkipPreAppliedAddrValue) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(30);
  notifyNode->nodeInfos.syncInfos.eventId = 100;
  notifyNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1000);

  auto waitNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 1,
                                                          INVALID_STREAM_ID, INVALID_TASK_ID);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->SetNodeId(40);
  waitNode->nodeInfos.syncInfos.eventId = 100;
  waitNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1000);

  auto resetNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RESET, 0, 2,
                                                           INVALID_STREAM_ID, INVALID_TASK_ID);
  resetNode->SetNodeType(SkNodeType::NODE_RESET);
  resetNode->SetNodeId(50);
  resetNode->nodeInfos.syncInfos.eventId = 100;
  resetNode->nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x1000);

  std::vector<SuperKernelBaseNode *> filteredNodes = {notifyNode.get(), waitNode.get(), resetNode.get()};
  graph->graphMap[30] = std::move(notifyNode);
  graph->graphMap[40] = std::move(waitNode);
  graph->graphMap[50] = std::move(resetNode);

  std::vector<SuperKernelBaseNode *> needUpdateNodes;
  SuperKernelScopePostProcessor postProcessor(*graph);

  EXPECT_TRUE(postProcessor.ApplyEventMemoryForFilteredNodes(filteredNodes, needUpdateNodes));
  EXPECT_TRUE(needUpdateNodes.empty());
}

// Test case for "advance node unsuccessful: next node not found in graph"
TEST_F(SuperKernelScopePostprocessTest, PostProcess_AdvanceNodeNextNotFound_Failed) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  // stream0 has preNodeId, will trigger needFrontWait
  auto stream0Kernel =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  stream0Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Kernel->SetNodeId(100);
  stream0Kernel->SetPreNodeId(1);  // has pre node, triggers needFrontWait
  stream0Kernel->SetNextNodeId(INVALID_TASK_ID);
  stream0Kernel->SetIsFusible(true);

  // stream1 has nextNodeId pointing to non-existent node
  auto stream1Kernel0 = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);
  stream1Kernel0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Kernel0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Kernel0->SetNodeId(200);
  stream1Kernel0->SetPreNodeId(INVALID_TASK_ID);
  stream1Kernel0->SetNextNodeId(999);  // next node does not exist
  stream1Kernel0->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Kernel.get(), stream1Kernel0.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 100;
  stream0.tailNodeIdx = 100;
  stream0.nodeSize = 2;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 200;
  stream1.tailNodeIdx = 200;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[100] = std::move(stream0Kernel);
  graph->graphMap[200] = std::move(stream1Kernel0);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::RESOURCE_INSUFFICIENT);
}

// Test case for "find kernel with reserve unsuccessful"
TEST_F(SuperKernelScopePostprocessTest, PostProcess_FindKernelWithReserveUnsuccessful_Failed) {
  auto notifyNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0,
                                                            INVALID_STREAM_ID, INVALID_TASK_ID);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->SetNodeId(100);
  notifyNode->SetPreNodeId(INVALID_TASK_ID);
  notifyNode->SetNextNodeId(INVALID_TASK_ID);
  notifyNode->nodeInfos.syncInfos.eventId = 100;
  notifyNode->SetIsFusible(true);

  auto waitNode = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 1,
                                                          INVALID_STREAM_ID, INVALID_TASK_ID);
  waitNode->SetNodeType(SkNodeType::NODE_WAIT);
  waitNode->SetNodeId(200);
  waitNode->SetPreNodeId(INVALID_TASK_ID);
  waitNode->SetNextNodeId(INVALID_TASK_ID);
  waitNode->nodeInfos.syncInfos.eventId = 100;
  waitNode->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {notifyNode.get(), waitNode.get()};

  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 100;
  stream0.tailNodeIdx = 100;
  stream0.nodeSize = 1;

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 200;
  stream1.tailNodeIdx = 200;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[100] = std::move(notifyNode);
  graph->graphMap[200] = std::move(waitNode);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::NO_TARGET_NODE);
}

// Test case for "stream sync resource insufficient" during candidate collection
TEST_F(SuperKernelScopePostprocessTest, PostProcess_StreamSyncResourceInsufficient_Failed) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  // stream0 has preNodeId and nextNodeId, requires frontWait and backBlock
  auto stream0Kernel =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  stream0Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Kernel->SetNodeId(100);
  stream0Kernel->SetPreNodeId(1);   // has pre node, triggers needFrontWait
  stream0Kernel->SetNextNodeId(2);  // has next node, triggers needBackBlock
  stream0Kernel->SetIsFusible(true);

  // stream1 also has preNodeId and nextNodeId
  auto stream1Kernel =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1, INVALID_STREAM_ID, 3);
  stream1Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Kernel->SetNodeId(200);
  stream1Kernel->SetPreNodeId(3);   // has pre node, triggers needFrontWait
  stream1Kernel->SetNextNodeId(4);  // has next node, triggers needBackBlock
  stream1Kernel->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Kernel.get(), stream1Kernel.get()};

  // Set nodeSize too small to satisfy subStream/subStreamEntry requirements
  // Each needs: frontWaitNodeCount (2) + backBlockNodeCount (2) = 4 nodes
  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 100;
  stream0.tailNodeIdx = 100;
  stream0.nodeSize = 1;  // too small for subStreamEntry (needs 4-1=3)

  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 200;
  stream1.tailNodeIdx = 200;
  stream1.nodeSize = 1;  // too small for subStreamEntry (needs 4-1=3)

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[100] = std::move(stream0Kernel);
  graph->graphMap[200] = std::move(stream1Kernel);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::RESOURCE_INSUFFICIENT);
}

// Test case for "unable to find main SK node in scope, skip update"
// This case: only one stream qualifies as both mainStream and subStreamEntry candidate
// When selecting it as main, no other stream can be entry sub stream
TEST_F(SuperKernelScopePostprocessTest, PostProcess_UnableToFindMainEntryCombination_Failed) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  // stream0: has preNodeId (triggers needFrontWait), has kernel, large nodeSize
  // Can be both mainStream and subStreamEntry candidate
  auto stream0Kernel0 =
      std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, 1);
  stream0Kernel0->SetNodeType(SkNodeType::NODE_KERNEL);
  stream0Kernel0->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream0Kernel0->SetNodeId(100);
  stream0Kernel0->SetPreNodeId(1);  // has pre, triggers needFrontWait
  stream0Kernel0->SetNextNodeId(INVALID_TASK_ID);
  stream0Kernel0->SetIsFusible(true);

  // stream1: has preNodeId, has kernel, but nodeSize too small for subStream
  // Can only be subStreamEntry candidate, NOT mainStream candidate
  auto stream1Kernel = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 1,
                                                               INVALID_STREAM_ID, INVALID_TASK_ID);
  stream1Kernel->SetNodeType(SkNodeType::NODE_KERNEL);
  stream1Kernel->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  stream1Kernel->SetNodeId(200);
  stream1Kernel->SetPreNodeId(1);
  stream1Kernel->SetNextNodeId(INVALID_TASK_ID);
  stream1Kernel->SetIsFusible(true);

  SuperKernelScopeInfo scopeInfo;
  scopeInfo.nodes_ = {stream0Kernel0.get(), stream1Kernel.get()};

  // stream0: nodeSize=1, needFrontWait=true, needBackBlock=false
  // subStreamNeedCount = 2*1 + 2*0 = 2, subStreamEntryNeedCount = 2-1 = 1
  // 1 < 2, so cannot be subStream candidate; 1 >= 1, so can be subStreamEntry candidate
  ScopeStreamInfo stream0;
  stream0.streamIdx = 0;
  stream0.headNodeIdx = 100;
  stream0.tailNodeIdx = 100;
  stream0.nodeSize = 1;

  // stream1: nodeSize=1, needFrontWait=true, needBackBlock=false
  // subStreamNeedCount = 2, subStreamEntryNeedCount = 1
  // Same as stream0, only subStreamEntry candidate
  ScopeStreamInfo stream1;
  stream1.streamIdx = 1;
  stream1.headNodeIdx = 200;
  stream1.tailNodeIdx = 200;
  stream1.nodeSize = 1;

  scopeInfo.scopeStreamInfos_ = {stream0, stream1};

  graph->graphMap[100] = std::move(stream0Kernel0);
  graph->graphMap[200] = std::move(stream1Kernel);

  SuperKernelScopePostProcessor postProcessor(*graph);
  bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::RESOURCE_INSUFFICIENT);
}

TEST_F(SuperKernelScopePostprocessTest, PostProcess_ThreeStreamsWithThreeNodesEach_ResourceInsufficient) {
  ScopedModelContext modelCtx(reinterpret_cast<aclmdlRI>(0x1));

  constexpr uint32_t streamCount = 3;
  constexpr uint32_t nodesPerStream = 3;
  SuperKernelScopeInfo scopeInfo;
  for (uint32_t streamIdx = 0; streamIdx < streamCount; ++streamIdx) {
    const uint64_t headNodeId = static_cast<uint64_t>(streamIdx + 1U) * 100U;
    for (uint32_t nodeIdx = 0; nodeIdx < nodesPerStream; ++nodeIdx) {
      const uint64_t nodeId = headNodeId + nodeIdx;
      auto kernelNode = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeIdx, streamIdx,
                                                                INVALID_STREAM_ID, INVALID_TASK_ID);
      kernelNode->SetNodeType(SkNodeType::NODE_KERNEL);
      kernelNode->nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
      kernelNode->SetNodeId(nodeId);
      kernelNode->SetPreNodeId(nodeIdx == 0 ? headNodeId - 1U : nodeId - 1U);
      kernelNode->SetNextNodeId(nodeId + 1U);
      kernelNode->SetIsFusible(true);
      scopeInfo.nodes_.push_back(kernelNode.get());
      graph->graphMap[nodeId] = std::move(kernelNode);
    }

    ScopeStreamInfo streamInfo;
    streamInfo.streamIdx = streamIdx;
    streamInfo.headNodeIdx = headNodeId;
    streamInfo.tailNodeIdx = headNodeId + nodesPerStream - 1U;
    streamInfo.nodeSize = nodesPerStream;
    scopeInfo.scopeStreamInfos_.push_back(streamInfo);
  }

  SuperKernelScopePostProcessor postProcessor(*graph);
  const bool result = postProcessor.PostProcess(scopeInfo);

  EXPECT_FALSE(result);
  EXPECT_TRUE(scopeInfo.extInfo_.filteredNodes.empty());
  EXPECT_EQ(scopeInfo.extInfo_.skMainNodeId, INVALID_TASK_ID);
  EXPECT_EQ(scopeInfo.extInfo_.processStatus, ScopeProcessStatus::RESOURCE_INSUFFICIENT);
}
